"""Mechanism C: does a Tofino stage hold a FLAT 24 TCAM blocks, or 2 columns
of 12 rows with a wide table's blocks chained inside one column?

WHY THIS EXISTS. The 2026-09-05 calibration study closed every stage-count
residual except one (reviews/p4_tofino_reference.md Sec 7, Mechanism C).
`joint_low_sd10` and `joint_low_sd12` take only 2 tree tables per stage while
using 14 and 16 of 24 blocks, with no PHV conflicts and the crossbar far from
its limit -- consistent with `mau_spec.h:88-90`'s `Tofino_tcam_rows=12,
Tofino_tcam_columns=2` if a table's blocks must chain inside ONE column. But
the naive form of that rule (`2*floor(12/w)` tables per stage) is REFUTED:
`independent_low_sd10` fits 3 tables where it allows 2, and
`independent_low_sd12` and `joint_high_sd8` break it too.

18 archived campaign rows are too thin and too confounded to settle it, so
this sweeps the question directly: N synthetic ternary tables of W blocks
each, W from 5 to 12, compiled through the real WSL2 p4c, and reads the
maximum tables the compiler actually put in one stage.

WHAT MAKES THE PROBE VALID -- three confounders removed by construction:

  * All N tables key the SAME field. The Ternary Match Input crossbar charges
    per distinct FIELD (Sec 7, Mechanism 2), so N private wide keys would
    exhaust its 64 bytes long before 24 blocks bound -- two 6-block tables are
    already 66 bytes. One shared field reproduces the real generator, where
    every tree of a task keys the identical code_* set, and leaves BLOCKS as
    the binding dimension. This is the whole reason the probe can see
    geometry at all.
  * Every table writes its own `@pa_solitary` result field, so Mechanism A
    (two tables whose actions write one PHV container cannot share a stage)
    cannot masquerade as a geometry limit.
  * `size = 512` -- exactly one block's worth of rows -- so a table's block
    count is driven purely by key WIDTH and nothing multiplies it.

WHERE THE INFORMATION IS. The two rules coincide at every width except 7 and
8, where flat-24 predicts 3 tables per stage and the column rule predicts 2.
The other widths are controls: they check that the tables really cost W
blocks each and that the compiler really fills to the cap.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_column_sweep.py

Resumable: points already recorded in --out are skipped.
"""
import argparse
import collections
import math
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.compiler_calibration import _committed_blocks, committed_table_stages
from src.p4gen.build_p4_script import (
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
)
# Re-exported, not redefined: this sweep is what MEASURED the column geometry,
# and the model is what has to obey it. Two copies of the packing rule could
# drift and the sweep would then certify a rule the estimator does not use.
from src.p4gen.evaluation import fits_two_columns
from src.p4gen.p4_compile import compile_p4

# Sec 4.1: a ternary entry carries a flat 4-bit overhead before the key is
# divided into TCAM_BLOCK_KEY_LENGTH-bit rows.
TERNARY_ENTRY_OVERHEAD_BITS = 4

# 5 is the narrowest width at which blocks bind before the 8-table crossbar
# cap (floor(24/5) = 4 < 8); 12 is the widest table a single stage can hold at
# all. Below 5 the sweep would be measuring the table-count cap instead.
BLOCK_WIDTHS = (5, 6, 7, 8, 9, 10, 11, 12)

KEY_FIELD = 'code'
DEFAULT_OUT = 'results/tcam_column_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_column_sweep'


def tables_per_stage_flat_24(block_width):
    """What the model currently assumes: TCAM_BLOCKS_PER_STAGE blocks in one
    undifferentiated pool."""
    return TCAM_BLOCKS_PER_STAGE // block_width


def tables_per_stage_two_columns(block_width):
    """The Mechanism C candidate: a table's blocks chain within one column, so
    a column of TCAM_ROWS_PER_STAGE rows holds floor(rows/w) tables and there
    are TCAM_COLUMNS_PER_STAGE columns. Never larger than the flat cap, so it
    can only ever be the tighter of the two."""
    return TCAM_COLUMNS_PER_STAGE * (TCAM_ROWS_PER_STAGE // block_width)


def key_bits_for(block_width):
    """The ternary key width that costs exactly `block_width` blocks at 512
    entries: the widest BYTE-ALIGNED key with ceil((bits + 4) / 44) == w.

    The obvious choice, 44*w - 4, is wrong for even w and the sweep's own
    control caught it: 44*w - 4 == 4 (mod 8) exactly when w is even, and Sec
    4.1.2's extra-block anomaly then charges one block MORE than the formula
    (measured 7, 9 and 11 blocks at intended widths 6, 8 and 10, and exactly
    w at every odd width). An even-w probe therefore silently measured a
    (w+1)-wide table -- fatal at w=8, where the two rules disagree and the
    substituted w=9 is a width at which they agree. Rounding down to a byte
    boundary avoids the anomaly and still lands in the same block bucket.

    Capped at the single-stage crossbar budget. At w=12 the uncapped 524 bits
    is 66 bytes against a 64-byte stage, so such a table could not be matched
    in one stage at all and the probe would be measuring the crossbar rather
    than the geometry; 512 bits still costs ceil(516/44) = 12 blocks."""
    if block_width < 1:
        raise ValueError('block_width must be >= 1, got %r' % (block_width,))
    bits = TCAM_BLOCK_KEY_LENGTH * block_width - TERNARY_ENTRY_OVERHEAD_BITS
    bits = 8 * (bits // 8)
    return min(bits, 8 * TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE)


def blocks_for_key_bits(key_bits):
    return math.ceil((key_bits + TERNARY_ENTRY_OVERHEAD_BITS)
                     / TCAM_BLOCK_KEY_LENGTH)


_PROGRAM_TEMPLATE = """\
/*
 * Mechanism C probe: {n_tables} independent ternary tables, {block_width}
 * TCAM blocks each ({key_bits}-bit key, {size} entries), ALL keying the one
 * shared field meta.{key_field} so the ternary crossbar charges it once and
 * TCAM blocks are the binding dimension. Every action writes its own
 * @pa_solitary field so no PHV container conflict can masquerade as a
 * geometry limit. Disposable, resource-oracle only -- generated by
 * scripts/tcam_column_sweep.py, see reviews/p4_tofino_reference.md Sec 7.
 */

#include <core.p4>
#include <tna.p4>

#include "headers.p4"
#include "util.p4"

{pragmas}\
struct metadata_t {{
    bit<{key_bits}> {key_field};
{result_fields}}}

parser SwitchIngressParser(
        packet_in pkt,
        out header_t hdr,
        out metadata_t ig_md,
        out ingress_intrinsic_metadata_t ig_intr_md) {{

    TofinoIngressParser() tofino_parser;

    state start {{
        tofino_parser.apply(pkt, ig_intr_md);
        transition parse_ethernet;
    }}

    state parse_ethernet {{
        pkt.extract(hdr.ethernet);
        transition select(hdr.ethernet.ether_type) {{
            ETHERTYPE_IPV4: parse_ipv4;
            default: accept;
        }}
    }}

    state parse_ipv4 {{
        pkt.extract(hdr.ipv4);
        transition accept;
    }}
}}

control SwitchIngressDeparser(
        packet_out pkt,
        inout header_t hdr,
        in metadata_t ig_md,
        in ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md) {{
    apply {{
        pkt.emit(hdr);
    }}
}}

control SwitchIngress(
        inout header_t hdr,
        inout metadata_t meta,
        in ingress_intrinsic_metadata_t ig_intr_md,
        in ingress_intrinsic_metadata_from_parser_t ig_prsr_md,
        inout ingress_intrinsic_metadata_for_deparser_t ig_dprsr_md,
        inout ingress_intrinsic_metadata_for_tm_t ig_tm_md) {{

{tables}
    apply {{
{applies}\
        ig_tm_md.ucast_egress_port = ig_intr_md.ingress_port;
        ig_tm_md.bypass_egress = 1w1;
    }}
}}

Pipeline(SwitchIngressParser(),
         SwitchIngress(),
         SwitchIngressDeparser(),
         EmptyEgressParser(),
         EmptyEgress(),
         EmptyEgressDeparser()) pipe;

Switch(pipe) main;
"""


def synthetic_program(block_width, n_tables,
                      size=TERNARY_MATCHING_ENTRIES_PER_BLOCK):
    """One probe program's P4 source."""
    if n_tables < 1:
        raise ValueError('n_tables must be >= 1, got %r' % (n_tables,))
    key_bits = key_bits_for(block_width)

    pragmas = ''.join('@pa_solitary("ingress", "ig_md.result_%d")\n' % i
                      for i in range(n_tables))
    result_fields = ''.join('    bit<1> result_%d;\n' % i
                            for i in range(n_tables))
    tables = ''
    for i in range(n_tables):
        # A distinct action per table: one shared action would let p4c merge
        # the tables, and one shared WRITTEN field would reintroduce exactly
        # the action dependency the @pa_solitary pragmas are here to remove.
        tables += (
            '    action set_result_{i}(bit<1> r) {{ meta.result_{i} = r; }}\n'
            '    table tern_table_{i} {{\n'
            '        key = {{ meta.{key} : ternary; }}\n'
            '        actions = {{ set_result_{i}; NoAction; }}\n'
            '        const default_action = NoAction();\n'
            '        size = {size};\n'
            '    }}\n'.format(i=i, key=KEY_FIELD, size=size))
    applies = ''.join('        tern_table_%d.apply();\n' % i
                      for i in range(n_tables))

    return _PROGRAM_TEMPLATE.format(
        n_tables=n_tables, block_width=block_width, key_bits=key_bits,
        size=size, key_field=KEY_FIELD, pragmas=pragmas,
        result_fields=result_fields, tables=tables, applies=applies)


def measure(logs_dir):
    """What the compiler actually did with one probe program, or None when the
    backend allocated nothing (same contract as
    compiler_calibration._committed_blocks -- a refused program has no
    committed placement, and inventing one would record a failed compile as a
    measurement).

    `max_tables_per_stage` is a max rather than a mean on purpose: the stage
    the compiler FILLED is the capacity, and a trailing stage holding the
    remainder says nothing about the limit."""
    blocks = _committed_blocks(logs_dir)
    if blocks is None:
        return None
    stages = committed_table_stages(logs_dir)

    probe = {name: stage for name, stage in stages.items()
             if name.startswith('tern_table_')}
    if not probe:
        return None
    per_stage = collections.Counter(probe.values())
    blocks_per_stage = collections.Counter()
    for name, stage in probe.items():
        blocks_per_stage[stage] += blocks[name]
    counted = {blocks[name] for name in probe}
    return {
        'n_tables_placed': len(probe),
        'occupied_stages': len(per_stage),
        'max_tables_per_stage': max(per_stage.values()),
        'max_blocks_per_stage': max(blocks_per_stage.values()),
        # One value if every table cost the same, which is what the sweep
        # intends; reported as-is rather than averaged so a surprise shows.
        'blocks_per_table': (counted.pop() if len(counted) == 1
                             else sorted(counted)),
    }


def verdict(rows):
    """Which rule the measurements support, over the discriminating widths
    only.

    Each row's 'block_width' must be the width the compiler ACTUALLY charged
    (`blocks_per_table`), never the width the program was built for. The
    sweep's own control found those diverge: at intended widths 6, 8 and 10
    the extra-block anomaly makes the real table 7, 9 and 11 blocks wide, and
    judging such a point at its intended width reports a discriminating
    result from a measurement that discriminates nothing.

    'flat_24' / 'two_columns' when every discriminating width agrees with that
    rule; 'split' when they disagree with each other -- the honest outcome if
    neither rule as stated survives, and the one that avoids repeating the
    "suggestive, not established" mistake of calling it from a single width;
    'refutes_both' when a stage held MORE than the flat cap; 'inconclusive'
    when no discriminating width was measured."""
    discriminating = [r for r in rows
                      # A table wider than a column measures the SPANNING
                      # question (the wide arm), not the sharing one: the
                      # column rule predicts 0 tables per stage there, which
                      # is not a prediction about how many share a stage. Only
                      # widths where both rules make a real, differing
                      # prediction can decide between them.
                      if tables_per_stage_two_columns(r['block_width']) > 0
                      and tables_per_stage_flat_24(r['block_width'])
                      != tables_per_stage_two_columns(r['block_width'])]
    if not discriminating:
        return 'inconclusive'

    verdicts = set()
    for row in discriminating:
        width, observed = row['block_width'], row['max_tables_per_stage']
        if observed > tables_per_stage_flat_24(width):
            return 'refutes_both'
        if observed == tables_per_stage_flat_24(width):
            verdicts.add('flat_24')
        elif observed == tables_per_stage_two_columns(width):
            verdicts.add('two_columns')
        else:
            verdicts.add('neither')
    if verdicts == {'flat_24'}:
        return 'flat_24'
    if verdicts == {'two_columns'}:
        return 'two_columns'
    return 'split'


def sweep_points(widths=BLOCK_WIDTHS):
    """(point_id, block_width, n_tables) for every compile the sweep runs.

    Two N per width. The first is the flat-24 cap itself: under flat-24 those
    tables all fit in ONE stage, under the column rule they cannot, so a
    single compile separates the rules at the discriminating widths. The
    second is a saturating N, high enough that at least one stage is filled to
    whatever the real cap is even if the compiler balances rather than packs
    greedily."""
    points = []
    for width in widths:
        saturating = 2 * tables_per_stage_flat_24(width) + 1
        for n_tables in sorted({tables_per_stage_flat_24(width), saturating}):
            # The key width goes in the id: it is what actually determines the
            # block cost, and pinning it here means a change to key_bits_for
            # cannot silently resume onto artifacts built from the old one.
            points.append(('w%dk%d_n%d' % (width, key_bits_for(width), n_tables),
                           width, n_tables))
    return points


def wide_table_points(widths=(7, 8, 12), entries_multiplier=2):
    """(point_id, block_width, n_tables, size) for the follow-up arm: ONE
    table too wide for a single column.

    The main sweep settles how many W-block tables share a stage, which is a
    statement about tables that each fit in a column. It says nothing about a
    table needing MORE than TCAM_ROWS_PER_STAGE blocks -- can its blocks span
    both columns, or does it need a second stage? That question is not
    academic: it decides whether _stage_shards should split a logical table at
    24 blocks (the flat pool) or at 12 (one column), and the answer must be
    measured rather than inferred from the main arm.

    Width is raised by ENTRIES, not by key bits: the key is already capped at
    the 64-byte crossbar (12 blocks), so `entries_multiplier` blocks of rows
    is the only way past it."""
    points = []
    for width in widths:
        size = entries_multiplier * TERNARY_MATCHING_ENTRIES_PER_BLOCK
        points.append(('wide_w%dk%d_s%d' % (width, key_bits_for(width), size),
                       width, 1, size))
    return points


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    return set(pd.read_csv(out_path)['point_id'])


def run_one_point(point_id, block_width, n_tables, output_root,
                  size=TERNARY_MATCHING_ENTRIES_PER_BLOCK):
    """Generates, compiles and measures ONE probe program. Raises on a
    toolchain failure; collect() isolates that per point."""
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(synthetic_program(block_width, n_tables, size=size))

    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an existing dir (see compile_p4's
        # docstring); a stale one means a prior attempt never finished.
        shutil.rmtree(compile_dir)
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')

    row = {
        'point_id': point_id, 'block_width': block_width,
        'n_tables': n_tables, 'key_bits': key_bits_for(block_width),
        'size': size,
        # What the table SHOULD cost: key width sets blocks per 512 entries,
        # and `size` multiplies that. The wide arm raises blocks through size
        # on purpose, so the control below has to compare against this rather
        # than against block_width, or it flags the wide arm as broken.
        'expected_blocks': block_width * math.ceil(
            size / TERNARY_MATCHING_ENTRIES_PER_BLOCK),
        'predicted_flat_24': tables_per_stage_flat_24(block_width),
        'predicted_two_columns': tables_per_stage_two_columns(block_width),
        'compile_errors': result.errors, 'stages_real': result.stages,
        'tcam_real': result.tcam,
    }
    measured = measure(os.path.join(compile_dir, 'pipe', 'logs'))
    if measured is None:
        row.update({'n_tables_placed': None, 'occupied_stages': None,
                    'max_tables_per_stage': None, 'max_blocks_per_stage': None,
                    'blocks_per_table': None})
    else:
        row.update(measured)
    return row


def collect(out, output_root, widths=BLOCK_WIDTHS, limit=None, arm='main'):
    """Compiles the remaining sweep points sequentially, writing each to `out`
    as it completes. Resumable by point_id, and one bad point does not lose
    the run -- the same shape as compiler_calibration.collect."""
    if arm == 'wide':
        points = [(pid, w, n, size) for pid, w, n, size in wide_table_points()]
    else:
        points = [(pid, w, n, TERNARY_MATCHING_ENTRIES_PER_BLOCK)
                  for pid, w, n in sweep_points(widths)]
    done = already_done(out)
    remaining = [p for p in points if p[0] not in done]
    if limit is not None:
        remaining = remaining[:limit]
    if not remaining:
        print('nothing to do -- every sweep point already recorded at %s' % out)
        return (pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()), []

    os.makedirs(output_root, exist_ok=True)
    file_exists = os.path.exists(out) and os.path.getsize(out) > 0
    failed = []
    for point_id, block_width, n_tables, size in remaining:
        print('compiling %s (%d tables x %d blocks, size=%d) ...'
              % (point_id, n_tables, block_width, size))
        started = time.time()
        try:
            row = run_one_point(point_id, block_width, n_tables, output_root,
                                size=size)
        except Exception as e:
            failed.append((point_id, '%s: %s' % (type(e).__name__, e)[:200]))
            print('  %s raised:\n%s' % (point_id, traceback.format_exc()))
            continue
        pd.DataFrame([row]).to_csv(out, mode='a', header=not file_exists,
                                    index=False)
        file_exists = True
        print('  [%s] %.1fs -- blocks/table=%s max tables/stage=%s '
              '(flat24 predicts %s, 2x12 predicts %s)'
              % (point_id, time.time() - started, row.get('blocks_per_table'),
                 row.get('max_tables_per_stage'), row['predicted_flat_24'],
                 row['predicted_two_columns']))

    frame = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    return frame, failed


def report(frame, failed=()):
    if frame.empty:
        print('\nno sweep points recorded yet.')
        return
    print('\n### Per-point measurements\n')
    columns = ['point_id', 'block_width', 'n_tables', 'size', 'expected_blocks',
               'blocks_per_table', 'max_tables_per_stage', 'predicted_flat_24',
               'predicted_two_columns', 'occupied_stages', 'stages_real']
    print(frame[[c for c in columns if c in frame.columns]].to_string(index=False))

    print('\n### Control: did each table really cost the width it was built for?\n')
    if {'blocks_per_table', 'expected_blocks'} <= set(frame.columns):
        wrong = frame[frame['blocks_per_table'].astype(str)
                      != frame['expected_blocks'].astype(str)]
        if len(wrong):
            print('%d point(s) where the compiler disagreed with key_bits_for '
                  '-- the probe is not measuring what it intends there:'
                  % len(wrong))
            print(wrong[['point_id', 'expected_blocks',
                         'blocks_per_table']].to_string(index=False))
        else:
            print('OK -- every table cost exactly its intended block width')

    print('\n### Verdict (widths 7 and 8 are the only discriminating ones)\n')
    measured = frame[frame['max_tables_per_stage'].notna()] if \
        'max_tables_per_stage' in frame.columns else frame.iloc[0:0]
    # Judged at the width the compiler CHARGED, not the width intended -- see
    # verdict()'s docstring and the control section above.
    rows = [{'block_width': int(r['blocks_per_table']),
             'max_tables_per_stage': int(r['max_tables_per_stage'])}
            for _, r in measured.iterrows()
            if str(r['blocks_per_table']).isdigit()]
    print(verdict(rows))
    for width in sorted({r['block_width'] for r in rows}):
        observed = max(r['max_tables_per_stage'] for r in rows
                       if r['block_width'] == width)
        flat, columns = (tables_per_stage_flat_24(width),
                         tables_per_stage_two_columns(width))
        if columns == 0:
            note = '   (wider than a column -- spanning arm, not a verdict)'
        elif flat != columns:
            note = '   <== discriminating'
        else:
            note = ''
        print('  w=%-2d observed max %d/stage   flat24=%d  2x12=%d%s'
              % (width, observed, flat, columns, note))

    if failed:
        print('\n%d point(s) raised a toolchain exception (retryable):' % len(failed))
        for point_id, summary in failed:
            print('  %s: %s' % (point_id, summary))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--widths', type=lambda v: tuple(int(x) for x in v.split(',')),
                        default=BLOCK_WIDTHS)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--arm', choices=('main', 'wide'), default='main',
                        help="'main' sweeps N tables of W blocks (W=5..12); "
                             "'wide' compiles ONE table too wide for a single "
                             "column, to settle whether a table can span both")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame, failed = collect(args.out, args.output_root, widths=args.widths,
                            limit=args.limit, arm=args.arm)
    report(frame, failed)


if __name__ == '__main__':
    main()
