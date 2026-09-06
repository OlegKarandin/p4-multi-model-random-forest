"""Mechanism G: when do two DIFFERENT ternary keys share a stage's TCAM?

WHY THIS EXISTS. `independent_low_sd9` is the last stage divergence left by
the 2026-09-05/06 calibration study (reviews/p4_tofino_reference.md Sec 7,
final bullet): the model packs 2 app tables (9 blocks each) and 2 ddos tables
(3 each) into one stage -- 24 blocks, 9+3 | 9+3 across the two TCAM columns,
61 of 64 crossbar bytes -- and the compiler never puts an app table and a ddos
table in the same stage at all. Four causes were ruled out and two hypotheses
refuted by measurement.

Reading p4c's own placement log for that row names the mechanism:

    table SwitchIngress.get_classification_tree_app_0 could not fit in stage 6
    find_ternary_stretch failed

`Memories::find_ternary_stretch` (bf-p4c/mau/tofino/memories.cpp:1761) does
not test a block TOTAL and does not test a column SUBSET-SUM, which is what
`evaluation.fits_two_columns` models. It scans column 0 then column 1, row 0
to 11, for a run of free rows, and it refuses to START that run on an odd row
unless the table's width is odd AND the midbyte it would then share with the
row above is compatible. Two TCAM rows 2i and 2i+1 share one crossbar midbyte
(Sec 4.1.1), so a stretch that starts odd splits a midbyte with whatever table
already sits above it.

Consequence, if the rule is what it looks like: a stage's free TCAM rows are
not fungible. `independent_low_sd9`'s stage 6 has NINE contiguous free rows
(column 1, rows 3..11, measured from resources.json) and a nine-block table
was refused, because row 3 is odd and belongs to a midbyte already claimed by
the table at row 2.

WHAT THIS SWEEP MEASURES. NA tables keying one wide field, NB tables keying a
second, narrower one, all in one stage's worth of blocks, compiled through the
real WSL2 p4c; it reads back which stage each table landed in AND the exact
(column, row) each block got. The confounders are removed the same way
scripts/tcam_column_sweep.py removes them -- every table writes its own
`@pa_solitary` result field (so Mechanism A cannot masquerade as geometry) and
`size = 512` (so a table's block count comes from its key width alone). What
this probe deliberately does NOT share is the KEY: two distinct fields is the
whole point, since a single shared key can never split a midbyte.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_stretch_sweep.py

Resumable: points already recorded in --out are skipped.
"""
import argparse
import collections
import json
import math
import os
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.compiler_calibration import committed_table_stages
from src.p4gen.build_p4_script import (
    TCAM_BLOCK_KEY_LENGTH,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_MATCHING_ENTRIES_PER_BLOCK,
)
from src.p4gen.p4_compile import compile_p4

# Sec 4.1: a ternary entry carries a flat 4-bit version/valid overhead before
# the key is divided into TCAM_BLOCK_KEY_LENGTH-bit rows.
TERNARY_ENTRY_OVERHEAD_BITS = 4

DEFAULT_OUT = 'results/tcam_stretch_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_stretch_sweep'


def as_fields(key):
    """A key spec as a tuple of field BIT widths.

    An int means one solid field of that many BYTES -- the shape that isolates
    geometry, since every crossbar byte it presents is fully used. A tuple
    means the real generator's shape: several `code_<feature>` fields whose
    widths are interval counts and therefore ragged, each contributing its own
    part-used final byte. The two are not interchangeable and the difference
    is what this sweep measures."""
    if isinstance(key, int):
        return (8 * key,)
    return tuple(key)


def key_bytes_for(key):
    """Crossbar bytes a key costs: Sec 4.1.1's per-FIELD byte rounding, which
    is `evaluation.ternary_table_key_bytes`."""
    return sum(math.ceil(bits / 8) for bits in as_fields(key))


def blocks_for_key(key):
    """The TCAM blocks this key costs, both arms of Sec 4.1's formula.

    The max is kept rather than assuming the arms agree, so a point whose real
    cost differs from its intent shows up as a control failure rather than as
    a silently mis-labelled measurement."""
    match_bits = sum(as_fields(key))
    return max(math.ceil((match_bits + TERNARY_ENTRY_OVERHEAD_BITS)
                         / TCAM_BLOCK_KEY_LENGTH),
               math.ceil(8 * key_bytes_for(key) / TCAM_BLOCK_KEY_LENGTH))


_PROGRAM_TEMPLATE = """\
/*
 * Mechanism G probe: {n_a} table(s) on a {bytes_a}-byte key ({blocks_a} TCAM
 * blocks each) and {n_b} on a {bytes_b}-byte key ({blocks_b} blocks each),
 * {total_blocks} blocks in total against a stage's {stage_blocks}. The two
 * keys are DIFFERENT fields on purpose: find_ternary_stretch's midbyte rule
 * can only bind between tables that do not share a key. Every action writes
 * its own @pa_solitary field so no PHV container conflict can masquerade as a
 * geometry limit. Disposable, resource-oracle only -- generated by
 * scripts/tcam_stretch_sweep.py, see reviews/p4_tofino_reference.md Sec 7.
 */

#include <core.p4>
#include <tna.p4>

#include "headers.p4"
#include "util.p4"

{key_pragmas}{pragmas}\
struct metadata_t {{
{key_fields}{result_fields}}}

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


def synthetic_program(key_a, n_a, key_b, n_b,
                      size=TERNARY_MATCHING_ENTRIES_PER_BLOCK):
    """One probe program's P4 source.

    Table emission order is A then B; `allocate_all_ternary` re-sorts by key
    width anyway (memories.cpp:1800), so program order is not what decides the
    layout, but keeping it fixed keeps the artifacts comparable."""
    if n_a < 1 or n_b < 0:
        raise ValueError('need n_a >= 1 and n_b >= 0, got %r and %r'
                         % (n_a, n_b))
    fields = {'a': as_fields(key_a), 'b': as_fields(key_b)}
    names = ([('a', i) for i in range(n_a)] + [('b', i) for i in range(n_b)])

    key_pragmas = ''.join(
        '@pa_solitary("ingress", "ig_md.key_%s%d")\n' % (tag, j)
        for tag in ('a', 'b') for j in range(len(fields[tag])))
    key_fields = ''.join(
        '    bit<%d> key_%s%d;\n' % (bits, tag, j)
        for tag in ('a', 'b') for j, bits in enumerate(fields[tag]))
    pragmas = ''.join('@pa_solitary("ingress", "ig_md.result_%s%d")\n'
                      % (tag, i) for tag, i in names)
    result_fields = ''.join('    bit<1> result_%s%d;\n' % (tag, i)
                            for tag, i in names)
    tables = ''
    for tag, i in names:
        # One key line per FIELD, exactly as build_p4_script emits one
        # `meta.code_<feature> : ternary` line per selected feature -- the
        # crossbar allocates per field, so a key's field split is part of
        # what is under test, not a formatting detail.
        key_lines = ''.join(
            '            meta.key_%s%d : ternary;\n' % (tag, j)
            for j in range(len(fields[tag])))
        # A distinct action per table: one shared action would let p4c merge
        # the tables, and one shared WRITTEN field would reintroduce exactly
        # the action dependency the @pa_solitary pragmas are here to remove.
        tables += (
            '    action set_result_{tag}{i}(bit<1> r) '
            '{{ meta.result_{tag}{i} = r; }}\n'
            '    table tern_{tag}{i} {{\n'
            '        key = {{\n{key_lines}        }}\n'
            '        actions = {{ set_result_{tag}{i}; NoAction; }}\n'
            '        const default_action = NoAction();\n'
            '        size = {size};\n'
            '    }}\n'.format(tag=tag, i=i, key_lines=key_lines, size=size))
    applies = ''.join('        tern_%s%d.apply();\n' % (tag, i)
                      for tag, i in names)

    blocks_a, blocks_b = blocks_for_key(key_a), blocks_for_key(key_b)
    return _PROGRAM_TEMPLATE.format(
        n_a=n_a, n_b=n_b,
        bytes_a=key_bytes_for(key_a), bytes_b=key_bytes_for(key_b),
        blocks_a=blocks_a, blocks_b=blocks_b,
        total_blocks=n_a * blocks_a + n_b * blocks_b,
        stage_blocks=TCAM_ROWS_PER_STAGE * TCAM_COLUMNS_PER_STAGE,
        key_pragmas=key_pragmas, key_fields=key_fields, pragmas=pragmas,
        result_fields=result_fields, tables=tables, applies=applies)


def committed_tcam_grid(logs_dir):
    """stage -> {(column, row): table basename}, the compiler's OWN committed
    TCAM allocation, read from resources.json.

    mau.resources.log gives a table's stage and its block COUNT;
    find_ternary_stretch's rule is about WHICH rows those blocks got, which
    only this file records. Returns {} when the backend never allocated (the
    same contract as compiler_calibration._committed_blocks)."""
    path = os.path.join(logs_dir, 'resources.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8', errors='replace') as handle:
        document = json.load(handle)
    grids = {}
    for stage in document['resources']['mau']['mau_stages']:
        cells = (stage.get('tcams') or {}).get('tcams') or []
        grid = {}
        for cell in cells:
            for usage in cell['usages']:
                grid[(cell['column'], cell['row'])] = \
                    usage['used_by'].split('.')[-1]
        if grid:
            grids[stage['stage_number']] = grid
    return grids


def spans(grid):
    """{table: (column, first row, block count)} for one stage's grid.

    A table whose blocks are NOT one contiguous run in one column is reported
    with a None column, so a surprise cannot be silently averaged away: every
    single-column placement in 53 archived compiles is contiguous, and the
    wide arm of scripts/tcam_column_sweep.py is the only thing that ever spans
    both columns."""
    cells = collections.defaultdict(list)
    for (column, row), table in grid.items():
        cells[table].append((column, row))
    out = {}
    for table, placed in cells.items():
        columns = {column for column, _ in placed}
        rows = sorted(row for _, row in placed)
        contiguous = rows == list(range(rows[0], rows[0] + len(rows)))
        single = columns.pop() if len(columns) == 1 and contiguous else None
        out[table] = (single, rows[0], len(placed))
    return out


def measure(logs_dir):
    """What the compiler did with one probe program, or None when the backend
    allocated nothing (a refused program has no committed placement, and
    inventing one would record a failed compile as a measurement)."""
    stages = committed_table_stages(logs_dir)
    if stages is None:
        return None
    probe = {name: stage for name, stage in stages.items()
             if name.startswith('tern_')}
    if not probe:
        return None
    grids = committed_tcam_grid(logs_dir)

    by_stage = collections.defaultdict(list)
    for name, stage in probe.items():
        by_stage[stage].append(name)
    shared = sorted(stage for stage, names in by_stage.items()
                    if {n[len('tern_')] for n in names} == {'a', 'b'})

    layout = {}
    for stage, grid in sorted(grids.items()):
        layout[stage] = {name: span for name, span in spans(grid).items()
                         if name.startswith('tern_')}
    return {
        'n_tables_placed': len(probe),
        'occupied_stages': len(by_stage),
        'stages_sharing_both_keys': len(shared),
        # The question the sweep exists to answer, as one boolean.
        'both_keys_in_one_stage': bool(shared),
        'max_tables_per_stage': max(len(v) for v in by_stage.values()),
        'layout': json.dumps({str(stage): {name: list(span)
                                           for name, span in tables.items()}
                              for stage, tables in layout.items()},
                             sort_keys=True),
    }


# independent_low_sd9's own two classification keys, field for field: its app
# trees key code_app_fwd_packet_length_max (179 bits) + code_packet_length_mean
# (204) = 49 crossbar bytes at 9 blocks, its ddos trees
# code_bwd_packet_length_mean (37) + code_ddos_fwd_packet_length_max (49) = 12
# bytes at 3 blocks. Read from the row's own generated P4, not reconstructed.
SD9_APP_KEY = (179, 204)
SD9_DDOS_KEY = (37, 49)


def sweep_points():
    """(point_id, key_a, n_a, key_b, n_b) for every compile the sweep runs.

    An int key is one solid field of that many bytes; a tuple is a list of
    field bit widths. The two arms answer different questions and the split is
    the point of the sweep:

      * SOLID arm -- same block counts and same crossbar byte totals as
        independent_low_sd9, but each key one fully-used field. Isolates
        GEOMETRY: block widths, column packing, start rows.
      * RAGGED arm -- sd9's real field widths. Same geometry, but every field
        contributes a part-used final crossbar byte, which is what decides
        which blocks carry a midbyte (Sec 4.1.1) and therefore which start
        rows find_ternary_stretch will accept.

    If the two arms differ, the constraint is not geometry and no width-based
    packing rule can express it."""
    return [
        # --- solid arm -------------------------------------------------
        # 24 blocks as 9+3 | 9+3: the packing the model builds for
        # independent_low_sd9 and the compiler never produced.
        ('a49x2_b12x2', 49, 2, 12, 2),
        # 24 blocks as (9 | 3+3+3+3) + 3: the arrangement the compiler
        # actually reached that row with, then refused to complete.
        ('a49x1_b12x5', 49, 1, 12, 5),
        # 21 blocks, one column free from row 0: the control that says the
        # refusal above is about WHERE the free rows are, not how many.
        ('a49x1_b12x4', 49, 1, 12, 4),
        # 12 blocks, nothing tight anywhere: a floor for the probe itself.
        ('a49x1_b12x1', 49, 1, 12, 1),
        # Four EVEN-width tables filling all 24 blocks on two keys, which is
        # also Sec 4.3's 2 x 32-byte crossbar saturation re-measured on the
        # current toolchain -- the measurement that refuted the 60-byte cap.
        ('a32x2_b32x2', 32, 2, 32, 2),
        # Four ODD-width (5-block) tables on two keys: 20 blocks, and the
        # second table in a column has to start on row 5.
        ('a27x2_b27x2', 27, 2, 27, 2),
        # --- ragged arm: sd9's real keys -------------------------------
        ('ragged_ax1_bx5', SD9_APP_KEY, 1, SD9_DDOS_KEY, 5),
        ('ragged_ax2_bx2', SD9_APP_KEY, 2, SD9_DDOS_KEY, 2),
        ('ragged_ax1_bx4', SD9_APP_KEY, 1, SD9_DDOS_KEY, 4),
    ]


def already_done(out_path):
    if not os.path.exists(out_path):
        return set()
    return set(pd.read_csv(out_path)['point_id'])


def run_one_point(point_id, key_a, n_a, key_b, n_b, output_root,
                  size=TERNARY_MATCHING_ENTRIES_PER_BLOCK):
    """Generates, compiles and measures ONE probe program. Raises on a
    toolchain failure; collect() isolates that per point."""
    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(synthetic_program(key_a, n_a, key_b, n_b, size=size))

    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        # p4c refuses to write into an existing dir (see compile_p4's
        # docstring); a stale one means a prior attempt never finished.
        shutil.rmtree(compile_dir)
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')

    blocks_a, blocks_b = blocks_for_key(key_a), blocks_for_key(key_b)
    bytes_a, bytes_b = key_bytes_for(key_a), key_bytes_for(key_b)
    row = {
        'point_id': point_id,
        'fields_a': ','.join(str(b) for b in as_fields(key_a)),
        'fields_b': ','.join(str(b) for b in as_fields(key_b)),
        'bytes_a': bytes_a, 'n_a': n_a, 'blocks_a': blocks_a,
        'bytes_b': bytes_b, 'n_b': n_b, 'blocks_b': blocks_b,
        'total_blocks': n_a * blocks_a + n_b * blocks_b,
        'total_bytes': bytes_a + bytes_b,
        'size': size,
        'compile_errors': result.errors, 'stages_real': result.stages,
        'tcam_real': result.tcam,
    }
    measured = measure(os.path.join(compile_dir, 'pipe', 'logs'))
    if measured is None:
        row.update({'n_tables_placed': None, 'occupied_stages': None,
                    'stages_sharing_both_keys': None,
                    'both_keys_in_one_stage': None,
                    'max_tables_per_stage': None, 'layout': None})
    else:
        row.update(measured)
    return row


def collect(out, output_root, limit=None):
    """Compiles the remaining sweep points sequentially, writing each to `out`
    as it completes. Resumable by point_id, and one bad point does not lose
    the run -- the same shape as compiler_calibration.collect."""
    points = sweep_points()
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
    for point_id, key_a, n_a, key_b, n_b in remaining:
        print('compiling %s (%d x %s + %d x %s) ...'
              % (point_id, n_a, as_fields(key_a), n_b, as_fields(key_b)))
        started = time.time()
        try:
            row = run_one_point(point_id, key_a, n_a, key_b, n_b,
                                output_root)
        except Exception as e:
            failed.append((point_id, '%s: %s' % (type(e).__name__, e)[:200]))
            print('  %s raised:\n%s' % (point_id, traceback.format_exc()))
            continue
        pd.DataFrame([row]).to_csv(out, mode='a', header=not file_exists,
                                   index=False)
        file_exists = True
        print('  [%s] %.1fs -- %s blocks over %s stage(s), both keys share a '
              'stage: %s' % (point_id, time.time() - started,
                             row['total_blocks'], row.get('occupied_stages'),
                             row.get('both_keys_in_one_stage')))

    frame = pd.read_csv(out) if os.path.exists(out) else pd.DataFrame()
    return frame, failed


def report(frame, failed=()):
    if frame.empty:
        print('\nno sweep points recorded yet.')
        return
    print('\n### Per-point measurements\n')
    columns = ['point_id', 'fields_a', 'blocks_a', 'n_a', 'fields_b',
               'blocks_b', 'n_b', 'total_blocks', 'total_bytes',
               'occupied_stages', 'both_keys_in_one_stage',
               'max_tables_per_stage', 'stages_real']
    print(frame[[c for c in columns if c in frame.columns]]
          .to_string(index=False))

    print('\n### Committed TCAM layout, per stage (column, first row, blocks)\n')
    for _, row in frame.iterrows():
        print('  %s:' % row['point_id'])
        if not isinstance(row.get('layout'), str):
            print('     (no committed allocation)')
            continue
        for stage, tables in sorted(json.loads(row['layout']).items(),
                                    key=lambda kv: int(kv[0])):
            rendered = ', '.join(
                '%s@col%s row%s x%s' % (name, span[0], span[1], span[2])
                for name, span in sorted(tables.items()))
            print('     stage %s: %s' % (stage, rendered))

    if failed:
        print('\n%d point(s) raised a toolchain exception (retryable):'
              % len(failed))
        for point_id, summary in failed:
            print('  %s: %s' % (point_id, summary))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--limit', type=int, default=None)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    frame, failed = collect(args.out, args.output_root, limit=args.limit)
    report(frame, failed)


if __name__ == '__main__':
    main()
