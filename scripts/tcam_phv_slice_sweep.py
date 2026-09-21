"""Does the version block depend on whether p4c ISOLATES a field's nibble tail?

WHY THIS EXISTS. `scripts/tcam_lane_sweep.py` showed the version charge on a
saturating key is decided by something positional, but its crude PHV-lane
model missed one point in ten. Reading the two disagreeing compiles'
`prog.bfa` showed what the real discriminator is, and it is not a lane
residue -- it is whether the nibble-clean TAIL of a field ends up as its own
crossbar entry:

  FREE  (lane_a4)   byte group 2: { 0: ig_md.key_a0.8-35(24..27) }   <- 4 bits
        two match lines, version rides the byte group's other nibble.

  PAYS  (lane_a3)   byte group 0: { 0: ig_md.key_a1.24-55(8..15) }   <- 8 bits
        three match lines, the third being `- { byte_config: 3 }`
        with NO group at all: a whole TCAM block holding only --version--.

So the question is not "which crossbar position can this byte reach" but
"does the compiler slice this field so its 1-4 leftover bits become a
standalone nibble entry, or does it fold them into a wider chunk that needs
a whole byte slot". This sweep measures that directly, from the assembly,
instead of inferring it from block counts.

THE DESIGN. Every point is B = 11 crossbar bytes, g = 2 groups, exactly
saturating, with exactly ONE nibble-clean field (S = 1). Exactly one byte
must therefore ride a byte group (midbyte), and the version charge fires iff
that occupant is a whole byte rather than a nibble. Only the clean field's
BIT WIDTH varies, which moves its leftover bits between container positions:

  tail 1-4    (W = 1-4, 33-36)    leftover in the container's byte 0
  tail 9-12   (W = 9-12, 41-44)   leftover in byte 1
  tail 17-20  (W = 17-20, 49-52)  leftover in byte 2
  tail 25-28  (W = 25-28, 57-60)  leftover in byte 3

with W <= 32 fitting a single container and W > 32 spanning two, so the same
tail is sampled in both regimes. Four points per cell rather than the one
the lane sweep had.

WHAT IS RECORDED. Per point, straight out of `prog.bfa`: every `byte group`
occupant and its BIT width, the number of `match:` lines (= TCAM blocks) and
whether one of them is a group-less version-only line; plus the committed
block count from resources.json as a cross-check, and the clean field's PHV
container from phv_allocation_summary.

Run it (from the repository root; needs the WSL2 p4c toolchain):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/tcam_phv_slice_sweep.py

24 real p4c compiles, a few seconds each. Resumable: points already present
in --out are skipped.
"""
import argparse
import os
import re
import shutil
import sys
import time
import traceback

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

from scripts.tcam_stretch_sweep import key_bytes_for, synthetic_program
from scripts.tcam_version_sweep import read_committed
from src.p4gen.p4_compile import compile_p4
from src.p4model.tables import codeword_to_blocks

DEFAULT_OUT = 'results/tcam_phv_slice_sweep.csv'
DEFAULT_OUTPUT_ROOT = 'results/tcam_phv_slice_sweep'

TOTAL_BYTES = 11          # saturating at g = 2: 10 whole bytes + 1 overflow

# Clean-field bit widths, grouped by which container byte their leftover bits
# land in, sampled both below and above the 32-bit container boundary.
CLEAN_WIDTHS = (
    [3, 4]                       # tail byte 0, single container   (control)
    + [11, 12]                   # tail byte 1, single container   (control)
    + [17, 18, 19, 20]           # tail byte 2, single container   (contested)
    + [25, 26, 27, 28]           # tail byte 3, single container   (contested)
    + [35, 36]                   # tail byte 0, two containers     (control)
    + [43, 44]                   # tail byte 1, two containers     (control)
    + [49, 50, 51, 52]           # tail byte 2, two containers     (contested)
    + [57, 58, 59, 60]           # tail byte 3, two containers     (contested)
)

_MATCH_LINE = re.compile(r'^\s*-\s*\{(.*)\}\s*$')
_BYTE_GROUP = re.compile(r'^\s*byte group (\d+):\s*\{(.*)\}\s*$')
_TERNARY_GROUP = re.compile(r'^\s*ternary group (\d+):')
_OCCUPANT_RANGE = re.compile(r'\((\d+)\.\.(\d+)\)')
_SLICE_RANGE = re.compile(r'\.(\d+)-(\d+)(?:\(|$|\s)')


def occupant_bits(occupant, field_widths):
    """BIT width of whatever p4c put in a byte group.

    `ig_md.key_a0.8-35(24..27)` -> the (24..27) sub-range, 4 bits (a nibble).
    `ig_md.key_a1.0-15`         -> the 0-15 slice, but a byte group holds at
                                   most 8 bits, so such an entry is the whole
                                   byte it names.
    `ig_md.key_a0`              -> the field itself, width from the caller."""
    match = _OCCUPANT_RANGE.search(occupant)
    if match:
        return int(match.group(2)) - int(match.group(1)) + 1
    match = _SLICE_RANGE.search(occupant)
    if match:
        return min(8, int(match.group(2)) - int(match.group(1)) + 1)
    for name, width in field_widths.items():
        if occupant.strip().endswith(name):
            return min(8, width)
    return None


def read_bfa_table(bfa_path, field_widths, table='tern_a0'):
    """The `ternary_match` stanza for `table`: its byte groups, its match
    lines, and whether one of those lines is a version-only block.

    Takes the FIRST stanza only -- p4c repeats the identical input_xbar under
    the table's `ternary_indirect` companion, which is the same allocation
    reported twice, not a second one."""
    with open(bfa_path, encoding='utf-8', errors='replace') as handle:
        lines = handle.read().splitlines()
    head = re.compile(r'^(\s*)ternary_match %s_\d+ ' % re.escape(table))
    start = indent = None
    for index, line in enumerate(lines):
        found = head.match(line)
        if found:
            start, indent = index, len(found.group(1))
            break
    if start is None:
        return {}

    byte_groups, match_lines, ternary_groups = [], [], 0
    in_match = False
    for line in lines[start + 1:]:
        if line.strip() and (len(line) - len(line.lstrip())) <= indent:
            break                       # next table stanza
        if _TERNARY_GROUP.match(line):
            ternary_groups += 1
            continue
        found = _BYTE_GROUP.match(line)
        if found:
            byte_groups.append((int(found.group(1)), found.group(2).strip()))
            continue
        if line.strip() == 'match:':
            in_match = True
            continue
        found = _MATCH_LINE.match(line)
        if in_match and found:
            match_lines.append(found.group(1))
            continue
        if in_match and line.strip() and not found:
            in_match = False

    occupants = []
    for number, body in byte_groups:
        for entry in body.split(','):
            if ':' not in entry:
                continue
            occupant = entry.split(':', 1)[1]
            occupants.append((number, occupant.strip(),
                              occupant_bits(occupant, field_widths)))

    version_only = [m for m in match_lines if 'group:' not in m]
    return {
        'ternary_groups': ternary_groups,
        'byte_group_occupants': '; '.join(
            '%d:%s=%sb' % (n, o, b) for n, o, b in occupants),
        'byte_group_min_bits': min([b for _n, _o, b in occupants if b], default=None),
        'byte_group_holds_whole_byte': any(b == 8 for _n, _o, b in occupants),
        'match_lines': len(match_lines),
        'version_only_blocks': len(version_only),
    }


def read_phv_container(logs_dir, field='ig_md.key_a0'):
    """The container(s) the clean field landed in, from p4c's PHV summary."""
    path = os.path.join(logs_dir, 'phv_allocation_summary_0.log')
    if not os.path.exists(path):
        return ''
    out = []
    with open(path, encoding='utf-8', errors='replace') as handle:
        for line in handle:
            if field not in line:
                continue
            cells = [c.strip() for c in line.strip().strip('|').split('|')]
            if cells and re.match(r'^[BHW]\d+$', cells[0]):
                out.append('%s%s' % (cells[0], cells[2] if len(cells) > 2 else ''))
    return ' '.join(out)


def run_point(point_id, clean_bits, output_root, size):
    solid_bytes = TOTAL_BYTES - -(-clean_bits // 8)
    fields = (clean_bits, 8 * solid_bytes)
    field_widths = {'key_a0': fields[0], 'key_a1': fields[1]}

    p4_dir = os.path.join(output_root, 'p4_src')
    os.makedirs(p4_dir, exist_ok=True)
    source = synthetic_program(fields, 1, (8,), 0, size=size)
    p4_path = os.path.join(p4_dir, point_id + '.p4')
    with open(p4_path, 'w', encoding='utf-8') as handle:
        handle.write(source)
    compile_dir = os.path.join(output_root, 'compiles', point_id)
    os.makedirs(os.path.dirname(compile_dir), exist_ok=True)
    if os.path.isdir(compile_dir):
        shutil.rmtree(compile_dir)
    started = time.time()
    result = compile_p4(p4_path, compile_dir,
                        include_path='p4/tofino_spike/common')

    logs = os.path.join(compile_dir, 'pipe', 'logs')
    committed = read_committed(logs)
    rec = committed.get('tern_a0')
    bfa = read_bfa_table(os.path.join(compile_dir, 'pipe', 'prog.bfa'),
                         field_widths)

    tail = clean_bits % 32 or 32
    row = {
        'point_id': point_id,
        'clean_bits': clean_bits,
        'solid_bits': fields[1],
        'key_bytes': key_bytes_for(fields),
        'tail_bits': tail,
        'tail_byte_index': -(-tail // 8) - 1,
        'containers': 'single' if clean_bits <= 32 else 'multi',
        'phv_containers': read_phv_container(logs),
        'pred_current': codeword_to_blocks(fields),
        'real_blocks': rec['blocks'] if rec else None,
        'compile_errors': result.errors if result.errors is not None else 0,
        'seconds': round(time.time() - started, 1),
    }
    row.update(bfa)
    return row


def summarize(rows):
    frame = pd.DataFrame(rows)
    if frame.empty:
        print('\n(no rows)')
        return
    print('\n### PHV-slice sweep -- B = 11 bytes, g = 2, S = 1\n')
    cols = ['point_id', 'clean_bits', 'tail_bits', 'tail_byte_index',
           'containers', 'byte_group_min_bits', 'byte_group_holds_whole_byte',
           'match_lines', 'version_only_blocks', 'real_blocks', 'pred_current']
    print(frame[[c for c in cols if c in frame.columns]].to_string(index=False))

    done = frame[frame['real_blocks'].notna()]
    if done.empty:
        return
    print('\n### Does "byte group holds a whole byte" predict the charge?\n')
    pays = done['real_blocks'] > 2
    whole = done['byte_group_holds_whole_byte'] == True     # noqa: E712
    print('  pays and byte group holds a whole byte : %d' % (pays & whole).sum())
    print('  free and byte group holds a nibble     : %d' % (~pays & ~whole).sum())
    print('  DISAGREEMENTS                          : %d'
          % ((pays & ~whole) | (~pays & whole)).sum())

    print('\n### Charge by (tail byte index, containers)\n')
    table = done.pivot_table(index='tail_byte_index', columns='containers',
                             values='real_blocks', aggfunc=lambda v: '/'.join(
                                 str(int(x)) for x in v))
    print(table.to_string())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', default=DEFAULT_OUT)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--size', type=int, default=512)
    args = parser.parse_args(argv)

    os.makedirs(args.output_root, exist_ok=True)
    done, rows = set(), []
    if os.path.exists(args.out):
        existing = pd.read_csv(args.out)
        rows = existing.to_dict('records')
        done = set(existing['point_id'])

    for clean_bits in CLEAN_WIDTHS:
        point_id = 'w%03d' % clean_bits
        if point_id in done:
            continue
        print('=== %s -- clean field %d bits' % (point_id, clean_bits), flush=True)
        try:
            row = run_point(point_id, clean_bits, args.output_root, args.size)
        except Exception:
            traceback.print_exc()
            row = {'point_id': point_id, 'clean_bits': clean_bits,
                   'compile_errors': -1}
        rows.append(row)
        pd.DataFrame(rows).to_csv(args.out, index=False)
        print('    byte_group=%s  whole_byte=%s  match_lines=%s  REAL=%s' % (
            row.get('byte_group_occupants'), row.get('byte_group_holds_whole_byte'),
            row.get('match_lines'), row.get('real_blocks')), flush=True)

    print('\nwrote %s (%d points)' % (args.out, len(rows)))
    summarize(rows)


if __name__ == '__main__':
    main()
