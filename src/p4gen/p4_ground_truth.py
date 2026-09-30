"""p4c's COMMITTED allocation, parsed from a compile's pipe/logs directory.

Library home of the parsers scripts/p4_artifact_replay.py used to own (spec
2026-09-29 §6.2: library code never imports from scripts/)."""
from dataclasses import dataclass
import json
import os
import re
from typing import Optional

from src.p4gen.p4_compile import parse_compile_logs


RESOURCE_TABLE_ROW = re.compile(
    r"^\|\s*(\S+)\s*\|\s*(-?\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|\s*(\d+)\s*\|"
    r"\s*(\d+)\s*\|\s*(\d+)\s*\|")


def committed_blocks(logs_dir):
    """table basename -> physical TCAM block count, from the COMMITTED
    allocation in mau.resources.log. Returns None when the backend never got
    as far as allocating resources (see replay_stage_depth's raise)."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    blocks = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = RESOURCE_TABLE_ROW.match(line)
        if match and not match.group(1).endswith('$action'):
            blocks[match.group(1).split('.')[-1]] = int(match.group(7))
    return blocks


def committed_table_stages(logs_dir):
    """table basename -> the stage the compiler placed it in, from the
    COMMITTED allocation in mau.resources.log.

    The measurement Mechanism B is checked against: a stage the placer spends
    entirely inside a gated register block can hold no table of the outer
    sequence, so it shows up here as a HOLE in the range pool's occupancy --
    a stage index between two occupied ones that no table landed in. Returns
    None when the backend never allocated resources, exactly as
    committed_blocks does."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    stages = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = RESOURCE_TABLE_ROW.match(line)
        if match and not match.group(1).endswith('$action'):
            stages[match.group(1).split('.')[-1]] = int(match.group(2))
    return stages


def committed_stages_real(logs_dir):
    """The compiler's FINAL stage count for a program.

    table_summary.log holds one "Number of stages in table allocation" line
    per placement round (INITIAL, then NOCC_TRY/REDO_PHV retries), and only
    the LAST is the allocation the compiler commits to -- cross-checked
    against mau.resources.log, which only ever reflects the committed one.
    Reading the first instead (a bare re.search) over-reported 4 of this
    study's 19 rows by 1-2 stages."""
    path = os.path.join(logs_dir, 'table_summary.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        last_round = handle.read().split('Table allocation done ')[-1]
    return int(re.search(r'stages in table allocation:\s*(\d+)',
                          last_round).group(1))


def phv_containers(logs_dir):
    """Whole-program PHV container count: the sum of containers_occupied over
    metrics.json's phv.normal groups. Tagalong (TPHV) containers are a
    separate, non-matchable pool and are not counted. Matches the pragmas
    archive manifest's phv_containers on all 73 designs
    (tests/test_p4_ground_truth.py). None when metrics.json is absent."""
    path = os.path.join(logs_dir, 'metrics.json')
    if not os.path.isfile(path):
        return None
    with open(path, encoding='utf-8') as handle:
        phv = json.load(handle).get('phv', {})
    return sum(group['containers_occupied'] for group in phv.get('normal', []))


@dataclass(frozen=True)
class P4cNumbers:
    """What one compile committed to. None always means "p4c did not get that
    far", never 0: blocks/table_blocks are None when the program never reached
    resource allocation (over 12 stages), stage_depth None when it never
    reached table placement."""
    stage_depth: Optional[int]
    blocks: Optional[int]
    table_blocks: Optional[dict]
    sram: Optional[int]
    map_ram: Optional[int]
    phv_containers: Optional[int]

    @property
    def allocated(self):
        return self.table_blocks is not None


def p4c_numbers(logs_dir):
    """P4cNumbers for one compile's pipe/logs directory. Never raises on a
    missing file."""
    def present(name):
        return os.path.isfile(os.path.join(logs_dir, name))

    table_blocks = committed_blocks(logs_dir) if present('mau.resources.log') else None
    try:
        stage_depth = (committed_stages_real(logs_dir)
                       if present('table_summary.log') else None)
    except AttributeError:
        stage_depth = None
    totals = parse_compile_logs(logs_dir)
    return P4cNumbers(
        stage_depth=stage_depth,
        blocks=None if table_blocks is None else sum(table_blocks.values()),
        table_blocks=table_blocks,
        sram=totals.sram, map_ram=totals.map_ram,
        phv_containers=phv_containers(logs_dir))
