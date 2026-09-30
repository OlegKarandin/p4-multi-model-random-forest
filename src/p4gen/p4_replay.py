"""Program-level replay: a generated P4 program (or its parsed form) through the
resource model, per table (spec 2026-09-29 sec 5.3, 6.6 --rescore).

Moved out of scripts/p4_artifact_replay.py so library code never imports from
scripts/; the script keeps thin wrappers with the old signatures."""
import math
import re
from collections import namedtuple
from dataclasses import dataclass

from src.p4model.packing import crossbar_stages_needed
from src.p4model.program import (PLACEMENT_PRIORITY, SHARED_TASK, TASKS,
                                 VOTE_EPILOGUE_STAGES)
from src.p4model.ranges import compiler_range_rows
from src.p4model.registers import gated_block_interior_stages, readiness_levels_for
from src.p4model.tables import codeword_to_blocks, tree_entries_to_blocks
from src.p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK
from src.p4model.usage import tree_readiness_levels


@dataclass(frozen=True)
class Program:
    """One parsed generated program: tables (name -> key fields), widths /
    bits (field -> crossbar bytes / declared bits), sizes (table -> declared
    entries)."""
    tables: dict
    widths: dict
    bits: dict
    sizes: dict


ReplayPlans = namedtuple('ReplayPlans', 'stage_depth range_plan ternary_plan '
                                        'range_names ternary_names')


def _parse_keys(text):
    """(table name -> [match key field names], field -> byte width,
    field -> BIT width) for one generated program.

    The widths come from the metadata declarations (`bit<N> code_<f>;` /
    `bit<N> <f>_val;`), byte-rounded per FIELD because that is how the ternary
    crossbar allocates -- see evaluation.codeword_fields_to_bytes. The raw bit
    widths come back too because a field that is not a whole number of bytes
    hands the crossbar a part-used byte, which is what decides whether a
    midbyte nibble survives for the version field
    (tables.tail_is_isolatable, consumed by tables.codeword_to_blocks's Sec
    2.3 isolation credit)."""
    widths, bits = {}, {}
    for pattern in (r"bit<(\d+)>\s+(code_\w+)\s*;", r"bit<(\d+)>\s+(\w+_val)\s*;"):
        for match in re.finditer(pattern, text):
            widths[match.group(2)] = math.ceil(int(match.group(1)) / 8)
            bits[match.group(2)] = int(match.group(1))
    tables, current = {}, None
    for line in text.splitlines():
        opened = re.match(r"\s*table\s+(\w+)\s*\{", line)
        if opened:
            current = opened.group(1)
            tables[current] = []
            continue
        key = re.match(r"\s*meta\.(\w+)\s*:\s*(ternary|range)\s*;", line)
        if key and current:
            tables[current].append(key.group(1))
    return tables, widths, bits
def _parse_sizes(text):
    """table name -> its declared `size = N`, the entry count the generator
    writes per table (build_p4_script: real per-tree and per-feature sizes)."""
    sizes, current = {}, None
    for line in text.splitlines():
        opened = re.match(r"\s*table\s+(\w+)\s*\{", line)
        if opened:
            current = opened.group(1)
            continue
        size = re.match(r"\s*size\s*=\s*(\d+)\s*;", line)
        if size and current:
            sizes[current] = int(size.group(1))
    return sizes

def parse_program_text(text):
    tables, widths, bits = _parse_keys(text)
    return Program(tables, widths, bits, _parse_sizes(text))


def parse_program(p4_path):
    with open(p4_path, encoding='utf-8', errors='replace') as handle:
        return parse_program_text(handle.read())


_TREE_TASK = re.compile(r'^get_classification_tree_(%s)_\d+$' % '|'.join(TASKS))
_RANGE_TABLE_CODE = re.compile(r'^table_\d+_(\w+)$')


def table_tasks(tables):
    """{table name: task label} for every classification tree and range table
    of one generated program -- the replay's counterpart of
    evaluation._pool_inputs' ternary_task/range_task labels (APP_TASK,
    DDOS_TASK, SHARED_TASK).

    A tree's task is in its name, get_classification_tree_<task>_<i>. A range
    table's is NOT, reliably: build_p4_script names it table_<n>_<resolved>,
    where resolved is task-prefixed (app_<f>/ddos_<f>) only when both models
    split feature f with DIFFERENT intervals; a feature one model selects
    alone, or both split identically, stays un-prefixed
    (_resolve_disjoint_feature_plan). So the name gives the code field the
    table writes, meta.code_<resolved>, and the task comes from which trees
    key on that field: one task's trees -> that task, both -> SHARED_TASK.
    A field no tree keys on (not emitted today) is labelled SHARED_TASK, so
    every tree waits for it -- the side that cannot under-count."""
    labels, readers = {}, {}
    for name, keys in tables.items():
        match = _TREE_TASK.match(name)
        if match:
            labels[name] = match.group(1)
            for key in keys:
                readers.setdefault(key, set()).add(match.group(1))
        elif name.startswith('get_classification_tree'):
            raise ValueError('%s: a classification tree must name its task (%s)'
                             % (name, ', '.join(TASKS)))
    for name in tables:
        match = _RANGE_TABLE_CODE.match(name)
        if match:
            tasks = readers.get('code_' + match.group(1), set())
            labels[name] = next(iter(tasks)) if len(tasks) == 1 else SHARED_TASK
    return labels

def declared_prices(program):
    """The model's own per-table price for every priced table of a program
    (replay_program's pricing loop)."""
    tables, bits, sizes = program.tables, program.bits, program.sizes
    blocks = {}
    for name, keys in tables.items():
        if not keys or name not in sizes:
            continue
        if name.startswith('get_classification_tree'):
            key_bits = tuple(sorted(bits[key] for key in keys))
            blocks[name] = (codeword_to_blocks(key_bits)
                            * tree_entries_to_blocks(sizes[name]))
        elif name.startswith('table_'):
            blocks[name] = -(-compiler_range_rows(sizes[name])
                             // TERNARY_MATCHING_ENTRIES_PER_BLOCK)
    return blocks


def replay_plans(row_id, program, blocks, readiness_levels=None):
    """Packs one program's range and classification pools with the model's
    packer, given a table -> block-count map. Shared by replay_stage_depth
    (p4c's committed counts) and replay_design (the model's own prices).
    Returns a ReplayPlans; the name lists are aligned with the plans' table_specs."""
    tables, widths, bits = program.tables, program.widths, program.bits
    # A range table keys on exactly one meta.<raw_feature>_val field. Levels
    # come from one schedule over the row's WHOLE feature set, not per
    # feature: the stateful-ALU cap is a property of the set (see
    # evaluation.register_stage_schedule). Two tables of different models may
    # name the same feature -- the register behind it is emitted once, so it
    # appears in the schedule once.
    row_features = []
    for name, keys in tables.items():
        if name.startswith('table_') and name in blocks and keys:
            feature = keys[0][:-len('_val')]
            if feature not in row_features:
                row_features.append(feature)
    if readiness_levels is None:
        levels = dict(zip(row_features, readiness_levels_for(row_features)))
    else:
        missing = [f for f in row_features if f not in readiness_levels]
        if missing:
            raise ValueError(
                '%s: readiness_levels names no level for %s; every feature with '
                'a range table must have one or the packer would place it at an '
                'arbitrary stage' % (row_id, ', '.join(sorted(missing))))
        levels = readiness_levels
    # Mechanism B: stages the placer spends wholly inside a gated register
    # block hold no table from the outer sequence, however empty they are.
    interior = gated_block_interior_stages(row_features)

    tasks = table_tasks(tables)
    range_specs, range_fields, range_levels, range_task = [], [], [], []
    range_names, ternary_names = [], []
    ternary_specs, ternary_fields, ternary_key_bits, ternary_task = [], [], [], []
    for name, keys in tables.items():
        if name not in blocks or not keys:
            continue
        fields = frozenset((key, widths[key]) for key in keys)
        width = sum(field_bytes for _, field_bytes in fields)
        if name.startswith('table_'):
            range_specs.append((blocks[name], width))
            range_fields.append(fields)
            range_levels.append(levels[keys[0][:-len('_val')]])
            range_task.append(tasks[name])
            range_names.append(name)
        elif name.startswith('get_classification_tree'):
            # blocks[name] is either the model's own standalone price
            # (codeword_to_blocks x rows, replay_design) or the count p4c
            # committed (replay_stage_depth). Either way it is what the table
            # is charged as its stage's FIRST key; the packer's ordered stage
            # simulation charges a table placed behind a different key its
            # lane leftover price instead (packing.crossbar_stages_needed).
            # Program order is kept: p4c breaks placement-priority ties by
            # taking the table listed LAST first.
            key_bits = tuple(sorted(bits[key] for key in keys))
            ternary_specs.append((blocks[name], width))
            ternary_fields.append(fields)
            ternary_key_bits.append(key_bits)
            ternary_task.append(tasks[name])
            ternary_names.append(name)

    range_plan = crossbar_stages_needed(range_specs, readiness_levels=range_levels,
                                         key_fields=range_fields,
                                         unavailable_stages=interior)
    # Per-task tree readiness and the seeded ternary pool, exactly as
    # usage.assemble_usage does it (audit C1): a tree waits for its own task's
    # range tables and the shared ones, and may land in a stage still holding
    # the other task's range tables, which count against that stage's limits.
    # key_field_bits and placement_priority (the generator's
    # @placement_priority per task) switch on the ordered stage simulation,
    # which decides both WHERE trees land and what a tree behind a different
    # key in its stage pays -- so they matter even when the block counts are
    # p4c's own.
    ternary_plan = crossbar_stages_needed(
        ternary_specs,
        readiness_levels=tree_readiness_levels(range_plan.table_stages,
                                               range_task, ternary_task),
        key_fields=ternary_fields, unavailable_stages=interior,
        key_field_bits=ternary_key_bits,
        seed_stages=range_plan.stage_loads,
        placement_priority=[PLACEMENT_PRIORITY[task] for task in ternary_task])

    predicted = (max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES)
    return ReplayPlans(predicted, range_plan, ternary_plan, range_names,
                       ternary_names)


def model_breakdown(program, row_id=''):
    """Everything the model says about one generated program, per table, by
    the program's own table names -- so spec sec 6.3's per-table comparison with
    p4c's mau.resources.log is a name join, not a mapping. The model side of
    every campaign verdict (plan O3): the path the 60/60 calibration gate
    validates, and the one --rescore reruns from a saved program."""
    plans = replay_plans(row_id, program, declared_prices(program))
    tables = []
    for kind, plan, names in (('range', plans.range_plan, plans.range_names),
                              ('ternary', plans.ternary_plan, plans.ternary_names)):
        for name, blocks, stage in zip(names, plan.table_blocks, plan.table_stages):
            tables.append({'table': name, 'kind': kind,
                           'blocks': int(blocks), 'stage': int(stage)})
    tables.sort(key=lambda t: (t['stage'], t['table']))
    return {'stage_depth': int(plans.stage_depth),
            'blocks': int(plans.range_plan.blocks + plans.ternary_plan.blocks),
            'tables': tables}
