"""Replaying already-compiled p4c artifacts through the resource model.

Pure parsing of a generated program (p4_src/<row>.p4) and of p4c's committed
logs (compiles/<row>/pipe/logs), plus two replays that pack the result with
the model's own packer. Split out of scripts/compiler_calibration.py so that
scripts/validation_table.py can replay held-out designs without importing
sklearn or the campaign backup: this module imports only src.p4model and the
standard library. compiler_calibration re-exports every name, so existing
callers are unaffected.
"""
import math
import os
import re

from src.p4model.packing import crossbar_stages_needed
from src.p4model.program import (PLACEMENT_PRIORITY, SHARED_TASK, TASKS,
                                  VOTE_EPILOGUE_STAGES)
from src.p4model.ranges import compiler_range_rows
from src.p4model.registers import gated_block_interior_stages, readiness_levels_for
from src.p4model.tables import codeword_to_blocks, tree_entries_to_blocks
from src.p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK
from src.p4model.usage import tree_readiness_levels


# The committed-allocation parsers live in the library now (spec 2026-09-29
# §6.2: library code never imports from scripts/). Re-exported under their
# old names so every script and test that imports them from here is unchanged.
from src.p4gen.p4_ground_truth import (  # noqa: F401
    RESOURCE_TABLE_ROW as _RESOURCE_TABLE_ROW,
    committed_blocks as _committed_blocks,
    committed_stages_real,
    committed_table_stages,
)


def _p4_table_keys(p4_path):
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
    with open(p4_path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
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


def committed_register_stages(logs_dir):
    """register base name -> the stage the compiler ran its RegisterAction in,
    from the COMMITTED allocation in mau.resources.log.

    The direct measurement behind evaluation.METER_ALUS_PER_STAGE: a stage's
    register count never exceeds 4 in any of these compiles, and the
    Percentage table reports 4 as 100% of the stage's Meter ALUs."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    stages = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = _RESOURCE_TABLE_ROW.match(line)
        if match and match.group(1).endswith('_reg'):
            name = match.group(1).split('.')[-1]
            stages[name[:-len('_reg')]] = int(match.group(2))
    return stages


def placement_round_states(logs_dir):
    """The placement rounds p4c actually ran, in order, e.g. ['INITIAL'] or
    ['INITIAL', 'NOCC_TRY1', 'REDO_PHV1'].

    NOCC_TRY is a container-conflicts-DISABLED re-placement of the identical
    program -- a free controlled experiment for Mechanism A, and the
    measurement its "PHV delta" column comes from. But p4c only runs it when
    the initial round leaves it something to retry, so a row with no NOCC_TRY
    round has no counterfactual at all and its delta is 0 by absence, not by
    measurement. Reading that 0 as "no PHV effect here" is what mis-attributed
    open_issues.md item 16; this function makes the distinction checkable."""
    path = os.path.join(logs_dir, 'table_summary.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    return re.findall(r'Table allocation done \d+ time\(s\), state = (\S+)', text)


def _latest_round_log(logs_dir, prefix):
    """The HIGHEST-numbered <prefix>_<N>.log in logs_dir.

    These logs exist once per placement round and round 0 is the DISCARDED
    initial allocation, exactly as table_summary.log's first stage count is
    (see committed_stages_real). Reading the lowest instead is what cost the
    first calibration pass its "+1 range packing" conclusion."""
    numbered = []
    for name in os.listdir(logs_dir):
        match = re.fullmatch(re.escape(prefix) + r'_(\d+)\.log', name)
        if match:
            numbered.append((int(match.group(1)), name))
    if not numbered:
        raise ValueError('no {}_<N>.log in {}'.format(prefix, logs_dir))
    return os.path.join(logs_dir, max(numbered)[1])


_PHV_ADVANCE_RE = re.compile(
    r'action dependency between (\S+) and table (\S+) due to PHV allocation '
    r'advances stage to (\d+)')


def phv_advanced_tables(logs_dir):
    """table basename -> the stage p4c's own placement log says a PHV-induced
    action dependency pushed it to (the LAST such stage, if it was pushed more
    than once).

    This is Mechanism A stated by the compiler rather than inferred: a Tofino
    stage has one action ALU per PHV CONTAINER, so two match tables whose
    actions write fields the allocator packed into one container cannot share
    a stage, and p4c logs the resulting advance verbatim:

        - action dependency between table_14_ddos_bwd_packet_length_max_0 and
          table table_13_ddos_bwd_iat_min_0 due to PHV allocation advances
          stage to 9

    The SECOND table named is the one that moves. Placement-log names carry a
    trailing `_<n>` suffix that mau.resources.log's do not (the generator's own
    table names never end in a digit -- they end in a feature name), so it is
    stripped to keep both keyed the same way."""
    with open(_latest_round_log(logs_dir, 'table_placement'),
              encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    advanced = {}
    for _blocker, moved, stage in _PHV_ADVANCE_RE.findall(text):
        name = re.sub(r'_\d+$', '', moved)
        advanced[name] = max(int(stage), advanced.get(name, 0))
    return advanced


def replay_stage_depth(row_id, artifacts_root, readiness_levels=None):
    """Returns (predicted_stage_depth, committed_stages_real) for one
    already-compiled calibration row.

    Feeds the estimator's packer the REAL per-table facts from the row's own
    generated P4 and committed compile logs -- key field identities and
    widths, physical block counts, per-feature readiness levels -- so the
    only thing being compared is stage PLACEMENT. Deliberately does not
    refit the archived model: that would fold the block model's own error
    (see this module's V5) into a stage measurement.

    readiness_levels (optional) overrides the model's own per-feature levels
    with a {raw_feature_name: level} mapping, so a caller can substitute a
    different level source and see what the packer then does with it. It
    exists for one measurement: feeding in an ORACLE -- levels read off the
    compiler's own committed register placement, one stage past each feature's
    last register -- moves the predicted depth on none of the 18 rows, which
    is what killed open_issues.md item 16's readiness-accuracy hypothesis.
    Default None keeps readiness_levels_for, i.e. the real model.

    Raises ValueError when the backend produced no resource allocation,
    which is how a program that does not fit the chip shows up (see
    independent_high_sd12: "tofino supports up to 12 stages, using 13").
    Returning a plausible number there would price an infeasible design as a
    cheap one."""
    logs_dir = os.path.join(artifacts_root, 'compiles', row_id, 'pipe', 'logs')
    blocks = _committed_blocks(logs_dir)
    if blocks is None:
        raise ValueError(
            '%s: the compiler never allocated resources for this program '
            '(mau.resources.log has no "Allocated Resource Usage" section), so '
            'there is no committed placement to replay. Its table placement '
            'needs %d stages against Tofino\'s 12.'
            % (row_id, committed_stages_real(logs_dir)))

    tables, widths, bits = _p4_table_keys(
        os.path.join(artifacts_root, 'p4_src', row_id + '.p4'))
    predicted, _, _ = _replay_plans(row_id, tables, widths, bits, blocks,
                                    readiness_levels)
    return predicted, committed_stages_real(logs_dir)


def _p4_table_sizes(p4_path):
    """table name -> its declared `size = N`, the entry count the generator
    writes per table (build_p4_script: real per-tree and per-feature sizes)."""
    with open(p4_path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
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


def replay_design(row_id, artifacts_root):
    """Returns (predicted_stage_depth, predicted_blocks) for one compiled
    row, END TO END: every table priced by the model's own per-table prices,
    placed by the model's own packer. Nothing is read from the compile except
    the program's table list, keys, entry counts and feature set -- all of
    which the model would also see before compiling. Compare replay_stage_depth,
    which feeds p4c's committed block counts in to isolate placement.

    Classification table: tables.codeword_to_blocks(key field bits) x
    tables.tree_entries_to_blocks(declared entries). Range table:
    ceil(ranges.compiler_range_rows(declared entries) / 512), the compile-time
    sizing. Works on rows p4c could not place too (no committed allocation is
    needed), which is how a held-out too-deep design is checked for
    never-under on depth."""
    p4_path = os.path.join(artifacts_root, 'p4_src', row_id + '.p4')
    tables, widths, bits = _p4_table_keys(p4_path)
    return replay_program(row_id, tables, widths, bits, _p4_table_sizes(p4_path))


def replay_program(row_id, tables, widths, bits, sizes):
    """replay_design's model half, on an already-parsed program: tables
    (name -> key fields), widths/bits (field -> crossbar bytes / declared
    bits) and sizes (table -> declared entries), as _p4_table_keys and
    _p4_table_sizes return them. Split out so a committed fixture can replay a
    design without the gitignored p4_src/ file
    (tests/fixtures/c1_replay_designs.json)."""
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
    predicted, range_plan, ternary_plan = _replay_plans(
        row_id, tables, widths, bits, blocks, None)
    return predicted, range_plan.blocks + ternary_plan.blocks


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


def _replay_plans(row_id, tables, widths, bits, blocks, readiness_levels):
    """Packs one program's range and classification pools with the model's
    packer, given a table -> block-count map. Shared by replay_stage_depth
    (p4c's committed counts) and replay_design (the model's own prices).
    Returns (predicted_stage_depth, range_plan, ternary_plan)."""
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
    return predicted, range_plan, ternary_plan
