"""Program-level replay: a generated P4 program (or its parsed form) through the
resource model, per table (spec 2026-09-29 sec 5.3, 6.6 --rescore).

Moved out of scripts/p4_artifact_replay.py so library code never imports from
scripts/; the script keeps thin wrappers with the old signatures."""
import math
import re
from collections import namedtuple
from dataclasses import dataclass, field

from p4model.errors import CrossbarKeyTooWide
from p4model.packing import (_alone_blocks, crossbar_stages_needed,
                              packed_alone_blocks)
from p4model.program import (PLACEMENT_PRIORITY, SHARED_TASK,
                                 VOTE_EPILOGUE_STAGES, feature_of_per_task_field,
                                 feature_of_per_tree_field)
from p4model.ranges import compiler_range_rows
from p4model.registers import readiness_levels_for
from p4model import lanes, packed, xbar
from p4model.tables import tree_entries_to_blocks
from p4model.target import TERNARY_MATCHING_ENTRIES_PER_BLOCK
from p4model.usage import tree_readiness_levels


@dataclass(frozen=True)
class Program:
    """One parsed generated program: tables (name -> key fields), widths /
    bits (field -> crossbar bytes / declared bits), sizes (table -> declared
    entries), priorities (table -> its @placement_priority, for every table
    that carries one).

    A packed per-tree program (spec 2026-11-11-packed-codes-stage-pin Sec 3.1)
    fills two more: containers (code_pk_<f> -> its @pa_container_size sizes in
    bits, low container first; empty for every other program) and key_slices
    ((table, field) -> [(hi, lo)], each `meta.<field>[hi:lo]` key line of that
    table in program order; every sliced key of any program is recorded).

    code_containers: every other code_<f> field's @pa_container_size sizes in
    bits, low container first (the generator pins every tree-key code field,
    build_p4_script.code_field_container_sizes) -- the containers the crossbar
    port lays the field's key bytes in (tree_key_slices)."""
    tables: dict
    widths: dict
    bits: dict
    sizes: dict
    priorities: dict = field(default_factory=dict)
    containers: dict = field(default_factory=dict)
    key_slices: dict = field(default_factory=dict)
    code_containers: dict = field(default_factory=dict)


ReplayPlans = namedtuple('ReplayPlans', 'stage_depth range_plan ternary_plan '
                                        'range_names ternary_names')

# The packed per-tree code field of feature f (build_p4_script.packed_field):
# meta.code_pk_<f>, written whole by range table table_<n>_<f>.
_PACKED_FIELD = 'code_pk_'


def _parse_keys(text):
    """(table name -> [match key field names], field -> byte width,
    field -> BIT width, (table, field) -> [(hi, lo)] key slices) for one
    generated program.

    The widths come from the metadata declarations (`bit<N> code_<f>;` /
    `bit<N> <f>_val;`), byte-rounded per FIELD because that is how the ternary
    crossbar allocates -- see evaluation.codeword_fields_to_bytes. The raw bit
    widths come back too because a field that is not a whole number of bytes
    hands the crossbar a part-used byte, which is what decides whether a
    midbyte nibble survives for the version field
    (tables.tail_is_isolatable, consumed by the ladder tables.codeword_to_blocks; the production price
    is lanes.table_blocks)."""
    widths, bits = {}, {}
    for pattern in (r"bit<(\d+)>\s+(code_\w+)\s*;", r"bit<(\d+)>\s+(\w+_val)\s*;"):
        for match in re.finditer(pattern, text):
            widths[match.group(2)] = math.ceil(int(match.group(1)) / 8)
            bits[match.group(2)] = int(match.group(1))
    # A key may read a slice of its field (`meta.code_x[26:0]`): a code field
    # declared at its container's full width so that p4c's action format has
    # no padding bits to condense another code into (results/
    # encoding_evidence_2026_11/adb_investigation). The crossbar sees the
    # slice, so the slice width is the field's width for pricing. A field
    # wider than 64 bits is keyed as several slices (p4c takes no wider
    # slice): one key entry, the slices' widths summed.
    # A packed field (code_pk_<f>) is the exception: every tree keys its OWN
    # slices of it, so its slices are no one key's width; they are kept per
    # (table, field) instead (tree_key_bytes) and its declared bits stay the
    # field's container bits.
    tables, current, sliced, slices = {}, None, {}, {}
    for line in text.splitlines():
        opened = re.match(r"\s*table\s+(\w+)\s*\{", line)
        if opened:
            current = opened.group(1)
            tables[current] = []
            continue
        key = re.match(r"\s*meta\.(\w+)(?:\[(\d+):(\d+)\])?\s*:\s*(ternary|range)\s*;", line)
        if key and current:
            if key.group(1) not in tables[current]:
                tables[current].append(key.group(1))
            if key.group(2) is not None:
                hi, lo = int(key.group(2)), int(key.group(3))
                slices.setdefault((current, key.group(1)), []).append((hi, lo))
                if not key.group(1).startswith(_PACKED_FIELD):
                    sliced[key.group(1)] = sliced.get(key.group(1), 0) + hi - lo + 1
    for field, width in sliced.items():
        bits[field] = width
        widths[field] = math.ceil(width / 8)
    return tables, widths, bits, slices
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

_PLACEMENT_PRIORITY = re.compile(r"@placement_priority\((\d+)\)\s*\n\s*table\s+(\w+)\s*\{")


def _parse_priorities(text):
    """table name -> the @placement_priority pragma directly above it (the
    generator emits one on every classification tree table, N - i for task i
    of N -- program.placement_priorities -- and, on a program generated since
    2026-11, N + 1 on every range table -- program.range_table_priority -- which
    the replay does not need: the model places the range pool first either way)."""
    return {name: int(priority)
            for priority, name in _PLACEMENT_PRIORITY.findall(text)}


# A gated register block: an `if (meta.fwd == 0|1) { ... }` in the apply block
# whose body executes a RegisterAction (`<name>_action.execute(`), which every
# program generated before decisions 1a / 7.1 option B (2026-11-14) carries.
# A direction test inside a RegisterAction body (`if (meta.fwd == 1) { value =
# ..._halve_unit.execute(...); }`) is not one.
_GATED_BLOCK = re.compile(
    r'if\s*\(\s*meta\.fwd\s*==\s*[01]\s*\)\s*\{[^{}]*?\b\w+_action\.execute\(')


# A packed code field's container pin (build_p4_script, spec 2026-11-11 Sec 3.1):
# sizes in bits, low container first -- packed.Layout.containers' own form.
_PACKED_CONTAINERS = re.compile(
    r'^@pa_container_size\("ingress", "ig_md\.(%s\w+)", ([\d, ]+)\)\s*$'
    % _PACKED_FIELD, re.M)

# Any code field's container pin; the packed ones are kept apart (containers).
_CODE_CONTAINERS = re.compile(
    r'^@pa_container_size\("ingress", "ig_md\.(code_\w+)", ([\d, ]+)\)\s*$', re.M)

# A table's @stage(n) line. Per-tree programs generated under spec 2026-11-11
# Sec 4.3 (heldout runs up to heldout8, before the crossbar port of spec
# 2026-11-13) carry them; nothing generates them now, and the model does not
# price them.
_STAGE_PIN = re.compile(r'^\s*@stage\(\d+\)', re.M)


def carries_stage_pins(text):
    """True when the program text has @stage lines, which parse_program_text
    refuses (src.verify's rescore checks a whole run with this up front)."""
    return bool(_STAGE_PIN.search(text))


def carries_gated_register_blocks(text):
    """True when the program executes RegisterActions inside `if (meta.fwd ==
    ...)` blocks (_GATED_BLOCK), which parse_program_text refuses."""
    return bool(_GATED_BLOCK.search(text))


def parse_program_text(text):
    """Raises ValueError for a program with @stage lines: the model no longer
    prices pinned placements, so such a program is replayed at the commit that
    generated it, never silently priced as if unpinned. Likewise for a program
    with a gated register block (carries_gated_register_blocks): the register
    schedule models one unconditional register sequence (decisions 1a / 7.1
    option B), not the block order, cursor floor and interior stages of the
    gated program."""
    if carries_stage_pins(text):
        raise ValueError('program carries @stage pins, which the model no longer '
                         'prices (spec 2026-11-13); replay it at the commit that '
                         'generated it')
    if carries_gated_register_blocks(text):
        raise ValueError('program carries a gated register block (`if (meta.fwd '
                         '== ...)` around RegisterAction executes), which the model '
                         'no longer prices (decisions 1a / 7.1 option B, '
                         '2026-11-14); replay it at the commit that generated it')
    tables, widths, bits, slices = _parse_keys(text)
    return Program(tables, widths, bits, _parse_sizes(text),
                   _parse_priorities(text),
                   containers={name: tuple(int(size) for size in sizes.split(','))
                               for name, sizes in _PACKED_CONTAINERS.findall(text)},
                   key_slices=slices,
                   code_containers={name: tuple(int(size) for size in sizes.split(','))
                                    for name, sizes in _CODE_CONTAINERS.findall(text)
                                    if not name.startswith(_PACKED_FIELD)})


def parse_program(p4_path):
    with open(p4_path, encoding='utf-8', errors='replace') as handle:
        return parse_program_text(handle.read())


# get_classification_tree_<task>_<i>: the task is everything between the
# prefix and the LAST _<digits> (task names are [a-z0-9_]+, so they may hold
# underscores and digits themselves).
_TREE_TASK = re.compile(r'^get_classification_tree_(\w+)_(\d+)$')
_RANGE_TABLE_CODE = re.compile(r'^table_\d+_(\w+)$')


def table_tasks(tables):
    """{table name: task label} for every classification tree and range table
    of one generated program -- the replay's counterpart of
    evaluation._pool_inputs' ternary_task/range_task labels (a task name,
    SHARED_TASK, or -- with three or more tasks -- a frozenset of task names;
    see usage.tree_readiness_levels).

    A tree's task is in its name, get_classification_tree_<task>_<i>. A range
    table's is NOT, reliably: build_p4_script names it table_<n>_<resolved>,
    where resolved is task-prefixed (app_<f>/ddos_<f>, or <t1>_<t2>_<f> for a
    group of tasks) only when the tasks selecting feature f split it with
    DIFFERENT intervals; a feature one task selects alone, or all of its
    selectors split identically, stays un-prefixed (resolve_feature_plan).
    So the name gives the code field the table writes, meta.code_<resolved>,
    and the task comes from which trees key on that field: one task's trees ->
    that task, every task's -> SHARED_TASK, some but not all -> the frozenset
    of them. A field no tree keys on (not emitted today) is labelled
    SHARED_TASK, so every tree waits for it -- the side that cannot
    under-count.

    Per-task codes (program.PER_TASK_ENCODING): table_<n>_<f> writes one
    code_tk_<task>_<f> per task with a threshold on f, so a reader of any of
    them is a reader of the table's feature. Per-tree codes
    (program.PER_TREE_ENCODING) likewise: the packed field code_pk_<f>, or an
    older program's code_tr_<task>_<i>_<f>, is read for the keying tree's task
    (code_written_by)."""
    labels, readers = {}, {}
    for name, keys in tables.items():
        match = _TREE_TASK.match(name)
        if match:
            labels[name] = match.group(1)
            for key in keys:
                readers.setdefault(code_written_by(key, match.group(1)),
                                   set()).add(match.group(1))
        elif name.startswith('get_classification_tree'):
            raise ValueError('%s: a classification tree must be named '
                             'get_classification_tree_<task>_<index>' % name)
    every_task = set(labels.values())
    for name in tables:
        match = _RANGE_TABLE_CODE.match(name)
        if match:
            tasks = readers.get('code_' + match.group(1), set())
            if len(tasks) == 1:
                labels[name] = next(iter(tasks))
            elif not tasks or tasks == every_task:
                labels[name] = SHARED_TASK
            else:
                labels[name] = frozenset(tasks)
    return labels


def code_written_by(key, task):
    """The code field a range table writes for a tree of `task` keying `key`:
    the key itself, or for a per-task code code_tk_<task>_<f>
    (feature_of_per_task_field), a per-tree code code_tr_<task>_<i>_<f>
    (feature_of_per_tree_field) or a packed field code_pk_<f> (spec
    2026-11-11 Sec 3.1; build_p4_script names its range table table_<n>_<f>),
    the field code_<f> that table_<n>_<f> is named after. The one place a new
    code encoding teaches the replay which range table a tree key comes from
    (table_tasks and tree_range_reads both read it). An archived code_tr_
    program (generated before 2026-11-11) is mapped here but no longer priced
    as it was generated -- its spill pins read as plain bytes -- so rescoring
    a pre-2026-11-11 per-tree run (campaign_2026_11e, heldout6) with this code
    can shift its numbers."""
    feature = feature_of_per_task_field(key, task)
    if feature is None:
        feature = feature_of_per_tree_field(key, task)
    if feature is None and key.startswith(_PACKED_FIELD):
        feature = key[len(_PACKED_FIELD):]
    return key if feature is None else 'code_' + feature


def tree_range_reads(tables):
    """{tree name: [range table names its key reads]} for one program, or None
    when some tree key matches no range table (a program this replay cannot
    map: the caller falls back to task labels). A range table table_<n>_<r>
    writes code_<r>, so a key is read from the table whose code_<r> is
    code_written_by(key, task)."""
    writer = {}
    for name in tables:
        match = _RANGE_TABLE_CODE.match(name)
        if match and tables[name]:
            writer['code_' + match.group(1)] = name
    reads = {}
    for name, keys in tables.items():
        match = _TREE_TASK.match(name)
        if not match:
            continue
        tables_read = []
        for key in keys:
            table = writer.get(code_written_by(key, match.group(1)))
            if table is None:
                return None
            if table not in tables_read:
                tables_read.append(table)
        reads[name] = tables_read
    return reads


def is_packed(program):
    """Whether `program` is a packed per-tree program: some classification
    tree keys a code_pk_ field (spec 2026-11-11 Sec 3.1)."""
    return any(key.startswith(_PACKED_FIELD)
               for name, keys in program.tables.items() if _TREE_TASK.match(name)
               for key in keys)


def range_action_bytes(program):
    """{range table name: action-data bus bytes its action writes}, plain
    bytes (the range pool counts them in 16-byte words itself, packing.
    crossbar_stages_needed's action_bytes): the code fields the trees read
    from it (code_written_by). A packed field is written whole, so it counts
    its container bytes, from its @pa_container_size pin -- packed.bus_bytes,
    what evaluation._pool_inputs charges (spec 2026-11-11 Sec 5 rule 3). Any
    other code counts its keyed bits, ceil(bits / 8) over the table's codes
    together, as evaluation._pool_inputs does from the forests; a sliced key
    counts its slices: an archived per-tree program (code_tr_, before
    2026-11-11) declared a wide code at its containers' width (rule A), but
    only the keyed bits carry data."""
    tables, bits, containers = program.tables, program.bits, program.containers
    written = {}
    for name, keys in tables.items():
        match = _TREE_TASK.match(name)
        if match:
            for key in keys:
                written.setdefault(code_written_by(key, match.group(1)), set()).add(key)
    action_bytes = {}
    for name in tables:
        match = _RANGE_TABLE_CODE.match(name)
        if match:
            fields = written.get('code_' + match.group(1), ())
            whole = sum(sum(containers[field]) // 8
                        for field in fields if field in containers)
            keyed = sum(int(bits[field]) for field in fields if field not in containers)
            action_bytes[name] = whole + -(-keyed // 8)
    return action_bytes


def tree_key_bytes(program, table):
    """One packed tree table's key as lanes.py byte tuples, in packed.key_bytes'
    format and order, rebuilt from the program text alone (spec 2026-11-11
    Sec 3.1, 3.2, 8): each code_pk_<f> key's slices give the code's byte
    offset (its lowest bit / 8) and live bits (the slices' widths summed), the
    field's @pa_container_size pin gives its containers, and packed.key_bytes
    reads the Layout those make -- the function the training pool prices
    from, so the two read one rule. Raises ValueError for a key a packed
    program cannot carry: a field with no container pin or no slice, slices
    that are not one contiguous run from a byte boundary, or a code running
    past its field's containers."""
    match = _TREE_TASK.match(table)
    if match is None:
        raise ValueError('%s is not a classification tree table' % table)
    task, tree = match.group(1), int(match.group(2))
    containers, place = {}, {}
    for key in program.tables[table]:
        if not key.startswith(_PACKED_FIELD) or key not in program.containers:
            raise ValueError('%s keys %s, which is not a packed code field with a '
                             '@pa_container_size pin' % (table, key))
        runs = sorted(program.key_slices.get((table, key), ()), key=lambda run: run[1])
        if not runs:
            raise ValueError('%s keys all of %s; a packed tree keys slices of it'
                             % (table, key))
        low = runs[0][1]
        for (hi, _), (_, next_lo) in zip(runs, runs[1:]):
            if next_lo != hi + 1:
                raise ValueError('%s: the slices of %s are not one contiguous run'
                                 % (table, key))
        width = runs[-1][0] - low + 1
        if low % 8 or low // 8 + -(-width // 8) > sum(program.containers[key]) // 8:
            raise ValueError("%s: %s[%d:%d] does not sit on whole bytes of the "
                             "field's containers" % (table, key, runs[-1][0], low))
        feature = key[len(_PACKED_FIELD):]
        containers[feature] = program.containers[key]
        place[(task, tree, feature)] = (low // 8, width)
    return packed.key_bytes(packed.Layout(containers, place), task, tree)


def tree_key_slices(program, table):
    """One tree table's key of whole code fields (every program but a packed
    one) as packing.crossbar_stages_needed's key_slices takes it: (field name,
    slices) per keyed field with bits, in key order. A field's slices are its
    @pa_container_size pin filled low first (xbar.fill_low); a field with no
    pin (a program generated before the pins) gets lanes.key_layout's choice
    for the key, which is what the pin would say."""
    keys = [key for key in program.tables[table] if program.bits.get(key, 0) > 0]
    lay = lanes.key_layout([program.bits[key] for key in keys])
    return tuple((key, xbar.fill_low(program.bits[key], program.code_containers[key])
                  if key in program.code_containers else tuple(lay(program.bits[key])))
                 for key in keys)


def _field_price(program, name):
    """A tree table's price per 512-row block of entries: its key alone in a
    stage, as p4c's ternary crossbar allocator lays out its pinned containers
    (xbar.alone_blocks; lanes.table_blocks prices training's keys the same
    way), and CrossbarKeyTooWide where no allocation exists."""
    key = tree_key_slices(program, name)
    if not key:
        return lanes.table_blocks(())
    price = _alone_blocks(key)
    if price is None:
        raise CrossbarKeyTooWide(
            "%s's key does not fit one stage's ternary crossbar" % name,
            sum(program.widths[k] for k in program.tables[name]))
    return price


def _tree_priority(program, name, task):
    """A tree table's @placement_priority: the pragma the program carries, or,
    for a program generated before the pragma existed, the two-task constant
    the generator has emitted since C5 (program.PLACEMENT_PRIORITY)."""
    if name in program.priorities:
        return program.priorities[name]
    if task not in PLACEMENT_PRIORITY:
        raise ValueError('%s carries no @placement_priority and %r is not one of '
                         'the two-task names (%s) a default exists for'
                         % (name, task, ', '.join(PLACEMENT_PRIORITY)))
    return PLACEMENT_PRIORITY[task]


def _label_tasks(range_task, ternary_task):
    """Every task name one program's labels use: the trees' tasks plus any a
    range-table label names."""
    names = set(ternary_task)
    for label in range_task:
        if isinstance(label, frozenset):
            names |= label
        elif label != SHARED_TASK:
            names.add(label)
    return tuple(sorted(names))


def _key_bits(bits, keys):
    """One tree's key bits, sorted: the tuple training hands the packer
    (tables.ternary_key_field_bits sorts the same way)."""
    return tuple(sorted(bits[key] for key in keys))


def _packed_price(program, name):
    """A packed tree table's price per 512-row block of entries: its key's
    stand-alone price (packing.packed_alone_blocks, p4c's crossbar allocation),
    as evaluation._pool_inputs charges it, and CrossbarKeyTooWide where training
    raises it (no allocation)."""
    key = tree_key_bytes(program, name)
    price = packed_alone_blocks(key)
    if price is None:
        raise CrossbarKeyTooWide(
            "%s's key of %d bytes does not fit a stage's ternary crossbar"
            % (name, len(key)), len(key))
    return price


def declared_prices(program):
    """The model's own per-table price for every priced table of a program
    (replay_program's pricing loop). A tree of a packed program (is_packed)
    costs packing.packed_alone_blocks of its tree_key_bytes, any other tree
    its key's price alone by its pinned containers (_field_price); either
    times tree_entries_to_blocks of its declared size."""
    tables, bits, sizes = program.tables, program.bits, program.sizes
    packed_program = is_packed(program)
    blocks = {}
    for name, keys in tables.items():
        if not keys or name not in sizes:
            continue
        if name.startswith('get_classification_tree'):
            per_row = (_packed_price(program, name) if packed_program
                       else _field_price(program, name))
            blocks[name] = per_row * tree_entries_to_blocks(sizes[name])
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
    # Who reads what (2026-11-08): per tree, the range tables its key reads
    # (rule E, usage.tree_readiness_levels); per range table and per feature,
    # how many trees read it -- p4c's tie order for registers
    # (register_stage_schedule) and for the range pool (table_dependents).
    reads = tree_range_reads(tables)
    table_readers = {}
    for tables_read in (reads or {}).values():
        for table in tables_read:
            table_readers[table] = table_readers.get(table, 0) + 1
    feature_readers = {}
    for name, count in table_readers.items():
        feature = tables[name][0][:-len('_val')]
        feature_readers[feature] = feature_readers.get(feature, 0) + count
    dependents = feature_readers if reads is not None else None
    if readiness_levels is None:
        levels = dict(zip(row_features, readiness_levels_for(
            row_features, dependents=dependents)))
    else:
        missing = [f for f in row_features if f not in readiness_levels]
        if missing:
            raise ValueError(
                '%s: readiness_levels names no level for %s; every feature with '
                'a range table must have one or the packer would place it at an '
                'arbitrary stage' % (row_id, ', '.join(sorted(missing))))
        levels = readiness_levels
    tasks = table_tasks(tables)
    bytes_written = range_action_bytes(program)
    packed_program = is_packed(program)
    range_specs, range_fields, range_levels, range_task = [], [], [], []
    range_names, ternary_names = [], []
    ternary_specs, ternary_fields, ternary_key_bits, ternary_task = [], [], [], []
    ternary_key_slices = []
    ternary_key_bytes = []
    for name, keys in tables.items():
        if name not in blocks or not keys:
            continue
        if packed_program and name.startswith('get_classification_tree'):
            # A packed tree (spec 2026-11-11 Sec 5 rules 1-2), as
            # evaluation._pool_inputs hands it to the packer: its key's bytes,
            # width len(key), and a private field (packed trees never share
            # crossbar bytes: each keys its own slices).
            key = tree_key_bytes(program, name)
            ternary_specs.append((blocks[name], len(key)))
            ternary_key_bytes.append(key)
            ternary_task.append(tasks[name])
            ternary_names.append(name)
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
            # (_field_price, xbar.alone_blocks, x rows; replay_design) or the
            # count p4c committed (replay_stage_depth). It sizes the table's
            # rows; what the table is charged is decided by the packer's
            # ordered stage simulation, which charges every table its
            # crossbar allocation on what the earlier tables in its stage left
            # (packing.crossbar_stages_needed).
            # Program order is kept: p4c breaks placement-priority ties by
            # taking the table listed LAST first.
            key_bits = _key_bits(bits, keys)
            ternary_specs.append((blocks[name], width))
            ternary_fields.append(fields)
            ternary_key_bits.append(key_bits)
            ternary_key_slices.append(tree_key_slices(program, name))
            ternary_task.append(tasks[name])
            ternary_names.append(name)

    range_plan = crossbar_stages_needed(
        range_specs, readiness_levels=range_levels, key_fields=range_fields,
        table_dependents=([table_readers.get(name, 0) for name in range_names]
                          if reads is not None else None),
        action_bytes=[bytes_written[name] for name in range_names])
    position = {name: index for index, name in enumerate(range_names)}
    tree_reads = None
    if reads is not None:
        tree_reads = [[position[table] for table in reads[name] if table in position]
                      for name in ternary_names]
    # Per-task tree readiness and the seeded ternary pool, exactly as
    # usage.assemble_usage does it (audit C1): a tree waits for its own task's
    # range tables and the shared ones, and may land in a stage still holding
    # the other task's range tables, which count against that stage's limits.
    # key_field_bits and placement_priority (the generator's
    # @placement_priority per task) switch on the ordered stage simulation,
    # which decides both WHERE trees land and what every tree pays on the
    # crossbar the earlier tables in its stage left (a later tree of the same
    # key included) -- so they matter even when the block counts are p4c's
    # own.
    ternary_levels = tree_readiness_levels(
        range_plan.table_stages, range_task, ternary_task,
        tasks=_label_tasks(range_task, ternary_task),
        tree_reads=tree_reads)
    priorities = [_tree_priority(program, name, task)
                  for name, task in zip(ternary_names, ternary_task)]
    if not packed_program:
        ternary_plan = crossbar_stages_needed(
            ternary_specs, readiness_levels=ternary_levels,
            key_fields=ternary_fields,
            key_field_bits=ternary_key_bits, key_slices=ternary_key_slices,
            seed_stages=range_plan.stage_loads,
            placement_priority=priorities)
        predicted = (max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES)
        return ReplayPlans(predicted, range_plan, ternary_plan, range_names,
                           ternary_names)

    # The packed pool, placed by key_bytes exactly as usage.assemble_usage
    # places it.
    ternary_plan = crossbar_stages_needed(
        ternary_specs, readiness_levels=ternary_levels,
        key_bytes=ternary_key_bytes, seed_stages=range_plan.stage_loads,
        placement_priority=priorities)
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
