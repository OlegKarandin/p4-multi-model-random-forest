"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py, errors.py, xbar.py (p4c's ternary crossbar allocation,
ported: every table's stand-alone and in-stage price) and lanes.py (key
layouts) only. Deliberately knows nothing about features,
registers or trees -- it is handed (blocks, byte_width) specs plus readiness
levels (and, for the classification pool, key field widths and placement
priorities) and returns a placement, which is what lets the same packer serve
both the range pool and the classification pool.

A table's block count is not a property of the table alone: a key placed in
a stage AFTER a different key gets only the crossbar slots the earlier keys
left, and can cost more blocks than it would alone. So for the classification
pool the placement and the charge are computed together, by an ORDERED stage
simulation (crossbar_stages_needed's key_field_bits and placement_priority;
audit C5): tables are placed in the order p4c's placer takes them, which the
generator pins with @placement_priority, and every stage's crossbar is
simulated as p4c allocates it (xbar.py, spec 2026-11-13; for every encoding
since 2026-11-13, the lane pricing of audit C5 before): a tree is charged the
blocks of its actual allocation, and a tree whose key no longer fits waits
for a later stage, as in p4c. Keys are whole code fields (key_slices) or, for
packed per-tree codes, byte lists (key_bytes, spec 2026-11-11 Sec 4-5)."""
import functools
import math
from dataclasses import dataclass

from p4model import lanes, xbar
from p4model.errors import CrossbarKeyTooWide, PlacementFailed
from p4model.tables import word_blocks
from p4model.target import (
    ACTION_DATA_BUS_WORDS,
    ACTION_RAM_WORD_BYTES,
    RANGE_KEY_SLOTS_PER_BYTE,
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_GROUPS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
    TERNARY_CROSSBAR_STAGE_BYTES,
)


@dataclass(frozen=True)
class StageLoad:
  """What ONE pool puts into ONE stage, in the three units the per-stage
  limits are written in. It is both an output (StagePlan.stage_loads, one per
  occupied stage) and an input (crossbar_stages_needed's seed_stages): a pool
  packed after another is handed the first pool's loads as stages that are
  already partly full.

  index  : the stage index.
  blocks : the TCAM block count of every table SHARD in the stage, as charged
           (its crossbar-allocation charge in the stage), one entry per column-sized
           shard. Kept per shard rather than summed because the 12x2 column
           packing (fits_two_columns) needs the widths, not the total.
  fields : the distinct crossbar key fields present, (field_id, field_bytes)
           pairs -- the per-stage byte limit charges their union, not their sum.
  tables : tables placed in the stage -- what the 8-table cap counts (the
           range pool counts shards, which is the same number: no range table
           is ever wider than one column).
  field_order : `fields` in the order the pool PLACED them into the stage (a
           field first appears with the first table keying it). The range pool
           fills it; the classification pool seeded with those loads allocates
           the range keys on its crossbar in this order (decision 7.4, S22),
           as p4c allocated them. A load with fields but no order cannot seed
           a classification pool (_crossbar_stage_test raises ValueError):
           the order decides the allocation, so it is never guessed."""
  index: int
  blocks: tuple
  fields: frozenset
  tables: int
  field_order: tuple = ()


@dataclass(frozen=True)
class StagePlan:
  """crossbar_stages_needed's placement, not just its size -- F10: the stage
  a pool is DONE at (depth) is not the same quantity as how many stages it
  OCCUPIES (occupied): a stage can fill at the 8-table crossbar cap and spill
  a table forward past every level actually requested, so depth must be read
  from where tables landed, not from max(readiness_levels) + 1.

  Every field describes THIS pool only. Seeds passed in through
  crossbar_stages_needed's seed_stages are never counted in occupied, depth,
  indices, blocks or stage_loads -- they belong to the pool that produced them."""
  occupied: int          # how many stage indices hold a table from this pool
  depth: int             # max(occupied index) + 1 -- the quantity a 12-stage ceiling reads
  indices: frozenset     # for assertions and debugging
  blocks: int            # total TCAM blocks actually CHARGED across every stage --
                         # not the naive per-table sum passed in as table_specs: in the
                         # classification pool a table of a key placed AFTER a different
                         # key in its stage is charged the blocks of its allocation on
                         # what the earlier keys left (xbar.py), which can exceed its
                         # stand-alone price. A per-STAGE placement fact, not a
                         # per-table one.
  table_stages: tuple = ()  # one stage index per table_specs entry, positionally
                         # aligned with it: the LATEST stage any shard of that table
                         # landed in, i.e. the stage after which its result exists. A
                         # table wider than one column is sharded (_stage_shards) and its
                         # shards can land in different stages; this is their max. What
                         # usage.assemble_usage reads to know when one task's range tables
                         # are done (per-task tree readiness, audit C1).
  stage_loads: tuple = ()  # one StageLoad per occupied stage, sorted by index: what
                         # this pool put where. Pass it as the NEXT pool's seed_stages
                         # so that pool sees these stages as partly full.
  table_blocks: tuple = ()  # one CHARGED block count per table_specs entry, positionally
                         # aligned like table_stages: the declared count in the
                         # declared-price paths, the sum of its chunks' charges (each
                         # its crossbar-allocation charge in its stage) in the ordered
                         # simulation. sum(table_blocks) == blocks. What model.json's
                         # per-table list reports (spec 2026-09-29 §5.3).


def stage_load_fits(loads):
  """Whether several pools' StageLoads for the SAME stage index fit together
  under the per-stage limits the classification placement enforces: <=
  TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE tables and a 12x2 column packing of
  every shard (fits_two_columns), restated as a predicate so a caller can
  assert it after the fact. The crossbar itself is not checked here: the
  placement's crossbar simulation (_crossbar_stage, xbar.py) already decided
  that every key in the stage has an allocation, and its charges are inside
  each load's blocks. (The union of the loads' field bytes against
  TERNARY_CROSSBAR_STAGE_BYTES was dropped from this test on 2026-11-13: the
  placement no longer enforces it, and nibble-sized bytes can share a
  midbyte, so it is not guaranteed to hold.)"""
  loads = list(loads)
  return (sum(load.tables for load in loads) <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
          and fits_two_columns([b for load in loads for b in load.blocks]))


def fits_two_columns(block_widths, rows=TCAM_ROWS_PER_STAGE,
                     columns=TCAM_COLUMNS_PER_STAGE):
  """Can these tables share one stage's TCAM?

  A stage's TCAM_BLOCKS_PER_STAGE blocks are not one pool: mau_spec.h:88-90
  gives 12 rows x 2 columns, and a table needing several blocks chains them
  down ONE column. So what decides a stage is the WIDTHS, not the total --
  three 8-block tables total exactly 24 and need two stages (8+8 overflows a
  12-row column) while four 6-block tables, also 24, fit in one (6+6 | 6+6).
  Measured against real p4c over synthetic tables of 5..12 blocks all keyed on
  one shared field, so TCAM blocks rather than the crossbar bound the result:
  scripts/tcam_column_sweep.py, reviews/p4_tofino_reference.md Appendix B
  "Mechanism C". Consistent with every one of the 18 calibration rows'
  committed placements, in both pools, with no exceptions.

  Exact subset-sum over achievable column loads, not a greedy fit: with two
  columns and a handful of tables the state space is trivial, and a greedy
  would report false violations. That matters historically -- the column rule
  was once written off as "refuted" on the strength of a per-width shortcut
  (`2*floor(12/w)`) applied to stages holding mixed widths, where the real
  packing is far more permissive: (7+5 | 6+6) fits four tables the shortcut
  rejects.

  Callers must pass shards no wider than a column (`_stage_shards` guarantees
  it). A single table needing MORE than a column really does span both --
  measured at 14, 16 and 24 blocks, each compiling into one stage -- and
  _stage_shards models that by splitting it into column-sized pieces."""
  widths = [int(w) for w in block_widths]
  if any(w > rows for w in widths):
    return False
  loads = {(0,) * columns}
  for width in sorted(widths, reverse=True):
    nxt = set()
    for load in loads:
      for i in range(columns):
        if load[i] + width <= rows:
          bumped = list(load)
          bumped[i] += width
          nxt.add(tuple(sorted(bumped)))
    if not nxt:
      return False
    loads = nxt
  return True


def _stage_shards(block_count, byte_width):
  """Splits one logical table into pieces no wider than a single TCAM column,
  so the packer never reports a stage count below what the table alone already
  forces.

  Splitting a table's ROWS is real: a table needing more blocks than one
  column holds genuinely spreads them further, each shard still carrying the
  table's full key width. Splitting a table's KEY across stages is not real: a
  key is one indivisible match, TCAM compares a whole row in one clock, and a
  stage's crossbar physically cannot deliver more than its
  target.TERNARY_CROSSBAR_STAGE_BYTES (66) byte slots -- a table whose key
  exceeds that is rejected by the compiler outright, not spread across
  stages. Raise rather than silently pricing that impossible design as
  `ceil(byte_width / budget)` stages (and rather than letting the range
  pool's eager placement look for a stage forever). The key-size rule itself
  (p4c's 524 live bits and the crossbar port's fit, decision 6a) is applied
  where keys are priced; this is the crossbar's own capacity, a guard.

  The split is at TCAM_ROWS_PER_STAGE (one column), not TCAM_BLOCKS_PER_STAGE
  (the whole stage), because fits_two_columns places each shard inside one
  column and could never place a wider one -- under eager placement an
  unplaceable shard would advance its stage index without end. Column-sized
  shards also reproduce the measurement: a 24-block table becomes 12 | 12,
  which fills both columns of ONE stage, exactly as real p4c does with single
  tables of 14, 16 and 24 blocks.

  Prior to finding 1.5a, the split was into n equal pieces, each rounded up.
  This charged a 13-block table as 14 and a 23-block one as 24, an arithmetic
  artifact of the equal split. Filling columns and leaving a remainder ensures
  the shards sum to the table. No table in the 19-row calibration archive
  exceeds 12 blocks, so this path rests on one confirming compile rather than
  the calibration corpus: `scripts/tcam_stage_shard_probe.py` places a
  13-block table as 12|1 in one stage, charged 13, matching this function's
  prediction. It is the one non-monotone (cost-lowering) change in this work,
  licensed because it corrects rounding rather than relaxing a measured
  limit."""
  if byte_width > TERNARY_CROSSBAR_STAGE_BYTES:
    # Live bits are not known here; 8 per byte is an upper bound, so the
    # magnitude (bits over 524) is one too. Reached only by a caller that skipped
    # the pricing paths' own checks (they refuse such a key first).
    raise CrossbarKeyTooWide(
        "table key of %d crossbar bytes is wider than a stage's %d crossbar byte "
        "slots" % (byte_width, TERNARY_CROSSBAR_STAGE_BYTES), 8 * byte_width)
  shards, remaining = [], block_count
  while remaining > TCAM_ROWS_PER_STAGE:
    shards.append((TCAM_ROWS_PER_STAGE, byte_width))
    remaining -= TCAM_ROWS_PER_STAGE
  shards.append((remaining, byte_width))
  return shards


# --------------------------------------------------------------------------
# The classification pool's ordered stage simulation (audit C5)
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class _Unit:
  """One placeable piece of a classification table: the whole table, or --
  for a table charged more than a stage's TCAM_BLOCKS_PER_STAGE -- one chunk
  of its rows."""
  table_idx: int
  chunk: int
  rows: int              # 512-row words -- what the key's per-row charge multiplies
  fields: frozenset
  bits: tuple            # the key: (field id, slices) per field (_field_units), or
                         # a packed key's lanes.py byte tuples (_packed_units)
  level: int
  priority: int


def _field_units(table_specs, key_fields, key_slices, levels, priorities):
  """table_specs -> _Units for keys of whole code fields (every encoding but
  per-tree). key_slices[idx] is table idx's key as xbar.field_key takes it,
  (field id, slices) per field. A table's rows are its blocks / the key's
  price alone in a stage (xbar.alone_blocks, which lanes.table_blocks
  returns), rounded up; a table charged more than one stage's
  TCAM_BLOCKS_PER_STAGE is split into chunks of whole rows that each fit a
  stage, every chunk keeping the full key."""
  units = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    _stage_shards(block_count, byte_width)   # raises for an impossible key
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    key = tuple(key_slices[idx])
    per_row = _alone_blocks(key)
    if per_row is None:
      raise CrossbarKeyTooWide(
          "table %d's key of %d bytes cannot be placed in one stage's ternary crossbar"
          % (idx, byte_width), sum(int(used) for _, slices in key for _, used in slices))
    units.extend(_row_chunks(idx, _rows_of(block_count, max(1, per_row)),
                             max(1, per_row), fields, key,
                             levels[idx], priorities[idx]))
  return units


@functools.lru_cache(maxsize=16384)
def _alone_blocks(key):
  """A key of whole fields alone in a stage: its blocks per 512-row word, or
  None when it cannot be placed in one stage (tables.word_blocks)."""
  return word_blocks(xbar.alone_blocks(key)) if key else 1


def _rows_of(block_count, per_row):
  """A table's 512-row words: its declared blocks over its key's price per word."""
  return max(1, math.ceil(block_count / per_row))


def _row_chunks(idx, rows, per_row, fields, bits, level, priority):
  """One table's _Units: the whole table (`rows` 512-row words, each costing
  `per_row` blocks), or -- charged more than one stage's TCAM_BLOCKS_PER_STAGE --
  chunks of whole words that each fit a stage; every chunk keeps the full key.

  Two splits exist and neither is exercised by the campaign: a table of more
  than 24 blocks (more than one stage's TCAM_BLOCKS_PER_STAGE) is cut into
  stage-sized chunks of floor(24 / per_row) words -- unreachable at this
  project's scale, no design has a tree that wide; and a chunk of more than
  12 blocks (a column, TCAM_ROWS_PER_STAGE) is cut again into 12-block column
  shards by _stage_shards, which rests on ONE compile
  (scripts/tcam_stage_shard_probe.py: a 13-block table places 12 | 1 in one
  stage). The paths are kept so an unusually wide table is still priced, not
  refused."""
  units = []
  remaining_rows = rows
  rows_per_chunk = max(1, TCAM_BLOCKS_PER_STAGE // per_row)
  chunk = 0
  while True:
    chunk_rows = min(remaining_rows, rows_per_chunk)
    units.append(_Unit(idx, chunk, chunk_rows, fields, bits, level, priority))
    remaining_rows -= chunk_rows
    chunk += 1
    if remaining_rows <= 0:
      break
  return units


def _ordered_stage_simulation(units, mode, seed_at, last_seed):
  """The classification pool's placement: p4c's table placer, stage by
  stage. At stage s, repeatedly take the FIRST remaining table, in the order
  (placement priority descending, then the table listed LAST first), whose
  level is <= s and that fits the stage as it now stands
  (_crossbar_stage_test in `mode`: 'fields' for keys of whole code fields,
  _field_units; 'packed' for packed per-tree keys, _packed_units); when none
  does, move to s + 1.
  Returns ({stage index: [(unit, charge), ...]}, {table_idx: latest stage})."""
  return _place_in_order(
      units,
      lambda here, unit, seed: _crossbar_stage_test(here + [unit], seed, mode),
      seed_at, last_seed)


def _place_in_order(units, admit, seed_at, last_seed):
  """The ordered placement loop of _ordered_stage_simulation.
  `admit(here, unit, seed)` returns the charges of `here + [unit]` in
  placement order, or None when the stage holding `seed` refuses the unit."""
  remaining = sorted(units, key=lambda u: (-u.priority, -u.table_idx, u.chunk))
  if not remaining:
    return {}, {}
  placed, charges = {}, {}
  stage = min(unit.level for unit in remaining)
  # Past every level and seed, an empty stage takes any
  # remaining unit alone (one key, charged its own blocks, <= one stage wide),
  # so the loop always ends; the bound only turns a bug into an error.
  horizon = (max(unit.level for unit in remaining) + len(units)
             + max(last_seed, 0) + 2)
  while remaining:
    if stage > horizon:
      raise PlacementFailed(
          "crossbar_stages_needed: %d classification table(s) could not be "
          "placed by stage %d" % (len(remaining), horizon))
    seed = seed_at(stage)
    here = placed.get(stage, [])
    for unit in remaining:
      if unit.level > stage:
        continue
      charged = admit(here, unit, seed)
      if charged is not None:
        placed[stage] = here + [unit]
        charges[stage] = charged
        remaining.remove(unit)
        break
    else:
      stage += 1
  table_stages = {}
  for index, stage_units in placed.items():
    for unit in stage_units:
      table_stages[unit.table_idx] = max(index,
                                         table_stages.get(unit.table_idx, index))
  return ({index: list(zip(placed[index], charges[index])) for index in placed},
          table_stages)


# --------------------------------------------------------------------------
# The packed classification pool: keys with a known layout (spec 2026-11-11)
# --------------------------------------------------------------------------

def packed_alone_blocks(key):
  """A packed tree key's price per 512-row word ALONE in a stage: the blocks
  of p4c's crossbar allocation of its bytes (xbar.tree_key) in an empty
  stage, or None when it cannot be placed in one stage (tables.word_blocks: no
  allocation in 12 groups, or a word over 12 blocks). Memoised on the key's
  shape: a byte's container identity (packed.key_bytes' 4th element) only
  matters between tables, so it is renamed to its first appearance in the key
  first. The lane count lanes.standalone priced it until 2026-11-13 (equal on
  1,581 of 1,591 development keys)."""
  names = {}
  return _packed_alone_blocks(tuple(
      item[:3] + (names.setdefault(item[3], len(names)),) if len(item) > 3 else item
      for item in key))


@functools.lru_cache(maxsize=16384)
def _packed_alone_blocks(key):
  key_bytes = xbar.tree_key(key, 0)
  if not xbar.StageXbar(tie='whole_first').allocate(key_bytes):
    return None
  return word_blocks(xbar.blocks_per_row(key_bytes))


def _packed_units(table_specs, key_fields, key_bytes, levels, priorities):
  """table_specs -> _Units for keys whose bytes are known (packed.key_bytes).
  A table's rows are its blocks / its key's price alone (packed_alone_blocks),
  rounded up; chunked as _field_units does. A key that cannot be placed in
  one stage (packed_alone_blocks None) is refused here, as CrossbarKeyTooWide
  (live bits unknown here: 0, magnitude 1)."""
  units = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    _stage_shards(block_count, byte_width)   # raises for an impossible key
    key = tuple(key_bytes[idx])
    per_row = packed_alone_blocks(key)
    if per_row is None:
      raise CrossbarKeyTooWide(
          "table %d's key of %d bytes cannot be placed in one stage's %d crossbar "
          "groups" % (idx, len(key), TERNARY_CROSSBAR_GROUPS_PER_STAGE), 0)
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    units.extend(_row_chunks(idx, _rows_of(block_count, per_row), per_row, fields,
                             key, levels[idx], priorities[idx]))
  return units


@functools.lru_cache(maxsize=65536)
def _crossbar_stage(seed_fields, units, mode='packed'):
  """p4c's ternary crossbar allocation of one stage (xbar.py, spec 2026-11-13):
  the seed's range keys first (seed_fields, in the range pool's placement
  order), then each tree TABLE's key
  in placement order, each on top of the ones before. `units` is a tuple of
  (table_idx, key byte tuple, rows) in placement order; units of one table
  share one allocation. mode 'packed': a key is a packed tree's byte tuples
  (packed.key_bytes -> xbar.tree_key), container identities shared across the
  stage's keys, so two keys naming one container byte would share its slot as
  in p4c (packed.layout_for gives no byte to two trees); mode 'fields': a key is
  (field id, slices) per code field (xbar.field_key), container numbers shared
  across the stage's keys, so tables keying the same field share its slots
  as in p4c. Returns {table_idx: blocks per 512-row word}, or None when a
  range key or a tree key does not fit. (A tree alone in an empty stage
  always fits: its rows were sized by the same allocation, packed_alone_blocks
  / _alone_blocks, and a key with none is refused before placement.)"""
  state = _stage_state(seed_fields, units, mode)
  return None if state is None else state[2]


@functools.lru_cache(maxsize=16384)
def _stage_state(seed_fields, units, mode):
  """_crossbar_stage's work, memoised per PREFIX of `units`: the placement
  loop asks for a stage's units, then the same units plus one, so each call
  extends the cached state of its prefix by one table instead of allocating
  the whole stage again. Returns (crossbar, numbering, per_row) -- never
  mutated once cached -- or None as _crossbar_stage does."""
  if not units:
    crossbar = xbar.StageXbar(tie='whole_first')
    for index, (_, field_slots) in enumerate(seed_fields):
      # A range field's size is counted in crossbar byte SLOTS (RANGE_LO + RANGE_HI
      # per key byte, tables.range_key_fields_for), xbar.range_key takes key bytes.
      if not crossbar.allocate(
          xbar.range_key(index, field_slots // RANGE_KEY_SLOTS_PER_BYTE)):
        return None
    return crossbar, {}, {}
  previous = _stage_state(seed_fields, units[:-1], mode)
  if previous is None:
    return None
  crossbar, numbering, per_row = previous
  table_idx, key, _ = units[-1]
  if table_idx in per_row:
    return previous
  crossbar, numbering = crossbar.clone(), dict(numbering)
  key_bytes = (xbar.tree_key(key, table_idx, numbering) if mode == 'packed'
               else xbar.field_key(key, numbering))
  placed = crossbar.find_alloc(key_bytes)
  if placed is None:
    return None
  # A word wider than a column cannot be placed in this stage either
  # (tables.word_blocks): the table waits, as in p4c.
  blocks = word_blocks(xbar.blocks_per_row(key_bytes))
  if blocks is None:
    return None
  crossbar.commit(placed)
  per_row = dict(per_row)
  per_row[table_idx] = blocks
  return crossbar, numbering, per_row


def _crossbar_stage_test(units, seed, mode='packed'):
  """The stage test of the classification pool: the charges of `units` (this pool's,
  in placement order) in one stage holding `seed`, or None when refused.

  The stage's crossbar is simulated as p4c allocates it (_crossbar_stage,
  xbar.py): its range tables, then its trees in placement order, each on the
  slots the earlier ones left. A tree whose key no longer fits is refused, so
  the placement tries the next table or stage, as p4c's placer does; a tree
  that fits is charged its rows x the blocks of its actual allocation. This
  replaces the stand-alone price, the 12-group rule and the @stage pins of
  spec 2026-11-11 (spec 2026-11-13: p4c's allocation reproduced slot for slot
  on 2,481 compiles). Also required:
    * at most TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE tables, seed included;
    * the blocks, seed shards included, pack 12x2 (fits_two_columns).
  A stage holding nothing else admits any single unit (its key was sized
  alone by the same allocation; a key with no allocation is refused before
  placement), so the placement always ends. The range tables' 8-word action
  data bus rule is the range pool's (crossbar_stages_needed), not this
  test's.
  Returns the charges, or None when the stage refuses the units."""
  if len(units) + seed.tables > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE:
    return None
  alone = (len(units) == 1 and seed.tables == 0 and not seed.blocks
           and not seed.fields)
  if seed.fields and not seed.field_order:
    raise ValueError(
        "a seed StageLoad with fields must carry field_order (the range pool's "
        "placement order, which decides the crossbar allocation)")
  per_row = _crossbar_stage(
      seed.field_order,
      tuple((unit.table_idx, unit.bits, unit.rows) for unit in units), mode)
  if per_row is None:
    return None
  charges = [unit.rows * per_row[unit.table_idx] for unit in units]
  if not alone and not fits_two_columns(
      list(seed.blocks) + [width for charge in charges
                           for width, _ in _stage_shards(charge, 1)]):
    return None
  return charges


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           key_field_bits=None,
                           seed_stages=(), placement_priority=None,
                           table_dependents=None, action_bytes=None,
                           key_bytes=None, key_slices=None):
  """Packs independent match tables into pipeline stages under every
  per-stage hardware limit simultaneously, and returns a StagePlan
  describing where the tables landed (not just how many stages that took).

  table_specs is one (block_count, byte_width) pair per independent P4
  table -- one per tree for the ternary classification tables, one per
  feature for the range-matching tables.

  key_fields (optional) is one frozenset of (field_id, field_bytes) per
  table, positionally aligned with table_specs, naming WHICH crossbar
  fields that table's key is made of. It exists because the Ternary Match
  Input crossbar charges per distinct FIELD placed on its byte slots, not
  per (table, field) pair: two tables in the same stage that match on the
  same field read the same slots and the field is charged ONCE. Every
  classification table of one task keys on the identical meta.code_<feature>
  field set, and under 'disjoint' both models' range tables for a shared
  feature key the identical meta.<feature>_val field. Measured against 19
  real p4c compiles (reviews/p4_tofino_reference.md §4.3): joint_low_sd7's
  stage 7 holds four tables on one 32-byte codeword and the compiler reports
  32 crossbar bytes, not 128. Passing key_fields=None gives each table a
  private synthetic field, so the union reduces to the per-table sum. In the
  classification pool the field SET is also the key's identity: tables with
  equal key_fields are one key.

  seed_stages (optional) is an iterable of StageLoad -- another pool's
  StagePlan.stage_loads -- naming stages that are already partly full. It
  exists because under 'disjoint' a task's trees wait only for THEIR OWN
  task's range tables (audit C1), so a classification tree can land in a
  stage still holding the other task's range tables. A seed's shards, fields
  and table count count against the stage's limits exactly as this pool's
  own would, and in the classification pool its range fields are also keys
  on the same crossbar, placed BEFORE any tree key (see below). Seeds are
  never charged again: the returned plan's blocks, occupied, depth, indices
  and stage_loads describe this pool alone. Seeds need readiness_levels
  (absolute stage indices); passing them without raises.

  Every stage must satisfy at once:
    * TCAM blocks, as a 12x2 COLUMN packing (fits_two_columns), not a flat
      <= TCAM_BLOCKS_PER_STAGE total
    * <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE independent tables
    * <= TERNARY_CROSSBAR_STAGE_BYTES bytes of DISTINCT key fields (the range
      pool; the classification pool's crossbar is decided by the ternary
      crossbar port's allocation instead, below)
  (RM-5/RM-6/RM-7, reviews/p4_tofino_reference.md §4.3.) These constraints
  are not separable: tables (20 blocks, 5 B), (20, 5), (1, 60) need 3 stages
  although each relaxation alone says 2.

  THE RANGE POOL (key_field_bits=None) is packed as before: every table is
  charged its declared block count wherever it lands; with readiness_levels,
  shards are placed eagerly in level order at the
  earliest legal stage with room (the compiler does not pack to the
  theoretical optimum, and neither does this -- M2's range pool really
  occupies 2 stages; a `load` score that once ordered a level's shards
  largest first was vacuous on every range table and is gone, decision 7.4 /
  S12); without them, first-fit-decreasing by blocks. A range table's
  key is one whole-byte 16-bit field, and 8 of them (the table cap) fill 16
  bytes, so no sharing price applies.

  table_dependents (2026-11-08, the range pool): one int per table, the
  classification trees reading its code. Given, tables of one level
  are placed in p4c's order -- more dependents first, then the table listed
  LATER -- instead of first-listed. p4c's own log over the range tables it
  compares (per_tree_model2/step12/range_reasons_out.txt): "fewer total
  dependencies in dominance frontier" picks the one more trees read in 714 of
  714 (campaign independent sample), 628 of 628 (joint-task), 2,361 of 2,430
  (per-tree); "default choice" picks the later-listed in every case. With the
  8-table stage cap binding, which table is left over decides which trees
  wait (campaign independent_M075_s20_k10). Only for the declared-price
  placement below; a classification pool (key_field_bits) refuses it.

  action_bytes (2026-11-09, the range pool): one int per table, the bytes
  its action puts on the action data bus: the bytes of the containers of
  every code field it writes (decision 4.3, design review S14;
  tables.range_action_bytes_for, packed.bus_bytes, p4_replay.range_action_bytes).
  Given, the tables sharing a stage take at most target.ACTION_DATA_BUS_WORDS
  (8) action RAM words between them, each table rounded up to whole 16-byte
  words: sum(ceil(bytes / 16)) <= 8 (spec 2026-11-11-packed-codes-stage-pin
  Sec 6, model4 probes bus_packed.py / bus_bytes.py); a table alone always
  takes an empty stage (a packed field over the bus is refused by
  packed.layout_for). p4c refuses a
  range table a stage whose bus it cannot fit ("could not fit in within the
  action data bus"). Only for the declared-price placement below.

  THE CLASSIFICATION POOL (key_field_bits given: one tuple of key-field BIT
  widths per table) is placed by an ORDERED STAGE SIMULATION (audit C5,
  reviews/model_audit_2026-09-27.md §7), which replaced the fitted 58-byte
  crowded-stage margin and its any-key-order search:

    * Order -- p4c's placer, pinned by the generator: stage by stage, the
      remaining table with the highest placement_priority (one int per
      table; the generator emits @placement_priority, ddos 2 above app 1 --
      program.PLACEMENT_PRIORITY), ties to the table listed LAST, that is
      ready (readiness level <= the stage) and fits; when none fits, the
      next stage. table_specs must therefore be in program order within a
      priority. Without readiness_levels every table is ready at stage 0;
      without placement_priority every table has priority 0.
    * Price and fit -- each stage's ternary crossbar is simulated as p4c
      allocates it (xbar.py, spec 2026-11-13; the lane pricing of audit C5
      until then): the seed's range keys first, then each table's key in
      placement order on the slots the earlier ones left, tables keying the
      same code field sharing its slots. A table is charged its rows x the
      blocks of its actual allocation; a table whose key no longer fits is
      refused there and waits. Also at most 8 tables, and the charged blocks
      pack 12x2.
    * key_slices (optional) -- one key per table as xbar.field_key takes it,
      (field id, slices) per code field, slices being (container kind, bits)
      low first: the containers the generator pins. Field ids name a field
      across tables (a disjoint field both tasks key is one field). Without
      it, each distinct key_fields set gets the layout lanes.key_fields gives
      its key_field_bits, sharing nothing with other keys.
  A table charged more than one stage's 24 blocks is split into chunks of
  whole rows that each fit a stage.

  EVERY classification pool is simulated, a single-key pool (every 'joint'
  design) included (2026-11-13). Until then a single-key pool was placed by
  the declared-price packer, every table charged its stand-alone price; but
  p4c charges a LATER table of the same key what its allocation on the
  earlier tables' slots costs, which can be a block less (joint-off_M035_s22
  _k06: 5, 5, 4, 5 in one stage). Simulated, the 2,400 joint-off and 2,400
  joint-off-al compiles of campaign_2026_11 price every tree table exactly
  (model7/findings.md Part 4).

  Measured under the ternary crossbar port (2026-11-13) on the step-2 replay
  of the 7,650 campaign_2026_11 compiles (model7/findings.md Part 4), and on
  programs of the current generator by tests/test_p4c_base_17b.py (12 Task 17b
  base designs, recompiled with the G3 pins; Task 23, results/encoding_evidence_2026_11/model8/impl/
  remeasure.md). The older calibration archive
  (compiler_calibration_pragmas_2026_09_29) predates decisions 1a / 7.1 option
  B and is refused by shape, so it no longer scores this placement.

  THE PACKED CLASSIFICATION POOL (key_bytes given: one lanes.py byte-list
  tuple per table, packed.key_bytes -- spec
  docs/superpowers/specs/2026-11-11-packed-codes-stage-pin-design.md Sec 4.2,
  4.3, Sec 5 rules 1-2) is placed in the same order as the ordered stage
  simulation, but priced and
  admitted differently (_crossbar_stage_test): each stage's ternary crossbar
  is simulated as p4c allocates it (xbar.py, spec 2026-11-13: p4c's slot
  assignment reproduced on every ternary table of 2,481 compiles), the
  seed's range keys first, then the trees in placement order. A tree is
  charged its rows x the blocks of its actual allocation; a tree that no
  longer fits is refused there and waits, as p4c's placer makes it wait.
  This replaced the stand-alone price, the 12-group rule and the @stage pins
  of spec 2026-11-11 (StagePlan's moved / unpinnable / first_refused, which
  only those pins needed, are gone). Needs readiness_levels; refuses key_field_bits,
  table_dependents and action_bytes.
  """
  if key_bytes is not None:
    if key_field_bits is not None:
      raise ValueError(
          "crossbar_stages_needed: key_bytes and key_field_bits both describe "
          "the classification keys; pass one (key_bytes for packed codes)")
    if readiness_levels is None:
      raise ValueError(
          "crossbar_stages_needed: key_bytes places tables at absolute stage "
          "indices, which needs readiness_levels")
    if len(key_bytes) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d key_bytes for %d table_specs; the "
          "two must be positionally aligned, one byte list per table"
          % (len(key_bytes), len(table_specs)))
    if table_dependents is not None or action_bytes is not None:
      raise ValueError(
          "crossbar_stages_needed: table_dependents and action_bytes belong to "
          "the range pool's declared-price placement, not the packed pool")
  if key_fields is not None:
    if len(key_fields) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d key_fields for %d table_specs; the "
          "two must be positionally aligned, one entry per table"
          % (len(key_fields), len(table_specs)))
    for idx, ((_, byte_width), fields) in enumerate(zip(table_specs, key_fields)):
      declared = sum(field_bytes for _, field_bytes in fields)
      if declared != byte_width:
        raise ValueError(
            "crossbar_stages_needed: table %d's key_fields sum to %d crossbar "
            "bytes but its spec declares %d -- both describe the same key, so "
            "a mismatch would mis-price every stage the table lands in"
            % (idx, declared, byte_width))

  if key_slices is not None:
    if key_field_bits is None:
      raise ValueError(
          "crossbar_stages_needed: key_slices describes the classification "
          "keys' containers; it goes with key_field_bits")
    if len(key_slices) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d key_slices for %d table_specs; the "
          "two must be positionally aligned, one key per table"
          % (len(key_slices), len(table_specs)))
  if key_field_bits is not None and len(key_field_bits) != len(table_specs):
    raise ValueError(
        "crossbar_stages_needed: got %d key_field_bits for %d table_specs; the "
        "two must be positionally aligned, one field-width tuple per table, or "
        "a table would be priced against another table's key shape"
        % (len(key_field_bits), len(table_specs)))
  if table_dependents is not None:
    if key_field_bits is not None:
      raise ValueError(
          "crossbar_stages_needed: table_dependents orders the declared-price "
          "range placement; the classification pool is ordered by "
          "placement_priority")
    if len(table_dependents) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d table_dependents for %d table_specs; "
          "the two must be positionally aligned"
          % (len(table_dependents), len(table_specs)))
  if action_bytes is not None:
    if key_field_bits is not None:
      raise ValueError(
          "crossbar_stages_needed: action_bytes budgets the range pool's "
          "declared-price placement, not the classification pool")
    if len(action_bytes) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d action_bytes for %d table_specs; "
          "the two must be positionally aligned"
          % (len(action_bytes), len(table_specs)))
  if placement_priority is not None:
    if key_field_bits is None and key_bytes is None:
      raise ValueError(
          "crossbar_stages_needed: placement_priority orders the "
          "classification pool's stage simulation, which needs key_field_bits "
          "or key_bytes")
    if len(placement_priority) != len(table_specs):
      raise ValueError(
          "crossbar_stages_needed: got %d placement_priority entries for %d "
          "table_specs; the two must be positionally aligned"
          % (len(placement_priority), len(table_specs)))

  seeds = {}
  for seed in seed_stages:
    if seed.index in seeds:
      raise ValueError(
          "crossbar_stages_needed: two seed_stages entries for stage %d; pass "
          "one StageLoad per stage (a StagePlan's stage_loads already is)"
          % seed.index)
    seeds[seed.index] = seed
  if seeds and readiness_levels is None:
    raise ValueError(
        "crossbar_stages_needed: seed_stages name absolute stage indices, "
        "which only the dependency-aware placement (readiness_levels) has; "
        "the pure packer would silently ignore them")

  def seed_at(index):
    return seeds.get(index) or StageLoad(index=index, blocks=(),
                                         fields=frozenset(), tables=0)

  def plan(loads, table_stages, table_blocks):
    loads = sorted(loads, key=lambda load: load.index)
    return StagePlan(occupied=len(loads),
                     depth=(loads[-1].index + 1) if loads else 0,
                     indices=frozenset(load.index for load in loads),
                     blocks=sum(sum(load.blocks) for load in loads),
                     table_stages=tuple(table_stages[idx]
                                        for idx in range(len(table_specs))),
                     stage_loads=tuple(loads),
                     table_blocks=tuple(table_blocks.get(idx, 0)
                                        for idx in range(len(table_specs))))

  def unit_plan(by_index, table_stages):
    # A classification placement ({stage: [(unit, charge), ...]}) as a StagePlan.
    charged = {}
    for placed in by_index.values():
      for unit, charge in placed:
        charged[unit.table_idx] = charged.get(unit.table_idx, 0) + charge
    loads = []
    for index, placed in by_index.items():
      loads.append(StageLoad(
          index=index,
          blocks=tuple(width for _, charge in placed
                       for width, _ in _stage_shards(charge, 1)),
          fields=frozenset().union(*(unit.fields for unit, _ in placed)),
          tables=len(placed)))
    return plan(loads, table_stages, charged)

  if key_bytes is not None:
    priorities = (list(placement_priority) if placement_priority is not None
                  else [0] * len(table_specs))
    by_index, table_stages = _ordered_stage_simulation(
        _packed_units(table_specs, key_fields, key_bytes,
                      list(readiness_levels), priorities),
        'packed', seed_at, max(seeds, default=0))
    return unit_plan(by_index, table_stages)

  if key_field_bits is not None:
    levels = (list(readiness_levels) if readiness_levels is not None
              else [0] * len(table_specs))
    priorities = (list(placement_priority) if placement_priority is not None
                  else [0] * len(table_specs))
    if key_slices is None:
      key_slices = []
      for idx, bits in enumerate(key_field_bits):
        identity = (key_fields[idx] if key_fields is not None
                    else ("<private>", idx))
        key_slices.append(tuple(((identity, field), slices) for field, slices
                                in lanes.key_fields(bits)))
    by_index, table_stages = _ordered_stage_simulation(
        _field_units(table_specs, key_fields,
                     [tuple(key) for key in key_slices], levels, priorities),
        'fields', seed_at, max(seeds, default=0))
    return unit_plan(by_index, table_stages)

  # ---- the range pool (and a caller passing neither key_field_bits nor
  # key_bytes): every table charged its declared blocks.
  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    for shard in _stage_shards(block_count, byte_width):
      shards.append((shard[0], shard[1], idx, fields))

  def bus_words(table_idx):
    # whole action RAM words: the bus is fed in 16-byte words
    return (-(-action_bytes[table_idx] // ACTION_RAM_WORD_BYTES)
            if action_bytes is not None else 0)

  def fits(stage, blocks, fields, table_idx):
    # stage: [fields present, shard blocks in placement order, seed,
    #         action-data bus words, fields in placement order]
    seed = stage[2]
    present = stage[0] | fields | seed.fields
    return (sum(field_bytes for _, field_bytes in present)
            <= TERNARY_CROSSBAR_STAGE_BYTES
            and len(stage[1]) + seed.tables + 1
            <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
            and fits_two_columns(list(seed.blocks) + stage[1] + [blocks])
            and (not stage[1]
                 or stage[3] + bus_words(table_idx) <= ACTION_DATA_BUS_WORDS))

  def place(stage, blocks, fields, table_idx):
    for field in sorted(fields, key=repr):
      if field not in stage[0]:
        stage[4].append(field)
    stage[0] |= fields
    stage[1].append(blocks)
    stage[3] += bus_words(table_idx)

  def empty(seed):
    return [set(), [], seed, 0, []]

  def as_loads(stages_by_index):
    return [StageLoad(index=index, blocks=tuple(stage[1]),
                      fields=frozenset(stage[0]), tables=len(stage[1]),
                      field_order=tuple(stage[4]))
            for index, stage in stages_by_index.items()]

  table_stages = {}
  table_blocks = {}
  if readiness_levels is None:
    stages = []
    for blocks, _width, table_idx, fields in sorted(
        shards, key=lambda s: (s[0], s[1]), reverse=True):
      for position, stage in enumerate(stages):
        if fits(stage, blocks, fields, table_idx):
          place(stage, blocks, fields, table_idx)
          break
      else:
        position = len(stages)
        stages.append(empty(seed_at(position)))
        place(stages[-1], blocks, fields, table_idx)
      table_stages[table_idx] = max(position,
                                    table_stages.get(table_idx, position))
      table_blocks[table_idx] = table_blocks.get(table_idx, 0) + blocks
    return plan(as_loads(dict(enumerate(stages))), table_stages, table_blocks)

  # Dependency-aware placement. Two differences from the packer above, both
  # chosen to track the REAL compiler rather than the theoretical optimum:
  #   1. A table may not occupy a stage index below its readiness level --
  #      its key value literally does not exist yet.
  #   2. Placement is EAGER (earliest legal stage with room), not "pack as
  #      few stages as possible". Measured: M2's range pool really occupies 2
  #      stages, which only eager placement reproduces.
  # A seeded stage this pool has not opened yet is tested through `fits`
  # with its seed; an unseeded empty stage always takes a lone shard.
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}
  if table_dependents is None:
    ordered = sorted(shards, key=lambda s: readiness_levels[s[2]])
  else:
    ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]],
                                            -table_dependents[s[2]], -s[2]))
  for blocks, _width, table_idx, fields in ordered:
    index = readiness_levels[table_idx]
    while not fits(by_index.get(index) or empty(seed_at(index)),
                   blocks, fields, table_idx):
      index += 1
    if index not in by_index:
      by_index[index] = empty(seed_at(index))
    place(by_index[index], blocks, fields, table_idx)
    table_stages[table_idx] = max(index, table_stages.get(table_idx, index))
    table_blocks[table_idx] = table_blocks.get(table_idx, 0) + blocks
  return plan(as_loads(by_index), table_stages, table_blocks)
