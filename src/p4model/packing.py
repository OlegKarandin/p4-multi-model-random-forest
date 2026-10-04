"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py, errors.py, tables.py's per-key price and lanes.py's
crossbar lane model only. Deliberately knows nothing about features,
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
generator pins with @placement_priority, and every stage is priced key by key
in that order with lanes.py."""
import functools
import math
from dataclasses import dataclass

from src.p4model import lanes
from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.target import (
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_GROUPS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
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
           (a later key's lane price included), one entry per column-sized
           shard. Kept per shard rather than summed because the 12x2 column
           packing (fits_two_columns) needs the widths, not the total.
  fields : the distinct crossbar key fields present, (field_id, field_bytes)
           pairs -- the 64-byte limit charges their union, not their sum.
  tables : tables placed in the stage -- what the 8-table cap counts (the
           range pool counts shards, which is the same number: no range table
           is ever wider than one column)."""
  index: int
  blocks: tuple
  fields: frozenset
  tables: int


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
                         # key in its stage is charged that key's lane LEFTOVER price
                         # (lanes.price_with_supply), which can exceed its standalone
                         # price. A per-STAGE placement fact, not a per-table one.
                         # Measured: dsp41's (84, 84) key costs 5 blocks alone and 7
                         # behind a 41-byte key in the same stage (lanes.py docstring).
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
                         # declared-price paths, the sum of its chunks' charges (a
                         # later key's lane price included) in the ordered
                         # simulation. sum(table_blocks) == blocks. What model.json's
                         # per-table list reports (spec 2026-09-29 §5.3).

  def __int__(self):     # transitional: `stages` is still the occupancy count
    return self.occupied


def stage_load_fits(loads):
  """Whether several pools' StageLoads for the SAME stage index fit together
  under the per-stage limits: <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
  tables, <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes of distinct key
  fields, and a 12x2 column packing of every shard (fits_two_columns). The check crossbar_stages_needed's seed_stages
  makes while placing (there on per-key byte SUMS, which are never below
  this union), restated as a predicate so a caller can assert it after the
  fact. The lane prices themselves are already inside each load's charged
  blocks."""
  loads = list(loads)
  fields = frozenset().union(*(load.fields for load in loads))
  total_bytes = sum(field_bytes for _, field_bytes in fields)
  return (sum(load.tables for load in loads) <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
          and total_bytes <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
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
  stage's crossbar physically cannot deliver more than
  TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes -- a table whose key exceeds that
  budget is rejected by the compiler outright, not spread across stages. Raise
  rather than silently pricing that impossible design as
  `ceil(byte_width / budget)` stages.

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
  if byte_width > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), byte_width)
  shards, remaining = [], block_count
  while remaining > TCAM_ROWS_PER_STAGE:
    shards.append((TCAM_ROWS_PER_STAGE, byte_width))
    remaining -= TCAM_ROWS_PER_STAGE
  shards.append((remaining, byte_width))
  return shards


# --------------------------------------------------------------------------
# The classification pool's ordered stage simulation (audit C5)
# --------------------------------------------------------------------------

# Bounded caches: a campaign prices many thousands of trials, each with its
# own keys, and only the few keys of the trial in hand are ever re-asked.
@functools.lru_cache(maxsize=4096)
def _lane_key_bytes(field_bit_widths):
  """lanes.key_bytes, memoised per key (the placement loop re-prices the same
  few keys many times)."""
  return tuple(lanes.key_bytes(field_bit_widths))


@functools.lru_cache(maxsize=4096)
def _first_key_groups(field_bit_widths):
  """How many groups the FIRST key of a stage occupies in the lane
  simulation: its production price lanes.table_blocks -- the price it is
  CHARGED (plan invariant 1) -- or its own lane price when that is higher
  (they are the same quantity since 2026-10-04; the max stays as a guard).
  A key with no lane fit at all occupies every group."""
  lane = lanes.standalone(_lane_key_bytes(field_bit_widths))
  if lane is None:
    groups = TERNARY_CROSSBAR_GROUPS_PER_STAGE
  else:
    groups = max(lanes.table_blocks(field_bit_widths), lane)
  return min(groups, TERNARY_CROSSBAR_GROUPS_PER_STAGE)


@functools.lru_cache(maxsize=16384)
def _stage_key_prices(seed_keys, keys):
  """Per-512-row-word block prices of one stage's distinct classification
  keys, priced one at a time in placement order -- lanes.stage_prices with
  the first key charged its production price, not its lane price.

  seed_keys : field-bit tuples of keys ANOTHER pool already placed in the
              stage (range tables, through seed_stages). They are placed
              first, occupy crossbar lanes like any key, and are never
              charged here.
  keys      : field-bit tuples of this pool's distinct keys, in placement
              order.

  The first key placed in the stage pays lanes.table_blocks (plan invariant 1:
  a stage holding one key is priced exactly as lanes.table_blocks
  prices it, which is what keeps every 'joint' design identical to
  threshold alignment's total_blocks) and occupies _first_key_groups groups
  with lanes.first_key_occupancy. Every later key pays
  lanes.price_with_supply, its LEFTOVER price in what the keys before it
  left, and then occupies lanes.later_key_occupancy. The first of this
  pool's keys behind a seed is a later key of the stage, but is never
  charged below lanes.table_blocks either.

  Returns one price per entry of `keys`, or None when some key (seed or
  not) has no lane-legal fit -- the stage does not fit."""
  free = mids = None
  for bits in seed_keys:
    key = list(_lane_key_bytes(bits))
    if free is None:
      groups = lanes.standalone(key)
      if groups is None:
        return None
      free, mids = lanes.first_key_occupancy(key, groups)
      continue
    price = lanes.price_with_supply(key, free, mids)
    if price is None:
      return None
    free, mids = lanes.later_key_occupancy(free, mids, price)
  prices = []
  for position, bits in enumerate(keys):
    key = list(_lane_key_bytes(bits))
    if free is None:
      prices.append(lanes.table_blocks(bits))
      free, mids = lanes.first_key_occupancy(key, _first_key_groups(bits))
      continue
    price = lanes.price_with_supply(key, free, mids)
    if price is None:
      return None
    if position == 0:
      price = max(price, lanes.table_blocks(bits))
    prices.append(price)
    free, mids = lanes.later_key_occupancy(free, mids, price)
  return tuple(prices)


@dataclass(frozen=True)
class _Unit:
  """One placeable piece of a classification table: the whole table, or --
  for a table charged more than a stage's TCAM_BLOCKS_PER_STAGE -- one chunk
  of its rows."""
  table_idx: int
  chunk: int
  blocks: int            # charged if its key is the stage's first
  rows: int              # 512-row words -- what a later key's price multiplies
  width: int             # crossbar bytes of the key
  fields: frozenset
  bits: tuple
  level: int
  priority: int


def _classification_units(table_specs, key_fields, key_field_bits, levels,
                          priorities):
  """table_specs -> _Units. A table's rows are its blocks / its key's
  lanes.table_blocks, rounded up (exact for every spec the model builds,
  which is lanes.table_blocks x tree_entries_to_blocks). A table charged more
  than one stage's TCAM_BLOCKS_PER_STAGE cannot sit in one stage and is split
  into chunks of whole rows that each can; every chunk keeps the full key."""
  units = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    _stage_shards(block_count, byte_width)   # raises for an impossible key
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    bits = tuple(key_field_bits[idx])
    per_row = max(1, lanes.table_blocks(bits))
    rows = max(1, math.ceil(block_count / per_row))
    rows_per_chunk = max(1, TCAM_BLOCKS_PER_STAGE // per_row)
    remaining_blocks, remaining_rows, chunk = block_count, rows, 0
    while True:
      chunk_rows = min(remaining_rows, rows_per_chunk)
      chunk_blocks = (remaining_blocks if chunk_rows == remaining_rows
                      else min(remaining_blocks, per_row * chunk_rows))
      units.append(_Unit(idx, chunk, chunk_blocks, chunk_rows, byte_width,
                         fields, bits, levels[idx], priorities[idx]))
      remaining_blocks -= chunk_blocks
      remaining_rows -= chunk_rows
      chunk += 1
      if remaining_rows <= 0:
        break
  return units


def _stage_charges(units, seed):
  """The charged blocks of every unit in one stage, in placement order, or
  None when the stage does not fit. `units` are this pool's, in the order
  they were placed; `seed` is another pool's StageLoad for the same stage.

  A stage fits when all of these hold:
    * at most TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE tables (seed included);
    * at most TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes of distinct fields;
    * every key has a lane price (_stage_key_prices);
    * the charged blocks, seed shards included, pack 12x2 (fits_two_columns).
  """
  if len(units) + seed.tables > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE:
    return None
  present = frozenset().union(*(unit.fields for unit in units)) | seed.fields
  if (sum(field_bytes for _, field_bytes in present)
      > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE):
    return None
  order, key_bits = [], {}
  for unit in units:
    if unit.fields not in key_bits:
      order.append(unit.fields)
      key_bits[unit.fields] = unit.bits
  seed_fields = sorted(seed.fields, key=repr)
  prices = _stage_key_prices(
      tuple((8 * field_bytes,) for _, field_bytes in seed_fields),
      tuple(key_bits[key] for key in order))
  if prices is None:
    return None
  charges = []
  for unit in units:
    position = order.index(unit.fields)
    if position == 0 and not seed_fields:
      charges.append(unit.blocks)
    elif position == 0:
      charges.append(max(unit.blocks, prices[0] * unit.rows))
    else:
      charges.append(prices[position] * unit.rows)
  shards = list(seed.blocks) + [width for charge in charges
                                for width, _ in _stage_shards(charge, 1)]
  if not fits_two_columns(shards):
    return None
  return charges


def _ordered_stage_simulation(table_specs, key_fields, key_field_bits, levels,
                              priorities, unavailable_stages, seed_at,
                              last_seed):
  """The classification pool's placement: p4c's table placer, stage by
  stage. At stage s, repeatedly take the FIRST remaining table, in the order
  (placement priority descending, then the table listed LAST first), whose
  level is <= s and that fits the stage as it now stands (_stage_charges);
  when none does, move to s + 1. A stage in unavailable_stages is skipped.
  Returns ({stage index: [(unit, charge), ...]}, {table_idx: latest stage})."""
  units = _classification_units(table_specs, key_fields, key_field_bits,
                                levels, priorities)
  remaining = sorted(units, key=lambda u: (-u.priority, -u.table_idx, u.chunk))
  if not remaining:
    return {}, {}
  placed, charges = {}, {}
  stage = min(unit.level for unit in remaining)
  # Past every level, seed and unavailable stage, an empty stage takes any
  # remaining unit alone (one key, charged its own blocks, <= one stage wide),
  # so the loop always ends; the bound only turns a bug into an error.
  horizon = (max(unit.level for unit in remaining) + len(units)
             + max(list(unavailable_stages) + [last_seed, 0]) + 2)
  while remaining:
    if stage > horizon:
      raise RuntimeError(
          "crossbar_stages_needed: %d classification table(s) could not be "
          "placed by stage %d" % (len(remaining), horizon))
    if stage in unavailable_stages:
      stage += 1
      continue
    seed = seed_at(stage)
    here = placed.get(stage, [])
    for unit in remaining:
      if unit.level > stage:
        continue
      charged = _stage_charges(here + [unit], seed)
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


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           unavailable_stages=frozenset(), key_field_bits=None,
                           seed_stages=(), placement_priority=None):
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

  unavailable_stages (optional) names stage indices no table may occupy,
  whatever their capacity -- gated_block_interior_stages produces them. It
  is only meaningful alongside readiness_levels: without those the returned
  plan has no absolute stage indices to exclude, so it is ignored.

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
    * <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes of DISTINCT key fields
  (RM-5/RM-6/RM-7, reviews/p4_tofino_reference.md §4.3.) These constraints
  are not separable: tables (20 blocks, 5 B), (20, 5), (1, 60) need 3 stages
  although each relaxation alone says 2.

  THE RANGE POOL (key_field_bits=None) is packed as before: every table is
  charged its declared block count wherever it lands; with readiness_levels,
  shards are placed eagerly in (level, largest-load-first) order at the
  earliest legal stage with room (the compiler does not pack to the
  theoretical optimum, and neither does this -- M2's range pool really
  occupies 2 stages); without them, first-fit-decreasing. A range table's
  key is one whole-byte 16-bit field, and 8 of them (the table cap) fill 16
  bytes, so no sharing price applies.

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
    * Price -- each stage's distinct keys are priced in the order their
      first table was placed (_stage_key_prices): the first key pays its
      declared blocks, i.e. lanes.table_blocks x rows (plan invariant 1: a
      one-key stage, as every 'joint' stage is, is priced exactly as
      before); every later key pays its lanes.price_with_supply LEFTOVER
      price per 512-row word. A seeded stage's range keys come first, so
      every tree key there is a later key -- never charged below
      lanes.table_blocks.
    * Fit -- at most 8 tables, at most 64 bytes of distinct key fields,
      every key priced, and the charged blocks pack 12x2. There is no
      mixed-key byte threshold below 64: the 62-byte refusal was retired
      2026-09-29, valid only while the generator pins every tree key's PHV
      layout (target.py).
  A table charged more than one stage's 24 blocks is split into chunks of
  whole rows that each fit a stage.

  A SINGLE-KEY POOL IS NOT SIMULATED. When every classification table keys
  the same field set -- every 'joint' design -- and no seed sits at or after
  the pool's lowest readiness level, there is no pricing question for an
  order to settle: every table is its stage's first (and only) key, charged
  its declared blocks. Such a pool is placed exactly as before C5, by the
  declared-price packer the range pool uses (eager (level, largest-load)
  shards; first-fit-decreasing without levels), so 'joint' blocks AND
  stage_depth are structurally identical to the pre-C5 model (plan
  invariant 1), not merely on the archived designs. The simulation's
  placement order is a p4c-order heuristic for bin packing as well as for
  pricing, and on single-key pools it packs differently from the old FFD in
  ~1% of realistic tree-size mixes (task-5 review fuzz); what it exists for
  -- which key a stage prices first -- only arises with two keys.
  tests/test_p4model_guards.py fuzzes this against an independent reference.

  Measured on the 43 pragma'd arm-D
  compiles (results/compiler_calibration_pinned/): stage_depth 42/43 (the
  miss, independent_high_sd12, is the one design where the pragma itself
  cost p4c a stage) and blocks 38/38.
  """
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

  if key_field_bits is not None and len(key_field_bits) != len(table_specs):
    raise ValueError(
        "crossbar_stages_needed: got %d key_field_bits for %d table_specs; the "
        "two must be positionally aligned, one field-width tuple per table, or "
        "a table would be priced against another table's key shape"
        % (len(key_field_bits), len(table_specs)))
  if placement_priority is not None:
    if key_field_bits is None:
      raise ValueError(
          "crossbar_stages_needed: placement_priority orders the "
          "classification pool's stage simulation, which needs key_field_bits")
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

  distinct_keys = {key_fields[idx] if key_fields is not None
                   else ("<private>", idx) for idx in range(len(table_specs))}
  lowest_level = min(readiness_levels, default=0) if readiness_levels else 0
  seed_reachable = any(index >= lowest_level for index in seeds)
  if key_field_bits is not None and (len(distinct_keys) > 1 or seed_reachable):
    levels = (list(readiness_levels) if readiness_levels is not None
              else [0] * len(table_specs))
    priorities = (list(placement_priority) if placement_priority is not None
                  else [0] * len(table_specs))
    by_index, table_stages = _ordered_stage_simulation(
        table_specs, key_fields, key_field_bits, levels, priorities,
        frozenset(unavailable_stages) if readiness_levels is not None
        else frozenset(), seed_at, max(seeds, default=0))
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

  # ---- the range pool, and a single-key classification pool (see the
  # docstring): every table charged its declared blocks, placed as before C5.
  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    for shard in _stage_shards(block_count, byte_width):
      shards.append((shard[0], shard[1], idx, fields))

  def load(shard):
    blocks, width, _, _ = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  def fits(stage, blocks, fields):
    # stage: [fields present, shard blocks in placement order, seed]
    seed = stage[2]
    present = stage[0] | fields | seed.fields
    return (sum(field_bytes for _, field_bytes in present)
            <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
            and len(stage[1]) + seed.tables + 1
            <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
            and fits_two_columns(list(seed.blocks) + stage[1] + [blocks]))

  def place(stage, blocks, fields):
    stage[0] |= fields
    stage[1].append(blocks)

  def empty(seed):
    return [set(), [], seed]

  def as_loads(stages_by_index):
    return [StageLoad(index=index, blocks=tuple(stage[1]),
                      fields=frozenset(stage[0]), tables=len(stage[1]))
            for index, stage in stages_by_index.items()]

  table_stages = {}
  table_blocks = {}
  if readiness_levels is None:
    stages = []
    for blocks, _width, table_idx, fields in sorted(shards, key=load,
                                                    reverse=True):
      for position, stage in enumerate(stages):
        if fits(stage, blocks, fields):
          place(stage, blocks, fields)
          break
      else:
        position = len(stages)
        stages.append(empty(seed_at(position)))
        place(stages[-1], blocks, fields)
      table_stages[table_idx] = max(position,
                                    table_stages.get(table_idx, position))
      table_blocks[table_idx] = table_blocks.get(table_idx, 0) + blocks
    return plan(as_loads(dict(enumerate(stages))), table_stages, table_blocks)

  # Dependency-aware placement. Three differences from the packer above, all
  # chosen to track the REAL compiler rather than the theoretical optimum:
  #   1. A table may not occupy a stage index below its readiness level --
  #      its key value literally does not exist yet.
  #   2. Placement is EAGER (earliest legal stage with room), not "pack as
  #      few stages as possible". Measured: M2's range pool really occupies 2
  #      stages, which only eager placement reproduces.
  #   3. A table may not occupy an unavailable_stages index at all -- stages
  #      the placer spends entirely inside a gated register block
  #      (gated_block_interior_stages); checked before `fits`, since an
  #      interior stage stays unusable however empty it is.
  # A seeded stage this pool has not opened yet is tested through `fits`
  # with its seed; an unseeded empty stage always takes a lone shard.
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}
  ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]], -load(s)))
  for blocks, _width, table_idx, fields in ordered:
    index = readiness_levels[table_idx]
    while (index in unavailable_stages or
           not fits(by_index.get(index) or empty(seed_at(index)),
                    blocks, fields)):
      index += 1
    if index not in by_index:
      by_index[index] = empty(seed_at(index))
    place(by_index[index], blocks, fields)
    table_stages[table_idx] = max(index, table_stages.get(table_idx, index))
    table_blocks[table_idx] = table_blocks.get(table_idx, 0) + blocks
  return plan(as_loads(by_index), table_stages, table_blocks)
