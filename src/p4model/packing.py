"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py and errors.py only. Deliberately knows nothing about
features, registers or trees -- it is handed (blocks, byte_width) specs plus
readiness levels and returns a placement, which is what lets the same packer
serve both the range pool and the classification pool."""
import itertools
import math
from dataclasses import dataclass

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.target import (
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
)


@dataclass(frozen=True)
class StagePlan:
  """crossbar_stages_needed's placement, not just its size -- F10: the stage
  a pool is DONE at (depth) is not the same quantity as how many stages it
  OCCUPIES (occupied): a stage can fill at the 8-table crossbar cap and spill
  a table forward past every level actually requested, so depth must be read
  from where tables landed, not from max(readiness_levels) + 1."""
  occupied: int          # how many stage indices hold a table from this pool
  depth: int             # max(occupied index) + 1 -- the quantity a 12-stage ceiling reads
  indices: frozenset     # for assertions and debugging
  blocks: int            # total TCAM blocks actually CHARGED across every stage --
                         # not the naive per-table sum passed in as table_specs, since
                         # a ragged key's group-offset penalty (see crossbar_stages_needed's
                         # ragged_keys) is a per-STAGE placement fact, not a per-table one.
                         # Measured to matter: independent_low_sd5 (compiler_calibration_v6,
                         # 2026-09-06) reported 13 from the naive sum where p4c used 16.

  def __int__(self):     # transitional: `stages` is still the occupancy count
    return self.occupied


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
  scripts/tcam_column_sweep.py, reviews/p4_tofino_reference.md Sec 7
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
  tables of 14, 16 and 24 blocks."""
  if byte_width > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE:
    raise CrossbarKeyTooWide(
        "table key is %d crossbar bytes; no stage supplies more than %d, so the "
        "compiler rejects this table rather than splitting it across stages"
        % (byte_width, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE), byte_width)
  n = max(1, math.ceil(block_count / TCAM_ROWS_PER_STAGE)) if block_count > 0 else 1
  return [(math.ceil(block_count / n), byte_width)] * n


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           unavailable_stages=frozenset(), ragged_keys=None):
  """Packs independent match tables into pipeline stages under ALL three
  per-stage hardware limits simultaneously, and returns a StagePlan
  describing where the tables landed (not just how many stages that took).

  table_specs is one (block_count, byte_width) pair per independent P4
  table -- one per tree for the ternary classification tables
  (build_p4_script.py:636-659), one per feature for the range-matching
  tables (build_p4_script.py:663-674).

  key_fields (optional) is one frozenset of (field_id, field_bytes) per
  table, positionally aligned with table_specs, naming WHICH crossbar
  fields that table's key is made of. It exists because the Ternary Match
  Input crossbar charges per distinct FIELD placed on its byte slots, not
  per (table, field) pair: two tables in the same stage that match on the
  same field read the same slots and the field is charged ONCE. That is not
  a corner case in this generator -- every classification table of one task
  keys on the identical meta.code_<feature> field set (see
  ternary_matching_resource_usage), and under 'disjoint' both models' range
  tables for a shared feature key the identical meta.<feature>_val field
  (build_p4_script._resolve_disjoint_feature_plan: the raw value field is
  always shared, and a code_<feature> field is shared whenever both models
  select the feature with identical intervals).

  Measured against 19 real p4c compiles (results/compiler_calibration/, and
  reviews/p4_tofino_reference.md Sec 7): joint_low_sd7's stage 7 holds four
  tables on one 32-byte codeword and the compiler reports 32 crossbar bytes,
  not 128; independent_low_sd6's stage 7 holds two 19-byte and two 4-byte
  tables and reports 23, not 46. Summing per table over-counted stages by up
  to 6 on that sample -- for a 41-byte codeword the sum-based rule computes
  floor(64/41) = 1 table per stage where the hardware takes 2 or more.

  Passing key_fields=None keeps the old, conservative per-table accounting
  exactly: each table is given a private synthetic field, so the union
  arithmetic below reduces to the sum it replaces. Every pre-existing caller
  is therefore unaffected.

  unavailable_stages (optional) names stage indices no table may occupy,
  whatever their capacity -- gated_block_interior_stages produces them.
  It is only meaningful alongside readiness_levels: without those the
  returned plan has no absolute stage indices to exclude, so the pure packer
  below ignores it.

  ragged_keys (optional) is one bool per table, positionally aligned with
  table_specs, saying whether that table's key leaves part-used crossbar
  bytes (ternary_key_is_ragged). It exists because a table's BLOCK COUNT is
  not a property of the table alone: the crossbar hands out groups in one
  consecutive run per key, so the second distinct key in a stage starts at
  the group offset the first one ended at, and a ragged key starting on an
  ODD group reaches one midbyte fewer than it does at offset 0 -- the midbyte
  at its low end is shared with the previous key's last group. p4c then pads
  the table with a whole extra TCAM. Confirmed from the pack format, not
  inferred: sharing, the table's memory unit 0 holds the 2-bit --version--
  field and NOTHING else (bits [41:0] empty); alone, unit 0 holds version
  plus 40 bits of match data. That is TableFormat::ternary_version()
  push_back()ing a block because no midbyte nibble was left for version --
  the same waste reviews/github_issue_tcam_version_bit_packing.md documents,
  reached by a new trigger.

  Measured directly, scripts/tcam_stretch_sweep.py, one artifact set:
  a table keying 179 + 204 bits (49 crossbar bytes, 9 groups) costs 9 TCAMs
  ALONE in a stage and 10 when a 12-byte key holds groups 0..2 ahead of it;
  the same block-and-byte geometry built from SOLID single fields costs 9
  either way. That +1 is the whole of independent_low_sd9's stage divergence
  (10 blocks + 5 x 3 = 25 > TCAM_BLOCKS_PER_STAGE), the last one the
  compiler-calibration study had open. Ref 7 "Mechanism G".

  Passing ragged_keys=None -- the default -- charges every table its declared
  block count at every offset, i.e. exactly the pre-existing pricing.

  RM-5/RM-6/RM-7 measured these limits
  on the Ternary Match Input crossbar specifically. A follow-up compile
  sweep (reviews/open_issues.md item 3, results_rmx_crossbar.csv) confirmed
  the 8-tables/stage cap generalizes to range tables at 16-bit width, but
  found range tables cost ~2x the crossbar xbar-units per byte that ternary
  tables do at the same width -- so reusing TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
  verbatim for the range pool is NOT known-conservative; once a design's
  range-table byte-budget (rather than the 8-table cap) becomes the binding
  constraint, this function likely UNDER-counts range_stages instead of
  over-counting it. The exact byte-width crossover for range tables was not
  pinned down (needs a >64-bit combined-width multi-field range sweep).
  Both pools are packed separately with this one function, since they are
  physically distinct table pools.

  Every stage must satisfy at once:
    * <= TCAM_BLOCKS_PER_STAGE                 TCAM blocks
    * <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE independent tables
    * <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE  bytes of DISTINCT key fields
                                               present in the stage (see
                                               key_fields above; without it,
                                               each table's key is its own)
  (the table-count and byte caps are RM-5/RM-6/RM-7,
  reviews/t12_required_changes.md Section 1.3, confirmed across key widths
  8-512 bits.)

  These constraints are NOT separable: solving each relaxation alone and
  taking the max can under-count. Counterexample -- tables
  (20 blocks, 5 B), (20, 5), (1, 60): the blocks-only bound is
  ceil(41/24) = 2 and the crossbar-only bound is 2, but no two of the three
  fit in one stage (20+20 = 40 blocks > 24; 5+60 = 65 bytes > 64), so the
  true answer is 3.

  Uses first-fit-decreasing, sorting by each table's most-loaded dimension
  (its largest fraction of a per-stage limit) descending: the hardest
  tables to place go first, which is what makes FFD behave well when the
  binding dimension differs from table to table. Sort order only affects
  tightness, never validity -- FFD only ever emits a packing in which every
  stage respects all three limits, so its stage count is always an upper
  bound on the true optimum. That is the safe direction for an estimator
  that must never under-count real hardware usage."""

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

  if ragged_keys is not None and len(ragged_keys) != len(table_specs):
    raise ValueError(
        "crossbar_stages_needed: got %d ragged_keys for %d table_specs; the "
        "two must be positionally aligned, one flag per table, or a table "
        "would be priced against another table's key shape"
        % (len(ragged_keys), len(table_specs)))

  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    # A caller that names no fields gets a PRIVATE synthetic one per table,
    # which makes the union accounting below numerically identical to the
    # per-table sum it replaces -- that is what keeps key_fields=None exactly
    # backwards compatible rather than approximately so.
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    ragged = bool(ragged_keys[idx]) if ragged_keys is not None else False
    for shard in _stage_shards(block_count, byte_width):
      shards.append((shard[0], shard[1], idx, fields, ragged))

  def load(shard):
    blocks, width, _, _, _ = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  def crossbar_bytes(fields):
    return sum(field_bytes for _, field_bytes in fields)

  def offsets_for(key_order):
    """Group offset each key starts at, given the order the crossbar hands
    groups out in. One group feeds one block and a key's groups are
    consecutive, so a key set's group count IS its per-table block count --
    counted once however many tables share it, since the crossbar charges per
    field, not per (table, field)."""
    offsets, running = {}, 0
    for key, key_blocks in key_order:
      offsets[key] = running
      running += key_blocks
    return offsets

  def charged(offsets, blocks, fields, ragged):
    return blocks + 1 if ragged and offsets[fields] % 2 else blocks

  def fits(stage, blocks, fields, ragged):
    # stage[3] is the per-shard (blocks, key, ragged) already here, stage[4]
    # the distinct keys with their group counts. A new key shifts every later
    # key's offset, so the whole stage is re-priced against the key set it
    # would HAVE -- a table already placed can become more expensive.
    #
    # Every ORDER of those keys is tried and the stage fits if any one of them
    # packs. The crossbar hands out groups in the order tables enter the
    # stage, which is p4c's placement order, not this packer's; assuming one
    # order would invent refusals the compiler does not make (measured: probe
    # point ragged_ax1_bx4 fits with the narrow key first and does not with
    # the wide one first). Distinct keys per stage are one to three here, so
    # this is a handful of permutations.
    keys = list(stage[4])
    if all(key != fields for key, _ in keys):
      keys.append((fields, blocks))
    if (crossbar_bytes(stage[1] | fields) > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
        or stage[2] + 1 > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE):
      return False
    shards_here = stage[3] + [(blocks, fields, ragged)]
    for key_order in itertools.permutations(keys):
      offsets = offsets_for(key_order)
      # The TCAM test is a column PACKING, not a running total against
      # TCAM_BLOCKS_PER_STAGE -- see fits_two_columns: three 8-block tables
      # sum to exactly 24 and still do not fit, four 6-block ones do.
      if fits_two_columns([charged(offsets, *shard) for shard in shards_here]):
        return True
    return False

  def place(stage, blocks, fields, ragged):
    stage[0] += blocks
    stage[1] |= fields
    stage[2] += 1
    stage[3].append((blocks, fields, ragged))
    if all(key != fields for key, _ in stage[4]):
      stage[4].append((fields, blocks))

  def opened(blocks, fields, ragged):
    return [blocks, set(fields), 1, [(blocks, fields, ragged)],
            [(fields, blocks)]]

  def stage_charged_blocks(stage):
    """The TCAM blocks ONE finished stage actually costs, ragged charge
    included -- the same question `fits()` already answers for placement,
    asked once more after the fact so the total can be reported.

    Tries every order of the stage's distinct keys, exactly as `fits()`
    does, and keeps the LARGEST total that still respects fits_two_columns.
    Not the order actually used to justify placement (place() only ever
    records arrival order, and fits() may have accepted a stage via a
    DIFFERENT permutation than that): this packer does not know which
    order p4c's placer will pick (see fits()'s own docstring), and 'largest
    feasible' is this module's standing rule for that uncertainty --
    consistent with never under-counting real hardware (crossbar_stages_
    needed's own FFD-upper-bound rationale). Validated against ground
    truth on the one case with more than one feasible order (ragged_ax1_
    bx4): the cheaper order (25) is infeasible by fits_two_columns, so the
    max over FEASIBLE orders is 22 -- exactly what p4c compiled.

    At least one permutation is guaranteed feasible: this stage exists
    because place()/opened() only ever commit a shard once `fits` (the
    same search) found one."""
    best = None
    for key_order in itertools.permutations(stage[4]):
      offsets = offsets_for(key_order)
      charged_list = [charged(offsets, blocks, fields, ragged)
                      for blocks, fields, ragged in stage[3]]
      if fits_two_columns(charged_list):
        total = sum(charged_list)
        best = total if best is None else max(best, total)
    return best

  if readiness_levels is None:
    # entry: [blocks_used, fields_present, tables_used, shards, key_order]
    stages = []
    for blocks, _width, _idx, fields, ragged in sorted(shards, key=load,
                                                       reverse=True):
      for stage in stages:
        if fits(stage, blocks, fields, ragged):
          place(stage, blocks, fields, ragged)
          break
      else:
        stages.append(opened(blocks, fields, ragged))

    return StagePlan(occupied=len(stages), depth=len(stages),
                      indices=frozenset(range(len(stages))),
                      blocks=sum(stage_charged_blocks(stage) for stage in stages))

  # Dependency-aware placement. Three differences from the packer above, all
  # chosen to track the REAL compiler rather than the theoretical optimum:
  #
  #   1. A table may not occupy a stage index below its readiness level --
  #      its key value literally does not exist yet.
  #   2. Placement is EAGER (earliest legal stage with room), not "pack as
  #      few stages as possible". The optimum would drop every table into the
  #      single latest stage; the compiler does not do that, and neither does
  #      this. Measured: M2's range pool really occupies 2 stages, which only
  #      eager placement reproduces.
  #   3. A table may not occupy an unavailable_stages index at all. Those are
  #      stages the placer spends entirely inside a gated register block, so
  #      no table of the outer sequence is even a candidate there -- see
  #      gated_block_interior_stages. Unlike the three capacity limits this
  #      is not a fullness test: an interior stage stays unusable however
  #      empty it is, so it is checked before `fits` rather than through it.
  #
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}  # index -> [blocks, fields_present, tables, shards, key_order]
  ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]], -load(s)))
  for blocks, _width, table_idx, fields, ragged in ordered:
    index = readiness_levels[table_idx]
    while (index in unavailable_stages or
           (index in by_index
            and not fits(by_index[index], blocks, fields, ragged))):
      index += 1
    if index in by_index:
      place(by_index[index], blocks, fields, ragged)
    else:
      by_index[index] = opened(blocks, fields, ragged)

  return StagePlan(occupied=len(by_index),
                    depth=(max(by_index) + 1) if by_index else 0,
                    indices=frozenset(by_index.keys()),
                    blocks=sum(stage_charged_blocks(stage)
                              for stage in by_index.values()))
