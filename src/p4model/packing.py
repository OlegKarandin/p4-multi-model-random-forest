"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py, errors.py and tables.py's crossbar geometry only.
Deliberately knows nothing about features, registers or trees -- it is handed
(blocks, byte_width) specs plus readiness levels and returns a placement, which
is what lets the same packer serve both the range pool and the classification
pool. The one thing it must ask tables.py is version_block_penalty: a table's
block count is not a property of the table alone, so the placement and the
charge have to be computed together (see crossbar_stages_needed's
key_field_bits)."""
import itertools
import math
from dataclasses import dataclass

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.tables import (codeword_bytes_to_blocks, codeword_to_blocks,
                                version_block_delta)
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
                         # not the naive per-table sum passed in as table_specs, since the
                         # version-block penalty (see crossbar_stages_needed's
                         # key_field_bits) depends on the group offset a key gets, which
                         # is a per-STAGE placement fact rather than a per-table one.
                         # Measured to matter: independent_low_sd5's three ddos trees cost
                         # 3 TCAM blocks each where their 11-byte key buys 2, so the naive
                         # sum reports 13 against p4c's 16 (resources.json, stage 6).

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
  the shards sum to the table. No table in the 19-row archive exceeds 12 blocks,
  so this path is unexercised by the calibration and therefore unvalidated
  against hardware -- it is the one non-monotone (cost-lowering) change in this
  work, licensed because it corrects rounding rather than relaxing a measured
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


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           unavailable_stages=frozenset(), key_field_bits=None):
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
  reviews/p4_tofino_reference.md §4.3): joint_low_sd7's stage 7 holds four
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

  key_field_bits (optional) is one tuple of key-field BIT widths per table,
  positionally aligned with table_specs. It exists because a table's BLOCK
  COUNT is not a property of the table alone: the crossbar hands out groups in
  one consecutive run per key, so the second distinct key in a stage starts at
  the group offset the first one ended at, and where a key's run starts decides
  whether any crossbar MIDBYTE keeps a free nibble for the mandatory 2-bit
  --version-- field. When none does, p4c pads the table with a whole extra
  TCAM. tables.version_block_penalty is the rule and carries the measurement;
  the short version is that it fires on a key that saturates the byte slots its
  groups supply, and that saturation depends on the offset.

  Confirmed from the pack format, not inferred: sharing, the table's memory
  unit 0 holds the version field and NOTHING else (bits [41:0] empty); alone,
  unit 0 holds version plus 40 bits of match data. That is
  TableFormat::ternary_version() push_back()ing a block because no midbyte
  nibble was left -- the same waste
  reviews/github_issue_tcam_version_bit_packing.md documents.

  Measured directly, scripts/tcam_stretch_sweep.py, one artifact set: a table
  keying 179 + 204 bits (49 crossbar bytes, 9 groups) costs 9 TCAMs ALONE in a
  stage and 10 when a 12-byte key holds groups 0..2 ahead of it; the same
  block-and-byte geometry built from SOLID single fields costs 9 either way.
  And directly in the calibration set, from resources.json rather than a probe:
  independent_low_sd5's three ddos trees each cost 3 TCAMs where their 11-byte
  key buys 2. Appendix B "Mechanism G".

  Passing key_field_bits=None -- the default -- charges every table its
  declared block count at every offset, i.e. exactly the pre-existing pricing.
  The range pool never passes it: a range table keys one whole-byte field with
  slack to spare, so the penalty is inert there by construction.

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
  reviews/archive/t12_required_changes.md Section 1.3, confirmed across key
  widths 8-512 bits; summarised in reviews/p4_tofino_reference.md §4.3.)

  A per-stage crossbar group cap was probed and not found: two solid keys of
  34 and 30 crossbar bytes need 7+6=13 groups in a stage that has 12, pass
  the 64-byte cap at exactly 64, and p4c placed them in one stage
  (scripts/tcam_group_cap_probe.py, point groups_13_bytes_64, with
  groups_12_bytes_59 as the control, probed 2026-09-15).

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

  if key_field_bits is not None and len(key_field_bits) != len(table_specs):
    raise ValueError(
        "crossbar_stages_needed: got %d key_field_bits for %d table_specs; the "
        "two must be positionally aligned, one field-width tuple per table, or "
        "a table would be priced against another table's key shape"
        % (len(key_field_bits), len(table_specs)))

  shards = []
  for idx, (block_count, byte_width) in enumerate(table_specs):
    # A caller that names no fields gets a PRIVATE synthetic one per table,
    # which makes the union accounting below numerically identical to the
    # per-table sum it replaces -- that is what keeps key_fields=None exactly
    # backwards compatible rather than approximately so.
    fields = (key_fields[idx] if key_fields is not None
              else frozenset({(("<private>", idx), byte_width)}))
    bits = tuple(key_field_bits[idx]) if key_field_bits is not None else None
    # Only the FIRST shard carries the key's field widths. A table wider than
    # one TCAM column is split into column-sized pieces (_stage_shards), but
    # the version field is stored once per table word, not once per shard, so
    # charging every piece would multiply a single 2-bit field's cost.
    for position, shard in enumerate(_stage_shards(block_count, byte_width)):
      shards.append((shard[0], shard[1], idx, fields,
                     bits if position == 0 else None))

  def load(shard):
    blocks, width, _, _, _ = shard
    return max(blocks / TCAM_BLOCKS_PER_STAGE,
               width / TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
               1 / TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE)

  def crossbar_bytes(fields):
    return sum(field_bytes for _, field_bytes in fields)

  def key_width(fields, bits):
    """Crossbar groups this KEY occupies, version block included.

    Not the table's block count: a table two blocks deep stores more rows
    through the same key, and depth does not move the next key along the
    crossbar (finding 1.4). The version block DOES -- p4c starts
    independent_low_sd5's app key at group 3 behind an 11-byte ddos key that
    buys 2 groups and pays a third for --version-- (see
    scripts/tcam_offset_harvest.py).

    bits is None for any caller that names no field widths (the range pool),
    and then the key's byte width is all there is; a range key is one
    whole-byte field with slack to spare, so no version charge is due.
    """
    if bits:
      return codeword_to_blocks(bits, 0)
    return codeword_bytes_to_blocks(crossbar_bytes(fields))

  def offsets_for(key_order):
    """Group offset each key starts at, given the order the crossbar hands
    groups out in. One group feeds one block, so a key's start is the running
    sum of the WIDTHS (key_width) of the keys ahead of it -- never their
    tables' block counts, which can run deeper than one group's worth of rows
    when a tree exceeds 512 entries or is sharded. A table's own extra depth
    stores more rows through the SAME key and does not move the crossbar
    along; only the version-block charge folded into key_width does that
    (finding 1.4 -- see key_width above)."""
    offsets, running = {}, 0
    for key, key_blocks in key_order:
      offsets[key] = running
      running += key_blocks
    return offsets

  def charged(offsets, blocks, fields, bits):
    # The DELTA, not the penalty: `blocks` already prices this key standalone,
    # i.e. at group offset 0, so a placement only adds what the offset changes.
    # Charging the penalty itself would re-bill a key that already paid in its
    # own declared block count -- independent_low_sd5's ddos key does exactly
    # that (it pays at offset 0), and would then be charged twice.
    if not bits:
      return blocks
    return blocks + version_block_delta(bits, offsets[fields])

  def fits(stage, blocks, fields, bits):
    # stage[3] is the per-shard (blocks, key, field-bits) already here, stage[4]
    # the distinct keys with their group counts. A new key shifts every later
    # key's offset, so the whole stage is re-priced against the key set it
    # would HAVE -- a table already placed can become more expensive.
    #
    # Every ORDER of those keys is tried and the stage fits only if ALL of
    # them pack. The crossbar hands out groups in the order tables enter the
    # stage, which is p4c's own placement order and not this packer's -- and
    # it is measurably NOT the cheapest one: in both artifacts where the two
    # differ, p4c gave the low groups to the key that made the OTHER one pay.
    # (probe ragged_ax1_bx4: the 12-byte key takes groups 0..2 and the 49-byte
    # key pays at group 3; independent_low_sd5: the ddos key takes groups 0..1
    # and pays there while the app key sits free at group 3.) Fitting on the
    # cheapest order would under-count -- ragged_ax1_bx5 really needs 2 stages
    # and packs into 1 if the wide key is allowed to claim group 0. Requiring
    # every order keeps the estimator on the safe side of the 12-stage gate,
    # and costs nothing on real data: over the 19 calibration rows this choice
    # changes no block count and no stage depth. Distinct keys per stage are one to three here, so
    # this is a handful of permutations.
    keys = list(stage[4])
    if all(key != fields for key, _ in keys):
      keys.append((fields, key_width(fields, bits)))
    if (crossbar_bytes(stage[1] | fields) > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
        or stage[2] + 1 > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE):
      return False
    shards_here = stage[3] + [(blocks, fields, bits)]
    for key_order in itertools.permutations(keys):
      offsets = offsets_for(key_order)
      # The TCAM test is a column PACKING, not a running total against
      # TCAM_BLOCKS_PER_STAGE -- see fits_two_columns: three 8-block tables
      # sum to exactly 24 and still do not fit, four 6-block ones do.
      if not fits_two_columns([charged(offsets, *shard)
                               for shard in shards_here]):
        return False
    return True

  def place(stage, blocks, fields, bits):
    stage[0] += blocks
    stage[1] |= fields
    stage[2] += 1
    stage[3].append((blocks, fields, bits))
    if all(key != fields for key, _ in stage[4]):
      stage[4].append((fields, key_width(fields, bits)))

  def opened(blocks, fields, bits):
    return [blocks, set(fields), 1, [(blocks, fields, bits)],
            [(fields, key_width(fields, bits))]]

  def stage_charged_blocks(stage):
    """The TCAM blocks ONE finished stage actually costs, version-block
    charge included -- the same question `fits()` already answers for
    placement, asked once more after the fact so the total can be reported.

    Tries every order of the stage's distinct keys, exactly as `fits()`
    does, and keeps the LARGEST total. Not the order tables arrived in
    (place() records that, but the crossbar does not follow it): this packer
    cannot know which order p4c's placer will pick, and 'largest' is this
    module's standing rule for that uncertainty -- consistent with never
    under-counting real hardware (crossbar_stages_needed's own FFD
    upper-bound rationale). It is also what the two measurements show, since
    in both of them p4c picked the EXPENSIVE order: probe ragged_ax1_bx4
    compiles to 22 blocks, not the 21 its cheap order would give, and
    independent_low_sd5's stage 6 to 12, not 9.

    No feasibility filter is applied. `fits` already requires EVERY order to
    pack before a shard joins an existing stage, so for those stages the
    filter would be a no-op. The one stage it could reject is a freshly
    `opened()` one, which is committed without a `fits` call -- a lone
    12-block shard that then takes the version charge would be 13, wider than
    a TCAM column. That stage is still real and still costs those blocks, so
    it must be counted rather than dropped."""
    return max(sum(charged(offsets_for(key_order), blocks, fields, bits)
                   for blocks, fields, bits in stage[3])
               for key_order in itertools.permutations(stage[4]))

  if readiness_levels is None:
    # entry: [blocks_used, fields_present, tables_used, shards, key_order]
    stages = []
    for blocks, _width, _idx, fields, bits in sorted(shards, key=load,
                                                     reverse=True):
      for stage in stages:
        if fits(stage, blocks, fields, bits):
          place(stage, blocks, fields, bits)
          break
      else:
        stages.append(opened(blocks, fields, bits))

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
  for blocks, _width, table_idx, fields, bits in ordered:
    index = readiness_levels[table_idx]
    while (index in unavailable_stages or
           (index in by_index
            and not fits(by_index[index], blocks, fields, bits))):
      index += 1
    if index in by_index:
      place(by_index[index], blocks, fields, bits)
    else:
      by_index[index] = opened(blocks, fields, bits)

  return StagePlan(occupied=len(by_index),
                    depth=(max(by_index) + 1) if by_index else 0,
                    indices=frozenset(by_index.keys()),
                    blocks=sum(stage_charged_blocks(stage)
                              for stage in by_index.values()))
