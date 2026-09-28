"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py, errors.py and tables.py's crossbar geometry only.
Deliberately knows nothing about features, registers or trees -- it is handed
(blocks, byte_width) specs plus readiness levels and returns a placement, which
is what lets the same packer serve both the range pool and the classification
pool. What it must ask tables.py is a key's own standalone price
(codeword_to_blocks): a table's block count is not a property of the table
alone -- a key sharing a CROWDED stage with a different key can cost one block
more than the same key alone -- so the placement and the charge have to be
computed together (see crossbar_stages_needed's key_field_bits and the
crowded-stage margin in charged())."""
import itertools
import math
from dataclasses import dataclass

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.tables import codeword_bytes_to_blocks, codeword_to_blocks
from src.p4model.target import (
    TCAM_BLOCKS_PER_STAGE,
    TCAM_COLUMNS_PER_STAGE,
    TCAM_ROWS_PER_STAGE,
    TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE,
    TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE,
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
           (crowded-stage margin included), one entry per shard. Kept per shard
           rather than summed because the 12x2 column packing
           (fits_two_columns) needs the widths, not the total.
  fields : the distinct crossbar key fields present, (field_id, field_bytes)
           pairs -- the 64-byte limit charges their union, not their sum.
  tables : shards placed in the stage -- what the 8-table cap counts."""
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
                         # not the naive per-table sum passed in as table_specs, since the
                         # crowded-stage margin (see crossbar_stages_needed's
                         # key_field_bits and charged()) adds a block to every table of a
                         # non-first key when two DIFFERENT keys fill more than 58 of a
                         # stage's crossbar bytes. That is a per-STAGE placement fact, not
                         # a per-table one. Measured to matter: the ragged 49-byte key
                         # (179, 204) costs 9 TCAM blocks alone in a stage and 10 beside a
                         # 12-byte key, 61 bytes in all (scripts/tcam_stretch_sweep.py).
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

  def __int__(self):     # transitional: `stages` is still the occupancy count
    return self.occupied


def stage_load_fits(loads):
  """Whether several pools' StageLoads for the SAME stage index fit together
  under the three per-stage limits: <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
  shards, <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE bytes of distinct key fields,
  and a 12x2 column packing of every shard (fits_two_columns). The check
  crossbar_stages_needed's seed_stages makes while placing, restated as a
  predicate so a caller can assert it after the fact. The crowded-stage rules
  are not part of it: they are measured on ternary keys and applied inside the
  ternary pool only (see crossbar_stages_needed's seed_stages)."""
  loads = list(loads)
  fields = frozenset().union(*(load.fields for load in loads))
  return (sum(load.tables for load in loads) <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE
          and sum(field_bytes for _, field_bytes in fields)
          <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
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


def crossbar_stages_needed(table_specs, readiness_levels=None, key_fields=None,
                           unavailable_stages=frozenset(), key_field_bits=None,
                           seed_stages=()):
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
  COUNT is not a property of the table alone: when two DIFFERENT keys crowd a
  stage (more than 58 of its 64 crossbar bytes), the later key's bytes must
  land on whatever crossbar positions the first one left free, p4c's
  allocator is greedy, and short of room it takes one extra half-byte slot --
  a whole extra TCAM block for a key with none spare. charged() below is that
  margin and carries the derivation; key_field_bits marks the ternary pool
  (the only one the crowded-stage rules apply to) and lets key_width ask
  tables.codeword_to_blocks for each key's standalone price.

  Confirmed from the pack format, not inferred: sharing, the table's memory
  unit 0 holds the version field and NOTHING else (bits [41:0] empty); alone,
  unit 0 holds version plus 40 bits of match data. That is
  TableFormat::ternary_version() push_back()ing a block because no half-byte
  slot was left -- the same waste
  reviews/github_issue_tcam_version_bit_packing.md documents.

  Measured directly, scripts/tcam_stretch_sweep.py, one artifact set: a table
  keying 179 + 204 bits (49 crossbar bytes) costs 9 TCAMs ALONE in a stage
  (ragged_ax1_bx5) and 10 when a 12-byte key shares that stage
  (ragged_ax1_bx4, ragged_ax2_bx2); the same block-and-byte geometry built
  from SOLID single fields costs 9 either way, because a solid key has no
  nibble-clean byte for the allocator to spend a half-byte slot on. 49 + 12 =
  61 bytes is a crowded stage, so the margin charges both -- over on the solid
  control, right on the ragged one: one-sided by design, and bounded at
  exactly +1 block per affected table.

  Passing key_field_bits=None -- the default -- charges every table its
  declared block count wherever it lands and skips both crowded-stage rules,
  i.e. exactly the pre-existing pricing. The range pool never passes it: a
  range table keys one whole-byte 16-bit field and 8 of them (the table cap)
  fill 16 bytes, nowhere near crowding a stage.

  RM-5/RM-6/RM-7 measured these limits
  on the Ternary Match Input crossbar specifically. A follow-up compile
  sweep (reviews/open_issues.md item 3, results_rmx_crossbar.csv) confirmed
  the 8-tables/stage cap generalizes to range tables at 16-bit width, but
  found range tables cost ~2x the crossbar xbar-units per byte that ternary
  tables do at the same width -- so reusing TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
  verbatim for the range pool is NOT known-conservative in general. In
  practice it is unreachable for this generator: the 8-table cap holds a
  range stage to at most 8 whole-byte 2-byte fields, 16 bytes, nowhere near
  the 64-byte budget, so the byte cap never binds on the range pool before
  the table-count cap does. The exact byte-width crossover was never pinned
  down (needs a >64-bit combined-width multi-field range sweep), but nothing
  this project generates reaches it.
  Both pools are packed separately with this one function, since they are
  physically distinct table pools.

  Every stage must satisfy at once:
    * TCAM blocks, as a 12x2 COLUMN packing (fits_two_columns), not a flat
      <= TCAM_BLOCKS_PER_STAGE total
    * <= TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE independent tables
    * <= TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE  bytes of DISTINCT key fields
                                               present in the stage (see
                                               key_fields above; without it,
                                               each table's key is its own)
  (the table-count and byte caps are RM-5/RM-6/RM-7,
  reviews/archive/t12_required_changes.md Section 1.3, confirmed across key
  widths 8-512 bits; summarised in reviews/p4_tofino_reference.md §4.3.)

  The ternary pool (the only caller passing key_field_bits) has one more
  pair of limits, for a stage holding two or more DIFFERENT keys -- the
  "crowded stage" (spec "F5"; measurements in target.py). A later key must
  take whatever crossbar groups the first left, and p4c routes what does not
  fit through midbyte nibbles at extra TCAM blocks. Above
  TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE = 58 bytes every table of
  the non-first key is charged +1 (charged()'s is_crowded); above
  TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62 the stage is refused, since
  the measured extra there reaches +2. One-sided: some real stages at 59-62
  bytes pay nothing (independent_low_sd12 shares 60 bytes free), and the rule
  prices those designs one stage deeper than p4c. There is still no per-stage
  group-supply term; target.py's group counts remain documentation only.

  seed_stages (optional) is an iterable of StageLoad -- another pool's
  StagePlan.stage_loads -- naming stages that are already partly full. It
  exists because the two pools are no longer strictly sequential under
  'disjoint' (audit C1): a task's trees wait only for THEIR OWN task's range
  tables, so a classification tree can land in a stage still holding the other
  task's range tables. A seed's shards, fields and table count count against
  the 8-table cap, the 64-byte limit and the 12x2 column packing of its stage
  exactly as this pool's own would, but they are never charged again: the
  returned plan's blocks, occupied, depth, indices and stage_loads describe
  this pool alone. Seeds are range tables in practice, and they take no part
  in the crowded-stage rules above -- those were measured on ternary keys
  sharing a stage with ternary keys, and a stage's crowding (key count, bytes
  over 58/62) is judged on this pool's keys only. Like unavailable_stages it
  needs readiness_levels (absolute stage indices); passing seeds without them
  raises.

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
    """This KEY's own standalone block price -- how far it moves the next key
    along offsets_for's running sum.

    Not the table's block count: a table two blocks DEEP stores more rows
    through the same key, and depth does not move the next key along. That is
    finding 1.4, and keeping it right is why this function exists at all.

    What the result is used for is narrow. offsets_for chains these widths so
    that charged() can ask one question -- is this key the FIRST distinct key
    in the ordering under test, i.e. is its offset 0 -- and nothing reads an
    absolute position out of it. A key's own price no longer depends on where
    it sits: tables.codeword_to_blocks has had no start_group parameter since
    the 2026-09-20 rewrite, because the consecutive-run/paired-midbyte premise
    that parameter modelled was read out of p4c's own assembly as false. The
    running sum is kept rather than an ordinal because every price this branch
    returns is >= 1, so "offset 0" and "first" are the same predicate and the
    sum states the intent without introducing a second concept.

    HONEST LIMIT, so nobody mistakes finding 1.4 for something that is still
    guarded: that same ">= 1" makes the width-vs-blocks distinction
    UNOBSERVABLE from outside crossbar_stages_needed. Any positive advance
    gives every key the same first/not-first verdict, so replacing all three
    key_width call sites with `blocks` -- the exact finding 1.4 bug -- changes
    no occupied, blocks or depth over 79 916 random multi-key configurations,
    and reproduces test_a_deep_table_costs_only_its_own_extra_depth's 7 and 10
    unchanged. That test therefore pins the margin arithmetic, not this
    invariant; the invariant is held by construction here and by this comment,
    since key_width and offsets_for are closures with no reachable surface of
    their own. It becomes observable again the moment anything downstream
    reads an offset as a POSITION rather than as a boolean, which is exactly
    when it would need a test.

    bits is None for any caller that names no field widths (the range pool),
    and then the key's byte width is all there is; a range key is one
    whole-byte field with slack to spare, so it can never pay the margin.

    `bits is not None`, not a truthiness check: an empty tuple `bits = ()` --
    a degenerate all-single-leaf forest -- is a real key, and
    codeword_to_blocks(()) is 1 because a version field still needs a physical
    block even when the key itself claims no bytes. The truthiness form this
    replaces sent that case to the byte branch and priced it at 0, which would
    have left the NEXT key at offset 0 and silently suppressed that key's
    margin -- an under-prediction, the one direction this model forbids.
    Unreachable by anything the generator produces today; fixed rather than
    re-documented now that there is no offset gap left to fix it alongside.
    """
    if bits is not None:
      return codeword_to_blocks(bits)
    return codeword_bytes_to_blocks(crossbar_bytes(fields))

  def offsets_for(key_order):
    """Where each distinct key sits in the order the crossbar serves them,
    as the running sum of the WIDTHS (key_width) of the keys ahead of it.

    Only `== 0` is ever read out of this: charged() needs to know which key is
    FIRST in the ordering under test, because that is the one key the
    crowded-stage margin never falls on. The sum is over the keys' own widths,
    never their tables' block counts, which can run deeper than one block's
    worth of rows when a tree exceeds 512 entries or is sharded -- a table's
    extra depth stores more rows through the SAME key and does not move the
    next key along (finding 1.4, see key_width above)."""
    offsets, running = {}, 0
    for key, key_blocks in key_order:
      offsets[key] = running
      running += key_blocks
    return offsets

  def crowded(key_count, stage_bytes):
    """Whether a stage is CROWDED: two or more different keys filling more
    than TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE bytes (target.py has
    the measurements). Ternary pool only -- the one caller passing
    key_field_bits."""
    return (key_field_bits is not None and key_count > 1
            and stage_bytes > TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE)

  def charged(offsets, blocks, fields, bits, is_crowded=False):
    """One shard's TCAM blocks IN THIS STAGE: its declared count plus the
    crowded-stage margin.

    `blocks` already carries the key's standalone price -- what a key alone in
    its stage, or first in the order the crossbar serves, actually costs.
    Sharing adds one thing, and only in a CROWDED stage (see crowded(): two
    different keys filling more than 58 of the stage's 64 crossbar bytes). The
    later key must take the crossbar groups the first left; its bytes must
    also land on positions matching their PHV container lane (byte k of a
    32-bit container only on a position congruent to k mod 4,
    input_xbar.cpp's is_better_group), and p4c's greedy allocator, short of
    room, parks bytes on midbyte nibbles -- one more half-byte slot, which a
    key with none spare pays for with a whole TCAM block (spec Sec 12.1, read
    out of prog.bfa). Rule: +1 to every table of a non-first key in a crowded
    stage, whatever the key's own slack.

    RETIRED 2026-09-25: the per-key saturation margin (+1 to a non-first key
    whose standalone price leaves no spare slot, crossbar_capacity(g) == B,
    in ANY shared stage). Every observation on disk says saturation is not
    the driver, crowding is: independent_high_sd7/sd8's saturated keys shared
    stages of 22 and 35 bytes free (the rule over-charged them +1/+3); the
    saturated 16-byte probe paid nothing even at 59-64 bytes
    (results/tcam_mixed_key_cap_sweep.csv); and the one saturated key that
    did pay -- (179, 204) beside a 12-byte key, 9 blocks alone and 10 shared
    (scripts/tcam_stretch_sweep.py) -- sat in a 61-byte, crowded stage, as
    does independent_low_sd9's refusal. Scored on every source, crowding
    alone is at least as exact on every gate and 0-under everywhere.

    "Not first" is `offsets[fields] != 0`; key_width is >= 1 for every key
    that can reach this branch, so offset 0 names exactly one key per
    ordering. Which key that is, this packer cannot know -- fits() and
    stage_charged_blocks() take the worst ordering instead.

    bits is None for callers that name no field widths (the range pool),
    whose keys are whole-byte fields with slack to spare; the margin is inert
    there by construction, so they short-circuit.
    """
    if bits is None:
      return blocks
    return blocks + (1 if offsets[fields] != 0 and is_crowded else 0)

  def fits(stage, blocks, fields, bits):
    # stage[2] is the per-shard (blocks, key, field-bits) already here, stage[3]
    # the distinct keys with their own widths. A second distinct key makes one
    # of the two pay the margin, so the whole stage is re-priced against the
    # key set it would HAVE -- a table already placed can become more
    # expensive.
    #
    # Every ORDER of those keys is tried and the stage fits if ANY of them
    # packs (reviews/final_model_check_2026-09-27.md section 1b). Which key
    # p4c's greedy allocator serves first is its business, not this packer's.
    # Probe ragged_ax1_bx4 -- one 49-byte app key (9 blocks) beside four
    # 12-byte ddos keys (3 blocks each) -- compiles into ONE stage at 22
    # blocks (results/tcam_stretch_sweep.csv): the order that serves the four
    # ddos keys first (12 blocks, no margin) then the app key (9 + 1 margin =
    # 10) fits in 24 blocks, even though the other order (app key first, ddos
    # keys paying the margin: 9 + 4x4 = 25) does not. Requiring EVERY order to
    # fit rejected this real, single-stage design; requiring only SOME order
    # reproduces it. Fitting on the cheapest order does not risk
    # under-counting a design that genuinely needs 2 stages: probe
    # ragged_ax1_bx5 (the same app key beside FIVE ddos keys) really does need
    # 2 stages, and both its orders overflow the 24-block budget regardless
    # (5x3 + 10 = 25; 9 + 5x4 = 29), so this rule still rejects it. stage_
    # charged_blocks() charges the worst order that still fits, matching this
    # function's feasibility test exactly.
    #
    # stage[4] is the stage's seed (seed_stages): another pool's shards, which
    # count against the table cap, the byte limit and the column packing but
    # not against the crowded-stage rules, whose byte count (stage_bytes) and
    # key count stay this pool's own.
    keys = list(stage[3])
    if all(key != fields for key, _ in keys):
      keys.append((fields, key_width(fields, bits)))
    seed = stage[4]
    stage_bytes = crossbar_bytes(stage[0] | fields)
    if (crossbar_bytes(stage[0] | fields | seed.fields)
        > TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE
        or stage[1] + seed.tables + 1 > TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE):
      return False
    # Two DIFFERENT keys past the mixed-key budget (target.py, spec "F5"): p4c
    # would route the later key's bytes through midbyte nibbles at up to +2
    # blocks a table, which no price here models, so the stage is refused
    # rather than under-charged. Pool-level gate: only the ternary pool passes
    # key_field_bits, and F5 was measured on ternary keys only.
    if (key_field_bits is not None and len(keys) > 1
        and stage_bytes > TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE):
      return False
    shards_here = stage[2] + [(blocks, fields, bits)]
    is_crowded = crowded(len(keys), stage_bytes)
    for key_order in itertools.permutations(keys):
      offsets = offsets_for(key_order)
      # The TCAM test is a column PACKING, not a running total against
      # TCAM_BLOCKS_PER_STAGE -- see fits_two_columns: three 8-block tables
      # sum to exactly 24 and still do not fit, four 6-block ones do.
      if fits_two_columns(list(seed.blocks)
                          + [charged(offsets, *shard, is_crowded=is_crowded)
                             for shard in shards_here]):
        return True
    return False

  def place(stage, blocks, fields, bits):
    stage[0] |= fields
    stage[1] += 1
    stage[2].append((blocks, fields, bits))
    if all(key != fields for key, _ in stage[3]):
      stage[3].append((fields, key_width(fields, bits)))

  def empty(seed):
    # entry: [fields_present, tables_used, shards, key_order, seed]
    return [set(), 0, [], [], seed]

  def opened(blocks, fields, bits, seed):
    stage = empty(seed)
    place(stage, blocks, fields, bits)
    return stage

  def stage_charged_blocks(stage):
    """The TCAM blocks ONE finished stage actually costs, per shard in
    stage[2]'s order (sum it for the stage's total), stage-sharing
    margin included -- the same question `fits()` already answers for
    placement, asked once more after the fact so the total can be reported.

    Tries every order of the stage's distinct keys, exactly as `fits()`
    does, and keeps the LARGEST total among the orders that actually FIT
    (fits_two_columns) -- `fits()` only requires SOME order to pack
    (reviews/final_model_check_2026-09-27.md section 1b), so an order that
    overflows the column budget is not a real placement p4c could choose and
    must not set the reported price. Not the order tables arrived in
    (place() records that, but the allocator does not follow it): this packer
    cannot know which of the FITTING orders p4c will serve first, and
    'largest fitting' is this module's standing rule for that uncertainty --
    consistent with never under-counting real hardware
    (crossbar_stages_needed's own FFD upper-bound rationale). It is also what
    the measurement shows, since p4c picked the more expensive of its two
    fitting-or-not orders there: probe ragged_ax1_bx4 compiles to 22 blocks --
    the ddos-keys-first order (12 + 9 + 1 margin) -- and not the 21 the same
    order's raw sum would give without the margin; the app-key-first order
    (25 blocks) does not fit at all and plays no part in the max.

    At least one order always fits: `fits()` already requires SOME order to
    pack before a shard joins an existing stage, and the only stage committed
    without a `fits` call -- a freshly `opened()` one with no seed -- holds a
    single key, which is therefore first in every ordering, pays no margin,
    and fits by construction (`_stage_shards` caps it at one column). A
    SEEDED stage is always entered through `fits()`, seed blocks included, so
    the same guarantee holds there. The seed's blocks take part in the column
    test but not in the returned list: they are the seeding pool's to report.
    Ties between fitting orders keep the first one found; they are equal in
    total, which is all any caller reads."""
    is_crowded = crowded(len(stage[3]), crossbar_bytes(stage[0]))
    worst = None
    for key_order in itertools.permutations(stage[3]):
      offsets = offsets_for(key_order)
      charged_blocks = [charged(offsets, blocks, fields, bits,
                                is_crowded=is_crowded)
                        for blocks, fields, bits in stage[2]]
      if (fits_two_columns(list(stage[4].blocks) + charged_blocks)
          and (worst is None or sum(charged_blocks) > sum(worst))):
        worst = charged_blocks
    return worst

  def finished(stages_by_index, table_stages):
    """The StagePlan for a finished placement: {index: stage} plus
    {table_idx: latest stage index any of its shards landed in}."""
    loads = []
    for index in sorted(stages_by_index):
      stage = stages_by_index[index]
      loads.append(StageLoad(index=index,
                             blocks=tuple(stage_charged_blocks(stage)),
                             fields=frozenset(stage[0]), tables=stage[1]))
    return StagePlan(occupied=len(loads),
                     depth=(loads[-1].index + 1) if loads else 0,
                     indices=frozenset(load.index for load in loads),
                     blocks=sum(sum(load.blocks) for load in loads),
                     table_stages=tuple(table_stages[idx]
                                        for idx in range(len(table_specs))),
                     stage_loads=tuple(loads))

  if readiness_levels is None:
    stages, table_stages = [], {}
    for blocks, _width, table_idx, fields, bits in sorted(shards, key=load,
                                                          reverse=True):
      for position, stage in enumerate(stages):
        if fits(stage, blocks, fields, bits):
          place(stage, blocks, fields, bits)
          break
      else:
        position = len(stages)
        stages.append(opened(blocks, fields, bits, seed_at(position)))
      table_stages[table_idx] = max(position,
                                    table_stages.get(table_idx, position))

    return finished(dict(enumerate(stages)), table_stages)

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
  # A stage this pool has not opened yet but a seed already partly fills is
  # tested through `fits` like any other shared stage; an unseeded empty stage
  # still opens unconditionally, exactly as before seed_stages existed.
  #
  # The result counts OCCUPIED stages, not the index span -- stages below the
  # lowest level hold register/hash work, not tables from this pool.
  by_index = {}  # index -> [fields_present, tables, shards, key_order, seed]
  table_stages = {}
  ordered = sorted(shards, key=lambda s: (readiness_levels[s[2]], -load(s)))
  for blocks, _width, table_idx, fields, bits in ordered:
    index = readiness_levels[table_idx]
    while (index in unavailable_stages or
           (index in by_index
            and not fits(by_index[index], blocks, fields, bits)) or
           (index not in by_index and index in seeds
            and not fits(empty(seeds[index]), blocks, fields, bits))):
      index += 1
    if index in by_index:
      place(by_index[index], blocks, fields, bits)
    else:
      by_index[index] = opened(blocks, fields, bits, seed_at(index))
    table_stages[table_idx] = max(index, table_stages.get(table_idx, index))

  return finished(by_index, table_stages)
