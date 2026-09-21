"""Where the tables land: TCAM column geometry and the ternary crossbar packer.

Depends on target.py, errors.py and tables.py's crossbar geometry only.
Deliberately knows nothing about features, registers or trees -- it is handed
(blocks, byte_width) specs plus readiness levels and returns a placement, which
is what lets the same packer serve both the range pool and the classification
pool. What it must ask tables.py is a key's own standalone price
(codeword_to_blocks) and the byte capacity that price buys
(crossbar_capacity): a table's block count is not a property of the table
alone -- a key that shares its stage with a DIFFERENT key can cost one block
more than the same key alone -- so the placement and the charge have to be
computed together (see crossbar_stages_needed's key_field_bits and the
stage-sharing margin in charged(), spec Sec 13.2 "Effect 4")."""
import itertools
import math
from dataclasses import dataclass

from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.tables import (codeword_bytes_to_blocks, codeword_to_blocks,
                                crossbar_capacity)
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
                         # stage-sharing margin (see crossbar_stages_needed's
                         # key_field_bits and charged()) adds a block to a key that shares
                         # its stage with a DIFFERENT key and has no spare half-byte slot
                         # of its own. That is a per-STAGE placement fact, not a per-table
                         # one. Measured to matter: the ragged 49-byte key (179, 204)
                         # costs 9 TCAM blocks alone in a stage and 10 when a 12-byte key
                         # shares it (scripts/tcam_stretch_sweep.py), and that difference
                         # is what keeps independent_low_sd9's app and ddos trees out of
                         # one stage.

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
  COUNT is not a property of the table alone: when two DIFFERENT keys share a
  stage, the second key's bytes must land on whatever crossbar positions the
  first one left free, p4c's allocator is greedy, and when it stumbles it
  takes one extra half-byte slot. A key that still has a spare slot absorbs
  that; a key whose standalone price has NO spare slot is left with nowhere to
  put the mandatory 2-bit --version-- field and pays a whole extra TCAM block
  for two bits. charged() below is that margin (spec Sec 13.2 "Effect 4") and
  carries the derivation; key_field_bits is what lets it ask
  tables.codeword_to_blocks for each key's standalone price and
  tables.crossbar_capacity whether that price leaves a byte slot spare.

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
  nibble-clean byte for the allocator to spend a half-byte slot on. The margin
  charges both, which is over on the solid control and right on the ragged one
  -- one-sided by design, and bounded at exactly +1 block per affected table
  (Sec 13.2). independent_low_sd5's three ddos trees, each costing 3 TCAMs
  where their 11-byte key's bytes alone buy 2, used to be cited here as a
  second measurement; that one is a per-TABLE fact now and is priced by
  tables.codeword_to_blocks, with no placement term involved.

  Passing key_field_bits=None -- the default -- charges every table its
  declared block count wherever it lands, i.e. exactly the pre-existing
  pricing, since the margin needs a key's field widths to know whether it is
  saturated. The range pool never passes it: a range table keys one whole-byte
  field with slack to spare, so the margin is inert there by construction.

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

  There is deliberately no per-stage crossbar BLOCK-supply term on top of
  those three. It was probed and not found where it was looked for: two solid
  keys of 34 and 30 crossbar bytes need 13 blocks' worth of crossbar supply in
  a stage that has 12 groups, sit exactly on the 64-byte cap, and p4c placed
  them in one stage anyway (scripts/tcam_group_cap_probe.py, point
  groups_13_bytes_64, with groups_12_bytes_59 as the control, probed
  2026-09-15). A residual near-cap effect does exist -- it shows up only when
  one key already holds 41+ of the stage's 64 bytes (spec F5) -- and spec Sec
  13.2 deliberately folds it into the same one-sided statement as the margin
  below rather than modelling it, with experiment E1 deferred.

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
    """This KEY's own standalone block price -- how far it moves the next key
    along offsets_for's running sum.

    Not the table's block count: a table two blocks DEEP stores more rows
    through the same key, and depth does not move the next key along. That is
    finding 1.4, pinned by tests/test_p4model_guards.py's
    test_a_deep_table_does_not_push_the_next_key_further_along_the_crossbar;
    only the key's own width may move the crossbar along.

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
    stage-sharing margin never falls on. The sum is over the keys' own widths,
    never their tables' block counts, which can run deeper than one block's
    worth of rows when a tree exceeds 512 entries or is sharded -- a table's
    extra depth stores more rows through the SAME key and does not move the
    next key along (finding 1.4, see key_width above)."""
    offsets, running = {}, 0
    for key, key_blocks in key_order:
      offsets[key] = running
      running += key_blocks
    return offsets

  def charged(offsets, blocks, fields, bits):
    """One shard's TCAM blocks IN THIS STAGE: its declared count plus the
    stage-sharing margin (spec Sec 13.2, "Effect 4").

    `blocks` already carries the key's standalone price -- what a key alone in
    its stage, or first in the order the crossbar serves, actually costs.
    Sharing adds exactly one thing. The second key's bytes must land on the
    crossbar positions the first key left free (byte k of a 32-bit PHV
    container may only sit on a position congruent to k mod 4,
    input_xbar.cpp's is_better_group comment), p4c's allocator is greedy, and
    when it stumbles it raises nibbles_needed by one and fills one more
    half-byte slot. A key with a spare slot absorbs that; a key with none has
    nowhere left for the mandatory 2-bit --version-- field and pays a whole
    extra TCAM block for two bits. Read out of prog.bfa rather than inferred:
    spec Sec 12.1, and the pack-format evidence in this function's own
    docstring.

    "No spare slot" is exactly saturation. g = codeword_to_blocks(bits) blocks
    supply crossbar_capacity(g) whole byte slots and the key needs B of them,
    so the key is saturated iff crossbar_capacity(g) == B. Measured: the
    ragged key (179, 204) is B = 49 at g = 9 with crossbar_capacity(9) = 49,
    and really does cost 9 alone (ragged_ax1_bx5) and 10 when a 12-byte key
    shares its stage (ragged_ax1_bx4, ragged_ax2_bx2 --
    scripts/tcam_stretch_sweep.py). The SOLID 49-byte control is saturated too
    but has no nibble-clean byte for the allocator to spend a slot on and
    costs 9 either way, so this rule is over on that synthetic control and
    never under -- which is the trade Sec 13.2 makes deliberately: the margin
    is one-sided and bounded at exactly +1 per affected table.

    Its measured cost on real designs, which Sec 13.2's prose understates
    ("on the 9 mixed archived stages no second key is saturated" -- two of
    them are): independent_high_sd7's stage 9 holds a 10-byte app key
    (crossbar_capacity(2) == 10) beside a 12-byte ddos key, and
    independent_high_sd8's holds a 19-byte app key beside three 16-byte ddos
    ones (crossbar_capacity(3) == 16). p4c charged neither anything extra, so
    usage.blocks reads 28 against tcam_real 27 and 41 against 38 --
    scripts/validation_table.py reports both as OVER. The margin is kept
    anyway because dropping it makes independent_low_sd9's stage_depth
    under-predict (10 against 11), and never under-predicting outranks
    exactness here. No spelling of the saturation test scored so far achieves
    both; see this task's report for the three that were tried.

    It is what keeps independent_low_sd9's app and ddos trees out of a shared
    stage: 2 app at 9 blocks + 2 ddos at 3 is exactly 24 and packs 9+3 | 9+3,
    so every capacity limit this model knows would let them share -- and p4c
    still refuses.

    "Not first" is `offsets[fields] != 0`; key_width is >= 1 for every key
    that can reach this branch, so offset 0 names exactly one key per
    ordering. Which key that is, this packer cannot know -- fits() and
    stage_charged_blocks() take the worst ordering instead.

    bits is None for callers that name no field widths (the range pool),
    whose keys are whole-byte fields with slack to spare; the margin is inert
    there by construction, so they short-circuit. An empty tuple would fall
    through to the general path and come out unsaturated anyway
    (crossbar_capacity(1) = 5 != 0 bytes), so the two spellings agree on it.
    """
    if bits is None:
      return blocks
    g = codeword_to_blocks(bits)
    saturated = crossbar_capacity(g) == crossbar_bytes(fields)
    return blocks + (1 if offsets[fields] != 0 and saturated else 0)

  def fits(stage, blocks, fields, bits):
    # stage[3] is the per-shard (blocks, key, field-bits) already here, stage[4]
    # the distinct keys with their own widths. A second distinct key makes one
    # of the two pay the margin, so the whole stage is re-priced against the
    # key set it would HAVE -- a table already placed can become more
    # expensive.
    #
    # Every ORDER of those keys is tried and the stage fits only if ALL of
    # them pack. Which key p4c's greedy allocator serves first is its business,
    # not this packer's, and it is measurably NOT the cheapest choice: in probe
    # ragged_ax1_bx4 the ragged 49-byte key is the one that pays, though
    # charging the four 12-byte tables instead would have been cheaper.
    # Fitting on the cheapest order would under-count -- ragged_ax1_bx5 really
    # needs 2 stages and packs into 1 if the wide key is allowed to be the free
    # one. Requiring every order keeps the estimator on the safe side of the
    # 12-stage gate, and costs nothing on real data: over the 19 calibration
    # rows this choice changes no block count and no stage depth. Distinct keys
    # per stage are one to three here, so this is a handful of permutations.
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
    """The TCAM blocks ONE finished stage actually costs, stage-sharing
    margin included -- the same question `fits()` already answers for
    placement, asked once more after the fact so the total can be reported.

    Tries every order of the stage's distinct keys, exactly as `fits()`
    does, and keeps the LARGEST total. Not the order tables arrived in
    (place() records that, but the allocator does not follow it): this packer
    cannot know which key p4c will serve first, and 'largest' is this
    module's standing rule for that uncertainty -- consistent with never
    under-counting real hardware (crossbar_stages_needed's own FFD
    upper-bound rationale). It is also what the measurement shows, since p4c
    picked the EXPENSIVE assignment there: probe ragged_ax1_bx4 compiles to
    22 blocks, not the 21 it would cost had the four 12-byte tables paid
    instead of the ragged 49-byte one.

    No feasibility filter is applied, and none is needed. `fits` already
    requires EVERY order to pack before a shard joins an existing stage, and
    the only stage committed without a `fits` call -- a freshly `opened()`
    one -- holds a single key, which is therefore first in every ordering and
    pays no margin; `_stage_shards` caps it at one column. The rule stands
    anyway: a stage this function is handed is real and costs its blocks, so
    it is counted rather than dropped."""
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
