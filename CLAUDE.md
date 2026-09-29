# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This repository implements multiple concurrent tree-based machine learning models (Random Forests) in P4 switches using feature sharing. The system trains Random Forest classifiers for two tasks:
- **Application Identification**: Classifying traffic flows into 3 application types (real-time apps like Skype, non-real-time apps like Dropbox, websites like Wikipedia)
- **DDoS Detection**: Detecting DDoS attacks vs benign traffic (binary classification)

The models are then compiled into P4 switch code that can perform real-time network traffic classification at wire speed.

## Architecture Overview

### Key Architectural Concepts

#### Feature Interval Encoding
- Decision tree thresholds are converted to integer values
- Features are discretized into intervals based on all split points across all trees
- Each interval is encoded as a binary codeword
- Codewords enable efficient TCAM-based matching in P4 switches

#### Multi-Model Feature Sharing
Two encoding approaches are implemented:
- **Joint Encoding**: Both models share the same feature intervals, reducing TCAM usage
- **Disjoint Encoding**: Models use independent feature intervals

The system measures resource usage in:
- **TCAM stages**: Pipeline stages in the switch
- **TCAM blocks**: Memory blocks (24 blocks per stage)
- **Codeword length**: Must stay under 512 bits (`MAX_CODEWORD_LENGTH`)

#### Model Training with Constraints
Training uses Optuna for hyperparameter optimization with custom objectives:
- Maximize classification accuracy
- Minimize TCAM resource usage (stages/blocks)
- Ensure codeword length constraints
- Penalties for constraint violations

### P4 Code Generation

Key constraints:
- A physical TCAM block holds `TERNARY_MATCHING_ENTRIES_PER_BLOCK = 512` rows x
  `TCAM_BLOCK_KEY_LENGTH = 44` key bits (`build_p4_script.py:16-17`), and both
  dimensions cost blocks.
- **The 44 key bits are charged as crossbar BYTES, not as raw bits.** One block is fed by one
  ternary crossbar group, and the crossbar allocates per key FIELD, byte-rounded. Every tree table
  keys one `meta.code_<feature>` field per feature, so its width is
  `tables.codeword_fields_to_bytes` = `B = sum(ceil(field_bits/8))` crossbar bytes, and that `B` is
  the only thing its block width depends on (`tables.codeword_to_blocks` — see "Ternary block cost"
  below). Practical consequence, unchanged:
  **shedding codeword bits is worth nothing unless it drops some one feature's own `ceil(w/8)`** —
  the old "44-bit band on the pooled codeword" intuition is the wrong lever, and
  `tables.codeword_bits_to_blocks` (the `ceil((bits+4)/44)` band, once `band_factor`) survives only
  as the empty-key floor.
- **Range tables are sized by p4c at COMPILE time from the declared interval count, not from the real
  row expansion** (`ranges.compiler_range_rows`): a quarter of the entries are priced at
  `min(8, 2*nibbles-1)` rows, the rest at 1, giving a per-block capacity of **206 intervals** at this
  project's 16-bit keys. `range_entry_count` answers a different question — whether the control plane's
  real entries fit INSIDE that fixed allocation — and now lives in
  `tables.range_deployment_overflow`. Do not compute blocks from it.
- A stage's 24 blocks are **12 rows x 2 columns**, and a table chains its blocks down ONE
  column, so which tables share a stage depends on their WIDTHS, not their total: three
  8-block tables sum to exactly 24 and need two stages; four 6-block tables (also 24) fit
  in one. `packing.fits_two_columns` is the test; a table needing more than a column
  does span both (measured at 14/16/24 blocks). Established over 19 real compiles by
  `scripts/tcam_column_sweep.py`. Affects stage count only, never a table's own block count.
  **`reviews/p4_tofino_reference.md` §4.3 explains this from first principles** (no P4
  background assumed); its Appendix B "Mechanism C" is the investigation record, including the
  per-width shortcut that once made the rule look refuted.
- A range key costs **more than one row**: `ranges.range_entry_count(lo, hi)`
  is a port of the real Tofino driver's `expand_range()` and gives the exact
  physical row count for installing `[lo, hi]`. Validated over 32 configs against
  the live `tofino_model`/`bf_switchd` control-plane path -- this project has no
  physical ASIC, so "hardware behaviour" here means the real driver and simulator,
  never silicon. (An older note here quoted "207 range entries per
  block" -- that was an averaged figure, superseded by the exact per-value
  model; do not use it.)
- Real usable capacity is 506-512 rows/block, not a flat 512: multi-row range
  entries need *contiguous* free rows (`pipe_mgr_tcam_find_next_free`), so
  insertion-order fragmentation can strand up to 6. Measured 2026-08-19 to have
  **zero effect at this project's scale** -- the largest range table observed is
  375 rows and every one fits in a single block either way -- so no safety
  margin is applied. Revisit only if tables ever approach 500 rows.
- **RETRACTED 2026-09-21 — "a key's crossbar run depends on where it STARTS."** There is no offset
  argument anywhere in the model any more. `tables.crossbar_groups_needed`, `_run_capacity`,
  `_full_midbytes`, `version_block_penalty` and `version_block_delta` are **deleted**, and
  `codeword_to_blocks` takes no `start_group`. Their shared premises were read out of p4c's own
  assembly (`prog.bfa`) and are false: a block pairs with **any** of a stage's 6 midbytes, not with a
  fixed pair partner; a key's groups need **not** be consecutive (measured runs `{0,1,3,4}`, `{0,3}`);
  and groups are **shared** between tables keying the same bytes. The old "the shifted arm is
  first-principles geometry awaiting measurement" caveat is moot rather than resolved — the arm is
  gone. The associated claim that `crossbar_groups_needed` is "monotone by construction, so it can
  only over-predict" was also false of the composed block count (negative `version_block_delta` was
  measured). The real effect that term was chasing is real and survives as a **stage-placement**
  margin, not a per-table price — see "Ternary block cost" below.
- **RETRACTED 2026-09-21 — "a per-stage crossbar GROUP cap was probed and NOT found; do not add
  one."** **FALSE on its own evidence.** `scripts/tcam_group_cap_probe.py`'s point
  `groups_13_bytes_64` was read as "7+6 = 13 groups in a stage that has 12"; 7 and 6 are **block**
  counts, the two tables **share group 5**, and the probe's assembly uses groups **0-11 — exactly
  12 of 12**, landing on the cap rather than past it. `target.py` now records
  `TERNARY_CROSSBAR_GROUPS_PER_STAGE = 12` and `TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE = 6` for
  provenance. **No group-BUDGET term consumes them**, and none should be added. The near-cap effect
  a budget would price (spec F5) is handled by the **ordered lane simulation** below (`lanes.py`, C5);
  no byte-total rule or group count exists.
- Also retracted with the above, from `scripts/tcam_spacer_sweep.py`'s header: **"p4c re-sorts
  ternary tables by key width, wider first."** Unsupported — over 39 compiles the probe landed at
  group 0 whether the spacer was wider or narrower. Drop the claim.

### Stage cost

`stage_depth` is **not** a bin-packing result. It is a placement: a 3-stage prologue before any
`RegisterAction` can run (metadata init, then the split `tbl_calc_flow_hash$precompute`/
`tbl_calc_flow_hash` pair — `program.FLOW_HASH_LEVEL = 3`), then the register schedule, then the
range pool, then the classification pool, then one vote epilogue (`VOTE_EPILOGUE_STAGES`). Three
constraints drive it and none is a capacity limit:

- A stage has **4 stateful ALUs** (`METER_ALUS_PER_STAGE`), and every emitted `RegisterAction` holds
  one for a whole stage — so 16–20 registers need ⌈n/4⌉ stages regardless of their chain depth.
- Registers are emitted **unconditional → `if (meta.fwd == 1)` → `if (meta.fwd == 0)`**, and p4c's
  placer has one work-list cursor, so each block is floored at the last stage the previous one used.
  A `bwd`-gated feature therefore prices one stage later than the symmetric `fwd`-gated one.
- A stage the cursor spends **fully inside** a gated block can hold no table from the outer sequence
  (`gated_block_interior_stages` → `crossbar_stages_needed(unavailable_stages=…)`). An empty stage
  in a committed placement is usually *forbidden*, not unfilled — **there is no range-pool fill
  limit; do not look for one** (two separate investigation passes lost time to this).

The ternary crossbar charges the **union of distinct key fields** in a stage, not the sum of
per-table key widths — every tree of one task keys the same `meta.code_<feature>` field, so it
never binds here; blocks do — and blocks bind by the **2x12 column geometry** above, not by a flat
total.

`packing._stage_shards` fills column-sized pieces and leaves a remainder instead of splitting into
*n* equal rounded-up pieces (finding 1.5a) — the old form charged a 13-block table as 14. Confirmed by
one compile (`scripts/tcam_stage_shard_probe.py`: a 13-block table places as 12|1 in one stage,
charged 13); no calibration design reaches that size, so it rests on that single point.

**SHIPPED 2026-09-28 — C1, per-task tree readiness under `disjoint`.** Before this, every
classification tree (both tasks) had to wait past the last stage of ANY range table, from EITHER
task — provably never too early (a tree can't key a field from a range table it doesn't read) but
sometimes needlessly late for `disjoint`, where an app tree only ever keys `meta.code_app_<feature>`
fields sourced from the app task's own range tables. `usage.py:tree_readiness_levels` fixes it: a
tree of task `t` is ready one stage past the last range table labelled `t` or `SHARED_TASK`, never
waiting on the other task's. Evidence: 28 real `disjoint` compiles never placed a tree before its
OWN task's last range table, but placed 8 of 168 trees before the OTHER task's
(`reviews/model_audit_scratch/per_task_variant.py`). This lets the two pools share a stage: the
classification pool's `crossbar_stages_needed` call is seeded with the range pool's
`StagePlan.stage_loads` (`evaluation._pool_inputs`'s `range_task`/`ternary_task` labels), and a
seeded range table always prices as the stage's FIRST key — ahead of any tree — so a tree sharing a
seeded stage is never priced below its own `codeword_to_blocks`. Under `joint`, every range table is
`SHARED_TASK`, so this is an exact no-op there (`tests/test_align_budget.py`'s `total_blocks`
check and the golden fixture's `joint` rows are byte-identical before/after). See
`reviews/p4_tofino_reference.md` §4.6.

### Ternary block cost

**Rewritten 2026-09-21** (spec `docs/superpowers/specs/2026-09-20-tcam-block-model-rewrite-design.md`
Sec 13). Read from p4c's own assembly rather than fitted: **a TCAM block is one crossbar group plus
one nibble, and the version field costs a whole block whenever no nibble is left over.** Three
sentences for the per-table price, one exact rule plus one one-sided sharing rule for placement.

*The per-table price* (`tables.codeword_to_blocks`, headline form
`tables.codeword_to_blocks_headline`):

1. A tree's key costs `B = sum_i ceil(w_i / 8)` crossbar bytes, one term per selected feature
   (`w_i` = interval count − 1).
2. A key of `B` bytes spans the smallest `g` blocks with `5g + floor((g-1)/2) >= B`
   (`tables.crossbar_capacity`): each block carries five bytes, each *pair* of blocks shares one
   more, and one half-byte per key is reserved for the compiler's version bits.
3. A table needs `ceil(leaves / 512) * g` blocks.

Ladder steps, which is what threshold alignment actually descends: `B` = 10, 16, 21, 27, 32, 38, 43,
49, 54, 60 → `g` = 2…11. Nothing about placement is visible to alignment, by design.

Production adds ONE measured refinement on top (`tables.tail_is_isolatable`, 34 compiles): a single
nibble-clean field whose 1-4 leftover bits p4c can split off as a free nibble saves one half-byte.
The credit is **capped at one field** — p4c's allocator guarantees at most one nibble-only midbyte
per table. Refinement on: exact on all 100 archived classification tables. Off: over by exactly +1
per tree on 8 of them. The paper states the headline; the refinement is an appendix note.

*Placement* adds exactly two things, and no offset term of any kind:

- **Row parity — exact, and vacuous here.** A block run of even height may start only on an even
  row; an odd-height run may start anywhere (`Memories::find_ternary_stretch`). Confirmed 317/317 in
  the archive — and **provably never paid under this packer's free ordering**, so
  `packing.fits_two_columns` is right to model column *loads* and ignore rows. Investigated and
  closed 2026-09-21; do not reopen it. Evidence in `reviews/p4_tofino_reference.md` Appendix B
  "Mechanism C, addendum".
- **Crowded stage — SHIPPED 2026-09-28 (C5): the fitted 58-byte margin is DELETED, replaced by an
  ordered crossbar-lane stage simulation.** `target.TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE`
  (58) no longer exists, and `packing.offsets_for`/`key_width`/`fits()`/`charged()` in their old
  form (the "charge the worst-case key ordering" apparatus, Finding D below) are **deleted**. Two
  things ship in their place:
  1. **Deterministic placement order, pinned at generation time, not guessed by the model.**
     `src/p4gen/build_p4_script.py` now emits `@placement_priority(2)` on every
     `get_classification_tree_ddos_*` table and `@placement_priority(1)` on every `..._app_*` one
     (`program.PLACEMENT_PRIORITY`, `build_p4_script.py:~1194`), plus `@pa_no_overlay` beside
     every `class_tree_*` field's `@pa_solitary` (`build_p4_script.py:~1489`; `code_*` fields now get their own
     `@pa_no_overlay` too, see the paragraph after this list) — **unconditionally, on every generated design**.
     p4c's own placer therefore serves stages in the order: highest `placement_priority` first,
     ties to "last listed in program order", among tables that are ready and fit. The model's
     `_ordered_stage_simulation` follows the identical rule, so which key is "first" in a shared
     stage is now known by construction, not searched.
  2. **Per-key pricing by lane simulation, not a flat byte margin.** A stage's first-placed key
     (by the pinned order) still pays its ordinary `codeword_to_blocks` × row-count price,
     unchanged from §"Ternary block cost" above — this is exactly why a `joint` pool (one key) is
     untouched by any of this. Every *later* key in a shared stage is priced by the new
     `src/p4model/lanes.py` module, which simulates the actual crossbar lane slots (`slot index
     mod 4`) the earlier key's layout left free, in three layers: `lanes.layout`/`relaxed_layout`
     predicts a key's PHV container layout from field bit widths alone (exact on 90.8% of 1130
     real fields; the relaxed fallback brings it to 148/148 against real compiled layouts);
     `lanes.standalone`/`price_with_supply` counts the cheapest legal block count into a free vs.
     partly-used crossbar (agrees with `codeword_to_blocks` on 74/74 real keys standalone; matches
     p4c on 133/133 real keys sharing a crowded stage); `lanes.stage_prices` composes both into a
     full per-stage simulation, which `packing.crossbar_stages_needed`'s `_stage_key_prices`
     mirrors (substituting the first key's charge for the plan-invariant reason above — the two
     prices agree on every real key measured). The 62-byte mixed-key refusal that used to sit on top was **RETIRED 2026-09-29** (spec
     `docs/superpowers/specs/2026-09-29-overlay-and-layout-pragmas-design.md`): its one real case,
     `M150_k7_s11`, was a PHV layout artifact. The generator now emits `@pa_no_overlay` on every
     `code_*` field and `@pa_container_size` on every tree-key `code_*` field, pinned to
     `lanes.key_layout`'s choice (`build_p4_script.code_field_container_sizes`); only the real 64-byte
     limit and the lanes decide a mixed stage. **The retirement is valid only while those pins are
     emitted.** There is no 58-byte threshold, no per-key +1, and no "worst ordering" search any
     more — the order is pinned, so there is exactly one ordering to price.
  **Why `@pa_no_overlay` on `code_*` (2026-09-29).** p4c lets two metadata fields with disjoint live
  ranges share a PHV container, and every such overlay adds a hidden placement edge (readers of the
  older field must be placed before the writer of the newer one) that the model cannot see; 125 range
  tables in 65 of 73 archived compiles were skipped at least once for it, and it is what produced the
  C6 miss `heldout_independent_M150_k14_s13` (via Tofino-1's backfill snapshot). Pinning OUR side of
  every overlay pair removes all of them (125 skipped tables -> 0). It changes PHV pressure, so the
  container split of each tree key is pinned too (`@pa_container_size`, from `lanes.key_layout`, the
  same function pricing uses). Cost: +4.7 PHV containers on average (max +13), 0 TCAM blocks.
  See `reviews/p4_tofino_reference.md` Appendix B "Mechanism H".
  A **single-key pool is not simulated at all**: every `joint` design, and any `disjoint` pool
  where no seeded stage (C1, above) is reachable, is placed by the pre-C5 declared-price packer,
  unchanged — proven structurally identical to the pre-C5 model by construction (not just
  measured), by a property-based fuzz test in `tests/test_p4model_guards.py`, independently
  re-confirmed by the final whole-branch review on 19,012 random joint pools with 0 differences.

**Accuracy — current gates (2026-09-29), superseding all earlier numbers quoted anywhere else in
this file's history.** The primary gate is `results/compiler_calibration_pragmas_2026_09_29/`: 73
designs (the 43 pinned + the 30 held-out designs, recompiled with the `code_*` pins so the program
p4c sees matches what the generator now emits). **Feasible designs (<= 12 stages, n=60):
`stage_depth` 60/60 exact, 0 under; all 73: 70/73; `blocks` 60/60 exact**; held-out feasible 22/22.
The older pinned archive (`results/compiler_calibration_pinned/`, compiled before the `code_*` pins) is
informational only and now scores 42 -> **41/43** (misses `independent_high_sd12` and
`margin_independent_M150_k7_s11`); the held-out archive `results/heldout_2026_09_28/` (also pre-pin)
was 28/30 stage_depth, 21/22 blocks. Per-table on that held-out draw, classification 144/144 and
range 443/443 exact (see `reviews/p4_tofino_reference.md` §4.6 for the caveat on what that draw does
and does not exercise).

**The remaining misses, all infeasible or archive artifacts — list them whenever this accuracy is
quoted, not just "0 under":**
1. `independent_high_sd12` — model 13 vs p4c 14, infeasible either way (both exceed
   `TOFINO_PIPELINE_STAGES = 12`).
2. `heldout_independent_M250_k14_s12` — model 15 vs p4c 16, infeasible; the extra stage is produced in
   p4c's `REDO_PHV2` retry rounds for a program that does not fit.
3. `margin_independent_M150_k7_s11` — on the pre-pin pinned archive AND the pre-pragma `tcam_margin_screen` adversarial archive (never on a compile with the `code_*` pins), model 11 vs p4c 12: a PHV
   layout artifact. Its compile with the `code_*` pins is exact.
Plus, unchanged: `margin_independent_M150_k5_s12`, pre-pragma adversarial replay ONLY, model 66 vs
p4c 68 blocks; its pinned compile costs exactly 66. And one OVER-prediction,
`heldout_independent_M150_k15_s16` (model 14, p4c 13), also infeasible.
`heldout_independent_M150_k14_s13` (the old C6 miss, model 11 vs 12) is **closed**: root-caused to PHV
overlay dependencies (Appendix B "Mechanism H"), exact with the pins.

Optuna risk, recorded: `train_model.objective` sets `stages_violation = stage_depth -
TOFINO_PIPELINE_STAGES` and `early_stopping.constraint_values` hands those magnitudes to Optuna's
constrained sampler, which ranks infeasible trials by total violation, so the one infeasible design
whose prediction dropped a stage relative to p4c (`heldout_independent_M250_k14_s12`, 16 -> 15) ranks
as slightly less infeasible; feasibility, the only thing selection reads, is unchanged.

For historical context only: the deleted 58-byte fitted margin, the per-key saturation margin it
replaced (retired 2026-09-25), and the any-order-fit search (Finding D, adopted 2026-09-27 then
superseded outright by C5's pinned order) are written up as closed investigation history in
`reviews/p4_tofino_reference.md` §4.3 and Appendix B "Mechanism G" — do not reintroduce any of
them; none of that machinery exists in the current code.

Still treat any future "exact N/N" claim sceptically, and check the right thing: **ask whether the
sample contains a `disjoint` stage holding two or more distinct keys** — the only shape C5's lane
pricing can affect. A sample with no such stage (e.g. an all-`joint` draw)
says nothing about C5 specifically, even if its totals look clean. Full record: `reviews/
p4_tofino_reference.md` §4 and Appendix A (the `lanes.py` derivation) and Appendix B (the mechanism
index the code's citations point at; §7 lists what is still open). The 62-byte refusal is gone, and
the range-placement-order gap is closed (Mechanism H).

## Legacy Code

`legacy/` holds superseded implementations kept for reference only. Nothing in it is imported by the active pipeline — do not extend or fix it.
