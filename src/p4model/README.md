# `src/p4model` — a calibrated Tofino resource-usage model

## 1. What this is

A calibrated surrogate for p4c's Tofino backend: given a set of decision-tree models and an
encoding (which features are selected, which thresholds they split on, whether two tasks share a
codeword), predict TCAM block usage and pipeline stage depth **without invoking the compiler**.
It exists so an Optuna search loop can reject thousands of infeasible designs a second — a real
`p4c` compile of one design takes ~2 minutes wall clock, ~17 s of it compiler CPU
(`reviews/p4_tofino_reference.md` §1.4).

Worked example — packing three per-tree classification tables (8 TCAM blocks, 4 crossbar bytes
each) into pipeline stages:

```python
from src.p4model.packing import crossbar_stages_needed

# One (block_count, byte_width) pair per independent table. See assemble_usage
# below for the real entry point that also prices the range-feature pool,
# the register schedule, and the whole-design stage/block totals.
plan = crossbar_stages_needed([(8, 4)] * 3)

print(plan.occupied)  # 2 -- two 8-block tables share a stage (8 | 8 across the
                       #      stage's 2 TCAM columns); the third needs its own
print(plan.blocks)     # 24 -- no key_field_bits given, so this call never
                       #      enters the C5 lane simulation (that only prices
                       #      a stage holding two or more DISTINCT keys, §2);
                       #      the charge is the naive 8+8+8 sum (§5)
```

The real per-design entry point is `src/p4model/usage.py:assemble_usage`, which takes a
serialized pool of both models' table specs, register names and readiness levels and returns a
`ResourceUsage` (stages, blocks, stage depth, register depth/count, …) plus both pools'
`StagePlan`s. See its docstring, and `tests/test_resource_model_golden.py` for a full worked
example that builds a pool from real fitted forests.

## 2. What is modelled

- **TCAM blocks for ternary classification tables** (one per decision tree) — crossbar bytes per
  key field, not raw codeword bits, priced up the block ladder `5g + floor((g-1)/2) >= B`
  (`src/p4model/tables.py:codeword_fields_to_bytes`, `crossbar_capacity`,
  `codeword_to_blocks_headline`, `codeword_to_blocks`, `ternary_matching_resource_usage`).
  `codeword_to_blocks_headline` is the published `S = 0` form; `codeword_to_blocks` adds the
  measured single-field isolation credit (`tail_is_isolatable`) and is what production charges.
- **TCAM blocks for range-match feature-encoding tables** (one per selected feature) — sized the
  way p4c sizes them at *compile* time, from the declared interval count, not from the real
  physical row expansion (`src/p4model/ranges.py:compiler_range_rows`,
  `src/p4model/tables.py:range_matching_resource_usage`).
- **Pipeline stage placement** for both table pools under all three simultaneously-binding
  per-stage limits (TCAM blocks, independent-table count, crossbar key bytes), plus — SHIPPED
  2026-09-28 (C5) — an ordered crossbar-lane stage simulation for the one case a per-table price
  cannot settle alone: two or more DIFFERENT keys sharing a stage. The generator now pins a
  deterministic placement order (`@placement_priority`, `@pa_no_overlay`,
  `src/p4gen/build_p4_script.py`, unconditional on every generated design), the packer follows the
  identical order, a stage's first-placed key pays its ordinary `codeword_to_blocks` price and
  every later key pays a leftover price computed by `src/p4model/lanes.py`'s crossbar-lane
  simulation (not a flat +1 margin), and a stage totalling more than 62 combined crossbar bytes
  across distinct keys is refused outright as a safety net (`TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE`)
  rather than priced. The older fitted 58-byte margin and its "charge the worst key ordering"
  search (`offsets_for`/`key_width`/the old `fits()`/`charged()`) are deleted, not just superseded —
  `src/p4model/packing.py:crossbar_stages_needed`, `fits_two_columns`, `_stage_shards`,
  `_ordered_stage_simulation`, `_stage_key_prices`, and `src/p4model/lanes.py`. A single-key pool
  (every `joint` design) is not routed through the simulation at all and is placed exactly as
  before C5 — proven structurally identical by a property-based fuzz test, not just measured.
- **The stateful register schedule** — which stage each per-feature `Register<>` runs in, given
  the flow-hash prologue, the 4-stateful-ALU-per-stage width limit, and the per-gate-block
  placement-cursor floor (`src/p4model/registers.py:register_stage_schedule`,
  `feature_readiness_level`, `gated_block_interior_stages`).
- **Overall pipeline depth** — prologue (flow hash) + register schedule + range pool + ternary
  pool + vote epilogue, checked against the hard 12-stage ceiling
  (`src/p4model/usage.py:assemble_usage`, `src/p4model/target.py:TOFINO_PIPELINE_STAGES`). SHIPPED
  2026-09-28 (C1): under `disjoint`, a task's classification trees are ready one stage past the
  last range table labelled that task or `SHARED_TASK`, never waiting on the OTHER task's range
  tables — `src/p4model/usage.py:tree_readiness_levels`. Confirmed on 28 real `disjoint` compiles
  never to place a tree too early. Under `joint` every range table is `SHARED_TASK`, so this is an
  exact no-op there.

Full derivations, with compile-row citations and dates, live in
`reviews/p4_tofino_reference.md` §4 ("Resource cost model"), which was rewritten from this
code on 2026-09-15 and cites every function below by name. The per-mechanism investigation
records — including the three retracted rules — are its **Appendix B**.

## 3. What is NOT modelled

- **SRAM and map-RAM blocks — observed only, not predicted.** The ground truth does not isolate
  variables: `independent_high_sd8` and `independent_high_sd10` have identical `register_depth`
  and `register_count` (8 and 19) yet differ 71 vs 75 real SRAM blocks. `scripts/validation_table.py`
  prints these alongside the predicted columns, explicitly labelled `OBSERVED, NOT PREDICTED`.
- **Exact-match / SRAM classification tables.** `src/p4model/tables.py:exact_match_resource_usage`
  returns `sram_blocks = None` *deliberately* — the per-block capacity for a plain exact-match key
  table depends on the compiler's `LayoutOption`/"ways" search, which is undocumented as a
  closed-form constant, and at this project's real feature scale the design needs on the order of
  `1.3×10³⁴` entries, so no such program can compile and no ground truth can ever exist.
  **The infeasibility finding is itself the result** — see the function's own docstring.
- **PHV container write conflicts** ("Mechanism A" in `reviews/p4_tofino_reference.md`
  Appendix B) — fixed at the source in the *generator* (`@pa_solitary` pragmas on every
  `class_tree_*`/`code_*` field), not modelled here.
- **Action data, hash units, the parser.**
- **Stateful register *capacity*.** `register_depth`/`register_count` report how deep the
  schedule runs and how many distinct registers it needs, never whether they physically *fit* —
  see `ResourceUsage`'s own docstring caveat in `src/p4model/usage.py`.

## 4. Known under-count risk

Range-match tables appear to cost roughly **2× the ternary crossbar units per byte** that
classification tables do, at equal key width (measured on 16-bit range keys against ternary keys
of comparable width). This project's real generator never reaches the crossover — every range
table keys exactly one 16-bit feature-value field, and 8 such tables (the hard per-stage
table-count cap) sit far under the 64-byte-per-stage crossbar budget even at that 2× rate — but if
a future design's **range-pool byte budget**, rather than the 8-table cap, becomes the binding
constraint, `crossbar_stages_needed` likely **under-counts** range stages. The exact byte-width
crossover was never pinned down (would need a >64-bit combined-width, multi-field range sweep).
See the function's own docstring in `src/p4model/packing.py` and
`reviews/p4_tofino_reference.md` §4.3.

## 5. Validation basis

**Current primary gate (2026-09-28, C1 + C5).** A unified 43-design set, compiled **with** the C5
generator pragmas that pin ternary-table placement order (`@placement_priority`, `@pa_no_overlay`
— the same 43 designs the older v6/extra/margin_screen archives below used to compile separately,
now re-compiled once so the model's placement decision and p4c's agree by construction rather than
chance): `results/compiler_calibration_pinned/`, scored by `scripts/validation_table.py`.

| quantity | comparable rows | exact | under-predictions |
|---|---|---|---|
| `blocks` | 38 of 38 | **38/38** | 0 |
| `stage_depth` | 43 of 43 | **42/43** (1 under, known — see below) | 1 (accepted) |

A frozen held-out batch (C6, 2026-09-28) of 30 designs never used for fitting, drawn by a
pre-recorded seed from `results/campaign_backup_20260825`, generated with the C5 pragmas and
compiled/scored exactly once (`results/heldout_2026_09_28/`): `stage_depth` **28/30 exact** (1
under, 1 over), `blocks` **21/22 exact** of 22 designs with a committed allocation (0 under, 1
over; 8 of the 30 exceed the 12-stage ceiling). Per-table, both the classification pool (144/144)
and the range pool (443/443) are exact against p4c's committed placement — see the caveat at the
end of this section before citing either per-table figure as validating C5 specifically.

**Three accepted under-predictions exist in the current state:**

1. `independent_high_sd12` (pinned primary gate) — model 13, p4c 14. The single design where
   pinning the tree placement order itself cost p4c a stage it did not need unpinned; infeasible
   either way, since both exceed `TOFINO_PIPELINE_STAGES = 12`, so nothing in the search loop is
   misled by it.
2. `margin_independent_M150_k5_s12` (pre-pragma adversarial replay only, `results/tcam_margin_screen/`)
   — model 66, unpinned p4c 68 blocks; p4c happened to serve its keys in a costlier order there.
   The **pinned** compile of the identical design costs exactly the model's predicted 66 — the
   miss is an artifact of replaying an archive p4c never compiled deterministically, not a pricing
   error, and does not appear on the primary gate. Outside what C5's accuracy claim covers, since
   that claim is conditioned on the pragmas being present (now always true for any newly-generated
   design).
3. `heldout_independent_M150_k14_s13` (C6 held-out) — model 11, p4c 12. Confirmed PRE-EXISTING (not
   a C1/C5 regression) by re-running four historical commits, including one from before this
   branch's work began, against the identical frozen design: all four reproduce the same 1-stage
   gap. The mechanism is in the RANGE POOL's placement ORDER, not its price — p4c spills one range
   table to a later stage than the model's eager bin-packing does, and a classification tree then
   waits one stage longer than C1 predicted for it. Out of scope for this run, left open.

Older archives, replayed informationally against current code (pre-pragma, i.e. without the C5
placement pinning): `results/compiler_calibration_v6.csv` (19 fitted designs) is `stage_depth`
19/19, `blocks` 17/17 of 17 comparable; `results/compiler_calibration_extra/` (8 held-out designs)
is `stage_depth` 8/8, `blocks` 5/5; `results/tcam_margin_screen/` (16 adversarial designs) is
`stage_depth` 16/16, `blocks` 15/16 (the `margin_independent_M150_k5_s12` miss above). These
pre-pragma numbers are historical — the pinned 43-design set above, not this list, is the number to
quote.

**A second gate exists because a per-design total hides a per-table error.** `blocks` sat at a
clean-looking 17/17 through an entire earlier calibration study with a 24-key **per-table** error
underneath it, because design totals let a +1 on one table cancel a −1 on another.
`scripts/tcam_table_scoreboard.py` scores every individual table observation this project has
collected — **405 rows across 14 CSVs** — and is run alongside `validation_table.py`. Current
result: the production per-table price is exact on **100/100** archived classification tables and
**50/50** held-out ones (the published headline form over-predicts 8 of the archived ones by
exactly 1 block), and the charged price has **0 under-predictions** on every placement the packer
emits; 16 rows sit at stages the packer refuses (> 62 bytes) and are reported, not scored.

**The F5 gap, closed 2026-09-25 and unaffected by the 2026-09-28 C5 rewrite.** `dsp41`/`dsp42` in
`results/tcam_discount_scan.csv` (key `(84, 84)`, model 5 blocks, p4c 7, beside a 41-42-byte
spacer) sit at 63-64 combined bytes, a stage the packer refuses outright — the refusal introduced
2026-09-25 survives unchanged as C5's safety net (`TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62`).
What changed 2026-09-28 is everything BELOW that refusal threshold: the fitted 58-byte "+1 to the
non-first key" margin is **deleted**, replaced by pinning the placement order at generation time
(`@placement_priority`/`@pa_no_overlay`) and pricing a later key by simulating the actual crossbar
lanes it inherits (`src/p4model/lanes.py`), rather than charging a flat margin against not knowing
which key p4c serves first. See §1 and §2 above, and `reviews/p4_tofino_reference.md` §4.3 for the
full mechanism and the `lanes.py` derivation (Appendix A).

Beneath the headline numbers:

- **144 compiled classification tables** behind the byte-to-block ladder (Appendix B "Mechanism D"),
  whose fixed points `B` = 10, 16, 21, 27, 32, 38, 43, 49, 54, 60 pin `crossbar_capacity` exactly.
- **19 real compiles** behind `packing.fits_two_columns`'s 12×2 TCAM column geometry
  (`scripts/tcam_column_sweep.py`, Appendix B "Mechanism C").
- **34 compiles** behind `tables.tail_is_isolatable`'s isolation table
  (`scripts/tcam_phv_slice_sweep.py`: 12 pays / 12 free / 0 disagreements against the direct
  observable), and behind the sharing rule's 62-byte refusal threshold (originally derived for the
  now-deleted 58/62 fitted margin, and unchanged by its replacement — C5's lane simulation only
  changed what happens below 62 bytes) **9 compiles / 40 tables**
  (`scripts/tcam_stretch_sweep.py`: the ragged 49-byte key at 9 blocks alone, 10 beside a 12-byte
  key), **47 probe compiles** (`scripts/tcam_mixed_key_cap_sweep.py`) and **16 real campaign
  designs** (`scripts/tcam_margin_screen.py`).
- **32 configs** behind `ranges.range_entry_count`'s exact per-value range decomposition (a live
  `tofino_model` real-insertion study, `.superpowers/sdd/task-4c-report.md`), plus a separate
  34-config width/offset sweep (`task-4b`) that never contradicted it.

**Evidence state — three things here are deliberately inexact or not hardware-confirmed**, and
should not be quoted as if they were exact:

1. **The 62-byte refusal is a safety net, not a proof.** SHIPPED 2026-09-28 (C5): the old fitted
   58-byte "+1 to the non-first key" margin this item used to describe is deleted — below 62 bytes,
   a shared stage is now priced by simulating the actual crossbar lanes (`src/p4model/lanes.py`),
   not a flat margin, and that simulation is exact on every real key measured so far (§1/§2 above).
   What remains inexact is the refusal itself: it can reject a placement the lane simulation could
   still legally price, because p4c's own greedy allocator has been observed to give up on a
   placement the simulation can compute (`M150_k7_s11`: 22+41=63 bytes, p4c moved the later key a
   stage; no simulation of any kind reproduced that particular give-up). Exact prediction there
   would mean simulating p4c's crossbar allocator's failure modes too, which this model
   deliberately does not do.
2. **`packing._stage_shards`' wider-than-one-column path.** No calibration design has a table over
   12 blocks. The finding-1.5a rounding fix rests on one probe compile
   (`scripts/tcam_stage_shard_probe.py`: a 13-block table places as 12|1 in one stage, charged 13).
   It is the one cost-*lowering* change in this work.
3. **The range pool's crossbar byte budget** — see §4 above; unreachable for this generator, but the
   crossover was never pinned down.

A per-stage crossbar **group** cap *was* found, contrary to what this section used to say: the probe
read as "13 groups in a stage that has 12" was counting **blocks**, and its two tables share a group
— the assembly uses groups 0-11, exactly 12 of 12. `target.py` records
`TERNARY_CROSSBAR_GROUPS_PER_STAGE = 12` and `TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE = 6` for
provenance, but **nothing consumes them**: no group-budget term is modelled. The residual it would
price, the F5 gap, is handled by C5's lane simulation plus the 62-byte refusal instead.

**Row parity (Finding C) was investigated and found vacuous.** Rows `2i`/`2i+1` share a half-byte
selector, so an even-height block run may start only on an even row. The rule is real (confirmed
317/317 in the archive) but is **provably never paid** under this packer's free ordering — place the
even-height tables first and every prefix sum stays even — so `fits_two_columns` is right to model
column loads and ignore rows. No code change; the permanent regression test is
`tests/test_p4model_guards.py::test_row_parity_never_rejects_a_stage_that_fits_by_size`, and the
four-part evidence is in `reviews/p4_tofino_reference.md` Appendix B "Mechanism C".

**Two models only** — every number above is for exactly two concurrent tasks (App ID + DDoS).

## 6. Not validated

- Designs with more than two concurrent models (N>2).
- Program shapes other than this generator's own table topology (one ternary table per tree, one
  range table per feature, the specific register-gating structure `catalog.py` encodes).
- Key widths outside those actually swept by the calibration studies cited above.

## 7. How to recalibrate for your own work

The package is split by what varies:

- **Different chip** → `target.py` (TCAM geometry, crossbar byte budget, per-stage ALU count —
  everything read off `mau_spec.h`/compiler sizing logic rather than this project's own choices).
- **Different P4 program** → `program.py` (this generator's own constants: feature-value bit
  width, the flow-hash prologue depth, the vote epilogue, the register block emission order).
- **Different features** → `catalog.py` (the feature → register dependency catalog; ships with
  this project's own 18-feature example set, explicitly meant to be replaced).
- **Different table topology** (anything other than one ternary table per tree and one range
  table per feature) → you are outside what this model validates at all; nothing here checks for
  that case.

A `Target` dataclass parameterizing the chip (so `target.py`'s module-level constants could
become an argument rather than an import) is a documented extension point, **deliberately not
built** — no second chip has ever needed calibrating against, so building the abstraction now
would be speculative.

## 8. Design notes

- **The package imports stdlib only** (`math`, `dataclasses`, `itertools`, `collections` —
  no `sklearn`, `numpy`, `pandas`, and no CWD-relative file reads; enforced by
  `tests/test_p4model_guards.py`). This is not a style preference: `p4/deploy_table_entries.py`
  runs inside `bfshell`'s embedded Python at real-switch deploy time, which has no `sklearn` and
  cannot import a dependency stack that assumes one. `range_entry_count`
  (`src/p4model/ranges.py`) in particular is shared verbatim between the training-time model
  (via `src/p4gen/evaluation.py`) and that live deploy script, specifically so there is one
  canonical implementation rather than two copies that could drift apart.
- **`src/p4model/__init__.py` is no longer a zero-import shim.** It eagerly re-exports the whole
  package, so any import from `src.p4model.*` — including `p4/deploy_table_entries.py`'s
  `from src.p4model.ranges import range_entry_count` — transitively imports every module
  (`.catalog`, `.errors`, `.names`, `.packing`, `.program`, `.ranges`, `.registers`, `.tables`,
  `.target`, `.usage`) and their stdlib-only imports (`dataclasses`, `math`, `re`, `itertools`,
  `collections`). Still safe under bfshell's embedded Python, but it does mean that deploy path
  now needs Python >= 3.7 for `dataclasses`.
- **The default-action discount (`ternary_matching_resource_usage`'s `dropped_per_tree`) takes a
  count, not a policy.** Which leaves get folded into a table's `default_action` instead of being
  installed explicitly is an *encoding* decision (this project uses the Planter RF_EB majority-class
  rule) that belongs to the caller, not to the cost model. Keeping only the resulting count here is
  what lets this module price a table's cost without knowing anything about decision trees.

## 9. Changelog of definition changes

Anyone comparing against archived campaign CSVs from before these dates is comparing against a
different definition of `stage_depth`/`blocks`, not a fresh anomaly:

- **2026-09-28** — **C1** (per-task tree readiness under `disjoint`) and **C5** (the fitted 58-byte
  crowded-stage margin deleted, replaced by pinned generator placement order plus a
  crossbar-lane stage simulation, `src/p4model/lanes.py`) both shipped. C1: a `disjoint` task's
  trees wait only for their OWN task's range tables, not both tasks' — closes
  `independent_high_sd12`'s and `margin_independent_M250_k4_s15`'s false-late placement. C5: below
  the (unchanged) 62-byte refusal, a shared stage's later key is now priced by simulation rather
  than a flat +1 margin, and `build_p4_script.py` unconditionally emits `@placement_priority`/
  `@pa_no_overlay` so the model's placement order and p4c's agree by construction. Primary gate
  moved to a single 43-design set compiled WITH the pragmas
  (`results/compiler_calibration_pinned/`): `stage_depth` 42/43, `blocks` 38/38. A frozen 30-design
  held-out batch (**C6**) scored once: `stage_depth` 28/30, `blocks` 21/22, found one genuine
  pre-existing (not C1/C5-caused) range-pool placement-order gap,
  `heldout_independent_M150_k14_s13`. See §5 above and `reviews/p4_tofino_reference.md` §4.3/§4.6.
- **2026-09-25** — the **crowded-stage rule**: a stage holding two different ternary keys that fill
  more than 58 crossbar bytes charges every non-first table +1, and more than 62 is refused
  (`target.TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE`,
  `TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE`). Closes the F5 under-prediction. The per-key
  **saturation margin** of 2026-09-21 is **retired** in the same change: it is now the only sharing
  charge. Fitted `blocks` 15/17 → 16/17 (`independent_high_sd7`/`sd8` exact again;
  `independent_low_sd12` now +2) and `stage_depth` 18/19 → 17/19 (`independent_low_sd12` +1). New
  gates: 8 held-out compiles and 16 adversarial real designs in `scripts/validation_table.py`; 97 new
  per-table observations in the scoreboard. `disjoint` only.

- **2026-09-21** — the whole **offset / consecutive-group-run mechanism retired**. p4c's own
  assembly falsified its premises (a block pairs with any midbyte, not a fixed partner; groups need
  not be consecutive; groups are shared between tables), so `crossbar_groups_needed`,
  `_run_capacity`, `_full_midbytes`, `version_block_penalty` and `version_block_delta` are deleted
  and `codeword_to_blocks` takes no `start_group`. Replaced by (a) a per-table price that is the
  block ladder `5g + floor((g-1)/2) >= B` plus a single-field isolation credit
  (`crossbar_capacity`, `codeword_to_blocks_headline`, `tail_is_isolatable`), and (b) the Sec 13.2
  **stage-sharing margin** (retired 2026-09-25) in the packer's `charged()`, +1 block to a not-first, exactly-saturated
  key. `blocks` went **17/17 → 15/17** (0 under, 2 over — the safe direction, and the price of not
  under-predicting `independent_low_sd9`'s `stage_depth`); `stage_depth` unchanged at 18/19.
  Row parity was investigated in the same pass and found **vacuous** under free ordering — no code
  change. A new per-table gate, `scripts/tcam_table_scoreboard.py` (308 observations), now runs
  alongside `scripts/validation_table.py`, because design totals hid a per-table error once already.
- **2026-09-05** — `FLOW_HASH_LEVEL` corrected 1 → 3 (p4c spends a metadata-init stage plus a
  split `tbl_calc_flow_hash$precompute`/`tbl_calc_flow_hash` pair before the first register can
  run) and `VOTE_EPILOGUE_STAGES = 1` added to `stage_depth` (the vote tables always occupy one
  further stage past the last classification table). Pre-2026-09-05 `stage_depth` values are
  **not comparable** to current ones.
- **2026-09-15** — three block/stage repairs, all in the **over**-predicting (safe) direction except
  the third. (1) `version_block_penalty` now asks `crossbar_groups_needed` for the run the key really
  occupies at its offset, instead of the offset-0 group count — closing a hole where clause (a)
  excused a run without checking the key fitted inside it. (2) A key's crossbar offset is now the
  running sum of the keys' **widths** ahead of it, not their tables' block counts: extra table depth
  stores more rows through the same key and does not move the crossbar along. (3) `_stage_shards`
  fills column-sized pieces and leaves a remainder rather than splitting into *n* equal rounded-up
  pieces, which used to charge a 13-block table as 14. `blocks` and `stage_depth` totals on the
  19-row calibration are unchanged by all three (17/17 and 18/19).
- **2026-09-07** — the ragged-key/odd-offset block charge **retracted** and replaced by
  `tables.version_block_penalty`; block counts compose as `max(bit_arm, crossbar + penalty)`, and
  the packer charges `version_block_delta` rather than the penalty. `blocks` went 12/17 → 17/17
  exact. Archived CSVs from before this date carry the old, over-firing block numbers on
  `independent`/`disjoint` rows.
- **2026-09-06** — `register_sram_bits` removed. `register_width_bits`
  (`src/p4model/catalog.py`) has no reader left inside the model itself; it is kept only because
  `build_p4_script.py` and `tests/test_feature_registers.py` still call it directly. Archived CSVs
  from before this date may still carry a `register_sram_bits` column that current code no longer
  produces.
