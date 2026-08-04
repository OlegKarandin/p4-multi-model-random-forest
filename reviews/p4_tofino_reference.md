# P4 / Tofino Reference: Environment, Toolchain, TNA Porting, and Resource Model

Consolidated reference for everything learned about targeting Intel Tofino (TNA architecture) with
this project's generated P4, and about the `open-p4studio` toolchain used to validate it. This
document describes **facts and working procedures**, not a task log — for the investigation history,
raw experiment data, and day-by-day narrative, see `t11_tofino_port_and_env.md` and
`t12_tcam_model_experiment_plan.md` in this same directory.

**Scope of everything below:** the compiler (`p4c`'s Tofino backend) and the `tofino_model` simulator
are used as a **resource-allocation oracle and functional simulator**. There is no physical Tofino
ASIC anywhere in this project. Nothing here is a latency or throughput measurement — the compiler
reports *stage/table/TCAM/SRAM allocation*, and `tofino_model` (rarely used so far) simulates packet
processing *correctness*, not timing.

---

## 1. Environment setup

### 1.1 Toolchain source and licensing

Since Intel discontinued Tofino, the SDE was open-sourced under **Apache-2.0** into the community
`p4lang` org. For simulation/compilation-only use (no hardware) there is no NDA, no SDE license, and
no hardware request required:

- `github.com/p4lang/open-p4studio` — build system, plus the **`tofino_model`** functional simulator
  (a binary blob, x86-64 only — its source is not open).
- `github.com/p4lang/p4c` — the Tofino backend is built into mainline `p4c` as a `--target`, not a
  separate compiler.

Still proprietary/unavailable: `tofino_model`'s own source, BSPs/SerDes drivers (hardware-only, not
needed here), the P4Insight GUI.

### 1.2 Building the toolchain

- **Platform: Linux x86-64 only** (the `tofino_model` binary is x86-64-only). A container is not
  required — **WSL2 Ubuntu-22.04 works fine as a native build/run environment** and is what this
  project actually used throughout (Docker was the originally planned isolation layer but was never
  necessary).
- Steps: `git submodule update --init --recursive`, then `./install.sh` or
  `./p4studio/p4studio profile apply ./p4studio/profiles/<profile>.yaml`.
- Resource budget: **~40–70 GB disk, ≥8 GB RAM (16 recommended), ~1–3 h build time.** Budget for at
  least one retry — a first build attempt can stall/fail for reasons not worth root-causing once a
  retry succeeds cleanly.
- Installed compiler ends up at `~/open-p4studio/install/bin/p4c`. Also present in the same `bin/`
  directory: `tofino_model`, `bf_switchd`, `bfshell` — everything needed for both static compilation
  and live functional/control-plane testing.

### 1.3 Compiling a P4 program for Tofino

```bash
~/open-p4studio/install/bin/p4c -b tofino -a tna -g --verbose 2 -o <outdir> <file>.p4
```

- `-b tofino -a tna` targets Tofino-1. **Tofino-2 is `-b tofino2 -a t2na`** (also valid:
  `tofino2a0-t2na`, `tofino2h-t2na`, `tofino2m-t2na`, `tofino2u-t2na` — check `p4c --help-targets`
  for the exact list on a given install). Both targets compiled the same project program cleanly with
  near-identical resource footprints (small deltas in Gateway/SRAM/Hash Bit tied to each target's
  different physical stage layout, no placement failures on either).
- **`-g --verbose 2` is required.** Without it, `pipe/logs/` is created but stays empty.
- Include paths: pass `-I resources` if the template/header files under `resources/` are referenced
  by relative path from the generated `.p4` file.
- There is no separate `bf-p4c` binary on this toolchain version — "the Tofino backend" is exactly
  `p4c -b tofino ...`, contrary to older documentation that names `bf-p4c` as a distinct tool.

### 1.4 Reading compiler output

A successful compile with `-g --verbose 2` produces `<outdir>/<prog>.tofino/` (or similar) containing:

| File | Contents |
|---|---|
| `<prog>.bfa` | Barefoot Assembly — human-readable table→stage mapping |
| `pipe/logs/table_summary.log` | Per-table min/max allowed stage ranges, `"Number of stages in table allocation: N"`, `"Number of tables allocated: N"`, and the **critical path length through the table dependency graph** — the true dependency-driven minimum stage count (can be lower than the actually-placed stage count, see §6) |
| `pipe/logs/table_placement_N.log` | `"Placement error(s):0 stages required:N"` — a nonzero placement-error count means the design does not fit |
| `pipe/logs/mau.resources.log` | Per-stage resource table with columns `Exact Match Input xbar`, `Ternary Match Input xbar`, `Hash Bit`, `Hash Dist Unit`, `Gateway`, `SRAM`, `Map RAM`, `TCAM`, `VLIW Instr`, `Meter ALU`, `Stats ALU`, `Stash`, `Exact/Tind Match Search/Result Bus`, `Action Data Bus Bytes`, `8/16/32-bit Action Slots`, `Logical TableID` — **contains both an absolute-count table and a percentage table, one after the other; a parser must anchor on `"Stage Number"` and stop after the first occurrence, or it will silently read percentages as if they were counts** |
| `pipe/logs/mau.characterize.log` | Per-table `Table Entries` as `used / capacity (headroom)` — the most direct source for TCAM/range block-boundary questions |
| `pipe/logs/resources.json` | Machine-readable aggregate (stages, per-table block usage, `used_by` naming the owning table per physical TCAM block) |
| `pipe/logs/phv_allocation_summary_0.log` | PHV container assignment — reveals when independent fields share one physical container (a source of false table dependencies, see §3.6) |
| `pipe/logs/table_dependency_summary.log` | Explicit dependency edges between tables, e.g. `D: OUTPUT ANTI_NEXT_TABLE_DATA` (write-after-write hazard) |
| `pipe/logs/phv.json`, `pipe/context.json`, `pipe/tofino.bin`, `bfrt.json` | Full compiled artifacts — also exactly what's needed to *install* the program (§1.5) |
| `pipe/logs/clot_allocation.log` | Tofino-2 only ("Compressed Local Own Tuple"), absent on Tofino-1 |

`*` markers in `table_summary.log` flag a table placed outside its allowed min/max-stage scope — i.e.
genuinely over budget, not just tightly packed.

A representative real compile (3 App trees + 1 DDoS tree, 4 shared features, M3-scale) took **~2
minutes wall clock**, of which only **~17 s was actual compiler CPU time** — the rest is filesystem
overhead crossing the WSL2/Windows boundary (see §1.6).

### 1.5 Live / functional testing (`tofino_model` + `bf_switchd` + `bfshell`)

The compiler's own output is already "install-ready" — no separate SDE build step is needed to run a
compiled program against the simulator:

1. Copy `bfrt.json`, `pipe/context.json`, `pipe/tofino.bin` to
   `~/open-p4studio/install/share/tofinopd/<program_name>/`.
2. Write a `.conf` manifest at
   `~/open-p4studio/install/share/p4/targets/tofino/<program_name>.conf` (copy the shape of an
   existing example such as `tna_exact_match.conf`, substituting the program name/paths).
3. `sudo veth_setup.sh` to bring up virtual ports.
4. Launch `tofino_model` and `bf_switchd -p <program_name>` as two coordinated background processes.
   `bf_switchd`'s device-ready check has been observed to stall on a `core_pll_ctrl0` PLL-lock
   simulation spinner for up to ~1 minute — this is environment timing, not a real failure; retry the
   readiness check (e.g. every 15 s) rather than treating an early timeout as fatal.
5. Drive the control plane with `bfshell -b <script>.py` (Python, via the installed `bfrt` client
   library) or interactively. **`bfshell` needs a real pty to show any output at all** — wrap it
   (`script -qec '...' /dev/null`) or output is silently swallowed even though the script runs
   correctly.

This path has been used successfully to: insert real range-match entries and observe
`[Not enough space]` failures at true physical capacity, install a full real M3-scale generated
program's tables/default-actions and read them back, and confirm control-plane insertion errors (see
§4.5, §6).

### 1.6 Known environment quirks

- **Cyrillic (or otherwise non-ASCII) path segments cause a compiler crash on larger programs.**
  Compiling a source tree that lives under a Windows path containing non-ASCII characters (accessed
  via WSL2's `/mnt/c/...`) failed identically across three independent invocation styles with a
  `cc1: fatal error: ... No such file or directory` where the offending path segment appears as a
  run of 3-digit octal byte values with no separators — a mojibake bug in one of `p4c`'s own
  sub-invocations (likely the C-preprocessor front-end) that only surfaces once the compiled program
  is large enough. Small spike programs compiled fine from the same path; a full multi-tree combined
  program did not. **Workaround: copy the `.p4` file plus its `resources/` include tree to a
  plain-ASCII WSL-native path (e.g. `~/some_dir`) before compiling anything non-trivial.**
- **Compiling directly from a Windows path under `/mnt/c/...` is slow** — dominated by 9P filesystem
  overhead crossing the WSL2/Windows boundary, not compiler CPU time (§1.4). Copying to a WSL-native
  path before compiling avoids this too.
- Regenerating a `.p4` file and then compiling it needs explicit re-verification that the file on
  disk is actually the freshly generated one — an unrelated process silently clobbering a
  just-generated file back to a stale version (from a leftover earlier run) has been observed. Always
  re-check file content/line-count immediately before compiling, not just trust an earlier print
  statement.
- Training pipelines with **unseeded random sampling** (e.g. class-balancing via
  `.sample()`/`np.random.choice` with no `random_state`) mean re-running the same generation script
  trains a structurally different tree every time, changing interval counts and codeword widths.
  Resource *footprint* (stage/table/TCAM/SRAM counts) has repeatedly been observed to stay stable
  across such re-draws for a fixed feature set and tree count/depth, but *exact* entry counts and
  codeword bit-widths will differ run to run. Pin a `random_state` before treating specific numeric
  results (not just qualitative shape) as reproducible.

---

## 2. What "porting to Tofino" actually means for this project

**It is not a mechanical backend swap.** The code generator (`build_p4_script.py`) only fills a
handful of marker points inside hand-written TNA template files under `resources/`
(`p4_template.p4`, `p4_headers.p4`, `p4_util.p4`, `action.p4`, `table.p4`,
`table_classification.p4`). Everything architecture-specific — the parser, the register
declarations and their per-packet update logic, the hash extern, the pipeline wrapper — lives in
those templates or is emitted directly by the generator; **most of the real porting work is template
and generator design, not a one-line backend flag change.**

The current generator (`build_p4_script.py` + `feature_registers.py`) emits a **complete TNA
program from a trained model and a selected feature set** — registers, feature-encoding tables,
per-tree classification tables, voting logic, and flow bookkeeping — and has been validated to
compile cleanly (0 errors) for single-task and combined dual-task configurations, on both Tofino-1
and Tofino-2. The sections below record the hardware/compiler rules this generator design has to
respect, and the resource-cost model that was reverse-engineered and validated against it.

---

## 3. v1model (BMv2) → TNA: rules, restrictions, and working patterns

### 3.1 Architecture-level changes

| Area | v1model | TNA |
|---|---|---|
| Include | `#include <v1model.p4>` | `#include <tna.p4>` (+ `core.p4`) |
| Pipeline wrapper | `V1Switch(...) main;` | `Pipeline(IngressParser, Ingress, IngressDeparser, EgressParser, Egress, EgressDeparser); Switch(pipe) main;` |
| Metadata | `standard_metadata` struct | TNA intrinsic-metadata structs (`ig_intr_md`, `ig_tm_md`, ...); field names differ (`ingress_global_timestamp` → `ig_intr_md.ingress_mac_tstamp`, `egress_spec` → `ig_tm_md.ucast_egress_port`, `ingress_port` → `ig_intr_md.ingress_port`) |
| Checksum controls | verify/compute-checksum controls | no TNA equivalent — delete them |
| Deparser | one deparser | split Ingress/Egress deparsers with TNA signatures |
| Registers | `register<T>` + `.read()`/`.write()`, unlimited sequential ops per packet | `Register<T,I>` + one `RegisterAction` per operation — see §3.2 |
| Hash | `hash(...)` extern with an algorithm enum | `Hash<T>(HashAlgorithm_t.CRC32).get({...})` — one `Hash<>` instance per field ordering, see §3.3 |

Exact intrinsic-metadata field names and struct layouts depend on the SDE/compiler version — verify
against the actual installed headers rather than assuming a fixed name across versions.

### 3.2 Registers — the central hardware constraint

A physical Tofino `Register` is bound to **one pipeline stage** and supports **at most one
`RegisterAction` execution per packet, period** — not "once per stage," once per packet, full stop
(two `RegisterAction`s that are both unconditionally reachable in the same packet's execution is
illegal even if they'd notionally land in different stages; two that are mutually exclusive via
`if`/`else`, so only one ever actually fires, are fine).

Concrete, compiler-enforced rules discovered by direct testing:

- **Hard cap: at most 4 `RegisterAction`s attached to a single `Register`.** A 5th `.execute()` site
  on the same register is a compile error (`"too many RegisterActions attached to the Register... The
  target architecture limits the number of RegisterActions attached to a single Register to 4."`).
  This is an architectural wall, not a resource tradeoff — a v1model design that legitimately touches
  one register 5+ times per packet (e.g. bulk-read-everything-at-the-end plus several scattered
  writes) **cannot be ported as-is**; it must be restructured to consolidate touches below the cap
  before anything else about the port matters.
- **No shift instruction in the stateful ALU.** Any expression like `value >> n` (even `>> 1`) inside
  a `RegisterAction` body fails to compile ("expression too complex" — the ALU instruction builder
  has no case for shift, only add/sub/bitwise/compare/div-mod). A running-mean/EWMA scheme built
  around a variable bit-shift has no direct TNA equivalent.
- **`MathUnit<T>`** is the working replacement for approximate scaled multiply/divide: a dedicated
  hardware LUT-based primitive whose result is just another operand fed into the instruction being
  built (so it does **not** count as a second register touch). Example pattern for a fixed-decay
  (α≈0.5) EWMA in exactly one touch:
  ```p4
  MathUnit<bit<16>>(MathOp_t.MUL, 1, 2) halve_unit;
  ...
  value = halve_unit.execute(value + current_sample);
  ```
  This computes `new_mean = (old_mean + current_sample) / 2` — a **deliberate redesign**, not a
  faithful port, of any original packet-count/power-of-2-gated EWMA scheme (different decay
  behaviour, approximate LUT division instead of exact arithmetic). Treat any such substitution as an
  accuracy-affecting change requiring its own re-validation, not a transparent optimization.
- **Legal `Register<T,I>` element types are exactly:** `bit<8>`, `int<8>`, `bit<16>`, `int<16>`,
  `bit<32>`, `int<32>`, `bit<1>`, `bit<64>`, or structs of one/two of those. **Any other width (e.g.
  `bit<19>`) is a hard compile error** ("Unsupported Register element type"), not a soft cost —
  choosing a feature-value bit-width that doesn't match one of these forces every backing register up
  to the next legal width (typically `bit<32>`). The measured cost of that forced widening
  (`bit<16>` vs. `bit<32>`, otherwise identical): **Action Data Bus Bytes double, and the value
  occupies two separate PHV containers instead of one** — real, but SRAM/Map RAM/TCAM block counts
  were observed unchanged at small (single-register) scale. This project settled on **16-bit** feature
  precision specifically because it is a legal native register width with no forced widening.
- **Combining several distinct `RegisterAction` results with nontrivial logic in one action** (e.g.
  `result = a ^ b ^ c` where `a`, `b`, `c` come from three separate register touches) can hit a
  different error: `"action spanning multiple stages... We currently support only single stage
  actions."` The fix is structural, not a workaround of the underlying limit: assign each touch's
  result to its own metadata field (adding extra match-table key fields if needed) rather than
  combining touches' results within one action body.

**Practical consolidation pattern** (validated, and now what the real generator implements): give
every register exactly one `RegisterAction`, execute it exactly once per packet, and carry its
returned value forward through metadata for every downstream use — never re-read the same register a
second time "for convenience." Where a v1model design legitimately needs the register's *pre-update*
value in one place and its *post-update* value in another, compute both from the single execute
call's return value and any locally-available operands, rather than adding a second touch.

### 3.3 Hash

`hash(...)` → `Hash<T>(HashAlgorithm_t.CRC32).get({...})`. **One `Hash<>` instance handles exactly one
field ordering** — computing both a forward-direction and a reverse-direction hash of the same 5-tuple
needs two separate `Hash<>` instances (and, in practice, two separate actions/tables), not one
instance called twice with different arguments. CRC configuration is not guaranteed bit-identical to
v1model's `hash()` extern — do not assume cross-target hash equivalence without checking.

### 3.4 Timestamps

TNA's `ig_intr_md.ingress_mac_tstamp` (and similar intrinsic timestamp fields) is **48-bit
nanoseconds**, not v1model's microsecond-granularity `standard_metadata.ingress_global_timestamp`.
Any inter-arrival-time (IAT) feature computed from timestamps needs an explicit rescale (e.g. `>> 10`
as a cheap ~1024x downshift, ~2.4% off a true µs conversion) — acceptable for a resource-oracle
deliverable, **not** for an accuracy claim without further validation.

### 3.5 Actions cannot branch on a shared parameter across logically-distinct outputs

A single shared, parameterized action of the shape `if (tree == i) { meta.class_tree_i = class; }`
(one action reused across all trees, branching on which field to write) is **rejected** by TNA's
action-analysis compiler pass, even for a trivially small (e.g. single-tree) case. **Fix: give each
logical branch its own dedicated, unconditional action** — e.g. one classify action per tree, each
unconditionally writing only its own output field, selected by which *table* fires rather than by an
in-action branch. This generalizes cleanly to any tree count once each tree already has its own
physical table (which per-tree classification tables do by construction).

### 3.6 Codeword / PHV layout matters for stage packing, independent of logic

Writing several **logically independent** feature-encoding tables' outputs into different bit-slices
of **one shared PHV metadata field** (e.g. one combined `bit<N> codeword`, each table setting its own
slice) creates a real compiler-visible write-after-write hazard (`OUTPUT ANTI_NEXT_TABLE_DATA` in
`table_dependency_summary.log`) between those tables, forcing them to serialize across stages even
though nothing about their actual logic depends on each other. **Splitting the shared field into
independent per-feature metadata fields** (each its own container, at the cost of some PHV padding)
removes the hazard and lets the compiler co-locate the tables in the same stage. Measured effect on a
3-feature/1-tree slice: **7 stages → 5 stages**, purely from this layout change, with zero change to
touch count or logic. **PHV layout is a first-class resource input on this target, not a backend
detail: the same allocator also decides TCAM *block* count for range-matched keys (§4.2's "PHV
container width" bullet), which is why the generator now pins those fields explicitly rather than
leaving them to the allocator.** This holds even in combination with other changes (e.g. maximum-legal register
touch counts) — it is an independent, additive win. Classification tables should correspondingly key
on **one separate ternary field per feature**, not one concatenated codeword field, for the same
reason (this is also what "Tier-3" / the per-feature-field template design in this project's
generator does).

### 3.7 Majority-vote / N-way branching logic compiles cheaply as an if-cascade

An unrolled Cartesian-product `if`-cascade over every class combination (e.g. 3 trees × 3 classes =
27 `if` blocks for a majority vote) was expected to be expensive but **compiles into only a couple of
extra pipeline stages** — the compiler packs many conditions into gateway hardware rather than one
stage per condition. (A table-based reformulation of the same logic is also possible and was found to
reduce Gateway resource usage further without changing stage count, if that resource matters more than
stages for a given design.)

### 3.8 Flow bookkeeping / bidirectional flow hashing

A **two-hash "test the other direction's slot, then test-and-set my own"** design (two separate
`Hash<>` instances/tables plus a short resolution sequence before `fwd`/flow-hash are known) is a
correct, validated, real-compile pattern for bidirectional flow tracking, but it costs several
pipeline stages sitting on the critical path ahead of every downstream register touch (measured: 4 of
10 ingress stages in one real combined-task program were pure flow-identification bookkeeping, not
feature computation). A **single symmetric/canonical hash** (computed over `{min(addr), max(addr),
protocol, min(port), max(port)}` so both directions of a flow hash identically) plus a stored
orientation bit is a cheaper alternative in principle — this project's generator now implements this
symmetric-hash design (validated: same real compiled program, 0 stage regression vs. the two-hash
predecessor, confirmed via direct inspection that the "does the other direction already exist"
resolution logic is gone).

### 3.9 `num_trees > 1` and shared codewords are both real, validated designs

- Any number of trees per task is supported once each tree gets its own dedicated classify action
  (§3.5) — validated up to at least 3 trees/3 classes in a real compile.
- Two tasks can either **share one codeword space** (union of both tasks' split thresholds; every
  classification table keys on the full union) or each get **its own, narrower codeword** covering
  only its own trees' features/thresholds (duplicating the feature-*encoding* tables per task, but
  **not** the underlying registers/feature-extraction pipeline, which stays fully shared either way).
  Both are real, compilable, measured designs. The shared-codeword design is simpler to generate and
  was found to cost only a modest amount of TCAM headroom in one bottleneck stage compared to the
  per-task alternative, at the model scale tested — see §4 for the concrete numbers and what
  determines when the per-task design would actually be worth its extra generator complexity.

---

## 4. Resource cost model

This section states the **cost model this project's `evaluation.py` implements today**, each formula
annotated with how it was validated. All of it targets the same physical 512-row TCAM block that both
range-match and ternary-match tables draw from.

### 4.1 TCAM — ternary (classification / decision tables)

```
blocks_per_tree = ceil(entries / 512) * ceil((codeword_bits + 4) / 44)
```

- **512** entries/block (`TERNARY_MATCHING_ENTRIES_PER_BLOCK`) — confirmed exact by a live sweep of
  declared entry counts from 1 to 2048; the boundary sits precisely at 512/513.
- **44** bits/row (`TCAM_BLOCK_KEY_LENGTH`) — confirmed exact once the +4 below is included; a naive
  `ceil(width/44)` alone under-counts by exactly one block at widths sitting just below a 44-bit
  multiple (confirmed at 41 and 88 bits, among 9 tested widths spanning 16–523 bits).
- **+4 bits is a flat, universal per-row overhead**, applying once per table entry regardless of how
  many separate match fields make up the key (not a per-field effect — an early hypothesis that a
  168-bit key split across 4 fields was paying a per-field penalty was tested and rejected; the +4
  appears identically whether the key is one 168-bit field or four fields summing to 168 bits). Likely
  valid/version bits inherent to how Tofino TCAM rows work; not otherwise documented.
- **Known unresolved edge case:** splitting one wide key into exactly **2** separate ternary match
  fields was measured to cost **one more physical block** than splitting the same total width into 1,
  3, or 4 fields (a genuine non-monotonic result — no per-field-overhead formula reproduces it). If a
  tree's classification table ever ends up keyed on exactly 2 selected features, this is an untuned
  blind spot in the model, not something the formula above accounts for.
- Validated end-to-end against the real compiler across real trained-model programs at multiple
  codeword widths (25, 145, 168 bits) and entry counts (10–70): **predicted block count matched the
  compiler's real allocation exactly, every case tested.**

### 4.2 TCAM — range (feature-encoding tables)

**What a P4 `size` attribute means for a range-match table matters and is easy to get backwards.**
Confirmed directly: `size` is a **logical** entry count (one declared range = one control-plane rule
the table can hold), *not* an already-physically-expanded row count. A naive per-interval formula like
`2 * floor(log2(value_span))` (charging expansion cost per interval on top of a flat divisor) **is
wrong and overcounts real usage by roughly 4×** — confirmed by comparing predicted vs. real compiled
block counts on real feature tables with 7–68 intervals each: every one of them fit in exactly 1
physical block regardless of interval count, where the naive formula predicted anywhere from 3 to 5.
**Caveat added later (see "PHV container width" below): "exactly 1 block" holds only when the key
field lands in a 16-bit PHV container. The same tables cost 2 blocks each when the allocator parks
their `bit<16>` key in a 32-bit container — interval count is still irrelevant, but container width
is not.**

The cost model this project's code now implements works interval-by-interval, using the **exact
per-value decomposition the real Tofino control-plane driver performs at insertion time** (traced from
`bf-drivers` source, function `expand_range()`; a faithful Python port lives in `evaluation.py` as
`range_entry_count`):

```python
def range_entry_count(lo, hi, nibble_widths=(4, 4, 4, 4)):
    """Exact port of expand_range() (bf-drivers/pipe_mgr_entry_format.c) --
    the real per-value range decomposition the control plane runs, not an estimate."""
    # greedy, largest-nibble-aligned-block-first decomposition; see evaluation.py for the full body
```

Then: for each feature, sum `range_entry_count(lo, hi)` over its trained model's real `(lo, hi)`
interval bounds, divide by 512 (the same physical block size ternary tables use — range and ternary
share the underlying TCAM block, differing only in rows-per-entry), and `ceil`.

**Why this, and not a simpler capacity-constant lookup:** the compiler itself, at *compile* time
(before it knows what values will actually be inserted), uses a different, distributional formula for
sizing a declared-but-not-yet-populated table — it assumes a fixed 25% of entries will need the
worst-case row cost and prices the rest at 1 row each (`min(8, 2*nibbles-1)` rows for the worst case,
at 16-bit key width = 4 nibbles → 7 rows worst case). That compile-time estimate reproduces the
compiler's own block-boundary behavior exactly for a table sized by declared count alone, and is
useful as an explanation of *why* the compiler behaves as it does — but `evaluation.py` runs *after*
training, already knows every real interval, and can cost each one exactly rather than guess a
distribution. **The exact per-value formula (`range_entry_count`, above) is what should be used
whenever the real intervals are known**, which is always true for this project's use case.

Key facts confirmed about real range-match behavior:

- **Aligned power-of-2 ranges always cost exactly 1 physical row**, regardless of width — an aligned
  power-of-2 range is always exactly one prefix.
- **Misaligned ranges cost more, following the nibble decomposition — not the old log2 formula's
  magnitude.** E.g. at width ≈500, the old formula predicted ~16 rows/entry; real cost averaged ~1.96.
  The *direction* of the old formula (misalignment costs more) was right; its *magnitude* was wrong by
  roughly an order of magnitude.
- **Real, physical fill-to-capacity is placement/order-dependent near the top of a block.** A block's
  nominal 512-row capacity is genuinely reachable, but only under a favorable insertion order (wide,
  multi-row entries packed first; narrow/single-row entries appended last). An unfavorable order
  (single-row entries inserted first, wide entries appended after) can leave several rows
  (empirically, up to ~6) unusable, because a multi-row range-expanded entry needs several *mutually
  contiguous* free rows, while a single-row entry can use any one free row regardless of contiguity.
  No fixed row reservation was found in the driver source that would explain this as a flat constant —
  it is a genuine packing/fragmentation effect, confirmed decisive by reordering the identical final
  entry set and observing a different real fill count (512 vs. 506). **There is no universally-correct
  flat safety margin** to subtract from 512; the margin needed depends on insertion order, which the
  cost model does not control. Treat block-boundary-adjacent predictions (within roughly the last 1%
  of a block) as having a small, order-dependent uncertainty rather than being exact — this only
  matters when a table's total physical-row usage lands very close to a 512-row multiple.
- **PHV container width of the key field decides blocks-per-entry, and nothing about the table does.**
  A `bit<16>` range key allocated into a **32-bit W container** costs **2 physical TCAM blocks**
  (`mau.characterize.log` reports `1 in 2 (88)` — one entry spanning two 44-bit words); the same key
  in a **16-bit H container** costs **1** (`1 in 1 (44)`). Established by three controlled sweeps over
  otherwise-identical range tables, each varying exactly one thing: declared `size` (11→256), action-
  data width (4→25 bits), and range key width (4→19 bits) **all had zero effect**; six byte-identical
  tables fed from one shared source expression all cost 1 block. On a real M2 program the split was
  visible directly in `phv_allocation_summary_0.log` — `flow_iat_max_val`→`W2` and
  `fwd_packet_length_max_val`→`W3` cost 2 blocks each, while `fwd_iat_max_val`→`H7` cost 1.
  **Fix: pin every range-key field with `@pa_container_size("ingress", "ig_md.<field>", 16)`.** On
  that same program pinning all four fields moved every field to an H container and took it from
  **14 TCAM blocks / 10 stages to 12 / 9**, 0 errors, where 12 is exactly what `evaluation.py`
  predicts. This is what makes the cost model's range term correct rather than accidentally correct;
  `build_p4_script.generate_P4_code` now emits one pragma per distinct raw value field. Note the
  compile-time *entry* accounting is unaffected either way — the compiler charged each declared range
  1 row against a 512-row capacity (`13 / 512`), applying no expansion multiplier at all.
- **A table's P4-declared `size` is enforced as a hard logical-entry cap independent of remaining
  physical capacity** — even an aligned, cheap-to-store set of ranges cannot exceed `size` entries,
  regardless of how much physical block headroom remains.
- Measured width→capacity table for a *table sized purely by declared logical count, no real values
  known* (i.e. the compiler's own compile-time estimate, not the exact per-value model): capacities of
  **512 (4-bit), 342 (8-bit), 256 (12-bit), 206 (16-bit), 187 (19-bit)**, with **20-bit range keys
  failing to compile outright** (a hard SDE ceiling). These are useful as sanity bounds but are *not*
  what `evaluation.py` should compute when real intervals are known (use `range_entry_count` instead).

### 4.3 Ternary Match Input crossbar — a separate, sometimes-binding stage-packing constraint

For a design with several **structurally independent** ternary or range-match tables (each on its own
key field, no shared match input) — which is close to how this project's per-tree classification
tables and per-feature range tables are actually built — table placement can be limited by a resource
distinct from either TCAM block count or the classic "24 TCAM blocks per stage" figure: the physical
**Ternary Match Input crossbar**. Two independent, measured limits apply per stage, whichever binds
first for a given table's key width:

```
bytes_per_table       = ceil(codeword_bits_for_this_table / 8)
tables_per_stage_cap  = min(8, 64 // bytes_per_table)
```

- **A hard cap of 8 independent tables per stage**, confirmed identically at every key width tested
  from 8 to 512 bits (i.e. narrow-key tables hit this cap first, since their byte cost never comes
  close to 64 bytes/stage).
- **A 64-byte-per-stage crossbar budget**, confirmed as the binding constraint once per-table key
  width grows past ~64 bits (8 bytes/table), including two exact 64-byte saturations (2 tables × 32
  bytes; 1 table × 64 bytes) — strong evidence this is a real hardware constant, not a coincidental
  fit.

This is codified in `evaluation.py` as `crossbar_stages_needed`, which packs every independent table
(one entry per tree for classification tables, one entry per feature for range tables) into stages
respecting **all three** per-stage limits simultaneously — TCAM blocks (≤24/stage), table count
(≤8/stage), and key bytes (≤64/stage) — using first-fit-decreasing bin-packing, because the three
constraints are not separable (a table set can individually satisfy each relaxed bound while still
needing an extra stage once all three apply jointly). This crossbar model has been directly measured
only for **ternary** tables; applying it to range-match tables as well is a conservative extrapolation
by analogy (it can only ever raise the computed stage count, never lower it), not a separately
confirmed finding — range tables may have their own, still-unmeasured per-stage limit distinct from
this one.

**Why this matters beyond being "interesting":** a design can have plenty of spare TCAM blocks in a
stage (e.g. only 16 of 24 used) and still be forced into an additional stage purely by the crossbar —
a cost model that only reasons about the 24-blocks-per-stage figure will under-count stages for
multi-table designs.

### 4.4 Table `size` and control-plane runtime footprint

This project's tables are populated **at runtime via the control plane**, not via `const entries`
baked into the `.p4` source. A consequence worth remembering when reading compiler output: a
compile-time-only resource report (`resources.json`, `mau.characterize.log`, etc.) for a table that
never has entries installed during that particular compile **constant-folds to a trivial physical
allocation regardless of the table's declared logical `size`** — e.g. a table declared with `size = 31`
(31 real intervals) reported exactly 1 physical TCAM row in a static, no-entries-installed compile,
which is not representative of the real physical row cost once actual values are inserted (§4.2 shows
some single ranges genuinely need up to 4 physical rows). **A static post-compile log read cannot
answer "how much physical TCAM does this table really use once populated" — only a live
`tofino_model` + `bf_switchd` + `bfshell` insertion test (§1.5) can**, and this project has confirmed
that methodology works end-to-end (real entries installed and read back correctly, including
default-action entries, against a real compiled multi-tree/multi-task program).

Correspondingly: **P4-declared `size` should be set from the real number of logical entries the
control plane will install** (`len(intervals)` for a range table, the real post-discount entry count
for a classification table) — an honest, entry-count-derived declaration is correct and necessary
regardless of whether it changes the compiled physical footprint at a given scale.

**An over-declared `size` is not harmlessly conservative when the key space is smaller than it.** For
an exact-match table the compiler caps `size` at the key's own cardinality and warns:

```
warning: Shrinking table SwitchIngress.vote_ddos: with 1 match bits, can only have 2 entries
```

This bit this project's `generate_voting_code`, which applied a `max(32, num_classes ** num_trees)`
floor. The real 1-tree/2-class DDoS vote table keys on a single `bit<1>` field — 2 entries is the
entire key space, and the table emits exactly 2 `const entries`. Declaring the exact count removed
the warning (9 → 8 on a real compile). A "safety margin" on `size` is not free: state the real count.

### 4.5 Default-action discounting (Planter-style)

Dropping every classification-table entry whose leaf's class equals that tree's majority class, and
installing the majority class as the table's `default_action` instead (matching the same real,
verified mechanism used by the Planter RF-tree generator), is implemented and validated:

- **Real, large reduction in control-plane programming load**: measured 51–65% fewer explicit ternary
  entries at a real M3-scale model.
- **Does not by itself reduce compiled physical TCAM/SRAM/Map RAM/Gateway usage** at the model scales
  tested here — Tofino was observed to pack a 51-entry and a 27-entry classification table into the
  *same* 4 physical TCAM rows, i.e. the entry-count reduction from this discount didn't cross a
  physical block-packing granularity boundary at this scale. A real hardware-resource benefit from
  this discount, if any, would only show up at a larger model scale where entry counts are large
  enough to cross a physical boundary — plausible but not confirmed.
- Live-verified end-to-end: a real compiled, discount-enabled program was installed against
  `tofino-model`/`bf_switchd`, and every table's default action (and every explicit entry) was read
  back and matched the generator's own computed values exactly.
- A `const default_action = <action>(<literal>);` declaration in the P4 source (as opposed to setting
  the default action from the control plane at runtime) was found, separately, to compile but is
  **not** what the live control-plane path uses — this project's real deployment path sets the default
  action via the control plane, matching how all other table entries are installed, and this is the
  path that has been live-verified.

### 4.6 Data-dependency-driven stage placement (`readiness levels`)

The crossbar/block model of §4.3 is a **bin-packer**: it answers "how few stages could these tables
fit in", which is a lower bound. The real compiler also obeys **data dependencies** — a feature's
range table cannot be placed before the register chain producing its key value has run — and it
places **eagerly**, at the earliest legal stage rather than the latest. Both effects are real and
both are derivable from this project's own `FEATURE_REGISTER_CATALOG`:

```
level = 1                                   # flow hash; every register is flow-hash indexed
      + 1 if the feature is fwd-gated       # flow_orientation_action must resolve meta.fwd first
      + one per RegisterAction in its chain # a "dependency" register feeds a "value" register
```

A register shared between two features (`flow_last_arrival_time`, executed once for both
`flow_iat_max` and `flow_iat_mean`) is executed once but still sits on **both** features' critical
paths, so it counts for both.

Validated against a real compile of the M2 program (3 App trees + 1 DDoS tree, 4 features):

| feature | gated | chain | level | real stage |
|---|---|---|---|---|
| `flow_iat_max` | no | last_arrival → max | 3 | 5 |
| `flow_iat_mean` | no | last_arrival → mean | 3 | 5 |
| `fwd_packet_length_max` | fwd | max | 3 | 5 |
| `fwd_iat_max` | fwd | last_arrival → max | **4** | **6** |

The derived levels reproduce the observed stage offsets `0/0/0/1` exactly. The range pool really
occupies **2** stages, not the 1 the pure packer predicted, and the classification tables (which read
every feature's codeword, so they sit one level past the last range table) occupy 1 more — **3 match
-table stages, which is what the compiler does**. Note the extra stage is *not* a capacity effect:
that stage holds 4 blocks / 4 tables / 8 key bytes against caps of 24 / 8 / 64.

`evaluation.py` implements this as `feature_readiness_level` / `readiness_levels_for` plus an optional
`readiness_levels` argument to `crossbar_stages_needed`, which then switches from first-fit-decreasing
to **eager placement** (earliest stage at or after the table's level with room, spilling forward when
full) and counts *occupied* stages rather than index span. Eager placement is the point: the
theoretical optimum would drop every table into the single latest stage, and the compiler does not do
that. Passing no levels keeps the original packer behaviour unchanged.

**Caveat:** this models register-chain depth only. It does not model gateway/action dependencies,
PHV-sharing hazards (§3.6), or the compiler's own placement heuristics, so it remains an
approximation of a greedy allocator — just a much closer one than pure packing.

---

## 5. Quick-reference constants (this compiler version, 9.13.4)

| Constant | Value | Meaning |
|---|---|---|
| `TERNARY_MATCHING_ENTRIES_PER_BLOCK` | 512 | rows per physical TCAM block (ternary **and** range tables share this) |
| `TCAM_BLOCK_KEY_LENGTH` | 44 | usable key bits per physical TCAM row |
| ternary per-entry overhead | +4 bits | flat, once per entry, applied before dividing by 44 |
| `TCAM_BLOCKS_PER_STAGE` | 24 | physical TCAM blocks available per pipeline stage |
| `TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE` | 8 | independent match tables per stage, hard cap |
| `TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` | 64 | total match-key bytes per stage, hard cap |
| max `RegisterAction`s per `Register` | 4 | hard compiler-enforced ceiling |
| legal `Register<T,...>` element widths | 1, 8, 16, 32, 64 bits | any other width is a compile error |
| max range-match key width | 19 bits | 20-bit range keys fail to compile |
| TNA timestamp resolution | 48-bit ns | vs. v1model's µs-scale fields |
| this project's decided feature precision | 16-bit | legal native register width, no forced widening |

**Range-match physical block capacity by key bit-width** (compiler's own compile-time, unknown-values
estimate — not what `evaluation.py` uses when real intervals are known, see §4.2):

| Key width (bits) | Capacity (entries) |
|---|---|
| 4 | 512 |
| 8 | 342 |
| 12 | 256 |
| 16 | 206 |
| 19 | 187 |
| 20 | fails to compile |

---

## 6. Operational notes and gotchas

- **Critical path length ≠ observed stage count.** `table_summary.log`'s own "critical path length
  through the table dependency graph" is the true dependency-driven minimum; the compiler's actual
  placement can use more stages than this minimum due to its own greedy bin-packing choices (e.g.
  adding a `const default_action` with a constant parameter to several sibling tables at once was
  observed, in one real run, to push table placement from sharing one stage to spreading across four
  — with *no* change in each table's own resource footprint). Don't conflate "the design needs N
  stages" with "the compiler happened to place it in N stages this run" — the latter can regress from
  compiler placement heuristics alone. **Not every gap between a model's stage count and the
  compiler's is heuristic noise, though**: the one-stage gap originally seen on the M2 program turned
  out to be a real, derivable register-chain dependency, now modelled (§4.6). Check for a genuine
  dependency before writing a divergence off as placement luck.
- **A synthetic or untrained model can hide real structural issues.** Always validate resource claims
  against a real trained model, not synthetic/toy data — several real bugs and TNA restrictions in
  this project were only found once a real model's actual codeword widths and interval counts were
  compiled, not when a hand-constructed toy example was used.
- **Table placement `*` markers are the authoritative "did it actually fit" signal** — a design with
  spare-looking resource counts can still have a `*` marker meaning it was forced outside its intended
  stage range; absence of `*` markers across an entire `table_summary.log` is the correct thing to
  check before declaring a compile "fits comfortably."
- **A compile that succeeds with 0 errors is not evidence of functional correctness** — it confirms
  the design is legal and fits the declared resource budget, nothing about whether it produces correct
  classification output on real traffic. Functional validation requires either the BMv2/Mininet path
  (already exercised for the v1model version of this project's generator) or a `tofino_model` packet
  I/O test (not yet exercised for the TNA version).

---

## 7. Open / unverified items

- **`tofino_model` packet-level functional simulation has not been exercised for the TNA-targeted
  generator.** Live control-plane insertion (installing real table entries and reading them back) has
  been validated end-to-end; sending real packets through the simulated pipeline and checking
  classification output has not.
- **Tofino-2's own hardware ceilings** (stage count, per-stage TCAM/logical-table/gateway budgets —
  likely different from Tofino-1's 12-stage/24-TCAM-block figures used throughout this document) have
  not been looked up or compared against. Only same-program parity between the two targets has been
  confirmed, not where either target's own ceiling sits.
- **The range-match crossbar model is an unmeasured analogy**, not a separately confirmed finding —
  applied conservatively (§4.3), but range tables may have their own distinct per-stage limit.
- **Why a 32-bit PHV container doubles a range key's TCAM words is unexplained** (§4.2). The effect is
  measured, reproducible, and now avoided by pinning, but the underlying microarchitectural reason
  (how the match input crossbar feeds container bytes to the range-match units) was not chased down —
  the pragma makes it moot for this project rather than answering it.
- **The readiness-level model (§4.6) covers register-chain depth only.** Gateway/action dependencies
  and PHV-sharing hazards can also delay a table and are not modelled; the M2 program happens to have
  none that bind. A design where they do would diverge again.
- **The RM-3-style "2-field costs more than 3/4-field" TCAM anomaly** (§4.1) is real, measured, and
  unexplained — flagged as a blind spot, not resolved.
- **The range-match block-boundary fill margin is understood qualitatively as placement/order-dependent
  (§4.2) but not reduced to a usable formula** — no safety-margin constant is currently applied for
  designs whose real physical-row usage lands very close to a 512-row multiple.
