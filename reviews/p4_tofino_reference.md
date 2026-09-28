# P4 / Tofino Reference: Environment, Toolchain, TNA Porting, and Resource Model

Consolidated reference for everything learned about targeting Intel Tofino (TNA architecture) with
this project's generated P4, and about the `open-p4studio` toolchain used to validate it. This
document describes **facts and working procedures**, not a task log — for the investigation history,
raw experiment data, and day-by-day narrative, see `t11_tofino_port_and_env.md` and
`t12_tcam_model_experiment_plan.md` in `reviews/archive/`.

**Where to start.** §4 is the resource cost model — what a table costs and why — and it is the half of
this document that `src/p4model/` implements; it was rewritten from that code on **2026-09-15**, with
§4.1/§4.1.1/§4.3 rewritten again on **2026-09-21** (the TCAM block price) and **2026-09-28** (per-task
tree readiness under `disjoint`, and the fitted crowded-stage margin's replacement by a crossbar-lane
stage simulation), and cites the implementing function by name throughout. **Appendix A** derives the
crossbar-lane mechanism those two 2026-09-21/28 rewrites both rest on, from p4c's own source, with
worked examples. **Appendix B** is the mechanism index the code's citations point at, including the
rules that were tried and retracted. `src/p4model/README.md` is the
short version of §4 for someone who only wants to use the model.

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

- `-b tofino -a tna` targets Tofino-1. **Tofino-2 is `-b tofino2 -a t2na`.** Real
  `p4c --help-targets` output on this install (v1.2.5.10), confirmed by actual invocation, not just
  documentation:
  ```
  tofino2a0-t2na    tofino2a0-v1model
  tofino2h-t2na      tofino2h-v1model
  tofino2m-t2na      tofino2m-v1model
  tofino2u-t2na      tofino2u-v1model
  tofino2-t2na       tofino2-v1model    tofino2-psa    tofino2-default
  tofino-tna         tofino-v1model     tofino-psa     tofino-default
  ```
  No `tofino3` target exists on this install — `-b tofino3 -a t3na` fails immediately with
  `p4c: error: Unknown backend: tofino3-t3na` (a clean "unsupported", not silently accepted).
  Both targets compiled the same project program cleanly with near-identical resource footprints
  (small deltas in Gateway/SRAM/Hash Bit tied to each target's different physical stage layout, no
  placement failures on either). **Important correction from real compiles (see §7): bare
  `-b tofino2` resolves to the same device model as `-b tofino2u` (20-stage `JBayUDevice`), not
  `-b tofino2m` (12-stage `JBayMDevice`) as a source-reading-only analysis previously guessed** —
  confirmed by a byte-for-byte diff of `table_summary.log`/`mau.resources.log` between the bare and
  `tofino2u` compiles of the same program. `tofino2h` (6-stage `JBayHDevice`) is a real, separately
  useful target: it correctly *fails* compilation (`error: tofino2h supports up to 6 stages, using 9`)
  once a program's logical stage requirement exceeds its ceiling, rather than silently mis-placing.
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
  program did not. **Fixed at the source (2026-09-05): `p4_compile.compile_p4` now copies the `.p4`
  file plus the include path's `*.p4` files into a fresh ASCII-only WSL-native scratch directory
  (`mktemp -d`, defaults to `/tmp`) and compiles there itself, then copies the result back out to the
  caller's requested `output_dir`** — every caller gets this for free; nothing to remember by hand
  any more. Confirmed against the real toolchain end to end with `test_full_eighteen_feature_pool_compiles`
  (the exact "full multi-tree combined program" shape that used to crash).
- **Compiling directly from a Windows path under `/mnt/c/...` is slow** — dominated by 9P filesystem
  overhead crossing the WSL2/Windows boundary, not compiler CPU time (§1.4). The scratch-dir copy
  above (added for the Cyrillic-path fix) incidentally avoids this too, since p4c now always compiles
  from the WSL-native scratch dir, never `/mnt/c/...` directly.
- **`wsl <command>` without `-e` mangles any script containing `$VAR`/`;`.** Confirmed by direct
  experiment (2026-09-05): `wsl bash -lc 'X=$(echo hi); echo "X=[$X]"'` prints `X=[]`, while
  `wsl -e bash -lc` of the identical string prints `X=[hi]` — without `-e`, `wsl` re-joins its argv
  into one string and re-parses it through an extra implicit shell layer, silently dropping the
  boundary protecting the `-lc` argument. A single flat command with no `$`/`;` (this project's
  compile invocation before the scratch-dir fix above) survives that mangling by luck; a
  multi-statement script does not. `p4_compile.compile_p4` now always invokes `wsl -e bash -lc '...'`
  for this reason — any future direct `wsl bash -lc` invocation with a nontrivial script should do
  the same.
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

This section states the **cost model `src/p4model/` implements today** (rewritten 2026-09-15 from the
code, function by function; §4.1/§4.1.1/§4.3 rewritten again 2026-09-21 when the TCAM block price was
replaced, and §4.3/§4.6 rewritten again 2026-09-28 for per-task tree readiness under `disjoint` (C1)
and the crossbar-lane stage simulation that replaced the fitted crowded-stage margin (C5)), each
formula annotated with how it was validated and — where it was not — with what is
missing. All of it targets the same physical 512-row TCAM block that both range-match and ternary-match
tables draw from. The per-mechanism investigation records, including the three superseded version-block
rules, live in **Appendix B**.

### 4.1 TCAM — ternary (classification / decision tables)

One classification table per decision tree (`build_p4_script.py:636-659`). Its block count is the
product of a **depth** term (how many 512-row blocks its leaves need) and a **width** term (how many
blocks one table word spans):

```
blocks_per_tree = tree_entries_to_blocks(entries) * codeword_to_blocks(field_bit_widths)
```

Both terms live in `src/p4model/tables.py`:

```
tree_entries_to_blocks(n)       = ceil(n / 512)                        # TERNARY_MATCHING_ENTRIES_PER_BLOCK

codeword_fields_to_bytes(iv)    = B = sum(ceil(w / 8) for w in field widths)   # crossbar BYTES
crossbar_capacity(g)            = 5g + floor((g - 1) / 2)                      # byte slots g blocks own

codeword_to_blocks_headline(ws) = min { g : crossbar_capacity(g) >= B }        # the published form
codeword_to_blocks(ws)          = the same ladder, with the Sec 2.3 isolation
                                  credit for ONE nibble-clean field (§4.1.1)
```

**There is no `start_group` argument, and no offset term anywhere in the model** — that whole
apparatus was retired on 2026-09-21; see §4.1.1.

Depth and width are independent and both cost. `ternary_matching_resource_usage` multiplies them per
tree; `entries_across_trees_to_blocks` is what one *step* of the width term is worth across a whole
forest (8–80 blocks across the golden fixture), which is the multiplier a training-side budget needs to
weigh a width step against a range step.

Four facts underneath:

- **512** entries/block (`TERNARY_MATCHING_ENTRIES_PER_BLOCK`) — confirmed exact by a live sweep of
  declared entry counts from 1 to 2048; the boundary sits precisely at 512/513.
- **44** bits/row (`TCAM_BLOCK_KEY_LENGTH`) — one crossbar group feeds one block and delivers 5 private
  bytes plus one nibble of a shared midbyte: 5×8 + 4 = 44 (§4.1.1).
- **+4 bits is the mandatory version/valid nibble**, once per table *entry* regardless of how many
  match fields make up the key. Confirmed at source level: `VERSION_BITS = 4`
  (`bf-p4c/mau/table_format.h:75`). In the current model it is not a separate term at all: the
  `floor((g-1)/2)` in `crossbar_capacity` is already one half-byte short of what `g` blocks
  physically own, which *is* the reservation for these bits (§4.1.1). (Reconciling note: elsewhere in
  this document and in `tables.py`'s own docstrings the same field is described as "the mandatory 2-bit
  `--version--` field" -- that is the semantic content, version plus valid; `VERSION_BITS = 4` is the
  whole nibble-sized *slot* p4c reserves to hold it, with the other 2 bits unused padding. The two
  numbers describe the field's meaning and its physical reservation respectively, not a disagreement.)
- **The crossbar allocates per key FIELD, and byte-rounds each one.** This is the term the old
  `ceil((bits + 4) / 44)` headline formula missed. `build_p4_script.py:630-635` emits one separate
  ternary key field per selected feature (`meta.code_<feature> : ternary`), each declared
  `bit<len(intervals) - 1>`. A table keying 15 such fields totalling 205 bits really presents
  **33 crossbar bytes = 264 bits** and needs 6 blocks, not 5. Rounding the *concatenation* would
  under-count.

> **Practical consequence for design.** Shedding codeword bits is worth nothing unless it drops some one
> feature's own `ceil(w / 8)`, or carries `B` past a ladder step. The ladder's steps are
> `B` = 10, 16, 21, 27, 32, 38, 43, 49, 54, 60 → `g` = 2…11. The old intuition — "get the pooled
> codeword under the next 44-bit band" — is the wrong lever; the pooled width no longer enters the
> price at all, and `codeword_bits_to_blocks` survives only as the empty-key floor (below).

**Evidence for the byte→block ladder.** Measured exact on **144 real compiled classification tables**
spanning three compile eras: the whole observed `key_bytes → blocks` ladder (4→1, 11→**3**, 16→3, 20→4,
26→5, 33→6, 37→7, 41→8, 49→9, 52→10, 60→11) is single-valued and lands on `codeword_to_blocks`
**with no separate version term at all** — which is the point of the 2026-09-21 rewrite. Under the old
model the map was only single-valued once a `version_block_penalty` was peeled off and added back
(`independent_low_sd5`'s three 11-byte ddos keys really cost 3 blocks, which `ceil(11 / 5.5) = 2` could
not produce); under `crossbar_capacity`, `capacity(2) = 10 < 11 <= 16 = capacity(3)` gives 3 directly.
The one row the *headline* form misses is 33→6, which it prices at 7 (`capacity(6) = 32`); the measured
isolation credit (§4.1.1) recovers it. Appendix B "Mechanism D".

The superseded `codeword_bytes_to_blocks` (`ceil(b * 8 / 44)`) is still in `tables.py`, but no longer
prices a classification table: its only remaining production reach is `packing.key_width`'s range-pool
branch, where a one-field whole-byte key makes the two rules agree anyway (the `scripts/tcam_*` probe
instruments also still use it to describe their own synthetic keys). The `+4/44`
`codeword_bits_to_blocks` band survives inside the cost model for exactly one reachable input, the
**empty key** — a
forest whose every tree is a single leaf, which this repo's own tests exercise. The ladder has no
answer there (`crossbar_capacity(0) = -1`, so `g = 0` can never satisfy `B = 0`), and a version field
still needs a physical block, so `codeword_to_blocks(())` returns `codeword_bits_to_blocks(0) = 1` as an
explicit early return rather than as an arm of a `max()`.

> **RETIRED 2026-09-21 — everything this section used to say about crossbar group RUNS and OFFSETS.**
> The retired text described a key as occupying a *consecutive run of `g` groups starting at an offset*,
> with midbyte `i` owned exclusively by the group pair `2i`/`2i+1`, and priced the version field from
> where that run started (`crossbar_groups_needed`, `_run_capacity`, `_full_midbytes`,
> `version_block_penalty`, `version_block_delta`, and a `codeword_to_blocks(widths, s)` composing them
> as `max(bit_arm, crossbar + penalty)`). Every one of those premises was read out of p4c's own
> assembly (`prog.bfa`) and found **false**: a block pairs with **any** of the stage's 6 byte groups, a
> key's groups need **not** be consecutive (measured runs `{0,1,3,4}`, `{0,3}`, `{5..11}`), and groups
> are **shared** between tables keying the same bytes. The functions are deleted; `codeword_to_blocks`
> takes no `start_group`.
>
> Two evidence-state caveats that used to live here are **moot rather than resolved**, and should not
> be repeated as open questions: *"the shifted-offset arm is first-principles geometry awaiting
> measurement"* (there is no arm), and *"`crossbar_groups_needed` is monotone by construction, so it can
> only over-predict"* (true of the group count, false of the composed block count — a negative
> `version_block_delta` was measured, and the whole delta is gone). The real effect the offset term was
> chasing is real, and survives as a **stage-placement margin**, not a per-table price: §4.3.

#### 4.1.1 Where the version field lives — the block ladder and the isolation credit

*(Rewritten 2026-09-21. The clause list `version_block_penalty` used to apply — clauses (a)–(d), a
group RUN, and a `start_group` — is **retired**; see the retirement box at the end of §4.1 for what was
falsified and why. The physical phenomenon it priced is unchanged and is §4.1.2.)*

**Crossbar geometry, corrected** (`bf-p4c/mau/tofino/input_xbar.h`, and p4c's own `prog.bfa` output):
a stage's ternary crossbar is **12 groups × 5 private bytes + 6 byte groups (midbytes) = 66 bytes**.
One TCAM block is fed by **one group plus at most one NIBBLE of one byte group**, and the byte group is
*chosen* — `match: - { group: 0, byte_group: 2, byte_config: 0 }`. So a block sees 5×8 + 4 = **44
bits**, but *which* half-byte it takes is an allocator decision, not a fixed wiring. The mandatory 2-bit
`--version--` field can live **only in a midbyte nibble**
(`reviews/github_issue_tcam_version_bit_packing.md` §1.3); p4c's crossbar sizing never reserves that
nibble, so a key whose own bytes consume every nibble its blocks reach falls through to
`TableFormat::ternary_version()`, which `push_back()`s **a whole extra TCAM block to hold two bits** —
a match line with no group at all: `- { byte_config: 3, dirtcam: 0x0 }`.

Three structural facts, each read directly off the assembly rather than inferred from block counts:

| fact | evidence |
|---|---|
| a block = one group (5 private bytes) + at most one nibble of a byte group | every `match:` line in every compile read |
| a block may pair with **any** byte group, not a fixed "pair partner" | `lane_a4`: group 0 with byte group 2 |
| a key's groups need **not** be consecutive, and groups are **shared** between tables keying the same bytes | measured runs `{0,1,3,4}`, `{0,3}`, `{5..11}`; `groups_13_bytes_64` has both tables on group 5 |

**The headline rule (`tables.crossbar_capacity`, `tables.codeword_to_blocks_headline`).** `g` blocks
supply `5g` private byte slots and `g` nibbles; two nibbles make one whole overflow byte, and one nibble
must be left for `--version--`. Netting that out gives a whole-byte capacity of

```
crossbar_capacity(g) = 5g + floor((g - 1) / 2)
blocks(B)            = min { g : crossbar_capacity(g) >= B }
```

with `B = sum_i ceil(w_i / 8)`. The ladder's fixed points — the widest key each `g` still holds — are
`B` = 10, 16, 21, 27, 32, 38, 43, 49, 54, 60 for `g` = 2…11, and all ten are pinned by real calibration
keys (`test_the_ladders_fixed_points_match_crossbar_capacity`). This is the whole published model: one
sentence, no offsets, no clause list.

**The refinement (`tables.tail_is_isolatable`, and the full `tables.codeword_to_blocks`).** The one
thing the ladder cannot see is whether p4c will split a field's 1–4 leftover bits off as a standalone
free nibble, or park a whole extra byte on a byte group to carry them. Production therefore runs the
ledger form

```
overflow(g) = max(0, B - 5g)
S_usable    = nibble-clean fields (1 <= bits % 8 <= 4) whose tail p4c can isolate
blocks      = min { g : 2*overflow(g) - min(overflow(g), S_usable, 1) + 1 <= g }
```

which is the same ladder with one half-byte of credit. Isolatability is a statement about which byte of
a 32-bit PHV container the tail lands in — `tail = bits mod 32`, `index = ceil(tail / 8) - 1`;
isolatable at index 0 or 1, at index 2 only for a field inside a single container, never at index 3.
Measured over 24 compiles (`scripts/tcam_phv_slice_sweep.py`) against the direct observable
`byte_group_holds_whole_byte`: **12 pays / 12 free / 0 disagreements.** (A "34 compiles" figure used
to be quoted here; `tcam_table_scoreboard.score_phv_slice_sweep`'s own docstring notes the CSV has
24 rows, matching this script's real size — 34 was a stale count copied from elsewhere. C2, below,
is a related but distinct scoreboard bug: it used to *score* this same 24-row probe as 12/24 instead
of 24/24 by merging the probe's two declared fields into one before charging the isolation credit;
fixed, the true figure has always been 24/24.)

The one clause in the formula above is **compiler behaviour, not block structure**: the credit is
**capped at one field** (`min(overflow, S_usable, 1)`). `IXBar::allocate_mid_bytes` / `free_mid_bytes`
guarantee at most one nibble-only midbyte per table by construction; a second credit is untested by any
corpus point and would under-predict 3.7% of random keys at `B ∈ {17, 28, 39, 50, 61}`.

A second piece of compiler behaviour, `overflow(g) <= ceil(g / 2)` (p4c's own sizing loop,
`IXBar::calculate_sizes`, `input_xbar.cpp:507-511`, which never plans more than `ceil(g/2)` midbytes for
`g` groups -- structure alone would allow up to 6), is deliberately **absent** from the formula rather
than merely undocumented: **it never binds as a separate test**. With the credit capped at 1, the ledger
clause `2*overflow(g) - credit + 1 <= g` already forces `overflow(g) <= floor(g/2) <= ceil(g/2)`,
confirmed by brute force over 200 000 random keys with 0 differences whether the clause is present or
absent. (An earlier version of this document claimed it binds on `joint_high_sd7`'s four 29-byte tables
-- that claim is false: at `B = 29, g = 5`, `overflow = 4`, the ledger clause alone already gives
`g = 6`, so dropping the sizing-loop clause changes nothing there either.)

**Which form ships where.** The headline (`S = 0`) form is what the paper states: exact on 92 of the 100
archived classification tables, over by exactly **+1 block per tree** on the other 8 (the two 33-byte,
15-feature keys), never under. The refined form is what the campaign runs: **exact on all 100**. The
refinement costs a paragraph about PHV container byte positions to explain, which is why it is an
appendix note rather than part of the model statement.

**The residual this leaves, stated honestly, and corrected (C3, 2026-09-28).** An older draft of this
section called the isolation outcome for 3+-field keys "not predictable ... the allocator's greedy
choice" — as if it were arbitrary. **That framing is wrong and is retracted.** The mechanism is not
mysterious: it is exactly the lane arithmetic **Appendix A** derives from p4c's own crossbar sizing
code, byte lane by byte lane, with two worked examples (`w019`/`w027`) that diverge on nothing but
which lane loses its tail. A lane checker built from a compile's REAL PHV layout reproduces p4c
**57/57** on the probe families where this section's width-only proxy (`tail_is_isolatable`,
`bits mod 32`) is only 48/57 (always conservative — errs toward NOT isolatable, never the other way),
and the two agree on **74/74** real design keys. The one thing that *is* genuinely unresolved is
narrower than "predictability" in general: for a key with **3 or more** candidate nibble-clean fields,
no width-only rule found so far predicts *which* field p4c's stateful greedy pass will pick to isolate
— `scripts/tcam_field_count_sweep.py` shows `(28, 32, 24)` pays and `(28, 40, 16)` frees, `(51, 24, 8)`
pays and `(51, 16, 16)` frees, with 44 targeted compiles finding no feature that separates them. That
narrow residual is accepted as a documented bias in the width-only proxy, not fitted further; it does
not mean the isolation credit itself is unpredictable — full first-principles resolution needs only the
real PHV layout, which Appendix A gives.

**MEASUREMENT.** The per-table gate is `scripts/tcam_table_scoreboard.py`, which scores every table
observation this project has collected — **405 rows across 14 result CSVs**, 50 of them from compiles
the model was never fitted on (`results/tcam_heldout_harvest.csv`) — because a design-level total lets
a +1 on one table cancel a −1 on another, and that is exactly how a 24-key per-table error survived an
entire calibration study behind a clean 17/17. Current standing (2026-09-28, re-run against this
document's live code):

| quantity | n | result |
|---|---|---|
| refined price, archived classification tables | 100 | **100 exact** |
| refined price, held-out classification tables | 50 | **50 exact** |
| headline price, archived classification tables | 100 | 92 exact, 8 over by 1 |
| charged price (refined + §4.3's lane-simulation placement price, C5), all 405 observations | 405 | 391 exact, 9 over, 5 under |
| charged price, only placements the packer actually emits (excludes the 16 refused rows below) | 389 | **0 under** |
| observations at a stage the packer refuses (two keys > 62 bytes, or no lane-legal fit) | 16 | reported, not scored — all 5 of the "under" rows above are inside this refused set |

**GATE (`blocks_charged`, 0 under-predictions on emitted placements): PASS.** The 16 refused rows
include the F5-gap pair `dsp41`/`dsp42` (below) and 14 crowded/wide probe keys from
`tcam_stretch_sweep`/`tcam_mixed_key_cap_sweep` at the 63-64-byte boundary or beyond the lane
simulation's own 12-group/6-midbyte capacity.

**The F5 gap — CLOSED 2026-09-25.** Until this date the charged price had 2 named under-predictions:
`dsp41` and `dsp42` in `results/tcam_discount_scan.csv`, key `(84, 84)`, `B` = 22, model **5** blocks,
p4c **7**, beside a spacer holding 41–42 of the stage's 64 crossbar bytes. A +1 margin could not
close a 2-block gap, and a first byte cap at 62 turned out to be too loose: a follow-up sweep of five
more probe shapes (`scripts/tcam_mixed_key_cap_sweep.py`, `results/tcam_mixed_key_cap_onset.csv`) and
16 real campaign designs compiled for the purpose (`scripts/tcam_margin_screen.py`) showed a +1 already
at 59–62 bytes, on keys that are **not** saturated. The 2026-09-25 fix was the **crowded-stage rule**:
above 58 combined distinct-key bytes every non-first table paid +1; above 62 the stage was refused.
`dsp41`/`dsp42` sat at a refused placement.

> **Superseded 2026-09-28 by C5, half of this closure survives unchanged.** The 58-byte fitted margin
> described in the paragraph above and the old table below is **retired** — §4.3 now prices a
> non-first key by simulating where the earlier key's blocks actually left free crossbar slots
> (`src/p4model/lanes.py`), not by a flat byte-count margin. But `dsp41`/`dsp42` were never *priced*
> by the margin in the first place — they were **refused**, and the > 62-byte refusal threshold
> (`TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62`) is the one piece of the old rule C5 keeps
> verbatim, as a safety net rather than a price (§4.3). So the F5 gap is still closed, by the same
> mechanism, under the new model.

> **The spec's publication sentence, now accurate as written.** Sec 13.2 reads *"Residual placement
> effects can cost a table one further block; the model charges this conservatively and was never
> observed to under-predict."* Before 2026-09-25 that was false against `dsp41`/`dsp42`. With the
> (now C5) sharing mechanism it holds on every source this project has: the 43-design pinned primary
> gate, the pre-pragma fitted/held-out/adversarial replays, and 389 scorable per-table observations
> (405 total, 16 refused). The price is over-prediction at a crowded stage p4c happens to share for
> free — never under-prediction.

**Joint designs are untouched by all of this**: every tree keys one shared field set, there is only ever
one distinct key in a stage, and no `joint` row in the calibration pays anything beyond its ladder
price. `packing.crossbar_stages_needed`'s docstring calls this out as a structural, not merely
empirical, invariant: a single-key pool is never routed through the stage simulation at all (§4.3).

**Sceptic's check, updated for C5.** Any future "exact N/N" claim should be checked against the right
question. Four earlier pieces of advice are all dead — "look for two different ragged keys sharing a
stage", "look for a ragged key at an odd offset", "look for a key that saturates", and (as of
2026-09-28) "look for a stage with more than 58 combined bytes shared by two keys" — that last one was
the right question for the retired fitted margin, not for the mechanism that replaced it. Ask instead
whether the sample contains a stage where two DIFFERENT keys share the crossbar at all: that is the
only shape §4.3's lane simulation prices differently from the plain per-table ladder, and a sample with
no shared-key stage says nothing about it.

#### 4.1.2 The physical anomaly this models: an entire block for 2 bits

*(The width-modulo predicate this section used to assert was retracted, and the later odd-offset rule
that replaced it was retracted too. Both retractions, with the data that killed each, are stated once
in Appendix B "Mechanism G". The version-block **clause list** that in turn replaced those has since
been retired as well (§4.1.1, 2026-09-21). Nothing in §4 asserts any of the three any more — but the
physical anomaly below is unaffected by all that churn. It is what every one of those rules was trying,
with varying success, to price.)*

The reproducible phenomenon is real and is what §4.1.1 prices. Three programs with **identical match
bits and identical crossbar bytes**:

| key | match bits | ternary xbar bytes | nibble-clean bytes | TCAM blocks | bits used / allocated |
|---|---|---|---|---|---|
| 4 × `bit<42>` | 168 | 22 | 1 | **4** | 172 / 176 (4 idle) |
| 2 × `bit<84>` | 168 | 22 | 0 | **5** | 172 / 220 (48 idle) |
| 2 × `bit<84>`, minus one unrelated line | 168 | 22 | 1 | **4** | 172 / 176 (4 idle) |

**That third row is the sharpest statement of the problem.** It is the identical program to row 2 with
`ig_tm_md.bypass_egress = 1w1;` deleted — a statement that does not mention the table, its key or its
actions. Deleting it frees bit `[0]` of container `B1`, PHV reshuffles, `key_field_0[7:0]` lands in
`W3[27:20]` whose byte 3 uses only bits `[3:0]` (nibble-clean), the nibble-clean count goes 0 → 1, and a
TCAM block comes back (`compile_logs_bypass/`). **A ternary table's TCAM cost is not a function of the
table — it is a function of the whole program.**

In the 2×84 case one entire block holds **nothing but the 2-bit version field** — the pack format's
first memory unit contains a single line, `Field --version-- [1:0] : in bits [43:42]`, with all five
byte slots empty. 2 of 44 bits used, and a 512-row block consumed out of the 24 in that stage. In the
4×42 layout one midbyte rider is a byte with only 2 used bits inside its low nibble
(`only_one_nibble_in_use()`), so its partner nibble stays free and version rides along for nothing.
That is precisely the isolation credit §4.1.1 now models (`tail_is_isolatable`), and the 4×42 / 2×84
pair is the cleanest demonstration of it: same bits, same bytes, one nibble-clean field of difference,
one TCAM block of difference.

**Two distinct defects, in different passes.** An earlier version of this section claimed they were
symmetric ("fixing either alone removes the block"); that is **false** and was retracted.

*Defect 1 — PHV leaves ternary key remainders straddling a nibble boundary.* This is what costs the
plain 2×84 program its block, and the `bypass_egress` row is the demonstration. The two 4-bit remainders
also occupy two separate containers, costing two crossbar bytes where one would do. PHV demonstrably
*can* co-pack — in the 4×42 compile it ganged three remainders into `B1` at `[2:1]`, `[4:3]`, `[6:5]` —
it simply fills partially-used containers bottom-up, and `B1` had only 7 bits free after
`bypass_egress`. *Why the remainders land where they do:* each container already had an unrelated TNA
intrinsic-metadata value parked at the bottom (`B1[0]` = `ig_intr_md_for_tm.bypass_egress`, `H0[8:0]` =
`ig_intr_md.ingress_port`), and PHV appends at the lowest free bit with no nibble-alignment rule.

**This defect is a *generator* problem and this project has fixed it at the source**, not modelled it:
`build_p4_script` now emits `@pa_solitary` on every `class_tree_*`/`code_*` field, which forbids sharing
a container and eliminates the squatters. It is "Mechanism A" in Appendix B, and it is explicitly out of
scope for `src/p4model` (see `src/p4model/README.md` §3).

*Defect 2 — crossbar sizing never reserves the version nibble.* Isolated by removing defect 1 with
`@pa_solitary`. The resulting layout has no shared containers and no straddling bytes, the ledger says
`g = 4` is feasible — and the compiler **still emits 5 blocks** (`compile_logs_solitary/`). So there is
no P4-level workaround, and **this is the defect §4.1.1 models.** The sizing helper is
`IXBar::increase_ternary_ixbar_space()`, `bf-p4c/mau/tofino/input_xbar.cpp:485-492`:

```cpp
void IXBar::increase_ternary_ixbar_space(int &groups_needed, int &nibbles_needed,
                                         bool /* requires_versioning */) {
    // (TODO): Try to optimize it in the future.
    if (groups_needed > nibbles_needed) nibbles_needed++;
    else                                groups_needed++;
}
```

The `requires_versioning` flag is threaded in from `calculate_sizes()` (line 494) and then **ignored —
its parameter name is commented out**, with a standing `TODO`. So crossbar sizing reserves capacity for
the match bytes only and never for the version nibble a later pass then unconditionally requires. A
second, quieter defect sits in the same loop: `(nibbles_needed + 1) / 2` prices two nibbles as one byte,
which is only true for a fully-used byte.

One further wiring constraint, needed only to reason about *which* byte can go where (it does not change
the counts in §4.1.1, and `src/p4model` does not model it): the crossbar is built from repeating 4-byte
sections (`REPEATING_CONSTRAINT_SECT = 4`), so a byte from **lane *k* of a 32-bit container** can only
occupy crossbar positions **≡ k (mod 4)**; a 16-bit container's bytes are restricted to even/odd
positions; an 8-bit container's are unrestricted. Midbytes sit at positions `11i + 5` = 5, 16, 27, 38,
49, 60 → ≡ 1, 0, 3, 2, 1, 0. Derived by combining `need_align_flags[4][4]`
(`bf-p4c/mau/input_xbar.cpp:453`) with `align_flags[]` (`bf-p4c/mau/tofino/input_xbar.cpp:461`).

**Not filed upstream.** Checked 2026-08-06 against `p4lang/p4c@main`, not just the local `8ffb734bd`
build: `increase_ternary_ixbar_space()`, `ternary_version()` and the `used_midbytes` loop in
`allocate_all_ternary_match()` are byte-identical upstream. Full issue-tracker search found nothing
matching. A verified self-contained reproduction and ready-to-file issue text live in
`reviews/github_issue_tcam_version_bit_packing.md` (still not filed).

#### 4.1.3 Current accuracy of the ternary block model

Two gates, deliberately, because one of them has already failed silently once. `blocks` sat at a clean
17/17 through an entire calibration study with a 24-key **per-table** error underneath it: design
totals let a +1 on one table cancel a −1 on another.

**Per table** — `scripts/tcam_table_scoreboard.py`, 405 observations across 14 CSVs. The production
(refined) price is exact on **100/100** archived and **50/50** held-out classification tables; the
published headline price over-predicts 8 archived ones by exactly 1 block per tree. The charged price
has **0 under-predictions** on every placement the packer emits (§4.1.1).

**Per design** — `scripts/validation_table.py`, end to end. As of C1 (audit §10, per-task tree
readiness) and C5 (the lane-simulation stage packing, §4.3), both merged 2026-09-28, the PRIMARY gate
is a single unified 43-design set compiled **with** the C5 generator pragmas
(`@placement_priority`, `@pa_no_overlay`; `results/compiler_calibration_pinned/`) — the same 43
designs the three older archives below used to compile separately, now re-compiled once with the
tree placement order pinned so the model's placement decision and p4c's agree by construction rather
than by chance:

| set | `stage_depth` | `blocks` |
|---|---|---|
| **PRIMARY — 43 pinned designs** (`results/compiler_calibration_pinned/`; v6's 19 + extra's 8 + margin_screen's 16, all re-compiled with the C5 pragmas) | **42/43**, 1 under (known, see below), 0 over | **38/38** of 38 comparable, 0 under, 0 over |
| pre-pragma, informational — 19 fitted (`results/compiler_calibration_v6.csv`) | **19/19** | **17/17** of 17 comparable |
| pre-pragma, informational — 8 held-out (`results/compiler_calibration_extra/`) | **8/8** | **5/5** of 5 comparable |
| pre-pragma, informational — 16 adversarial (`results/tcam_margin_screen/`; C4: the set originally compiled to choose the now-retired 58/62 fitted margin) | **16/16** | **15/16**, 1 under (known, see below) |

**The one accepted `stage_depth` under-prediction, on the primary gate:** `independent_high_sd12`
(model 13, p4c 14). This is the single design in the pinned archive where pinning the tree placement
order itself cost p4c a stage it did not need at 13 without the pragmas (audit §7.3/§7.6) — the design
is past the 12-stage feasibility ceiling either way, so nothing in the Optuna search loop is misled by
it. **The one accepted `blocks` under-prediction, on the pre-pragma adversarial replay only:**
`margin_independent_M150_k5_s12` (model 66, unpinned p4c 68 — p4c happened to serve its keys in an
order costing 2 more blocks there). The **pinned** compile of that identical design costs exactly the
model's predicted 66; the miss is an artifact of replaying an archive p4c did not compile
deterministically, not a pricing error, and does not appear on the primary gate.

Per-table, the same replacement also passes the scoreboard's charged-price gate 0-under on every
emitted placement (§4.1.1), and reproduces `reviews/model_audit_scratch/proto_model.py`'s prototype
(`--c1`, `FILL=lane`) design-for-design on all 43 pinned compiles (the Phase-3 parity gate the SDD
plan required before trusting the production code).

`blocks` was 12/17 before 2026-09-07, 17/17 until the 2026-09-21 rewrite, 15/17 until 2026-09-25,
16/17 by the 2026-09-27 any-order fit rule, and now (pre-pragma) 17/17 again after C1 fixed
`independent_low_sd12`'s placement. `stage_depth` followed the same arc to 19/19 pre-pragma. Older
pre-`@pa_solitary` figures describe a different generator and are not comparable. **Neither quantity
has ever been observed to under-predict a FEASIBLE design** — the two accepted misses above are
either past the 12-stage ceiling regardless, or an artifact of an unpinned replay the primary gate
does not use.

### 4.2 TCAM — range (feature-encoding tables)

One range-matching table per selected feature (`build_p4_script.py:663-674`, keyed
`meta.<feature>_val : range`). **Its block count comes from the compiler's own COMPILE-time sizing of
the DECLARED interval count, not from the physical row expansion the control plane will actually
perform** — `src/p4model/tables.py:range_matching_resource_usage`:

```
feature_blocks = ceil( compiler_range_rows(len(intervals), key_bit_width) / 512 )
```

with (`src/p4model/ranges.py:compiler_range_rows`)

```
worst   = min(8, 2 * nibbles(key_bit_width) - 1)        # RANGE_WORST_CASE_ROWS_CAP, nibble geometry
quarter = entry_count // 4                              # RANGE_WORST_CASE_ENTRY_FRACTION
rows    = quarter * worst + (entry_count - quarter)
```

**Provenance.** `RANGE_WORST_CASE_ENTRY_FRACTION`, `RANGE_WORST_CASE_ROWS_CAP` and the whole shape of
`compiler_range_rows` above are not fitted parameters — they are a direct transcription of p4c's own
compile-time sizing pass, `RangeEntries::preorder`/`postorder`
(`bf-p4c/mau/resource_estimate.cpp:1628-1693`, `resource_estimate.h:213-214`). Tier H: the compiler's
rule is a 3-line closed form, and copying it is the correct model, not overfitting.

The compiler never sees the interval bounds — this project's range tables are populated at runtime via
the control plane, never via `const entries` (§4.4) — so it cannot cost them exactly. It applies a fixed
distributional guess instead: a quarter of the declared entries are priced at the worst-case row count
for the key's nibble geometry, the rest at one row each. At this project's 16-bit key width that is 4
nibbles → `worst = 7`, and the largest declared entry count whose rows still fit 512 is **206 intervals
per block**. The same formula reproduces all five independently measured per-block capacities as its own
largest fitting `entry_count`: **512 (4-bit), 342 (8-bit), 256 (12-bit), 206 (16-bit), 187 (19-bit)**,
with 20-bit range keys failing to compile outright (`MAX_RANGE_KEY_BITS`, a hard SDE ceiling).
Appendix B "Mechanism E".

**`range_entry_count` answers a different question and must not be used for blocks.**
`src/p4model/ranges.py:range_entry_count` is an exact port of `expand_range()`
(`bf-drivers/src/pipe_mgr/pipe_mgr_entry_format.c`, the real Tofino driver): it gives the true number of
physical TCAM rows the control plane needs to *install* one range key `[lo, hi]`, at **insertion** time.
Verified by hand-trace against `reviews/cited_papers/tofino_results_2.odt.pdf` slide 11's worked example
(`[10,300]` on 16 bits → exactly 4 entries, matching the slide's sub-range boundaries, not just the
count). Never contradicted by two live `tofino_model` studies either: a 34-config width/offset sweep
(`.superpowers/sdd/task-4b-report.md`) and a 32-config real-insertion study
(`.superpowers/sdd/task-4c-report.md`). **Using it to compute blocks under-counts** — measured: a 478-entry table priced at 1 block against p4c's committed 3.

**The deploy-time question it *does* answer** is kept separate, as
`src/p4model/tables.py:range_deployment_overflow`: given the real trained intervals, will they fit
*inside* the blocks the compiler already committed? A committed block count is fixed in the binary; the
control plane cannot grow a table, it just gets `[Not enough space]` partway through insertion. So a
design can be perfectly feasible on blocks and still be undeployable. It can genuinely happen —
`compiler_range_rows` budgets 2.5 rows per entry at 16-bit keys while a single maximally-misaligned
range costs up to 7 — but measured reality averages ~1.96 rows/entry and every row of the calibration
study clears its allocation by at least 1.66×. It is a guard against a tail, not a routine constraint,
which is why it ships as a documented, caller-less oracle -- nothing in the Optuna loop or the P4 generator calls it as a build-time assertion -- rather than as a term inside the block cost.

Facts confirmed about real range-match behaviour:

- **Aligned power-of-2 ranges always cost exactly 1 physical row**, regardless of width — an aligned
  power-of-2 range is exactly one prefix.
- **Misaligned ranges cost more, following the nibble decomposition, not the old `2*floor(log2(span))`
  formula's magnitude.** At width ≈500 that old formula predicted ~16 rows/entry; real cost averaged
  ~1.96. Its *direction* was right, its *magnitude* wrong by about an order of magnitude. (The old
  formula overcounted real usage by roughly 4× against compiled block counts on real feature tables with
  7–68 intervals each.)
- **PHV container width of the key field decides TCAM words per entry, and nothing about the table
  does.** A `bit<16>` range key allocated into a **32-bit W container** costs **2 physical TCAM words per
  entry** (`mau.characterize.log` reports `1 in 2 (88)`); the same key in a **16-bit H container** costs
  **1** (`1 in 1 (44)`). Established by three controlled sweeps over otherwise-identical range tables,
  each varying one thing: declared `size` (11→256), action-data width (4→25 bits) and range key width
  (4→19 bits) **all had zero effect**. Mechanism traced to `resource_estimate.cpp:1628-1653`
  (`RangeEntries::preorder`), which counts nibble-halves of the **container** bytes the field's real PHV
  placement spans, not nibbles of the field's own logical width.
  **Fix: `generate_P4_code` pins every range-key field with
  `@pa_container_size("ingress", "ig_md.<field>", 16)`.** On a real M2 program, pinning all four fields
  took it from **14 TCAM blocks / 10 stages to 12 / 9**, 0 errors. `range_matching_resource_usage`'s
  depth-only arithmetic is correct **because of** that pragma, not by coincidence; without it the
  function would under-count by up to 2× per table.
- **A table's P4-declared `size` is a hard logical-entry cap** independent of remaining physical
  capacity — even cheap, aligned ranges cannot exceed `size` entries.
- **Real fill-to-capacity is insertion-order-dependent near the top of a block.** A block's nominal
  512-row capacity is reachable, but only under a favourable order (wide multi-row entries first), since
  a multi-row range-expanded entry needs several *mutually contiguous* free rows
  (`pipe_mgr_tcam_find_next_free`) while a single-row entry can use any free row. An unfavourable order
  can strand up to ~6 rows (512 vs 506 observed on the identical final entry set). **There is no
  universally-correct flat safety margin**, so none is applied in the model. It was instead fixed at the
  insertion point: `p4/deploy_table_entries.py` sorts each range table's entries by descending physical
  row cost before installing them via `bf_rt`. Verified on a heterogeneous 213-entry set
  (`.superpowers/sdd/task-8-sort-order-verify-report.md`), where sorted and a realistic non-adversarial
  unsorted order both reached 512/512 — a tie, confirming the sort is never worse, not that it wins.
  Re-checked 2026-08-19 at this project's own scale: the largest range table observed is **375 rows**
  and every one fits a single block either way, so the effect is currently inert. Revisit only if tables
  approach 500 rows.

### 4.3 Stage packing: TCAM column geometry and the Ternary Match Input crossbar

`src/p4model/packing.py:crossbar_stages_needed` packs a pool of independent match tables into pipeline
stages and returns a `StagePlan` — where the tables landed (`indices`, `depth`), how many stage indices
hold one (`occupied`), and the blocks actually **charged** (`blocks`), which is not the naive per-table
sum. Both pools go through the same function: the ternary classification pool (one table per tree) and
the range pool (one table per feature) are physically distinct table pools and are packed separately.

**A stage's 24 TCAM blocks are 12 rows × 2 columns, not one flat pool.** `mau_spec.h:88-90` gives
`Tofino_tcam_rows = 12`, `Tofino_tcam_columns = 2`, with an explicit source comment that the figure is
correct for Tofino 1, 2 and 3. A table needing several blocks chains them down **one column**, so what
decides a stage is the tables' **widths**, not their total:

- three 8-block tables total exactly 24 and need **two** stages (8+8 overflows a 12-row column);
- four 6-block tables, also 24, fit in **one** (6+6 | 6+6).

`packing.fits_two_columns` is the test. It is an **exact subset-sum over achievable column loads**, not
a greedy fit: with two columns and a handful of tables the state space is trivial, and a greedy reports
false violations. That matters historically — the column rule was once written off as "refuted" on the
strength of a per-width shortcut (`2 * floor(12 / w)`) applied to stages holding *mixed* widths, where
the real packing is far more permissive: `(7+5 | 6+6)` fits four tables that shortcut rejects.

**Evidence: 19 real p4c compiles** over synthetic tables of 5..12 blocks, all keyed on one shared field
so that TCAM blocks and not the crossbar bound the result — `scripts/tcam_column_sweep.py`,
`results/tcam_column_sweep.csv` (16 points) and `results/tcam_column_sweep_wide.csv` (3 points).
Consistent with every one of the 18 calibration rows' committed placements, in both pools, with no
exceptions. Appendix B "Mechanism C". **This affects stage count only, never a table's own block count.**

**A single table wider than a column really does span both** — measured at 14, 16 and 24 blocks, each
compiling into one stage. `_stage_shards` models that by splitting the table into column-sized pieces:
fill 12-block columns and leave the remainder. Splitting a table's **rows** is real; splitting its
**key** is not (a key is one indivisible match, and a key wider than
`TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` is rejected by the compiler rather than spread across stages, so
`_stage_shards` raises `CrossbarKeyTooWide` instead of pricing an impossible design). Only the **first**
shard carries the key's field widths, because the version field is stored once per table word, not once
per shard.

> Prior to **finding 1.5a** (2026-09) the split was into *n equal pieces, each rounded up*, which charged
> a 13-block table as 14 and a 23-block one as 24 — an arithmetic artifact. **No table in the 19-row
> archive exceeds 12 blocks**, so this path is unexercised by the calibration and therefore
> **unvalidated against hardware**. It is the one cost-*lowering* change in this work, licensed because
> it corrects rounding rather than relaxing a measured limit.

**Every stage must satisfy three limits at once:**

| limit | constant | evidence |
|---|---|---|
| TCAM blocks, as a 12×2 column packing | `TCAM_BLOCKS_PER_STAGE = 24` | above |
| independent match tables | `TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8` | hard cap, confirmed identically at every key width tested from 8 to 512 bits |
| bytes of **distinct key fields** present in the stage | `TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE = 64` | binding once per-table key width passes ~8 bytes; two exact 64-byte saturations observed (2 tables × 32 bytes; 1 table × 64 bytes) |

*(the table-count and byte caps were first written up as RM-5/RM-6/RM-7,
`reviews/archive/t12_required_changes.md` Section 1.3. Provenance since confirmed against the p4c
source directly: `TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE = 8` is `TERNARY_TABLES_MAX`,
`bf-p4c/mau/tofino/memories.h:54`.)*

**These constraints are not separable** — solving each relaxation alone and taking the max can
under-count. Counterexample: tables (20 blocks, 5 B), (20, 5), (1, 60). The blocks-only bound is
`ceil(41/24) = 2` and the crossbar-only bound is 2, but no two of the three fit in one stage
(20+20 = 40 blocks > 24; 5+60 = 65 bytes > 64), so the true answer is 3.

**The crossbar charges the UNION of distinct key fields in a stage, not the sum of per-table key
widths.** Two tables in the same stage matching on the same field read the same byte slots and the field
is charged once. That is not a corner case here: every classification table of one task keys the
identical `meta.code_<feature>` field set, and under `disjoint` both models' range tables for a shared
feature key the identical `meta.<feature>_val` field. Measured: `joint_low_sd7`'s stage 7 holds four
tables on one 32-byte codeword and the compiler reports **32** crossbar bytes, not 128;
`independent_low_sd6`'s stage 7 holds two 19-byte and two 4-byte tables and reports **23**, not 46.
Summing per table over-counted stages by up to 6 on that sample. (Passing `key_fields=None` gives every
table a private synthetic field, which reduces the union arithmetic to the old per-table sum exactly.)

**RETRACTED 2026-09-21 — "a per-stage crossbar GROUP cap was probed and NOT found."** This section used
to read: *"Two solid keys of 34 and 30 crossbar bytes need 7 + 6 = 13 groups in a stage that has only
12, pass the 64-byte cap at exactly 64 — and p4c placed them in one stage. Do not reinstate a group cap
without contrary evidence."* That is **false on its own evidence.** The "7" and "6" are *block* counts,
not group counts; the two tables **share group 5**; and the probe's own assembly
(`scripts/tcam_group_cap_probe.py`, point `groups_13_bytes_64`) uses groups **0–11 — exactly 12 of 12**.
The probe landed *on* the cap and was read as having sailed past it. The cap is real:
`src/p4model/target.py` now records `TERNARY_CROSSBAR_GROUPS_PER_STAGE = 12`
(`MAX_TERNARY_GROUPS`, `bf-p4c/ir/tofino.def:38`) and `TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE = 6`.

**Nothing consumes those two constants as a budget**, and that is intentional — documentation and
provenance only. The near-cap effect a flat group *budget* would price (spec F5) is instead handled
by the **crossbar-lane stage simulation** below (`src/p4model/lanes.py`, C5, 2026-09-28), which counts
which of a stage's actual 12 groups and 6 midbytes each key's bytes can legally occupy, not a group
*count*.

**Where the sharing charge is applied — C5, 2026-09-28: the fitted crowded-stage margin is retired,
replaced by an ordered crossbar-lane stage simulation.** A table's block count is not a property of
the table alone (§4.1.1), so placement and charge must be computed together. `crossbar_stages_needed`
takes `key_field_bits` (one bit-width tuple per table; only the classification pool passes it), and,
for any stage where two or more *distinct* keys could actually meet — the only case a per-table price
cannot settle alone — routes the classification pool through the ordered stage simulation described
here (audit §10 C5, `reviews/model_audit_2026-09-27.md` §7). A pool where every table keys the same
field set (every `joint` design, and any `disjoint` stage where only one classification key is
present) is **not** simulated at all: it is placed exactly as before C5, by the declared-price packer
the range pool also uses, so `joint` `blocks` and `stage_depth` are structurally identical to the
pre-C5 model — not merely identical on the archived designs, but provably so by construction
(`packing.crossbar_stages_needed`'s own docstring calls this the plan invariant the SDD run's stop
rules required; fuzzed against an independent reference in `tests/test_p4model_guards.py`).

**1. Order — pinned by the generator, not searched by the model.** Which tree p4c's placer serves
first in a stage used to be an open question this document called "the allocator's choice" and priced
with a symmetric worst-case margin (below). It no longer is: `build_p4_script.py` now emits
`@placement_priority(2)` on every `get_classification_tree_ddos_*` table and `@placement_priority(1)`
on every `..._app_*` one (`program.PLACEMENT_PRIORITY`), plus `@pa_no_overlay` beside every
`class_tree_*` field's existing `@pa_solitary` (never on `code_*` — doing so once cost p4c an extra
stage by pushing fields into wider PHV containers, §7.3 arm C in the audit). Confirmed to pin p4c's
own placement choice: on 20 compiles at `reviews/model_audit_scratch/priority_exp_ddos_first_noovct/`
(audit "arm D") and 43 at the primary gate below, the packer's placement order and p4c's agree by
construction, not by coincidence — stage by stage, the remaining table with the highest
`placement_priority`, ties broken by "last listed in program order", that is ready
(its readiness level, §4.6, has been reached) and fits; when none fits, the next stage.

**2. Price — each stage's first key pays its standalone ladder price; every later key pays a LEFTOVER
price computed by simulating the actual crossbar slots the earlier key left free.** The first
distinct key placed in a stage (by the pinned order above) is charged exactly `codeword_to_blocks` ×
its row count, unchanged from §4.1.1 — this is why the plan invariant above holds and why a `joint`
pool is never routed through this machinery at all. Every later key in the same stage is priced by
`src/p4model/lanes.py`, a module ported verbatim from the audit's validated scratch prototypes
(`width_layout.py`, `fastlane.py`, `stage_sim_fast.py`), in three layers:

  - **Layout from width** (`lanes.layout`/`relaxed_layout`/`key_bytes`, audit §7.4). The model runs
    before any compile exists, so it has no real PHV log; it predicts one from each field's bit width
    alone (`w <= 8` → an 8-bit container, `9..16` → 16-bit, `17..32` → 32-bit from bit 0, wider
    fields as a top 32-bit slice plus low-order 32-bit slices) — exact on **90.8%** of 1130 real
    `code_*` fields surveyed, with a relaxed fallback (whole 32-bit containers split into 16+narrow
    when the plain rule would price worse than production's own `codeword_to_blocks`) that brings the
    lane price computed from predicted widths to equal the lane price computed from the REAL compiled
    PHV layout on **148/148** real keys.
  - **Lane price by counting** (`lanes.standalone`/`price_with_supply`, audit §7.2 and Appendix A).
    Every crossbar byte slot has a lane (`slot index mod 4`; Appendix A below derives this from p4c's
    own crossbar geometry). A 32-bit container's byte `k` may sit only in a lane-`k` slot, a 16-bit
    container's only at matching parity, an 8-bit container's anywhere. `standalone` computes the
    smallest legal block count for a key with the WHOLE crossbar free — it agrees with
    `tables.codeword_to_blocks` on **74/74** real design keys, which is the empirical form of the
    same plan invariant. `price_with_supply` computes the cheapest legal fit into a PARTLY used
    crossbar — the free groups and midbytes the earlier key's `first_key_occupancy` (fullest-lane
    midbyte choice, lane-aware group fill — the row in the audit's §7.5 comparison table that beat
    every alternative, 334/336 probe keys exact, 0 under) left behind. Reproduces p4c on **133/133**
    real keys sharing a crowded stage, given where the other key actually landed.
  - **Stage simulation** (`lanes.stage_prices`, audit §7.5-7.6). Keys are priced one at a time in the
    pinned placement order: the first pays `standalone`, every later key pays `price_with_supply` into
    what is left, and the packer's own `_stage_key_prices` mirrors this exactly except that the FIRST
    key is charged production's `codeword_to_blocks` (row-scaled) rather than the bare lane price, per
    the plan invariant — the two prices agree on every real key measured, so this substitution changes
    nothing observable.

**3. Fit — the same three per-stage limits as before, plus one refusal that survives from the old
rule unchanged.** At most `TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE` (8) tables, every key must have a
legal lane price, the charged blocks must pack the stage's 12×2 columns (`fits_two_columns`) — and
when two or more DIFFERENT keys are present, their combined crossbar bytes must not exceed
`TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62`, or the stage is refused outright, whatever the lane
simulation says it could price. This is a **greedy-give-up safety net**, not a pricing margin: p4c's
own allocator can abandon a later key the simulation can still legally price. `M150_k7_s11`'s stage 8
would hold 22 + 41 = 63 combined bytes; p4c gave the first key groups `{0,1,3,4}`, the later (app) key
then needed 9 blocks, and the real placer moved it to the next stage instead — no simulation of any
kind reproduced that choice (audit §7.6), so stages past 62 bytes are refused rather than priced.
Without this refusal, `M150_k7_s11`'s `stage_depth` under-predicts (11 vs the real 12).

**Measured end to end, on the 43-design primary gate (`results/compiler_calibration_pinned/`,
compiled WITH the pinning pragmas): `stage_depth` 42/43 exact — the one miss,
`independent_high_sd12`, is the design where the pragmas themselves cost p4c a stage, infeasible
either way — and `blocks` 38/38 exact** (of 38 designs with a committed `mau.resources.log`; §4.1.3
has the full breakdown and the pre-pragma informational replays). The same code reproduces
`reviews/model_audit_scratch/proto_model.py`'s validated prototype (`--c1`, `FILL=lane`)
design-for-design on all 43 — the parity gate the implementation plan required before trusting the
production code over the scratch prototype it was ported from.

**Interaction with C1 (per-task readiness, §4.6).** Under `disjoint`, a task's trees now wait only for
their OWN task's range tables (§4.6), so a task with an early-finishing range pool can seed its trees
into a stage the OTHER task's range tables are still occupying. When that happens, the range tables'
`meta.<feature>_val` fields are keys placed on that stage's crossbar **first**, ahead of any tree key,
via `crossbar_stages_needed`'s `seed_stages` argument (`StagePlan.stage_loads` from the range pool)
— so a tree landing beside a seeded range table is always pricing as a LATER key, never below its own
`codeword_to_blocks`. Seeds are never charged again (the range pool already charged them); they only
constrain what the classification pool's lane simulation sees as already occupied.

**RETIRED 2026-09-28 — the fitted 58-byte crowded-stage margin, its any-order-fit refinement
(2026-09-27), and the per-key saturation margin before that (2026-09-21).** Kept here as the
investigation record the mechanism above replaced, not as a description of current behaviour.

From 2026-09-25 to 2026-09-28 the model instead read `target.py`'s
`TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE = 58` (since **deleted** from `target.py` entirely —
it is not merely unused): when two DIFFERENT keys shared a stage and together filled more than 58 of
its 64 crossbar bytes, every table of the non-first key was charged **+1 block flat**, whatever its
own slack; above 62 the stage was refused (the one number that survives, now as the lane simulation's
own refusal threshold above). Because the model could not simulate WHICH key p4c's greedy allocator
served first, `fits()` required **every** ordering of a stage's distinct keys to pack column-wise and
`charged()` priced the **largest** total among orderings that fit (the "any-order fit rule",
2026-09-27, `reviews/final_model_check_2026-09-27.md` section 1b) — a margin against not knowing the
order, superseded outright once C5 made the order a pinned, known fact rather than an unknown to
margin against.

Measured evidence for the retired rule, for the record: scored against 19 fitted, 8 held-out and 16
adversarial real designs compiled to stress it (`scripts/tcam_margin_screen.py`) plus 405 per-table
observations, a flat +1 above 58 bytes (refused above 62) was the only tested rule with 0
under-predictions everywhere, beating plain byte caps at 58/60/62 alone and a "credit-dependent key"
saturation predicate. Measured per combined distinct-key bytes: nothing paid at ≤ 58 bytes on any real
stage or probe shape; +1 on several real and probe keys at 59-62 bytes (`independent_low_sd5`'s real
`(54, 56)` app key, the ragged `(179, 204)`, each 11-byte ddos tree of real design `M150_k5_s12`); +2
at 63-64 bytes (`(84, 84)`, `(54, 56)`, `(27, 52)`). Its cost, also for the record: some real stages
at 59-62 bytes shared for free anyway (`independent_low_sd12` shares 60 bytes free; three campaign
designs share 59 free), so a handful of designs at the 12-stage ceiling were priced one stage or block
deeper than p4c placed them — the single reason the pre-C5 model's fitted `blocks` sat at 16/17 rather
than the 17/17 the pre-pragma replay (§4.1.3) reports today. A **per-key saturation margin** (+1 to
any non-first key whose standalone price was exactly saturated, in ANY shared stage) preceded the
58-byte rule from 2026-09-21 and was itself retired on 2026-09-25, over-predicting
`independent_high_sd7`/`sd8` by +1/+3 in stages of only 22 and 35 free bytes where p4c charged
nothing; a proposed `tail_is_isolatable` root cause for that over-prediction was separately refuted by
measured keys of exactly the shape it blamed (`w035`/`w036`/`w043`/`w044`, isolated by p4c despite
starting outside a single 32-bit container) — closed, not reopened by the C5 rewrite.

The **range pool never passes `key_field_bits`**: a range table keys one 16-bit field (2 bytes), and the
8-table cap holds a stage to 16 such bytes, nowhere near crowding it. Neither the two retired margins
above nor the current lane simulation apply there by construction, and `range_plan.blocks` is provably
identical to the naive `range_blocks` sum.

**Row parity is real, and is deliberately not modelled** — see Appendix B "Mechanism C, addendum".
Rows `2i`/`2i+1` share a half-byte selector, so an even-height block run may start only on an even row.
Confirmed 317/317 across the archive with zero exceptions, and **provably never paid** under a packer
that is free to choose the order it fills a column: place the even-height tables first and every prefix
sum stays even. `fits_two_columns` is therefore right to model column *loads* and ignore which physical
row a table starts on.

**Known under-count risk (unresolved).** A follow-up compile sweep (`reviews/open_issues.md` item 3,
`results_rmx_crossbar.csv`, N = 1, 7, 8, 9, 17 independent 16-bit range tables) confirmed the
8-tables/stage cap generalises to range tables at 16-bit width — but found range tables cost **~2× the
crossbar units per byte** that ternary tables do at the same width (4 vs 2). So reusing
`TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` verbatim for the range pool is **not known-conservative**: once a
design's range-table byte budget rather than the 8-table cap becomes binding, `crossbar_stages_needed`
likely **under**-counts range stages. **Unreachable for this project's own generator** — every range
table it emits keys exactly one 16-bit field, and 8 such tables consume 32 of the real 66 units/stage
(48%), or 48 even under a hypothetical stretch to the 19-bit SDE ceiling — so the table-count cap always
binds first. The exact byte-width crossover was never pinned down; it would need a >64-bit
combined-width multi-field range sweep.

**Why this matters beyond being "interesting":** a design can have plenty of spare TCAM blocks in a
stage (16 of 24 used) and still be forced into another stage purely by the crossbar or by the column
geometry. A cost model that reasons only about a flat 24-blocks-per-stage figure will under-count
stages for multi-table designs.

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

### 4.6 Stage depth: the register schedule and readiness levels

**`stage_depth` is not a bin-packing result — it is a placement.** `src/p4model/usage.py:assemble_usage`
lays the pipeline out in this order, and `stage_depth` is what the 12-stage ceiling
(`TOFINO_PIPELINE_STAGES`) is checked against:

```
[ 3-stage prologue ] [ register schedule ] [ range pool ] [ classification pool ] [ vote epilogue ]
   FLOW_HASH_LEVEL        METER_ALUS_PER_STAGE = 4                                  VOTE_EPILOGUE_STAGES = 1

stage_depth = max(range_plan.depth, ternary_plan.depth) + VOTE_EPILOGUE_STAGES
```

- **`FLOW_HASH_LEVEL = 3`** (`src/p4model/program.py`). Every per-flow register is indexed by
  `meta.flow_hash`, and three stages elapse before the first `RegisterAction` can run: a metadata-init
  table at stage 0, `tbl_calc_flow_hash$precompute` at stage 1, `tbl_calc_flow_hash` at stage 2.
  Measured over all **121 range tables in the 19 real compiles** of `results/compiler_calibration/`:
  every committed placement opens that way. Under the old value of 1, `real_stage − level` had a floor
  of exactly +2 in every one of the 19 rows; at 3 the floor is 0, i.e. levels now name the earliest
  stage the compiler really uses. *(Values above the floor are tables the packer legitimately pushed
  later — placement, not origin.)*
- **`VOTE_EPILOGUE_STAGES = 1`** (`src/p4model/program.py`). The `vote_app`/`vote_ddos` tables read every
  tree's class, so they always sit one stage past the last classification table. Measured as exactly 1
  in all 19 compiles, with no exceptions and no scaling. Leaving it out understated every design's depth
  by a whole stage.
- **`METER_ALUS_PER_STAGE = 4`** (`src/p4model/target.py`). Read off the compiler's own arithmetic
  rather than fitted: `mau.resources.log`'s percentage table reports a Meter ALU count of 4 as 100.00%
  (`joint_high_sd7` stages 3-6, among others). Swept over 2/3/4/5/6/8 against the 18 committed
  calibration placements, **only 4** reproduces the compiler's last-register stage on every row — its
  neighbours manage 13, 11, 10 and 8 of 18.

#### The register schedule

`src/p4model/registers.py:register_stage_schedule` decides which stage each per-feature `Register<>`
runs in. **Three constraints drive it, and none of them is a capacity limit:**

1. **Stateful-ALU width.** Every emitted `RegisterAction` holds one of a stage's four stateful ALUs for
   the whole stage, so 16–20 registers need `ceil(n / 4)` stages **regardless of how deep any single
   feature's dependency chain runs**. This is the constraint that actually binds at campaign scale;
   chain depth alone reports 5 stages on each of the six k ≥ 13 calibration rows where the compiler
   really uses 7.
2. **The per-block placement-cursor floor.** Registers are emitted unconditional → `if (meta.fwd == 1)`
   → `if (meta.fwd == 0)` (`REGISTER_BLOCK_ORDER`, matching `generate_P4_registers_and_apply`'s three
   `_execute_lines` calls). p4c's table placer walks the control block with **one work-list cursor**, so
   it cannot begin the second gated block before the first is fully placed. Each block is therefore
   floored at the last stage the previous one used, and a `bwd`-gated feature prices one stage later
   than the symmetric `fwd`-gated one.
3. **Gated-block interiors are unusable** (below).

The schedule is a greedy earliest-level-first list schedule, run once per block, with critical-path
(height) tie-breaking so `ORIENTATION_REGISTER` — which carries every gated feature's whole chain — is
never displaced by a leaf register. **Measured against the 18 committed placements in the original v6
calibration set, the makespan matches the compiler's on every row** (7,7,7,7,4,5,4,4,4,4,7,7,7,4,4,4,4,4).
Adding the per-block floor left all 18 intact while raising per-register agreement from 88 to 148 of
170. Ties inside a level do move individual registers, but never the makespan — verified over 300
random feature-order shuffles of every calibration row, where the last-register stage never moved.
Re-run against the full 43-design archive (audit §8, `reviews/model_audit_scratch/register_check.py`,
of which 38 had a committed allocation): **makespan 38/38**, per-register agreement 362/411 exact (the
rest +-1, symmetric ALU-slot swaps that never move the makespan), and per-feature readiness-level
agreement 265/314 — but range-pool `stage_depth` still 38/38, because those per-feature swaps never
propagate to the quantity the model actually predicts. Not worth pinning the schedule further to close
the 265/314 gap.

`readiness_levels_for` then gives each feature's range table a level one stage past that feature's last
register. `feature_readiness_level` remains the per-feature *depth* rule
(`FLOW_HASH_LEVEL` + 1 if fwd-/bwd-gated + one per `RegisterAction` in the chain), but it prices one
feature at a time and cannot see the ALU cap, which is a property of the whole selected set.

#### Gated-block interiors — and there is NO range-pool fill limit

Tofino has no program counter: every table hands the next stage a next-table pointer, so p4c's placer
walks the control block with a work-list cursor and a table becomes a candidate only when the cursor
reaches it. `generate_P4_registers_and_apply` emits the gated register blocks **before** every match
table, so while the cursor is inside `if (meta.fwd == 1) { ... }` the entire range and classification
pools are out of reach — **regardless of how empty the stage is**. In `independent_high_sd10`'s stage 5
the TCAM is 0/24, the ternary crossbar 0/66 and the logical table IDs 4/16.

> **This is a control-flow constraint, not a capacity one. THERE IS NO RANGE-POOL FILL LIMIT — do not
> look for one.** Two separate investigation passes lost time hunting for a capacity rule that would
> explain these empty stages. An empty stage in a committed placement is usually *forbidden*, not
> merely unfilled.

`registers.gated_block_interior_stages` returns exactly the forbidden indices. A block spanning stages
`[first, last]` blocks `range(first + 1, last)` and nothing else: in the stage the cursor *descends into*
the block, outer tables can still be back-filled (p4c logs them verbatim as
`potential backfill ... before tbl_prog951`), and in the stage it pops back out they are candidates
again — so **a two-stage gated block costs nothing at all**. Measured against all 18 calibration rows:
the range pool's committed occupancy has a hole on exactly **5** of them, and this returns precisely
those holes — right rows, right indices, nothing on the other 13. Two of the five
(`independent_high_sd6`, `joint_high_sd8`) still cost **+0** stages overall because their range tables
were not going to occupy that stage anyway, which is why the caller applies this as a **placement
constraint** (`crossbar_stages_needed(unavailable_stages=…)`) and never as a per-row penalty.
Appendix B "Mechanism B".

#### Placement, not optimisation

With `readiness_levels`, `crossbar_stages_needed` switches from first-fit-decreasing to **eager**
placement: earliest legal stage at or after the table's level with room, spilling forward when full, and
skipping `unavailable_stages` outright. Eager is the point — the theoretical optimum would drop every
table into the single latest stage, and the compiler does not do that. Measured: M2's range pool really
occupies **2** stages, which only eager placement reproduces; 2 range + 1 classification = 3 match-table
stages, exactly the compiler's own placement, where the pure packer predicted 2.

The classification pool's start level is `range_plan.depth` — derived from where range tables actually
**landed**, not from `max(range_levels) + 1`, the earliest stage one was merely *allowed* to start. The
8-table crossbar cap can spill a range table past its level, and the naive form would then schedule a
classification table into a stage a range table still occupies. `assemble_usage` asserts the two pools'
stage indices stay disjoint.

Under **`joint`**, every tree's readiness level is `range_plan.depth` and stops there — the paragraph
above is the whole story. Under **`disjoint`**, one further correction applies before the classification
pool can be placed at all: see C1, next.

#### Per-task tree readiness under `disjoint` (C1, 2026-09-28)

**Deliberate simplification, corrected.** Before this change, `assemble_usage` gave every
classification tree the SAME readiness level: one past the last stage of ANY range table, from
EITHER task. That over-constrains `disjoint` designs specifically — a `disjoint` app tree keys only
`meta.code_app_<feature>` fields, which come from the app task's OWN range tables, and reading a
ddos-task range table's fields is impossible for it by construction (`build_p4_script.py` never emits
a table keying a field a tree does not use). `usage.py:tree_readiness_levels` fixes this: a tree of
task `t` is ready at `1 + max(stage of every range table labelled t or SHARED_TASK)`, never waiting on
a range table that belongs only to the other task; a task with no range table of its own falls back to
`FLOW_HASH_LEVEL + 1`.

**Evidence.** `reviews/model_audit_scratch/per_task_variant.py`, 28 real `disjoint` compiles: p4c
places 8 of 168 trees before the OTHER task's last range table finished (in 3 of the 28 designs), and
**never** places a tree before its OWN task's last range table. So the per-task rule is never too
early, only sometimes usefully earlier than the old single-level rule. On the calibration archive it
fixes two placements outright: `independent_high_sd12` (14 → 13 predicted stages, matching p4c) and
`margin_independent_M250_k4_s15` (12 → 11). Re-run on the full replay of all 43 archived compiles (v6
+ extra + `tcam_margin_screen`): exactly those two designs move, and no `blocks` total changes
anywhere — the joint invariant (below) holds throughout.

**Mechanics.** Because a `disjoint` task can now start its trees before the OTHER task's range tables
have finished, the two pools can share a stage — something that could never happen under the old
single-level rule. `crossbar_stages_needed`'s classification-pool call is therefore SEEDED with the
range pool's `StagePlan.stage_loads` (`usage.py`, `packing.py`'s `seed_stages` argument): a seeded
range table's shards count against that stage's block, table-count and column budget exactly as this
pool's own tables would, and — because §4.3's stage simulation treats the range table's
`meta.<feature>_val` field as a key placed on the crossbar FIRST, before any tree — a tree sharing a
seeded stage always prices as a LATER key, never below its own `codeword_to_blocks`. Seeds are never
charged twice; the range pool already priced them.

**The joint invariant, preserved.** Under `joint` every range table is `SHARED_TASK`, so every tree's
level still reduces to exactly `range_plan.depth` — the single level it always had — and no seeded
stage is reachable by a `joint` pool in practice (`tests/test_align_budget.py`'s
`total_blocks == usage.blocks` check, and the golden fixture's 8 `joint` rows, are both
byte-identical before and after C1). This matters beyond tidiness: `src/training/threshold_alignment.py`
prices keys with `codeword_to_blocks` alone and never calls `packing.py` at all, so it depends on
`joint` pricing staying exactly as it was — which C1 (and C5, §4.3) both preserve by construction, not
by re-validating alignment separately.

#### Current accuracy, and what is still not modelled

The primary end-to-end gate is now the same 43-design pinned set §4.3 reports: `stage_depth` **42/43**
exact (the one miss, `independent_high_sd12`, is under-predicted by 1 — model 13, p4c 14 — but the
design is infeasible either way, since both exceed `TOFINO_PIPELINE_STAGES = 12`, so the miss does not
affect real usability), `blocks` **38/38** of 38 comparable. Replayed pre-pragma
(informational, `scripts/validation_table.py` against `results/compiler_calibration_v6.csv` alone):
**`stage_depth` is exact on 19 of 19 rows, 0 under-predictions** — C1 closed the one remaining miss
(`independent_high_sd12`, 14 → 13) that a single shared range-readiness level could not. Older
pre-`@pa_solitary` figures (mean error 0.78 stages, exact on 9) describe a different `stage_depth`
definition and a generator defect since fixed; they are not comparable.

**Held-out, frozen protocol (C6, 2026-09-28).** A batch of 30 campaign designs drawn from
`results/campaign_backup_20260825` by a pre-recorded seed, excluded from every calibration/fitting
step, generated with the C5 pragmas and compiled once (`results/heldout_2026_09_28/`, scored once,
never tuned on): `stage_depth` **28/30 exact, 1 under, 1 over**; `blocks` **21/22 exact, 0 under, 1
over** (of 22 designs with a committed allocation; 8 exceed 12 stages). Per-table, both the
classification pool (144/144) and the range pool (443/443) are exact against p4c's committed
placement.

**Composition caveat — do not cite this batch as "the held-out test validated C5" without it.**
The draw's actual composition is **24 `joint` / 6 `independent` designs**
(`results/heldout_2026_09_28/manifest.json`, verified by counting `arm_slug`), not an even 25/5
split. This matters specifically for C5 (§4.3), whose ordered lane simulation only activates on a
`disjoint` stage holding two or more DISTINCT keys — a `joint` pool never reaches it at all (plan
invariant 1), so 24 of the 30 designs say nothing about the new mechanism by construction. Worse:
**no table in this batch was ever actually charged a lane-leftover price different from its own
standalone price** — every table that shared a stage with another key still priced at its ordinary
`codeword_to_blocks`, so this specific draw exercises the pinned-order *placement* logic (C1/C5's
ordering) but provides essentially no direct evidence for the *leftover-lane pricing* path that is
C5's actual novel arithmetic (§4.3's `price_with_supply`). The per-table range score (443/443) is
also weaker evidence than it looks: most range tables in this project's designs key exactly one
16-bit feature field and are trivially 1 block each, so a near-perfect range score is a low bar;
the classification per-table figure (144/144) is the more meaningful number, since classification
tables are where the byte-ladder and lane-sharing mechanisms actually bind. In short: this batch is
solid end-to-end evidence for the pipeline as a whole and for C1's readiness fix, but weak,
underpowered evidence for C5's lane-pricing mechanism specifically — that mechanism's real
validation is the 43-design pinned gate above and the dedicated probes cited in §4.3 (`tcam_mixed_
key_cap_sweep.py`, `tcam_margin_screen.py`, the `lanes.py` audit sweeps in Appendix A).

**A genuine, PRE-EXISTING, still-open model gap, found by that held-out batch.** `heldout_independent_M150_k14_s13`
(`disjoint`, 57 combined key bytes — below the crowded-stage threshold, so not a C5/lane-mechanism
issue): the model predicts `stage_depth = 11`, p4c places it at **12**. Replaying p4c's OWN committed
block counts through this model's placement logic still gives 11, so the gap is in the RANGE POOL'S
placement order, not its price: p4c spills one range table (`table_0_app_bwd_iat_max`) to stage 9
(the model packs all 27 range tables into stages 4-8, 8 per stage either way), and under C1 the app
task's classification trees then wait for that table and land one stage later than the model expects
(stage 10 vs. 9). This predates the whole 2026-09-27/28 rewrite — replaying 4 historical commits' code
against this same frozen design reproduces the same 1-stage gap under each of them — so it is not a
C1/C5 regression, and it is out of scope for this run: no fix is proposed here, and it should be
treated as a known, open limitation of the range-pool bin-packing order (§4.3's "known under-count
risk" paragraph is a related but distinct gap — the byte-budget crossover, not this ordering
question).

**All three accepted under-predictions in the current state, together.** §4.1.3 above lists the
first two against the ternary block gates; this held-out batch adds the third. Listed together so
neither this section nor §4.1.3 is read as exhaustive on its own:

1. `independent_high_sd12` (primary pinned gate, §4.1.3) — model 13, p4c 14 stages. Infeasible
   either way (both exceed `TOFINO_PIPELINE_STAGES = 12`).
2. `margin_independent_M150_k5_s12` (§4.1.3) — model 66, p4c 68 blocks, but **only on the
   pre-pragma adversarial replay** of `results/tcam_margin_screen/`; this design was compiled
   WITHOUT the C5 generator pragmas. The **pinned** compile of the identical design costs exactly
   66, matching the model, so this miss sits outside what C5's accuracy claim covers — that claim
   is conditioned on the pragmas being present, which they now always are for any newly-generated
   design.
3. `heldout_independent_M150_k14_s13` (this section) — model 11, p4c 12 stages, the C6 held-out
   range-pool placement-order gap described above.

**Not modelled:** gateway/action dependencies, PHV-sharing hazards (§3.6), and the compiler's own
placement heuristics. A follow-up pass compiled the real generator's output at two scales (M2's 3/1
trees and a larger 8/4-tree build), each with and without the `@pa_container_size` pins. Removing the
pins **does** produce a genuine cross-feature dependency edge in `table_dependency_summary.log` — the
`fwd_last_arrival_time_action` table picks up `OUTPUT ANTI_NEXT_TABLE_DATA` /
`ANTI_TABLE_READ ANTI_ACTION_READ ANTI_NEXT_TABLE_DATA` on the unrelated `flow_iat_max_action` /
`flow_iat_mean_action` tables purely from sharing PHV container `W2` — so the gap is not imaginary. But
in all four compiles the range tables' real stage stayed 5/5/5/6, identical to the prediction; the
compiler had slack to absorb the edge for free. A push to 20/10 trees to force the issue was abandoned
when the Python-side codeword generation did not finish in ~14.5 minutes. Full detail:
`reviews/open_issues.md` item 5. This is "Mechanism A" in Appendix B, and it is fixed in the
*generator* (`@pa_solitary` on every written field) rather than modelled.

---

## 5. Quick-reference constants (this compiler version, 9.13.4)

| Constant | Value | Meaning |
|---|---|---|
| `TERNARY_MATCHING_ENTRIES_PER_BLOCK` | 512 | rows per physical TCAM block (ternary **and** range tables share this) |
| `TCAM_BLOCK_KEY_LENGTH` | 44 | usable key bits per physical TCAM row = one crossbar group = **5.5 crossbar bytes** (§4.1.1) |
| `CROSSBAR_PRIVATE_BYTES_PER_GROUP` | 5 | private bytes per group; the 6th half-byte is the shared midbyte |
| `CODEWORD_KEY_OVERHEAD_BITS` | +4 bits | the version/valid nibble, once per entry. No longer a separate term: `crossbar_capacity`'s `floor((g-1)/2)` already withholds one half-byte for it. `codeword_bits_to_blocks` carries it only for the empty-key floor (§4.1) |
| `TCAM_BLOCKS_PER_STAGE` | 24 | physical TCAM blocks per stage — **but as 12 rows × 2 columns**, not a flat pool (§4.3) |
| `TCAM_ROWS_PER_STAGE` / `TCAM_COLUMNS_PER_STAGE` | 12 / 2 | `mau_spec.h:88-90`; a table chains its blocks down ONE column |
| `TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE` | 8 | independent match tables per stage, hard cap (`bf-p4c/mau/tofino/memories.h:54`, `TERNARY_TABLES_MAX`) |
| `TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` | 64 | bytes of **distinct key fields** per stage, hard cap |
| `TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE` | 62 | when two DIFFERENT keys share a stage, a refusal safety net above this many combined crossbar bytes — the one figure surviving from the retired fitted crowded-stage margin (§4.3, C5, 2026-09-28) |
| `TERNARY_CROSSBAR_GROUPS_PER_STAGE` / `TERNARY_CROSSBAR_BYTE_GROUPS_PER_STAGE` | 12 / 6 | ternary groups (5 private bytes each) and byte groups/midbytes (2 nibbles each) per stage (`bf-p4c/ir/tofino.def:38`, `MAX_TERNARY_GROUPS`). As of C5 (2026-09-28) these ARE consumed — by `src/p4model/lanes.py`'s crossbar-lane simulation, which uses them as the physical slot count a later key's leftover price is computed against — not as a flat group-count budget (§4.3, Appendix A). Corrects an earlier "no group cap was found" reading of the same probe |
| `METER_ALUS_PER_STAGE` | 4 | stateful ALUs per stage; every emitted `RegisterAction` holds one for a whole stage (§4.6; `bf-p4c/mau/memories.h:67`, `mau/resource_estimate.h:32`) |
| `FLOW_HASH_LEVEL` | 3 | stages of metadata-init / hash-precompute / hash before the first `RegisterAction` (§4.6) |
| `VOTE_EPILOGUE_STAGES` | 1 | the vote tables always sit one stage past the last classification table (§4.6) |
| `RANGE_WORST_CASE_ENTRY_FRACTION` / `RANGE_WORST_CASE_ROWS_CAP` | 4 / 8 | p4c's compile-time range sizing: one entry in 4 priced at `min(8, 2·nibbles − 1)` rows (§4.2; `bf-p4c/mau/resource_estimate.cpp:1628-1693`, `.h:213-214`, `RangeEntries::preorder`/`postorder`) |
| max `RegisterAction`s per `Register` | 4 | hard compiler-enforced ceiling |
| legal `Register<T,...>` element widths | 1, 8, 16, 32, 64 bits | any other width is a compile error |
| max range-match key width | 19 bits | 20-bit range keys fail to compile |
| TNA timestamp resolution | 48-bit ns | vs. v1model's µs-scale fields |
| this project's decided feature precision | 16-bit | legal native register width, no forced widening |
| Tofino-1 (`TofinoDevice`) pipeline stages | 12 | source-confirmed (`device.h:186`); this project's assumed figure throughout |
| Tofino-2 pipeline stages (variant-dependent) | H=6, M=12, U=20 | internal codename `JBay`; bare `-b tofino2` resolves to **U (20 stages)**, confirmed by real compile diff, not M as source-reading alone suggested |
| TCAM blocks/stage across generations | 24 | `mau_spec.h:88-90`, explicit source comment "correct data for Tof.1 + Tof.2 + Tof.3" — not Tofino-1-specific |

**Range-match physical block capacity by key bit-width** — the compiler's own compile-time,
unknown-values estimate, which **is** what decides blocks (`ranges.compiler_range_rows`, §4.2). The
exact per-value model (`range_entry_count`) answers the separate *deployment* question:

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

**Mechanism index.** The lettered mechanisms A–G that `src/p4model/*.py` and `scripts/tcam_*.py`
cite — including the retracted ones — are catalogued in **Appendix B**, not here. This section
lists only what is still open.

- **`tofino_model` packet-level functional simulation has not been exercised for the TNA-targeted
  generator.** Live control-plane insertion (installing real table entries and reading them back) has
  been validated end-to-end; sending real packets through the simulated pipeline and checking
  classification output has not.
- **Tofino-2's own hardware ceilings — RESOLVED for stages/TCAM, including which variant bare
  `-b tofino2` resolves to. Gateway-count ceiling constant still open.** Found in the real `p4c`
  source: `device.h:186` gives `TofinoDevice` (Tof-1) 12 stages, confirming this document's figure at
  the source; Tofino-2's internal codename `JBay` defaults to 20 stages, with named variants
  `JBayHDevice`=6, `JBayMDevice`=12, `JBayUDevice`=20. `mau_spec.h:88-90` gives
  `Tofino_tcam_rows=12, Tofino_tcam_columns=2` → 24 blocks/stage with an explicit source comment
  "correct data for Tof.1 + Tof.2 + Tof.3" — the 24-blocks-per-stage figure is confirmed shared across
  generations, not Tofino-1-specific.

  **Empirically confirmed by real compiles** (`temp/validate/rf_validate.p4`, 22 tables, 9 logical
  stages, compiled clean against `tofino`, bare `tofino2`, `tofino2m`, `tofino2u`, and `tofino2h`):
  bare `-b tofino2` produces output **byte-for-byte identical** to `-b tofino2u` (max table scope
  stage 19, 20-row `mau.resources.log`) and **structurally different** from `-b tofino2m` (max table
  scope stage 11, 12-row `mau.resources.log`) — bare `-b tofino2` is Tofino2U (20 stages), not
  Tofino2M (12 stages) as this document previously inferred from source reading alone. `-b tofino2h`
  (6-stage ceiling) genuinely fails this same 9-stage program: `error: tofino2h supports up to 6
  stages, using 9`, then `error: Due to errors, no binary will be generated` (exit code 2) — the
  frontend/table-placement stage-count (9) is identical across every target (placement is
  target-agnostic), but the backend assembler step correctly rejects once the device's real physical
  ceiling is exceeded, confirming H/M/U are enforced, not just descriptive labels. Full findings and
  the reproduction recipe: `reviews/open_issues.md` item 4.

  Gateway-count-per-stage **ceiling constant** was not found as an explicit source constant
  (`getGatewaySpec()` is virtual; implementation not chased down) — still open. Per-program gateway
  *usage* (not the ceiling) is visible directly in the real compiles above.
- **The range-match crossbar byte-budget question — RESOLVED (imprecise but unreachable in practice).**
  A real compile sweep (N=1,7,8,9,17 independent range tables, 16-bit width,
  `p4/tofino_spike/t12_experiments/results_rmx_crossbar.csv`) confirms the 8-tables/stage cap
  generalizes to range tables (identical split point to ternary's RM-6 data). Range tables do cost
  ~2x the crossbar xbar-units/table that ternary tables do at the same width (4 vs 2), so reusing
  `TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE` verbatim for the range pool (as `crossbar_stages_needed`,
  `src/p4model/packing.py`, currently does) is not known-conservative in the abstract — the model can
  under-count range_stages once the byte-budget rather than the table-count cap binds, the wrong
  direction for a "never under-count" cost model. Follow-up arithmetic on the same collected data (no
  new compiles needed) found this crossover is unreachable for this project's real generator: every
  range table here keys on exactly one feature fixed at 16 bits (`FEATURE_VALUE_BIT_WIDTH`,
  `src/p4model/program.py`), and the real per-stage crossbar budget (66 units, read from
  `mau.resources.log`'s own percentage column) is far larger than 8 such tables can ever consume (32
  units at 16 bits, 48 units even under a hypothetical stretch to the 19-bit hard SDE ceiling) — the
  8-table cap always binds first. Full arithmetic in `reviews/open_issues.md` item 3.
- **Why a 32-bit PHV container doubles a range key's TCAM words — RESOLVED (still moot).** Mechanism
  found in `resource_estimate.cpp:1628-1653`, class `RangeEntries::preorder`: it counts nibble-halves
  of **container** bytes the field's real PHV placement spans (via `AllocSlice`), not nibbles of the
  field's own logical width. A byte-aligned 16-bit field in an H container gives `range_nibbles=4`
  exactly (reproducing RM-1's measured 206-capacity/7-lines-per-entry); packed non-byte-aligned into a
  wider W container alongside other fields, the same field can straddle more physical container bytes,
  inflating the row estimate — directly explaining the 1→2 block doubling. The `@pa_container_size`
  pragma still makes it moot in practice.
- **The stage model (§4.6) does not cover gateway/action dependencies or PHV-sharing hazards —
  investigated, real hazard confirmed, no stage divergence demonstrated.** A follow-up pass compiled the
  real generator's own output at two scales (M2's 3/1 trees and a larger 8/4-tree build), each with and
  without the `@pa_container_size` pins that normally close off this hazard (pins removed by editing the
  generated `.p4` text, not production code). Removing the pins does produce a genuine cross-feature
  dependency edge in `table_dependency_summary.log` (the `fwd_last_arrival_time_action`/`fwd_iat_max`
  register table picks up an `OUTPUT ANTI_NEXT_TABLE_DATA`/`ANTI_TABLE_READ ANTI_ACTION_READ
  ANTI_NEXT_TABLE_DATA` dependency on the unrelated `flow_iat_max_action`/`flow_iat_mean_action` tables
  via shared PHV container `W2`) — so the gap is not imaginary. But in all four compiles the range-match
  tables' real stage stayed 5/5/5/6, identical to the model's prediction and the original M2
  measurement; the compiler had enough slack in the preceding stages to absorb the extra edge for free.
  A further push to 20/10 trees to test whether more PHV pressure could turn this into an actual stage
  delay was abandoned — the Python-side codeword/table-entry generation itself (pre-existing, unrelated
  code) did not finish within ~14.5 minutes and was killed before reaching `p4c`. Full detail:
  `reviews/open_issues.md` item 5. **Note the scope statement this bullet used to carry is out of
  date:** since 2026-09, §4.6 models more than register-chain depth — it also models the
  4-stateful-ALU-per-stage width limit, the per-gate-block placement-cursor floor, and gated-block
  interiors (Appendix B "Mechanism B").
- **The "2-field costs more than 3/4-field" ternary TCAM anomaly — root cause traced, and the two
  width-predicate rules this section used to assert are both RETRACTED.** The physical mechanism is real
  and is now modelled (§4.1.1, §4.1.2): the mandatory 2-bit per-row version field can ride for free only
  in a crossbar midbyte nibble, and when a key's own bytes consume every midbyte its groups reach, the
  packer allocates an entire new, near-empty TCAM block for those 2 bits alone — confirmed concretely in
  a real compile where that block uses only 2 of its 44 bits. Root cause of the fragmentation that
  triggers it in unpinned programs: an earlier, unrelated PHV-container-allocation pass, confirmed in
  `phv_allocation_summary_0.log` (Appendix B "Mechanism A"; fixed in the generator with `@pa_solitary`).
  **The two predictive rules formerly stated here as exact were falsified against this project's own
  data, and the clause-list rule that replaced them was itself superseded on 2026-09-21 when p4c's
  assembly falsified its geometry; Appendix B "Mechanism G" states each of the three once, with the
  counterexamples.** Not filed upstream — `p4lang/p4c`'s issue tracker has nothing on this and the relevant code is unchanged on
  current `main`; a minimal reproduction and draft bug-report text live in
  `reviews/github_issue_tcam_version_bit_packing.md`, not filed.
- **The range-match block-boundary fill margin — RESOLVED.** Rather than a flat safety-margin constant
  (shown in §4.2 to have no universally-correct value), the actual fix was applied at the insertion
  point: `p4/deploy_table_entries.py` now sorts each range table's entries by descending physical
  row-cost before installing, reproducing the favorable insertion order §4.2/Task 4c proved reaches
  full nominal 512-row capacity. Verified against the live `tofino_model`/`bf_switchd` control-plane
  path (this project has no physical ASIC — see the scope note at the top) with a heterogeneous
  213-entry set
  (`.superpowers/sdd/task-8-sort-order-verify-report.md`): sorted and a realistic non-adversarial
  unsorted order both reached exactly 512/512 in this test (a tie — the unsorted arm wasn't
  adversarial enough to lose rows), confirming the sort is never worse, consistent with the
  `pipe_mgr_tcam_find_next_free` contiguity mechanism identified in §4.2.

---

## Appendix A. The half-byte credit, explained by crossbar lanes

The mechanism behind §4.1.1's isolation credit (`tables.tail_is_isolatable`) and §4.3's lane-based
sharing price (`src/p4model/lanes.py`, C5) is one and the same rule, read directly off p4c's crossbar
sizing code rather than fitted. This appendix states it once, with the two worked examples that pin
it down. Source: `bf-p4c/mau/tofino/input_xbar.cpp` (`align_flags` l.461, `free_mid_bytes` l.879, the
`allocTable` loop l.1361-1390 — midbytes are allocated BEFORE groups), `input_xbar.h:51-52` (5 bytes
per group, 11 bytes per big group-pair). Evidence: `prog.bfa` of `results/tcam_phv_slice_sweep`
probes `w019`/`w027`, and traced recompiles of both.

**A.1 Slots, groups, midbytes, lanes.** Every crossbar slot has a lane, `slot number mod 4`:

```
slot #   0  1  2  3  4    5    6  7  8  9 10   11 12 13 14 15   16   17 18 19 20 21  ...  27  ...  38
        [   group 0    ] [mb0] [   group 1   ] [   group 2    ] [mb1] [   group 3    ]  ... [mb2] ... [mb3]
lane     0  1  2  3  0    1    2  3  0  1  2    3  0  1  2  3    0    1  2  3  0  1         3        2
```

A group is 5 slots — one of each lane, plus one EXTRA in whichever lane it starts on (group 0 starts
at lane 0, group 1 at lane 2, group 2 at lane 3, group 3 at lane 1, cycling every 4 groups). Any two
groups therefore supply 2 slots per lane, plus 2 extras in lanes the allocator chooses. Midbyte lanes
cycle `1, 0, 3, 2, 1, 0` across the stage's 6 midbytes.

**A.2 One TCAM block** = 1 group (5 whole-byte slots) + one nibble of a midbyte. Two blocks = 10
whole-byte slots + one midbyte, split `[version nibble | spare nibble]`. **The 2-bit `--version--`
field always takes a whole nibble**, never fewer bits, regardless of how few of its 2 bits are live —
this is the whole reason a key with no free nibble anywhere it reaches pays an entire extra block for
2 bits (§4.1.2 shows the physical anomaly this produces).

**A.3 Byte lanes.** Byte `k` of a 32-bit (`W`) container may sit only in a lane-`k` slot; a 16-bit
(`H`) container's bytes only at matching parity (`k mod 2`); an 8-bit (`B`) container's byte anywhere.
This is `src/p4model/lanes.py`'s `_tail_allowed`/`_fits`, and it is the reason a key's crossbar price
depends on ITS REAL PHV CONTAINER LAYOUT, not just its byte count.

**A.4 The rule.** An 11-byte key (three `W`-container bytes each needing a full lane, two more in a
fourth lane — lane counts `(3, 3, 3, 2)`) fits 2 blocks only if its 3-4-bit tail takes the spare
midbyte nibble AND the other 10 bytes still fit the two groups' lane-by-lane capacity (2 slots/lane +
2 extras in lanes of the allocator's choosing). With only 2 extras available, at most two lanes may
hold 3 bytes each; a third lane needing 3 is impossible for any 2-group choice. The tail nibble only
helps when removing it relieves an over-full lane — it cannot rescue a key more than one lane over.

**A.5 The two probes, at IDENTICAL lane counts.** Both keys are 11 bytes across three `W` containers,
both `(3, 3, 3, 2)` before the tail — and they diverge purely on which lane loses its tail:

```
 w027 (27 + 56 bits)                          w019 (19 + 64 bits)
        lane: 0    1    2    3                        lane: 0    1    2    3
  W0        [a1] [a1] [a1] [a1]                 W0        [a1] [a1] [a1] [a1]
  W1        [a0] [a0] [a0] [t ]  tail lane 3    W1        [a1] [a1] [a1] [a1]
  W2        [a1] [a1] [a1]                      W2        [a0] [a0] [t ]        tail lane 2
```

`w019`: remove the tail → lane counts `(3, 3, 2, 2)` → the 2 extras needed sit in lanes 0 and 1 →
groups 0 + 3 supply exactly that; the tail lands in midbyte 3 (lane 2) → **2 blocks, isolatable.**

`w027`: remove the tail → lane counts `(3, 3, 3, 1)` → 3 extras needed, only 2 exist → impossible for
ANY 2-group choice. p4c's real attempt (tail in midbyte 2, groups 0 + 1 → supply `(3, 2, 3, 2)`):
one lane-1 byte is left homeless while a lane-3 slot sits empty (`prog.bfa`: "free bytes placed 4" in
group 1). Its fallback: park a WHOLE lane-1 byte in midbyte 0 instead of a tail nibble → supply
`(3, 2, 3, 2)` now fits exactly — but a whole byte fills BOTH nibbles of that midbyte, leaving none
for the version field → a version-only block (`- { byte_config: 3 }`, no group attached) → **3
blocks, not isolatable.**

**A.6 How the model relates.** `tables.tail_is_isolatable` (§4.1.1) is a WIDTH-ONLY proxy for this
rule: it infers the tail's lane from `field_bits mod 32` alone, assuming the field starts at a
container boundary (27 → byte 3, not isolatable; 19 → byte 2, single container, isolatable) — rather
than walking a compile's real PHV layout the way this appendix's mechanism does. A lane checker built
from real PHV layouts agrees with the width-only proxy on **74/74** real design keys, and is exact
(**57/57**) on synthetic probe families where the proxy is only 48/57 (always conservative — errs
toward NOT isolatable, never the other way). `src/p4model/lanes.py` (§4.3, C5) implements the full
mechanism, not the width-only proxy, for the one place it matters at compile-prediction time: pricing
a LATER key's leftover fit into a crowded stage, where predicting the real PHV layout from widths
alone (`lanes.layout`/`relaxed_layout`, audit §7.4) is accurate on 148/148 real keys once the relaxed
fallback is included.

---

## Appendix B. Mechanism index and investigation record

The effects this project named by letter while calibrating the cost model. `src/p4model/*.py` and
`scripts/tcam_*.py` cite these letters directly; this appendix is what they cite. Each entry says what
the effect is, **where it lives today** (modelled here, fixed in the generator, or retracted), and what
evidence backs it.

| # | Effect | Status | Implemented / recorded in | Section |
|---|---|---|---|---|
| **A** | PHV parks unrelated program state at the bottom of a shared container, leaving a ternary key's remainder straddling a nibble boundary and costing crossbar bytes (and sometimes a block) | **Fixed in the GENERATOR, not modelled** | `build_p4_script` emits `@pa_solitary` on every `class_tree_*`/`code_*` field | §4.1.2, §4.6 |
| **B** | A stage the placer spends wholly INSIDE a gated register block can hold no table from the outer sequence, however empty it is | **Current** | `registers.gated_block_interior_stages` → `packing.crossbar_stages_needed(unavailable_stages=…)` | §4.6 |
| **C** | A stage's 24 TCAM blocks are 12 rows × 2 COLUMNS, and a table chains its blocks down one column | **Current** (with a row-parity addendum investigated 2026-09-21 and found **vacuous**) | `packing.fits_two_columns`, `packing._stage_shards` | §4.3 |
| **D** | Ternary blocks are charged from crossbar BYTES per key FIELD, not from raw codeword bits | **Current** | `tables.codeword_fields_to_bytes` → `tables.crossbar_capacity` / `tables.codeword_to_blocks` | §4.1 |
| **E** | p4c sizes a range table at COMPILE time from the declared entry count, pricing a quarter of them at the key's worst-case row count | **Current** | `ranges.compiler_range_rows` | §4.2 |
| **F** | — | **Does not exist** | no effect was ever assigned to this letter, and nothing in `src/` or `scripts/` cites it (verified by grep 2026-09-15); a 2026-09-06 plan records it as used briefly in a probe script and then dropped | — |
| **G** | The extra TCAM block a ternary key pays because no crossbar midbyte nibble is left for the mandatory 2-bit `--version--` field | **Phenomenon current; FOUR rules under this heading are now retired — G1, G2, the `version_block_penalty` clause list (2026-09-21), and the fitted 58-byte crowded-stage margin plus its any-order-fit refinement (2026-09-28)** | the per-table charge is folded into `tables.crossbar_capacity`'s ladder; the placement half is now `packing.crossbar_stages_needed`'s ordered crossbar-lane simulation, `src/p4model/lanes.py` (C5) | §4.1.1, §4.1.2, §4.3, Appendix A |

*Housekeeping:* the letters are the only labelling series the current code cites. There is **no
Mechanism F**. A 2026-09-06 restructuring plan also mentions an earlier numbered series (1/2/3); no
citation of it survives anywhere in `src/` or `scripts/`, so it is not reconstructed here.

---

### Mechanism A — PHV container sharing

Real, reproduced, and **out of scope for `src/p4model`**: it is a property of the whole compiled program
(what unrelated values happened to sit at the bottom of a container), not of any table the model can
see, so it cannot be predicted from a design description. This project removed it at the source instead,
with `@pa_solitary` on every field the classification tables write. Evidence and the `bypass_egress`
demonstration are in §4.1.2; the range-pool case, and the compile pairs with and without
`@pa_container_size`, are in `reviews/open_issues.md` items 16 and 5.

Consequence for reading old numbers: **pre-`@pa_solitary` calibration figures describe a different
generator.** Figures from that era (mean error 0.78 stages, `stage_depth` exact on 9 of 19) are not
comparable with the current ones in §4.1.3 and §4.6, and their remaining under-predictions were this
mechanism.

### Mechanism B — gated-block interiors

Stated in full in §4.6. The short form: p4c's placer has one work-list cursor, the gated register blocks
are emitted before every match table, and every stage strictly *between* where the cursor enters and
leaves a gated block is unreachable for the outer sequence. Exact on the 5 of 18 calibration rows whose
range pool has an occupancy hole, and silent on the other 13.

**This is the record of the dead end it ended.** Before it, two separate investigation passes looked for
a *capacity* rule that would explain those empty stages — some limit on how many range tables a stage
may hold. **No such limit exists. There is no range-pool fill limit; do not look for one.** The stages
in question sit at 0/24 TCAM blocks and 0/66 crossbar bytes: they are forbidden, not full.

### Mechanism C — the 2×12 TCAM column geometry

Stated in §4.3. The investigation record worth keeping is the **near-miss**: the rule was once written
off as "refuted" because a per-width shortcut (`2 * floor(12 / w)`) was applied to stages holding mixed
widths. The real packing is far more permissive than that shortcut — `(7+5 | 6+6)` places four tables
the shortcut rejects — so the refutation was an artifact of the test, not of the rule.
`fits_two_columns` therefore does an exact subset-sum instead. Confirmed afterwards by 19 real compiles
(`scripts/tcam_column_sweep.py`; `results/tcam_column_sweep.csv`, 16 points, and
`results/tcam_column_sweep_wide.csv`, 3 points at 14/16/24 blocks), and consistent with all 18
calibration rows' committed placements.

#### Mechanism C, addendum: row parity investigated and found vacuous (2026-09-21)

**Decision: real, hardware-accurate, and provably vacuous under this packer's free-ordering model. No
code change.** `fits_two_columns` is right to model column LOADS and ignore which physical row a table
starts on. The permanent regression test is
`tests/test_p4model_guards.py::test_row_parity_never_rejects_a_stage_that_fits_by_size`.

The rule under test, read from `Memories::find_ternary_stretch`
(`bf-p4c/mau/tofino/memories.cpp:1761-1797`) and adopted in the rewrite design's two-line form: **a run
of even height may only start on an even row; a run of odd height may start anywhere.** Rows `2i` and
`2i+1` share the half-byte selector, which is why. It refines Mechanism C's column geometry, which is
why it is recorded here.

Four independent checks, all of which had to agree before the finding was closed.

**1. Enumeration — parity-feasible-under-some-order ≡ size-feasible.** Exhaustive over every multiset of
block heights from `1..12` in two 12-row columns: **8 618 multisets checked, 0 disagreements** between
`fits_two_columns(heights)` and an exhaustive parity-aware search over all orders and column
assignments. (Supersets of a size-infeasible multiset are pruned; both tests are monotone in the
multiset, so no case is lost.)

The structural reason — which is why this is not a coincidence of this grid size — is one line: **place
every even-height table first.** Prefix sums of even numbers stay even, so every even-height run
naturally begins on an even row, and the odd-height runs may then go anywhere. Any multiset that fits by
SIZE therefore fits under parity for SOME order; the converse is immediate, since parity only removes
options.

*Note for anyone tempted to "just add parity to the existing DP":* that DP walks heights in **descending**
order, and a parity-aware DP locked to that order would wrongly reject `{5, 4}` in one column — 5 first
leaves the next free row at 5, which is odd, so the even-height 4 cannot start there; take the 4 first
and both fit (rows 0–3, then 4–8), which is what real p4c does. **Free ordering is exactly what makes
the constraint disappear, and a fixed order reintroduces it as an artifact.**

**2. Archive — the rule holds without exception, and costs almost nothing.** Read with
`scripts/tcam_stretch_sweep.committed_tcam_grid` / `.spans`. Two survey scopes, kept separate because
they are separate corpora:

- the **20-compile** archive the rewrite design itself cites (`results/compiler_calibration_v6`'s 19 plus
  `independent_low_sd9`): **317 committed runs, 103 starting on an odd row, 0 even-height odd starts** —
  reproducing the design's own counts exactly and independently;
- widened to **all 27 archived compiles** (v6 + `compiler_calibration_extra`): **465 runs, 151 odd
  starts, still 0 even-height odd starts**.

Wasted rows, over the **183 committed columns in 138 stages** of that widened survey: exactly **two**
columns contain an interior hole, and both are the parity rule visibly at work —

> `independent_high_sd6` stage 9 and `independent_high_sd7` stage 9, column 0: a 3-high `ddos_0` run at
> rows 0–2, then a **2-high** (even) `app_0` run that cannot start on odd row 3 and takes rows 4–5. Row
> 3 is wasted.

Both columns use 6 of 12 rows, so neither hole spilled anything. That is the only wasted-row signature
p4c leaves anywhere in the archive.

**3. Counterfactual — parity replayed in p4c's OWN placement order.** For every archived column, tables
were taken in p4c's actual bottom-up order and re-packed tight under the parity rule. **Zero spills:**
no column exceeded 12 rows, and only **2 of 183** columns needed a parity gap at all — the same two
above. So even without the free-ordering argument, the rule never binds on real data.

**4. Exposure — where a wasted row could possibly flip a verdict.** Column-load histogram over the 183
archived columns (rows used → count): `1:10, 2:19, 3:20, 4:11, 5:5, 6:5, 7:30, 8:33, 9:11, 10:12, 11:5,
12:22`. **27 columns, across 20 of the 138 stages, sit at 11 or 12 of 12 rows** — 22 at 12, 5 at 11.
Those are the only places a single wasted row could change a verdict. None of them contains a parity gap
(the only two gaps are in 6-row columns), and every one of the 138 stages' height multisets was
re-checked individually: `fits_two_columns` agrees with parity-feasible-under-some-order on all 138
(39 distinct multisets), **0 disagreements**.

**Conclusion.** Row parity is a real hardware rule, confirmed 317/317 in the archive, whose only possible
cost is at most one wasted row per column per stage — and under free ordering it is provably never paid.
Modelling it would add a row-level concept to a packer that has none, and would change no verdict on any
data this project has. This is a **closed** investigation, not an open question.

### Mechanism D — crossbar bytes, not codeword bits

Stated in §4.1. The record: `ceil((bits + 4) / 44)` was accidentally right for years because it
coincides with the byte rule on **one dense key field**, and this generator's keys are one field *per
feature*. The observed byte→block ladder (4→1, 11→3, 16→3, 20→4, 26→5, 33→6, 37→7, 41→8, 49→9, 52→10,
60→11) is single-valued across **144 compiled classification tables** from three compile eras. Until
2026-09-21 it was quoted as `4→1, 11→2, …` and was single-valued only once Mechanism G's version charge
was peeled off and added back, since the three 11-byte keys in that corpus really cost 3 blocks. Under
`tables.crossbar_capacity` the version half-byte is inside the ladder, `capacity(2) = 10 < 11`, and 11→3
falls out directly — the map is single-valued with **no separate version term at all**. The one row the
published `S = 0` form misses is 33→6, which it prices at 7; the measured isolation credit recovers it
(§4.1.1).

The superseded identity `n_features + n_trees · ceil((codeword + 4) / 44)` under-counted and should not
be reused.

### Mechanism E — compile-time range sizing

Stated in §4.2. The record: the natural move — feed the *driver's* exact per-value expansion
(`range_entry_count`) into the block count — is **wrong**, because blocks are decided at compile time
when the compiler has not seen a single interval bound. Measured cost of getting this backwards: a
478-entry table priced at 1 block against p4c's committed 3. The two quantities now live in separate
functions (`compiler_range_rows` for blocks, `range_entry_count` for deployment), with
`range_deployment_overflow` as the explicit bridge.

An older note in this repo quoted "207 range entries per block". That was an averaged figure and is
superseded by `compiler_range_rows`; the correct figure at this project's 16-bit key width is **206
declared intervals per block**.

### Mechanism G — the version block, and four superseded placement rules

**Current rule (§4.1.1, §4.3), as of 2026-09-28.** The phenomenon is unchanged: a ternary key pays one
extra TCAM block when no crossbar nibble is left free for the mandatory 2-bit `--version--` field. It is
no longer a separate *term*. The per-table half is folded into the block ladder itself —
`tables.crossbar_capacity(g) = 5g + floor((g-1)/2)` withholds one half-byte per key, and
`tables.codeword_to_blocks` takes the smallest `g` whose capacity covers the key's `B` crossbar bytes,
with a capped credit for one nibble-clean field whose tail p4c can isolate. Exact on 100/100 archived
classification tables; the published `S = 0` form is exact on 92 and over by +1 on the other 8. The
placement half has been rewritten twice since: a **per-key saturation margin** (2026-09-21 to
2026-09-25: +1 to a non-first key whose standalone price was exactly saturated, in any shared stage),
then a **fitted 58/62-byte crowded-stage margin** (2026-09-25 to 2026-09-28: +1 flat to every table
of a non-first key once two different keys shared more than 58 combined bytes, refused above 62), and
now — as of C5, 2026-09-28 — an **ordered crossbar-lane stage simulation**
(`src/p4model/lanes.py`, Appendix A) that prices a later key by literally simulating which crossbar
slots the earlier key left free, rather than margining against not knowing. Only the 62-byte figure
survives from the middle rule, repurposed as a refusal safety net rather than a price (§4.3).

**G3 — SUPERSEDED 2026-09-21: the `version_block_penalty` clause list.** Between 2026-09-07 and this
date, Mechanism G was implemented as `tables.version_block_penalty(field_bit_widths, start_group)`, a
four-clause ledger (a)–(d) over a **consecutive run of crossbar groups starting at an offset**, with
midbyte `i` owned exclusively by the group pair `2i`/`2i+1`, charged into a placement as
`tables.version_block_delta`. It was exact on 100/100 archived tables and on two out-of-sample sets, so
it is not retracted for being *wrong on its own corpus* — it is superseded because **its structural
premises are false**, read directly out of p4c's assembly (`prog.bfa`): a block pairs with **any** of a
stage's 6 byte groups; a key's groups need **not** be consecutive (measured `{0,1,3,4}`, `{0,3}`,
`{5..11}`); and groups are **shared** between tables keying the same bytes. Two consequences of those
false premises were measurable defects in their own right: `version_block_delta` could be **negative**,
discounting a key and making the packer charge fewer blocks than the tables' own declared sum; and
clause (d)'s contiguous-ordering reachability over-fired on 24 observations. The functions
(`crossbar_groups_needed`, `_run_capacity`, `_full_midbytes`, `_midbyte_slot_indices`,
`_clean_byte_can_reach_a_midbyte`, `version_block_penalty`, `version_block_delta`) are **deleted**.
Two caveats that used to be filed as open questions against this rule are therefore **moot rather than
resolved**, and should not be reinstated as open items: the unmeasured "shifted-offset arm", and the
"monotone by construction" safety claim.

The real effect the offset term was chasing — the same 49-byte ragged key costing 9 blocks alone and 10
beside a different key — is real, and survives today as §4.3's lane simulation pricing that key against
the actual crossbar slots left free by whichever key p4c's pinned placement order puts first (audit C5,
Appendix A) — no longer as a flat one-sided margin (the fitted 58/62-byte crowded-stage rule that sat
between this offset term and the current mechanism from 2026-09-25 to 2026-09-28 is retired; §4.3
has its record).

Two *earlier* rules were asserted under this heading before G3 and are **both retracted**. Neither is
implemented anywhere, and neither should be reinstated.

**G1 — RETRACTED: "anomalies occur whenever `field_width mod 8 == 4` with 2 or more such fields."**
Asserted as "the exact predictive rule" on the strength of 6 data points. It has **at least five
counterexamples in this project's own data**: 2×44, 2×36, 3×20, 3×4 and 6×4 all satisfy the predicate
and are not anomalous. The 2×20-bit "minimal repro" it was built on is not the same phenomenon at all —
two `bit<20>` fields cost 6 crossbar bytes and a group has only 5 private slots, while a single
`bit<40>` field covering the same 40 match bits costs 5 bytes and fits in one block. The `6×28` case
listed alongside it as anomalous is also not anomalous: it costs 23 crossbar bytes, which genuinely
needs 5 groups. The predicate was a coincidence of field widths; the real variable is crossbar-byte
saturation.

**G2 — RETRACTED: "+1 block to a RAGGED key sitting at an ODD crossbar group offset."**
Retracted 2026-09-07 on `resources.json`'s own per-table numbers. Scored against the data it was fitted
to, the predicate **fired on 5 stages where p4c charged nothing**, and on the single stage where p4c
*did* charge a block it named the **wrong table**: `independent_low_sd5`'s three ddos trees are
penalised at group offset **0** while the app key sits free at offset 3 — the exact inverse of the
parity predicate, which reached the right stage total (12) only by arithmetic coincidence. Replacing it
took `usage.blocks` from 12/17 to **17/17 exact**.

The offset still matters — but through **geometry, not parity**. `scripts/tcam_stretch_sweep.py`: the
same 49-byte ragged key costs 9 blocks at offset 0 and 10 at offset 3, while its **solid** control costs
9 at both, because the solid key has no nibble-clean byte to consume the half-midbyte. That is what
survives today, not as a parity rule but as **geometry priced by simulation**: §4.3's crossbar-lane
stage simulation (`src/p4model/lanes.py`, C5, 2026-09-28) prices exactly this effect by computing which
lanes the second key's bytes can legally reach given what the first key left free, superseding the
fitted 58/62-byte crowded-stage margin that sat here from 2026-09-25 to 2026-09-28, which in turn
superseded clause (c)/(d) of §4.1.1's own now-deleted clause list (2026-09-21 rewrite; there is no
clause list left in the per-table price at all) -- so the geometry has always lived entirely in
placement, not in the per-table ledger, only the placement MECHANISM has changed twice since.

**Composition bug caught on the way in.** A 12-compile out-of-sample sweep
(`scripts/tcam_version_sweep.py`, `results/tcam_version_sweep.csv`) confirmed the replacement rule and
exposed one error the archived corpus could not: the block count must be
`max(bit_arm, crossbar + penalty)`, **not** `max(bit_arm, crossbar) + penalty`, because the bit arm's
`+4` overhead bits *are* the version/valid field. Measured on three geometries the archive never
covered: a solid 11-byte key is 3 TCAMs not 4, 22 bytes 5 not 6, 33 bytes 7 not 8.

**`joint` designs were never affected by any of this** — every tree keys one shared field set at offset
0, and no `joint` row in the calibration has a single penalised table.

---

### Open / unverified, last updated 2026-09-28

*Three items this list carried on 2026-09-15 are now **CLOSED**, and are recorded here as closed so
nobody reopens them:*

- ~~**`version_block_delta` can be NEGATIVE, and the packer charges it**~~ — **CLOSED 2026-09-21 by
  deletion.** The composed `codeword_to_blocks` was not monotone in `start_group`, so a key could price
  lower at a shifted offset than at 0 and `packing.charged` subtracted the difference from the stage
  total (reachable with only two distinct keys, i.e. by ordinary `disjoint` designs). There is no
  `start_group` and no delta any more: the per-table price does not depend on placement at all, and the
  placement term is a one-sided margin that can only *add*. The accompanying open question — "is the
  discount an artifact or real physics?" — is moot; the geometry it rested on was false (Mechanism G3).
- ~~**The shifted-offset arm of `crossbar_groups_needed` is geometry, not measurement**~~ — **CLOSED
  2026-09-21: the arm is deleted.** Note this is *moot*, not *resolved*: the 12-compile sweep that went
  looking for a usable odd-offset point still returned zero usable points, and nobody ever measured the
  arm. It simply no longer exists to be measured.
- ~~**Row parity might need modelling in `fits_two_columns`**~~ — **CLOSED 2026-09-21: real, and
  provably vacuous** under this packer's free ordering. Four-part evidence in Appendix B "Mechanism C,
  addendum".

*Still open:*

- ~~**The near-cap crossbar group budget (spec "F5")**~~ — **CLOSED 2026-09-25** by the (since
  superseded) crowded-stage rule; the refusal it introduced above 62 combined bytes survives unchanged
  under C5. `dsp41`/`dsp42` still sit at a placement the packer refuses; 0 under-predictions remain on
  any source.
- ~~**The fitted crowded-stage rule over-predicts where p4c happens to find room**~~ — **the rule
  itself is CLOSED 2026-09-28**, replaced by C5's crossbar-lane stage simulation (§4.3, Appendix A),
  which prices a later key by simulation rather than a flat margin. The over-prediction risk this item
  used to describe (some real stages at 59-62 shared bytes pay nothing, so a margin prices them one
  block/stage too deep) is retired along with the rule that caused it — the pre-pragma replay of the
  fitted, held-out and adversarial archives is now 19/19, 8/8 and 16/16 exact on `stage_depth`, and
  17/17, 5/5 and 15/16 on `blocks` (§4.1.3), with the one remaining `blocks` miss an artifact of an
  UNPINNED replay, not a pricing error. **A new, narrower risk replaces it**, next item.
- **The 62-byte refusal is still a safety net, not a proof.** C5's stage simulation can legally PRICE
  a later key in stages the model still refuses outright above 62 combined bytes (`M150_k7_s11`, §4.3)
  because p4c's own greedy allocator is observed to give up on a placement the simulation can still
  compute. This remains a one-sided margin against an allocator behaviour that is not itself simulated,
  same as before C5 — only the priced side of the rule (below 62 bytes) moved from a fitted margin to a
  mechanism.
- **A genuine range-pool PLACEMENT-ORDER gap, found by the C6 frozen held-out batch (2026-09-28).**
  `heldout_independent_M150_k14_s13`: `stage_depth` predicted 11, p4c places at 12. Not a pricing bug —
  replaying p4c's own committed block counts through this model's placement logic still gives 11 — and
  not new: replaying 4 historical commits' code against the same frozen design reproduces the same
  1-stage gap under every one of them, so it predates this entire 2026-09-27/28 rewrite. The mechanism:
  p4c spills one range table to a later stage than the model's eager bin-packing does (§4.6's
  "Placement, not optimisation"), and under C1 a classification tree then waits one stage longer than
  predicted for that table. Out of scope for this run; see §4.6's "Current accuracy" for the full
  account. This is a DIFFERENT gap from the next item, which is about byte BUDGET, not fill ORDER.
- **The isolation term is not predictable from a key's shape for keys of 3+ fields.** 44 targeted
  compiles found no feature separating the pays/free outcomes (§4.1.1). This is now understood to be a
  narrow residual of the width-only proxy specifically (Appendix A gives the full, predictable
  mechanism once the real PHV layout is known — C3, 2026-09-28), not evidence that the outcome itself
  is unpredictable. Accepted as a documented bias in the proxy, with a stated sample size; not to be
  fitted further.
- **`_stage_shards`' wider-than-a-column path rests on a single confirming compile, not the
  calibration corpus.** No table in the 19-row archive exceeds 12 blocks, so `scripts/tcam_stage_shard_probe.py`'s
  one 13-block probe (places as 12|1 in one stage, charged 13, matching the finding-1.5a rounding fix) is
  the only hardware evidence for this path, not a broad validation. It is the one cost-lowering change in
  this work.
- **The range pool's crossbar byte budget is not known-conservative** (§4.3). Range tables cost ~2× the
  crossbar units per byte that ternary tables do; the crossover is unreachable for this generator but
  was never pinned down. (Distinct from the C6-found placement-order gap above: this item is about
  BUDGET, that one is about FILL ORDER within a budget neither item disputes.)
- **`tofino_model` packet-level functional simulation has not been exercised for the TNA generator.**
  Live control-plane insertion is validated end-to-end; sending real packets through the simulated
  pipeline and checking classification output is not.
- **A gateway-count-per-stage ceiling constant was not found in source** (`getGatewaySpec()` is virtual;
  the implementation was not chased down). Per-program gateway *usage* is visible in the real compiles.
- **The width-based PHV layout rule (§4.3, `lanes.layout`) is p4c-version-specific.** It was surveyed
  against 1130 real fields on p4c 9.13.4; re-run `phv_layout_survey.py` after any compiler upgrade
  before trusting the lane simulation's leftover prices again.
- **Everything here is a compiler-and-simulator result. There is no physical Tofino ASIC in this
  project**, and nothing in this document is a latency or throughput measurement.
