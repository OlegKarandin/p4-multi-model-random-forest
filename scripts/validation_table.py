"""Validation table: this project's resource-model predictions (recomputed
with CURRENT code) against real Tofino p4c ground truth.

PRIMARY GATE -- THE PINNED ARCHIVE (results/compiler_calibration_pinned/,
ground truth results/compiler_calibration_pinned.csv, written by
scripts/build_pinned_calibration_csv.py). 43 designs generated with the
@placement_priority / @pa_no_overlay pragmas the generator now emits (audit
C5; "arm D" of reviews/model_audit_2026-09-27.md §7.3) and compiled one at a
time. The pragmas pin the order p4c places the classification trees in --
the order packing.crossbar_stages_needed's ordered stage simulation replays --
so this is the only archive whose compiles match what the generator produces
today. Every design is replayed END TO END from its generated program
(scripts/p4_artifact_replay.replay_design): the model's own per-table prices,
packed by the model's own packer. CURRENT STANDING (2026-09-28):
stage_depth 42/43, blocks 38/38 (5 designs never got a TCAM allocation: they
need more than 12 stages). The one miss, KNOWN_PINNED_MISSES below, is
independent_high_sd12: predicted 13, p4c 14 -- the one design where the
pragma itself cost p4c a stage (13 without it); infeasible either way. It is
an UNDER-prediction and is printed as one, deliberately un-silenced. The
same numbers, design for design, as the audit's prototype
(reviews/model_audit_scratch/proto_model.py --c1, FILL=lane).

PRE-PRAGMA, INFORMATIONAL. Three older archives were compiled WITHOUT the
pragmas, so p4c chose its own tree order there, which the model no longer
assumes. They are still reported, for continuity, but are not the gate:

  * FITTED (results/compiler_calibration_v6.csv, 19 rows). Predictions are
    recomputed by replaying tests/fixtures/resource_model_golden.json through
    src/p4model/usage.py's assemble_usage (no sklearn, no campaign backup).
    CAPTION CONSTRAINT: these are 19 ARCHIVED compiles re-evaluated against
    current model code, not 19 fresh compiles. stage_depth 19/19, blocks
    17/17 (2026-09-28).
  * HELD OUT (results/compiler_calibration_extra/, 8 designs never used to
    fit the model): stage_depth 8/8, blocks 5/5.
  * ADVERSARIAL (results/tcam_margin_screen/, 16 real campaign designs, USED
    TO CHOOSE the retired 58/62 crowded-stage margin -- not held out):
    stage_depth 16/16, blocks 15/16 -- margin_independent_M150_k5_s12 reads
    66 against p4c's 68, an UNDER-prediction of 2 blocks. Unpinned, p4c
    served that design's keys in an order that cost 2 more blocks; its pinned
    compile (the primary gate) costs 66, exactly as predicted (audit §7.3:
    the pragma made it 2 blocks cheaper). Printed, not silenced.

WHY THE CSV'S OWN PREDICTION COLUMNS ARE NOT USED. results/
compiler_calibration_v6.csv carries its own stage_depth/blocks columns, but
those were written by whatever model code was live the moment each row was
collected -- not necessarily today's. independent_low_sd5 is the documented
case: its CSV blocks=13 predates the StagePlan.blocks fix; recomputing from
the golden fixture gives 16, matching tcam_real=16
(results/compiler_calibration_verify.csv). Only stages_real/tcam_real/
sram_real/map_ram_real (p4c's own numbers, which cannot go stale) are used as
ground truth. The CSV's own stage_depth/blocks columns are read ONLY by the
drift check, whose job is precisely to catch a stale-column case like this
one: it prints independent_high_sd12's stage_depth (13, exact, since per-task
tree readiness) against the CSV's stale 14, deliberately un-silenced.

Run (from the repository root):
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/validation_table.py
"""
import argparse
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
TESTS_DIR = os.path.join(ROOT, "tests")
if TESTS_DIR not in sys.path:
    sys.path.insert(0, TESTS_DIR)

import pandas as pd  # noqa: E402

from test_resource_model_golden import rebuild_pool  # noqa: E402

from src.p4model.usage import assemble_usage  # noqa: E402
from scripts.p4_artifact_replay import replay_design  # noqa: E402

DEFAULT_FIXTURE = os.path.join(ROOT, "tests", "fixtures", "resource_model_golden.json")
DEFAULT_CSV = os.path.join(ROOT, "results", "compiler_calibration_v6.csv")
# PRIMARY GATE: the 43 pragma'd compiles (see the module docstring). Ground
# truth only; predictions are recomputed from p4_src/ every run.
PINNED_ROOT = os.path.join(ROOT, "results", "compiler_calibration_pinned")
PINNED_CSV = os.path.join(ROOT, "results", "compiler_calibration_pinned.csv")
# The pinned archive's one known miss, printed with its reason rather than
# silenced: an UNDER-prediction on a design p4c cannot fit anyway.
KNOWN_PINNED_MISSES = {
    "independent_high_sd12": (
        "stage_depth 13 vs p4c 14 (UNDER by 1, design infeasible either way: "
        "> 12 stages). The one design where the @pa_no_overlay/"
        "@placement_priority pragmas themselves cost p4c a stage -- 13 without "
        "them (reviews/model_audit_2026-09-27.md §7.3/§7.6)"),
    "margin_independent_M150_k7_s11": (
        "stage_depth 11 vs p4c 12 (UNDER by 1) since the 62-byte refusal was "
        "retired: this archive predates the code_* layout pins, and unpinned "
        "p4c split the ddos key's fields into whole W containers and moved the "
        "app key a stage. Its pinned compile (results/"
        "compiler_calibration_pragmas_2026_09_29) is 11, exact"),
}
# PRE-PRAGMA, INFORMATIONAL (see the module docstring). HELD OUT: 8 real
# compiles never used to fit or re-tune the model. Its CSV's blocks/
# stage_depth columns are stale predictions from the model that was live when
# the rows were collected -- only the *_real columns are read here.
HELDOUT_ROOT = os.path.join(ROOT, "results", "compiler_calibration_extra")
HELDOUT_CSV = os.path.join(ROOT, "results", "compiler_calibration_extra.csv")
# PRE-PRAGMA, INFORMATIONAL. 16 REAL campaign designs, USED TO CHOOSE the
# retired 58/62 crowded-stage margin (scripts/tcam_margin_screen.py) --
# adversarial evidence, not a held-out generalisation check.
CROWDED_ROOT = os.path.join(ROOT, "results", "tcam_margin_screen")
CROWDED_CSV = os.path.join(ROOT, "results", "tcam_margin_screen_compiled.csv")

# The one already-corrected discrepancy between the CSV's own (stale) blocks
# column and a fresh recompute -- see results/compiler_calibration_verify.csv
# and this module's docstring.
KNOWN_BLOCKS_COLUMN_CORRECTION = {
    "independent_low_sd5": (
        "known correction -- CSV blocks=13 predates the StagePlan.blocks fix; "
        "recomputed 16 matches tcam_real=16 "
        "(results/compiler_calibration_verify.csv)"),
}

# EMPTY, deliberately. It once held 5 rows priced by the retracted Mechanism G
# rule. The lookup is kept so a future regression lands as a NEW entry with an
# explanation, instead of silently widening KNOWN_BLOCKS_COLUMN_CORRECTION.
KNOWN_MECHANISM_G_ROWS = {}


def _clean_cell(value):
    """None for a blank CSV cell (pandas NaN); an int for a whole-valued
    float (a numeric column with any blank cell is upcast to float64 by
    pandas even where every present value is an integer); the value
    unchanged otherwise."""
    if isinstance(value, float):
        if math.isnan(value):
            return None
        if value.is_integer():
            return int(value)
    return value


def predictions_from_fixture(fixture_path):
    """{row_id: ResourceUsage}, recomputed by replaying every row of the
    golden fixture at fixture_path through assemble_usage. Reuses
    tests/test_resource_model_golden.rebuild_pool as the sole deserializer --
    see that module's own docstring for why the fixture pins predictions
    without sklearn or a campaign backup."""
    with open(fixture_path, encoding="utf-8") as handle:
        fixture = json.load(handle)
    predictions = {}
    for row in fixture["rows"]:
        pool = rebuild_pool(row)
        usage, _range_plan, _ternary_plan = assemble_usage(pool)
        predictions[row["row_id"]] = usage
    return predictions


def load_ground_truth(csv_path):
    """{row_id: {column: value}} from csv_path, blank cells as None. Every
    column is kept (not just the *_real ones) so the drift check can also
    read the CSV's own stage_depth/blocks columns."""
    frame = pd.read_csv(csv_path)
    truth = {}
    for _, row in frame.iterrows():
        record = {col: _clean_cell(value) for col, value in row.to_dict().items()}
        truth[record["row_id"]] = record
    return truth


def join_rows(predictions, truth, predicted_attr, real_key):
    """[{'row_id', 'predicted', 'real'}, ...], one per prediction row_id, in
    predictions' own order (which follows the fixture's row order -- the same
    order build_sample produced the CSV in, see dump_resource_fixtures.py).
    real is None when truth has no row for this id, or the named column is
    blank there."""
    pairs = []
    for row_id, usage in predictions.items():
        predicted = getattr(usage, predicted_attr)
        real = truth.get(row_id, {}).get(real_key)
        pairs.append({"row_id": row_id, "predicted": predicted, "real": real})
    return pairs


def heldout_pairs(root=HELDOUT_ROOT, csv_path=HELDOUT_CSV):
    """(stage_pairs, blocks_pairs) for the held-out archive, each row replayed
    END TO END by p4_artifact_replay.replay_design: the model's own per-table
    prices, packed by the model's own packer, from nothing but the generated
    program. There is no golden-fixture row for these designs, which is why
    they go through the program text rather than assemble_usage; on the 19
    fixture rows the two paths give identical stage_depth and blocks."""
    truth = load_ground_truth(csv_path)
    stage_pairs, blocks_pairs = [], []
    for row_id, record in truth.items():
        depth, blocks = replay_design(row_id, root)
        stage_pairs.append({"row_id": row_id, "predicted": depth,
                            "real": record.get("stages_real")})
        blocks_pairs.append({"row_id": row_id, "predicted": blocks,
                             "real": record.get("tcam_real")})
    return stage_pairs, blocks_pairs


def pinned_pairs(root=PINNED_ROOT, csv_path=PINNED_CSV):
    """(stage_pairs, blocks_pairs) for the PRIMARY gate, the pinned archive,
    replayed end to end exactly as heldout_pairs replays the held-out one."""
    return heldout_pairs(root, csv_path)


def aggregate(pairs):
    """n / n_compared / exact / mean_error / max_error / under / over over
    `pairs` (each a {'row_id', 'predicted', 'real'} dict, 'real' possibly
    None). Errors are UNSIGNED (abs(predicted - real)) so an under-prediction
    on one row can never cancel an over-prediction on another in mean_error --
    a signed mean would hide exactly the failure mode this table exists to
    catch. under/over are reported separately and DO carry a sign convention:
    over = predicted above real (the safe direction for a capacity model),
    under = predicted below (unsafe). mean_error/max_error are None (not 0 or
    NaN) when n_compared == 0, so a caller can't mistake 'nothing to compare'
    for 'compared and matched'."""
    n = len(pairs)
    compared = [(p["predicted"], p["real"]) for p in pairs if p["real"] is not None]
    n_compared = len(compared)
    exact = sum(1 for predicted, real in compared if predicted == real)
    under = sum(1 for predicted, real in compared if predicted < real)
    over = sum(1 for predicted, real in compared if predicted > real)
    if n_compared:
        errors = [abs(predicted - real) for predicted, real in compared]
        mean_error = sum(errors) / n_compared
        max_error = max(errors)
    else:
        mean_error = None
        max_error = None
    return {
        "n": n,
        "n_compared": n_compared,
        "exact": exact,
        "mean_error": mean_error,
        "max_error": max_error,
        "under": under,
        "over": over,
    }


def format_table(pairs, label):
    """Fixed-width per-row text table: row_id, predicted <label>, real, diff
    (predicted - real), and a note column that reads UNDER for an
    under-prediction, OVER for an over-prediction, blank for an exact match
    or a row with no ground truth (real is None renders as '--', never
    raises)."""
    header = "{:<24} {:>16} {:>10} {:>8} {}".format(
        "row_id", "predicted_" + label, "real", "diff", "note")
    lines = [header, "-" * len(header)]
    for pair in pairs:
        row_id = pair["row_id"]
        predicted = pair["predicted"]
        real = pair["real"]
        if real is None:
            real_str, diff_str, note = "--", "--", ""
        else:
            diff = predicted - real
            real_str = str(real)
            diff_str = str(diff)
            note = "UNDER" if diff < 0 else ("OVER" if diff > 0 else "")
        lines.append("{:<24} {:>16} {:>10} {:>8} {}".format(
            row_id, predicted, real_str, diff_str, note))
    return "\n".join(lines)


def _format_aggregate(stats):
    if stats["n_compared"] == 0:
        return "  n={} n_compared=0 -- no ground truth to compare against".format(
            stats["n"])
    return ("  n={} n_compared={} exact={} mean_error={:.2f} max_error={} "
            "under={} over={}").format(
        stats["n"], stats["n_compared"], stats["exact"], stats["mean_error"],
        stats["max_error"], stats["under"], stats["over"])


def _print_section(title, pairs, label):
    print("\n### {} -- predicted vs p4c\n".format(title))
    print(format_table(pairs, label))
    print(_format_aggregate(aggregate(pairs)))


def _print_sram_map_ram(truth, row_ids):
    print("\n### SRAM and map-RAM -- OBSERVED, NOT PREDICTED\n")
    header = "{:<24} {:>10} {:>12}".format("row_id", "sram_real", "map_ram_real")
    print(header)
    print("-" * len(header))
    for row_id in row_ids:
        record = truth.get(row_id, {})
        sram = record.get("sram_real")
        map_ram = record.get("map_ram_real")
        print("{:<24} {:>10} {:>12}".format(
            row_id, "--" if sram is None else sram,
            "--" if map_ram is None else map_ram))
    print(
        "\nSpec 2 excludes SRAM/map-RAM from prediction: the ground truth "
        "does not isolate variables. SRAM tracks register count but "
        "contributes independently with tree count, and differs between "
        "arms at matched k -- independent_high_sd7 sram_real=62 vs "
        "joint_high_sd7 sram_real=53. It even differs between two rows with "
        "IDENTICAL register_depth/register_count: independent_high_sd8 and "
        "independent_high_sd10 both have register_depth=8, register_count=19, "
        "yet sram_real is 71 vs 75.")


def _print_drift_check(predictions, truth, row_ids):
    print("\n### drift check\n")
    print(
        "The CSV's own stage_depth/blocks columns were written by whatever "
        "model code was live when each row was collected -- they should not "
        "move on a re-recompute against the CURRENT code (a verbatim "
        "extraction changes nothing observable). A disagreement here is a "
        "finding, not something to reconcile away.\n")

    stage_drift = []
    blocks_drift = []
    for row_id in row_ids:
        usage = predictions.get(row_id)
        if usage is None:
            continue
        record = truth.get(row_id, {})
        csv_stage_depth = record.get("stage_depth")
        if csv_stage_depth is not None and usage.stage_depth != csv_stage_depth:
            stage_drift.append((row_id, usage.stage_depth, csv_stage_depth))
        csv_blocks = record.get("blocks")
        if csv_blocks is not None and usage.blocks != csv_blocks:
            blocks_drift.append((row_id, usage.blocks, csv_blocks))

    print("stage_depth vs the CSV's own stage_depth column:")
    if stage_drift:
        print("  {} row(s) disagree (UNEXPECTED -- investigate):".format(
            len(stage_drift)))
        for row_id, recomputed, csv_value in stage_drift:
            print("    {}: recomputed={} csv={}".format(
                row_id, recomputed, csv_value))
    else:
        print("  OK -- no drift on any of the {} rows".format(len(row_ids)))

    print("\nblocks vs the CSV's own blocks column:")
    if not blocks_drift:
        print("  OK -- no drift on any of the {} rows".format(len(row_ids)))
    else:
        unexplained = []
        for row_id, recomputed, csv_value in blocks_drift:
            if row_id in KNOWN_BLOCKS_COLUMN_CORRECTION:
                print("    {}: recomputed={} csv={} -- {}".format(
                    row_id, recomputed, csv_value,
                    KNOWN_BLOCKS_COLUMN_CORRECTION[row_id]))
            elif row_id in KNOWN_MECHANISM_G_ROWS:
                print(
                    "    {}: recomputed={} csv={} -- known finding "
                    "mechanism_g_over_application_2026_09_07 (delta +{}), "
                    "carried over from Task 11: see "
                    "tests/fixtures/resource_model_golden.json's "
                    "metadata.known_findings and "
                    "scripts/dump_resource_fixtures.py's module docstring "
                    "for the full mechanistic explanation".format(
                        row_id, recomputed, csv_value,
                        KNOWN_MECHANISM_G_ROWS[row_id]))
            else:
                unexplained.append((row_id, recomputed, csv_value))
        if unexplained:
            print("  {} row(s) disagree with NO known explanation -- "
                  "investigate:".format(len(unexplained)))
            for row_id, recomputed, csv_value in unexplained:
                print("    {}: recomputed={} csv={}".format(
                    row_id, recomputed, csv_value))


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--fixture", default=DEFAULT_FIXTURE)
    parser.add_argument("--csv", dest="csv_path", default=DEFAULT_CSV)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)

    if (os.path.isfile(PINNED_CSV)
            and os.path.isdir(os.path.join(PINNED_ROOT, "compiles"))):
        print("## PRIMARY GATE -- 43 pragma'd compiles "
              "(results/compiler_calibration_pinned)")
        pin_stage, pin_blocks = pinned_pairs()
        _print_section("stage_depth (pinned)", pin_stage, "stage_depth")
        _print_section("blocks (pinned)", pin_blocks, "blocks")
        for row_id, reason in KNOWN_PINNED_MISSES.items():
            print("  known miss, %s: %s" % (row_id, reason))
    else:
        print("(pinned archive %s not present -- PRIMARY GATE SKIPPED)"
              % PINNED_ROOT)

    print("\n\n## PRE-PRAGMA, INFORMATIONAL -- archives compiled without "
          "@placement_priority; p4c chose its own tree order there")
    predictions = predictions_from_fixture(args.fixture)
    truth = load_ground_truth(args.csv_path)
    row_ids = list(predictions.keys())

    stage_pairs = join_rows(predictions, truth, "stage_depth", "stages_real")
    blocks_pairs = join_rows(predictions, truth, "blocks", "tcam_real")

    _print_section("stage_depth (fitted, pre-pragma)", stage_pairs, "stage_depth")
    _print_section("blocks (fitted, pre-pragma)", blocks_pairs, "blocks")
    _print_sram_map_ram(truth, row_ids)
    _print_drift_check(predictions, truth, row_ids)

    if os.path.isdir(os.path.join(HELDOUT_ROOT, "compiles")):
        print("\n\n## HELD OUT, PRE-PRAGMA -- 8 compiles never used to fit "
              "the model (results/compiler_calibration_extra)")
        held_stage, held_blocks = heldout_pairs()
        _print_section("stage_depth (held out)", held_stage, "stage_depth")
        _print_section("blocks (held out)", held_blocks, "blocks")
    else:
        print("\n(held-out archive %s not present -- section skipped)"
              % HELDOUT_ROOT)

    if os.path.isfile(CROWDED_CSV):
        print("\n\n## ADVERSARIAL, PRE-PRAGMA -- 16 real campaign designs "
              "used to choose the retired 58/62 margin "
              "(results/tcam_margin_screen)")
        crowd_stage, crowd_blocks = heldout_pairs(CROWDED_ROOT, CROWDED_CSV)
        _print_section("stage_depth (crowded real designs)", crowd_stage,
                       "stage_depth")
        _print_section("blocks (crowded real designs)", crowd_blocks, "blocks")
        print("  margin_independent_M150_k5_s12's UNDER: unpinned, p4c served "
              "its keys in an order costing 2 more blocks; the pinned compile "
              "of the same design costs exactly the predicted 66.")


if __name__ == "__main__":
    main()
