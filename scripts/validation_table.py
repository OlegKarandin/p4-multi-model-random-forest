"""Validation table: this project's resource-model predictions (recomputed
with CURRENT code) against real Tofino p4c ground truth, over all 19 rows of
the compiler-calibration sample (Spec 4.7).

CAPTION CONSTRAINT. These are 19 ARCHIVED p4c compiles re-evaluated against
current model code -- not 19 fresh compiles. Published wording must say so.
Ground truth (stages_real, tcam_real, sram_real, map_ram_real) is read from
results/compiler_calibration_v6.csv and came from p4c. Predictions are
recomputed here by replaying tests/fixtures/resource_model_golden.json through
src/p4model/usage.py's assemble_usage, so this script needs neither sklearn nor
the campaign backup and reproduces in under a second.

WHY THE CSV'S OWN PREDICTION COLUMNS ARE NOT USED. results/
compiler_calibration_v6.csv carries its own stage_depth/blocks columns, but
those were written by whatever model code was live the moment each row was
collected -- not necessarily today's. independent_low_sd5 is the documented
case: its CSV blocks=13 predates the StagePlan.blocks fix; recomputing from
the golden fixture (frozen AFTER that fix) gives 16, matching tcam_real=16
(results/compiler_calibration_verify.csv). This script never reads the CSV's
stage_depth/blocks columns as predictions -- only stages_real/tcam_real/
sram_real/map_ram_real (p4c's own numbers, which cannot go stale) are used as
ground truth. The CSV's own stage_depth/blocks columns are read ONLY by the
drift check below, whose job is precisely to catch a stale-column case like
this one.

KNOWN FINDING carried over from Task 11 (id
mechanism_g_over_application_2026_09_07, tests/fixtures/
resource_model_golden.json's metadata.known_findings[0] and scripts/
dump_resource_fixtures.py's own module docstring): on 5 rows -- all
group='independent'/encoding='disjoint' with every ternary table
ternary_ragged=True -- recomputed blocks is HIGHER than tcam_real (+1 on
independent_low_sd7/independent_high_sd6/independent_high_sd7/
independent_high_sd8, +3 on independent_low_sd6). This is a pre-existing,
already-diagnosed over-application of the ragged-key group-offset charge
("Mechanism G"), not something this Tier 1 extraction task introduced or is
scoped to fix. So blocks comes out exact on 12 of 17 comparable rows here,
not 17 of 17 -- see the drift check section below for the accounting.
stage_depth is unaffected: it remains exact on 18 of 19, the sole miss being
independent_high_sd12 (predicts 14 against stages_real=13, the safe/over
direction, on a design already past the 12-stage ceiling).

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

DEFAULT_FIXTURE = os.path.join(ROOT, "tests", "fixtures", "resource_model_golden.json")
DEFAULT_CSV = os.path.join(ROOT, "results", "compiler_calibration_v6.csv")

# The one already-corrected discrepancy between the CSV's own (stale) blocks
# column and a fresh recompute -- see results/compiler_calibration_verify.csv
# and this module's docstring.
KNOWN_BLOCKS_COLUMN_CORRECTION = {
    "independent_low_sd5": (
        "known correction -- CSV blocks=13 predates the StagePlan.blocks fix; "
        "recomputed 16 matches tcam_real=16 "
        "(results/compiler_calibration_verify.csv)"),
}

# The 5-row Mechanism G over-application finding, carried over from Task 11
# (tests/fixtures/resource_model_golden.json metadata.known_findings /
# scripts/dump_resource_fixtures.py's module docstring). Recomputed blocks is
# higher than the CSV's own (also-stale, pre-Mechanism-G) blocks column here
# by the same amount it is higher than tcam_real.
KNOWN_MECHANISM_G_ROWS = {
    "independent_low_sd6": 3,
    "independent_low_sd7": 1,
    "independent_high_sd6": 1,
    "independent_high_sd7": 1,
    "independent_high_sd8": 1,
}


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
    predictions = predictions_from_fixture(args.fixture)
    truth = load_ground_truth(args.csv_path)
    row_ids = list(predictions.keys())

    stage_pairs = join_rows(predictions, truth, "stage_depth", "stages_real")
    blocks_pairs = join_rows(predictions, truth, "blocks", "tcam_real")

    _print_section("stage_depth", stage_pairs, "stage_depth")
    _print_section("blocks", blocks_pairs, "blocks")
    _print_sram_map_ram(truth, row_ids)
    _print_drift_check(predictions, truth, row_ids)


if __name__ == "__main__":
    main()
