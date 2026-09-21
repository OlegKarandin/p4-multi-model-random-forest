"""The per-table regression gate (2026-09-20 rewrite design Sec 6.4).

WHY THIS EXISTS. scripts/validation_table.py scores DESIGN totals, where a +1
on one table and a -1 on another cancel out -- which is exactly how a real
per-table error survived an entire calibration study behind a clean-looking
17/17 (see that script's docstring). This script is the missing check: score
every individual table observation this project has ever collected against
p4c -- 308 rows spread across 11 CSV files in results/ -- and require ZERO
under-predictions on the quantity the production packer will actually charge.

THREE PREDICTED QUANTITIES, per row:
  blocks_headline -- tables.codeword_to_blocks_headline(field_bit_widths),
                      the Sec 13.1 "S = 0" ladder, the one sentence the paper
                      states.
  blocks_refined  -- tables.codeword_to_blocks(field_bit_widths), the
                      production per-table price (headline plus the Sec 2.3
                      isolation credit, capped at one field per Sec 6.1).
  blocks_charged  -- blocks_refined, PLUS ONE if the Sec 13.2 stage-sharing
                      margin applies: the table's key is not the first
                      distinct key in its stage, AND its own standalone price
                      has no spare half-byte slot (crossbar_capacity(g) == B,
                      i.e. it is exactly saturated). blocks_charged is the
                      quantity a downstream 12-stage feasibility gate will
                      rely on once Task 4 wires the packer to it -- headline
                      and refined alone are EXPECTED to under-predict on the
                      stage-sharing rows; only blocks_charged must hit 0/308.

ONE ADAPTER PER SOURCE CSV, feeding ONE shared scoring core (score_observation
below) -- the 11 files' schemas differ enough (single key vs. two, an
explicit measured_start_group column vs. none, a JSON layout blob) that a
common adapter would be more contorted than 11 small ones (spec Sec 6.4).

APPLYING THE MARGIN -- the one genuinely hard judgment call, resolved per
source as follows (see task-3-report.md for the full account):

  * tcam_offset_harvest, tcam_discount_scan, tcam_offset_scan,
    tcam_offset_probe, tcam_version_sweep, tcam_spacer_sweep: each row (or,
    for version_sweep/spacer_sweep, the single scored "a" key) carries its own
    measured_start_group -- direct per-table ground truth. not_first =
    measured_start_group is not null and > 0, taken literally, in preference
    to any row-level boolean (both_in_one_stage / shared_stage).
  * tcam_phv_slice_sweep, tcam_ledger_divergence_sweep, tcam_lane_sweep,
    tcam_field_count_sweep: single-key probes by construction (no second key
    ever appears) -- the margin never applies, full stop, exactly as the task
    brief states for these four regardless of any column they happen to carry.
  * tcam_stretch_sweep: no measured_start_group column exists here, and this
    file's own `blocks_a`/`blocks_b` columns are STALE -- they hold the
    predicted price a superseded formula gave when the sweep script ran, not
    p4c's real per-table count (verified against the `layout` JSON, which
    IS read from resources.json; two tables -- ragged_ax1_bx4's tern_a0 and
    ragged_ax2_bx2's tern_a1 -- cost 10 blocks for real while blocks_a says
    9, exactly Sec 12.1's F7 finding). This adapter therefore takes ground
    truth and per-table stage placement from `layout` alone, never from
    blocks_a/blocks_b. not_first per table = some table of the OTHER key
    shares this table's TCAM stage in the committed layout; this file has no
    crossbar-byte evidence to say which of the two keys is truly first, so
    both keys are treated as margin candidates whenever both are saturated
    (conservative, over-predicting, and safe) -- except on the two ragged
    rows, where only the ragged key is ever saturated, so the margin fires
    on exactly the table that needs it and nowhere else.

Run (from the repository root):
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/tcam_table_scoreboard.py
"""
import argparse
import ast
import json
import math
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from src.p4model.tables import (  # noqa: E402
    codeword_fields_to_bytes_from_bits,
    codeword_to_blocks,
    codeword_to_blocks_headline,
    crossbar_capacity,
)

RESULTS_DIR = os.path.join(ROOT, "results")
DEFAULT_OUT = os.path.join(RESULTS_DIR, "tcam_table_scoreboard.csv")

# Every source this scoreboard reads, so tests and main() share one list
# rather than two copies drifting apart.
SOURCE_FILES = [
    "tcam_offset_harvest.csv",
    "tcam_discount_scan.csv",
    "tcam_offset_scan.csv",
    "tcam_offset_probe.csv",
    "tcam_stretch_sweep.csv",
    "tcam_phv_slice_sweep.csv",
    "tcam_ledger_divergence_sweep.csv",
    "tcam_version_sweep.csv",
    "tcam_spacer_sweep.csv",
    "tcam_lane_sweep.csv",
    "tcam_field_count_sweep.csv",
]


def sources_present(results_dir=RESULTS_DIR):
    """Whether every file SOURCE_FILES names is on disk under results_dir --
    the condition tests skip on (the CSVs are real p4c output, gitignored,
    generated by one-shot instrument scripts, never regenerated here)."""
    return all(os.path.isfile(os.path.join(results_dir, name))
               for name in SOURCE_FILES)


def _parse_int_list(value):
    """A comma-separated string of ints -> a tuple, e.g. '84,84' -> (84, 84),
    '88' -> (88,). The convention every *_scan/*_sweep CSV in this corpus
    uses for a probe's field bit widths, except tcam_offset_harvest (a
    Python-tuple-formatted string, parsed separately with ast.literal_eval)."""
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def score_observation(source, identifier, field_bit_widths, observed_blocks,
                      not_first, note=""):
    """The one scoring core every adapter feeds. field_bit_widths is the
    table's key, exactly what tables.codeword_to_blocks takes. not_first is
    this specific table's OWN evidence of whether its key was first in its
    stage -- never a row-level "sharing was possible" flag. The Sec 13.2
    margin fires only when not_first AND the table's own standalone price is
    exactly saturated (no spare half-byte slot): a table with slack never
    pays regardless of placement, and a first-placed table never pays
    regardless of slack."""
    field_bit_widths = tuple(field_bit_widths)
    key_bytes = codeword_fields_to_bytes_from_bits(field_bit_widths)
    headline = codeword_to_blocks_headline(field_bit_widths)
    refined = codeword_to_blocks(field_bit_widths)
    saturated = crossbar_capacity(refined) == key_bytes
    margin_applied = bool(not_first) and saturated
    charged = refined + (1 if margin_applied else 0)
    return {
        "source": source,
        "identifier": identifier,
        "field_bit_widths": field_bit_widths,
        "key_bytes": key_bytes,
        "observed_blocks": int(observed_blocks),
        "not_first": bool(not_first),
        "saturated": saturated,
        "margin_applied": margin_applied,
        "blocks_headline": headline,
        "blocks_refined": refined,
        "blocks_charged": charged,
        "diff_headline": headline - int(observed_blocks),
        "diff_refined": refined - int(observed_blocks),
        "diff_charged": charged - int(observed_blocks),
        "note": note,
    }


def _not_first_from_measured_start_group(value):
    """The direct per-table ground truth every measured_start_group-bearing
    source shares: a table whose run starts strictly after group 0 was not
    the first distinct key in its stage. Taken LITERALLY (no special-casing
    of start_group_ambiguous) -- see the module docstring and task-3-report.md
    for why a literal reading is the one the brief specifies and the one that
    never under-predicts on the rows this project has measured."""
    return pd.notna(value) and value > 0


# --- one adapter per source CSV ------------------------------------------

def score_offset_harvest(path):
    """100 archived classification-table observations. field_bits is a
    Python-tuple-formatted string; distinct_keys_in_stage documents sharing
    was POSSIBLE, but measured_start_group is this file's per-table ground
    truth of whether it actually happened for THIS table."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = ast.literal_eval(r["field_bits"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        rows.append(score_observation(
            "tcam_offset_harvest", "%s/%s" % (r["row_id"], r["table"]),
            field_bits, r["observed_blocks"], not_first))
    return rows


def score_probe_family(path, source_name):
    """Shared adapter for tcam_discount_scan.csv, tcam_offset_scan.csv and
    tcam_offset_probe.csv -- identical schema (point_id, probe_fields,
    probe_real_blocks, measured_start_group, ...), differing only in which
    probe geometry each sweep varied."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["probe_fields"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        rows.append(score_observation(
            source_name, r["point_id"], field_bits, r["probe_real_blocks"],
            not_first))
    return rows


def score_stretch_sweep(path):
    """9 rows -> 40 table observations. blocks_a/blocks_b are a STALE
    predicted price (scripts/tcam_stretch_sweep.blocks_for_key, the retired
    band/xbar formula) frozen at sweep time, not p4c's real per-table count --
    ground truth and per-table stage placement both come from the `layout`
    JSON (stage -> {table: [column, start_row, height]}), which IS read from
    the compiler's own resources.json (tcam_stretch_sweep.committed_tcam_grid).
    height (layout[...][table][2]) is the real observed block count.

    not_first for one table = some table of the OTHER key (tag 'a' vs 'b')
    shares this table's stage in the committed layout. This file carries no
    crossbar-byte evidence to say which of two co-located keys is truly
    first, so both are treated as margin candidates -- score_observation only
    actually charges the margin when the table's own price is ALSO
    saturated, which is what keeps this conservative choice from
    over-firing: the narrower key in every row here (the 12/27/32-byte 'b'
    or 'a' partner alongside a non-saturated width) never has spare-free
    headroom taken away by this, and the two rows where it matters
    (ragged_ax1_bx4's tern_a0, ragged_ax2_bx2's tern_a1) are exactly the
    F7 rows this margin exists to cover -- real 10, refined 9, charged 10."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        point_id = r["point_id"]
        fields = {
            "a": _parse_int_list(r["fields_a"]),
            "b": _parse_int_list(r["fields_b"]),
        }
        layout = json.loads(r["layout"])
        for stage, tables in layout.items():
            tags_here = {name[len("tern_"):][0] for name in tables}
            for name, span in tables.items():
                tag = name[len("tern_"):][0]
                other_tag = "b" if tag == "a" else "a"
                shares_stage_with_other_key = other_tag in tags_here
                height = span[2]
                rows.append(score_observation(
                    "tcam_stretch_sweep", "%s/%s" % (point_id, name),
                    fields[tag], height, shares_stage_with_other_key,
                    note="ground truth and not_first both read from layout, "
                         "not blocks_a/blocks_b -- see adapter docstring"))
    return rows


def score_phv_slice_sweep(path):
    """24 single-key isolation probes. field_bit_widths is reconstructed from
    clean_bits + solid_bits (one field per row); key_bytes cross-checked
    against the CSV's own column. No sharing column at all -- the margin
    never applies, full stop."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        total_bits = int(r["clean_bits"]) + int(r["solid_bits"])
        expected_key_bytes = math.ceil(total_bits / 8)
        if expected_key_bytes != int(r["key_bytes"]):
            raise ValueError(
                "tcam_phv_slice_sweep %s: clean_bits+solid_bits=%d bytes-round "
                "to %d, CSV's own key_bytes says %d" % (
                    r["point_id"], total_bits, expected_key_bytes,
                    r["key_bytes"]))
        rows.append(score_observation(
            "tcam_phv_slice_sweep", r["point_id"], (total_bits,),
            r["real_blocks"], False))
    return rows


def score_ledger_divergence_sweep(path):
    """13 single-key probes, 1-5 fields each (the CSV's `fields` column is a
    comma-separated list despite carrying exactly one field on some rows --
    NOT always a single int, contrary to an earlier draft of this task's
    brief; parsed generically like every other *_sweep source). No sharing
    column -- the margin never applies, full stop."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields"])
        rows.append(score_observation(
            "tcam_ledger_divergence_sweep", r["point_id"], field_bits,
            r["real_blocks"], False))
    return rows


def score_version_sweep(path):
    """12 rows; only the "a" key is scored (real_blocks_b, present on 2 rows,
    is the spacer/pushed-in key's own count, not one of this task's 308 --
    matching the explicit instruction for tcam_spacer_sweep's analogous
    real_blocks_b, and confirmed by the row-count arithmetic: scoring "b" too
    would give 14 observations from this file, not the 12 the brief's total
    requires). measured_start_group is this file's per-table ground truth,
    including on the two rows (c_solid_shifted, d_ragged_shifted) where it is
    0 with start_group_ambiguous=True -- taken literally per the module
    docstring, which also means not_first is False there and no margin
    fires; both rows are already exact at blocks_refined (3 == 3), so this
    reading costs nothing towards the 0-under gate either way."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields_a"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        rows.append(score_observation(
            "tcam_version_sweep", r["point_id"], field_bits,
            r["real_blocks_a"], not_first))
    return rows


def score_spacer_sweep(path):
    """12 rows; only the "a" (probe) key is scored -- real_blocks_b is the
    spacer table's own block count, explicitly not one of this task's 88
    distinct probe keys per the brief. measured_start_group is this file's
    per-table ground truth."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields_a"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        rows.append(score_observation(
            "tcam_spacer_sweep", r["point_id"], field_bits,
            r["real_blocks_a"], not_first))
    return rows


def score_lane_sweep(path):
    """10 single-key PHV-lane probes. No sharing column -- the margin never
    applies, full stop, regardless of the measured_start_group column this
    file happens to carry (always 0: a single key alone in its stage has
    nothing to share with)."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields"])
        rows.append(score_observation(
            "tcam_lane_sweep", r["point_id"], field_bits, r["real_blocks"],
            False))
    return rows


def score_field_count_sweep(path):
    """10 single-key field-count probes. No sharing column -- the margin
    never applies, full stop, for the same reason as tcam_lane_sweep."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields"])
        rows.append(score_observation(
            "tcam_field_count_sweep", r["point_id"], field_bits,
            r["real_blocks"], False))
    return rows


def score_all(results_dir=RESULTS_DIR):
    """Every observation from every source, in SOURCE_FILES order."""
    path = lambda name: os.path.join(results_dir, name)  # noqa: E731
    rows = []
    rows += score_offset_harvest(path("tcam_offset_harvest.csv"))
    rows += score_probe_family(path("tcam_discount_scan.csv"), "tcam_discount_scan")
    rows += score_probe_family(path("tcam_offset_scan.csv"), "tcam_offset_scan")
    rows += score_probe_family(path("tcam_offset_probe.csv"), "tcam_offset_probe")
    rows += score_stretch_sweep(path("tcam_stretch_sweep.csv"))
    rows += score_phv_slice_sweep(path("tcam_phv_slice_sweep.csv"))
    rows += score_ledger_divergence_sweep(path("tcam_ledger_divergence_sweep.csv"))
    rows += score_version_sweep(path("tcam_version_sweep.csv"))
    rows += score_spacer_sweep(path("tcam_spacer_sweep.csv"))
    rows += score_lane_sweep(path("tcam_lane_sweep.csv"))
    rows += score_field_count_sweep(path("tcam_field_count_sweep.csv"))
    return rows


def _tally(rows, diff_key):
    exact = sum(1 for r in rows if r[diff_key] == 0)
    over = sum(1 for r in rows if r[diff_key] > 0)
    under = sum(1 for r in rows if r[diff_key] < 0)
    return {"n": len(rows), "exact": exact, "over": over, "under": under}


def summary_table(rows):
    """One row per (source, quantity), DataFrame.to_string(index=False)
    house style."""
    sources = sorted({r["source"] for r in rows})
    quantities = [
        ("blocks_headline", "diff_headline"),
        ("blocks_refined", "diff_refined"),
        ("blocks_charged", "diff_charged"),
    ]
    records = []
    for source in sources:
        source_rows = [r for r in rows if r["source"] == source]
        for label, diff_key in quantities:
            tally = _tally(source_rows, diff_key)
            records.append({"source": source, "quantity": label, **tally})
    for label, diff_key in quantities:
        tally = _tally(rows, diff_key)
        records.append({"source": "TOTAL", "quantity": label, **tally})
    return pd.DataFrame.from_records(records)


def under_predictions(rows, diff_key="diff_charged"):
    return [r for r in rows if r[diff_key] < 0]


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=DEFAULT_OUT)
    parser.add_argument("--results-dir", default=RESULTS_DIR)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = score_all(args.results_dir)

    frame = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    frame.to_csv(args.out, index=False)
    print("wrote %s (%d observations)" % (args.out, len(rows)))

    print("\n### Per-source / per-quantity tallies\n")
    print(summary_table(rows).to_string(index=False))

    unders = under_predictions(rows, "diff_charged")
    print("\n### blocks_charged under-predictions: %d\n" % len(unders))
    for r in unders:
        print("  %s / %s: field_bit_widths=%s key_bytes=%d "
              "blocks_refined=%d blocks_charged=%d observed_blocks=%d "
              "not_first=%s saturated=%s" % (
                  r["source"], r["identifier"], r["field_bit_widths"],
                  r["key_bytes"], r["blocks_refined"], r["blocks_charged"],
                  r["observed_blocks"], r["not_first"], r["saturated"]))

    print("\nGATE (blocks_charged, 0 under-predictions required): %s"
          % ("PASS" if not unders else "FAIL -- %d under-prediction(s)" % len(unders)))

    return rows


if __name__ == "__main__":
    main()
