"""The per-table regression gate (2026-09-20 rewrite design Sec 6.4).

WHY THIS EXISTS. scripts/validation_table.py scores DESIGN totals, where a +1
on one table and a -1 on another cancel out -- which is exactly how a real
per-table error survived an entire calibration study behind a clean-looking
17/17 (see that script's docstring). This script is the missing check: score
every individual table observation this project has ever collected against
p4c -- 405 rows across 14 CSV files in results/, 50 of them from compiles the
model was never fitted on -- and require ZERO under-predictions on the
quantity the production packer will actually charge, at every placement the
packer can emit.

THREE PREDICTED QUANTITIES, per row:
  blocks_headline -- tables.codeword_to_blocks_headline(field_bit_widths),
                      the Sec 13.1 "S = 0" ladder, the one sentence the paper
                      states.
  blocks_refined  -- tables.codeword_to_blocks(field_bit_widths), the
                      production per-table price (headline plus the Sec 2.3
                      isolation credit, capped at one field per Sec 6.1).
  blocks_charged  -- blocks_refined, PLUS ONE if the crowded-stage margin
                      applies: the table's key is not the first distinct key
                      in its stage AND two different keys fill more than 58 of
                      the stage's crossbar bytes (target.py). Headline and
                      refined alone are EXPECTED to under-predict on crowded
                      rows; only blocks_charged must hit 0 under.

Rows whose stage holds two different keys past 62 bytes are marked
placement_refused: the packer never produces that stage, so they are reported
and excluded from the gate. (Until 2026-09-25 blocks_charged instead added the
per-key SATURATION margin -- +1 whenever a saturated key was not first in any
shared stage. Retired; see src/p4model/packing.py's charged().)

ONE ADAPTER PER SOURCE CSV, feeding ONE shared scoring core (score_observation
below) -- the files' schemas differ enough (single key vs. two, an explicit
measured_start_group column vs. none, a JSON layout blob) that a common
adapter would be more contorted than one small adapter each (spec Sec 6.4).

WHICH KEY IS "NOT FIRST", resolved per source:

  * tcam_offset_harvest, tcam_heldout_harvest, tcam_discount_scan,
    tcam_offset_scan, tcam_offset_probe, tcam_mixed_key_cap_sweep/onset,
    tcam_version_sweep, tcam_spacer_sweep: each row (or the single scored
    "a" key) carries its own measured_start_group -- direct per-table ground
    truth. not_first = measured_start_group is not null and > 0, taken
    literally. The probe-family files also record both keys' bytes, so they
    set crowded/placement_refused.
  * tcam_phv_slice_sweep, tcam_ledger_divergence_sweep, tcam_lane_sweep,
    tcam_field_count_sweep: single-key probes by construction -- no margin.
  * tcam_stretch_sweep: ground truth and per-table stage placement come from
    the `layout` JSON (its blocks_a/blocks_b columns are a STALE predicted
    price). not_first per table = some table of the OTHER key shares its
    stage; the file cannot say which key p4c served first, so both are
    candidates -- the packer's own worst-order choice. Stage bytes are the two
    keys' widths, so crowded and refused are set here too.
  * tcam_offset_harvest and tcam_heldout_harvest do not record stage byte
    totals, so crowded stays False there. That can only LOWER blocks_charged
    below the packer's own charge, which keeps the gate conservative.

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
from src.p4model.target import (  # noqa: E402
    TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE,
    TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE,
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
    # HELD OUT: results/compiler_calibration_extra/, never used for fitting.
    # Written by scripts/tcam_heldout_harvest.py.
    "tcam_heldout_harvest.csv",
    # The mixed-key byte cap's own evidence (scripts/tcam_mixed_key_cap_sweep.py):
    # five probe shapes at 59-64 combined bytes, and two of them at 20-58.
    "tcam_mixed_key_cap_sweep.csv",
    "tcam_mixed_key_cap_onset.csv",
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
                      not_first, note="", placement_refused=False,
                      crowded=False):
    """The one scoring core every adapter feeds. field_bit_widths is the
    table's key, exactly what tables.codeword_to_blocks takes. not_first is
    this specific table's OWN evidence of whether its key was first in its
    stage -- never a row-level "sharing was possible" flag. The margin fires
    only when not_first AND crowded (below): a first-placed table never pays,
    and neither does any table in an uncrowded stage. `saturated`
    (crossbar_capacity(g) == B) is still recorded per row for reading the
    table, but no longer charges anything.

    placement_refused marks an observation whose STAGE the packer would never
    produce -- two different keys past
    target.TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE. Its price is still
    recorded, but under_predictions() skips it: the model avoids that
    placement rather than pricing it (the F5 rows, dsp41/dsp42).

    crowded marks a table whose stage holds two different keys filling more
    than target.TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE (58) bytes:
    the packer charges every non-first table there +1.
    Only sources that record the stage's byte total can set it; elsewhere it
    stays False, which can only LOWER blocks_charged below the packer's, so
    the under-prediction gate stays conservative."""
    field_bit_widths = tuple(field_bit_widths)
    key_bytes = codeword_fields_to_bytes_from_bits(field_bit_widths)
    headline = codeword_to_blocks_headline(field_bit_widths)
    refined = codeword_to_blocks(field_bit_widths)
    saturated = crossbar_capacity(refined) == key_bytes
    # Since 2026-09-25 the only sharing charge is the crowded-stage margin;
    # `saturated` is still recorded, for reading the table, but charges nothing.
    margin_applied = bool(not_first) and bool(crowded)
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
        "placement_refused": bool(placement_refused),
        "crowded": bool(crowded),
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
    """Shared adapter for tcam_discount_scan.csv, tcam_offset_scan.csv,
    tcam_offset_probe.csv and the two tcam_mixed_key_cap_*.csv -- identical schema (point_id, probe_fields,
    probe_real_blocks, measured_start_group, ...), differing only in which
    probe geometry each sweep varied."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["probe_fields"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        # Spacer and probe are always two DIFFERENT keys, so their sum is the
        # stage's distinct-key byte load whenever p4c co-located them.
        stage_bytes = (r["spacer_bytes"] + r["probe_key_bytes"]
                       if bool(r["both_in_one_stage"]) else 0)
        refused = stage_bytes > TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE
        crowded = stage_bytes > TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE
        rows.append(score_observation(
            source_name, r["point_id"], field_bits, r["probe_real_blocks"],
            not_first, placement_refused=refused, crowded=crowded))
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
    first, so both are treated as margin candidates -- the same worst-order
    choice the packer makes. The stage's distinct-key bytes are the two keys'
    own widths, so crowded (> 58) and refused (> 62) are computed here too:
    a49 + b12 = 61 is crowded, a32 + b32 = 64 is refused."""
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
                stage_bytes = (sum(codeword_fields_to_bytes_from_bits(fields[t])
                                   for t in tags_here)
                               if shares_stage_with_other_key else 0)
                height = span[2]
                rows.append(score_observation(
                    "tcam_stretch_sweep", "%s/%s" % (point_id, name),
                    fields[tag], height, shares_stage_with_other_key,
                    note="ground truth and not_first both read from layout, "
                         "not blocks_a/blocks_b -- see adapter docstring",
                    placement_refused=(stage_bytes
                                       > TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE),
                    crowded=(stage_bytes
                             > TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE)))
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


def score_heldout_harvest(path):
    """50 classification tables from the 8 HELD-OUT compiles
    (results/compiler_calibration_extra/, scripts/tcam_heldout_harvest.py).
    Same per-table ground truth as score_offset_harvest: measured_start_group
    says whether this table's key was first in its stage."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        rows.append(score_observation(
            "tcam_heldout_harvest", "%s/%s" % (r["row_id"], r["table"]),
            _parse_int_list(r["field_bits"]), r["observed_blocks"],
            _not_first_from_measured_start_group(r["measured_start_group"])))
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
    rows += score_heldout_harvest(path("tcam_heldout_harvest.csv"))
    rows += score_probe_family(path("tcam_mixed_key_cap_sweep.csv"),
                               "tcam_mixed_key_cap_sweep")
    rows += score_probe_family(path("tcam_mixed_key_cap_onset.csv"),
                               "tcam_mixed_key_cap_onset")
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
    """Rows priced below p4c, among placements the packer can emit."""
    return [r for r in rows if r[diff_key] < 0 and not r["placement_refused"]]


def refused_placements(rows):
    """Rows observed at a stage the packer refuses (mixed-key byte cap)."""
    return [r for r in rows if r["placement_refused"]]


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

    refused = refused_placements(rows)
    print("\n### placements the packer refuses (two keys > %d bytes): %d\n"
          % (TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE, len(refused)))
    for r in refused:
        print("  %s / %s: field_bit_widths=%s blocks_charged=%d observed_blocks=%d"
              % (r["source"], r["identifier"], r["field_bit_widths"],
                 r["blocks_charged"], r["observed_blocks"]))

    print("\nGATE (blocks_charged, 0 under-predictions on emitted placements): %s"
          % ("PASS" if not unders else "FAIL -- %d under-prediction(s)" % len(unders)))

    return rows


if __name__ == "__main__":
    main()
