"""The per-table regression gate (2026-09-20 rewrite design Sec 6.4).

WHY THIS EXISTS. scripts/validation_table.py scores DESIGN totals, where a +1
on one table and a -1 on another cancel out -- which is exactly how a real
per-table error survived an entire calibration study behind a clean-looking
17/17 (see that script's docstring). This script is the missing check: score
every individual table observation this project has ever collected against
p4c -- 405 rows across 14 CSV files in results/, 50 of them from compiles the
model was never fitted on -- and require ZERO under-predictions on the
quantity the production packer will actually charge, at every placement the
packer can emit. Since 2026-10-04 blocks_charged is priced with lanes.table_blocks
(pinned fill-low layout); probes compiled without pins under-predict by one on
24-remainder fields -- listed in the test, not a model error on generated designs.

THREE PREDICTED QUANTITIES, per row:
  blocks_headline -- tables.codeword_to_blocks_headline(field_bit_widths),
                      the Sec 13.1 "S = 0" ladder, the one sentence the paper
                      states.
  blocks_refined  -- tables.codeword_to_blocks(field_bit_widths), the
                      LADDER per-table price (the production price is now
                      lanes.table_blocks; headline plus the Sec 2.3
                      isolation credit, capped at one field per Sec 6.1).
  blocks_charged  -- what src/p4model/packing.py's ordered stage simulation
                      charges the table (audit C5): blocks_refined when its
                      key is the first distinct key in its stage (or the
                      source cannot say what was placed ahead of it); when a
                      DIFFERENT key sits ahead of it in the same stage, its
                      lane LEFTOVER price behind that key
                      (packing._stage_key_prices, the packer's own pricing
                      function). Headline and refined alone are EXPECTED to
                      under-predict on shared stages; only blocks_charged
                      must hit 0 under.

Rows whose stage the packer would never produce are marked placement_refused
and excluded from the gate: a key the lane simulation cannot fit into the
stage (the 62-byte mixed-key net was retired 2026-09-29). (Until
2026-09-28 blocks_charged instead added the fitted crowded-stage margin, +1
to a non-first key when two different keys filled more than 58 bytes; before
2026-09-25, the per-key SATURATION margin. Both retired.)

ONE ADAPTER PER SOURCE CSV, feeding ONE shared scoring core (score_observation
below) -- the files' schemas differ enough (single key vs. two, an explicit
measured_start_group column vs. none, a JSON layout blob) that a common
adapter would be more contorted than one small adapter each (spec Sec 6.4).

WHICH KEY IS "NOT FIRST", AND WHAT IS AHEAD OF IT, resolved per source:

  * tcam_discount_scan, tcam_offset_scan, tcam_offset_probe,
    tcam_mixed_key_cap_sweep/onset, tcam_spacer_sweep: the probe (key 'a')
    carries its own measured_start_group -- direct per-table ground truth,
    not_first = measured_start_group is not null and > 0, taken literally.
    Its stage-mate is the spacer, one solid field of spacer_bytes bytes
    (tcam_stretch_sweep.as_fields), whenever p4c put both in one stage --
    ahead of it when not_first, behind it otherwise.
  * tcam_version_sweep: the same, with key 'b' (fields_b) as the scored key
    'a''s stage-mate when the row's shared_stage is true.
  * tcam_phv_slice_sweep, tcam_ledger_divergence_sweep, tcam_lane_sweep,
    tcam_field_count_sweep: single-key probes by construction -- nothing
    ahead, charged the refined price.
  * tcam_stretch_sweep: ground truth and per-table stage placement come from
    the `layout` JSON (its blocks_a/blocks_b columns are a STALE predicted
    price). The file cannot say which of two co-located keys p4c served
    first, so both orders are scored and the WORSE one that the packer could
    emit is charged; a stage of two keys up to 64 bytes is priced by the lanes.
  * tcam_offset_harvest and tcam_heldout_harvest record measured_start_group
    but not the other keys of the stage, so no stage-mate is known and
    blocks_charged is the refined price. The design-level lane simulation on
    those archives is scored end to end by scripts/validation_table.py.

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
from src.p4model.packing import _stage_key_prices  # noqa: E402

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
    if isinstance(value, float) and value.is_integer():
        # a single-field column with blank cells elsewhere is read as float
        # (tcam_version_sweep's fields_b: 40.0 for the 40-bit field)
        value = int(value)
    return tuple(int(part) for part in str(value).split(",") if part.strip())


def price_in_stage(field_bit_widths, ahead=(), behind=()):
    """(charged blocks, refused) for a key sharing a stage with the keys
    `ahead` of it and `behind` it (field-bit tuples, in placement order) --
    the packer's own pricing (packing._stage_key_prices): the first key pays
    codeword_to_blocks, every later key its lane leftover price. refused is
    True when some key has no lane-legal fit (charged is then None). Keys behind this one cannot change its price, only refuse
    the stage."""
    field_bit_widths = tuple(field_bit_widths)
    sequence = (tuple(tuple(key) for key in ahead) + (field_bit_widths,)
                + tuple(tuple(key) for key in behind))
    prices = _stage_key_prices((), sequence)
    if prices is None:
        return None, True
    return prices[len(ahead)], False


def score_observation(source, identifier, field_bit_widths, observed_blocks,
                      not_first, note="", stage_mates=(), orders=None):
    """The one scoring core every adapter feeds. field_bit_widths is the
    table's key, exactly what tables.codeword_to_blocks takes. not_first is
    this specific table's OWN evidence of whether its key was first in its
    stage -- never a row-level "sharing was possible" flag.

    stage_mates names the OTHER keys p4c put in the same stage (field-bit
    tuples), when the source records them. They are placed ahead of this key
    when not_first, behind it otherwise, and the table is charged what the
    packer would charge it there (price_in_stage): its lane leftover price
    behind them, or its refined price as the first key. A table with no
    recorded stage-mates is charged the refined price.

    orders (optional) replaces that for a source that cannot say which of
    several co-located keys p4c served first: a list of candidate
    (ahead, behind) splits of the stage-mates. The WORST charge among the
    orders the packer could emit is used.

    placement_refused marks an observation whose STAGE the packer would never
    produce under any candidate order (price_in_stage). Its refined price is
    still recorded, but under_predictions() skips it: the model avoids that
    placement rather than pricing it (dsp41/dsp42, 63-64 combined bytes).

    `saturated` (crossbar_capacity(g) == B) is still recorded per row for
    reading the table, but charges nothing."""
    field_bit_widths = tuple(field_bit_widths)
    key_bytes = codeword_fields_to_bytes_from_bits(field_bit_widths)
    headline = codeword_to_blocks_headline(field_bit_widths)
    refined = codeword_to_blocks(field_bit_widths)
    saturated = crossbar_capacity(refined) == key_bytes
    if orders is None:
        stage_mates = tuple(stage_mates)
        orders = [(stage_mates, ()) if not_first else ((), stage_mates)]
    charges = []
    for ahead, behind in orders:
        price, refused = price_in_stage(field_bit_widths, ahead, behind)
        if not refused:
            charges.append(price)
    placement_refused = not charges
    charged = max(charges) if charges else refined
    return {
        "source": source,
        "identifier": identifier,
        "field_bit_widths": field_bit_widths,
        "key_bytes": key_bytes,
        "observed_blocks": int(observed_blocks),
        "not_first": bool(not_first),
        "saturated": saturated,
        "behind_other_key": charged != refined,
        "placement_refused": placement_refused,
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
    probe geometry each sweep varied. The spacer is key 'b', one solid field
    of spacer_bytes bytes (tcam_offset_scan.run_one ->
    tcam_stretch_sweep.synthetic_program); it is the probe's stage-mate
    whenever both landed in one stage -- ahead of it when the probe did not
    start at group 0."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["probe_fields"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        spacer = (8 * int(r["spacer_bytes"]),)
        shared = bool(r["both_in_one_stage"])
        rows.append(score_observation(
            source_name, r["point_id"], field_bits, r["probe_real_blocks"],
            not_first, stage_mates=(spacer,) if shared else ()))
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
    crossbar evidence to say which of two co-located keys is truly first, so
    both orders are candidates and the worse emittable one is charged; a
    stage of two keys up to 64 bytes is priced by the lanes."""
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
                         "not blocks_a/blocks_b -- see adapter docstring",
                    orders=([((), (fields[other_tag],)),
                             ((fields[other_tag],), ())]
                            if shares_stage_with_other_key else None)))
    return rows


def score_phv_slice_sweep(path):
    """24 single-key isolation probes, each key TWO fields --
    `(clean_bits, solid_bits)`, exactly the pair the probe itself builds and
    scores (`fields = (clean_bits, 8 * solid_bytes)`,
    `tcam_phv_slice_sweep.py:193`, fed straight to `codeword_to_blocks`). Do
    not merge them into one `(clean_bits + solid_bits,)` field: solid_bits is
    always a whole-byte field with no tail of its own, so merging does not
    change key_bytes (both round to the same crossbar-byte total) but it DOES
    silently drop the Sec 2.3 isolation credit, which only fires on the true
    nibble-clean field (clean_bits) -- codeword_to_blocks then charges the
    version-only-block worst case on every row, over-predicting 16 of the
    24 (C2). key_bytes is cross-checked against the CSV's own column via the
    combined total, which is round-trip safe either way. No sharing column at
    all -- the margin never applies, full stop."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        clean_bits = int(r["clean_bits"])
        solid_bits = int(r["solid_bits"])
        total_bits = clean_bits + solid_bits
        expected_key_bytes = math.ceil(total_bits / 8)
        if expected_key_bytes != int(r["key_bytes"]):
            raise ValueError(
                "tcam_phv_slice_sweep %s: clean_bits+solid_bits=%d bytes-round "
                "to %d, CSV's own key_bytes says %d" % (
                    r["point_id"], total_bits, expected_key_bytes,
                    r["key_bytes"]))
        rows.append(score_observation(
            "tcam_phv_slice_sweep", r["point_id"], (clean_bits, solid_bits),
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
    docstring, which also means not_first is False there and the key is
    charged its refined price; both rows are exact at it (3 == 3). Key 'b'
    (fields_b) is the scored key's stage-mate when shared_stage is true."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields_a"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        other = (_parse_int_list(r["fields_b"])
                 if pd.notna(r["fields_b"]) and str(r["fields_b"]).strip() else ())
        shared = bool(r["shared_stage"]) and bool(other)
        rows.append(score_observation(
            "tcam_version_sweep", r["point_id"], field_bits,
            r["real_blocks_a"], not_first,
            stage_mates=(other,) if shared else ()))
    return rows


def score_spacer_sweep(path):
    """12 rows; only the "a" (probe) key is scored -- real_blocks_b is the
    spacer table's own block count, explicitly not one of this task's 88
    distinct probe keys per the brief. measured_start_group is this file's
    per-table ground truth; the spacer, one solid field of spacer_bytes bytes,
    is the probe's stage-mate when both landed in one stage."""
    frame = pd.read_csv(path)
    rows = []
    for _, r in frame.iterrows():
        field_bits = _parse_int_list(r["fields_a"])
        not_first = _not_first_from_measured_start_group(r["measured_start_group"])
        spacer = (8 * int(r["spacer_bytes"]),)
        shared = bool(r["both_in_one_stage"])
        rows.append(score_observation(
            "tcam_spacer_sweep", r["point_id"], field_bits,
            r["real_blocks_a"], not_first,
            stage_mates=(spacer,) if shared else ()))
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
    """Rows observed at a stage the packer refuses (no lane-legal fit)."""
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
              "not_first=%s behind_other_key=%s" % (
                  r["source"], r["identifier"], r["field_bit_widths"],
                  r["key_bytes"], r["blocks_refined"], r["blocks_charged"],
                  r["observed_blocks"], r["not_first"],
                  r["behind_other_key"]))

    refused = refused_placements(rows)
    print("\n### placements the packer refuses (no lane fit): %d\n" % len(refused))
    for r in refused:
        print("  %s / %s: field_bit_widths=%s blocks_charged=%d observed_blocks=%d"
              % (r["source"], r["identifier"], r["field_bit_widths"],
                 r["blocks_charged"], r["observed_blocks"]))

    print("\nGATE (blocks_charged, 0 under-predictions on emitted placements): %s"
          % ("PASS" if not unders else "FAIL -- %d under-prediction(s)" % len(unders)))

    return rows


if __name__ == "__main__":
    main()
