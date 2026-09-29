"""Writes results/compiler_calibration_pinned.csv: p4c's own stage and TCAM
numbers for the 43 designs of the PINNED archive,
results/compiler_calibration_pinned/ (p4_src/ + compiles/).

WHAT THE ARCHIVE IS. The 43 designs of results/compiler_calibration_v6,
compiler_calibration_extra and tcam_margin_screen, regenerated with the
generator's @placement_priority (ddos trees 2, app trees 1) and
@pa_no_overlay on every class_tree_* field, and compiled one at a time --
"arm D" of the model audit's key-order experiment
(reviews/model_audit_2026-09-27.md §7.3, originally
reviews/model_audit_scratch/priority_exp_ddos_first_noovct/, copied here
byte for byte). The pragmas pin the order p4c places the classification
trees in, which is what the ordered stage simulation in
src/p4model/packing.py replays; archives compiled WITHOUT them
(compiler_calibration_v6, _extra, tcam_margin_screen) are pre-pragma and
informational only.

GROUND TRUTH ONLY. Parsed with the same parser every other calibration CSV
uses (src.p4gen.p4_compile.parse_compile_logs, as compiler_calibration.
run_one_row stores it): stages_real is the LAST placement round's stage
count (table_summary.log), tcam_real / sram_real / map_ram_real are the
committed totals from mau.resources.log -- blank when the backend never
allocated (the design needs more than Tofino's 12 stages). Cross-checked
here against scripts/p4_artifact_replay's committed_stages_real and the sum
of _committed_blocks, the two readings the replay scripts use; a
disagreement raises rather than writing a CSV the two paths would read
differently. No prediction columns: scripts/validation_table.py recomputes
predictions with current code every time.

Run (from the repository root):
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/build_pinned_calibration_csv.py
"""
import argparse
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import pandas as pd  # noqa: E402

from scripts.p4_artifact_replay import (  # noqa: E402
    _committed_blocks, committed_stages_real)
from src.p4gen.p4_compile import parse_compile_logs  # noqa: E402

PINNED_ROOT = os.path.join(ROOT, "results", "compiler_calibration_pinned")
PINNED_CSV = os.path.join(ROOT, "results", "compiler_calibration_pinned.csv")


def group_of(row_id):
    """'joint' or 'independent' -- the P4 encoding family the design was
    generated with, read from its row id (joint_*, heldout_joint-*_*,
    independent_*, margin_independent_*, heldout_independent_*)."""
    return "independent" if "independent" in row_id else "joint"


def pinned_rows(root=PINNED_ROOT):
    """One ground-truth record per compiled design under root/compiles/, in
    sorted row-id order."""
    records = []
    for row_id in sorted(os.listdir(os.path.join(root, "compiles"))):
        compile_dir = os.path.join(root, "compiles", row_id)
        if not os.path.isfile(os.path.join(root, "p4_src", row_id + ".p4")):
            raise ValueError("%s: compiled but no p4_src/%s.p4" % (row_id, row_id))
        result = parse_compile_logs(compile_dir)
        logs = os.path.join(compile_dir, "pipe", "logs")
        stages = committed_stages_real(logs)
        blocks = _committed_blocks(logs)
        committed_tcam = sum(blocks.values()) if blocks else None
        if result.stages != stages or result.tcam != committed_tcam:
            raise ValueError(
                "%s: parse_compile_logs says stages=%s tcam=%s, the replay "
                "parsers say stages=%s tcam=%s" % (
                    row_id, result.stages, result.tcam, stages, committed_tcam))
        records.append({
            "row_id": row_id,
            "group": group_of(row_id),
            "stages_real": result.stages,
            "tcam_real": result.tcam,
            "sram_real": result.sram,
            "map_ram_real": result.map_ram,
            "allocated": blocks is not None,
        })
    return records


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default=PINNED_ROOT)
    parser.add_argument("--out", default=PINNED_CSV)
    args = parser.parse_args(argv)
    frame = pd.DataFrame(pinned_rows(args.root))
    frame.to_csv(args.out, index=False)
    print("wrote %s: %d designs, %d with a TCAM allocation"
          % (args.out, len(frame), int(frame["allocated"].sum())))


if __name__ == "__main__":
    main()
