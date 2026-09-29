"""Does the generator emit the pins the evidence archive was compiled with?

results/compiler_calibration_pragmas_2026_09_29/ holds 73 programs with the
spec 2026-09-29 pins INJECTED into frozen P4 (inject_and_compile.py there),
not generated. Two checks close that gap:

  archive  -- for all 73 designs, rebuild the generator's resolved plan from
              the frozen program's own key structure (which code_* field each
              task's trees key, and its width) and run
              build_p4_script.code_field_container_sizes on it; the result must
              equal manifest.json's pa_container_size_code_pins, and the
              declared code_* count its pa_no_overlay_code_fields. Fast, no
              campaign data.
  regen    -- for the named designs, refit the forests from the campaign
              backup, run the real generator, and compare with the archived
              injected program: non-pragma lines identical in order, pragma
              lines equal as multisets (spec: "byte-identical modulo pragma
              order"). Slow; needs the campaign data and backup.

Run (from the repository root):
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/verify_pragma_generation.py archive
  "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" scripts/verify_pragma_generation.py regen \\
      heldout_joint-off_M100_k14_s16 heldout_independent_M150_k14_s13 margin_independent_M50_k8_s13
"""
import argparse
import collections
import json
import os
import re
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from scripts.p4_artifact_replay import _p4_table_keys  # noqa: E402
from src.p4gen.build_p4_script import code_field_container_sizes  # noqa: E402

ARCHIVE = os.path.join(ROOT, "results", "compiler_calibration_pragmas_2026_09_29")
_ROW = re.compile(r"^(?:heldout_|margin_)?(.+)_M(\d+)_k(\d+)_s(\d+)$")


def pins_from_program(p4_path):
    """(pins, declared): what code_field_container_sizes gives for the key
    structure of the program at p4_path, and the code_* fields it declares."""
    with open(p4_path, encoding="utf-8") as handle:
        text = handle.read()
    declared = {name: int(width) for width, name in
                re.findall(r"bit<(\d+)> (code_[A-Za-z0-9_]+);", text)}
    tables, _widths, bits = _p4_table_keys(p4_path)
    keyed = collections.defaultdict(set)   # code field -> tasks keying it
    for name, keys in tables.items():
        match = re.match(r"get_classification_tree_(app|ddos)_\d+$", name)
        if match:
            for key in keys:
                keyed[key].add(match.group(1))
    plan = collections.OrderedDict()
    for field, width in declared.items():
        # Only len(intervals) is read, so a placeholder list of the right
        # length stands in for the real intervals.
        plan[field[len("code_"):]] = (None, [None] * (width + 1), keyed.get(field, set()))
    tasks = sorted({task for tasks in keyed.values() for task in tasks})
    pins = {"code_" + name: sizes
            for name, sizes in code_field_container_sizes(plan, tasks).items()}
    return pins, set(declared)


def check_archive(root=ARCHIVE):
    with open(os.path.join(root, "manifest.json"), encoding="utf-8") as handle:
        manifest = json.load(handle)
    problems = []
    for design in manifest["designs"]:
        row_id = design["row_id"]
        pins, declared = pins_from_program(os.path.join(root, "p4_src", row_id + ".p4"))
        if pins != design["pa_container_size_code_pins"]:
            problems.append("%s: pins differ: generator %s, injected %s" % (
                row_id, pins, design["pa_container_size_code_pins"]))
        if len(declared) != design["pa_no_overlay_code_fields"]:
            problems.append("%s: %d code_* fields, %d no-overlay pragmas injected" % (
                row_id, len(declared), design["pa_no_overlay_code_fields"]))
    return problems


def _split(text):
    body, pragmas = [], collections.Counter()
    for line in text.splitlines():
        if line.strip().startswith("@"):
            pragmas[line.strip()] += 1
        else:
            body.append(line)
    return body, pragmas


def regenerate_and_compare(row_id, root=ARCHIVE):
    """Refit, generate, compare with root/p4_src/<row_id>.p4 (see module
    docstring). Unaligned arms only: the three designs this is run on are
    independent or joint-off, so no alignment step is needed."""
    from scripts.compiler_calibration import CAMPAIGN_BACKUP_DIR
    from scripts.replay_alignment import load_backup, refit_pair
    from src.main import load_campaign_data
    from src.p4gen.build_p4_script import (
        generate_P4_code, get_feature_intervals, get_joint_feature_intervals)

    arm, M, k, split = _ROW.match(row_id).groups()
    if arm.startswith("joint-d"):
        raise ValueError("%s: aligned arm; regen does not replay alignment" % row_id)
    backup = load_backup(CAMPAIGN_BACKUP_DIR)
    hits = backup[(backup["arm_slug"] == arm) & (backup["M"] == int(M))
                  & (backup["k"] == int(k)) & (backup["split"] == int(split))]
    if len(hits) != 1:
        return ["%s: %d campaign rows match" % (row_id, len(hits))]
    row = hits.iloc[0]
    model_app, model_ddos, *_ = refit_pair(row, load_campaign_data())
    names_app = row["features_app"].split(";")
    names_ddos = row["features_ddos"].split(";")
    if arm == "independent":
        intervals_app = get_feature_intervals(model_app, names_app)
        intervals_ddos = get_feature_intervals(model_ddos, names_ddos)
    else:
        intervals_app = intervals_ddos = get_joint_feature_intervals(
            model_app, names_app, model_ddos, names_ddos)
    with tempfile.TemporaryDirectory() as scratch:
        path = generate_P4_code(
            3, 2, model_app, model_ddos,
            feature_intervals_app=intervals_app, feature_intervals_ddos=intervals_ddos,
            output_dir=scratch + os.sep, output_filename=row_id + ".p4",
            selected_features_app=names_app, selected_features_ddos=names_ddos)
        with open(path, encoding="utf-8") as handle:
            generated = handle.read()
    with open(os.path.join(root, "p4_src", row_id + ".p4"), encoding="utf-8") as handle:
        archived = handle.read()
    gen_body, gen_pragmas = _split(generated)
    arc_body, arc_pragmas = _split(archived)
    problems = []
    if gen_body != arc_body:
        first = next((i for i, (a, b) in enumerate(zip(gen_body, arc_body)) if a != b),
                     min(len(gen_body), len(arc_body)))
        problems.append("%s: non-pragma line %d differs:\n  generated: %r\n  archived:  %r" % (
            row_id, first, gen_body[first] if first < len(gen_body) else None,
            arc_body[first] if first < len(arc_body) else None))
    if gen_pragmas != arc_pragmas:
        problems.append("%s: pragma sets differ: only generated %s; only archived %s" % (
            row_id, sorted((gen_pragmas - arc_pragmas).elements()),
            sorted((arc_pragmas - gen_pragmas).elements())))
    return problems


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("mode", choices=("archive", "regen"))
    parser.add_argument("rows", nargs="*")
    args = parser.parse_args(argv)
    if args.mode == "archive":
        problems = check_archive()
    else:
        problems = [p for row_id in args.rows for p in regenerate_and_compare(row_id)]
    for problem in problems:
        print(problem)
    print("PASS" if not problems else "FAIL -- %d problem(s)" % len(problems))
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
