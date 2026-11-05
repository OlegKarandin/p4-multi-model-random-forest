"""N-task generator compatibility gate (spec 2026-10-11, Testing item 1).

Two checks, both of which must report zero differences for N = 2:

1. ``--campaign DIR``: regenerate the P4 program of a sample of campaign rows
   from their saved forests (forests/<row_id>.joblib) exactly as
   row_artifacts._generate does, and diff it byte for byte against the
   archived designs/<row_id>.p4, apart from the range tables'
   @placement_priority, which every program since 2026-11 carries and no
   archived one does. The model side is checked too:
   model_breakdown(parse_program(<archived p4>)) must equal the archived
   model.json's stage_depth / blocks / tables.

2. ``--snapshot DIR --out FILE`` / ``--compare FILE``: model_breakdown of
   every p4_src/<row>.p4 of a calibration archive, written as JSON (before a
   refactor) or compared with one written earlier (after it).

The campaign data is not in git, so this is a command, not a test.

Usage (PolimiML env):
  python -m scripts.ntask_regression --campaign <run dir> [--all]
  python -m scripts.ntask_regression --snapshot <archive> --out before.json
  python -m scripts.ntask_regression --snapshot <archive> --compare before.json
"""
import argparse
import json
import os
import re
import sys
import tempfile

from p4gen.p4_replay import model_breakdown, parse_program

# Arm slug -> encoding, as in the campaign manifest (independent: disjoint;
# joint-off and its aligned twins: joint).
ARM_ENCODING = {'independent': 'disjoint', 'joint-off': 'joint',
                'joint-off-al': 'joint'}
SAMPLE_M = ('015', '035', '075', 'inf')
SAMPLE_SPLITS = (0, 7, 13, 24)
SAMPLE_K = (1, 2, 4, 6, 9, 12, 14, 17)


def _without_range_priority(program_bytes):
    """The program without the range tables' @placement_priority(3) (the
    two-task value, program.range_table_priority): the one line a current
    generator adds to an archived program."""
    return re.sub(rb'@placement_priority\(3\)\n\s*', b'', program_bytes)


def _arm_of(row_id):
    return row_id.split('_M')[0]


def sample_rows(run_dir, take_all=False):
    designs = os.path.join(run_dir, 'designs')
    rows = sorted(n[:-3] for n in os.listdir(designs) if n.endswith('.p4'))
    if take_all:
        return rows
    keep = []
    for row in rows:
        arm, rest = row.split('_M')
        m_token, split, k = rest.split('_')
        if (m_token in SAMPLE_M and int(split[1:]) in SAMPLE_SPLITS
                and int(k[1:]) in SAMPLE_K):
            keep.append(row)
    return keep


def regenerate(run_dir, row_id, scratch):
    from p4gen.build_p4_script import generate_P4_code
    from src.training.feature_selection import (
        _derive_feature_intervals, _derive_joint_feature_intervals)
    from src.training.row_artifacts import load_forests
    stored = load_forests(os.path.join(run_dir, 'forests', row_id + '.joblib'))
    model_app, model_ddos = stored['app'], stored['ddos']
    names_app, names_ddos = stored['features_app'], stored['features_ddos']
    if ARM_ENCODING[_arm_of(row_id)] == 'joint':
        intervals_app = intervals_ddos = _derive_joint_feature_intervals(
            model_app, model_ddos, names_app, names_ddos)
    else:
        intervals_app = _derive_feature_intervals(model_app, names_app)
        intervals_ddos = _derive_feature_intervals(model_ddos, names_ddos)
    written = generate_P4_code(
        3, 2, model_app, model_ddos,
        feature_intervals_app=intervals_app, feature_intervals_ddos=intervals_ddos,
        output_dir=scratch + os.sep, output_filename=row_id + '.p4',
        selected_features_app=names_app, selected_features_ddos=names_ddos)
    with open(written, 'rb') as handle:
        data = handle.read()
    # generate_P4_code writes in text mode, so on Windows every newline comes
    # out as CRLF; the archive was generated on Linux. Undo exactly that
    # translation and nothing else (the archived files hold no CR at all).
    if os.linesep != '\n':
        data = data.replace(os.linesep.encode(), b'\n')
    return data


def check_campaign(run_dir, take_all=False):
    rows = sample_rows(run_dir, take_all)
    p4_diffs, model_diffs = [], []
    with tempfile.TemporaryDirectory() as scratch:
        for index, row in enumerate(rows):
            archived_path = os.path.join(run_dir, 'designs', row + '.p4')
            with open(archived_path, 'rb') as handle:
                archived = handle.read()
            if _without_range_priority(regenerate(run_dir, row, scratch)) != archived:
                p4_diffs.append(row)
            with open(os.path.join(run_dir, 'designs', row + '.model.json'),
                      encoding='utf-8') as handle:
                stored = json.load(handle)
            replay = model_breakdown(parse_program(archived_path), row)
            if any(replay[key] != stored[key]
                   for key in ('stage_depth', 'blocks', 'tables')):
                model_diffs.append(row)
            if (index + 1) % 50 == 0:
                print(f'{index + 1}/{len(rows)} p4 diffs {len(p4_diffs)} '
                      f'model diffs {len(model_diffs)}', flush=True)
    arms = sorted({_arm_of(r) for r in rows})
    print(f'campaign rows {len(rows)} (arms {", ".join(arms)}); '
          f'P4 byte differences {len(p4_diffs)}; model differences {len(model_diffs)}')
    for row in p4_diffs:
        print('P4 DIFF', row)
    for row in model_diffs:
        print('MODEL DIFF', row)
    return not p4_diffs and not model_diffs


def snapshot(archive_dir):
    src = os.path.join(archive_dir, 'p4_src')
    rows = sorted(n[:-3] for n in os.listdir(src) if n.endswith('.p4'))
    result = {}
    for row in rows:
        # A probe program the replay refuses is recorded as its error, so a
        # refactor that changes WHICH programs are refused shows up too.
        try:
            result[row] = model_breakdown(parse_program(os.path.join(src, row + '.p4')), row)
        except Exception as exc:  # noqa: BLE001 -- recorded, compared
            result[row] = {'error': f'{type(exc).__name__}: {exc}'}
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--campaign', metavar='DIR')
    parser.add_argument('--all', action='store_true',
                        help='every design of the run, not the sample')
    parser.add_argument('--snapshot', metavar='ARCHIVE')
    parser.add_argument('--out', metavar='FILE')
    parser.add_argument('--compare', metavar='FILE')
    args = parser.parse_args(argv)
    ok = True
    if args.campaign:
        ok &= check_campaign(args.campaign, args.all)
    if args.snapshot:
        current = snapshot(args.snapshot)
        if args.out:
            with open(args.out, 'w', encoding='utf-8') as handle:
                json.dump(current, handle, sort_keys=True, indent=1)
            print(f'snapshot of {len(current)} designs written to {args.out}')
        if args.compare:
            with open(args.compare, encoding='utf-8') as handle:
                before = json.load(handle)
            changed = sorted(r for r in set(before) | set(current)
                             if before.get(r) != current.get(r))
            print(f'archive designs {len(current)}; model breakdown differences '
                  f'{len(changed)}')
            for row in changed:
                print('BREAKDOWN DIFF', row)
            ok &= not changed
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
