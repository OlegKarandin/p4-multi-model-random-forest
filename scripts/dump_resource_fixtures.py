"""Dumps tests/fixtures/resource_model_golden.json: a golden fixture pinning
every predicted number the resource model produces, over all 19 real rows of
the compiler-calibration sample (results/campaign_backup_20260825).

WHY THIS EXISTS. Task 11 of the p4model extraction plan (Spec 4.6). Every
prior calibration finding (see CLAUDE.md's compiler-calibration note) lives in
committed CSVs that compare the model's predictions against a real p4c
compile. Nothing pins whether a REFACTOR of src/p4model/ changes those
predictions at all -- and re-running the real compiler to check would take
the several minutes this script itself takes, times however many refactor
steps land. This script serializes the resource model's inputs and outputs at
the `evaluation._pool_inputs` / `src.p4model.usage.assemble_usage` seam, once,
so tests/test_resource_model_golden.py can replay all 19 rows through the
packing and accounting core with NO sklearn, NO fitted forests, and NO
campaign backup at test time -- any diff there means the restructuring
changed a prediction, full stop.

Key-field SETS are interned per row (not per table): `ternary_fields` is the
same frozenset repeated once per tree in a model, and a 15-feature model's
interval lists are long, so storing it once per table would inflate the file
by orders of magnitude for nothing. Field ids are opaque integers assigned in
first-seen order; `field_names` records a human-readable label for each,
never read back by the replay test. Widths are stored in BITS (spec 4.5);
bytes are ceil(bits / 8), reconstructed on load, and this script asserts that
identity before writing so a mismatch fails at dump time rather than baking a
wrong number into the fixture.

SCOPE. Refits the 19 calibration rows' 38 forests via
scripts.replay_alignment.refit_pair (no new Optuna search) and calls
evaluation._pool_inputs / usage.assemble_usage on each -- no real p4c compile,
no production code touched. This is a one-off data-generation step, not part
of the test suite; the fixture it produces is what the suite actually reads.

Run (from the repository root; takes several minutes -- 38 forests refit):
  PYTHONPATH=. "C:/Users/olegk/miniconda3/envs/PolimiML/python.exe" \\
      scripts/dump_resource_fixtures.py --out tests/fixtures/resource_model_golden.json

KNOWN FINDING (Mechanism G over-application, unresolved, out of scope for this
Tier 1 extraction -- tracked, not fixed here). Cross-checking this fixture's
`usage.blocks` against results/compiler_calibration_v6.csv's `blocks` column
turned up 5 rows that disagree (this model's current blocks always higher,
by +1 to +3 depending on the row), beyond the one already-documented
independent_low_sd5 exception (where the CSV's 13 is itself stale and the
corrected value is 16, per results/compiler_calibration_verify.csv and
CLAUDE.md): independent_low_sd6 (+3), independent_low_sd7 (+1),
independent_high_sd6 (+1), independent_high_sd7 (+1), independent_high_sd8
(+1). All 5 are group='independent' (encoding='disjoint') with every one of
their ternary tables' ternary_ragged=True; every 'joint' row agrees with the
CSV exactly.

Diagnosed mechanistically (no live p4c recompile): for all 5 rows,
`range_blocks + sum(block_count for block_count, _ in ternary_table_specs)`
(the PRE-Mechanism-G naive per-table sum this model used before
StagePlan.blocks/stage_charged_blocks existed) reproduces the CSV's stale
`blocks` value EXACTLY -- and the CSV's value is independently confirmed to
equal `tcam_real` (real p4c ground truth) on all 5 rows. So unlike
independent_low_sd5 (where the naive pre-fix sum was the one wrong against
ground truth, and Mechanism G's charge corrected it to match), on these 5
rows it is the CURRENT model's Mechanism-G-charged `ternary_plan.blocks` that
disagrees with real hardware, entirely via the ragged-key group-offset
penalty (crossbar_stages_needed's `ragged_keys`/`charged`/`stage_charged_
blocks` logic in src/p4model/packing.py) -- i.e. Mechanism G appears to
OVER-fire on these 5 rows' stages, charging block(s) real p4c does not. The
per-row delta is not uniformly +1: independent_low_sd6's single occupied
ternary stage is charged +3 over its naive sum where the other 4 rows are
each +1 (on 1-2 occupied stages) -- the exact per-stage mechanics of that
larger delta were not further isolated here (out of scope for this Tier 1
task; a candidate follow-up for whoever picks up this finding).

This is a pre-existing property of the ALREADY-LANDED Mechanism G rule,
present in the working tree before this extraction plan's Task 1, not
something Tasks 1-10's verbatim relocation introduced -- recalibrating the
rule would be a genuine model-behavior change, out of scope for a Tier 1
plan whose whole premise is "move verbatim, prove nothing changed" (see
CLAUDE.md and reviews/p4_tofino_reference.md Sec 7 "Mechanism G" for the
rule's own history). This fixture therefore pins the model's CURRENT
predictions on these 5 rows (22, 39, 24, 28, 39) as-is, not a hand-corrected
value -- exactly the numbers a Tier 2 validation-table comparison against
tcam_real (scripts/validation_table.py, a separate task) is meant to surface
as a finding, not something this fixture should paper over.
"""
import argparse
import datetime
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy
import sklearn

from scripts.compiler_calibration import (
    CAMPAIGN_BACKUP_DIR,
    add_group_and_band,
    build_sample,
)
from scripts.replay_alignment import load_backup, refit_pair
from src.main import load_campaign_data
from src.p4gen.evaluation import _pool_inputs
from src.p4model.errors import CrossbarKeyTooWide
from src.p4model.program import FEATURE_VALUE_BIT_WIDTH
from src.p4model.usage import assemble_usage

DEFAULT_OUT = 'tests/fixtures/resource_model_golden.json'
USE_DEFAULT_ACTION_DISCOUNT = False  # run_one_row's own value; see this
                                     # module's docstring and _metadata.

# build_sample's default (group x k_band x stage_depth) cross product names
# 24 cells; 4 of them have NO archived row at all (structurally absent from
# results/campaign_backup_20260825, reviews/p4_tofino_reference.md's
# "4 missing strata") and always come back from build_sample's `missing` --
# that is an established, permanent fact about the archive, not something
# this dump run could ever fix, so it is not grounds to abort here. Of the 20
# cells that DO have an archived row, one -- `joint_low_sd8` -- refits fine
# but its own refit pair raises CrossbarKeyTooWide (65 crossbar bytes, cap
# 64) inside _pool_inputs itself: a permanently infeasible design, not a
# toolchain flake (same reference doc: "1 cell (joint_low_sd8) is permanently
# infeasible ... deterministic, will re-raise identically on every future
# run of this exact archived row"). 20 - 1 = 19, exactly the calibration
# sample's own row count -- see EXPECTED_MISSING_CELLS/EXPECTED_INFEASIBLE_ROW
# below, which build_fixture checks its run against explicitly rather than
# silently reproducing this shape by accident.
EXPECTED_MISSING_CELLS = frozenset({
    ('independent', 'high', 5), ('joint', 'high', 5),
    ('joint', 'high', 10), ('joint', 'high', 12),
})
EXPECTED_INFEASIBLE_ROW = 'joint_low_sd8'


def _field_display_name(raw_id):
    """Human-readable label for a raw field identity, for field_names only.

    A range field's raw id is already the normalised feature name (a plain
    string). A ternary field's raw id is (normalised_name, intervals) --
    only the name is kept; the intervals are what makes two same-named
    ternary fields distinct ids in the first place, and are not needed for a
    diagnostic label."""
    if isinstance(raw_id, tuple):
        return raw_id[0]
    return raw_id


def serialize_pool(pool):
    """Interns every (range_fields / ternary_fields) key-field SET pool uses.

    Returns (key_field_sets, field_names, range_key_field_set_ids,
    ternary_key_field_set_ids):
      key_field_sets            : list of [[field_id, bits], ...] lists, one
                                  per DISTINCT key-field set seen (range or
                                  ternary), in first-seen order.
      field_names               : {str(field_id): display name}, diagnostic
                                  only, never read back.
      range_key_field_set_ids   : one index into key_field_sets per range
                                  table, positionally aligned with
                                  pool['range_table_specs'].
      ternary_key_field_set_ids : likewise for pool['ternary_table_specs'].

    Asserts ceil(bits / 8) == the byte width the pool itself carries for
    every field, before anything is written -- a mismatch here means this
    script's own understanding of a field's width (FEATURE_VALUE_BIT_WIDTH
    for range; max(len(intervals) - 1, 0) for ternary, exactly
    ternary_key_fields' own formula) disagrees with the pool that produced
    the byte width, and the fixture must not paper over that."""
    field_ids = {}
    field_names = {}
    set_index = {}
    key_field_sets = []

    def intern_field(raw_id):
        if raw_id not in field_ids:
            new_id = len(field_ids)
            field_ids[raw_id] = new_id
            field_names[str(new_id)] = _field_display_name(raw_id)
        return field_ids[raw_id]

    def intern_set(fields, is_ternary):
        pairs = []
        for raw_id, byte_width in fields:
            if is_ternary:
                _name, intervals = raw_id
                bits = max(len(intervals) - 1, 0)
            else:
                bits = FEATURE_VALUE_BIT_WIDTH
            assert math.ceil(bits / 8) == byte_width, (
                'serialize_pool: field %r recorded as %d bytes but its bit '
                'width %d implies %d bytes' % (
                    raw_id, byte_width, bits, math.ceil(bits / 8)))
            pairs.append([intern_field(raw_id), bits])
        pairs.sort()
        key = tuple(tuple(pair) for pair in pairs)
        if key not in set_index:
            set_index[key] = len(key_field_sets)
            key_field_sets.append(pairs)
        return set_index[key]

    range_key_field_set_ids = [intern_set(fields, False)
                               for fields in pool['range_fields']]
    ternary_key_field_set_ids = [intern_set(fields, True)
                                 for fields in pool['ternary_fields']]
    return key_field_sets, field_names, range_key_field_set_ids, ternary_key_field_set_ids


def serialize_row(row_id, group, encoding, pool, usage, range_plan, ternary_plan):
    """One fixture row: row_id/group/encoding, the interned key-field sets,
    the pool inputs (assemble_usage's own 13-key contract, plus
    emitted_features for the interior-stages replay test), and the outputs
    (every ResourceUsage field, both StagePlans' occupied/depth/blocks/
    sorted indices)."""
    key_field_sets, field_names, range_ids, ternary_ids = serialize_pool(pool)

    inputs = {
        'range_table_specs': [list(spec) for spec in pool['range_table_specs']],
        'ternary_table_specs': [list(spec) for spec in pool['ternary_table_specs']],
        'range_key_field_set_ids': range_ids,
        'ternary_key_field_set_ids': ternary_ids,
        'range_levels': list(pool['range_levels']),
        'ternary_ragged': list(pool['ternary_ragged']),
        'interior_stages': sorted(pool['interior_stages']),
        'emitted_features': list(pool['emitted_features']),
        'register_names': list(pool['register_names']),
        'range_entries': pool['range_entries'],
        'range_blocks': pool['range_blocks'],
        'ternary_entries': pool['ternary_entries'],
        'codeword_length': pool['codeword_length'],
    }
    outputs = {
        'usage': {
            'stages': usage.stages,
            'blocks': usage.blocks,
            'stage_depth': usage.stage_depth,
            'range_entries': usage.range_entries,
            'ternary_entries': usage.ternary_entries,
            'codeword_length': usage.codeword_length,
            'register_depth': usage.register_depth,
            'register_count': usage.register_count,
            'range_depth': usage.range_depth,
            'ternary_depth': usage.ternary_depth,
            'range_tables': usage.range_tables,
            'ternary_tables': usage.ternary_tables,
        },
        'range_plan': {
            'occupied': range_plan.occupied,
            'depth': range_plan.depth,
            'indices': sorted(range_plan.indices),
            'blocks': range_plan.blocks,
        },
        'ternary_plan': {
            'occupied': ternary_plan.occupied,
            'depth': ternary_plan.depth,
            'indices': sorted(ternary_plan.indices),
            'blocks': ternary_plan.blocks,
        },
    }
    return {
        'row_id': row_id,
        'group': group,
        'encoding': encoding,
        'key_field_sets': key_field_sets,
        'field_names': field_names,
        'inputs': inputs,
        'outputs': outputs,
    }


def _metadata(campaign_dir):
    return {
        'generated': datetime.date.today().isoformat(),
        'campaign_dir': campaign_dir,
        'sklearn': sklearn.__version__,
        'numpy': numpy.__version__,
        'use_default_action_discount': USE_DEFAULT_ACTION_DISCOUNT,
        'note': ('Inputs captured at evaluation._pool_inputs; outputs from '
                 'usage.assemble_usage.'),
        'known_findings': [
            {
                'id': 'mechanism_g_over_application_2026_09_07',
                'rows': ['independent_low_sd6', 'independent_low_sd7',
                        'independent_high_sd6', 'independent_high_sd7',
                        'independent_high_sd8'],
                'row_deltas': {'independent_low_sd6': 3, 'independent_low_sd7': 1,
                              'independent_high_sd6': 1, 'independent_high_sd7': 1,
                              'independent_high_sd8': 1},
                'summary': (
                    "This fixture's usage.blocks is higher than results/"
                    "compiler_calibration_v6.csv's blocks column on these 5 "
                    "rows -- by +1 (independent_low_sd7, independent_high_sd6, "
                    "independent_high_sd7, independent_high_sd8) or +3 "
                    "(independent_low_sd6); see row_deltas above. All 5 are "
                    "group='independent'/encoding='disjoint' with every "
                    "ternary table's ternary_ragged=True; every 'joint' row "
                    "agrees with the CSV exactly. Diagnosed mechanistically: "
                    "range_blocks + the naive per-table sum of "
                    "ternary_table_specs' block counts (the PRE-Mechanism-G "
                    "formula) reproduces the CSV's stale blocks value exactly "
                    "on all 5 rows, and that CSV value is independently "
                    "confirmed to equal tcam_real (real p4c ground truth) on "
                    "all 5. So here -- unlike independent_low_sd5, where the "
                    "naive sum was the one wrong and Mechanism G's ragged-key "
                    "group-offset charge (crossbar_stages_needed's "
                    "ragged_keys/charged/stage_charged_blocks logic, "
                    "src/p4model/packing.py) corrected it to match ground "
                    "truth (13 -> 16) -- Mechanism G's charge is what "
                    "disagrees with ground truth here, appearing to over-fire "
                    "on these 5 rows' stages. The larger independent_low_sd6 "
                    "delta (+3 on a single occupied ternary stage, vs +1 on "
                    "the other rows' 1-2 occupied stages) was not further "
                    "isolated -- out of scope for this Tier 1 task. This is a "
                    "pre-existing property of the already-landed Mechanism G "
                    "rule, not a regression introduced by the p4model "
                    "extraction (Tasks 1-10 are verbatim relocations, "
                    "independently reviewed byte-for-byte faithful). "
                    "Recalibrating the rule is out of scope for this Tier 1 "
                    "extraction, whose premise is 'move verbatim, prove "
                    "nothing changed' -- tracked here as a known model-"
                    "calibration finding for a Tier 2 validation-table "
                    "comparison (scripts/validation_table.py) to surface "
                    "against tcam_real, not reconciled away in this fixture. "
                    "independent_low_sd5's own +3 (13 -> 16) is NOT part of "
                    "this finding: it is the already-documented, already-"
                    "corrected exception (results/"
                    "compiler_calibration_verify.csv)."
                ),
            },
        ],
    }


def build_fixture(campaign_dir=CAMPAIGN_BACKUP_DIR):
    """Refits all 19 calibration rows' model pairs and serializes the
    resource model's inputs/outputs for each.

    Raises RuntimeError if `missing` names any cell OTHER than the 4
    permanently-archive-absent ones (see EXPECTED_MISSING_CELLS above) --
    that would mean the archive's shape has changed since this was written
    and the 19-row assumption below needs re-checking, not silently
    adjusting. Raises whatever exception a row's own refit/_pool_inputs call
    raises UNLESS it is the one documented, permanent, deterministic
    exclusion (EXPECTED_INFEASIBLE_ROW raising CrossbarKeyTooWide) -- any
    other row raising, or that one row raising something else, means a new,
    uninvestigated finding and must not be papered over here. Finally checks
    the assembled row count is exactly 19, so a silently-shrunk (or grown)
    fixture cannot pass unnoticed."""
    backup = load_backup(campaign_dir)
    if 'infeasible' in backup.columns:
        backup = backup[backup['infeasible'].isna() | (backup['infeasible'] == '')]
    frame = add_group_and_band(backup)
    rows, missing = build_sample(frame, on_missing='skip')
    if set(missing) != EXPECTED_MISSING_CELLS:
        raise RuntimeError(
            'build_sample reported a different set of missing cells than the '
            'documented, permanent 4 (reviews/p4_tofino_reference.md) -- '
            'expected {} got {}; the archive may have changed and the 19-row '
            'assumption needs re-checking'.format(
                sorted(EXPECTED_MISSING_CELLS), sorted(missing)))

    data = load_campaign_data()
    out = []
    for row_id, group, _band, _stage_depth, archived_row in rows:
        print('refitting {} ...'.format(row_id))
        encoding = 'joint' if group == 'joint' else 'disjoint'
        model_app, model_ddos = refit_pair(archived_row, data)[:2]
        names_app = archived_row['features_app'].split(';')
        names_ddos = archived_row['features_ddos'].split(';')
        try:
            pool = _pool_inputs(model_app, model_ddos, names_app, names_ddos, encoding,
                                use_default_action_discount=USE_DEFAULT_ACTION_DISCOUNT)
            usage, range_plan, ternary_plan = assemble_usage(pool)
        except CrossbarKeyTooWide as e:
            if row_id != EXPECTED_INFEASIBLE_ROW:
                raise RuntimeError(
                    '{} raised CrossbarKeyTooWide ({}), but only {} is a '
                    'documented, permanent exclusion -- this is a new finding, '
                    'not something to skip silently'.format(
                        row_id, e, EXPECTED_INFEASIBLE_ROW))
            print('  {} is the documented permanently-infeasible row '
                  '(CrossbarKeyTooWide: {}) -- excluded, matching '
                  'reviews/p4_tofino_reference.md'.format(row_id, e))
            continue
        out.append(serialize_row(row_id, group, encoding, pool,
                                 usage, range_plan, ternary_plan))

    if len(out) != 19:
        raise RuntimeError(
            'expected exactly 19 rows in the fixture, got {}: {}'.format(
                len(out), sorted(r['row_id'] for r in out)))
    return {'metadata': _metadata(campaign_dir), 'rows': out}


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--campaign-dir', default=CAMPAIGN_BACKUP_DIR)
    parser.add_argument('--out', default=DEFAULT_OUT)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    fixture = build_fixture(args.campaign_dir)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as handle:
        json.dump(fixture, handle, indent=1, sort_keys=True)
    print('wrote {} rows to {}'.format(len(fixture['rows']), args.out))


if __name__ == '__main__':
    main()
