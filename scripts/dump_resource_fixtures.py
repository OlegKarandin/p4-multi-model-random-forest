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

RESOLVED FINDING (Mechanism G superseded, 2026-09-07). This fixture used to
record a 5-row over-application of "Mechanism G" -- the rule that charged +1
TCAM block to any ragged key landing on an ODD crossbar group offset. It is
now resolved, and the rule is gone.

What settled it was reading p4c's own per-table numbers instead of comparing
row totals: `resources.json` records both the TCAM units each table got
(`tcams.tcams`) and the exact crossbar bytes its key was given (`xbar_bytes`,
byte_type 'ternary'). Over the 100 classification tables of the 19 compiles in
results/compiler_calibration_v6/, exactly 3 cost more than
`ceil(8 * key_bytes / 44)` blocks -- independent_low_sd5's three ddos trees --
and Mechanism G got that stage right only by accident: it charged the app key
at group offset 3, where the compiler charged the ddos key at offset 0.

`tables.version_block_penalty` replaces it with the physical question -- does
any crossbar midbyte in the key's group run keep a free nibble for the
mandatory 2-bit --version-- field? That predicts all 100 tables exactly, and
takes usage.blocks from 12/17 to 17/17 against tcam_real with no
under-prediction. See tests/test_tcam_block_ledger.py and
reviews/p4_tofino_reference.md Appendix B "Mechanism G".

SUPERSEDED IN TURN (2026-09-21). `tables.version_block_penalty` and the whole
offset premise under it are gone -- reading p4c's own assembly showed a block
may pair with ANY of a stage's midbytes and groups need not be consecutive, so
a key's own price cannot depend on where it starts. The paragraph above is
kept as the record of how the earlier retraction was settled, not as a
description of live code. What prices a key now is `tables.codeword_to_blocks`
(no offset argument), and the sharing effect lives in
`src/p4model/packing.py`'s stage-sharing margin. That change re-pinned two of
this fixture's rows: see `known_findings` entry
`stage_sharing_margin_2026_09_21` below, which records an OVER-prediction
rather than a correction.

`ternary_ragged` is no longer serialized: the penalty prices a key by its
field BIT widths, and key_field_sets already carries those.
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
                'id': 'mechanism_g_superseded_2026_09_07',
                'rows': [],
                'row_deltas': {},
                'summary': (
                    "RESOLVED. Mechanism G -- '+1 TCAM block for a ragged key "
                    "at an odd crossbar group offset' -- was replaced on "
                    "2026-09-07 by tables.version_block_penalty, which asks "
                    "whether any midbyte in the key's crossbar group run keeps "
                    "a free nibble for the mandatory 2-bit --version-- field. "
                    "Scored per TABLE against p4c's own resources.json over "
                    "the 100 classification tables of the 19 compiles in "
                    "results/compiler_calibration_v6/, the new rule is exact: "
                    "3 penalties predicted, 3 observed, no false alarms and no "
                    "misses. usage.blocks goes from 12/17 to 17/17 exact "
                    "against tcam_real, with no under-prediction; stage_depth "
                    "is unchanged at 18/19. The old rule's 5-row "
                    "over-application (independent_low_sd6 +3, "
                    "independent_low_sd7 / independent_high_sd6 / "
                    "independent_high_sd7 / independent_high_sd8 +1 each) is "
                    "gone, and so is the accident that made it look right on "
                    "independent_low_sd5: p4c charges that stage's ddos key at "
                    "group offset 0, not its app key at offset 3."),
            },
            {
                'id': 'stage_sharing_margin_2026_09_21',
                'rows': ['independent_high_sd7', 'independent_high_sd8'],
                'row_deltas': {'independent_high_sd7': 1,
                               'independent_high_sd8': 3},
                'summary': (
                    "RE-PINNED, and the delta is an OVER-prediction -- read "
                    "this before treating these two rows as ground truth. The "
                    "2026-09-20 TCAM block model rewrite deleted the "
                    "per-table offset mechanism (codeword_to_blocks has no "
                    "start_group parameter any more; version_block_penalty "
                    "and version_block_delta are gone) and moved the real "
                    "effect it was chasing into a stage-PLACEMENT margin in "
                    "src/p4model/packing.py's charged(): a table whose key is "
                    "not the first distinct key in its stage, and whose "
                    "standalone price leaves no spare whole-byte crossbar "
                    "slot (crossbar_capacity(g) == B), pays one extra TCAM "
                    "block for the mandatory 2-bit --version-- field (spec "
                    "Sec 13.2 'Effect 4'). The margin is REQUIRED: without it "
                    "independent_low_sd9's stage_depth under-predicts, 10 "
                    "against p4c's 11, which this project forbids. Its cost "
                    "is these two rows, both disjoint designs whose "
                    "classification stage really does hold two keys in p4c's "
                    "own committed placement (resources.json stage 9) and "
                    "where p4c charged nothing extra: sd7's 10-byte app key "
                    "is crossbar_capacity(2) == 10 and sd8's 16-byte ddos key "
                    "is crossbar_capacity(3) == 16, so the margin fires on "
                    "each and the model reports 28 against tcam_real 27 and "
                    "41 against 38. That contradicts Sec 13.2's own prose "
                    "('on the 9 mixed archived stages no second key is "
                    "saturated'), which was never re-scored against the "
                    "archive under this spelling of the saturation test. "
                    "usage.blocks goes from 17/17 exact to 15/17, 0 under, 2 "
                    "over; stage_depth stays 18/19, 0 under. Never "
                    "under-predicting outranks exactness here, so the margin "
                    "lands and the fixture follows it -- but these are a "
                    "known, quantified over-prediction, not a correction, and "
                    "scripts/validation_table.py still reports both as "
                    "OVER."),
            },
            {
                'id': 'crowded_stage_rule_2026_09_25',
                'rows': ['independent_low_sd12'],
                'row_deltas': {'independent_low_sd12': 2},
                'summary': (
                    "RE-PINNED, and the delta is an OVER-prediction. The "
                    "crowded-stage rule (src/p4model/target.py "
                    "TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE = 58, "
                    "TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE = 62) charges "
                    "every non-first key +1 when two different keys fill more "
                    "than 58 crossbar bytes of one stage, and refuses more "
                    "than 62. independent_low_sd12's real stage 10 shares a "
                    "40-byte app key and a 20-byte ddos key at 60 bytes and "
                    "p4c charged nothing, so this row now reads blocks 80 "
                    "against tcam_real 78 and stage_depth 13 against "
                    "stages_real 12 (a feasible design priced past the "
                    "12-stage ceiling). Kept because the same 59-62-byte band "
                    "costs +1 on independent_low_sd5's real (54, 56) key "
                    "behind a spacer (results/tcam_mixed_key_cap_sweep.csv) "
                    "and on the real campaign design M150_k5_s12 "
                    "(results/tcam_margin_screen_compiled.csv), and byte "
                    "totals cannot separate the paying stages from this free "
                    "one; without the rule those are under-predictions. Other "
                    "rows unchanged."),
            },
            {
                'id': 'saturation_margin_retired_2026_09_25',
                'rows': ['independent_high_sd7', 'independent_high_sd8'],
                'row_deltas': {'independent_high_sd7': -1,
                               'independent_high_sd8': -3},
                'summary': (
                    "RE-PINNED to exact. 'stage_sharing_margin_2026_09_21' "
                    "above over-charged these two rows +1/+3 because a "
                    "SATURATED key shared a stage (22 and 35 combined bytes) "
                    "and p4c charged nothing. That per-key saturation margin "
                    "is retired: every observation on disk showed crowding "
                    "(> 58 combined distinct-key bytes), not saturation, "
                    "deciding who pays -- the saturated 16-byte probe paid "
                    "nothing even at 59-64 bytes "
                    "(results/tcam_mixed_key_cap_sweep.csv), and the one "
                    "saturated key that did pay, (179, 204) beside a 12-byte "
                    "key, sat in a crowded 61-byte stage the crowded-stage "
                    "rule charges anyway. blocks now 28 -> 27 and 41 -> 38, "
                    "matching tcam_real; stage_depth unchanged."),
            },
            {
                'id': 'any_order_fit_rule_2026_09_27',
                'rows': ['independent_low_sd12'],
                'row_deltas': {'independent_low_sd12': -1},
                'summary': (
                    "RE-PINNED, and the delta is an IMPROVEMENT. "
                    "src/p4model/packing.py's fits() used to require EVERY "
                    "ordering of a stage's distinct keys to pack before a "
                    "shard could join it, and stage_charged_blocks() charged "
                    "the largest total over ALL orderings; both now only "
                    "require/consider orderings that actually fit_two_columns "
                    "(reviews/final_model_check_2026-09-27.md section 1b). "
                    "independent_low_sd12's real stage 10 packs one order "
                    "(ddos key first, no margin; app key second, +1 margin) "
                    "but not the other (app key first; ddos key pays the "
                    "margin and overflows the column budget) -- requiring "
                    "every order wrongly called the whole design infeasible "
                    "one stage too deep. This row now reads ternary blocks 69 "
                    "against tcam_real (unchanged), stage_depth 12 against "
                    "stages_real 12 (EXACT, was 13, the design's own 12-stage "
                    "ceiling) and usage.blocks 79 against 78 (was 80). Every "
                    "other row is unchanged: no other archived stage has two "
                    "different keys where only one ordering fits. Matches the "
                    "review's own scratch experiment exactly: fitted "
                    "stage_depth 17/19 -> 18/19, this row's blocks error "
                    "+2 -> +1."),
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
