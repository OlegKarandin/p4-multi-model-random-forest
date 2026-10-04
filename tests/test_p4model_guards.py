"""Structural guarantees for src/p4model: each name lives in exactly one place,
and the old import paths resolve to that same object.

Identity (`is`) assertions, not equality: a re-export that rebinds rather than
aliases would still compare equal on an int, then drift silently the first time
one side is edited."""
import os
import subprocess
import sys
import textwrap

import pytest

from src.p4gen import build_p4_script as bps
from src.p4gen import evaluation as ev
from src.p4model import errors, program, target

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
HEAVY = ("sklearn", "numpy", "pandas", "scipy", "optuna", "matplotlib")


def _run_isolated(source, cwd):
    """Run `source` in a fresh interpreter, returning stdout.

    A subprocess, not an importlib dance: sklearn/numpy are already in THIS
    process's sys.modules (pytest imported evaluation), so an in-process check
    could never tell whether p4model pulled them in or found them already
    there."""
    env = dict(os.environ, PYTHONPATH=REPO_ROOT)
    result = subprocess.run([sys.executable, "-c", textwrap.dedent(source)],
                            cwd=cwd, env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()

TARGET_NAMES = (
    "TCAM_BLOCKS_PER_STAGE", "TCAM_ROWS_PER_STAGE", "TCAM_COLUMNS_PER_STAGE",
    "TCAM_BLOCK_KEY_LENGTH", "TERNARY_MATCHING_ENTRIES_PER_BLOCK",
    "TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE",
    "TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE", "MAX_CODEWORD_LENGTH",
    "METER_ALUS_PER_STAGE", "TOFINO_PIPELINE_STAGES", "MAX_RANGE_KEY_BITS",
    "RANGE_WORST_CASE_ENTRY_FRACTION", "RANGE_WORST_CASE_ROWS_CAP",
    "CODEWORD_KEY_OVERHEAD_BITS",
)

PROGRAM_NAMES = (
    "FEATURE_VALUE_BIT_WIDTH", "RANGE_TABLE_KEY_BYTES", "FLOW_HASH_LEVEL",
    "VOTE_EPILOGUE_STAGES", "ORIENTATION_REGISTER", "REGISTER_BLOCK_ORDER",
    "APP_TASK", "DDOS_TASK", "TASKS", "SHARED_TASK", "PLACEMENT_PRIORITY",
)

# Retired by audit C5 (2026-09-28): the fitted 58-byte crowded-stage margin
# and the helpers that served it. Nothing may bring them back by name.
RETIRED_NAMES = (
    (target, "TERNARY_CROSSBAR_MIXED_KEY_FREE_BYTES_PER_STAGE"),
    # Retired by spec 2026-09-29: the 62-byte mixed-key refusal (valid only while the generator pins code_* layouts).
    (target, "TERNARY_CROSSBAR_MIXED_KEY_BYTES_PER_STAGE"),
)


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_target_holds_every_chip_constant(name):
    assert hasattr(target, name)


@pytest.mark.parametrize("name", PROGRAM_NAMES)
def test_program_holds_every_this_program_constant(name):
    assert hasattr(program, name)


@pytest.mark.parametrize("module,name", RETIRED_NAMES)
def test_the_crowded_stage_margin_is_retired(module, name):
    assert not hasattr(module, name)


def test_the_packer_has_no_key_order_search_left():
    # offsets_for/key_width/crowded/charged were closures inside
    # crossbar_stages_needed; the any-key-order search iterated
    # itertools.permutations. The ordered stage simulation replaces all of it.
    import inspect

    from src.p4model import packing

    source = inspect.getsource(packing)
    for gone in ("offsets_for", "def key_width", "def crowded", "def charged",
                 "permutations", "_SEED_KEY", "MIXED_KEY_FREE"):
        assert gone not in source, gone


def test_the_two_namespaces_do_not_overlap():
    # The whole point of the split: a reader recalibrating for their own program
    # must be able to tell which constants are the chip and which are our
    # choices. A name in both namespaces defeats that.
    assert not (set(TARGET_NAMES) & set(PROGRAM_NAMES))


def test_max_num_flows_stayed_in_build_p4_script():
    # It has no Python reader inside the model (register_sram_bits was its only
    # one, dropped in Task 1) -- it is codegen text emitted into
    # resources/p4_template.p4, so it belongs with the generator.
    assert hasattr(bps, "MAX_NUM_FLOWS")
    assert not hasattr(program, "MAX_NUM_FLOWS")


BPS_REEXPORTS = (
    "TCAM_BLOCKS_PER_STAGE", "TCAM_ROWS_PER_STAGE", "TCAM_COLUMNS_PER_STAGE",
    "TCAM_BLOCK_KEY_LENGTH", "TERNARY_MATCHING_ENTRIES_PER_BLOCK",
    "TERNARY_CROSSBAR_MAX_TABLES_PER_STAGE",
    "TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE", "MAX_CODEWORD_LENGTH",
)


@pytest.mark.parametrize("name", BPS_REEXPORTS)
def test_build_p4_script_reexports_are_the_same_object(name):
    assert getattr(bps, name) is getattr(target, name)


EV_CONSTANT_REEXPORTS = (
    "TOFINO_PIPELINE_STAGES", "CODEWORD_KEY_OVERHEAD_BITS",
    "VOTE_EPILOGUE_STAGES", "FLOW_HASH_LEVEL", "METER_ALUS_PER_STAGE",
    "FEATURE_VALUE_BIT_WIDTH", "RANGE_TABLE_KEY_BYTES", "MAX_RANGE_KEY_BITS",
    "RANGE_WORST_CASE_ENTRY_FRACTION", "RANGE_WORST_CASE_ROWS_CAP",
    "ORIENTATION_REGISTER",
)


@pytest.mark.parametrize("name", EV_CONSTANT_REEXPORTS)
def test_evaluation_constant_reexports_are_the_same_object(name):
    home = target if hasattr(target, name) else program
    assert getattr(ev, name) is getattr(home, name)


def test_exceptions_are_shared_not_duplicated():
    # capacity_ceiling.py:126-128 and train_model.py:32-34 catch these by
    # identity through evaluation's namespace. Two classes would mean a raise
    # from p4model sails straight through their except clause.
    assert ev.CodewordTooLong is errors.CodewordTooLong
    assert ev.CrossbarKeyTooWide is errors.CrossbarKeyTooWide


def test_register_block_order_lost_its_underscore():
    # Spec 4.1's one deliberate rename: it stops being a private detail of one
    # module and becomes a documented program.py constant.
    assert program.REGISTER_BLOCK_ORDER == (None, "fwd", "bwd")


def test_names_and_catalog_reexports_are_the_same_object():
    from src.p4gen import feature_registers as fr
    from src.p4model import catalog, names

    assert bps.normalise_feature_name is names.normalise_feature_name
    assert ev.normalise_feature_name is names.normalise_feature_name
    assert fr.FEATURE_REGISTER_CATALOG is catalog.FEATURE_REGISTER_CATALOG
    assert ev.FEATURE_REGISTER_CATALOG is catalog.FEATURE_REGISTER_CATALOG
    assert fr.register_names_for is catalog.register_names_for
    assert fr.register_width_bits is catalog.register_width_bits


def test_catalog_imports_normalise_at_module_level_not_inside_a_function():
    # feature_registers.py:449 used a function-local import to dodge the
    # build_p4_script <-> feature_registers cycle. names.py imports nothing but
    # `re`, so the cycle cannot exist -- if the dodge is still there, something
    # reintroduced it.
    import inspect

    from src.p4model import catalog

    assert "import" not in inspect.getsource(catalog.register_names_for)


PACKING_REEXPORTS = ("StagePlan", "fits_two_columns", "_stage_shards",
                     "crossbar_stages_needed")


@pytest.mark.parametrize("name", PACKING_REEXPORTS)
def test_packing_reexports_are_the_same_object(name):
    from src.p4model import packing

    assert getattr(ev, name) is getattr(packing, name)


def test_stage_shards_stays_reachable_as_a_private_name():
    # tests/test_evaluation.py reaches ev._stage_shards directly, and
    # scripts/tcam_column_sweep.py:70 imports fits_two_columns from evaluation.
    # Both are in the shim contract (spec 4.3): re-exporting one underscore name
    # beats rewriting a passing test during a verbatim move.
    assert callable(ev._stage_shards)


TABLES_REEXPORTS = (
    "range_matching_resource_usage", "range_deployment_overflow",
    "codeword_fields_to_bytes", "codeword_fields_to_bytes_from_bits",
    "codeword_bits_to_blocks", "ternary_key_field_bits",
    "codeword_bytes_to_blocks", "exact_match_resource_usage",
    "range_key_fields_for", "ternary_key_fields",
    "tree_entries_to_blocks", "entries_across_trees_to_blocks",
)


@pytest.mark.parametrize("name", TABLES_REEXPORTS)
def test_tables_reexports_are_the_same_object(name):
    from src.p4model import tables

    assert getattr(ev, name) is getattr(tables, name)


def test_the_planter_discount_policy_did_not_move_into_the_model():
    # The majority-class rule is this generator's encoding convention, not chip
    # physics -- reviews/p4_tofino_reference.md 4.5 records that it does not
    # even reduce physical TCAM at tested scales. p4model asks "how many entries
    # does this table have"; build_p4_script answers "my encoding sheds K".
    # Keeping the policy out is also what keeps p4model free of build_p4_script,
    # which imports numpy and sklearn.tree at module scope.
    import inspect

    from src.p4model import tables

    assert not hasattr(tables, "most_common_class_and_dropped_codewords")
    assert hasattr(bps, "most_common_class_and_dropped_codewords")

    params = inspect.signature(tables.ternary_matching_resource_usage).parameters
    assert "dropped_per_tree" in params
    assert "use_default_action_discount" not in params


def test_the_evaluation_wrapper_still_accepts_the_boolean_flag():
    # src/main.py:234 and p4_gen_config.py:17 thread the bool through the real
    # pipeline, so the old signature has to keep working.
    import inspect

    params = inspect.signature(ev.ternary_matching_resource_usage).parameters
    assert "use_default_action_discount" in params


REGISTERS_REEXPORTS = ("feature_readiness_level", "register_stage_schedule",
                       "gated_block_interior_stages", "readiness_levels_for")


@pytest.mark.parametrize("name", REGISTERS_REEXPORTS)
def test_registers_reexports_are_the_same_object(name):
    from src.p4model import registers

    assert getattr(ev, name) is getattr(registers, name)


def test_registers_reports_schedule_and_depth_but_not_sram():
    # Task 1: total register bits answered no question. What survives is the
    # schedule (which drives stage placement) and the depth/count read off it.
    from src.p4model import registers

    assert not hasattr(registers, "register_sram_bits")


def test_resource_usage_is_the_same_class():
    from src.p4model import usage

    assert ev.ResourceUsage is usage.ResourceUsage


def test_resource_usage_field_order_is_pinned():
    # The golden fixture serializes these by name, but scripts/ and
    # src/training/ build rows positionally in places -- a reordering would be a
    # silent data corruption, so pin it here rather than discover it later.
    import dataclasses

    from src.p4model import usage

    assert [f.name for f in dataclasses.fields(usage.ResourceUsage)] == [
        "stages", "blocks", "stage_depth", "range_entries", "ternary_entries",
        "codeword_length", "register_depth", "register_count", "range_depth",
        "ternary_depth", "range_tables", "ternary_tables",
    ]


def test_importing_p4model_pulls_in_no_heavy_dependency():
    # The invariant that replaces p4/range_expansion.py's duplication. If this
    # fails, p4/deploy_table_entries.py stops working under bfshell's embedded
    # Python -- which has no sklearn -- and the failure would only surface on a
    # real switch.
    out = _run_isolated("""
        import sys
        import src.p4model
        print(','.join(sorted(m for m in {heavy} if m in sys.modules)))
    """.format(heavy=HEAVY), cwd=REPO_ROOT)
    assert out == "", "src.p4model dragged in: " + out


def test_importing_every_p4model_submodule_pulls_in_no_heavy_dependency():
    # __init__.py is deliberately import-light, so importing the package alone
    # could pass while a submodule is dirty. Import them all.
    out = _run_isolated("""
        import sys
        import src.p4model.catalog, src.p4model.errors, src.p4model.names
        import src.p4model.packing, src.p4model.program, src.p4model.ranges
        import src.p4model.registers, src.p4model.tables, src.p4model.target
        import src.p4model.usage
        print(','.join(sorted(m for m in {heavy} if m in sys.modules)))
    """.format(heavy=HEAVY), cwd=REPO_ROOT)
    assert out == "", "a p4model submodule dragged in: " + out


def test_p4model_imports_and_computes_from_an_unrelated_cwd(tmp_path):
    # build_p4_script reads three .p4 templates from CWD-relative paths at
    # IMPORT time (build_p4_script.py:87,88,95 via PATH = "resources/"), so
    # anything importing it must run from the repo root. p4model must not
    # inherit that -- an installed package has no idea where the repo is.
    out = _run_isolated("""
        from src.p4model.packing import crossbar_stages_needed
        plan = crossbar_stages_needed([(8, 4)] * 3)
        print(plan.occupied)
    """, cwd=str(tmp_path))
    # Three 8-block tables sum to exactly 24 but need two stages: a table chains
    # its blocks down ONE 12-row column, so 8+8 overflows.
    assert out == "2"


def test_the_cost_decomposition_names_say_what_they_take():
    """Design §5.2. Each name is a true function of its stated input, so a
    wrong composition stops being representable rather than merely
    currently-avoided.

    `codeword_bytes_to_blocks` deliberately carries NO version charge: the
    real per-table price depends on the width MULTISET, never on a byte
    total -- 11 bytes costs 2 blocks or 3 depending on the field split -- so a
    signature promising otherwise would assert a dependency the quantity does
    not have.
    """
    from src.p4model import tables

    assert tables.codeword_fields_to_bytes({'a': [0] * 5, 'b': [0] * 5}) == 2
    assert tables.codeword_bytes_to_blocks(11) == 2
    assert tables.codeword_bits_to_blocks(40) == 1
    assert tables.codeword_bits_to_blocks(41) == 2      # 41 + 4 > 44
    for retired in ('ternary_table_key_bytes', 'crossbar_block_width',
                    'band_factor'):
        assert not hasattr(tables, retired), retired


def test_a_deep_table_pays_a_later_keys_price_once_per_row_word():
    """A table deeper than one 512-row word stores more rows through the SAME
    key: the key's crossbar lanes are laid out once, and every word repeats
    that layout. So a key that pays a leftover price behind a different key
    pays it once per word -- the ordered stage simulation's `price x rows`
    (audit C5; reviews/model_audit_scratch/proto_model.py charges the same).

    (This replaces finding 1.4's test, which pinned `offsets_for`'s running
    sum of key widths. offsets_for, key_width and the crowded-stage margin
    they served are deleted: the ordered simulation reads the ORDER keys were
    placed in directly, and no offset exists any more.)

    The spacer key (360,) is placed first (placement priority 2) and pays its
    standalone 9 blocks; the probe key (54, 56) -- independent_low_sd5's real
    app key, 3 blocks alone -- costs 4 per word in the lanes the spacer left
    (results/tcam_mixed_key_cap_sweep.csv measured exactly 4 behind such a
    spacer). One word: 9 + 4 = 13. Two words (declared 6 blocks): 9 + 8 = 17.
    """
    from src.p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 45)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[spacer, probe],
                  key_field_bits=[(360,), (54, 56)], placement_priority=[2, 1])

    shallow = crossbar_stages_needed([(9, 45), (3, 14)], **kwargs)
    deep = crossbar_stages_needed([(9, 45), (6, 14)], **kwargs)

    assert (shallow.occupied, deep.occupied) == (1, 1), (shallow, deep)
    assert (shallow.blocks, deep.blocks) == (13, 17), (shallow, deep)


# --------------------------------------------------------------------------
# The sharing charge (src/p4model/packing.py's ordered stage simulation, audit
# C5, 2026-09-28): tables are placed in p4c's order (placement priority, then
# the table listed LAST first), and each stage's distinct keys are priced in
# that order -- the first key its declared blocks, every later key its lane
# LEFTOVER price in the crossbar lanes the keys before it left
# (src/p4model/lanes.py). Two different keys share a stage whenever the lanes
# price them and the real 64-byte crossbar limit holds; the 62-byte refusal
# that used to sit on top was retired by spec 2026-09-29 once the generator
# pinned every key's PHV layout.
#
# It replaced the fitted CROWDED-STAGE margin (+1 to every non-first key when
# two different keys filled more than 58 of a stage's 64 crossbar bytes, and
# an any-key-order fit search), which scored 39/43 stage_depth and 31/38
# blocks on the pragma'd compiles against the simulation's 42/43 and 38/38
# (reviews/model_audit_2026-09-27.md §7.6). The per-key saturation margin
# before that was retired 2026-09-25.
#
# The first four tests stood in tests/test_tcam_block_ledger.py (then named
# test_version_block.py) until the 2026-09-20 TCAM block model rewrite, which
# re-scoped that file to PER-TABLE facts only. Two tests that pinned
# `codeword_to_blocks(bits, start_group)`'s offset sensitivity are NOT
# restored: reading p4c's own assembly showed that premise false (rewrite
# design Sec 2).
# --------------------------------------------------------------------------
def test_the_packer_charges_sd5s_stage_nine_blocks_an_accepted_under_of_three():
    # Modelled on independent_low_sd5's stage 6, where p4c charged its app and
    # ddos trees 3 TCAM blocks each (resources.json): 12 in all. Under the lane
    # price (2026-10-04) the ddos key (27, 52) costs 2 per tree, so the model
    # charges 3 + 3 * 2 = 9 -- an UNDER of 3, the accepted greedy-miss class
    # (golden fixture known_findings 'lane_price_2026_10'). The ladder's 3 was
    # right by luck.
    from src.p4model.lanes import table_blocks
    from src.p4model.packing import crossbar_stages_needed
    from src.p4model.tables import codeword_to_blocks

    app = frozenset({(("code", "app_flm"), 7), (("code", "app_plm"), 7)})
    ddos = frozenset({(("code", "ddos_bplm"), 4), (("code", "ddos_plm"), 7)})
    assert codeword_to_blocks((27, 52)) == 3
    assert table_blocks((27, 52)) == 2
    assert table_blocks((54, 56)) == 3
    plan = crossbar_stages_needed(
        [(3, 14), (2, 11), (2, 11), (2, 11)],
        key_fields=[app, ddos, ddos, ddos],
        key_field_bits=[(54, 56), (27, 52), (27, 52), (27, 52)],
        placement_priority=[1, 2, 2, 2])
    assert plan.blocks == 9


def test_a_stage_of_one_shared_key_is_charged_exactly_its_declared_blocks():
    # Every 'joint' design keys every tree on the identical field set, so a
    # joint stage has exactly ONE distinct key, which is its first key and is
    # charged its declared blocks (plan invariant 1 -- what keeps joint
    # designs identical to threshold alignment's total_blocks). The key below
    # is 60 bytes, specifically so this test cannot pass merely because the
    # key happened to be small: one key may use the whole crossbar.
    from src.p4model.lanes import table_blocks
    from src.p4model.packing import crossbar_stages_needed
    from src.p4model.tables import codeword_to_blocks

    key = frozenset({(("code", "f"), 60)})
    assert codeword_to_blocks((480,)) == 11
    assert table_blocks((480,)) == 11
    plan = crossbar_stages_needed(
        [(11, 60)] * 3, key_fields=[key] * 3,
        key_field_bits=[(480,)] * 3)
    assert plan.blocks == 33


def test_omitting_key_field_bits_prices_every_table_at_its_declared_width():
    # Without field widths there is no lane simulation: every table costs what
    # it declares. That is what keeps the range pool -- the one caller that
    # passes no bits -- on exactly its pre-existing pricing.
    from src.p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed([(3, 14), (2, 11), (2, 11), (2, 11)])
    assert plan.blocks == 9


def test_placement_priority_needs_key_field_bits_and_one_entry_per_table():
    from src.p4model.packing import crossbar_stages_needed

    with pytest.raises(ValueError, match="key_field_bits"):
        crossbar_stages_needed([(1, 2)], placement_priority=[1])
    with pytest.raises(ValueError, match="placement_priority"):
        crossbar_stages_needed([(1, 2), (1, 2)], key_field_bits=[(8,), (8,)],
                               placement_priority=[1])


def test_the_generator_emits_the_placement_priority_the_packer_replays():
    # The simulation's order is only right if it is the order the generator
    # pins: every classification tree table in the committed golden program
    # (p4/p4_code_RF_models_shared.p4, regenerated and diffed by
    # test_build_p4_script_tna.py) carries @placement_priority equal to
    # program.PLACEMENT_PRIORITY of its task, and the trees are listed app
    # first, then ddos, each by index -- evaluation._pool_inputs' order.
    import re

    from src.p4model.program import PLACEMENT_PRIORITY

    path = os.path.join(REPO_ROOT, "p4", "p4_code_RF_models_shared.p4")
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    found = re.findall(
        r"@placement_priority\((\d+)\)\s*\n\s*table get_classification_tree_"
        r"(app|ddos)_(\d+)\s*\{", text)
    assert found
    for priority, task, _index in found:
        assert int(priority) == PLACEMENT_PRIORITY[task], (task, priority)
    listed = [(task, int(index)) for _p, task, index in found]
    assert listed == sorted(listed, key=lambda t: (t[0] != "app", t[1]))
    assert len(re.findall(r"table get_classification_tree_", text)) == len(found)


def test_two_different_ragged_keys_do_not_share_a_stage():
    # independent_low_sd9, the row a sharing price exists for. Its app trees
    # key 179 + 204 bits = 49 crossbar bytes at 9 blocks; its ddos trees key
    # 37 + 49 bits = 12 bytes at 3 blocks. 2 app + 2 ddos is exactly 24 blocks
    # and packs both columns as 9+3 | 9+3, so every BLOCK/TABLE-COUNT limit
    # would let them share a stage -- and p4c refuses. The simulation places
    # the five ddos trees first (priority 2), and an app tree behind them
    # costs 10, not 9, in the lanes they left (15 + 10 > 24): the app trees go
    # 2 | 2 | 1 in three more stages. p4c's committed placement exactly
    # (resources.json: 5 ddos | 2 app | 2 app | 1 app, 5 x 9 + 5 x 3 = 60
    # blocks). The retired crowded margin co-located one app tree with the
    # ddos trees and charged 61. Measured on the same key by
    # scripts/tcam_stretch_sweep.py: 9 blocks alone, 10 beside a 12-byte key.
    from src.p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] * 5 + [(3, 12)] * 5,
        readiness_levels=[0] * 10,
        key_fields=[app] * 5 + [ddos] * 5,
        key_field_bits=[(179, 204)] * 5 + [(37, 49)] * 5,
        placement_priority=[1] * 5 + [2] * 5)
    assert (plan.occupied, plan.blocks) == (4, 60)
    assert [load.blocks for load in plan.stage_loads] == [
        (3, 3, 3, 3, 3), (9, 9), (9, 9), (9,)]


def test_the_ragged_key_pays_one_block_behind_four_narrow_tables():
    # scripts/tcam_stretch_sweep.py's ragged_ax1_bx4: a 49-byte app key (9
    # blocks) beside four 12-byte ddos keys (3 blocks each). p4c places all
    # five in ONE stage at 22 TCAM blocks (results/tcam_stretch_sweep.csv):
    # the ddos key first (4 x 3 = 12, one column) and the app key second at
    # 10, the other column. The simulation, ddos first by priority, agrees.
    from src.p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4,
        readiness_levels=[0] * 5,
        key_fields=[app] + [ddos] * 4,
        key_field_bits=[(179, 204)] + [(37, 49)] * 4,
        placement_priority=[1] + [2] * 4)
    assert (plan.occupied, plan.blocks) == (1, 22)


_SPACER = frozenset({(("code", "spacer"), 41)})
_PROBE = frozenset({(("code", "probe_a"), 11), (("code", "probe_b"), 11)})


def test_the_lane_price_alone_keeps_two_probes_off_a_crowded_spacer():
    # results/tcam_discount_scan.csv dsp41: a (84, 84) probe key (22 crossbar
    # bytes, 4 blocks alone under the pinned fill-low layout (the unpinned
    # dsp41 compile charged 5)) behind a 41-byte spacer costs p4c 7 blocks, not 5,
    # and p4c kept both in ONE stage. The lane simulation charges exactly that
    # 7 (spacer placed first by priority). With two probes it is 8 | 7+7,
    # which a 12-row column cannot hold, so the second probe moves on -- the
    # price keeps them apart, no byte threshold needed (the 62-byte refusal
    # was retired by spec 2026-09-29).
    from src.p4model.packing import crossbar_stages_needed

    one = crossbar_stages_needed(
        [(8, 41), (4, 22)], readiness_levels=[0] * 2,
        key_fields=[_SPACER, _PROBE], key_field_bits=[(328,), (84, 84)],
        placement_priority=[2, 1])
    assert (one.occupied, one.blocks, one.table_stages) == (1, 15, (0, 0))

    two = crossbar_stages_needed(
        [(8, 41), (4, 22), (4, 22)], readiness_levels=[0] * 3,
        key_fields=[_SPACER, _PROBE, _PROBE],
        key_field_bits=[(328,), (84, 84), (84, 84)],
        placement_priority=[2, 1, 1])
    assert (two.occupied, two.blocks, two.table_stages) == (2, 19, (0, 1, 0))


def test_a_41_plus_22_byte_mixed_stage_now_shares_one_stage():
    # The dsp41 shape (not M150_k7_s11's real keys): 41 + 22 = 63 bytes of
    # two different keys, refused by the old 62-byte
    # net; with the generator's layout pins p4c places both in one stage
    # (spec 2026-09-29 Sec 5.2), and so does the packer. Probe listed last,
    # so placed first at its own 4; the spacer behind it pays its lane
    # leftover 8.
    from src.p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed(
        [(8, 41), (4, 22)], readiness_levels=[0] * 2,
        key_fields=[_SPACER, _PROBE], key_field_bits=[(328,), (84, 84)])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 12, (0, 0))


def test_a_later_key_pays_its_lane_leftover_price():
    # results/tcam_mixed_key_cap_sweep.csv: independent_low_sd5's real app key
    # (54, 56) -- 14 bytes, 3 blocks alone -- costs 4 behind a solid spacer at
    # 59-62 combined bytes. Placed second (the spacer's priority is higher),
    # the probe is priced in the lanes a 45-byte spacer left: 4, as measured
    # -- 9 + 4 = 13 in one stage. Placed FIRST, the same probe pays its own
    # 3 and the spacer, behind it, still finds its 9: 12. The order is the
    # generator's to pin, not the packer's to guess.
    from src.p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 45)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[spacer, probe],
                  key_field_bits=[(360,), (54, 56)])
    spacer_first = crossbar_stages_needed(
        [(9, 45), (3, 14)], placement_priority=[2, 1], **kwargs)
    probe_first = crossbar_stages_needed(
        [(9, 45), (3, 14)], placement_priority=[1, 2], **kwargs)
    assert (spacer_first.occupied, spacer_first.blocks) == (1, 13)
    assert spacer_first.stage_loads[0].blocks == (9, 4)
    assert (probe_first.occupied, probe_first.blocks) == (1, 12)


def test_a_later_key_with_lanes_to_spare_pays_nothing():
    # Across six probe shapes no probe paid anything at 58 combined bytes or
    # below (tcam_mixed_key_cap_sweep/onset). Behind a 44-byte spacer the
    # (54, 56) probe still finds lanes for its own 3 blocks: 12. (The retired
    # margin read this as its "free edge"; the simulation needs no threshold.)
    from src.p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 44), (3, 14)], readiness_levels=[0] * 2,
        key_fields=[spacer, probe], key_field_bits=[(352,), (54, 56)],
        placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks) == (1, 12)
    assert plan.stage_loads[0].blocks == (9, 3)


def test_the_mixed_key_byte_cap_never_touches_a_single_key_stage():
    # The cap is about a SECOND key finding the groups the first left; one key
    # alone may use the whole 64-byte budget (measured: one table at exactly
    # 64 bytes, reviews/p4_tofino_reference.md Sec 4.3). Two trees of one
    # 64-byte key share their field, so the stage still holds 64 bytes.
    from src.p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "wide"), 64)})
    plan = crossbar_stages_needed(
        [(12, 64), (12, 64)], readiness_levels=[0] * 2,
        key_fields=[key, key], key_field_bits=[(512,), (512,)])
    assert plan.occupied == 1


def test_a_table_is_sharded_as_full_columns_plus_a_remainder():
    """Finding 1.5a. The equal split rounded each piece up, so a 13-block table
    was charged 14. A column holds TCAM_ROWS_PER_STAGE blocks; fill columns and
    leave the remainder, and the shards sum to the table."""
    from src.p4model.packing import _stage_shards

    assert _stage_shards(13, 30) == [(12, 30), (1, 30)]
    assert _stage_shards(23, 30) == [(12, 30), (11, 30)]
    assert _stage_shards(25, 30) == [(12, 30), (12, 30), (1, 30)]


def test_shards_always_sum_to_the_table_and_fit_a_column():
    """The two invariants the split must never break, over the whole reachable
    range: nothing is lost, and no shard is wider than one column."""
    from src.p4model.packing import _stage_shards
    from src.p4model.target import TCAM_ROWS_PER_STAGE

    for blocks in range(0, 60):
        shards = _stage_shards(blocks, 30)

        assert sum(b for b, _w in shards) == blocks, blocks
        assert all(b <= TCAM_ROWS_PER_STAGE for b, _w in shards), blocks
        assert shards, blocks


def test_a_table_inside_one_column_is_not_sharded():
    """The common case, and the one the whole 19-row archive lives in: no
    archived table exceeds 12 blocks, so this path is the only one the
    calibration ever exercises."""
    from src.p4model.packing import _stage_shards

    assert _stage_shards(12, 30) == [(12, 30)]
    assert _stage_shards(1, 30) == [(1, 30)]
    assert _stage_shards(0, 30) == [(0, 30)]


def _parity_feasible_under_some_order(heights, rows, columns):
    """Is there ANY order and column assignment placing all of `heights` under
    p4c's row-parity rule -- a run of EVEN height may only start on an EVEN
    row, a run of odd height may start anywhere
    (Memories::find_ternary_stretch, rewrite design Sec 6.2/13.2)?

    Exhaustive with memoisation on (tables left, column loads); a column's
    load IS the next free row in it, since runs are placed bottom-up with no
    deliberate holes."""
    seen = set()

    def search(remaining, loads):
        if not remaining:
            return True
        if (remaining, loads) in seen:
            return False
        seen.add((remaining, loads))
        for i, height in enumerate(remaining):
            if i and remaining[i - 1] == height:
                continue                       # same multiset branch already tried
            rest = remaining[:i] + remaining[i + 1:]
            for column in range(columns):
                if loads[column] + height > rows:
                    continue
                if height % 2 == 0 and loads[column] % 2:
                    continue                   # even run may not start on an odd row
                bumped = list(loads)
                bumped[column] += height
                if search(rest, tuple(bumped)):
                    return True
        return False

    return search(tuple(sorted(heights)), (0,) * columns)


def test_row_parity_never_rejects_a_stage_that_fits_by_size():
    """Step 3a. p4c's row-parity rule costs this packer nothing, so
    fits_two_columns is right to model column LOADS and ignore rows.

    Proof by exhaustion over every multiset of block heights 1..12 that is
    size-feasible in two 12-row columns: parity-feasible-under-some-order is
    EQUIVALENT to fits_two_columns on all 8 618 of them, with no disagreement
    in either direction. The reason it must be so: place every EVEN height
    first, and the prefix sums of even numbers stay even, so every even run
    lands on an even row; odd runs then go anywhere. A multiset that fits by
    size therefore always fits under parity for SOME order.

    A superset of a size-infeasible multiset is size-infeasible and
    parity-infeasible alike (both tests are monotone in the multiset), so
    pruning there loses no case.

    Confirmed against the archive, not just by construction: over the 20
    compiles the rewrite design cites, 317 committed runs include 103 that
    start on an ODD row and ZERO even-height odd starts -- the rule itself
    holds without exception -- and exactly 2 of 183 committed columns contain
    a parity gap at all (independent_high_sd6 and _sd7, stage 9: a 3-high ddos
    run at rows 0-2, then a 2-high app run forced off row 3 to row 4). Both
    gaps sit in columns using 6 of 12 rows, so neither spilled anything.
    """
    from src.p4model.packing import fits_two_columns
    from src.p4model.target import TCAM_COLUMNS_PER_STAGE, TCAM_ROWS_PER_STAGE

    rows, columns = TCAM_ROWS_PER_STAGE, TCAM_COLUMNS_PER_STAGE
    checked = 0

    def walk(prefix, largest):
        nonlocal checked
        if prefix:
            checked += 1
            by_size = fits_two_columns(prefix)
            assert by_size == _parity_feasible_under_some_order(
                prefix, rows, columns), prefix
            if not by_size:
                return                        # monotone: every superset fails both
        for height in range(min(largest, rows), 0, -1):
            walk(prefix + [height], height)   # non-increasing, so each multiset once

    walk([], rows)
    assert checked == 8618, checked


def test_a_saturated_key_in_an_uncrowded_shared_stage_pays_nothing():
    # independent_high_sd7's real stage 9: a SATURATED 10-byte app key
    # (crossbar_capacity(2) == 10) beside a 12-byte ddos key, 22 bytes in all,
    # and p4c charged nothing (tcam_offset_harvest.csv). The retired per-key
    # saturation margin charged it +1; the lane simulation finds the later
    # key's lanes free and charges nothing, whichever key goes first.
    from src.p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_%d" % i), 1) for i in range(10)})
    ddos = frozenset({(("code", "ddos_%d" % i), 1) for i in range(12)})
    plan = crossbar_stages_needed(
        [(2, 10), (3, 12)], readiness_levels=[0] * 2,
        key_fields=[app, ddos],
        key_field_bits=[(1, 1, 1, 2, 3, 3, 3, 3, 4, 8),
                        (1, 1, 1, 2, 2, 2, 2, 2, 4, 4, 4, 5)])
    assert (plan.occupied, plan.blocks) == (1, 5)


# --- seed_stages: a pool packed into stages another pool already partly fills
# (audit C1). Seeds count against the table cap, the byte limit and the column
# packing, and are never charged or reported by the seeded pool.

def _seed(index, blocks, byte_widths, tables=None):
    from src.p4model.packing import StageLoad

    return StageLoad(index=index, blocks=tuple(blocks),
                     fields=frozenset((("val", index, i), w)
                                      for i, w in enumerate(byte_widths)),
                     tables=len(blocks) if tables is None else tables)


def test_a_seed_counts_against_the_eight_table_cap():
    from src.p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "k"), 2)})
    plan = crossbar_stages_needed(
        [(1, 2), (1, 2)], readiness_levels=[5, 5], key_fields=[key, key],
        seed_stages=[_seed(5, [1] * 7, [2] * 7)])
    # 7 seeded + 1 = 8 at stage 5; the second table spills to 6.
    assert plan.table_stages in ((5, 6), (6, 5))
    assert plan.indices == {5, 6}


def test_a_seed_counts_against_the_64_byte_limit():
    from src.p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "k"), 8)})
    plan = crossbar_stages_needed(
        [(2, 8)], readiness_levels=[5], key_fields=[key],
        seed_stages=[_seed(5, [1], [60])])
    assert plan.table_stages == (6,)


def test_a_seed_counts_against_the_column_packing():
    from src.p4model.packing import crossbar_stages_needed

    seed = _seed(5, [12, 8], [2, 2])
    # 12 | 8 + 8 = 16 overflows the second column; 12 | 8 + 4 fits.
    assert crossbar_stages_needed(
        [(8, 5)], readiness_levels=[5],
        seed_stages=[seed]).table_stages == (6,)
    assert crossbar_stages_needed(
        [(4, 5)], readiness_levels=[5],
        seed_stages=[seed]).table_stages == (5,)


def test_a_seed_is_never_charged_or_reported_by_the_seeded_pool():
    from src.p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed(
        [(3, 5)], readiness_levels=[5],
        seed_stages=[_seed(5, [2, 2], [2, 2]), _seed(9, [4], [2])])
    assert (plan.occupied, plan.depth, plan.indices, plan.blocks) == (
        1, 6, frozenset({5}), 3)
    assert [(load.index, load.blocks, load.tables)
            for load in plan.stage_loads] == [(5, (3,), 1)]


def test_seeds_leave_an_unseeded_placement_unchanged():
    # Seeds at stages the pool never reaches change nothing at all.
    from src.p4model.packing import crossbar_stages_needed

    specs = [(9, 45), (3, 14), (6, 10), (6, 10)]
    fields = [frozenset({(("code", "s"), 45)}),
              frozenset({(("code", "a"), 7), (("code", "b"), 7)}),
              frozenset({(("code", "c"), 10)}), frozenset({(("code", "c"), 10)})]
    bits = [(360,), (54, 56), (80,), (80,)]
    kwargs = dict(readiness_levels=[4] * 4, key_fields=fields,
                  key_field_bits=bits)
    assert crossbar_stages_needed(specs, **kwargs) == crossbar_stages_needed(
        specs, seed_stages=[_seed(0, [12, 12], [2, 2]), _seed(3, [1], [2])],
        **kwargs)


def test_seed_range_keys_are_placed_ahead_of_every_tree_key():
    # A seed's range fields sit on the same ternary crossbar, and the range
    # tables were placed before any tree reached the stage, so the lane
    # simulation places them FIRST: every tree key there is a later key.
    # Nothing was ever measured with a range key beside a tree key (before
    # audit C1 the two pools never shared a stage), so they are not exempted.
    # Without a seed the spacer (listed last, so placed first) and the probe
    # share stage 0 at 9 + 3 = 12. With two 2-byte range keys already there,
    # 4 + 44 + 14 = 62 bytes are within the 64-byte crossbar, but the lanes are not: the
    # spacer still fits behind the range keys, the probe then finds no legal
    # fit and moves to stage 1. The seed itself is never charged.
    from src.p4model.packing import crossbar_stages_needed

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    kwargs = dict(readiness_levels=[0] * 2, key_fields=[probe, spacer],
                  key_field_bits=[(54, 56), (352,)])
    alone = crossbar_stages_needed([(3, 14), (9, 44)], **kwargs)
    seeded = crossbar_stages_needed([(3, 14), (9, 44)],
                                    seed_stages=[_seed(0, [1, 1], [2, 2])],
                                    **kwargs)
    assert (alone.occupied, alone.blocks, alone.table_stages) == (1, 12, (0, 0))
    assert (seeded.occupied, seeded.blocks, seeded.table_stages) == (2, 12, (1, 0))


def test_a_tree_key_behind_a_seed_pays_its_leftover_price_never_less():
    # The 60-byte (480,) key costs 11 blocks alone. Behind one 2-byte range key
    # its lanes are short and it costs 12; the first tree key behind a seed is
    # charged max(codeword_to_blocks, lane leftover), so never below 11.
    from src.p4model.packing import crossbar_stages_needed

    key = frozenset({(("code", "wide"), 60)})
    plan = crossbar_stages_needed(
        [(11, 60)], readiness_levels=[0], key_fields=[key],
        key_field_bits=[(480,)], seed_stages=[_seed(0, [1], [2])])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 12, (0,))
    small = frozenset({(("code", "small"), 2)})
    plan = crossbar_stages_needed(
        [(1, 2)], readiness_levels=[0], key_fields=[small],
        key_field_bits=[(16,)], seed_stages=[_seed(0, [1], [2])])
    assert plan.blocks == 1


def test_a_seed_and_tree_keys_share_a_stage_up_to_64_bytes_if_the_lanes_fit():
    # 44 + 14 + 6 = 64 bytes, three range keys ahead of two tree keys. The old
    # 62-byte net refused this outright; now only the lanes and the real
    # 64-byte limit decide. The lanes still cannot fit the spacer behind the
    # seed and the probe, so it moves on -- same placement, new reason.
    from src.p4model.packing import crossbar_stages_needed, stage_load_fits

    spacer = frozenset({(("code", "spacer"), 44)})
    probe = frozenset({(("code", "a"), 7), (("code", "b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 44), (3, 14)], readiness_levels=[0] * 2,
        key_fields=[spacer, probe], key_field_bits=[(352,), (54, 56)],
        seed_stages=[_seed(0, [1, 1, 1], [2, 2, 2])])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (2, 12, (1, 0))
    # Judged after the fact: two pools totalling exactly 64 bytes now fit;
    # 66 bytes are past the real crossbar limit and still do not.
    assert stage_load_fits([_seed(0, [1], [6]), _seed(0, [9, 3], [58])])
    assert not stage_load_fits([_seed(0, [1], [8]), _seed(0, [9, 3], [58])])


def test_seed_stages_need_readiness_levels_and_one_load_per_stage():
    from src.p4model.packing import crossbar_stages_needed

    with pytest.raises(ValueError, match="readiness_levels"):
        crossbar_stages_needed([(1, 2)], seed_stages=[_seed(0, [1], [2])])
    with pytest.raises(ValueError, match="two seed_stages entries"):
        crossbar_stages_needed([(1, 2)], readiness_levels=[0],
                               seed_stages=[_seed(0, [1], [2]),
                                            _seed(0, [1], [2])])


def test_table_stages_is_each_tables_latest_shard_in_both_branches():
    from src.p4model.packing import crossbar_stages_needed

    # Pure packer: a 30-block table is 12 | 12 | 6 shards; the two full ones
    # fill stage 0, the 6 and the 1-block table share stage 1.
    plan = crossbar_stages_needed([(30, 5), (1, 2)])
    assert plan.table_stages == (1, 1)
    assert [load.blocks for load in plan.stage_loads] == [(12, 12), (6, 1)]
    # Dependency-aware: the same table from level 3 ends at stage 4.
    plan = crossbar_stages_needed([(30, 5), (1, 2)], readiness_levels=[3, 7])
    assert plan.table_stages == (4, 7)
    assert sum(sum(load.blocks) for load in plan.stage_loads) == plan.blocks


def test_stage_load_fits_checks_the_combined_limits():
    from src.p4model.packing import stage_load_fits

    assert stage_load_fits([_seed(0, [12], [2]), _seed(0, [6, 6], [30])])
    assert not stage_load_fits([_seed(0, [12], [2]), _seed(0, [8, 8], [2])])
    assert not stage_load_fits([_seed(0, [1] * 5, [2]), _seed(0, [1] * 4, [2])])
    assert not stage_load_fits([_seed(0, [1], [40]), _seed(0, [1], [30])])


def test_a_classification_table_wider_than_a_stage_is_split_by_whole_rows():
    # A 30-block table of a 5-block key (6 row words) cannot sit in one
    # stage's 24 blocks. The ordered simulation (two keys here, so it runs)
    # splits it into chunks of whole words that each can -- 4 words (20
    # blocks, 12 | 8) then 2 (10) -- every chunk keeping the full key; the
    # table is done at the later stage. The 1-block second key, placed after
    # the first chunk, finds lanes to spare.
    from src.p4model.packing import crossbar_stages_needed

    wide = frozenset({(("code", "w"), 25)})
    small = frozenset({(("code", "s"), 2)})
    plan = crossbar_stages_needed([(30, 25), (1, 2)], key_fields=[wide, small],
                                  key_field_bits=[(200,), (16,)],
                                  placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (2, 31, (1, 0))
    assert [load.blocks for load in plan.stage_loads] == [(12, 8, 1), (10,)]


def test_a_classification_table_wider_than_a_column_stays_in_one_stage():
    # 15 blocks spans both columns of ONE stage (12 | 3, measured at 14/16/24
    # blocks, _stage_shards) and counts as ONE table against the 8-table cap.
    from src.p4model.packing import crossbar_stages_needed

    wide = frozenset({(("code", "w"), 25)})
    small = frozenset({(("code", "s"), 2)})
    plan = crossbar_stages_needed([(15, 25), (1, 2)], key_fields=[wide, small],
                                  key_field_bits=[(200,), (16,)],
                                  readiness_levels=[2, 2],
                                  placement_priority=[2, 1])
    assert (plan.occupied, plan.blocks, plan.table_stages) == (1, 16, (2, 2))
    assert plan.stage_loads[0].tables == 2


# --- Plan invariant 1, structurally: a single-key classification pool (every
# 'joint' design) is placed and charged exactly as before audit C5. The
# ordered stage simulation's placement order once leaked into single-key
# pools and moved stage_depth on ~1% of realistic tree-size mixes (task-5
# review fuzz, 224/20 000) while the archive happened to avoid them. The
# reference below is an INDEPENDENT restatement of the pre-C5 placement
# (packing.py at 5e97e6b, where a single key never paid the old margin):
# column-sized shards, placed eagerly in (level, largest-load-first) order at
# the earliest legal stage (first-fit-decreasing without levels), a stage
# taking a shard while it keeps <= 8 shards, <= 64 bytes of distinct fields
# and a 12x2 column packing, every shard charged its declared blocks
# (declared = lanes.table_blocks x words since 2026-10-04).

def _pre_c5_single_key_reference(specs, levels, unavailable):
    from src.p4model.packing import fits_two_columns

    shards = []
    for idx, (blocks, width) in enumerate(specs):
        remaining = blocks
        while remaining > 12:
            shards.append((12, width, idx))
            remaining -= 12
        shards.append((remaining, width, idx))

    def load(shard):
        return max(shard[0] / 24, shard[1] / 64, 1 / 8)

    stages = {}                     # index -> shard blocks; one key, so width is fixed

    def fits(here, blocks, width):
        return (width <= 64 and len(here) + 1 <= 8
                and fits_two_columns(here + [blocks]))

    table_stage = {}
    if levels is None:
        for blocks, width, idx in sorted(shards, key=load, reverse=True):
            index = 0
            while index in stages and not fits(stages[index], blocks, width):
                index += 1
            stages.setdefault(index, []).append(blocks)
            table_stage[idx] = max(index, table_stage.get(idx, index))
    else:
        for blocks, width, idx in sorted(shards, key=lambda s: (levels[s[2]], -load(s))):
            index = levels[idx]
            while index in unavailable or (
                    index in stages and not fits(stages[index], blocks, width)):
                index += 1
            stages.setdefault(index, []).append(blocks)
            table_stage[idx] = max(index, table_stage.get(idx, index))
    depth = max(stages) + 1 if stages else 0
    return (len(stages), depth, sum(map(sum, stages.values())),
            tuple(table_stage[i] for i in range(len(specs))))


def _single_key_config(rng, dist):
    from src.p4model.errors import CrossbarKeyTooWide
    from src.p4model.lanes import table_blocks

    while True:
        if dist == "small":
            bits = tuple(rng.randint(1, 12) for _ in range(rng.randint(1, 4)))
        else:
            bits = tuple(rng.randint(1, 60) for _ in range(rng.randint(1, 16)))
        width = sum(-(-b // 8) for b in bits)
        if width > 64:
            continue
        try:
            per_row = table_blocks(bits)
        except CrossbarKeyTooWide:
            continue
        break
    trees = rng.randint(1, 40 if dist == "extreme" else 30)

    def rows():
        if dist == "one":
            return 1
        if dist == "mix":           # ~85% one-word trees, the realistic shape
            return 1 if rng.random() < 0.85 else rng.choice([2, 2, 3, 4])
        if dist == "small":
            return rng.choice([1, 2])
        return rng.choice([1, 2, 3, 4, 5, 6])

    specs = [(per_row * rows(), width) for _ in range(trees)]
    key = frozenset({(("code", i), -(-b // 8)) for i, b in enumerate(bits)})
    level = rng.randint(3, 8)
    unavailable = frozenset(s for s in range(level, level + 4) if rng.random() < 0.15)
    priority = [rng.choice([1, 2]) for _ in range(trees)]
    return specs, key, bits, [level] * trees, unavailable, priority


@pytest.mark.parametrize("dist", ["one", "mix", "small", "extreme"])
def test_a_single_key_pool_is_placed_exactly_as_before_the_ordered_simulation(dist):
    import random

    from src.p4model.packing import StageLoad, crossbar_stages_needed

    rng = random.Random("single-key-" + dist)
    for case in range(1500):
        specs, key, bits, levels, unavailable, priority = _single_key_config(rng, dist)
        pure = case % 4 == 0
        kwargs = dict(key_fields=[key] * len(specs), key_field_bits=[bits] * len(specs),
                      placement_priority=priority)
        if pure:
            plan = crossbar_stages_needed(specs, **kwargs)
            expected = _pre_c5_single_key_reference(specs, None, frozenset())
        else:
            # A range seed BELOW the trees' level (every 'joint' design: trees
            # wait for every range table) must not change anything either.
            seed = StageLoad(index=levels[0] - 1, blocks=(1, 1),
                             fields=frozenset({(("val", 0), 2), (("val", 1), 2)}),
                             tables=2)
            plan = crossbar_stages_needed(specs, readiness_levels=levels,
                                          unavailable_stages=unavailable,
                                          seed_stages=[seed], **kwargs)
            expected = _pre_c5_single_key_reference(specs, levels, unavailable)
        assert (plan.occupied, plan.depth, plan.blocks, plan.table_stages) == expected, (
            dist, case, specs, levels, sorted(unavailable))
        assert plan.blocks == sum(blocks for blocks, _ in specs)


def test_a_second_key_or_a_reachable_seed_switches_the_simulation_on():
    # The single-key shortcut must not swallow a real pricing question: a
    # 12-byte key ahead of the ragged 49-byte one still makes it pay 10, and a
    # range seed IN the trees' stage still counts as a key placed first.
    from src.p4model.packing import StageLoad, crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    two = crossbar_stages_needed(
        [(9, 49)] + [(3, 12)] * 4, readiness_levels=[0] * 5,
        key_fields=[app] + [ddos] * 4,
        key_field_bits=[(179, 204)] + [(37, 49)] * 4,
        placement_priority=[1] + [2] * 4)
    assert two.blocks == 22
    wide = frozenset({(("code", "wide"), 60)})
    seed = StageLoad(index=0, blocks=(1,), fields=frozenset({(("val", 0), 2)}),
                     tables=1)
    seeded = crossbar_stages_needed(
        [(11, 60)], readiness_levels=[0], key_fields=[wide],
        key_field_bits=[(480,)], seed_stages=[seed])
    assert seeded.blocks == 12
