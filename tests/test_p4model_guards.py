"""Structural guarantees for src/p4model: each name lives in exactly one place,
and the old import paths resolve to that same object.

Identity (`is`) assertions, not equality: a re-export that rebinds rather than
aliases would still compare equal on an int, then drift silently the first time
one side is edited."""
import pytest

from src.p4gen import build_p4_script as bps
from src.p4gen import evaluation as ev
from src.p4model import errors, program, target

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
)


@pytest.mark.parametrize("name", TARGET_NAMES)
def test_target_holds_every_chip_constant(name):
    assert hasattr(target, name)


@pytest.mark.parametrize("name", PROGRAM_NAMES)
def test_program_holds_every_this_program_constant(name):
    assert hasattr(program, name)


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
    "ternary_table_key_bytes", "band_factor", "ternary_key_is_ragged",
    "crossbar_block_width", "exact_match_resource_usage",
    "range_key_fields_for", "ternary_key_fields",
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
