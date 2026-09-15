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
    "codeword_fields_to_bytes", "codeword_fields_to_bytes_from_bits",
    "codeword_bits_to_blocks", "ternary_key_field_bits",
    "version_block_penalty",
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
    penalty depends on the width MULTISET and the group offset, never on a byte
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


def test_a_deep_table_does_not_push_the_next_key_further_along_the_crossbar():
    """Finding 1.4. A table two blocks DEEP stores more rows through the same
    key; a key is one indivisible match and occupies one run of crossbar
    groups. Charging row depth to the next key's start group is a bug --
    latent, because no calibration tree exceeds 512 entries.

    The second key is chosen so the difference is VISIBLE: (77, 42) is 16
    crossbar bytes wide and costs 3 blocks at an even group offset, 4 at an
    odd one. The first key is 3 blocks wide either way.

      correct: key B starts at key A's WIDTH,  3 -> odd  -> B costs 4
      buggy:   key B starts at key A's BLOCKS, 6 -> even -> B costs 3

    Measured on the fixed code: shallow 7, deep 10, delta 3. On the shipped
    code the deep stage comes out at 9 and the delta at 2.
    """
    from src.p4model.packing import crossbar_stages_needed

    key_a = (5,) * 11                    # 11 crossbar bytes, 3 blocks wide
    key_b = (77, 42)                     # 16 crossbar bytes, 3 at even / 4 at odd
    fields = [frozenset({(('a', 0), 11)}), frozenset({(('b', 0), 16)})]

    shallow = crossbar_stages_needed(
        [(3, 11), (3, 16)], key_fields=fields, key_field_bits=[key_a, key_b])
    deep = crossbar_stages_needed(
        [(6, 11), (3, 16)], key_fields=fields, key_field_bits=[key_a, key_b])

    assert (shallow.blocks, deep.blocks) == (7, 10), (shallow, deep)


def test_the_second_keys_offset_parity_is_what_the_previous_key_decides():
    """The property the test above exercises, stated directly so a future
    reader can see why (77, 42) was chosen rather than any 16-byte key."""
    from src.p4model.tables import codeword_to_blocks

    assert [codeword_to_blocks((77, 42), s) for s in range(6)] == [3, 4, 3, 4, 3, 4]
    assert [codeword_to_blocks((5,) * 11, s) for s in range(6)] == [3, 3, 3, 3, 3, 3]


def test_the_version_block_does_advance_the_next_keys_offset():
    """The other half of the ruling, and the half the archive measured:
    independent_low_sd5's ddos key is 11 bytes = 2 crossbar groups, costs 3
    blocks because it saturates, and p4c starts the app key at group 3 -- the
    key's block WIDTH, not its group count. So the version block consumes a
    group and the advance must include it."""
    from src.p4model.tables import codeword_to_blocks, crossbar_groups_needed

    assert crossbar_groups_needed((5,) * 11, 0) == 2
    assert codeword_to_blocks((5,) * 11, 0) == 3


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
