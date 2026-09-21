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
    """Finding 1.4. A table two blocks DEEP stores more rows through the SAME
    key. Depth must not move the next key along `offsets_for`'s running sum --
    only the key's own width may. Charging row depth there is a bug, latent
    because no calibration tree exceeds 512 entries.

    What makes the difference visible is the stage-sharing margin (packing.py
    `charged`, spec Sec 13.2): key B, (77, 42), is 16 crossbar bytes priced at
    3 blocks, and crossbar_capacity(3) is exactly 16 -- a SATURATED key, so it
    pays one extra block in every ordering where it is not served first. Key A
    is 11 bytes at 3 blocks against a capacity of 16, so it never pays. Both
    stages below therefore cost `A + B + 1` in the worst ordering, and the
    only thing that may change between them is A's own declared depth:

      shallow: A 3 + B 3 + margin 1 = 7
      deep:    A 6 + B 3 + margin 1 = 10   (delta 3 == A's extra depth, exactly)

    A model that charged A's block count instead of A's width would still land
    on 7 and 10 here, because there is no longer any quantity the running sum
    feeds other than "is this offset 0" -- so the guard is now that the delta
    between the two is A's depth and NOTHING else, i.e. deep - shallow == 3.
    """
    from src.p4model.packing import crossbar_stages_needed

    key_a = (5,) * 11                    # 11 crossbar bytes, 3 blocks, 5 spare
    key_b = (77, 42)                     # 16 crossbar bytes, 3 blocks, saturated
    fields = [frozenset({(('a', 0), 11)}), frozenset({(('b', 0), 16)})]

    shallow = crossbar_stages_needed(
        [(3, 11), (3, 16)], key_fields=fields, key_field_bits=[key_a, key_b])
    deep = crossbar_stages_needed(
        [(6, 11), (3, 16)], key_fields=fields, key_field_bits=[key_a, key_b])

    assert (shallow.blocks, deep.blocks) == (7, 10), (shallow, deep)
    assert deep.blocks - shallow.blocks == 3, (shallow, deep)


# --------------------------------------------------------------------------
# The stage-sharing margin (src/p4model/packing.py `charged`, spec Sec 13.2
# "Effect 4"): a table whose key is not the first distinct key in its stage,
# and whose standalone price leaves no spare whole-byte crossbar slot
# (`crossbar_capacity(g) == B`), pays one extra TCAM block for the mandatory
# 2-bit --version-- field.
#
# These four tests stood in tests/test_version_block.py until the 2026-09-20
# TCAM block model rewrite. That file was re-scoped by Task 2 to PER-TABLE
# facts only -- a key's price alone, which no longer depends on where the key
# sits -- so the sharing tests move here, alongside the other packing-level
# guards. Two further tests that stood here
# (test_the_second_keys_offset_parity_is_what_the_previous_key_decides and
# test_the_version_block_does_advance_the_next_keys_offset) are NOT restored:
# they pinned `codeword_to_blocks(bits, start_group)`'s offset sensitivity and
# `crossbar_groups_needed`, and reading p4c's own assembly showed that premise
# false (a block may pair with ANY of a stage's midbytes, not a fixed partner;
# groups need not be consecutive -- rewrite design Sec 2). There is no offset
# parameter left anywhere for them to assert against.
# --------------------------------------------------------------------------
def test_the_packer_charges_sd5s_stage_the_twelve_blocks_p4c_charged():
    # Modelled on independent_low_sd5's stage 6, where p4c charged its app and
    # ddos trees 3 TCAM blocks each (resources.json): one app table plus three
    # ddos tables is 12. Both keys price at 3 blocks in
    # tables.codeword_to_blocks -- the ddos key's 11 bytes saturate two blocks
    # and the version bits push it to three, which is a per-TABLE fact, not a
    # placement one. Neither key is saturated AT three blocks
    # (crossbar_capacity(3) = 16, against 11 and 14 bytes), so the margin stays
    # silent and the stage costs exactly the declared sum. This is the shape
    # that used to be cited as the retired offset mechanism's proof; the new
    # model reaches the same 12 with no placement term involved at all.
    from src.p4model.packing import crossbar_stages_needed
    from src.p4model.tables import codeword_to_blocks

    app = frozenset({(("code", "app_flm"), 7), (("code", "app_plm"), 7)})
    ddos = frozenset({(("code", "ddos_bplm"), 4), (("code", "ddos_plm"), 7)})
    assert codeword_to_blocks((27, 52)) == 3
    plan = crossbar_stages_needed(
        [(3, 14), (3, 11), (3, 11), (3, 11)],
        key_fields=[app, ddos, ddos, ddos],
        key_field_bits=[(54, 56), (27, 52), (27, 52), (27, 52)])
    assert plan.blocks == 12


def test_a_stage_of_one_shared_key_is_never_charged_the_sharing_margin():
    # Every 'joint' design keys every tree on the identical field set, so a
    # joint stage has exactly ONE distinct key and that key is first in every
    # ordering -- the margin cannot fire, however saturated the key is. The
    # key below IS saturated (32 bytes, 6 blocks, crossbar_capacity(6) = 32),
    # so this pins the "not first" half of the rule rather than the "spare
    # slot" half. None of the 8 joint calibration rows shows a single table
    # charged above its declared width.
    from src.p4model.packing import crossbar_stages_needed
    from src.p4model.tables import codeword_to_blocks, crossbar_capacity

    key = frozenset({(("code", "f"), 32)})
    assert crossbar_capacity(codeword_to_blocks((256,))) == 32
    plan = crossbar_stages_needed(
        [(6, 32)] * 3, key_fields=[key] * 3,
        key_field_bits=[(256,)] * 3)
    assert plan.blocks == 18


def test_omitting_key_field_bits_prices_every_table_at_its_declared_width():
    # Without field widths there is no way to ask whether a key is saturated,
    # so the margin is skipped entirely and every table costs what it declares.
    # That is what keeps the range pool -- the one caller that passes no bits
    # -- on exactly its pre-existing pricing.
    from src.p4model.packing import crossbar_stages_needed

    plan = crossbar_stages_needed([(3, 14), (2, 11), (2, 11), (2, 11)])
    assert plan.blocks == 9


def test_two_different_ragged_keys_do_not_share_a_stage():
    # independent_low_sd9, the row the margin exists for. Its app trees key
    # 179 + 204 bits = 49 crossbar bytes at 9 blocks, and crossbar_capacity(9)
    # is exactly 49 -- saturated. Its ddos trees key 37 + 49 bits = 12 bytes at
    # 3 blocks against a capacity of 16 -- five slots spare. 2 app + 2 ddos is
    # exactly 24 blocks and packs both columns as 9+3 | 9+3, so every capacity
    # limit this model knows would let them share a stage; p4c refuses, and
    # charging the app key its margin in any ordering where it is not served
    # first is what reproduces that. The measurement behind it:
    # scripts/tcam_stretch_sweep.py's ragged_ax1_bx5 puts the 49-byte key alone
    # at 9 blocks and ragged_ax1_bx4 puts it beside a 12-byte key at 10.
    from src.p4model.packing import crossbar_stages_needed

    app = frozenset({(("code", "app_a"), 23), (("code", "app_b"), 26)})
    ddos = frozenset({(("code", "ddos_a"), 5), (("code", "ddos_b"), 7)})
    plan = crossbar_stages_needed(
        [(9, 49)] * 5 + [(3, 12)] * 5,
        readiness_levels=[0] * 10,
        key_fields=[app] * 5 + [ddos] * 5,
        key_field_bits=[(179, 204)] * 5 + [(37, 49)] * 5)
    assert plan.occupied == 4
    assert plan.blocks == 61


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
