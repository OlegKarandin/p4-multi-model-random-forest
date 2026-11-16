"""Program-level replay in the library (spec 2026-09-29 §5.3, §6.6 --rescore).
The c1 fixture holds already-parsed programs (tests/fixtures/c1_replay_designs.json),
so these run without results/."""
import json
import os
import re

import pytest

from p4gen import p4_replay as rp
from p4model import refusals
from p4model.errors import ProgramShapeRefused

_C1 = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "fixtures", "c1_replay_designs.json")


def _designs():
    with open(_C1, encoding="utf-8") as handle:
        return sorted(json.load(handle)["designs"].items())


def _program(design):
    return rp.Program(design["tables"], design["widths"], design["bits"], design["sizes"])


@pytest.mark.parametrize("name,design", _designs(), ids=[n for n, _ in _designs()])
def test_breakdown_totals_equal_the_scripts_replay(name, design):
    from scripts.p4_artifact_replay import replay_program
    depth, blocks = replay_program(name, design["tables"], design["widths"],
                                   design["bits"], design["sizes"])
    breakdown = rp.price_program(_program(design), row_id=name)
    assert (breakdown["stage_depth"], breakdown["blocks"]) == (depth, blocks)
    assert sum(t["blocks"] for t in breakdown["tables"]) == blocks


@pytest.mark.parametrize("name,design", _designs(), ids=[n for n, _ in _designs()])
def test_breakdown_names_every_priced_table_once(name, design):
    breakdown = rp.price_program(_program(design), row_id=name)
    names = [t["table"] for t in breakdown["tables"]]
    assert len(names) == len(set(names))
    assert set(names) == set(rp.declared_prices(_program(design)))
    for entry in breakdown["tables"]:
        assert entry["kind"] == ("range" if entry["table"].startswith("table_") else "ternary")


def test_parse_program_text_reads_keys_widths_and_sizes():
    text = """
    struct metadata_t {
        bit<12> code_a;
        bit<16> a_val;
    }
    table table_0_a {
        key = {
            meta.a_val : range;
        }
        size = 9;
    }
    table get_classification_tree_app_0 {
        key = {
            meta.code_a : ternary;
        }
        size = 40;
    }
    """
    program = rp.parse_program_text(text)
    assert program.tables == {"table_0_a": ["a_val"],
                              "get_classification_tree_app_0": ["code_a"]}
    assert program.bits == {"code_a": 12, "a_val": 16}
    assert program.widths == {"code_a": 2, "a_val": 2}
    assert program.sizes == {"table_0_a": 9, "get_classification_tree_app_0": 40}


def test_parse_program_text_prices_a_sliced_key_at_its_slice_width():
    # A code field declared at its container's full width (so p4c's action
    # format has no padding bits to condense another code into) is keyed on
    # the slice holding the code; the crossbar sees the slice, so the model
    # prices the slice width, not the declaration.
    # A field wider than 64 bits is keyed as several slices (p4c asserts on a
    # wider one): one key entry, widths summed.
    text = """
    struct metadata_t {
        bit<32> code_tr_app_0_a;
        bit<8> code_tr_app_0_b;
        bit<80> code_tr_app_0_c;
    }
    table get_classification_tree_app_0 {
        key = {
            meta.code_tr_app_0_a[26:0] : ternary;
            meta.code_tr_app_0_b : ternary;
            meta.code_tr_app_0_c[63:0] : ternary;
            meta.code_tr_app_0_c[73:64] : ternary;
        }
        size = 40;
    }
    """
    program = rp.parse_program_text(text)
    assert program.tables == {"get_classification_tree_app_0": ["code_tr_app_0_a", "code_tr_app_0_b", "code_tr_app_0_c"]}
    assert program.bits == {"code_tr_app_0_a": 27, "code_tr_app_0_b": 8, "code_tr_app_0_c": 74}
    assert program.widths == {"code_tr_app_0_a": 4, "code_tr_app_0_b": 1, "code_tr_app_0_c": 10}


_FULL_WIDTH = """
    struct metadata_t {
        bit<8> code_tk_app_a;
        bit<32> code_tk_ddos_a;
        bit<16> a_val;
    }
@pa_container_size("ingress", "ig_md.code_tk_app_a", 8)
@pa_container_size("ingress", "ig_md.code_tk_ddos_a", 32)
    action set_code_a (bit<8> code_app, bit<32> code_ddos) {
        meta.code_tk_app_a = code_app;
        meta.code_tk_ddos_a = code_ddos;
    }
    table table_0_a {
        key = {
            meta.a_val : range;
        }
        size = 9;
    }
    table get_classification_tree_app_0 {
        key = {
            meta.code_tk_app_a[2:0] : ternary;
        }
        size = 4;
    }
    table get_classification_tree_app_1 {
        key = {
            meta.code_tk_app_a[2:0] : ternary;
        }
        size = 4;
    }
    table get_classification_tree_ddos_0 {
        key = {
            meta.code_tk_ddos_a[19:0] : ternary;
        }
        size = 21;
    }
"""


def test_replay_live_width_from_slice():
    # Decision 4.3: a code field is declared at its containers' full width and every
    # tree keys the slice of its live bits; the replay prices the slice (the live
    # width), lays it in the pinned containers, and counts the action's bus bytes as
    # the containers it writes (S14): 1 + 4 = 5, not ceil((3 + 20) / 8) = 3.
    program = rp.parse_program_text(_FULL_WIDTH)
    assert program.bits["code_tk_app_a"] == 3 and program.widths["code_tk_app_a"] == 1
    assert program.bits["code_tk_ddos_a"] == 20 and program.widths["code_tk_ddos_a"] == 3
    assert program.action_writes == {"set_code_a": ["code_tk_app_a", "code_tk_ddos_a"]}
    assert rp.range_action_bytes(program) == {"table_0_a": 5}
    assert rp.tree_key_slices(program, "get_classification_tree_ddos_0") == (
        ("code_tk_ddos_a", (("W", 20),)),)
    # The same live bits declared at their live width price the same tables; only
    # the declaration differs.
    live = (_FULL_WIDTH.replace("bit<8> code_tk_app_a", "bit<3> code_tk_app_a")
            .replace("bit<32> code_tk_ddos_a", "bit<20> code_tk_ddos_a"))
    # (price_program: the fragment has no prologue / pins for check_program; with
    # the live width the field is no longer its containers' width, which
    # check_program refuses -- unexpected_shape:code_width.)
    assert rp.price_program(rp.parse_program_text(live)) == rp.price_program(program)


def test_replay_refuses_slices_that_do_not_hold_the_live_bits_from_bit_0():
    with pytest.raises(ProgramShapeRefused, match="one_run_from_bit_0"):
        rp.parse_program_text(_FULL_WIDTH.replace("code_tk_app_a[2:0] : ternary;\n        }\n"
                                                  "        size = 4;\n    }\n    table "
                                                  "get_classification_tree_app_1",
                                                  "code_tk_app_a[3:1] : ternary;\n        }\n"
                                                  "        size = 4;\n    }\n    table "
                                                  "get_classification_tree_app_1"))
    with pytest.raises(ProgramShapeRefused, match="same_live_bits"):
        rp.parse_program_text(_FULL_WIDTH.replace(
            "meta.code_tk_app_a[2:0] : ternary;\n        }\n        size = 4;\n    }\n"
            "    table get_classification_tree_ddos_0",
            "meta.code_tk_app_a[1:0] : ternary;\n        }\n        size = 4;\n    }\n"
            "    table get_classification_tree_ddos_0"))


def test_declared_prices_charge_classification_tables_with_table_blocks():
    from p4gen.p4_replay import Program, declared_prices
    program = Program(
        tables={'get_classification_tree_ddos_0': ['code_a', 'code_b']},
        widths={'code_a': 4, 'code_b': 7},
        bits={'code_a': 27, 'code_b': 52},
        sizes={'get_classification_tree_ddos_0': 600})       # 2 words
    # (27, 52): 3 per word, p4c's charge (independent_low_sd5); the lane count said 2
    assert declared_prices(program) == {'get_classification_tree_ddos_0': 6}


# --- per-tree key order: training lists a tree's fields as the program text does ---

def _two_feature_forest(names, n_classes, seed):
    """Depth-2 trees over two features, the first binary (every split at threshold 0, so
    two such forests discretize it identically: a shared disjoint field)."""
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from p4gen import build_p4_script as bps
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 3000, size=(400, 2))
    X[:, 0] = rng.randint(0, 2, size=400)
    y = ((1500 * X[:, 0] + X[:, 1]) // (4500 // n_classes)).clip(0, n_classes - 1)
    return bps.dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=4, max_depth=2, random_state=seed).fit(X, y))


def test_disjoint_key_order_is_the_texts_with_a_shared_feature_after_a_ddos_only_one(tmp_path):
    """Disjoint: a ddos tree keying a shared field (flow_iat_max, listed with the app
    fields) and a ddos-only one that sorts before it (bwd_iat_total, listed after them)
    keys them in the program's order, and training's key slices list them in that order
    too (the order decides the crossbar container numbering by first use)."""
    from p4gen import build_p4_script as bps
    from p4gen.evaluation import _pool_inputs
    names_app, names_ddos = ["flow_iat_max", "fwd_packet_length_max"], ["flow_iat_max", "bwd_iat_total"]
    app, ddos = _two_feature_forest(names_app, 3, 2), _two_feature_forest(names_ddos, 2, 5)
    ia, idd = bps.get_feature_intervals(app, names_app), bps.get_feature_intervals(ddos, names_ddos)
    assert ia["flow_iat_max"] == idd["flow_iat_max"] and "bwd_iat_total" in idd
    written = bps.generate_P4_code(
        3, 2, app, ddos, feature_intervals_app=ia, feature_intervals_ddos=idd,
        output_dir=str(tmp_path) + os.sep, output_filename="order.p4",
        selected_features_app=names_app, selected_features_ddos=names_ddos, encoding="disjoint")
    program = rp.parse_program(written)
    pool = _pool_inputs(app, ddos, names_app, names_ddos, "disjoint")
    plan = bps._resolve_disjoint_feature_plan(ia, idd)
    raw = {"code_" + name: feature for name, (feature, _, _) in plan.items()}
    trees = ["get_classification_tree_app_%d" % i for i in range(len(app.estimators_))] + [
        "get_classification_tree_ddos_%d" % i for i in range(len(ddos.estimators_))]
    both = False
    for name, slices in zip(trees, pool["ternary_key_slices"]):
        # training names each field as the program does (the crossbar numbers containers by name)
        text = rp.tree_key_slices(program, name)
        assert [field_id for field_id, _ in slices] == [field for field, _ in text], name
        both |= [raw[field] for field, _ in text] == ["flow_iat_max", "bwd_iat_total"]
    assert both      # some ddos tree keys the shared field and the ddos-only one


def test_a_split_on_a_feature_without_a_code_field_is_refused():
    """A tree splitting a feature its task has no code field for (intervals that are not
    its trees') would be keyed on nothing for that split: ValueError in the generator and
    in training."""
    import pytest as _pytest
    from p4gen import build_p4_script as bps
    from p4gen.evaluation import _tree_keys
    from p4model.program import APP_TASK, TaskSpec
    names = ["flow_iat_max", "fwd_packet_length_max"]
    forest = _two_feature_forest(names, 3, 2)
    own = bps.get_feature_intervals(forest, names)
    missing = {f: iv for f, iv in own.items() if f != "flow_iat_max"}
    task = TaskSpec(APP_TASK, 3, list(forest.estimators_), names, missing)
    with _pytest.raises(ValueError, match="flow_iat_max"):
        bps.tree_key_features(task, missing.__contains__)
    nodes = bps.tree_nodes_for(forest, names)
    trees = [(APP_TASK, i, n) for i, n in enumerate(nodes.values())]
    with _pytest.raises(ValueError, match="flow_iat_max"):
        _tree_keys(trees, {APP_TASK: missing}, {APP_TASK: list(missing)})


# --- training == replay on saved DEV forests (decision 2.3, key pruning) -------

_ROW = re.compile(r'^(?P<arm>.+)_M\w+_s(?P<split>\d+)_k(?P<k>\d+)$')
# encoding -> (arm, campaign whose forests/ holds it); DEV splits 0-24 only.
_SAVED_ARMS = {
    "joint": ("joint-off", "campaign_2026_11"),
    "disjoint": ("independent", "campaign_2026_11"),
    "per_task": ("joint-task", "campaign_2026_11d"),
    "per_tree": ("joint-tree", "campaign_2026_11e"),
}


def _results_dir():
    """The checkout's results/ (outside git): beside tests/, or, from a git
    worktree under .claude/worktrees/, the main checkout's."""
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for root in (here, os.path.normpath(os.path.join(here, "..", "..", ".."))):
        if os.path.isdir(os.path.join(root, "results", "campaign_2026_11", "forests")):
            return os.path.join(root, "results")
    return None


def _saved_designs(encoding, per_group=15, seed=20261114):
    """per_group designs with k <= 8 and per_group with k >= 9, drawn
    deterministically from the arm's saved DEV forests."""
    import random
    arm, campaign = _SAVED_ARMS[encoding]
    folder = os.path.join(_results_dir(), campaign, "forests")
    rows = []
    for name in sorted(os.listdir(folder)):
        match = _ROW.match(name[:-len(".joblib")]) if name.endswith(".joblib") else None
        if match and match["arm"] == arm and int(match["split"]) <= 24:
            rows.append((os.path.join(folder, name), int(match["k"])))
    rng = random.Random("%d-%s" % (seed, encoding))
    out = []
    for low in (True, False):
        pool = sorted(path for path, k in rows if (k <= 8) == low)
        out += rng.sample(pool, min(per_group, len(pool)))
    return out


@pytest.mark.skipif(_results_dir() is None, reason="results/ (saved forests) not present")
@pytest.mark.parametrize("encoding", sorted(_SAVED_ARMS))
def test_training_equals_replay_pruned(tmp_path, encoding):
    """30 saved designs per encoding: training's price (evaluation._pool_inputs ->
    assemble_usage) equals the replay of the generated program, stage depth and
    blocks, now that every tree keys only its own features; a design one path
    refuses, the other refuses the same way."""
    from p4gen import build_p4_script as bps
    from p4gen.evaluation import multi_model_memory_evaluation
    from src.training.dataset import replace_means
    from src.training.row_artifacts import load_forests

    designs = _saved_designs(encoding)
    assert len(designs) == 30
    compared = 0
    for path in designs:
        stored = load_forests(path)
        app, ddos = stored["app"], stored["ddos"]
        # saved before Task 38: each Mean under the name of its replacement
        names_app = replace_means(stored["features_app"])
        names_ddos = replace_means(stored["features_ddos"])
        if encoding == "joint":
            ia = idd = bps.get_joint_feature_intervals(app, names_app, ddos, names_ddos)
        else:
            ia = bps.get_feature_intervals(app, names_app)
            idd = bps.get_feature_intervals(ddos, names_ddos)
        try:
            usage = multi_model_memory_evaluation(app, ddos, names_app, names_ddos, encoding)
        except RuntimeError as exc:
            trained, refused = type(exc), exc
        else:
            trained = None
        try:
            written = bps.generate_P4_code(
                3, 2, app, ddos, feature_intervals_app=ia, feature_intervals_ddos=idd,
                output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
                selected_features_app=names_app, selected_features_ddos=names_ddos,
                encoding=encoding)
        except RuntimeError as exc:
            assert trained is type(exc), (path, trained, exc)
            continue
        replay = rp.model_breakdown(rp.parse_program(written))
        if trained is not None:
            # a refusal the generator does not apply (key bytes, PHV, placement):
            # the replay of the program names the same constraint
            assert replay["refusal"] == refusals.of(refused)[0], (path, replay["refusal"])
            continue
        assert (replay["stage_depth"], replay["blocks"]) == (usage.stage_depth, usage.blocks), path
        compared += 1
    # the sample is of trained (feasible) designs: nearly all are priced both ways
    assert compared >= 25, compared
