"""Program-level replay in the library (spec 2026-09-29 §5.3, §6.6 --rescore).
The c1 fixture holds already-parsed programs (tests/fixtures/c1_replay_designs.json),
so these run without results/."""
import json
import os
import re

import pytest

from p4gen import p4_replay as rp

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
    breakdown = rp.model_breakdown(_program(design), row_id=name)
    assert (breakdown["stage_depth"], breakdown["blocks"]) == (depth, blocks)
    assert sum(t["blocks"] for t in breakdown["tables"]) == blocks


@pytest.mark.parametrize("name,design", _designs(), ids=[n for n, _ in _designs()])
def test_breakdown_names_every_priced_table_once(name, design):
    breakdown = rp.model_breakdown(_program(design), row_id=name)
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


def test_declared_prices_charge_classification_tables_with_table_blocks():
    from p4gen.p4_replay import Program, declared_prices
    program = Program(
        tables={'get_classification_tree_ddos_0': ['code_a', 'code_b']},
        widths={'code_a': 4, 'code_b': 7},
        bits={'code_a': 27, 'code_b': 52},
        sizes={'get_classification_tree_ddos_0': 600})       # 2 words
    # (27, 52): 3 per word, p4c's charge (independent_low_sd5); the lane count said 2
    assert declared_prices(program) == {'get_classification_tree_ddos_0': 6}


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
    from src.training.row_artifacts import load_forests

    designs = _saved_designs(encoding)
    assert len(designs) == 30
    compared = 0
    for path in designs:
        stored = load_forests(path)
        app, ddos = stored["app"], stored["ddos"]
        names_app, names_ddos = stored["features_app"], stored["features_ddos"]
        if encoding == "joint":
            ia = idd = bps.get_joint_feature_intervals(app, names_app, ddos, names_ddos)
        else:
            ia = bps.get_feature_intervals(app, names_app)
            idd = bps.get_feature_intervals(ddos, names_ddos)
        try:
            usage = multi_model_memory_evaluation(app, ddos, names_app, names_ddos, encoding)
        except RuntimeError as exc:
            trained = type(exc)
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
            # training-only refusals (codeword length, crossbar width) are not
            # generator refusals; the replay prices the program regardless
            continue
        assert (replay["stage_depth"], replay["blocks"]) == (usage.stage_depth, usage.blocks), path
        compared += 1
    # the sample is of trained (feasible) designs: nearly all are priced both ways
    assert compared >= 25, compared
