"""Control-plane entries of the per-tree program (build_p4_script.generate_per_tree_table_entries).

Asserted on MEANING, like test_per_task_codes' entry tests: the entries are played as the switch plays them (a range table's
action writes each tree's code, every tree table matches value/mask on each key slice) and must give every tree's own class.
The slicing is rule A of the 2026-11 action-data-bus investigation; that the compiled program loads and runs these entries
was checked live on tofino-model (results/encoding_evidence_2026_11/adb_investigation/step7_live)."""
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.switch_semantics import per_tree_class_indices, vote_winner
from p4model.program import APP_TASK, DDOS_TASK, TaskSpec, per_tree_field

FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_mean"]


def _forest(n_classes, seed, depth, n_trees=2):
    # the label follows feature 0 in fine steps, so deep trees split it many times: codes wider than 64 bits
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 5000, size=(4000, len(FEATURES)))
    y = ((X[:, 0] // 37) + (X[:, 1] > 2500) + (X[:, 2] // 1700)) % n_classes
    return bps.dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=n_trees, max_depth=depth, random_state=seed).fit(X, y))


def _tasks(app, ddos):
    return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), FEATURES, bps.get_feature_intervals(ddos, FEATURES)),
            TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, bps.get_feature_intervals(app, FEATURES))]


_SLICE = re.compile(r"^(code_tr_\w+?)_(\d+)_(\d+)_$")


def _play(entries, row):
    """{tree table: class} for one input, the entries interpreted as the switch does."""
    codes, by_table = {}, {}
    for entry in entries:
        by_table.setdefault(entry["table_name"], []).append(entry)
    for table, rows in by_table.items():
        if not table.startswith("table_"):
            continue
        feature = rows[0]["action"][len("set_code_"):]
        hits = [e for e in rows if int(e["key_fields"][feature + "_val"]["start"]) <= row[feature]
                <= int(e["key_fields"][feature + "_val"]["end"])]
        assert len(hits) == 1, (table, row)
        for param, value in hits[0]["action_params"].items():
            task, tree = re.match(r"^code_(app|ddos)_(\d+)$", param).groups()
            codes["code_" + per_tree_field(task, int(tree), feature)] = int(value)
    classes = {}
    for table, rows in by_table.items():
        if not table.startswith("get_classification_tree_"):
            continue

        def matches(entry):
            for name, spec in entry["key_fields"].items():
                m = _SLICE.match(name)
                field, hi, lo = (m[1], int(m[2]), int(m[3])) if m else (name, None, 0)
                value = codes[field] >> lo
                if hi is not None:
                    value &= (1 << (hi - lo + 1)) - 1
                if value & int(spec["mask"], 16) != int(spec["value"], 16):
                    return False
            return True
        hits = [e for e in rows if not e["is_default_action"] and matches(e)]
        assert len(hits) <= 1, (table, row)
        hits = hits or [e for e in rows if e["is_default_action"]]
        assert len(hits) == 1, (table, row)
        classes[table] = int(hits[0]["action_params"]["class"])
    return classes


def test_bfrt_key_name_keeps_a_slices_bounds():
    assert bps.bfrt_key_name("meta.code_tr_app_0_bwd_packet_length_mean[63:0]") == "code_tr_app_0_bwd_packet_length_mean_63_0_"
    assert bps.bfrt_key_name("meta.code_tr_app_0_bwd_packet_length_mean[89:64]") == "code_tr_app_0_bwd_packet_length_mean_89_64_"
    assert bps.bfrt_key_name("meta.code_flow_iat_max") == "code_flow_iat_max"
    assert bps.bfrt_key_name("meta.flow_iat_max_val") == "flow_iat_max_val"


@pytest.mark.parametrize("width, pinned, slices", [
    (3, 8, None),               # a narrow code keeps its padding: two may share a byte
    (8, 8, None),
    (27, 32, [(26, 0)]),        # declared bit<32>, keyed on its own 27 bits
    (32, 32, None),             # fills its container: nothing to pad
    (36, 40, [(35, 0)]),
    (64, 64, None),
    (74, 80, [(63, 0), (73, 64)]),   # p4c asserts on one slice wider than 64 bits
    (90, 96, [(63, 0), (89, 64)]),
    (200, 208, [(63, 0), (127, 64), (191, 128), (199, 192)]),
])
def test_per_tree_key_slices_is_rule_a(width, pinned, slices):
    assert bps.per_tree_key_slices(width, pinned) == slices


def test_fixture_has_codes_that_are_split_sliced_and_whole():
    app, ddos = _forest(3, 0, depth=9), _forest(2, 1, depth=3)
    names = {n for e in bps.generate_per_tree_table_entries(_tasks(app, ddos))
             if e["table_name"].startswith("get_") for n in e["key_fields"]}
    assert any(n.endswith("_63_0_") for n in names), "no code wider than 64 bits"
    assert any(_SLICE.match(n) and not n.endswith("_63_0_") and n.split("_")[-2] == "0" for n in names), "no single slice"
    assert any(not _SLICE.match(n) for n in names), "no whole key"


@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("seed", range(3))
def test_entries_classify_every_input_exactly_as_each_tree(seed, discount):
    app, ddos = _forest(3, seed, depth=9), _forest(2, seed + 10, depth=4)
    entries = bps.generate_per_tree_table_entries(_tasks(app, ddos), use_default_action_discount=discount)
    rng = np.random.RandomState(seed)
    X = np.vstack([np.array([[v] * len(FEATURES) for v in (0, 1, 2500, 4999, 5000, 65535)]),
                   rng.randint(0, 5100, size=(250, len(FEATURES)))])
    trees = {APP_TASK: per_tree_class_indices(app, X), DDOS_TASK: per_tree_class_indices(ddos, X)}
    for s, values in enumerate(X):
        got = _play(entries, dict(zip(FEATURES, values)))
        want = {"get_classification_tree_%s_%d" % (t, i): int(c) for t in trees for i, c in enumerate(trees[t][:, s])}
        assert got == want
        assert vote_winner([got["get_classification_tree_app_%d" % i] for i in range(2)], 3) == \
            vote_winner(list(trees[APP_TASK][:, s]), 3)


def test_each_tree_writes_its_own_thermometer_code():
    app, ddos = _forest(3, 2, depth=5), _forest(2, 3, depth=3)
    tasks = _tasks(app, ddos)
    own = {t.name: bps.per_tree_intervals(t) for t in tasks}
    for entry in bps.generate_per_tree_table_entries(tasks):
        if not entry["table_name"].startswith("table_"):
            continue
        feature = entry["action"][len("set_code_"):]
        x = int(entry["key_fields"][feature + "_val"]["start"])
        for param, value in entry["action_params"].items():
            task, tree = re.match(r"^code_(app|ddos)_(\d+)$", param).groups()
            iv = own[task][int(tree)][feature]
            idx = next(j for j, (lo, hi) in enumerate(iv) if lo <= x <= hi)
            assert int(value) == int(bps.thermometer_code(len(iv), idx), 2)
