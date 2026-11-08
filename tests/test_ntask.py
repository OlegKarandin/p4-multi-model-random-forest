"""N-task resource model and generator (spec 2026-10-11).

The N = 2 compatibility gate (byte-identical programs, identical breakdowns on
every archived design) is scripts/ntask_regression.py, run against data that
is not in git; these tests pin the N-generic behaviour on synthetic forests:
N = 1 and N = 3 programs generate, parse back through p4_replay and price;
placement priorities are N..1; disjoint grouping on a hand-made 3-task plan;
and the two-task wrapper is exactly the N-task generator with (ddos, app)."""
import json
import os
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.p4_replay import (model_breakdown, parse_program,
                                 parse_program_text, table_tasks)
from p4model.program import (APP_TASK, DDOS_TASK, PLACEMENT_PRIORITY,
                                 SHARED_TASK, TaskSpec, placement_priorities)
from p4model.usage import tree_readiness_levels

FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_mean",
            "flow_iat_min", "bwd_packet_length_max"]


def _forest(n_classes, seed, features=FEATURES, n_trees=2, depth=3):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2000, size=(300, len(features)))
    y = (X[:, seed % len(features)] // (2000 // n_classes)).clip(0, n_classes - 1)
    return RandomForestClassifier(n_estimators=n_trees, max_depth=depth,
                                  random_state=seed).fit(X, y)


def _task(name, n_classes, seed, features=FEATURES, intervals=None, **kwargs):
    forest = _forest(n_classes, seed, features, **kwargs)
    ordered = list(features)
    if intervals is None:
        intervals = bps.get_feature_intervals(forest, features)
    else:
        # Hand-made intervals do not come from these trees, so no codewords:
        # without the ordered feature list the tables are sized by leaves.
        ordered = None
    return TaskSpec(name=name, num_classes=n_classes, trees=list(forest.estimators_),
                    features=ordered, intervals=intervals)


def _generate(tmp_path, tasks, name="design.p4"):
    path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep,
                                   output_filename=name)
    with open(path, encoding="utf-8") as handle:
        return path, handle.read()


# --- task descriptor and placement priority -------------------------------

def test_placement_priority_is_n_minus_index():
    assert placement_priorities(["a", "b", "c"]) == {"a": 3, "b": 2, "c": 1}
    assert placement_priorities(["only"]) == {"only": 1}
    # N = 2 in the (ddos, app) order reproduces the constant every two-task
    # caller has used since C5.
    assert placement_priorities([DDOS_TASK, APP_TASK]) == PLACEMENT_PRIORITY
    assert PLACEMENT_PRIORITY == {DDOS_TASK: 2, APP_TASK: 1}


@pytest.mark.parametrize("name", ["App", "a-b", "", "shared", "x y"])
def test_task_names_must_be_lowercase_identifiers_and_not_reserved(name):
    with pytest.raises(ValueError):
        TaskSpec(name=name, num_classes=2)


def test_task_names_must_be_unique(tmp_path):
    a = _task("a", 2, 1)
    with pytest.raises(ValueError, match="unique"):
        bps.generate_p4_program([a, a], output_dir=str(tmp_path) + os.sep)


# --- disjoint grouping ------------------------------------------------------

def test_disjoint_grouping_on_a_hand_made_three_task_plan():
    one = [(0, 10), (11, 65535)]
    two = [(0, 20), (21, 65535)]
    three = [(0, 30), (31, 65535)]
    x = {"f_all": one, "f_xy": one, "f_each": one, "f_x": one}
    y = {"f_all": one, "f_xy": one, "f_each": two, "f_y": two}
    z = {"f_all": one, "f_xy": two, "f_each": three}
    plan = bps.resolve_feature_plan([("x", x), ("y", y), ("z", z)])
    assert {name: (raw, models) for name, (raw, _, models) in plan.items()} == {
        "f_all": ("f_all", {"x", "y", "z"}),   # one group = every selecting task
        "x_y_f_xy": ("f_xy", {"x", "y"}),       # group named in task order
        "z_f_xy": ("f_xy", {"z"}),
        "x_f_each": ("f_each", {"x"}),
        "y_f_each": ("f_each", {"y"}),
        "z_f_each": ("f_each", {"z"}),
        "f_x": ("f_x", {"x"}),                  # only one task selects it
        "f_y": ("f_y", {"y"}),
    }
    # Program order is the reverse of task order (lowest priority first),
    # each task's features in its own order, a group at its first member.
    assert list(plan) == ["f_all", "z_f_xy", "z_f_each", "x_y_f_xy", "y_f_each",
                          "f_y", "x_f_each", "f_x"]


def test_two_task_grouping_matches_the_legacy_resolver():
    one = [(0, 10), (11, 65535)]
    two = [(0, 20), (21, 65535)]
    app = {"a": one, "both": one, "x": one}
    ddos = {"d": two, "both": one, "x": two}
    legacy = bps._resolve_disjoint_feature_plan(app, ddos)
    assert bps.resolve_feature_plan([(DDOS_TASK, ddos), (APP_TASK, app)]) == legacy
    assert list(legacy) == ["a", "both", "app_x", "d", "ddos_x"]


# --- generated programs ------------------------------------------------------

def test_single_task_program_generates_parses_and_prices(tmp_path):
    path, text = _generate(tmp_path, [_task("solo", 2, 3)])
    assert re.findall(r"@placement_priority\((\d+)\)\s*\n\s*table get_classification_tree_solo_(\d+)",
                      text) == [("1", "0"), ("1", "1")]
    assert "table vote_solo" in text
    assert "_app" not in text and "_ddos" not in text
    breakdown = model_breakdown(parse_program(path))
    assert breakdown["stage_depth"] > 0 and breakdown["blocks"] > 0
    assert {t["table"] for t in breakdown["tables"] if t["kind"] == "ternary"} == {
        "get_classification_tree_solo_0", "get_classification_tree_solo_1"}


@pytest.mark.parametrize("encoding", ["joint", "disjoint"])
def test_three_task_program_generates_parses_and_prices(tmp_path, encoding):
    tasks = [_task("ddos", 2, 1), _task("app", 3, 2), _task("third", 4, 5, n_trees=1)]
    if encoding == "joint":
        joint = bps.feature_intervals_from_nodes(bps.merge_tree_nodes(*(
            bps.tree_nodes_for(type("F", (), {"estimators_": t.trees})(), FEATURES)
            for t in tasks)))
        tasks = [TaskSpec(t.name, t.num_classes, t.trees, t.features, joint) for t in tasks]
    path, text = _generate(tmp_path, tasks)
    found = re.findall(r"@placement_priority\((\d+)\)\s*\n\s*table get_classification_tree_(\w+)_(\d+)",
                       text)
    assert {(task, int(p)) for p, task, _ in found} == {("ddos", 3), ("app", 2), ("third", 1)}
    # Tree ids are per task, from 0, with no global offset.
    assert sorted((task, int(i)) for _, task, i in found) == [
        ("app", 0), ("app", 1), ("ddos", 0), ("ddos", 1), ("third", 0)]
    for task in ("ddos", "app", "third"):
        assert "table vote_%s" % task in text
        assert "bit<%d> classification_%s;" % ({"ddos": 1, "app": 2, "third": 2}[task], task) in text
    program = parse_program(path)
    trees = {name: p for name, p in program.priorities.items()
             if name.startswith("get_classification_tree_")}
    assert trees == {name: int(p) for p, task, i in found
                     for name in ["get_classification_tree_%s_%s" % (task, i)]}
    # Every range table outranks every tree: N + 1, program.range_table_priority.
    assert {p for name, p in program.priorities.items() if name not in trees} == {4}
    assert {name for name in program.priorities if name not in trees} == {
        name for name in program.tables if name.startswith("table_")}
    breakdown = model_breakdown(program)
    assert breakdown["stage_depth"] > 0 and breakdown["blocks"] > 0
    assert sum(t["kind"] == "ternary" for t in breakdown["tables"]) == 5


def test_disjoint_three_task_program_names_groups_and_labels_their_range_tables(tmp_path):
    one = [(0, 10), (11, 65535)]
    two = [(0, 20), (21, 65535)]
    common = ["flow_iat_max", "bwd_iat_mean"]
    tasks = [
        _task("x", 2, 1, common, intervals={"flow_iat_max": one, "bwd_iat_mean": one}),
        _task("y", 2, 2, common, intervals={"flow_iat_max": one, "bwd_iat_mean": two}),
        _task("z", 2, 3, common, intervals={"flow_iat_max": two, "bwd_iat_mean": two}),
    ]
    path, text = _generate(tmp_path, tasks)
    for field in ("code_x_y_flow_iat_max", "code_z_flow_iat_max",
                  "code_x_bwd_iat_mean", "code_y_z_bwd_iat_mean"):
        assert re.search(r"bit<\d+> %s;" % field, text), field
    # Registers stay per distinct raw feature: one value field each.
    assert text.count("bit<16> flow_iat_max_val;") == 1
    labels = table_tasks(parse_program(path).tables)
    range_labels = {name.split("_", 2)[2]: label for name, label in labels.items()
                    if name.startswith("table_")}
    assert range_labels == {"x_y_flow_iat_max": frozenset({"x", "y"}),
                            "z_flow_iat_max": "z", "x_bwd_iat_mean": "x",
                            "y_z_bwd_iat_mean": frozenset({"y", "z"})}
    model_breakdown(parse_program(path))


def test_two_task_wrapper_is_the_n_task_generator_in_ddos_app_order(tmp_path):
    clf_app, clf_ddos = _forest(3, 2), _forest(2, 1)
    app_iv = bps.get_feature_intervals(clf_app, FEATURES)
    ddos_iv = bps.get_feature_intervals(clf_ddos, FEATURES)
    legacy = bps.generate_P4_code(
        3, 2, clf_app, clf_ddos, feature_intervals_app=app_iv,
        feature_intervals_ddos=ddos_iv, output_dir=str(tmp_path) + os.sep,
        output_filename="legacy.p4", selected_features_app=FEATURES,
        selected_features_ddos=FEATURES)
    _, generic = _generate(tmp_path, [
        TaskSpec(DDOS_TASK, 2, list(clf_ddos.estimators_), FEATURES, ddos_iv),
        TaskSpec(APP_TASK, 3, list(clf_app.estimators_), FEATURES, app_iv)], "generic.p4")
    with open(legacy, encoding="utf-8") as handle:
        assert handle.read() == generic


# --- replay labels and readiness ---------------------------------------------

def test_tree_readiness_waits_only_for_range_tables_serving_the_trees_task():
    stages = (4, 6, 8)
    labels = [frozenset({"x", "y"}), "z", SHARED_TASK]
    assert tree_readiness_levels(stages, labels, ["x", "y", "z"],
                                 tasks=("x", "y", "z")) == [9, 9, 9]
    # x and y read only the stage-4 table, but rule E keeps them from
    # preceding z's table at 6: they may share its stage.
    assert tree_readiness_levels(stages[:2], labels[:2], ["x", "y", "z"],
                                 tasks=("x", "y", "z")) == [6, 6, 7]
    with pytest.raises(ValueError, match="unknown task"):
        tree_readiness_levels((4,), [frozenset({"x", "w"})], ["x"], tasks=("x",))


def test_replay_recovers_any_task_name_from_tree_tables():
    tables = {
        "get_classification_tree_t_one_0": ["code_a", "code_t_one_t_two_b"],
        "get_classification_tree_t_two_3": ["code_t_one_t_two_b", "code_c"],
        "get_classification_tree_t3_1": ["code_c"],
        "table_0_a": ["a_val"], "table_1_t_one_t_two_b": ["b_val"],
        "table_2_c": ["c_val"],
    }
    assert table_tasks(tables) == {
        "get_classification_tree_t_one_0": "t_one",
        "get_classification_tree_t_two_3": "t_two",
        "get_classification_tree_t3_1": "t3",
        "table_0_a": "t_one",
        "table_1_t_one_t_two_b": frozenset({"t_one", "t_two"}),
        "table_2_c": frozenset({"t_two", "t3"}),
    }


def test_program_without_pragmas_falls_back_to_the_two_task_priorities():
    text = ("struct m {\n bit<4> code_a;\n bit<16> a_val;\n}\n"
            "    table get_classification_tree_app_0 {\n        key = {\n"
            "            meta.code_a : ternary;\n        }\n        size = 3;\n    }\n")
    assert parse_program_text(text).priorities == {}


def test_feature_plan_rejects_colliding_resolved_names():
    # Task "a" splitting "b_c" alone and task "a_b" splitting "c" alone would
    # both resolve to "a_b_c"; the second must not silently replace the first.
    one = [(0, 10), (11, 65535)]
    two = [(0, 20), (21, 65535)]
    with pytest.raises(ValueError, match="a_b_c"):
        bps.resolve_feature_plan([("a", {"b_c": one, "c": one}),
                                  ("a_b", {"b_c": two, "c": two})])


# --- control-plane entries ---------------------------------------------------

def _int_forest(n_classes, seed, features=FEATURES, n_trees=2, depth=3):
    """A forest with integer thresholds, as every deployed forest has
    (dt_thresholds_float_to_int), so the switch and sklearn agree exactly on
    integer inputs."""
    return bps.dt_thresholds_float_to_int(
        _forest(n_classes, seed, features, n_trees=n_trees, depth=depth))


def _tasks(members, encoding):
    """TaskSpecs for (name, n_classes, forest, features) members: one joint
    interval set over every forest, or each task's own."""
    if encoding == "joint":
        joint = bps.feature_intervals_from_nodes(bps.merge_tree_nodes(*(
            bps.tree_nodes_for(forest, names) for _, _, forest, names in members)))
        intervals = [joint] * len(members)
    else:
        intervals = [bps.get_feature_intervals(forest, names)
                     for _, _, forest, names in members]
    return [TaskSpec(name, n_classes, list(forest.estimators_), names, iv)
            for (name, n_classes, forest, names), iv in zip(members, intervals)]


def _switch_tree_classes(entries, row):
    """Each tree table's class for one input, by interpreting the entries the
    way the switch does: every range table writes its code field from the
    value it keys on, then every tree table matches those codes ternary-wise.
    Exactly one entry of each table may match (intervals are disjoint and a
    tree's leaves partition its input space); a tree with no match takes its
    control-plane default."""
    codes, by_table = {}, {}
    for entry in entries:
        by_table.setdefault(entry["table_name"], []).append(entry)
    for table, rows in by_table.items():
        if not table.startswith("table_"):
            continue
        hits = [e for e in rows
                for field, span in e["key_fields"].items()
                if int(span["start"]) <= row[field[:-len("_val")]] <= int(span["end"])]
        assert len(hits) == 1, (table, row)
        codes["code_" + hits[0]["action"][len("set_code_"):]] = int(hits[0]["action_params"]["code"])
    classes = {}
    for table, rows in by_table.items():
        if not table.startswith("get_classification_tree_"):
            continue
        hits = [e for e in rows if not e["is_default_action"]
                and all(codes[field] & int(spec["mask"], 16) == int(spec["value"], 16)
                        for field, spec in e["key_fields"].items())]
        assert len(hits) <= 1, (table, row)
        if not hits:
            hits = [e for e in rows if e["is_default_action"]]
        assert len(hits) == 1, (table, row)
        classes[table] = int(hits[0]["action_params"]["class"])
    return classes


@pytest.mark.parametrize("encoding,discount", [
    ("joint", False), ("disjoint", False), ("disjoint", True)])
def test_entries_classify_every_input_exactly_as_each_tree_does(tmp_path, encoding, discount):
    from p4gen.switch_semantics import per_tree_class_indices
    members = [("ddos", 2, _int_forest(2, 1), FEATURES),
               ("app", 3, _int_forest(3, 2), FEATURES),
               # A third task on a feature subset: under disjoint some
               # features are read by two tasks only, some by one.
               ("third", 4, _int_forest(4, 5, FEATURES[:3], n_trees=1), FEATURES[:3])]
    tasks = _tasks(members, encoding)
    path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep,
                                   output_filename="design.p4",
                                   use_default_action_discount=discount,
                                   entries_filename="entries.json")
    with open(os.path.join(str(tmp_path), "entries.json"), encoding="utf-8") as handle:
        entries = json.load(handle)
    assert entries == bps.generate_table_entries(tasks, use_default_action_discount=discount)
    # Every table the entries address is one the program declares, and the
    # key fields are the ones it declares for that table.
    declared = parse_program(path).tables
    for entry in entries:
        assert entry["table_name"] in declared
        if not entry["is_default_action"]:
            assert sorted(entry["key_fields"]) == sorted(declared[entry["table_name"]])
    rng = np.random.RandomState(7)
    X = rng.randint(0, 2000, size=(200, len(FEATURES)))
    expected = {}
    for name, _, forest, names in members:
        columns = [FEATURES.index(f) for f in names]
        for i, per_tree in enumerate(per_tree_class_indices(forest, X[:, columns])):
            expected["get_classification_tree_%s_%d" % (name, i)] = per_tree
    for s, values in enumerate(X):
        got = _switch_tree_classes(entries, dict(zip(FEATURES, values)))
        assert got == {table: int(classes[s]) for table, classes in expected.items()}


def test_entries_for_joint_two_tasks_equal_the_legacy_main_path(tmp_path):
    app, ddos = _int_forest(3, 2), _int_forest(2, 1)
    # main.generate_p4_from_models' path: one joint interval set over app's
    # trees then ddos', ddos re-keyed past app's trees.
    nodes = bps.merge_tree_nodes(bps.tree_nodes_for(app, FEATURES),
                                 bps.tree_nodes_for(ddos, FEATURES))
    intervals = bps.feature_intervals_from_nodes(nodes)
    paths = bps.get_root_to_leaf_paths(nodes)
    bps.get_table_entries(paths, intervals, bps.generate_codewords(paths, intervals),
                          len(app.estimators_), path_to_output=str(tmp_path) + os.sep)
    with open(os.path.join(str(tmp_path), "table_entries.json"), encoding="utf-8") as handle:
        legacy = json.load(handle)
    tasks = [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), FEATURES, intervals),
             TaskSpec(APP_TASK, 3, list(app.estimators_), FEATURES, intervals)]
    assert bps.generate_table_entries(tasks) == legacy


def test_entries_need_each_tasks_ordered_feature_list():
    one = [(0, 10), (11, 65535)]
    task = _task("x", 2, 1, ["flow_iat_max"], intervals={"flow_iat_max": one})
    with pytest.raises(ValueError, match="features"):
        bps.generate_table_entries([task])
