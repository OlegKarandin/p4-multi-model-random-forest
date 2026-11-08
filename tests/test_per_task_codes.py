"""Per-task codes (program.PER_TASK_ENCODING), the range-table priority and the
66-byte stage check, on synthetic forests.

One threshold table per feature, as under 'joint'; its action writes one code per
task with a threshold on the feature; each tree keys only its own task's codes.
What is pinned here: the plan and the program text, that training's pricing equals
the replay of the generated program, that real control-plane entries classify every
input exactly as each tree does, and that the arm identity flows through every
place that spells it. The real-compiler checks are not in git (results/ is not)."""
import json
import os
import re

import numpy as np
import pytest
from sklearn.ensemble import RandomForestClassifier

from p4gen import build_p4_script as bps
from p4gen.evaluation import _pool_inputs, multi_model_memory_evaluation
from p4gen.p4_replay import model_breakdown, parse_program, parse_program_text, table_tasks
from p4gen.switch_semantics import per_tree_class_indices, switch_predict, vote_winner
from p4model import lanes
from p4model.errors import CrossbarKeyTooWide
from p4model.packing import StageLoad, _stage_shards, crossbar_stages_needed, stage_load_fits
from p4model.program import (APP_TASK, DDOS_TASK, PER_TASK_ENCODING, SHARED_TASK, TaskSpec,
                             feature_of_per_task_field, per_task_field, range_table_priority)
from p4model.target import (MAX_CODEWORD_LENGTH, TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE,
                            TERNARY_CROSSBAR_STAGE_BYTES)

FEATURES = ["flow_iat_max", "fwd_packet_length_max", "bwd_iat_mean",
            "flow_iat_min", "bwd_packet_length_max"]


def _forest(n_classes, seed, features=FEATURES, n_trees=3, depth=4, leaf=1):
    rng = np.random.RandomState(seed)
    X = rng.randint(0, 2000, size=(600, len(features)))
    y = ((X[:, seed % len(features)] + X[:, (seed + 1) % len(features)] // 3)
         // (2600 // n_classes)).clip(0, n_classes - 1)
    return bps.dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=n_trees, max_depth=depth, min_samples_leaf=leaf,
        random_state=seed).fit(X, y))


def _own(forest, names=FEATURES):
    return bps.get_feature_intervals(forest, names)


def _tasks(app, ddos, names_app=FEATURES, names_ddos=FEATURES):
    return [TaskSpec(DDOS_TASK, 2, list(ddos.estimators_), names_ddos, _own(ddos, names_ddos)),
            TaskSpec(APP_TASK, 3, list(app.estimators_), names_app, _own(app, names_app))]


def _generate(tmp_path, app, ddos, names_app=FEATURES, names_ddos=FEATURES, **kwargs):
    path = bps.generate_P4_code(
        3, 2, app, ddos, feature_intervals_app=_own(app, names_app),
        feature_intervals_ddos=_own(ddos, names_ddos),
        output_dir=str(tmp_path) + os.sep, output_filename="design.p4",
        selected_features_app=names_app, selected_features_ddos=names_ddos,
        encoding=PER_TASK_ENCODING, **kwargs)
    with open(path, encoding="utf-8") as handle:
        return path, handle.read()


# --- the plan ---------------------------------------------------------------

def _iv(*thresholds):
    return bps.get_feature_intervals_from_thresholds([("f", t) for t in thresholds])["f"]


def test_plan_writes_one_code_per_task_with_a_threshold_and_the_table_holds_the_union():
    tables, fields = bps.resolve_per_task_plan([
        (DDOS_TASK, {"b": _iv(5, 9), "a": _iv(3)}),
        (APP_TASK, {"b": _iv(5, 7, 20), "c": _iv(1)})])
    # Sorted by feature, as feature_intervals_from_nodes sorts the merged intervals.
    assert list(tables) == ["a", "b", "c"]
    assert tables["b"][0] == _iv(5, 7, 9, 20)
    # Program order: the lowest-priority task, app, first.
    assert tables["b"][1] == [(APP_TASK, "tk_app_b", 3), (DDOS_TASK, "tk_ddos_b", 2)]
    # A task with no threshold on a feature has no field for it.
    assert tables["a"] == (_iv(3), [(DDOS_TASK, "tk_ddos_a", 1)])
    assert tables["c"] == (_iv(1), [(APP_TASK, "tk_app_c", 1)])
    assert sorted(fields) == ["tk_app_b", "tk_app_c", "tk_ddos_a", "tk_ddos_b"]
    assert fields["tk_ddos_b"] == ("b", _iv(5, 9), {DDOS_TASK})


def test_plan_union_equals_the_joint_intervals_of_the_same_forests():
    app, ddos = _forest(3, 2), _forest(2, 1)
    tables, _ = bps.resolve_per_task_plan([(DDOS_TASK, _own(ddos)), (APP_TASK, _own(app))])
    joint = bps.get_joint_feature_intervals(app, FEATURES, ddos, FEATURES)
    assert {feature: union for feature, (union, _) in tables.items()} == joint
    assert list(tables) == list(joint)


def test_plan_rejects_colliding_field_names():
    with pytest.raises(ValueError, match="produced twice"):
        bps.resolve_per_task_plan([("a", {"b_c": _iv(1)}), ("a_b", {"c": _iv(1)})])


def test_field_helpers_undo_each_other_even_with_underscores_in_the_names():
    assert per_task_field("a_b", "c_d") == "tk_a_b_c_d"
    assert feature_of_per_task_field("code_tk_a_b_c_d", "a_b") == "c_d"
    assert feature_of_per_task_field("tk_a_b_c_d", "a_b") == "c_d"
    assert feature_of_per_task_field("code_tk_a_b_c_d", "a") == "b_c_d"
    assert feature_of_per_task_field("code_c_d", "a") is None


# --- the program ------------------------------------------------------------

def test_program_declares_and_pins_every_code_field(tmp_path):
    app, ddos = _forest(3, 2), _forest(2, 1)
    _, text = _generate(tmp_path, app, ddos)
    own = {APP_TASK: _own(app), DDOS_TASK: _own(ddos)}
    for task, intervals in own.items():
        names = list(intervals)
        widths = [len(intervals[f]) - 1 for f in names]
        lay = lanes.key_layout(widths)
        for feature, width in zip(names, widths):
            field = "code_tk_%s_%s" % (task, feature)
            assert "bit<%d> %s;" % (width, field) in text
            # All three pragmas: dropping @pa_solitary let p4c pack code fields of
            # different tables into one container and made stage results false.
            assert '@pa_solitary("ingress", "ig_md.%s")' % field in text
            assert '@pa_no_overlay("ingress", "ig_md.%s")' % field in text
            assert ('@pa_container_size("ingress", "ig_md.%s", %s)'
                    % (field, ", ".join(map(str, lanes.container_sizes(width, lay))))) in text
    # No single-code field is left over.
    assert not re.search(r"bit<\d+>\s+code_(?!tk_)\w+;", text)


def test_a_range_action_writes_each_tasks_code_and_each_tree_keys_only_its_own(tmp_path):
    app, ddos = _forest(3, 2), _forest(2, 1)
    _, text = _generate(tmp_path, app, ddos)
    own_app, own_ddos = _own(app), _own(ddos)
    shared = sorted(set(own_app) & set(own_ddos))[0]
    params = ("bit<%d> code_app, bit<%d> code_ddos"
              % (len(own_app[shared]) - 1, len(own_ddos[shared]) - 1))
    assert "action set_code_%s (%s) {" % (shared, params) in text
    assert "meta.code_tk_app_%s = code_app;" % shared in text
    assert "meta.code_tk_ddos_%s = code_ddos;" % shared in text
    tables = parse_program_text(text).tables
    for name, keys in tables.items():
        if name.startswith("get_classification_tree_app_"):
            assert keys == ["code_tk_app_" + f for f in sorted(own_app)]
        elif name.startswith("get_classification_tree_ddos_"):
            assert keys == ["code_tk_ddos_" + f for f in sorted(own_ddos)]
    # One range table per feature of the union, keyed on the raw value.
    union = sorted(set(own_app) | set(own_ddos))
    assert [(n, k) for n, k in tables.items() if n.startswith("table_")] == [
        ("table_%d_%s" % (i, f), [f + "_val"]) for i, f in enumerate(union)]


def test_everything_but_the_codes_matches_the_one_code_program(tmp_path):
    """Registers, table order and sizes are the shared design's, so a difference
    between the two encodings on p4c can only come from the code fields."""
    app, ddos = _forest(3, 2), _forest(2, 1)
    _, per_task = _generate(tmp_path, app, ddos)
    joint = bps.get_joint_feature_intervals(app, FEATURES, ddos, FEATURES)
    path = bps.generate_P4_code(
        3, 2, app, ddos, feature_intervals_app=joint, feature_intervals_ddos=joint,
        output_dir=str(tmp_path) + os.sep, output_filename="one.p4",
        selected_features_app=FEATURES, selected_features_ddos=FEATURES)
    with open(path, encoding="utf-8") as handle:
        one_code = handle.read()

    def without_codes(text):
        return [line for line in text.splitlines()
                if not re.search(r"code", line) and "bit<" not in line]
    assert without_codes(per_task) == without_codes(one_code)
    assert (parse_program_text(per_task).sizes == parse_program_text(one_code).sizes)


def test_old_one_code_programs_still_replay(tmp_path):
    app, ddos = _forest(3, 2), _forest(2, 1)
    joint = bps.get_joint_feature_intervals(app, FEATURES, ddos, FEATURES)
    path = bps.generate_P4_code(
        3, 2, app, ddos, feature_intervals_app=joint, feature_intervals_ddos=joint,
        output_dir=str(tmp_path) + os.sep, output_filename="one.p4",
        selected_features_app=FEATURES, selected_features_ddos=FEATURES)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    # A program generated before the range-table priority carries none.
    old = re.sub(r"@placement_priority\(3\)\n\s*", "", text)
    assert "@placement_priority(3)" not in old
    assert model_breakdown(parse_program_text(old)) == model_breakdown(parse_program_text(text))
    usage = multi_model_memory_evaluation(app, ddos, FEATURES, FEATURES, "joint")
    breakdown = model_breakdown(parse_program_text(old))
    assert (breakdown["stage_depth"], breakdown["blocks"]) == (usage.stage_depth, usage.blocks)


# --- the range-table priority (all arms) ------------------------------------

@pytest.mark.parametrize("priorities,expected", [
    ({"ddos": 2, "app": 1}, 3), ({"only": 1}, 2), ({"a": 3, "b": 2, "c": 1}, 4), ({}, 1)])
def test_range_table_priority_is_one_above_the_largest_tree_priority(priorities, expected):
    assert range_table_priority(priorities) == expected


@pytest.mark.parametrize("arm", ["joint", "disjoint", "per_task"])
def test_every_range_table_outranks_every_tree_in_every_arm(tmp_path, arm):
    app, ddos = _forest(3, 2), _forest(2, 1)
    joint = bps.get_joint_feature_intervals(app, FEATURES, ddos, FEATURES)
    intervals = {"joint": (joint, joint)}.get(arm, (_own(app), _own(ddos)))
    path = bps.generate_P4_code(
        3, 2, app, ddos, feature_intervals_app=intervals[0], feature_intervals_ddos=intervals[1],
        output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
        selected_features_app=FEATURES, selected_features_ddos=FEATURES,
        encoding=arm)
    program = parse_program(path)
    ranges = [n for n in program.tables if n.startswith("table_")]
    trees = [n for n in program.tables if n.startswith("get_classification_tree_")]
    assert ranges and trees
    assert {program.priorities[n] for n in ranges} == {3}
    assert {program.priorities[n] for n in trees} == {1, 2}


def test_three_task_range_priority_is_four(tmp_path):
    members = [TaskSpec(name, n, list(f.estimators_), FEATURES, _own(f))
               for name, n, f in (("a", 2, _forest(2, 1)), ("b", 3, _forest(3, 2)),
                                  ("c", 2, _forest(2, 3)))]
    path = bps.generate_p4_program(members, output_dir=str(tmp_path) + os.sep,
                                   output_filename="d.p4")
    program = parse_program(path)
    assert {program.priorities[n] for n in program.tables if n.startswith("table_")} == {4}
    assert {program.priorities[n] for n in program.tables
            if n.startswith("get_classification_tree_")} == {1, 2, 3}


# --- search path equals replay path -------------------------------------------

def _rng_design(seed):
    rng = np.random.RandomState(seed)
    depth_app, depth_ddos = rng.randint(2, 11, size=2)
    trees_app, trees_ddos = rng.randint(1, 8, size=2)
    k = rng.randint(1, len(FEATURES) + 1)
    names = list(rng.choice(FEATURES, size=k, replace=False))
    return (_forest(3, 2 + seed, names, n_trees=trees_app, depth=depth_app),
            _forest(2, 1 + seed, names, n_trees=trees_ddos, depth=depth_ddos), names)


@pytest.mark.parametrize("seed", range(40))
def test_training_prices_exactly_what_the_generated_program_replays(tmp_path, seed):
    app, ddos, names = _rng_design(seed)
    usage = multi_model_memory_evaluation(app, ddos, names, names, PER_TASK_ENCODING)
    path, _ = _generate(tmp_path, app, ddos, names, names)
    breakdown = model_breakdown(parse_program(path))
    assert (breakdown["stage_depth"], breakdown["blocks"]) == (usage.stage_depth, usage.blocks)


def test_forests_with_no_common_feature_label_each_range_table_with_its_task(tmp_path):
    """Trees wait only for the range tables writing their own task's codes, the
    readiness rule 'disjoint' has, on one table per feature: labels must agree
    between training and replay, or the paths would drift."""
    app = _forest(3, 2, FEATURES[:2], depth=3)
    ddos = _forest(2, 1, FEATURES[2:4], depth=3)
    pool = _pool_inputs(app, ddos, FEATURES[:2], FEATURES[2:4], PER_TASK_ENCODING)
    union = sorted(set(_own(app, FEATURES[:2])) | set(_own(ddos, FEATURES[2:4])))
    assert pool["emitted_features"] == union
    path, _ = _generate(tmp_path, app, ddos, FEATURES[:2], FEATURES[2:4])
    tables = parse_program(path).tables
    labels = table_tasks(tables)
    assert pool["range_task"] == [labels["table_%d_%s" % (i, f)] for i, f in enumerate(union)]
    assert set(pool["range_task"]) == {APP_TASK, DDOS_TASK}


def test_a_feature_both_tasks_split_is_shared_and_keys_stay_distinct(tmp_path):
    app, ddos = _forest(3, 2), _forest(2, 1)
    pool = _pool_inputs(app, ddos, FEATURES, FEATURES, PER_TASK_ENCODING)
    both = set(_own(app)) & set(_own(ddos))
    assert both
    for feature, label in zip(pool["emitted_features"], pool["range_task"]):
        assert (label == SHARED_TASK) == (feature in both)
    n_app = len(app.estimators_)
    # Identical widths must not merge two tasks' fields into one crossbar field.
    assert not pool["ternary_fields"][0] & pool["ternary_fields"][n_app]
    # The key bits are the task's own thresholds, not the union's.
    assert pool["ternary_key_bits"][0] == tuple(sorted(
        len(iv) - 1 for iv in _own(app).values()))
    assert pool["codeword_length"] == max(sum(len(iv) - 1 for iv in _own(app).values()),
                                          sum(len(iv) - 1 for iv in _own(ddos).values()))


def test_per_task_keys_never_cost_more_blocks_on_a_lone_table_than_the_union():
    app, ddos = _forest(3, 2), _forest(2, 1)
    one_code = _pool_inputs(app, ddos, FEATURES, FEATURES, "joint")
    per_task = _pool_inputs(app, ddos, FEATURES, FEATURES, PER_TASK_ENCODING)
    assert one_code["range_table_specs"] == per_task["range_table_specs"]
    assert one_code["ternary_entries"] == per_task["ternary_entries"]
    for one, task in zip(one_code["ternary_table_specs"], per_task["ternary_table_specs"]):
        assert task[1] <= one[1]    # crossbar bytes


# --- control-plane entries ----------------------------------------------------

def _switch_tree_classes(entries, row):
    """Each tree table's class for one input, by interpreting the entries as the
    switch does: a range table's action writes code_tk_<task>_<f> from its
    parameter code_<task>, then every tree table matches ternary-wise. Exactly one
    entry of each table may match."""
    codes, by_table = {}, {}
    for entry in entries:
        by_table.setdefault(entry["table_name"], []).append(entry)
    for table, rows in by_table.items():
        if not table.startswith("table_"):
            continue
        hits = [e for e in rows for field, span in e["key_fields"].items()
                if int(span["start"]) <= row[field[:-len("_val")]] <= int(span["end"])]
        assert len(hits) == 1, (table, row)
        feature = hits[0]["action"][len("set_code_"):]
        for param, value in hits[0]["action_params"].items():
            assert param.startswith("code_")
            codes["code_" + per_task_field(param[len("code_"):], feature)] = int(value)
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


def _votes(classes, task, n_classes):
    trees = sorted((int(t.rsplit("_", 1)[1]), c) for t, c in classes.items()
                   if t.startswith("get_classification_tree_" + task + "_"))
    return vote_winner([c for _, c in trees], n_classes)


@pytest.mark.parametrize("discount", [False, True])
@pytest.mark.parametrize("seed", range(6))
def test_entries_classify_every_input_exactly_as_each_tree_and_the_forest(tmp_path, seed, discount):
    app, ddos, names = _rng_design(seed)
    tasks = _tasks(app, ddos, names, names)
    path = bps.generate_p4_program(
        tasks, output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
        use_default_action_discount=discount, entries_filename="entries.json",
        encoding=PER_TASK_ENCODING)
    with open(os.path.join(str(tmp_path), "entries.json"), encoding="utf-8") as handle:
        entries = json.load(handle)
    assert entries == bps.generate_table_entries(
        tasks, use_default_action_discount=discount, encoding=PER_TASK_ENCODING)
    # Every table the entries address exists, with the key fields it declares.
    declared = parse_program(path).tables
    for entry in entries:
        assert entry["table_name"] in declared
        if not entry["is_default_action"]:
            assert sorted(entry["key_fields"]) == sorted(declared[entry["table_name"]])
    rng = np.random.RandomState(seed)
    # Every boundary of every feature and its neighbours, then random values.
    X = rng.randint(0, 65536, size=(150, len(names)))
    edges = np.vstack([X[:1]] * 0 + [np.full((1, len(names)), v)
                                     for v in (0, 1, 500, 1000, 1999, 2000, 65535)])
    X = np.vstack([edges, rng.randint(0, 2100, size=(150, len(names))), X])
    trees_app = per_tree_class_indices(app, X)
    trees_ddos = per_tree_class_indices(ddos, X)
    predicted_app, predicted_ddos = switch_predict(app, X), switch_predict(ddos, X)
    for s, values in enumerate(X):
        got = _switch_tree_classes(entries, dict(zip(names, values)))
        expected = {"get_classification_tree_app_%d" % i: int(c)
                    for i, c in enumerate(trees_app[:, s])}
        expected.update({"get_classification_tree_ddos_%d" % i: int(c)
                         for i, c in enumerate(trees_ddos[:, s])})
        assert got == expected
        assert app.classes_[_votes(got, APP_TASK, 3)] == predicted_app[s]
        assert ddos.classes_[_votes(got, DDOS_TASK, 2)] == predicted_ddos[s]


def test_each_code_written_is_the_tasks_own_thermometer():
    entries = bps._range_table_entries(
        "t", "a", "f_val", _iv(5, 7, 9),
        [("code_app", _iv(5, 9)), ("code_ddos", _iv(7))])
    by_span = {(e["key_fields"]["f_val"]["start"], e["key_fields"]["f_val"]["end"]):
               e["action_params"] for e in entries}
    assert by_span == {("0", "5"): {"code_app": "3", "code_ddos": "1"},
                       ("6", "7"): {"code_app": "1", "code_ddos": "1"},
                       ("8", "9"): {"code_app": "1", "code_ddos": "0"},
                       ("10", "65535"): {"code_app": "0", "code_ddos": "0"}}


def test_a_table_interval_that_does_not_refine_a_tasks_own_is_refused():
    with pytest.raises(ValueError, match="refine"):
        bps._range_table_entries("t", "a", "f_val", _iv(5), [("code_app", _iv(3))])


# --- N tasks -------------------------------------------------------------------

def _three_tasks():
    forests = {"a": _forest(2, 1), "b": _forest(3, 2), "c": _forest(4, 5, FEATURES[:3], n_trees=1)}
    names = {"a": FEATURES, "b": FEATURES, "c": FEATURES[:3]}
    classes = {"a": 2, "b": 3, "c": 4}
    return [TaskSpec(t, classes[t], list(forests[t].estimators_), names[t], _own(forests[t], names[t]))
            for t in ("a", "b", "c")], forests, names


def test_three_task_program_replays_and_labels_a_subset_of_tasks_with_a_frozenset(tmp_path):
    tasks, forests, names = _three_tasks()
    path = bps.generate_p4_program(tasks, output_dir=str(tmp_path) + os.sep, output_filename="d.p4",
                                   encoding=PER_TASK_ENCODING)
    program = parse_program(path)
    assert {t: sorted(k) for t, k in program.tables.items() if t == "get_classification_tree_c_0"} == {
        "get_classification_tree_c_0": sorted("code_tk_c_" + f for f in _own(forests["c"], names["c"]))}
    labels = table_tasks(program.tables)
    own = {t.name: set(t.intervals) for t in tasks}
    for i, f in enumerate(sorted(own["a"] | own["b"] | own["c"])):
        readers = {t for t in own if f in own[t]}
        expected = SHARED_TASK if len(readers) == 3 else (
            frozenset(readers) if len(readers) == 2 else next(iter(readers)))
        assert labels["table_%d_%s" % (i, f)] == expected
    breakdown = model_breakdown(program)
    assert breakdown["stage_depth"] > 0 and breakdown["blocks"] > 0


def test_three_task_entries_classify_exactly_as_each_tree(tmp_path):
    tasks, forests, names = _three_tasks()
    entries = bps.generate_table_entries(tasks, encoding=PER_TASK_ENCODING)
    X = np.random.RandomState(3).randint(0, 2100, size=(120, len(FEATURES)))
    expected = {}
    for task in tasks:
        cols = [FEATURES.index(f) for f in names[task.name]]
        for i, classes in enumerate(per_tree_class_indices(forests[task.name], X[:, cols])):
            expected["get_classification_tree_%s_%d" % (task.name, i)] = classes
    for s, values in enumerate(X):
        got = _switch_tree_classes(entries, dict(zip(FEATURES, values)))
        assert got == {table: int(classes[s]) for table, classes in expected.items()}


# --- the 66-byte stage check --------------------------------------------------

def _load(index, fields, blocks=(1,), tables=1):
    return StageLoad(index=index, blocks=tuple(blocks), fields=frozenset(fields), tables=tables)


def test_the_stage_limit_is_66_while_the_single_key_limit_stays_64():
    assert TERNARY_CROSSBAR_STAGE_BYTES == 66
    assert TERNARY_CROSSBAR_MAX_BYTES_PER_STAGE == 64
    assert MAX_CODEWORD_LENGTH == 512
    assert stage_load_fits([_load(0, [("a", 33)]), _load(0, [("b", 33)])])
    assert not stage_load_fits([_load(0, [("a", 33)]), _load(0, [("b", 34)])])
    with pytest.raises(CrossbarKeyTooWide):
        _stage_shards(1, 65)
    with pytest.raises(CrossbarKeyTooWide):
        lanes.table_blocks((8,) * 65)


def _two_key_plan(bytes_a, bytes_b):
    """Two tables with single-field keys of whole bytes, ddos-style first."""
    bits_a, bits_b = (8 * bytes_a,), (8 * bytes_b,)
    specs = [(lanes.table_blocks(bits_a), bytes_a), (lanes.table_blocks(bits_b), bytes_b)]
    fields = [frozenset({("a", bytes_a)}), frozenset({("b", bytes_b)})]
    return crossbar_stages_needed(specs, readiness_levels=[0, 0], key_fields=fields,
                                  key_field_bits=[bits_a, bits_b],
                                  placement_priority=[1, 2])


def test_a_pair_past_66_bytes_always_splits_and_a_pair_within_is_up_to_the_lanes():
    assert _two_key_plan(33, 34).occupied == 2
    # Below the limit the byte check says nothing: this pair fits one stage.
    assert _two_key_plan(16, 16).occupied == 1
    # 65-66 bytes: the check no longer refuses, so the plan is the lane fit's, which
    # either prices both keys or refuses the second one (a real rise).
    a, b = [8 * 33], [8 * 33]
    free, mids = lanes.first_key_occupancy(lanes.key_bytes(a), lanes.table_blocks(a))
    fits = lanes.price_with_supply(lanes.key_bytes(b), free, mids) is not None
    assert _two_key_plan(33, 33).occupied == (1 if fits else 2)


# --- the arm ------------------------------------------------------------------

def test_arm_identity_flows_through_config_row_id_jobs_and_the_loader():
    from src.main import select_arm_slugs
    from src.reporting.campaign_data import _expected_arm_slug
    from src.training.campaign_run import row_id, split_csv_name
    from src.training.campaign_runner import Job, plan_jobs
    from src.training.config import TrainConfig, arm_encoding
    (arm, cfg), = select_arm_slugs(["joint-task"])
    assert arm == "joint-task" and arm_encoding(arm) == PER_TASK_ENCODING
    assert cfg.arm_slug(PER_TASK_ENCODING) == "joint-task"
    assert not cfg.alignment_enabled
    joint_off = dict(select_arm_slugs(["joint-off"]))["joint"]
    assert cfg == joint_off     # the same config: only the encoding differs
    job = Job(arm=arm, cfg=cfg, M=35, split=7)
    assert job.slug == "joint-task" and job.key == "joint-task_M035_s07"
    assert os.path.basename(job.csv_path("run")) == split_csv_name(
        cfg.n_trees, cfg.max_depth, 35, "joint-task", 7)
    assert row_id("joint-task", 35, 7, 9) == "joint-task_M035_s07_k09"
    assert [j.key for j in plan_jobs([(arm, cfg)], [15, 75], [0], "no_such_dir")] == [
        "joint-task_M015_s00", "joint-task_M075_s00"]
    assert _expected_arm_slug("joint-task", False, "") == "joint-task"
    with pytest.raises(ValueError):
        arm_encoding("joint_task")


def test_the_primary_preset_is_unchanged():
    from src.main import select_arms
    from src.training.config import arm_encoding
    assert [cfg.arm_slug(arm_encoding(arm)) for arm, cfg in select_arms("primary")] == [
        "independent", "joint-off", "joint"]


def test_row_artifacts_of_the_arm_replay_to_the_trainings_numbers(tmp_path):
    from src.training import row_artifacts as ra
    app, ddos = _forest(3, 2), _forest(2, 1)
    usage = multi_model_memory_evaluation(app, ddos, FEATURES, FEATURES, PER_TASK_ENCODING)
    ctx = ra.RowContext(str(tmp_path), "joint-task", 35, "abc")
    model = ra.write_row_artifacts(ctx, "joint-task_M035_s00_k05", app, ddos, FEATURES, FEATURES,
                                   PER_TASK_ENCODING, usage)
    assert model["model_paths_differ"] is False
    assert model["generator_error"] is None
    assert (tmp_path / "designs" / "joint-task_M035_s00_k05.p4").is_file()
    assert "code_tk_" in (tmp_path / "designs" / "joint-task_M035_s00_k05.p4").read_text()
