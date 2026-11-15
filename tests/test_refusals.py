"""Decision 8.2 / 9.2 (Task 12): the replay checks the program shape it assumes
(program_checks.check_program) and applies every refusal, one NAMED constraint per
refusal source (p4model.refusals.CONSTRAINTS), so training, --rescore,
--model-archive and outside users share one implementation."""
import collections
import json
import os
import re

import numpy as np
import pytest

from p4gen import build_p4_script as bps
from p4gen.p4_replay import (model_breakdown, parse_program_text, price_program,
                             tree_range_reads)
from p4gen.program_checks import check_program
from p4model import errors, lanes, packed, phv, refusals, target
from p4model.errors import ProgramShapeRefused
from p4model.program import ENCODINGS, JOINT_ENCODING, PER_TREE_ENCODING
from p4model.registers import register_stage_schedule
from p4model.target import PHV_INGRESS_CONTAINERS
from src.training import early_stopping
from src.training.row_artifacts import replayed_fields
from src.training.train_model import design_refusal
from tests.test_per_task_codes import _own, _rng_design

# The pins the generator writes, one per line: @pa_<pin>("ingress", "ig_md.<field>"...)
# or @placement_priority(N) above `table <name> {`.
_PIN_LINE = re.compile(r'^\s*@(pa_\w+)\("ingress",\s*"ig_md\.(\w+)".*$|'
                       r'^\s*@(placement_priority)\(\d+\)\s*$')


def _design_text(tmp_path, app, ddos, names, encoding, name="d.p4"):
    """The program row_artifacts._generate writes for the design (joint: the joint
    intervals; every other encoding: each model's own)."""
    if encoding == JOINT_ENCODING:
        joint = bps.get_joint_feature_intervals(app, names, ddos, names)
        intervals = (joint, joint)
    else:
        intervals = (_own(app, names), _own(ddos, names))
    path = bps.generate_P4_code(3, 2, app, ddos, feature_intervals_app=intervals[0],
                                feature_intervals_ddos=intervals[1],
                                output_dir=str(tmp_path) + os.sep, output_filename=name,
                                selected_features_app=names, selected_features_ddos=names,
                                encoding=encoding)
    with open(path, encoding="utf-8") as handle:
        return handle.read()


def _texts(tmp_path, encoding, count, min_features=1):
    """(seed, text) of the first `count` _rng_design seeds the generator builds (a
    stump is refused, NoSplitTree) with at least min_features features."""
    found = []
    for seed in range(60):
        app, ddos, names = _rng_design(seed)
        if len(names) < min_features:
            continue
        try:
            found.append((seed, _design_text(tmp_path, app, ddos, names, encoding)))
        except errors.NoSplitTree:
            continue
        if len(found) == count:
            return found
    raise AssertionError("too few buildable designs")


# --- check_program on what the generator writes ------------------------------------

@pytest.mark.parametrize("encoding", ENCODINGS)
def test_every_generated_program_passes_check_program(tmp_path, encoding):
    for seed, text in _texts(tmp_path, encoding, 6):
        program = parse_program_text(text)
        check_program(program)
        breakdown = model_breakdown(program)
        assert breakdown["refusal"] in (None, "stages_over"), (encoding, seed)
        assert breakdown["stage_depth"] is not None


def _identity_sample():
    """The newest identity sample of the model8 plan (impl/identity/t<N>/), which
    lives outside git: in this checkout's results/ or any parent directory's."""
    relative = os.path.join("results", "encoding_evidence_2026_11", "model8", "impl", "identity")
    here = os.path.dirname(os.path.abspath(__file__))
    while True:
        folder = os.path.join(here, relative)
        if os.path.isdir(folder):
            # t<N> or t<N><letter> (t17b); t17b is newer than t17, older than t18
            tags = [name for name in os.listdir(folder)
                    if re.fullmatch(r"t\d+[a-z]?", name) and os.path.isdir(os.path.join(folder, name))]
            if tags:
                return os.path.join(folder, max(
                    tags, key=lambda t: (int(re.match(r"t(\d+)", t).group(1)), t)))
            return None
        parent = os.path.dirname(here)
        if parent == here:
            return None
        here = parent


def test_every_program_in_the_identity_sample_passes_check_program():
    sample = _identity_sample()
    if sample is None:
        pytest.skip("the identity sample (results/, not in git) is not in this checkout "
                    "or any parent directory")
    names = sorted(name for name in os.listdir(sample) if name.endswith(".p4"))
    assert len(names) > 100
    for name in names:
        with open(os.path.join(sample, name), encoding="utf-8") as handle:
            check_program(parse_program_text(handle.read()))


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_each_pin_the_generator_emits_is_required(tmp_path, encoding):
    """The program with any one pin line removed is refused missing_pin:<pin>:<field>."""
    (_, text), = _texts(tmp_path, encoding, 1, min_features=3)
    lines = text.split("\n")
    seen = collections.Counter()
    for index, line in enumerate(lines):
        match = _PIN_LINE.match(line)
        if not match:
            continue
        if match.group(3):
            pin = match.group(3)
            field = re.match(r"\s*table\s+(\w+)", lines[index + 1]).group(1)
        else:
            pin, field = match.group(1), match.group(2)
        cut = parse_program_text("\n".join(lines[:index] + lines[index + 1:]))
        with pytest.raises(ProgramShapeRefused) as refused:
            check_program(cut)
        assert refused.value.reason == "missing_pin:%s:%s" % (pin, field)
        breakdown = model_breakdown(cut)
        assert (breakdown["refusal"], breakdown["refusal_reason"]) == (
            "program_shape", "missing_pin:%s:%s" % (pin, field))
        assert breakdown["stage_depth"] is None and breakdown["blocks"] is None
        seen[pin] += 1
    expected = {"pa_no_overlay", "pa_solitary", "pa_container_size", "placement_priority"}
    # pa_solitary is on the class_tree_* fields only since gate G3 (3.5), plus
    # now_pseudo_us in a program with a timestamp since review A1b
    assert seen["pa_solitary"] == sum(1 for line in lines if ("class_tree_" in line
                                                              or "now_pseudo_us" in line)
                                      and "@pa_solitary" in line)
    assert ("calc_timestamp();" in text) == ('"ig_md.now_pseudo_us"' in text)
    assert set(seen) == expected, seen


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_each_pin_dropped_at_g3_is_refused_when_present(tmp_path, encoding):
    """A text carrying a pin gate G3 dropped (3.1 fwd no-overlay, 3.3 class_tree_*
    no-overlay, 3.5 code_* solitary) is an older generator's program: refused
    unexpected_shape:pin:<pin>:<field>, although the pin changes no price."""
    (_, text), = _texts(tmp_path, encoding, 1, min_features=3)
    program = parse_program_text(text)
    first = text.index("@pa_")
    dropped = [("pa_no_overlay", "fwd"), ("pa_no_overlay", sorted(program.class_fields)[0])]
    codes = sorted(name for name in program.declared if name.startswith("code_"))
    dropped.append(("pa_solitary", codes[0]))
    for pin, field in dropped:
        added = text[:first] + '@%s("ingress", "ig_md.%s")\n' % (pin, field) + text[first:]
        with pytest.raises(ProgramShapeRefused) as refused:
            check_program(parse_program_text(added))
        assert refused.value.reason == "unexpected_shape:pin:%s:%s" % (pin, field)
        assert price_program(parse_program_text(added)) == price_program(program)


def test_a_pre_task_2_program_is_refused_as_a_gated_register_block(tmp_path):
    # Before decisions 1a / 7.1 option B (2026-11-14) direction-owned registers were
    # executed inside `if (meta.fwd == ...)` blocks: wrap one call site that way.
    (_, text), = _texts(tmp_path, JOINT_ENCODING, 1, min_features=3)
    calls = re.findall(r"^\t+meta\.\w+ = (?!flow_orientation)\w+_action\.execute\(meta\.flow_hash\);$",
                       text, re.M)
    assert calls
    gated = text.replace(calls[-1], "\t\t\tif (meta.fwd == 1) {\n%s\n\t\t\t}" % calls[-1], 1)
    for program in (parse_program_text(gated),):
        with pytest.raises(ProgramShapeRefused, match="^gated_register_block$"):
            check_program(program)
        assert model_breakdown(program)["refusal_reason"] == "gated_register_block"


@pytest.mark.parametrize("edit,reason", [
    (lambda t: re.sub(r"(\n\s*@placement_priority\(\d+\)\s*\n\s*table get_classification_tree_app_0)",
                      r"\n    @stage(9)\1", t, count=1), "stage_pin"),
    (lambda t: t.replace("\t\t\tcalc_flow_hash();\n", "", 1), "unexpected_shape:prologue"),
    (lambda t: re.sub(r"\t\t\tvote_ddos\.apply\(\);\n", "", t, count=1),
     "unexpected_shape:vote_tables"),
    (lambda t: re.sub(r"(\t\t\ttable_0_\w+\.apply\(\);\n)((?:.|\n)*?)(\t\t\tvote_app\.apply\(\);\n)",
                      r"\2\1\3", t, count=1), "unexpected_shape:apply_order"),
    (lambda t: re.sub(r"bit<16> (\w+_val);", r"bit<32> \1;", t, count=1),
     "unexpected_shape:val_width:"),
    (lambda t: re.sub(r"@placement_priority\(3\)", "@placement_priority(7)", t, count=1),
     "unexpected_shape:range_priority:table_"),       # range tables not one priority
    # a code field's split, same total, not lanes.container_sizes of its live width
    (lambda t: re.sub(r'(@pa_container_size\("ingress", "ig_md\.code_\w+"), (16|32)\)',
                      lambda m: "%s, %d, %d)" % (m.group(1), int(m.group(2)) // 2,
                                                 int(m.group(2)) // 2), t, count=1),
     "unexpected_shape:code_container:code_"),
    # the pre-17b template: the forwarding assignments last, after the votes
    (lambda t: re.sub(r"(\n[ \t]*ig_tm_md\.ucast_egress_port = ig_intr_md\.ingress_port;\n"
                      r"[ \t]*ig_tm_md\.bypass_egress = 1w1;\n)((?:.|\n)*?)(\n    \}\n\}\n\nPipeline)",
                      r"\2\1\3", t, count=1), "unexpected_shape:egress_assignment"),
    # a second copy of one of them after the votes
    (lambda t: re.sub(r"(\t\t\tvote_ddos\.apply\(\);\n)", r"\1\t\t\tig_tm_md.bypass_egress = 1w1;\n",
                      t, count=1), "unexpected_shape:egress_assignment"),
    # the pre-A1 flow hash, one part at a time (Task 31 review Important 1)
    (lambda t: t.replace("Hash<bit<12>>(HashAlgorithm_t.CRC32)", "Hash<bit<32>>(HashAlgorithm_t.CRC32)", 1),
     "unexpected_shape:flow_hash"),
    (lambda t: re.sub(r'\t@symmetric\("hdr\.tcp\.src_port", "hdr\.tcp\.dst_port"\)\n', "", t, count=1),
     "unexpected_shape:flow_hash"),
    (lambda t: t.replace("\t\t\thdr.ipv4.src_addr,\n\t\t\thdr.ipv4.dst_addr,\n",
                         "\t\t\thdr.ipv4.src_addr ^ hdr.ipv4.dst_addr,\n", 1), "unexpected_shape:flow_hash"),
    (lambda t: re.sub(r"(\t\t\thdr\.tcp\.dst_port\n\t\t\}\))", r"\1 & 0xFFF", t, count=1),
     "unexpected_shape:flow_hash"),
    # a sliced key without its bf_rt @name (pre-B1), or with another one
    (lambda t: re.sub(r' @name\("[^"]*"\);', ";", t, count=1), "unexpected_shape:key_name:"),
    (lambda t: re.sub(r' @name\("([^"]*)"\);', r' @name("x\1");', t, count=1),
     "unexpected_shape:key_name:"),
])
def test_a_program_of_another_shape_is_refused_with_its_reason(tmp_path, edit, reason):
    (_, text), = _texts(tmp_path, JOINT_ENCODING, 1, min_features=3)
    changed = edit(text)
    assert changed != text
    with pytest.raises(ProgramShapeRefused) as refused:
        check_program(parse_program_text(changed))
    assert refused.value.reason.startswith(reason), refused.value.reason


PRE_A1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "pre_a1")


@pytest.mark.parametrize("row_id", ["joint-off_M075_s19_k07", "joint-task_M075_s13_k04"])
def test_the_pre_a1_fixture_texts_are_refused_for_their_flow_hash(row_id):
    """Task 31 review Important 1: two texts of the generator before A1
    (tests/fixtures/p4c_base_17b at 7429a09^, copied to tests/fixtures/pre_a1) that
    the replay used to accept and price 2 stages low (model 8, p4c 10): XOR inputs,
    Hash<bit<32>>, `& 0xFFF`. They are refused now; the current texts of the same
    designs pass."""
    with open(os.path.join(PRE_A1, row_id + ".p4"), encoding="utf-8") as handle:
        old = parse_program_text(handle.read())
    assert old.flow_hash == (32, (), "&^")
    with pytest.raises(ProgramShapeRefused, match="^unexpected_shape:flow_hash$"):
        check_program(old)
    assert model_breakdown(old)["refusal_reason"] == "unexpected_shape:flow_hash"
    current = os.path.join(os.path.dirname(PRE_A1), "p4c_base_17b", row_id + ".p4")
    with open(current, encoding="utf-8") as handle:
        check_program(parse_program_text(handle.read()))


@pytest.mark.parametrize("treeless", ["app", "ddos"])
def test_a_program_with_a_tree_less_task_passes(tmp_path, treeless):
    # The generator numbers priorities over every task (ddos 2, app 1, range 3),
    # also when one task has no trees: the check reads the relations the model uses
    # (one priority per task with trees, distinct; one range priority above them; a
    # vote table exactly per task with trees), not those numbers.
    from tests.test_per_task_codes import FEATURES, _forest
    app, ddos = _forest(3, 2), _forest(2, 1)
    path = bps.generate_P4_code(
        3, 2, None if treeless == "app" else app, None if treeless == "ddos" else ddos,
        feature_intervals_app=_own(app, FEATURES), feature_intervals_ddos=_own(ddos, FEATURES),
        output_dir=str(tmp_path) + os.sep, output_filename="treeless.p4",
        selected_features_app=FEATURES, selected_features_ddos=FEATURES)
    with open(path, encoding="utf-8") as handle:
        text = handle.read()
    assert "get_classification_tree_%s_" % treeless not in text
    program = parse_program_text(text)
    check_program(program)
    breakdown = model_breakdown(program)
    assert breakdown["refusal"] is None and breakdown["stage_depth"] is not None
    # still refused: the range tables' priority not above the trees'
    tree_priority = max(program.priorities[name] for name in program.tables
                        if name.startswith("get_classification_tree_"))
    lowered = re.sub(r"@placement_priority\(3\)(\s*\n\s*table table_)",
                     r"@placement_priority(%d)\1" % tree_priority, text)
    with pytest.raises(ProgramShapeRefused, match="^unexpected_shape:range_priority:"):
        check_program(parse_program_text(lowered))


def test_two_tasks_with_one_tree_priority_are_refused(tmp_path):
    (_, text), = _texts(tmp_path, JOINT_ENCODING, 1, min_features=3)
    same = re.sub(r"@placement_priority\(2\)(\s*\n\s*table get_classification_tree_ddos_)",
                  r"@placement_priority(1)\1", text)
    assert same != text
    with pytest.raises(ProgramShapeRefused, match="^unexpected_shape:tree_priority:"):
        check_program(parse_program_text(same))


def test_model_refused_verdicts_are_verdicts():
    from src.verify.verdicts import MODEL_REFUSED_PREFIX, VERDICTS, is_verdict
    assert all(is_verdict(v) for v in VERDICTS)
    assert is_verdict(MODEL_REFUSED_PREFIX + "stage_pin")
    assert not is_verdict("MODEL_REFUSED") and not is_verdict(None)
    assert MODEL_REFUSED_PREFIX + "stage_pin" not in VERDICTS


def test_feasibility_frontier_counts_every_constraint():
    """scripts/feasibility_frontier.py's F6 diagnostic over refusals.CONSTRAINTS (it
    read early_stopping's old three attribute names before decision 9.2)."""
    import optuna
    from types import SimpleNamespace
    from scripts import feasibility_frontier as ff
    complete = optuna.trial.TrialState.COMPLETE

    def trial(constraint=None, magnitude=0, state=complete):
        return SimpleNamespace(state=state,
                               user_attrs=refusals.violations(constraint, magnitude))
    # "key_bytes_over" is the pre-Task-32 name, read as key_too_wide.
    trials = [trial(), trial("key_bytes_over", 3), trial("blocks_over", 5),
              trial("blocks_over", 1), trial("program_shape", 1),
              SimpleNamespace(state=complete, user_attrs={}),          # never set: infeasible
              trial("stages_over", 2, state=optuna.trial.TrialState.FAIL)]
    assert ff.trial_violation_type(refusals.violations("phv_containers_short", 2)) == (
        "phv_containers_short")
    assert ff.trial_violation_type({}) is None
    summary = ff.summarize_trials(trials)
    assert summary["n_trials_run"] == 7 and summary["n_feasible"] == 1
    assert summary["any_feasible"] and summary["dominant_violation_type"] == "blocks_over"
    assert {name: summary["n_" + name] for name in refusals.CONSTRAINTS} == dict(
        refusals.violations(None), key_too_wide=1, blocks_over=2, program_shape=1)


# --- named constraints (9.2) ----------------------------------------------------------

def test_every_refusal_exception_carries_its_constraint_and_magnitude():
    cases = [
        (errors.CrossbarKeyTooWide("", target.MAX_TERNARY_KEY_BITS + 7), "key_too_wide", 7),
        (errors.CrossbarKeyTooWide("", 300), "key_too_wide", 1),    # no crossbar fit
        (errors.PhvContainersExhausted("", 3), "phv_containers_short", 3),
        (errors.ActionDataTooWide("", target.ACTION_DATA_BUS_BYTES + 2), "bus_bytes_over", 2),
        (errors.ActionDataTooWide("", target.ACTION_DATA_BUS_BYTES), "bus_bytes_over", 1),
        (errors.RangeEntriesDoNotFit("", 12), "range_rows_over", 12),
        (errors.NoSplitTree("", "app_0"), "no_split_tree", 1),
        (errors.PlacementFailed(""), "placement_failed", 1),
        (ProgramShapeRefused("stage_pin"), "program_shape", 1),
    ]
    for exc, constraint, magnitude in cases:
        assert isinstance(exc, refusals.REFUSAL_ERRORS)
        assert refusals.of(exc) == (constraint, magnitude)
        assert constraint in refusals.CONSTRAINTS
    assert refusals.reason_of(ProgramShapeRefused("stage_pin")) == "stage_pin"
    assert refusals.stages_over(target.TOFINO_PIPELINE_STAGES + 2) == 2
    assert refusals.stages_over(target.TOFINO_PIPELINE_STAGES) == 0
    assert refusals.blocks_over(40, 35) == 5 and refusals.blocks_over(40, float("inf")) == 0
    assert refusals.blocks_over(40, None) == 0


def test_constraint_values_follow_the_constraint_order():
    class Trial:
        user_attrs = {name: float(i) for i, name in enumerate(refusals.CONSTRAINTS)}
    assert early_stopping.constraint_values(Trial()) == [
        float(i) for i in range(len(refusals.CONSTRAINTS))]
    assert list(refusals.violations("blocks_over", 3)) == list(refusals.CONSTRAINTS)


def test_an_older_runs_constraint_name_is_read_as_the_current_one():
    """Task 32 renamed key_bytes_over to key_too_wide; a trial recorded under the
    old name still reads as refused, through refusals.LEGACY_CONSTRAINTS."""
    assert refusals.canonical("key_bytes_over") == "key_too_wide"
    assert refusals.violations("key_bytes_over", 2) == refusals.violations("key_too_wide", 2)

    class Old:
        user_attrs = {("key_bytes_over" if name == "key_too_wide" else name): 0.0
                      for name in refusals.CONSTRAINTS}
        user_attrs["key_bytes_over"] = 4.0
    values = early_stopping.constraint_values(Old())
    assert values[refusals.CONSTRAINTS.index("key_too_wide")] == 4.0
    assert sum(values) == 4.0


@pytest.fixture
def _fresh_key_prices():
    """lanes._table_blocks caches prices; a test that moves a key limit must not
    see prices cached under the real one."""
    lanes._table_blocks.cache_clear()
    yield
    lanes._table_blocks.cache_clear()


def _flat_forest(n_features, n_classes):
    """A forest whose trees are stumps (a constant input leaves nothing to split)."""
    from sklearn.ensemble import RandomForestClassifier
    X = np.zeros((60, n_features), dtype=int)
    y = np.arange(60) % n_classes
    return bps.dt_thresholds_float_to_int(RandomForestClassifier(
        n_estimators=2, max_depth=3, random_state=0).fit(X, y))


@pytest.mark.parametrize("encoding", ENCODINGS)
def test_training_and_replay_give_the_same_refusal_name(tmp_path, monkeypatch, encoding,
                                                        _fresh_key_prices):
    """20 designs per encoding that training refuses (design_refusal, the objective's
    rule): the replay of each design's program (model_breakdown) -- or, where the
    generator itself refuses to write it, the generator -- names the same constraint.
    Limits are moved to make refusals: the key limit to 1 live bit, the 16-bit PHV budget
    to 6 containers (not per-tree: its layout refuses at generation), the budget M
    to 12 blocks, the action data bus to 1 byte (per-tree only: the program is written
    under the real bus, then training and the replay meet the moved one); and an app
    forest of stumps (no_split_tree, refused by the generator too)."""
    regimes = (["key", "budget", "no_split"]
               + (["bus"] if encoding == PER_TREE_ENCODING else ["phv"]))
    seen = collections.Counter()
    for seed in range(300):
        if sum(seen.values()) >= 20 and len({r for r, _ in seen}) == len(regimes):
            break
        regime = regimes[seed % len(regimes)]
        max_blocks = 12 if regime == "budget" else float("inf")
        app, ddos, names = _rng_design(seed)
        if regime == "no_split":
            app = _flat_forest(len(names), 3)
        text = replayed = None
        if regime == "bus":
            try:
                text = _design_text(tmp_path, app, ddos, names, encoding)
            except refusals.REFUSAL_ERRORS:
                continue
        with monkeypatch.context() as patch:
            if regime == "key":
                patch.setattr(target, "MAX_TERNARY_KEY_BITS", 1)
            elif regime == "phv":
                tight = dict(PHV_INGRESS_CONTAINERS)
                tight[16] = 6
                patch.setattr(phv, "PHV_INGRESS_CONTAINERS", tight)
            elif regime == "bus":
                patch.setattr(packed, "ACTION_DATA_BUS_BYTES", 1)
            lanes._table_blocks.cache_clear()
            _, trained, _ = design_refusal(app, ddos, names, names, encoding, max_blocks)
            if trained is None:
                continue
            if text is None:
                try:
                    text = _design_text(tmp_path, app, ddos, names, encoding)
                except refusals.REFUSAL_ERRORS as exc:
                    replayed = exc.constraint
            if replayed is None:
                replayed = model_breakdown(parse_program_text(text),
                                           max_blocks=max_blocks)["refusal"]
        lanes._table_blocks.cache_clear()
        assert replayed == trained, (encoding, seed, regime)
        seen[(regime, trained)] += 1
    assert sum(seen.values()) >= 20, seen
    assert {regime for regime, _ in seen} == set(regimes), seen
    assert ("no_split", "no_split_tree") in seen, seen
    if encoding == PER_TREE_ENCODING:
        assert ("bus", "bus_bytes_over") in seen, seen


@pytest.mark.parametrize("encoding", [e for e in ENCODINGS if e != PER_TREE_ENCODING])
def test_a_key_with_no_allocation_is_refused_before_the_phv_count_on_both_paths(
        tmp_path, monkeypatch, encoding, _fresh_key_prices):
    """Interim review minor 1: a design with BOTH a key the crossbar cannot allocate
    (within MAX_TERNARY_KEY_BITS) and a PHV shortage is refused key_too_wide by training
    (per tree, lanes.table_blocks) and by the replay (check_trees runs the allocation
    before the PHV count). The program is written first, under the real limits."""
    from p4model import packing, xbar
    for seed in range(40):
        app, ddos, names = _rng_design(seed)
        try:
            text = _design_text(tmp_path, app, ddos, names, encoding)
        except refusals.REFUSAL_ERRORS:
            continue
        break
    tight = dict(PHV_INGRESS_CONTAINERS)
    tight[16] = 0
    tight[8] = 0
    monkeypatch.setattr(phv, "PHV_INGRESS_CONTAINERS", tight)
    monkeypatch.setattr(xbar, "alone_blocks", lambda key: None)
    lanes._table_blocks.cache_clear()
    packing._alone_blocks.cache_clear()
    try:
        _, trained, _ = design_refusal(app, ddos, names, names, encoding, float("inf"))
        replayed = model_breakdown(parse_program_text(text))["refusal"]
    finally:
        packing._alone_blocks.cache_clear()
    assert trained == replayed == "key_too_wide", (trained, replayed)
    monkeypatch.setattr(xbar, "alone_blocks", lambda key: 3)
    lanes._table_blocks.cache_clear()
    packing._alone_blocks.cache_clear()
    try:
        _, trained, _ = design_refusal(app, ddos, names, names, encoding, float("inf"))
        replayed = model_breakdown(parse_program_text(text))["refusal"]
    finally:
        packing._alone_blocks.cache_clear()
    assert trained == replayed == "phv_containers_short", (trained, replayed)


def test_a_refused_replay_is_neither_feasible_nor_priced():
    refused = {"stage_depth": None, "blocks": None, "tables": [],
               "refusal": "program_shape", "refusal_reason": "stage_pin"}
    fields = replayed_fields(refused, 9, 20, 35)
    assert fields["model_refusal"] == "stage_pin"
    assert not fields["hw_feasible"] and not fields["budget_feasible"]
    assert fields["stage_depth"] is None and fields["tables"] is None
    over = {"stage_depth": 13, "blocks": 40, "tables": [], "refusal": "blocks_over",
            "refusal_reason": "blocks_over"}
    fields = replayed_fields(over, 13, 40, 35)
    assert fields["model_refusal"] is None
    assert not fields["hw_feasible"] and not fields["budget_feasible"]
    assert replayed_fields(over, 13, 40, float("inf"))["budget_feasible"]


# --- real p4c output: the register schedule on the option-B probe compiles -------------

_PROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures", "p4c_probe_7_1b")

# p4c's stage vs the model's, per register, where they differ (Task 2's probe report).
_KNOWN_REGISTER_MISSES = {
    ("joint-off_M025_s01_k17", "bwd_packet_length_max"): (4, 5),
    ("joint-off_M025_s01_k17", "fwd_last_arrival_time"): (5, 4),
    ("joint-tree_M035_s00_k17", "fwd_last_arrival_time"): (5, 4),
    ("joint-tree_M035_s00_k17", "fwd_packet_length_min"): (4, 5),
}


def test_the_register_schedule_matches_p4c_on_the_probe_compiles(monkeypatch):
    """Six programs compiled by p4c (results/encoding_evidence_2026_11/model8/impl/
    probe_7_1b, Task 2 of the model8 plan, option B: one unconditional register
    block), committed with p4c's per-register stages (mau.resources.log). Only the
    REGISTER schedule is tested: the texts predate later generator changes (the vote
    export, key pruning, full-width code fields), so they are not what the
    generator writes today -- check_program refuses the three joint ones
    (unexpected_shape:code_width, pre-4.3 live-width declarations) and, since
    Task 17b, the three per-tree ones (unexpected_shape:egress_assignment, the
    forwarding assignments last) -- and the
    schedule reads only their range tables, tree keys and register call sites.
    Model vs p4c (73/77 registers in p4c's stage): k<=8 35/35, k>=9 38/42; the four
    misses are pinned below. The texts also predate the symmetric 12-bit hash
    (review A1): their hash prologue is 3 stages, so the schedule is run with
    program.FLOW_HASH_LEVEL = 3, the value of their generator (registers.py reads
    the constant through the module, so the patch reaches the schedule)."""
    from p4model import program as program_constants
    monkeypatch.setattr(program_constants, "FLOW_HASH_LEVEL", 3)
    with open(_PROBE + ".json", encoding="utf-8") as handle:
        p4c = json.load(handle)
    assert len(p4c) == 6
    misses, total = {}, 0
    for name, numbers in sorted(p4c.items()):
        assert numbers["errors"] == 0 and numbers["compilation_succeeded"]
        with open(os.path.join(_PROBE, name + ".p4"), encoding="utf-8") as handle:
            program = parse_program_text(handle.read())
        tables = program.tables
        features = []
        for table, keys in tables.items():
            if table.startswith("table_") and keys[0][:-len("_val")] not in features:
                features.append(keys[0][:-len("_val")])
        dependents = {}
        for read in tree_range_reads(tables).values():
            for table in read:
                feature = tables[table][0][:-len("_val")]
                dependents[feature] = dependents.get(feature, 0) + 1
        placed = register_stage_schedule(features, dependents=dependents,
                                         call_order=program.register_calls)
        for register, stage in numbers["register_stages"].items():
            total += 1
            if placed.get(register) != stage:
                misses[(name, register)] = (stage, placed.get(register))
    assert total == 77
    assert misses == _KNOWN_REGISTER_MISSES
