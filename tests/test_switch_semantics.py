"""What the generated P4 program computes -- which is NOT rf.predict."""
import numpy as np
import pytest
from itertools import product
from sklearn.ensemble import RandomForestClassifier

from p4gen import switch_semantics as ss
from p4gen.build_p4_script import INFINITE, dt_thresholds_float_to_int, generate_voting_code


def _forest(n_estimators=7, n_classes=3, min_samples_leaf=200, n=1200, seed=0):
    """Impure leaves on purpose: min_samples_leaf up to 200 at max_depth <= 10 is
    the search's normal regime, and it is where hard and soft voting diverge."""
    rng = np.random.default_rng(seed)
    X = np.clip(rng.integers(0, 90000, size=(n, 5)), 0, INFINITE).astype(float)
    y = np.array([c % n_classes for c in range(n)])
    rf = RandomForestClassifier(
        n_estimators=n_estimators, max_depth=10, min_samples_leaf=min_samples_leaf,
        min_samples_split=10, random_state=seed, n_jobs=1).fit(X, y)
    return dt_thresholds_float_to_int(rf), X, y


def test_per_tree_indices_are_what_the_switch_carries():
    """meta.class_tree_<task>_<i> holds one class per tree -- the leaf's
    argmax, which is exactly what a sub-estimator's predict returns."""
    rf, X, y = _forest()

    idx = ss.per_tree_class_indices(rf, X)

    assert idx.shape == (len(rf.estimators_), X.shape[0])
    assert idx.dtype == np.intp
    for tree_idx, estimator in enumerate(rf.estimators_):
        assert np.array_equal(idx[tree_idx], estimator.predict(X).astype(np.intp))


def test_vote_winner_breaks_ties_toward_the_smallest_class_index():
    """Order-independence is the reason this rule is the specification:
    statistics.mode returns the FIRST-ENCOUNTERED mode, so its winner depends on
    tree ordering, which is arbitrary."""
    assert ss.vote_winner([1, 0, 2], 3) == 0
    assert ss.vote_winner([2, 1, 0], 3) == 0
    assert ss.vote_winner([0, 2, 1, 1, 2], 3) == 1
    assert ss.vote_winner([1, 1, 0], 3) == 1
    assert ss.vote_winner([0], 3) == 0


def test_vote_winner_is_invariant_to_tree_order():
    for arr in ([1, 0, 2], [0, 2, 1, 1, 2], [2, 2, 0, 1, 1, 1, 0]):
        expected = ss.vote_winner(arr, 3)
        for permuted in (list(reversed(arr)), sorted(arr), sorted(arr, reverse=True)):
            assert ss.vote_winner(permuted, 3) == expected, arr


def test_the_generated_vote_table_uses_exactly_vote_winner():
    """The switch's rule and this module's rule must be one rule. Parse the
    generated const entries back out and compare every one."""
    for num_trees, num_classes in ((3, 3), (5, 3), (7, 3), (2, 2), (3, 2)):
        table, _ = generate_voting_code(num_trees, num_classes, 'app')

        entries = {}
        for line in table.splitlines():
            line = line.strip()
            if not line.startswith('('):
                continue
            key_text, action = line.split(') : ')
            key = tuple(int(v) for v in key_text.lstrip('(').split(', '))
            entries[key] = int(action.split('(')[1].rstrip(');').rstrip(')'))

        assert len(entries) == num_classes ** num_trees, (num_trees, num_classes)
        for key, winner in entries.items():
            assert winner == ss.vote_winner(list(key), num_classes), (key, winner)


def test_switch_predict_is_the_hard_vote_over_the_per_tree_classes():
    rf, X, y = _forest()

    got = ss.switch_predict(rf, X)

    idx = ss.per_tree_class_indices(rf, X)
    expected = np.array([rf.classes_[ss.vote_winner(idx[:, i].tolist(), rf.n_classes_)]
                         for i in range(X.shape[0])])
    assert np.array_equal(got, expected)


def test_switch_predict_diverges_from_rf_predict_on_impure_leaves():
    """The defect this task exists for: rf.predict averages predict_proba (SOFT)
    while the switch votes hard. Measured on the real app dataset at 7 trees /
    min_samples_leaf=200 this is 6.6% of flows and 1.2 accuracy points."""
    rf, X, y = _forest(n_estimators=7, n_classes=3, min_samples_leaf=200)

    assert not np.array_equal(ss.switch_predict(rf, X), rf.predict(X))


def test_a_single_tree_forest_cannot_diverge():
    """With one tree the hard vote IS the tree's own argmax, which is what
    rf.predict reduces to. A useful sanity anchor: any divergence here would
    mean the per-tree class extraction is wrong."""
    rf, X, y = _forest(n_estimators=1)

    assert np.array_equal(ss.switch_predict(rf, X), rf.predict(X))


def test_switch_accuracy_scorer_matches_accuracy_of_switch_predict():
    from sklearn.metrics import accuracy_score

    rf, X, y = _forest()

    assert ss.switch_accuracy_scorer(rf, X, y) == pytest.approx(
        accuracy_score(y, ss.switch_predict(rf, X)))


def test_the_scorer_is_usable_by_permutation_importance():
    """P2 Task 5 passes it to permutation_importance, which calls a scorer as
    scorer(estimator, X, y) -- so the signature has to be exactly that."""
    from sklearn.inspection import permutation_importance

    rf, X, y = _forest(n=400)

    result = permutation_importance(
        rf, X, y, scoring=ss.switch_accuracy_scorer, n_repeats=2, random_state=0)

    assert result.importances_mean.shape == (X.shape[1],)


# ---------------------------------------------------------------------------
# Decision 1a / 7.1 option B: the feature registers' per-packet semantics.
# A small interpreter runs the GENERATED register logic (the apply block's
# .execute() call sites and every RegisterAction body, as P4 text) over a
# packet trace, and every <feature>_val is compared with the statistic computed
# directly from the flow's packets.
# ---------------------------------------------------------------------------

import random
import re

from p4gen.build_p4_script import generate_P4_registers_and_apply
from p4model.catalog import FEATURE_REGISTER_CATALOG

# The 17 features every saved forest pool has (the catalog's 18 minus
# max_packet_length, which no pool contains).
_POOL_FEATURES = [f for f in FEATURE_REGISTER_CATALOG if f != "max_packet_length"]
_MASK = 0xFFFF


def _p4_expr(text):
    text = text.replace("(bit<16>)", "")
    text = re.sub(r"\w+_halve_unit\.execute\(", "_halve(", text)
    text = re.sub(r"(\w+)_action\.execute\(meta\.flow_hash\)", r'_run("\1")', text)
    # saturating subtraction (C1f's gap): a |-| b = max(a - b, 0)
    text = re.sub(r"([\w.\[\]\"]+) \|-\| ([\w.\[\]\"]+)", r"_ssub(\1, \2)", text)
    text = re.sub(r"\b\d+w(0x[0-9A-Fa-f]+|\d+)\b", r"\1", text)      # 1w1, 32w0xFFFFFFFF
    text = text.replace("&&", " and ").replace("||", " or ")
    text = text.replace("hdr.ipv4.total_len", 'pkt["len"]').replace("hdr.ipv4.src_addr", 'pkt["src"]')
    return re.sub(r"meta\.(\w+)", r'meta["\1"]', text)


def _p4_to_python(lines):
    """P4 statements (one per line) -> Python source: if / else / assignments /
    calls, every assignment wrapped to the body's width (`_w`). A condition is
    evaluated exactly, without wrapping: the stateful ALU compares on 34-bit
    zero-extended operands (p4c CheckStatefulAlu; Task 33b live: a chain's first
    packet, `now - 0xFFFFFFFF > 65535`, does not fire), unlike the P4 text's mod
    2^32."""
    out, depth = [], 0
    for line in (line.strip() for line in lines):
        if not line:
            continue
        if line == "}":
            depth -= 1
            continue
        if line == "} else {":
            out.append("  " * (depth - 1) + "else:")
            continue
        opened = re.match(r"^if \((.*)\) \{$", line)
        if opened:
            out.append("  " * depth + "if " + _p4_expr(opened.group(1)) + ":")
            depth += 1
            continue
        assert line.endswith(";"), line
        assign = re.match(r"^([\w.]+) = (.*);$", line)
        if assign:
            out.append("  " * depth + "%s = _w(%s)" % (_p4_expr(assign.group(1)),
                                                      _p4_expr(assign.group(2))))
        else:
            out.append("  " * depth + _p4_expr(line[:-1]))
    assert depth == 0
    return "\n".join(out) or "pass"


class _Switch:
    """The generated register logic for one feature set, interpreted."""

    def __init__(self, features):
        regs, actions, apply_code, _ = generate_P4_registers_and_apply(
            {f: None for f in features})
        self.initial = {name: int(re.sub(r"^\d+w", "", init or "0"), 0) for init, name in re.findall(
            r"Register<bit<\d+>, bit<\d+>>\(MAX_NUM_FLOWS(?:, (\w+))?\) (\w+)_reg;", regs)}
        self.bodies = {}
        # every RegisterAction, the orientation's (flow_forward_srcaddr_reg) included
        for register, width, body in re.findall(
                r"\((\w+)_reg\) \w+_action = \{\n\t\tvoid apply\(inout bit<(\d+)> value, "
                r"out bit<\d+> rv\) \{\n(.*?)\n\t\t\}\n\t\};", actions, re.S):
            name = "flow_orientation" if register == "flow_forward_srcaddr" else register
            source = "def body(value, meta, pkt):\n  rv = None\n%s\n  return value, rv\n" % (
                "\n".join("  " + line for line in _p4_to_python(body.splitlines()).splitlines()))
            mask = (1 << int(width)) - 1
            scope = {"_w": lambda x, mask=mask: x & mask, "_halve": lambda x: (x & _MASK) // 2,
                     "_ssub": lambda a, b: max(a - b, 0)}
            exec(source, scope)
            self.bodies[name] = scope["body"]
        self.initial["flow_orientation"] = self.initial.pop("flow_forward_srcaddr")
        self.apply = compile(_p4_to_python(apply_code.splitlines()), "<apply>", "exec")
        self.state = {}

    def packet(self, pkt):
        meta = {}

        def run(name):
            key = (name, pkt["flow"])
            value, rv = self.bodies[name](self.state.get(key, self.initial[name]), meta, pkt)
            self.state[key] = value
            return rv

        def calc_flow_hash():
            meta["flow_hash"] = pkt["flow"]

        def calc_timestamp():
            meta["now_pseudo_us"] = pkt["now"]

        exec(self.apply, {"_w": lambda x: x & _MASK, "_run": run, "meta": meta, "pkt": pkt,
                          "calc_flow_hash": calc_flow_hash, "calc_timestamp": calc_timestamp})
        return meta


def _trace(n=40, flows=3, seed=7):
    rng = random.Random(seed)
    now, packets = 100, []
    ends = {flow: (10 + 2 * flow, 11 + 2 * flow) for flow in range(flows)}
    for i in range(n):
        flow = i % flows if i < 2 * flows else rng.randrange(flows)
        # the first two packets of every flow go one each way
        src = ends[flow][0 if i < flows else 1] if i < 2 * flows else rng.choice(ends[flow])
        # one idle time past the 16-bit range (V1o's clamp), the rest short
        now += 100000 if i == 25 else rng.randint(0, 50)
        packets.append({"flow": flow, "src": src, "now": now, "len": rng.randint(40, 1500)})
    return packets


def _expected(feature, history):
    """CICFlowMeter's statistic of `feature` over its own direction's packets of the
    flow so far (history: [(fwd, now, len)]), in the switch's encoding (V1o + C1f,
    Task 34): an IAT of g ticks is min(g + 1, 65535), the chain's first packet has
    none; Max / Min / Mean of no IAT are 0; means are the alpha = 1/2 EWMA the
    catalog documents (from 0, over the IATs only)."""
    entry = FEATURE_REGISTER_CATALOG[feature]
    gate = entry.get("gated_by")
    own = [(now, length) for fwd, now, length in history
           if gate is None or fwd == (gate == "fwd")]
    value_body = [r for r in entry["registers"] if r["role"] == "value"][0]["body"]
    if "packet_length" in value_body:
        samples = [length for _, length in own]
    else:
        samples = [min(now - previous + 1, 0xFFFF)
                   for (now, _), (previous, _) in zip(own[1:], own[:-1])]
    if value_body.startswith("running_max"):
        return max(samples, default=0)
    if value_body.startswith("running_min"):
        return min(samples, default=0 if value_body == "running_min_iat" else 0xFFFF)
    # The P4 text's Mean: bit<16> `value + x`, then halved -- the sum wraps at 2^16 (a
    # clamped 65535 IAT does). Pre-existing and left as is (the Mean registers are a
    # separate decision; p4c even compiles value + f(x), Task 33b).
    mean = 0
    for sample in samples:
        mean = ((mean + sample) & 0xFFFF) // 2
    return mean


def test_option_b_switch_semantics():
    switch = _Switch(_POOL_FEATURES)
    histories = {}
    first_src = {}
    directions = set()
    for pkt in _trace():
        fwd = first_src.setdefault(pkt["flow"], pkt["src"]) == pkt["src"]
        directions.add((pkt["flow"], fwd))
        history = histories.setdefault(pkt["flow"], [])
        history.append((fwd, pkt["now"], pkt["len"]))
        before = dict(switch.state)
        meta = switch.packet(pkt)
        assert meta["fwd"] == int(fwd)
        for feature in _POOL_FEATURES:
            # written on every packet, both directions
            assert feature + "_val" in meta, (feature, pkt)
            assert meta[feature + "_val"] == _expected(feature, history), (feature, pkt)
        # the other direction's registers are read, never updated
        for entry in FEATURE_REGISTER_CATALOG.values():
            gate = entry.get("gated_by")
            if gate is None or (gate == "fwd") == fwd:
                continue
            for reg in entry["registers"]:
                key = (reg["name"], pkt["flow"])
                if key in before:
                    assert switch.state[key] == before[key], (reg["name"], pkt)
    assert len(directions) == 6  # 3 flows, both directions each
