"""Model constants of the disjoint_per_tree encoding (arm independent-tree; spec
2026-11-17-independent-tree-encoding-design.md): the registry entry, is_packed, and
packed.layout_for's n_vals (K1: one _val field per RAW feature)."""
import pytest

from p4gen import evaluation
from p4model import packed, phv, target
from p4model.errors import PhvContainersExhausted
from p4model.program import (APP_TASK, DDOS_TASK, DISJOINT_ENCODING, DISJOINT_PER_TREE_ENCODING,
                             ENCODINGS, JOINT_ENCODING, PER_TASK_ENCODING, PER_TREE_ENCODING,
                             is_packed)
from tests.test_packed_pool import _features
from tests.test_per_task_codes import _rng_design


def test_new_encoding_is_registered():
    assert DISJOINT_PER_TREE_ENCODING == "disjoint_per_tree"
    assert DISJOINT_PER_TREE_ENCODING in ENCODINGS
    assert ENCODINGS[-1] == "disjoint_per_tree"
    assert is_packed(PER_TREE_ENCODING) and is_packed(DISJOINT_PER_TREE_ENCODING)
    for encoding in (JOINT_ENCODING, DISJOINT_ENCODING, PER_TASK_ENCODING):
        assert not is_packed(encoding)


def _args(seed):
    app, ddos, names = _rng_design(seed)
    own = {task: [{f: i for f, i in evaluation.feature_intervals_from_nodes({0: nodes}).items()
                   if len(i) >= 2}
                  for nodes in evaluation.tree_nodes_for(clf, names).values()]
           for task, clf in ((APP_TASK, app), (DDOS_TASK, ddos))}
    return packed.widths_of(own), _features(app, ddos, names), [APP_TASK, DDOS_TASK]


@pytest.mark.parametrize("seed", range(6))
def test_layout_for_default_n_vals_is_unchanged(seed):
    widths, features, order = _args(seed)
    base = packed.layout_for(widths, features, order)
    with_code = {f for f in features if any(f in tree for trees in widths.values() for tree in trees)}
    assert packed.layout_for(widths, features, order, n_vals=None) == base
    assert packed.layout_for(widths, features, order, n_vals=len(with_code)) == base


def test_n_vals_changes_only_the_budget(monkeypatch):
    widths = {APP_TASK: [{"app_f": 10}], DDOS_TASK: [{"ddos_f": 10}]}
    features, order = ["app_f", "ddos_f"], [APP_TASK, DDOS_TASK]
    two = packed.layout_for(widths, features, order, n_vals=2)
    one = packed.layout_for(widths, features, order, n_vals=1)
    assert (one.containers, one.place) == (two.containers, two.place)
    assert sorted(two.containers.values()) == [(16,), (16,)]
    # exactly one 16-bit container spare once one _val field is counted
    spare = len(target.PHV_TEMPLATE_FIELDS[16]) + 1 + 2
    monkeypatch.setitem(target.PHV_INGRESS_CONTAINERS, 16, spare)
    monkeypatch.setitem(phv.PHV_INGRESS_CONTAINERS, 16, spare)
    with pytest.raises(PhvContainersExhausted):
        packed.layout_for(widths, features, order, n_vals=2)
    packed.layout_for(widths, features, order, n_vals=1)
