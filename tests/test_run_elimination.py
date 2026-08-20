"""One parameterised elimination loop replaces two near-duplicate ones."""
import json

import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.training import feature_selection as fs
from src.training.config import TrainConfig
from src.training.errors import NoFeasibleSolution
from src.training.splits import make_task_splits
from src.training.train_model import TrainResult

FEATURE_NAMES = ['Flow.IAT.Max', 'Fwd.IAT.Max', 'Fwd.Packet.Length.Max']


def _splits(seed=0):
    rng = np.random.default_rng(seed)
    n = 400
    X_app = rng.integers(0, 500, size=(n, 3)).astype(float)
    y_app = np.array([c % 3 for c in range(n)])
    X_ddos = rng.integers(0, 500, size=(n, 3)).astype(float)
    y_ddos = np.array([-1, 1] * (n // 2))
    return (make_task_splits(X_app, y_app, random_state=42),
            make_task_splits(X_ddos, y_ddos, random_state=42))


def _fake_trainer(raise_at_k=(), record=None):
    """Stands in for the TrainResult-returning trainer (P5 gap 2 -- was a
    7-tuple prior to Task 7)."""
    def trainer(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                val_select_A, val_select_B, features_A, features_B,
                max_blocks, encoding, cfg, warm_start_params=None):
        k = X_A.shape[1]
        if record is not None:
            record.append({'k': k, 'encoding': encoding,
                           'features_A': list(features_A),
                           'features_B': list(features_B),
                           'n_train_A': len(y_A),
                           'n_val_align_A': len(val_align_A[1]),
                           'n_val_select_A': len(val_select_A[1])})
        if k in raise_at_k:
            raise NoFeasibleSolution(k=k, max_blocks=max_blocks)
        model_A = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A)
        model_B = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_B, y_B)
        return TrainResult(
            model_A=model_A, model_B=model_B, stages=1, blocks=1,
            acc_sel_A=0.71, acc_sel_B=0.91, best_params={'n_estimators_A': 1},
            rel_shortfall=0.0, n_trials_run=1, n_feasible=1,
            align_attempted=None, align_accepted=None,
            intervals_before=None, intervals_after=None)
    return trainer


def _run(monkeypatch, arm, raise_at_k=(), record=None):
    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained',
        _fake_trainer(raise_at_k, record))
    app, ddos = _splits()
    return fs._run_elimination(
        arm=arm, split_idx=10, app=app, ddos=ddos,
        feature_names=list(FEATURE_NAMES), max_blocks=25, cfg=TrainConfig())


def test_independent_arm_produces_one_row_per_k(monkeypatch):
    rows = _run(monkeypatch, 'independent')

    assert sorted(r['k'] for r in rows) == [1, 2, 3]
    assert {r['arm'] for r in rows} == {'independent'}
    assert {r['method'] for r in rows} == {'single'}


def test_joint_arm_produces_one_row_per_k(monkeypatch):
    rows = _run(monkeypatch, 'joint')

    assert sorted(r['k'] for r in rows) == [1, 2, 3]
    assert {r['arm'] for r in rows} == {'joint'}
    assert {r['method'] for r in rows} == {'multi'}


def test_an_unknown_arm_is_rejected(monkeypatch):
    """'disjoint' is the ENCODING name, not an arm name -- easily typo'd in
    place of 'independent'. An unvalidated arm would silently run the
    independent path while writing the typo verbatim into row['arm'],
    corrupting a results file's identity column."""
    import pytest

    app, ddos = _splits()
    with pytest.raises(ValueError, match='arm'):
        fs._run_elimination(
            arm='disjoint', split_idx=10, app=app, ddos=ddos,
            feature_names=list(FEATURE_NAMES), max_blocks=25, cfg=TrainConfig())


def test_the_arm_chooses_the_encoding(monkeypatch):
    record = []
    _run(monkeypatch, 'independent', record=record)
    assert {c['encoding'] for c in record} == {'disjoint'}

    record = []
    _run(monkeypatch, 'joint', record=record)
    assert {c['encoding'] for c in record} == {'joint'}


def test_the_joint_arm_always_passes_one_shared_feature_set(monkeypatch):
    """Under joint encoding every tree's codeword spans the MERGED interval
    pool, so a feature used by only one task widens BOTH tasks' codewords.
    Sharing the feature set is part of the treatment, not an implementation
    convenience."""
    record = []
    _run(monkeypatch, 'joint', record=record)

    for call in record:
        assert call['features_A'] == call['features_B']


def test_the_independent_arm_lets_the_feature_sets_diverge(monkeypatch):
    """Independent is unconstrained: each task ranks features by its OWN
    permutation importance, so nothing forces the two elimination orders to
    match. Feed genuinely different importance rankings per task -- app's
    rises left-to-right (drops the lowest-index feature first), ddos's falls
    left-to-right (drops the highest-index feature first) -- and confirm the
    feature sets actually diverge, not just that their lengths stay equal.
    """
    call_count = {'n': 0}

    def _fake_importance(model, X, y, **kwargs):
        call_count['n'] += 1
        k = X.shape[1]
        ascending = np.arange(1, k + 1, dtype=float)
        # _run_elimination calls importance_app then importance_ddos each
        # iteration, so odd calls are app, even calls are ddos.
        values = ascending if call_count['n'] % 2 == 1 else ascending[::-1]
        return type('Importance', (), {'importances_mean': values})()

    monkeypatch.setattr(fs, 'permutation_importance', _fake_importance)

    record = []
    _run(monkeypatch, 'independent', record=record)

    # Both start from the same list, so the sets are equal at k=3 (the first
    # trainer call, before any drop has happened yet).
    assert record[0]['features_A'] == record[0]['features_B']
    # After the first drop: app dropped the lowest-index feature (index 0),
    # ddos dropped the highest-index one -- the sets MUST now differ.
    assert record[1]['features_A'] != record[1]['features_B']
    assert all(len(c['features_A']) == len(c['features_B']) for c in record)


def test_the_trainer_receives_the_train_bucket_not_the_whole_dataset(monkeypatch):
    """55% train, 15% val_align, 15% val_select -- confirm the buckets that
    reach the trainer are the split ones, not the full arrays."""
    record = []
    _run(monkeypatch, 'joint', record=record)

    n_total = 400
    assert record[0]['n_train_A'] / n_total < 0.60
    assert record[0]['n_train_A'] / n_total > 0.50
    assert record[0]['n_val_align_A'] / n_total < 0.20
    assert record[0]['n_val_select_A'] / n_total < 0.20


def test_selection_time_accuracies_are_recorded_separately_from_test_ones(monkeypatch):
    """acc_sel_* come from val_select (what the search optimised); acc_* come
    from X_test (which nothing selects on). Conflating them would make the
    reported numbers optimistic."""
    rows = _run(monkeypatch, 'joint')

    assert all(r['acc_sel_app'] == 0.71 for r in rows)
    assert all(r['acc_sel_ddos'] == 0.91 for r in rows)
    assert all(r['acc_app'] != 0.71 for r in rows)


def test_an_infeasible_middle_k_still_records_and_continues(monkeypatch):
    """F3b, now in one place instead of two."""
    rows = _run(monkeypatch, 'joint', raise_at_k=(2,))

    by_k = {r['k']: r for r in rows}
    assert sorted(by_k) == [1, 2, 3]
    assert by_k[2]['infeasible'].startswith('no feasible solution at k=2')
    assert by_k[2]['acc_app'] is None
    assert by_k[1]['infeasible'] == ''


def test_an_infeasible_first_k_breaks_with_one_row(monkeypatch):
    rows = _run(monkeypatch, 'independent', raise_at_k=(3,))

    assert len(rows) == 1
    assert rows[0]['k'] == 3


def test_hardware_validation_survives_an_interleaved_infeasible_row(monkeypatch):
    """Regression test for 8a4746d ("Decouple hardware-validation joins from
    results list position"): an infeasible row between two feasible ones must
    not misattribute a neighboring iteration's real-compile result, and the
    infeasible row itself must carry the same three keys (all None) that
    `_process_single_split`'s docstring promises every row gets.

    `tests/test_feature_selection_infeasible.py` (deleted when this file was
    added) was the purpose-built regression test for exactly this bug; this
    replaces that coverage against the merged loop.
    """
    from src.p4gen.p4_compile import CompileResult

    class _FakeFuture:
        def __init__(self, k):
            self.k = k

        def result(self, timeout=None):
            return CompileResult(errors=0, warnings=0, stages=100 + self.k,
                                  tables=100 + self.k, tcam=100 + self.k)

    def _fake_kickoff(validate_on_hardware, hardware_output_dir, split_idx, method, k,
                       model_app, model_ddos, names_app, names_ddos, encoding, config=None):
        # Stands in for a real compile handle -- encodes k so the assertions
        # below can tell whether a row's numbers came from ITS OWN iteration
        # or got misattributed to a neighbor.
        return _FakeFuture(k)

    monkeypatch.setattr(fs, '_kickoff_hardware_validation', _fake_kickoff)
    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained',
        _fake_trainer(raise_at_k=(2,)))

    app, ddos = _splits()
    rows = fs._run_elimination(
        arm='joint', split_idx=10, app=app, ddos=ddos,
        feature_names=list(FEATURE_NAMES), max_blocks=25, cfg=TrainConfig(),
        validate_on_hardware=True, hardware_output_dir='unused/')

    by_k = {r['k']: r for r in rows}
    assert sorted(by_k) == [1, 2, 3]

    # k=3 and k=1 are feasible: each gets ITS OWN compile result, not a
    # neighbor's -- k=2's infeasible row sits between them and carries no
    # compile handle at all, so it must not shift the attribution.
    assert by_k[3]['stages_real'] == 103
    assert by_k[3]['tcam_real'] == 103
    assert by_k[3]['compile_errors'] == 0

    assert by_k[1]['stages_real'] == 101
    assert by_k[1]['tcam_real'] == 101
    assert by_k[1]['compile_errors'] == 0

    # k=2 is infeasible: no compile ever ran for it, so all three stay None
    # -- this is the fix for Important #1 (the row used to be missing these
    # keys entirely).
    assert by_k[2]['stages_real'] is None
    assert by_k[2]['tcam_real'] is None
    assert by_k[2]['compile_errors'] is None


# ---------------------------------------------------------------------------
# P5 gap 3: the row dicts must carry feature-name provenance, best_params,
# and the alignment diagnostics -- with the None/''/0 distinction preserved.
# ---------------------------------------------------------------------------

def test_feasible_rows_carry_provenance_and_alignment_fields_including_a_real_zero(monkeypatch):
    """A real align_accepted of 0 is falsy in Python -- `if not value` would
    write it as '', erasing the difference between "alignment accepted
    nothing" and "alignment never ran". This trainer returns a real 0 to
    catch exactly that bug."""
    def trainer(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                val_select_A, val_select_B, features_A, features_B,
                max_blocks, encoding, cfg, warm_start_params=None):
        model_A = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A)
        model_B = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_B, y_B)
        from src.training.train_model import TrainResult
        return TrainResult(
            model_A=model_A, model_B=model_B, stages=1, blocks=1,
            acc_sel_A=0.71, acc_sel_B=0.91, best_params={'a': 1},
            rel_shortfall=0.05, n_trials_run=7, n_feasible=4,
            align_attempted=3, align_accepted=0,
            intervals_before=12, intervals_after=9)

    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained', trainer)

    app, ddos = _splits()
    rows = fs._run_elimination(
        arm='joint', split_idx=10, app=app, ddos=ddos,
        feature_names=list(FEATURE_NAMES), max_blocks=25, cfg=TrainConfig())

    assert len(rows) == 3
    for row in rows:
        assert row['best_params'] == json.dumps({'a': 1})
        assert row['rel_shortfall'] == 0.05
        assert row['n_trials_run'] == 7
        assert row['n_feasible'] == 4
        assert row['align_attempted'] == 3
        # The load-bearing assertion: 0 must survive as 0, not become ''.
        assert row['align_accepted'] == 0
        assert row['align_accepted'] != ''
        assert row['intervals_before'] == 12
        assert row['intervals_after'] == 9

    # k=3 is the first iteration -- names are still the untouched full list.
    by_k = {r['k']: r for r in rows}
    assert by_k[3]['features_app'] == ';'.join(FEATURE_NAMES)
    assert by_k[3]['features_ddos'] == ';'.join(FEATURE_NAMES)


def test_alignment_fields_are_empty_string_not_none_or_zero_when_alignment_never_ran(monkeypatch):
    """The default fake trainer returns None for the four alignment fields
    (mirrors the disjoint arm / alignment_enabled=False case). The CSV must
    carry '' there, not a literal None and not 0."""
    rows = _run(monkeypatch, 'independent')

    assert len(rows) == 3
    for row in rows:
        assert row['align_attempted'] == ''
        assert row['align_accepted'] == ''
        assert row['intervals_before'] == ''
        assert row['intervals_after'] == ''
        assert row['rel_shortfall'] == 0.0  # a real number, since it ran and returned 0.0
        assert row['n_trials_run'] == 1
        assert row['n_feasible'] == 1


def test_infeasible_row_best_params_is_empty_string_not_the_previous_ks_params(monkeypatch):
    """warm_start_params is still bound from the previous k when a later k is
    infeasible. A naive json.dumps(best_params) there would silently write
    the PREVIOUS k's params. This is the single most dangerous bug in this
    task: it produces plausible, wrong data rather than a visible failure."""
    def trainer(X_A, y_A, X_B, y_B, val_align_A, val_align_B,
                val_select_A, val_select_B, features_A, features_B,
                max_blocks, encoding, cfg, warm_start_params=None):
        k = X_A.shape[1]
        if k == 2:
            from src.training.errors import NoFeasibleSolution
            raise NoFeasibleSolution(k=k, max_blocks=max_blocks)
        model_A = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_A, y_A)
        model_B = RandomForestClassifier(
            n_estimators=1, max_depth=2, random_state=0).fit(X_B, y_B)
        from src.training.train_model import TrainResult
        return TrainResult(
            model_A=model_A, model_B=model_B, stages=1, blocks=1,
            acc_sel_A=0.71, acc_sel_B=0.91, best_params={'k': k},
            rel_shortfall=0.0, n_trials_run=1, n_feasible=1,
            align_attempted=None, align_accepted=None,
            intervals_before=None, intervals_after=None)

    monkeypatch.setattr(
        'src.training.train_model.train_multi_RF_Optuna_multi_constrained', trainer)

    app, ddos = _splits()
    rows = fs._run_elimination(
        arm='joint', split_idx=10, app=app, ddos=ddos,
        feature_names=list(FEATURE_NAMES), max_blocks=25, cfg=TrainConfig())

    by_k = {r['k']: r for r in rows}
    assert sorted(by_k) == [1, 2, 3]
    assert by_k[3]['best_params'] == json.dumps({'k': 3})
    # The trap: must be '', and specifically must NOT be k=3's params.
    assert by_k[2]['best_params'] == ''
    assert by_k[2]['best_params'] != json.dumps({'k': 3})
    # k=1 ran after the infeasible k=2 and is feasible again -- its own
    # params, not a stale k=3 or k=2 value.
    assert by_k[1]['best_params'] == json.dumps({'k': 1})

    # The infeasible row's other new fields must also be '', not a stale
    # carry-over from the previous iteration.
    for key in ('rel_shortfall', 'n_trials_run', 'n_feasible',
                'align_attempted', 'align_accepted',
                'intervals_before', 'intervals_after'):
        assert by_k[2][key] == ''
    # features_* ARE expected on the infeasible row too (names_app/names_ddos
    # are in scope regardless of whether training succeeded).
    assert by_k[2]['features_app'] != ''
    assert len(by_k[2]['features_app'].split(';')) == 2


def test_features_round_trip_through_the_semicolon_join(monkeypatch):
    """Feature names contain dots but never semicolons, so ';'.join/split is
    a lossless round trip."""
    rows = _run(monkeypatch, 'independent')

    for row in rows:
        names = row['features_app'].split(';')
        assert len(names) == row['k']
        for name in names:
            assert name in FEATURE_NAMES
        # Confirms the join actually used ';' and not something that would
        # collide with a dot in a feature name.
        assert all('.' in n for n in names)
