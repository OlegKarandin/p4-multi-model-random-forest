"""Gap 6 (phase P5): the run manifest. One manifest_<runid>.json per
invocation of compare_independent_joint_mapping, so a reader can confirm the
arms differ as claimed and which code produced which numbers.

Nothing anywhere in this repo wrote JSON provenance or read a git SHA before
this. Three things this module must get right, each with its own test group
below: delta_align: None must survive JSON round-tripping distinguishably
from 0.0 (the whole joint-dinf vs joint-d000 distinction); git provenance
must degrade to a recorded None rather than raise, both when the tree is
merely dirty (the NORMAL state here) and when git/the repo is unavailable
entirely; and two close-together invocations must not collide on a filename.
"""
import json
import subprocess

import pytest

from src.reporting.manifest import (
    build_manifest,
    generate_runid,
    git_provenance,
    library_versions,
    write_run_manifest,
)
from src.training.config import TrainConfig


PRIMARY_ARMS = [
    ('independent', TrainConfig()),
    ('joint', TrainConfig(alignment_enabled=False)),
    ('joint', TrainConfig(delta_align=0.0)),
]


# ---------------------------------------------------------------------------
# Round-trip through json.load
# ---------------------------------------------------------------------------

def test_manifest_round_trips_through_json_load(tmp_path):
    path = write_run_manifest(
        arms=PRIMARY_ARMS, M_values=[25, 40], n_splits=2,
        n_rows_app=1234, n_rows_ddos=5678,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))

    assert path is not None
    with open(path) as f:
        loaded = json.load(f)

    assert loaded['M_values'] == [25, 40]
    assert loaded['n_splits'] == 2
    assert loaded['dataset_rows'] == {'app': 1234, 'ddos': 5678}
    assert len(loaded['arms']) == 3
    assert 'timestamp_utc' in loaded
    assert 'git' in loaded and 'sha' in loaded['git'] and 'dirty' in loaded['git']
    assert 'library_versions' in loaded


def test_manifest_records_no_candidate_pre_filter_explicitly():
    """The findings spec's endpoint_ratio_cap column predates P3's removal of
    the filter; the manifest states positively that none is in effect,
    rather than carrying an always-empty column."""
    manifest = build_manifest(
        arms=PRIMARY_ARMS, M_values=[25], n_splits=2,
        n_rows_app=10, n_rows_ddos=10)

    assert manifest['candidate_pre_filter']['active'] is False
    note = manifest['candidate_pre_filter']['note'].lower()
    assert 'no candidate pre-filter' in note or 'no candidate pre filter' in note


def test_manifest_records_library_versions():
    versions = library_versions()
    for name in ('python', 'numpy', 'pandas', 'sklearn', 'optuna'):
        assert name in versions
        assert versions[name]  # non-empty string, not swallowed to None


# ---------------------------------------------------------------------------
# delta_align: None vs 0.0 -- the whole joint-dinf vs joint-d000 distinction
# ---------------------------------------------------------------------------

def test_delta_align_none_survives_the_round_trip_distinguishable_from_zero(tmp_path):
    arms = [
        ('joint', TrainConfig(delta_align=None)),
        ('joint', TrainConfig(delta_align=0.0)),
    ]
    path = write_run_manifest(
        arms=arms, M_values=[25], n_splits=2,
        n_rows_app=10, n_rows_ddos=10,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))

    with open(path) as f:
        loaded = json.load(f)

    dinf_config = loaded['arms'][0]['config']
    d000_config = loaded['arms'][1]['config']

    assert dinf_config['delta_align'] is None
    assert d000_config['delta_align'] == 0.0
    assert dinf_config['delta_align'] != d000_config['delta_align']
    # Not stringified either -- a naive str(cfg.delta_align) would make both
    # "None" and "0.0" survive as strings, which is a subtler way to lose the
    # distinction than outright coercion to a shared sentinel.
    assert not isinstance(d000_config['delta_align'], str)

    # And the raw JSON text itself must contain a real `null`, not the string
    # "null" or "None" -- guards against a stringifying encoder.
    raw = path
    with open(raw) as f:
        text = f.read()
    assert '"delta_align": null' in text or '"delta_align":null' in text


def test_delta_align_of_zero_is_not_coerced_to_none_or_dropped():
    manifest = build_manifest(
        arms=[('joint', TrainConfig(delta_align=0.0))],
        M_values=[25], n_splits=2, n_rows_app=10, n_rows_ddos=10)

    cfg_dict = manifest['arms'][0]['config']
    assert 'delta_align' in cfg_dict
    assert cfg_dict['delta_align'] == 0.0
    assert cfg_dict['delta_align'] is not None


# ---------------------------------------------------------------------------
# The M grid and n_splits actually used, not the defaults
# ---------------------------------------------------------------------------

def test_records_the_M_grid_and_n_splits_actually_passed_in_not_the_default():
    manifest = build_manifest(
        arms=PRIMARY_ARMS, M_values=[25], n_splits=2,
        n_rows_app=10, n_rows_ddos=10)

    # Today's default grid/n_splits (main.py's run_main), guarded against
    # accidentally leaking in instead of the value actually passed.
    assert manifest['M_values'] != [25, 40, 50, 60, 75, 90, 100]
    assert manifest['n_splits'] != 15
    assert manifest['M_values'] == [25]
    assert manifest['n_splits'] == 2


# ---------------------------------------------------------------------------
# git provenance: dirty is recorded, not fatal; unavailable degrades to None
# ---------------------------------------------------------------------------

def _run_git(args, cwd):
    subprocess.run(['git'] + args, cwd=cwd, check=True,
                    capture_output=True)


def _init_repo(tmp_path):
    repo = tmp_path / 'repo'
    repo.mkdir()
    _run_git(['init', '-q'], cwd=str(repo))
    _run_git(['config', 'user.email', 'test@example.com'], cwd=str(repo))
    _run_git(['config', 'user.name', 'Test'], cwd=str(repo))
    (repo / 'a.txt').write_text('hello')
    _run_git(['add', 'a.txt'], cwd=str(repo))
    _run_git(['commit', '-q', '-m', 'initial'], cwd=str(repo))
    return repo


def test_dirty_working_tree_is_recorded_as_a_flag_not_raised():
    """Dirty is the NORMAL state in this repository: the standing rule means
    .md files are routinely uncommitted. A dirty tree must never fail
    provenance collection -- it must show up as dirty: True."""
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        from pathlib import Path
        repo = _init_repo(Path(tmp))
        (repo / 'a.txt').write_text('modified, uncommitted')

        result = git_provenance(cwd=str(repo))

        assert result['sha'] is not None
        assert len(result['sha']) == 40
        assert result['dirty'] is True
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_clean_working_tree_is_recorded_as_not_dirty():
    import tempfile
    import shutil
    tmp = tempfile.mkdtemp()
    try:
        from pathlib import Path
        repo = _init_repo(Path(tmp))

        result = git_provenance(cwd=str(repo))

        assert result['sha'] is not None
        assert result['dirty'] is False
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def test_git_unavailable_or_not_a_repo_degrades_to_null_rather_than_raising(tmp_path):
    """tmp_path is not inside any git repository (or is at worst inside this
    one by accident of pytest's tmp dir location, which it never is), so
    `git rev-parse HEAD` fails there -- this must not raise."""
    not_a_repo = tmp_path / 'not_a_repo'
    not_a_repo.mkdir()

    result = git_provenance(cwd=str(not_a_repo))

    assert result == {'sha': None, 'dirty': None}


def test_git_executable_missing_degrades_to_null_rather_than_raising(monkeypatch, tmp_path):
    """Simulates git itself being unavailable (e.g. not on PATH), independent
    of whether cwd is a repo -- must not raise even when the subprocess call
    cannot find the executable at all."""
    import src.reporting.manifest as manifest_mod

    def _raise_not_found(*args, **kwargs):
        raise FileNotFoundError("git executable not found")

    monkeypatch.setattr(manifest_mod.subprocess, 'check_output', _raise_not_found)

    result = git_provenance(cwd=str(tmp_path))

    assert result == {'sha': None, 'dirty': None}


def test_git_provenance_of_this_actual_repository_does_not_raise():
    """Smoke test against the real repository this file lives in -- whatever
    its dirty state happens to be, this must return a well-shaped dict."""
    result = git_provenance()
    assert 'sha' in result
    assert 'dirty' in result


# ---------------------------------------------------------------------------
# Provenance must never crash the campaign
# ---------------------------------------------------------------------------

def test_a_manifest_that_cannot_be_serialised_is_skipped_not_raised(tmp_path, capsys):
    """An un-JSON-able row count (or any other build failure) must degrade to
    'no manifest written' rather than take down the calling campaign -- and
    must do so LOUDLY: the entire justification for swallowing the exception
    here is that a WARNING is printed instead, so that property must itself
    be asserted, not just inferred from the code not crashing."""
    class NotJSONable:
        pass

    path = write_run_manifest(
        arms=PRIMARY_ARMS, M_values=[25], n_splits=2,
        n_rows_app=NotJSONable(), n_rows_ddos=10,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))

    assert path is None
    # And critically: no partial directory/file was left behind either.
    assert not (tmp_path / 'manifests').exists()

    # The degradation is visible, not silent: a WARNING line (and its
    # traceback) actually fires, both on stderr so the identifying line and
    # its detail are on the same stream -- a future edit that deleted the
    # print/traceback.print_exc() calls would still return None here, but
    # would fail this assertion.
    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'WARNING: failed to write run manifest' in captured.err
    assert 'NotJSONable' in captured.err or 'not JSON serializable' in captured.err


def test_an_unwritable_directory_is_skipped_not_raised(tmp_path, monkeypatch, capsys):
    import src.reporting.manifest as manifest_mod

    def _raise(*args, **kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(manifest_mod.os, 'makedirs', _raise)

    path = write_run_manifest(
        arms=PRIMARY_ARMS, M_values=[25], n_splits=2,
        n_rows_app=10, n_rows_ddos=10,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))

    assert path is None

    captured = capsys.readouterr()
    assert captured.out == ''
    assert 'WARNING: failed to write run manifest' in captured.err
    assert 'disk full' in captured.err


# ---------------------------------------------------------------------------
# Collision-proof runid
# ---------------------------------------------------------------------------

def test_two_runids_generated_back_to_back_never_collide():
    ids = {generate_runid() for _ in range(200)}
    assert len(ids) == 200


def test_two_manifests_written_back_to_back_land_in_different_files(tmp_path):
    path1 = write_run_manifest(
        arms=PRIMARY_ARMS, M_values=[25], n_splits=2,
        n_rows_app=10, n_rows_ddos=10,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))
    path2 = write_run_manifest(
        arms=PRIMARY_ARMS, M_values=[40], n_splits=2,
        n_rows_app=10, n_rows_ddos=10,
        directory=str(tmp_path / 'manifests'), cwd=str(tmp_path))

    assert path1 != path2
    import os
    assert os.path.isfile(path1)
    assert os.path.isfile(path2)
