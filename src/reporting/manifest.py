"""Run-level provenance for phase P5's schema (spec C.2, gap 6).

Tasks 7-9 put diagnostics and provenance on each result ROW (which alignment
guards fired, how many trials ran, ...). This module is the companion at the
RUN level: one manifest_<runid>.json per invocation of
compare_independent_joint_mapping, recording which arms were compared, the
grid actually swept, the dataset sizes, the git commit the numbers came from,
and the library versions -- what a reader needs to confirm the arms differ as
claimed and to reproduce the run.

Nothing anywhere else in this repository writes JSON provenance or reads a
git SHA; this is the first such code, so every git call is defensive: `git`
being missing, `cwd` not being inside a repository, or the tree being dirty
(the NORMAL state here -- CLAUDE.md's standing rule leaves .md files
routinely uncommitted) must all degrade to a recorded value rather than raise
and take down a ~40 h campaign over metadata.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import platform
import subprocess
import traceback
import uuid


def generate_runid():
    """Collision-proof id for one campaign invocation.

    The campaign is chunked per M via --M, so several invocations in close
    succession are the expected pattern, not an edge case -- a plain
    second-resolution timestamp is not enough. Microsecond-resolution UTC
    plus 8 hex chars of a uuid4 makes two runids share a value only by a
    roughly 1-in-4-billion coincidence at the very same microsecond.
    """
    ts = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')
    return '{}_{}'.format(ts, uuid.uuid4().hex[:8])


def git_provenance(cwd=None):
    """Best-effort git SHA + dirty flag for `cwd` (default: process cwd).

    Never raises. `git` not being installed, `cwd` not being inside a
    repository, or any other subprocess failure all degrade to
    {'sha': None, 'dirty': None} -- there is no code path here that can take
    down the caller. A dirty tree is recorded as `dirty: True`; it is the
    ordinary state of this repository, never treated as an error.
    """
    try:
        sha = subprocess.check_output(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd,
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode().strip()
    except Exception:
        return {'sha': None, 'dirty': None}

    try:
        status = subprocess.check_output(
            ['git', 'status', '--porcelain'], cwd=cwd,
            stderr=subprocess.DEVNULL, timeout=10,
        ).decode()
        dirty = bool(status.strip())
    except Exception:
        dirty = None

    return {'sha': sha, 'dirty': dirty}


def library_versions():
    """Versions of the libraries the training/search path actually depends
    on (train_model.py imports sklearn + optuna; dataset.py imports
    numpy + pandas). Missing a library degrades that one entry to None
    rather than failing the whole manifest."""
    versions = {'python': platform.python_version()}
    for name in ('numpy', 'pandas', 'sklearn', 'optuna'):
        try:
            module = __import__(name)
            versions[name] = getattr(module, '__version__', None)
        except Exception:
            versions[name] = None
    return versions


_NO_PREFILTER_NOTE = (
    "No candidate pre-filter is in effect for this run. The delta-derived "
    "shift_mass pre-filter (and, before it, the endpoint_ratio_cap ratio "
    "cap) was removed in P3 (Tasks 7-8): every alignment candidate that "
    "clears overlap_threshold is now presented to the accept/reject "
    "guards unfiltered. overlap_threshold itself still governs which range "
    "pairs are CANDIDATES in the first place; that is a separate concern "
    "recorded per-arm below, not a pre-filter on top of it."
)


def build_manifest(arms, M_values, n_splits, n_rows_app, n_rows_ddos, cwd=None):
    """The JSON-able dict for one campaign invocation.

    arms : list of (arm, TrainConfig) pairs, the same shape PRIMARY_ARMS /
        SENSITIVITY_ARMS use. Each is recorded as dataclasses.asdict(cfg) --
        NOT via the CSV-row label helpers (delta_align_label etc.), which
        deliberately collapse None/off/suppressed cases to '' for the row
        schema. The manifest wants the raw config: delta_align: None must
        stay distinguishable from delta_align: 0.0, which is the entire
        joint-dinf (accept every move) vs joint-d000 (accept only harmless
        moves) distinction, and asdict() does not coerce None to anything.
    """
    encoded_arms = []
    for arm, cfg in arms:
        encoding = 'joint' if arm == 'joint' else 'disjoint'
        encoded_arms.append({
            'arm': arm,
            'encoding': encoding,
            'slug': cfg.arm_slug(encoding),
            'config': asdict(cfg),
        })

    return {
        'timestamp_utc': datetime.now(timezone.utc).isoformat(),
        'git': git_provenance(cwd=cwd),
        'arms': encoded_arms,
        'M_values': list(M_values),
        'n_splits': n_splits,
        'dataset_rows': {'app': n_rows_app, 'ddos': n_rows_ddos},
        'candidate_pre_filter': {
            'active': False,
            'note': _NO_PREFILTER_NOTE,
        },
        'library_versions': library_versions(),
    }


def write_run_manifest(arms, M_values, n_splits, n_rows_app, n_rows_ddos,
                        directory=None, cwd=None):
    """Build and write results/manifests/manifest_<runid>.json for one
    invocation of compare_independent_joint_mapping.

    NOT results/: skip_existing there treats any file's existence as a
    cell-completion marker, and a manifest sitting alongside the per-(arm, M)
    CSVs could make the runner believe a cell was already computed.

    Best-effort end to end and deliberately ordered so a failure never
    leaves a partial trace: the JSON payload is built and serialised to a
    string FIRST, and the manifests directory is only created (and the file
    only written) once that has succeeded. Any failure anywhere in this --
    a value that will not serialise, a full disk, an unwritable directory,
    git being absent -- is caught, logged, and swallowed. This is provenance
    metadata about a ~40 h campaign, not the campaign itself, and must never
    be what costs the campaign its results. Returns the written path, or
    None if nothing was written.
    """
    if directory is None:
        directory = os.path.join('results', 'manifests')

    try:
        runid = generate_runid()
        manifest = build_manifest(
            arms, M_values, n_splits, n_rows_app, n_rows_ddos, cwd=cwd)
        payload = json.dumps(manifest, indent=2)

        os.makedirs(directory, exist_ok=True)
        path = os.path.join(directory, 'manifest_{}.json'.format(runid))
        tmp_path = path + '.partial'
        with open(tmp_path, 'w') as f:
            f.write(payload)
        os.replace(tmp_path, path)
        return path
    except Exception:
        print('WARNING: failed to write run manifest -- continuing without '
              'provenance for this invocation:')
        traceback.print_exc()
        return None
