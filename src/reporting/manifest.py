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
import sys
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
    "No candidate pre-filter is in effect for this run. Two heuristic "
    "pre-filters existed historically and both are gone: the original "
    "endpoint_ratio_cap ratio-based cap (removed P3 Task 7, replaced by a "
    "delta-derived shift_mass_cap veto) and that veto itself (removed P3 "
    "Task 8, because it silently discarded confirmed-harmless moves at the "
    "zero alignment tolerance). shift_mass_cap's now-dead implementation "
    "lingered as "
    "unused code after Task 8 until this cleanup campaign's own Task 6 "
    "deleted it outright. overlap_threshold itself is gone too (removed "
    "this campaign's Task 7): every alignment candidate is now admitted by "
    "three unconditional, named correctness checks in "
    "threshold_alignment.py -- still_overlaps, structurally_alignable and "
    "target_is_well_formed -- with no threshold of any kind gating "
    "admission. endpoint_ratio and calculate_range_overlap were pure "
    "diagnostics (visible via candidate_log, never compared to any "
    "threshold by admission) between P3 Task 7 and this cleanup campaign's "
    "Task 14, which deleted both outright once their only remaining role "
    "was reporting a value nothing read. shift_mass remains, still a pure "
    "diagnostic, not a filter -- do not confuse it with the removed *_cap "
    "veto mechanisms. There is no per-arm threshold recorded below "
    "(TrainConfig has no such field any more) for a reader to consult.\n\n"
    "Separately, a provenance caveat for reading the results CSV alongside "
    "this manifest: manifests recorded before commit 8065df9 (this cleanup "
    "campaign's Task 1, the joint_interval_count fix) have an accompanying "
    "results CSV whose intervals_before/intervals_after columns were "
    "computed by an incorrect tuple-union formula; manifests from 8065df9 "
    "onward accompany a CSV using the corrected pooled-threshold formula. "
    "Compare this manifest's git.sha against 8065df9 before comparing "
    "those two columns across runs from different points in the campaign."
)


def build_manifest(arms, M_values, n_splits, n_rows_app, n_rows_ddos, cwd=None):
    """The JSON-able dict for one campaign invocation.

    arms : list of (arm, TrainConfig) pairs, the same shape PRIMARY_ARMS uses.
        Each is recorded as dataclasses.asdict(cfg) -- the RAW config, not the
        CSV-row label helpers, which deliberately collapse off/suppressed cases
        to '' for the row schema.

        The sharpest case for that choice is gone: until 2026-09-15 the config
        carried `delta_align`, where None (accept every move -- joint-dinf) had
        to stay distinguishable from 0.0 (accept only harmless moves --
        joint-d000), and asdict() is what does not coerce None to anything.
        Track 5's pre-registered verdict was delta_helps = FALSE (mean_d000
        0.7956173344395895 vs mean_d020 0.7861922400433382,
        cells_favouring_d020 14/24), the field is gone, and no Optional field
        remains for an encoder to flatten. The rule stands anyway: a label is
        for the row schema, a manifest is for provenance.
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
        # Both lines go to stderr -- not just the traceback -- so the line
        # someone would actually grep a campaign log for (the identifying
        # WARNING) lands on the same stream as its detail, rather than being
        # split across stdout/stderr.
        print('WARNING: failed to write run manifest -- continuing without '
              'provenance for this invocation:', file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return None


# ---------------------------------------------------------------------------
# The compiler-verified campaign's run-level manifest (spec §5.2): ONE
# run_manifest.json per run directory, grown by every invocation that adds
# splits to that run rather than one file per invocation.
# ---------------------------------------------------------------------------

# The per-table TCAM price the run was TRAINED with (spec 2026-10-04 Sec II.4):
# 'lane' = lanes.table_blocks. results/campaign_2026_10 was trained with the
# 'ladder' (tables.codeword_to_blocks) and carries that value by hand.
BLOCK_PRICE = 'lane'

SEED_RULES = {'data': '42 + split', 'optuna': '1000 * split + k'}

# Fields that define the run's grid. A later invocation into the same run
# directory must agree on all of them, or the run would silently mix grids.
_GRID_FIELDS = ('arms', 'M_values', 'dataset_rows', 'seed_rules', 'block_price')


def env_hash():
    """sha256 of `conda list --explicit`, or None. Never raises: conda being
    absent (a plain venv, a container) is an ordinary state, not an error."""
    import hashlib
    for exe in filter(None, (os.environ.get('CONDA_EXE'), 'conda')):
        try:
            listing = subprocess.check_output(
                [exe, 'list', '--explicit'],
                stderr=subprocess.DEVNULL, timeout=120)
        except Exception:
            continue
        return hashlib.sha256(listing).hexdigest()
    return None


def _json_M(M):
    """M as it is stored in the manifest: an int, or the string 'inf' (strict
    JSON has no Infinity)."""
    return 'inf' if M == float('inf') else int(M)


def write_campaign_manifest(run_dir, arms, M_values, splits, n_rows_app, n_rows_ddos,
                            cwd=None):
    """Create or extend <run_dir>/run_manifest.json and return its content.

    The first invocation writes build_manifest(...)'s content plus run_dir,
    splits, seed_rules, env_hash, p4c_image, open_p4studio_commit and a
    one-element `batches` list. Every later invocation checks that the grid
    fields (arms, M_values, dataset_rows, seed_rules) match what is on disk
    -- raising ValueError otherwise -- then appends its own
    {splits, started_utc, git} batch and widens `splits` to the union.

    Unlike write_run_manifest this is NOT best-effort: it runs before any
    training, so a grid mismatch should stop the invocation.
    """
    from src.training.campaign_run import atomic_write_text, canonical_json, run_paths

    path = run_paths(run_dir).manifest
    splits = sorted(int(s) for s in splits)
    fresh = build_manifest(arms, [_json_M(M) for M in M_values], len(splits),
                           n_rows_app, n_rows_ddos, cwd=cwd)
    fresh['seed_rules'] = dict(SEED_RULES)
    fresh['block_price'] = BLOCK_PRICE
    # Round-trip through JSON so tuples compare equal to the lists on disk.
    fresh = json.loads(json.dumps(fresh))
    batch = {'splits': splits, 'started_utc': fresh['timestamp_utc'],
             'git': fresh['git']}

    if os.path.exists(path):
        with open(path, encoding='utf-8') as f:
            manifest = json.load(f)
        for field in _GRID_FIELDS:
            if manifest.get(field) != fresh[field]:
                raise ValueError(
                    '{}: {} differs from this invocation ({!r} on disk, {!r} now); '
                    'use a new --run directory for a different grid'.format(
                        path, field, manifest.get(field), fresh[field]))
        manifest['batches'].append(batch)
        manifest['splits'] = sorted(set(manifest['splits']) | set(splits))
        manifest['n_splits'] = len(manifest['splits'])
    else:
        manifest = fresh
        manifest.update({
            'run_dir': run_dir,
            'splits': splits,
            'env_hash': env_hash(),
            'p4c_image': os.environ.get('THESIS_P4C_IMAGE') or None,
            'open_p4studio_commit': os.environ.get('THESIS_P4STUDIO_COMMIT') or None,
            'batches': [batch],
        })

    atomic_write_text(path, canonical_json(manifest))
    return manifest
