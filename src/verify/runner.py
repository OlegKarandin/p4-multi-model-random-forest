"""The resumable p4c verifier (spec 2026-09-29 §6): compile every design of a
campaign run, keep a small tarball of p4c's logs, classify model vs p4c.

Resume rule: `verify/<row_id>.json` is written atomically and LAST, so it is the
only done marker. A leftover `verify/<row_id>.json.partial`, a half-written
tarball or a compile temp dir never counts as done; the next run redoes the row.
`verification.csv` is always rebuilt from the per-row JSON files.

Tarball layout (`verify/<row_id>.tar.gz`), shared by the writer and `rescore`:
    pipe/logs/<name>                 every KEPT_LOGS file p4c produced
    pipe/logs/table_placement_<N>.log  the highest-numbered placement round
    pipe/prog.bfa
    p4c_output.txt                   failed compiles only: p4c's captured output
so extracting it into a directory gives `p4c_numbers(<dir>/pipe/logs)` the same
input the original compile output did.
"""
import csv
import datetime
import functools
import hashlib
import io
import json
import os
import re
import shutil
import statistics
import tarfile
import tempfile
import time
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from src.p4gen.p4_compile import P4CompileTimeout, compile_p4
from src.p4gen.p4_ground_truth import (
    committed_blocks, committed_stages_real, p4c_numbers)
from src.p4gen.p4_replay import model_breakdown, parse_program
from src.p4model.target import MAX_CODEWORD_LENGTH, TOFINO_PIPELINE_STAGES
from src.reporting.manifest import git_provenance
from src.training.campaign_run import (
    atomic_write_bytes, atomic_write_text, canonical_json, run_paths)
from src.verify.verdicts import classify

KEPT_LOGS = ('table_summary.log', 'mau.resources.log', 'phv_allocation_summary_0.log',
             'table_dependency_summary.log', 'metrics.json', 'pragmas.log')

# verification.csv's columns, in order, and exactly the keys of every
# verify/<row_id>.json (spec §6.5, plus the O3 training-path fields, `M` and
# `failure`, the machine-readable reason behind a COMPILE_ERROR/TIMEOUT:
# generator_error | p4c_timeout | p4c_toolchain | p4c_errors, else None).
VERIFICATION_COLUMNS = (
    'row_id', 'p4_sha256', 'M',
    'model_stage_depth', 'model_blocks', 'model_hw_feasible', 'model_budget_feasible',
    'model_training_stage_depth', 'model_training_blocks', 'model_paths_differ',
    'p4c_stage_depth', 'p4c_blocks', 'p4c_sram', 'p4c_map_ram', 'p4c_phv_containers',
    'p4c_errors', 'p4c_warnings', 'compile_seconds',
    'verdict', 'p4c_over_budget', 'p4c_over_stages', 'unverified', 'failure',
    'tables_differing',
    'p4c_image', 'open_p4studio_commit', 'model_git_commit', 'verifier_git_commit',
    'verified_utc',
)

_FAILURE_VERDICT = {'generator_error': 'COMPILE_ERROR', 'p4c_toolchain': 'COMPILE_ERROR',
                    'p4c_errors': 'COMPILE_ERROR', 'p4c_timeout': 'TIMEOUT'}
_PLACEMENT_LOG = re.compile(r'^table_placement_(\d+)\.log$')


@functools.lru_cache(maxsize=1)
def _verifier_commit():
    return git_provenance().get('sha')


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat(timespec='seconds')


def _read_json(path):
    with open(path, encoding='utf-8') as handle:
        return json.load(handle)


def _sha256(path):
    if not os.path.isfile(path):
        return None
    with open(path, 'rb') as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _M_value(model):
    """model.json's M (None for an unbudgeted row) as classify wants it."""
    return float('inf') if model.get('M') is None else model['M']


def _M_column(model):
    return 'inf' if model.get('M') is None else model['M']


# ---------------------------------------------------------------- tarball


def _kept_files(output_dir):
    """(source path, arcname) for every file the tarball keeps."""
    logs = os.path.join(output_dir, 'pipe', 'logs')
    kept = []
    for name in KEPT_LOGS:
        path = os.path.join(logs, name)
        if os.path.isfile(path):
            kept.append((path, 'pipe/logs/' + name))
    rounds = []
    if os.path.isdir(logs):
        for name in os.listdir(logs):
            match = _PLACEMENT_LOG.match(name)
            if match:
                rounds.append((int(match.group(1)), name))
    if rounds:
        name = max(rounds)[1]
        kept.append((os.path.join(logs, name), 'pipe/logs/' + name))
    bfa = os.path.join(output_dir, 'pipe', 'prog.bfa')
    if os.path.isfile(bfa):
        kept.append((bfa, 'pipe/prog.bfa'))
    return kept


def _write_tarball(path, output_dir, p4c_output=None):
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode='w:gz') as tar:
        for source, arcname in _kept_files(output_dir):
            tar.add(source, arcname=arcname)
        if p4c_output is not None:
            data = p4c_output.encode('utf-8', errors='replace')
            info = tarfile.TarInfo('p4c_output.txt')
            info.size = len(data)
            tar.addfile(info, io.BytesIO(data))
    atomic_write_bytes(path, buffer.getvalue())


def _p4c_from_tarball(path):
    """P4cNumbers re-parsed from a kept tarball, or None when there is none."""
    if not os.path.isfile(path):
        return None
    with tempfile.TemporaryDirectory() as tmp:
        with tarfile.open(path) as tar:
            tar.extractall(tmp, filter='data')
        return p4c_numbers(os.path.join(tmp, 'pipe', 'logs'))


# ---------------------------------------------------------------- one row


def _compile_with_retry(compile_fn, p4_path, output_dir, timeout, retry_timeout):
    """(result, failure, message, seconds of the last attempt). One retry,
    with the long timeout, on a timeout or toolchain failure; the second
    failure decides TIMEOUT vs COMPILE_ERROR."""
    for attempt, limit in enumerate((timeout, retry_timeout)):
        if os.path.exists(output_dir):  # p4c wants to create output_dir itself
            shutil.rmtree(output_dir)
        start = time.monotonic()
        try:
            result = compile_fn(p4_path, output_dir, timeout_seconds=limit)
        except P4CompileTimeout as exc:
            failure, message = 'p4c_timeout', str(exc)
        except RuntimeError as exc:
            failure, message = 'p4c_toolchain', str(exc)
        else:
            return result, None, None, time.monotonic() - start
        seconds = time.monotonic() - start
    return None, failure, message, seconds


def _record(model, row_id, p4_path, verdict, p4c=None, result=None, failure=None,
            compile_seconds=None, provenance=None):
    provenance = provenance or {}
    return {
        'row_id': row_id,
        'p4_sha256': _sha256(p4_path),
        'M': _M_column(model),
        'model_stage_depth': model.get('stage_depth'),
        'model_blocks': model.get('blocks'),
        'model_hw_feasible': model.get('hw_feasible'),
        'model_budget_feasible': model.get('budget_feasible'),
        'model_training_stage_depth': model.get('training_stage_depth'),
        'model_training_blocks': model.get('training_blocks'),
        'model_paths_differ': model.get('model_paths_differ'),
        'p4c_stage_depth': None if p4c is None else p4c.stage_depth,
        'p4c_blocks': None if p4c is None else p4c.blocks,
        'p4c_sram': None if p4c is None else p4c.sram,
        'p4c_map_ram': None if p4c is None else p4c.map_ram,
        'p4c_phv_containers': None if p4c is None else p4c.phv_containers,
        'p4c_errors': provenance.get('p4c_errors',
                                     None if result is None else result.errors),
        'p4c_warnings': provenance.get('p4c_warnings',
                                       None if result is None else result.warnings),
        'compile_seconds': (None if compile_seconds is None
                            else round(compile_seconds, 3)),
        'verdict': verdict.verdict,
        'p4c_over_budget': verdict.p4c_over_budget,
        'p4c_over_stages': verdict.p4c_over_stages,
        'unverified': verdict.unverified,
        'failure': failure,
        'tables_differing': list(verdict.tables_differing),
        'p4c_image': provenance.get('p4c_image', os.environ.get('THESIS_P4C_IMAGE')),
        'open_p4studio_commit': provenance.get(
            'open_p4studio_commit', os.environ.get('THESIS_P4STUDIO_COMMIT')),
        'model_git_commit': model.get('git_commit'),
        'verifier_git_commit': _verifier_commit(),
        'verified_utc': provenance.get('verified_utc', _utc_now()),
    }


def _load_model(paths, row_id):
    path = os.path.join(paths.designs, row_id + '.model.json')
    if not os.path.isfile(path):
        raise FileNotFoundError(
            f"{row_id}: feasible row has no designs/{row_id}.model.json "
            "(training wrote the CSV row but not its artifacts)")
    return _read_json(path)


def _generator_failed(paths, row_id, model):
    return (os.path.isfile(os.path.join(paths.designs, row_id + '.generator_error.txt'))
            or model.get('generator_error') is not None)


def verify_row(run_dir, row_id, compile_fn=None, timeout=300, retry_timeout=1800):
    compile_fn = compile_fn or compile_p4
    paths = run_paths(run_dir).ensure()
    model = _load_model(paths, row_id)
    p4_path = os.path.join(paths.designs, row_id + '.p4')
    M = _M_value(model)

    # Generator error FIRST: the model fields are null, so classify must not
    # see this row, and there is no program to compile.
    if _generator_failed(paths, row_id, model):
        verdict = classify(model, None, M, 'COMPILE_ERROR')
        record = _record(model, row_id, p4_path, verdict, failure='generator_error')
    else:
        with tempfile.TemporaryDirectory(prefix='verify_' + row_id + '_') as tmp:
            output_dir = os.path.join(tmp, 'out')
            result, failure, message, seconds = _compile_with_retry(
                compile_fn, p4_path, output_dir, timeout, retry_timeout)
            if failure is None and (result.errors or 0) > 0 and not os.path.isfile(
                    os.path.join(output_dir, 'pipe', 'logs', 'table_summary.log')):
                failure, message = 'p4c_errors', result.output
            p4c = p4c_numbers(os.path.join(output_dir, 'pipe', 'logs'))
            verdict = classify(model, p4c, M, _FAILURE_VERDICT.get(failure))
            _write_tarball(os.path.join(paths.verify, row_id + '.tar.gz'), output_dir,
                           p4c_output=message if failure else None)
        record = _record(model, row_id, p4_path, verdict, p4c, result, failure, seconds)

    atomic_write_text(os.path.join(paths.verify, row_id + '.json'), canonical_json(record))
    return record


# ---------------------------------------------------------------- run level


def _feasible_csv_rows(paths):
    ids = set()
    if not os.path.isdir(paths.rows):
        return ids
    for name in sorted(os.listdir(paths.rows)):
        if not name.endswith('.csv'):
            continue
        with open(os.path.join(paths.rows, name), newline='', encoding='utf-8') as handle:
            for row in csv.DictReader(handle):
                if row.get('row_id') and not (row.get('infeasible') or '').strip():
                    ids.add(row['row_id'])
    return ids


def pending_rows(run_dir):
    paths = run_paths(run_dir)
    designed = set()
    if os.path.isdir(paths.designs):
        designed = {name[:-len('.model.json')] for name in os.listdir(paths.designs)
                    if name.endswith('.model.json')}
    # A feasible CSV row with no model.json is kept pending so verify_row
    # reports it as an error on every run, never skipped silently.
    wanted = designed | _feasible_csv_rows(paths)
    return sorted(row_id for row_id in wanted
                  if not os.path.isfile(os.path.join(paths.verify, row_id + '.json')))


def _csv_cell(value):
    if isinstance(value, list):
        return json.dumps(value, sort_keys=True, separators=(',', ':'))
    return '' if value is None else value


def merge_verification(run_dir):
    paths = run_paths(run_dir).ensure()
    records = [_read_json(os.path.join(paths.verify, name))
               for name in os.listdir(paths.verify) if name.endswith('.json')]
    records.sort(key=lambda r: r['row_id'])
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator='\n')
    writer.writerow(VERIFICATION_COLUMNS)
    for record in records:
        writer.writerow([_csv_cell(record.get(col)) for col in VERIFICATION_COLUMNS])
    atomic_write_text(paths.verification_csv, buffer.getvalue())
    return paths.verification_csv


def run(run_dir, workers=1, compile_fn=None, min_free_bytes=5 * 1024 ** 3):
    """Verify every pending row. Returns the exit code: 0 all done, 1 some row
    raised (reported, left pending), 2 stopped on low disk."""
    compile_fn = compile_fn or compile_p4
    run_paths(run_dir).ensure()
    todo = pending_rows(run_dir)
    stopped, errors = False, []
    in_flight = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        while todo or in_flight:
            while todo and len(in_flight) < workers and not stopped:
                free = shutil.disk_usage(run_dir).free
                if free < min_free_bytes:
                    print(f"stopped: free disk {free / 1024 ** 3:.1f} GB < "
                          f"{min_free_bytes / 1024 ** 3:g} GB; restart resumes", flush=True)
                    stopped = True
                    break
                row_id = todo.pop(0)
                in_flight[pool.submit(verify_row, run_dir, row_id, compile_fn)] = row_id
            if stopped:
                todo = []
            if not in_flight:
                break
            done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
            for future in done:
                row_id = in_flight.pop(future)
                try:
                    record = future.result()
                except Exception as exc:  # reported, row stays pending
                    errors.append(row_id)
                    print(f"error: {row_id}: {type(exc).__name__}: {exc}", flush=True)
                else:
                    print(f"{row_id}: {record['verdict']}", flush=True)
    merge_verification(run_dir)
    if stopped:
        return 2
    return 1 if errors else 0


def _rescored_model(paths, row_id, model):
    """model.json with its model side recomputed by the current code from the
    saved program. Falls back to model.json as stored when the program is
    missing. model.json itself is never rewritten (it is byte-deterministic
    training output)."""
    p4_path = os.path.join(paths.designs, row_id + '.p4')
    if _generator_failed(paths, row_id, model) or not os.path.isfile(p4_path):
        return model
    fresh = model_breakdown(parse_program(p4_path), row_id)
    model = dict(model)
    model.update({
        'stage_depth': fresh['stage_depth'],
        'blocks': fresh['blocks'],
        'tables': fresh['tables'],
        'model_paths_differ': (fresh['stage_depth'], fresh['blocks'])
                              != (model.get('training_stage_depth'),
                                  model.get('training_blocks')),
        'hw_feasible': bool(fresh['stage_depth'] <= TOFINO_PIPELINE_STAGES
                            and (model.get('codeword_bits') or 0) <= MAX_CODEWORD_LENGTH),
        'budget_feasible': model.get('M') is None or fresh['blocks'] <= model['M'],
    })
    return model


def rescore(run_dir):
    """Reclassify every verified row with the current model code and the
    p4c numbers re-parsed from its tarball. Never compiles. Keeps the
    compile's own provenance (p4c image, open-p4studio commit, verified_utc,
    p4c errors/warnings, compile_seconds); verifier_git_commit becomes the
    rescoring commit."""
    paths = run_paths(run_dir).ensure()
    for name in sorted(os.listdir(paths.verify)):
        if not name.endswith('.json'):
            continue
        old = _read_json(os.path.join(paths.verify, name))
        row_id = old['row_id']
        model = _rescored_model(paths, row_id, _load_model(paths, row_id))
        failure = old.get('failure')
        if failure == 'p4c_errors' or failure not in _FAILURE_VERDICT:
            p4c = _p4c_from_tarball(os.path.join(paths.verify, row_id + '.tar.gz'))
        else:
            p4c = None
        verdict = classify(model, p4c, _M_value(model), _FAILURE_VERDICT.get(failure))
        record = _record(model, row_id, os.path.join(paths.designs, row_id + '.p4'),
                         verdict, p4c, None, failure, old.get('compile_seconds'),
                         provenance={k: old.get(k) for k in (
                             'p4c_errors', 'p4c_warnings', 'p4c_image',
                             'open_p4studio_commit', 'verified_utc')})
        atomic_write_text(os.path.join(paths.verify, name), canonical_json(record))
    return merge_verification(run_dir)


# ---------------------------------------------------------------- archive


def _stages_or_none(logs_dir):
    try:
        return committed_stages_real(logs_dir)
    except (OSError, AttributeError):
        return None


def _blocks_or_none(logs_dir):
    try:
        return committed_blocks(logs_dir)
    except OSError:
        return None


def _verify_archived_design(archive_dir, row, compile_fn):
    p4_path = os.path.join(archive_dir, 'p4_src', row + '.p4')
    stored_logs = os.path.join(archive_dir, 'compiles', row, 'pipe', 'logs')
    with tempfile.TemporaryDirectory(prefix='archive_' + row + '_') as tmp:
        output_dir = os.path.join(tmp, 'out')
        start = time.monotonic()
        compile_fn(p4_path, output_dir, timeout_seconds=1800)
        seconds = time.monotonic() - start
        new_logs = os.path.join(output_dir, 'pipe', 'logs')
        new_stages, new_blocks = _stages_or_none(new_logs), _blocks_or_none(new_logs)
    old_stages, old_blocks = _stages_or_none(stored_logs), _blocks_or_none(stored_logs)
    differing = []
    if new_blocks is None or old_blocks is None:
        if (new_blocks is None) != (old_blocks is None):
            differing.append({'table': '<allocation>', 'new': new_blocks is not None,
                              'archived': old_blocks is not None})
    else:
        for table in sorted(set(new_blocks) | set(old_blocks)):
            if new_blocks.get(table) != old_blocks.get(table):
                differing.append({'table': table, 'new': new_blocks.get(table),
                                  'archived': old_blocks.get(table)})
    return {'row': row, 'stages_new': new_stages, 'stages_archived': old_stages,
            'tables_differing': differing,
            'match': new_stages == old_stages and not differing,
            'compile_seconds': round(seconds, 3)}


def verify_archive(archive_dir, workers=1, compile_fn=None):
    """Recompile every p4_src/<row>.p4 of a calibration archive and compare
    committed stages and per-table blocks with compiles/<row>/pipe/logs (the
    environment acceptance test, spec §8.2). Prints one line per design, the
    total and the median compile time; the caller exits 1 on any mismatch."""
    compile_fn = compile_fn or compile_p4
    rows = sorted(name[:-3] for name in os.listdir(os.path.join(archive_dir, 'p4_src'))
                  if name.endswith('.p4'))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(
            lambda row: _verify_archived_design(archive_dir, row, compile_fn), rows))
    for r in results:
        line = (f"{r['row']}: {'OK' if r['match'] else 'DIFF'} stages "
                f"{r['stages_new']} (archived {r['stages_archived']})")
        for t in r['tables_differing']:
            line += f"; {t['table']} {t['new']} vs archived {t['archived']}"
        print(line)
    matched = sum(r['match'] for r in results)
    print(f"total: {matched}/{len(results)} match")
    if results:
        print(f"median compile seconds: "
              f"{statistics.median(r['compile_seconds'] for r in results):.1f}")
    return results
