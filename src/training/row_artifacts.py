"""Per-row artifacts of the compiler-verified campaign: the generated P4
program, the model's own breakdown of it (model.json) and the trained forests.

model.json is byte-deterministic: no timestamp and no absolute path is ever
written into it.
"""

import os
import shutil
import tempfile
import dataclasses

import joblib
import sklearn

from src.p4gen.build_p4_script import generate_P4_code
from src.p4gen.p4_replay import model_breakdown, parse_program
from src.p4model.target import MAX_CODEWORD_LENGTH, TOFINO_PIPELINE_STAGES
from src.training.campaign_run import (
    atomic_write_text, canonical_json, run_paths)


@dataclasses.dataclass(frozen=True)
class RowContext:
    run_dir: str
    arm_slug: str
    M: float
    git_commit: str


def _generate(ctx, row_id, model_app, model_ddos, names_app, names_ddos, encoding):
    """The row's program, written into a private temp dir and then moved
    atomically, so a crash never leaves a half-written designs/<row_id>.p4
    that the verifier would compile."""
    # Deferred: feature_selection imports this module, so a top-level import
    # would be circular.
    from src.training.feature_selection import (
        _derive_feature_intervals, _derive_joint_feature_intervals)
    if encoding == 'joint':
        intervals_app = intervals_ddos = _derive_joint_feature_intervals(
            model_app, model_ddos, names_app, names_ddos)
    else:
        intervals_app = _derive_feature_intervals(model_app, names_app)
        intervals_ddos = _derive_feature_intervals(model_ddos, names_ddos)
    paths = run_paths(ctx.run_dir).ensure()
    scratch = tempfile.mkdtemp(prefix=row_id + '.', dir=paths.designs)
    try:
        written = generate_P4_code(
            3, 2, model_app, model_ddos,
            feature_intervals_app=intervals_app, feature_intervals_ddos=intervals_ddos,
            output_dir=scratch + os.sep, output_filename=row_id + '.p4',
            selected_features_app=names_app, selected_features_ddos=names_ddos)
        final = os.path.join(paths.designs, row_id + '.p4')
        os.replace(written, final)
        return final
    finally:
        shutil.rmtree(scratch, ignore_errors=True)


def _save_forests(paths, row_id, model_app, model_ddos, names_app, names_ddos):
    payload = {'app': model_app, 'ddos': model_ddos,
               'features_app': list(names_app), 'features_ddos': list(names_ddos),
               'sklearn': sklearn.__version__}
    final = os.path.join(paths.forests, row_id + '.joblib')
    tmp = final + '.partial'
    joblib.dump(payload, tmp, compress=3)
    os.replace(tmp, final)


def load_forests(path):
    stored = joblib.load(path)
    if stored.get('sklearn') != sklearn.__version__:
        raise RuntimeError(
            f"{path} was saved with scikit-learn {stored.get('sklearn')}, "
            f"but this environment runs {sklearn.__version__}")
    return stored


def write_row_artifacts(ctx, row_id, model_app, model_ddos, names_app, names_ddos,
                        encoding, usage, extra=None):
    """`extra`: a mapping merged into model.json after every computed field
    (the alignment twins' `source_row_id` / `identical_to`); it may not
    override a computed key."""
    paths = run_paths(ctx.run_dir).ensure()
    budgeted = ctx.M != float('inf')
    model = {
        'row_id': row_id,
        'git_commit': ctx.git_commit,
        'M': int(ctx.M) if budgeted else None,
        'budgeted': budgeted,
        'training_stage_depth': int(usage.stage_depth),
        'training_blocks': int(usage.blocks),
        'stages': int(usage.stages),
        'register_depth': int(usage.register_depth),
        'register_count': int(usage.register_count),
        'range_entries': int(usage.range_entries),
        'ternary_entries': int(usage.ternary_entries),
        'codeword_bits': int(usage.codeword_length),
    }
    try:
        p4_path = _generate(ctx, row_id, model_app, model_ddos,
                            names_app, names_ddos, encoding)
    except ValueError as exc:
        message = str(exc)
        atomic_write_text(os.path.join(paths.designs, row_id + '.generator_error.txt'),
                          message)
        model.update({'stage_depth': None, 'blocks': None, 'tables': None,
                      'model_paths_differ': None, 'hw_feasible': False,
                      'budget_feasible': False, 'generator_error': message})
    else:
        replay = model_breakdown(parse_program(p4_path), row_id)
        model.update({
            'stage_depth': replay['stage_depth'],
            'blocks': replay['blocks'],
            'tables': replay['tables'],
            'model_paths_differ': (replay['stage_depth'], replay['blocks'])
                                  != (usage.stage_depth, usage.blocks),
            'hw_feasible': bool(replay['stage_depth'] <= TOFINO_PIPELINE_STAGES
                                and model['codeword_bits'] <= MAX_CODEWORD_LENGTH),
            'budget_feasible': (not budgeted) or replay['blocks'] <= ctx.M,
            'generator_error': None,
        })
    if extra:
        clash = set(extra) & set(model)
        if clash:
            raise ValueError(f'extra may not override model.json keys {sorted(clash)}')
        model.update(extra)
    _save_forests(paths, row_id, model_app, model_ddos, names_app, names_ddos)
    atomic_write_text(os.path.join(paths.designs, row_id + '.model.json'),
                      canonical_json(model))
    return model


def write_trial_table(ctx, row_id, rows):
    """trials/<row_id>.csv: every trial of the row's search with b/c against
    the reference (trial_selection.trial_rows), so the selection rule can be
    re-run offline at any select_alpha. Written atomically."""
    import csv
    import io
    paths = run_paths(ctx.run_dir).ensure()
    buffer = io.StringIO()
    if rows:
        writer = csv.DictWriter(buffer, fieldnames=list(rows[0]), lineterminator='\n')
        writer.writeheader()
        writer.writerows(rows)
    path = os.path.join(paths.trials, row_id + '.csv')
    atomic_write_text(path, buffer.getvalue())
    return path
