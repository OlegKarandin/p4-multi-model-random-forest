"""Campaign identity helpers and run layout (spec §4, §5.1-§5.2, §5.4).

Provides row identity, filename tokens, and CSV parsing for the compiler-verified
campaign. Supports deterministic campaign execution via seeded Optuna and atomic
file writes for crash safety.
"""

import dataclasses
import json
import os
from pathlib import Path


INF = float('inf')
DEFAULT_M_GRID = (15, 25, 35, 50, 75, INF)
DEVELOPMENT_SPLITS = frozenset({10, 11, 12, 13})


def m_token(M):
    """The filename/row_id token for a TCAM budget: zero-padded to 3 digits so
    lexical order is numeric order, or 'inf' for the unbudgeted cell (spec §4)."""
    if M == INF:
        return 'inf'
    if M != int(M) or not 0 <= M < 1000:
        raise ValueError('M must be an integer in [0, 1000) or inf, got {!r}'.format(M))
    return '{:03d}'.format(int(M))


def row_id(arm_slug, M, split, k):
    """Return the row identity string for this (arm, M, split, k) tuple.

    Format: {arm_slug}_M{m_token}_s{split:02d}_k{k:02d}
    Example: joint_M035_s07_k09
    """
    return '{}_{}_s{:02d}_k{:02d}'.format(arm_slug, 'M' + m_token(M), split, k)


def split_csv_name(n_trees, max_depth, M, arm_slug, split):
    """Return the CSV filename for one (arm, M, split).

    Format: rf_t{n_trees}_d{max_depth}_M{m_token}_{arm_slug}_s{split:02d}.csv
    Example: rf_t7_d14_M035_joint_s07.csv
    """
    return 'rf_t{}_d{}_M{}_{}_s{:02d}.csv'.format(
        n_trees, max_depth, m_token(M), arm_slug, split
    )


def parse_splits(text):
    """Parse splits string: '0-9' or '0,3,5' or a mix like '0-2,7'.

    Returns sorted, unique list of integers.
    Raises ValueError on empty input, negative split, reversed range, or non-integer.
    """
    if not text:
        raise ValueError('--splits: empty input')

    splits = set()
    try:
        for part in text.split(','):
            if '-' in part:
                # Range
                lo_str, hi_str = part.split('-')
                lo = int(lo_str)
                hi = int(hi_str)
                if lo > hi:
                    raise ValueError('--splits: reversed range {}-{}'.format(lo, hi))
                if lo < 0 or hi < 0:
                    raise ValueError('--splits: negative split in range {}-{}'.format(lo, hi))
                splits.update(range(lo, hi + 1))
            else:
                # Single value
                val = int(part)
                if val < 0:
                    raise ValueError('--splits: negative split {}'.format(val))
                splits.add(val)
    except ValueError as e:
        # Re-raise if already a splits error
        if 'splits' in str(e):
            raise
        # Otherwise wrap the parsing error
        raise ValueError('--splits: {}'.format(text))

    return sorted(splits)


def parse_M_grid(text):
    """Parse M grid string: '15,25,inf' gives [15, 25, inf].

    Raises ValueError on M <= 0 and on any value m_token cannot name
    (non-integers such as 15.5, values >= 1000, 'INF'), so a bad budget fails
    at parse time instead of inside row_id hours into a run.
    """
    result = []
    try:
        for part in text.split(','):
            part = part.strip()
            if part == 'inf':
                result.append(INF)
            else:
                M = float(part)
                if M <= 0:
                    raise ValueError('M must be > 0, got {}'.format(M))
                m_token(M)
                result.append(int(M))
    except OverflowError:
        raise ValueError('--M: {!r} (only lowercase inf names the unbudgeted cell)'.format(text))
    except ValueError as e:
        if 'must be' in str(e):
            raise
        raise ValueError('--M-grid: {}'.format(text))
    return list(dict.fromkeys(result))  # a repeated M names the same cell


def optuna_seed(split, k):
    """Return the Optuna seed for this (split, k).

    Formula: 1000 * split + k
    The seed does not depend on arm or M (common random numbers).
    """
    return 1000 * split + k


@dataclasses.dataclass(frozen=True)
class RunPaths:
    """Layout for a campaign run directory.

    Attributes:
        root: Run directory root (e.g., results/campaign_2026_10/)
        rows: Directory for per-split CSV files
        designs: Directory for .p4 and .model.json files
        forests: Directory for joblib forest files
        trials: Directory for per-row trial tables
        verify: Directory for .tar.gz verification outputs
        verification_csv: Path to verification.csv
        manifest: Path to run_manifest.json
    """
    root: str
    rows: str
    designs: str
    forests: str
    trials: str
    verify: str
    verification_csv: str
    manifest: str

    def ensure(self):
        """Create all directories. Returns self."""
        for attr in ('rows', 'designs', 'forests', 'trials', 'verify'):
            Path(getattr(self, attr)).mkdir(parents=True, exist_ok=True)
        return self


def run_paths(run_dir):
    """Return a RunPaths for the given run directory."""
    root = run_dir
    return RunPaths(
        root=root,
        rows=os.path.join(root, 'rows'),
        designs=os.path.join(root, 'designs'),
        forests=os.path.join(root, 'forests'),
        trials=os.path.join(root, 'trials'),
        verify=os.path.join(root, 'verify'),
        verification_csv=os.path.join(root, 'verification.csv'),
        manifest=os.path.join(root, 'run_manifest.json'),
    )


def canonical_json(obj):
    """Serialize to JSON with sorted keys, 1-space indent, and trailing newline."""
    return json.dumps(obj, sort_keys=True, indent=1) + '\n'


def atomic_write_bytes(path, data):
    """Write bytes atomically: write to .partial, then os.replace.

    Creates parent directories as needed.
    """
    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)
    partial_path = str(path) + '.partial'
    with open(partial_path, 'wb') as f:
        f.write(data)
    os.replace(partial_path, path)


def atomic_write_text(path, text):
    """Write text atomically: write to .partial, then os.replace.

    Creates parent directories as needed.
    """
    atomic_write_bytes(path, text.encode('utf-8'))
