"""Two compiler-log readers the TCAM probe scripts share: `read_committed`
(a compile's committed resources.json, per table: blocks, ternary crossbar
bytes, stage) and `measured_start_group` (the crossbar group a key's run
begins at, from its lowest byte).

History. This module was the 12-compile
out-of-sample test of `tables.version_block_penalty` (2026-09-07): arms on the
saturation edge, clause (d) and the start offset. The penalty was deleted
with the offset-taking `codeword_to_blocks` on 2026-09-21 (2026-09-20 rewrite
design Sec 13.1), which left the sweep's `predict()` / `run_point()` / `main()`
calling a name that no longer exists. They were removed on 2026-11-13 (see git
history); the sweep's data lives in the gitignored
results/tcam_version_sweep.csv, which tests/test_tcam_block_ledger.py and
scripts/tcam_table_scoreboard.py read directly. Only the two readers below,
which many probe scripts import, remain.
"""
import json
import os

TERNARY_XBAR_BASE = 128        # ternary crossbar bytes are numbered from here
PAIR_BYTES = 11                # 5 private + 1 midbyte + 5 private


def read_committed(logs_dir):
    """{table -> {'blocks': n, 'xbar_bytes': [relative byte numbers]}} from the
    compiler's own resources.json -- the same file the archived compiles were
    re-scored from, so this sweep and that scoring read the identical quantity."""
    path = os.path.join(logs_dir, 'resources.json')
    if not os.path.exists(path):
        return {}
    with open(path, encoding='utf-8') as handle:
        doc = json.load(handle)
    out = {}
    for stage in doc['resources']['mau']['mau_stages']:
        for unit in stage['tcams']['tcams']:
            for use in unit['usages']:
                rec = out.setdefault(use['used_by'].split('.')[-1],
                                     {'blocks': 0, 'xbar_bytes': set(),
                                      'stage': stage['stage_number']})
                rec['blocks'] += 1
        for byte in stage['xbar_bytes']['bytes']:
            if byte.get('byte_type') != 'ternary':
                continue
            for use in byte['usages']:
                name = use['used_by'].split('.')[-1]
                if name in out:
                    out[name]['xbar_bytes'].add(
                        byte['byte_number'] - TERNARY_XBAR_BASE)
    for rec in out.values():
        rec['xbar_bytes'] = sorted(rec['xbar_bytes'])
    return out


def measured_start_group(xbar_bytes):
    """The crossbar group a key's run begins at, from its lowest byte.

    A pair of groups spans PAIR_BYTES bytes as 5 private, 1 midbyte, 5 private,
    so relative byte r sits in group 2*(r//11) when r%11 < 5 and 2*(r//11)+1
    when r%11 > 5. A byte at r%11 == 5 IS the midbyte and belongs to both, so
    it is reported as ambiguous rather than guessed."""
    if not xbar_bytes:
        return None, None
    low = min(xbar_bytes)
    pair, within = divmod(low, PAIR_BYTES)
    if within < 5:
        return 2 * pair, False
    if within == 5:
        return 2 * pair, True          # shared midbyte: could be either group
    return 2 * pair + 1, False

