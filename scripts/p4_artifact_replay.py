"""Replaying already-compiled p4c artifacts through the resource model.

Pure parsing of a generated program (p4_src/<row>.p4) and of p4c's committed
logs (compiles/<row>/pipe/logs), plus two replays that pack the result with
the model's own packer. Split out of scripts/compiler_calibration.py so that
scripts/validation_table.py can replay held-out designs without importing
sklearn or the campaign backup: this module imports only src.p4model and the
standard library. compiler_calibration re-exports every name, so existing
callers are unaffected.
"""
import os
import re

from src.p4gen.p4_replay import (  # noqa: F401
    Program, declared_prices, parse_program, replay_plans, table_tasks)


# The committed-allocation parsers live in the library now (spec 2026-09-29
# §6.2: library code never imports from scripts/). Re-exported under their
# old names so every script and test that imports them from here is unchanged.
from src.p4gen.p4_ground_truth import (  # noqa: F401
    RESOURCE_TABLE_ROW as _RESOURCE_TABLE_ROW,
    committed_blocks as _committed_blocks,
    committed_stages_real,
    committed_table_stages,
)


def _p4_table_keys(p4_path):
    """(table name -> [match key field names], field -> byte width,
    field -> BIT width) for one generated program (see p4_replay.parse_program)."""
    program = parse_program(p4_path)
    return program.tables, program.widths, program.bits


def committed_register_stages(logs_dir):
    """register base name -> the stage the compiler ran its RegisterAction in,
    from the COMMITTED allocation in mau.resources.log.

    The direct measurement behind evaluation.METER_ALUS_PER_STAGE: a stage's
    register count never exceeds 4 in any of these compiles, and the
    Percentage table reports 4 as 100% of the stage's Meter ALUs."""
    path = os.path.join(logs_dir, 'mau.resources.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    if 'Allocated Resource Usage' not in text:
        return None
    stages = {}
    for line in text.partition('Allocated Resource Usage')[2].splitlines():
        match = _RESOURCE_TABLE_ROW.match(line)
        if match and match.group(1).endswith('_reg'):
            name = match.group(1).split('.')[-1]
            stages[name[:-len('_reg')]] = int(match.group(2))
    return stages


def placement_round_states(logs_dir):
    """The placement rounds p4c actually ran, in order, e.g. ['INITIAL'] or
    ['INITIAL', 'NOCC_TRY1', 'REDO_PHV1'].

    NOCC_TRY is a container-conflicts-DISABLED re-placement of the identical
    program -- a free controlled experiment for Mechanism A, and the
    measurement its "PHV delta" column comes from. But p4c only runs it when
    the initial round leaves it something to retry, so a row with no NOCC_TRY
    round has no counterfactual at all and its delta is 0 by absence, not by
    measurement. Reading that 0 as "no PHV effect here" is what mis-attributed
    open_issues.md item 16; this function makes the distinction checkable."""
    path = os.path.join(logs_dir, 'table_summary.log')
    with open(path, encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    return re.findall(r'Table allocation done \d+ time\(s\), state = (\S+)', text)


def _latest_round_log(logs_dir, prefix):
    """The HIGHEST-numbered <prefix>_<N>.log in logs_dir.

    These logs exist once per placement round and round 0 is the DISCARDED
    initial allocation, exactly as table_summary.log's first stage count is
    (see committed_stages_real). Reading the lowest instead is what cost the
    first calibration pass its "+1 range packing" conclusion."""
    numbered = []
    for name in os.listdir(logs_dir):
        match = re.fullmatch(re.escape(prefix) + r'_(\d+)\.log', name)
        if match:
            numbered.append((int(match.group(1)), name))
    if not numbered:
        raise ValueError('no {}_<N>.log in {}'.format(prefix, logs_dir))
    return os.path.join(logs_dir, max(numbered)[1])


_PHV_ADVANCE_RE = re.compile(
    r'action dependency between (\S+) and table (\S+) due to PHV allocation '
    r'advances stage to (\d+)')


def phv_advanced_tables(logs_dir):
    """table basename -> the stage p4c's own placement log says a PHV-induced
    action dependency pushed it to (the LAST such stage, if it was pushed more
    than once).

    This is Mechanism A stated by the compiler rather than inferred: a Tofino
    stage has one action ALU per PHV CONTAINER, so two match tables whose
    actions write fields the allocator packed into one container cannot share
    a stage, and p4c logs the resulting advance verbatim:

        - action dependency between table_14_ddos_bwd_packet_length_max_0 and
          table table_13_ddos_bwd_iat_min_0 due to PHV allocation advances
          stage to 9

    The SECOND table named is the one that moves. Placement-log names carry a
    trailing `_<n>` suffix that mau.resources.log's do not (the generator's own
    table names never end in a digit -- they end in a feature name), so it is
    stripped to keep both keyed the same way."""
    with open(_latest_round_log(logs_dir, 'table_placement'),
              encoding='utf-8', errors='replace') as handle:
        text = handle.read()
    advanced = {}
    for _blocker, moved, stage in _PHV_ADVANCE_RE.findall(text):
        name = re.sub(r'_\d+$', '', moved)
        advanced[name] = max(int(stage), advanced.get(name, 0))
    return advanced


def replay_stage_depth(row_id, artifacts_root, readiness_levels=None):
    """Returns (predicted_stage_depth, committed_stages_real) for one
    already-compiled calibration row.

    Feeds the estimator's packer the REAL per-table facts from the row's own
    generated P4 and committed compile logs -- key field identities and
    widths, physical block counts, per-feature readiness levels -- so the
    only thing being compared is stage PLACEMENT. Deliberately does not
    refit the archived model: that would fold the block model's own error
    (see this module's V5) into a stage measurement.

    readiness_levels (optional) overrides the model's own per-feature levels
    with a {raw_feature_name: level} mapping, so a caller can substitute a
    different level source and see what the packer then does with it. It
    exists for one measurement: feeding in an ORACLE -- levels read off the
    compiler's own committed register placement, one stage past each feature's
    last register -- moves the predicted depth on none of the 18 rows, which
    is what killed open_issues.md item 16's readiness-accuracy hypothesis.
    Default None keeps readiness_levels_for, i.e. the real model.

    Raises ValueError when the backend produced no resource allocation,
    which is how a program that does not fit the chip shows up (see
    independent_high_sd12: "tofino supports up to 12 stages, using 13").
    Returning a plausible number there would price an infeasible design as a
    cheap one."""
    logs_dir = os.path.join(artifacts_root, 'compiles', row_id, 'pipe', 'logs')
    blocks = _committed_blocks(logs_dir)
    if blocks is None:
        raise ValueError(
            '%s: the compiler never allocated resources for this program '
            '(mau.resources.log has no "Allocated Resource Usage" section), so '
            'there is no committed placement to replay. Its table placement '
            'needs %d stages against Tofino\'s 12.'
            % (row_id, committed_stages_real(logs_dir)))

    tables, widths, bits = _p4_table_keys(
        os.path.join(artifacts_root, 'p4_src', row_id + '.p4'))
    predicted, _, _ = _replay_plans(row_id, tables, widths, bits, blocks,
                                    readiness_levels)
    return predicted, committed_stages_real(logs_dir)


def _p4_table_sizes(p4_path):
    """table name -> its declared `size = N` (see p4_replay.parse_program)."""
    return parse_program(p4_path).sizes


def replay_design(row_id, artifacts_root):
    """Returns (predicted_stage_depth, predicted_blocks) for one compiled
    row, END TO END: every table priced by the model's own per-table prices,
    placed by the model's own packer. Nothing is read from the compile except
    the program's table list, keys, entry counts and feature set -- all of
    which the model would also see before compiling. Compare replay_stage_depth,
    which feeds p4c's committed block counts in to isolate placement.

    Classification table: tables.codeword_to_blocks(key field bits) x
    tables.tree_entries_to_blocks(declared entries). Range table:
    ceil(ranges.compiler_range_rows(declared entries) / 512), the compile-time
    sizing. Works on rows p4c could not place too (no committed allocation is
    needed), which is how a held-out too-deep design is checked for
    never-under on depth."""
    p4_path = os.path.join(artifacts_root, 'p4_src', row_id + '.p4')
    tables, widths, bits = _p4_table_keys(p4_path)
    return replay_program(row_id, tables, widths, bits, _p4_table_sizes(p4_path))


def replay_program(row_id, tables, widths, bits, sizes):
    """replay_design's model half, on an already-parsed program: tables
    (name -> key fields), widths/bits (field -> crossbar bytes / declared
    bits) and sizes (table -> declared entries), as _p4_table_keys and
    _p4_table_sizes return them. Split out so a committed fixture can replay a
    design without the gitignored p4_src/ file
    (tests/fixtures/c1_replay_designs.json)."""
    program = Program(tables, widths, bits, sizes)
    plans = replay_plans(row_id, program, declared_prices(program))
    return plans.stage_depth, plans.range_plan.blocks + plans.ternary_plan.blocks



def _replay_plans(row_id, tables, widths, bits, blocks, readiness_levels):
    """Thin wrapper over p4_replay.replay_plans; returns
    (predicted_stage_depth, range_plan, ternary_plan)."""
    plans = replay_plans(row_id, Program(tables, widths, bits, {}), blocks,
                         readiness_levels)
    return plans.stage_depth, plans.range_plan, plans.ternary_plan
