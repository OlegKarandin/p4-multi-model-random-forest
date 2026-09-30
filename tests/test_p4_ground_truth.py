"""p4c's COMMITTED allocation, parsed without scripts/ (spec 2026-09-29 §6.2).

Fixtures are inline text in the real files' shapes (copied from the pragmas
archive's logs), so these run without results/. The archive-backed test at the
bottom checks the PHV count against the 73 archived compiles when present."""
import json
import os

import pytest

from src.p4gen import p4_ground_truth as gt

_RESOURCES = """\
Allocated Resource Usage
| Table Name | Stage | a | b | c | d | TCAM |
| ingress.table_0_flow_iat_max | 3 | 0 | 0 | 0 | 0 | 1 |
| ingress.table_0_flow_iat_max$action | 3 | 0 | 0 | 0 | 0 | 0 |
| ingress.get_classification_tree_app_0 | 7 | 0 | 0 | 0 | 0 | 5 |
"""

_SUMMARY_TWO_ROUNDS = """Table allocation done 1 time(s), state = INITIAL
Number of stages in table allocation: 11
Table allocation done 2 time(s), state = REDO_PHV1
Number of stages in table allocation: 12
"""

_METRICS = {"phv": {
    "normal": [{"bit_width": 8, "containers_occupied": 16},
               {"bit_width": 16, "containers_occupied": 29},
               {"bit_width": 32, "containers_occupied": 14}],
    "tagalong": [{"bit_width": 8, "containers_occupied": 2}]}}


def _logs(tmp_path, resources=None, summary=None, metrics=None):
    logs = tmp_path / "pipe" / "logs"
    logs.mkdir(parents=True)
    if resources is not None:
        (logs / "mau.resources.log").write_text(resources)
    if summary is not None:
        (logs / "table_summary.log").write_text(summary)
    if metrics is not None:
        (logs / "metrics.json").write_text(json.dumps(metrics))
    return str(logs)


def test_committed_blocks_and_stages_skip_action_rows(tmp_path):
    logs = _logs(tmp_path, resources=_RESOURCES)
    assert gt.committed_blocks(logs) == {"table_0_flow_iat_max": 1,
                                         "get_classification_tree_app_0": 5}
    assert gt.committed_table_stages(logs) == {"table_0_flow_iat_max": 3,
                                               "get_classification_tree_app_0": 7}


def test_no_allocation_section_means_none_not_zero(tmp_path):
    # A program over 12 stages never reaches allocation (independent_high_sd12).
    logs = _logs(tmp_path, resources="tofino supports up to 12 stages, using 13\n")
    assert gt.committed_blocks(logs) is None
    assert gt.committed_table_stages(logs) is None


def test_committed_stages_reads_the_last_round(tmp_path):
    logs = _logs(tmp_path, summary=_SUMMARY_TWO_ROUNDS)
    assert gt.committed_stages_real(logs) == 12


def test_phv_containers_counts_normal_containers_only(tmp_path):
    # 16 + 29 + 14 = 59, the pragmas archive manifest's phv_containers for
    # heldout_independent_M100_k10_s19; tagalong containers are not counted.
    assert gt.phv_containers(_logs(tmp_path, metrics=_METRICS)) == 59


def test_p4c_numbers_on_a_compile_that_never_allocated(tmp_path):
    logs = _logs(tmp_path, resources="no allocation\n", summary=_SUMMARY_TWO_ROUNDS)
    numbers = gt.p4c_numbers(logs)
    assert numbers.stage_depth == 12
    assert numbers.blocks is None and numbers.table_blocks is None
    assert not numbers.allocated


def test_p4c_numbers_on_an_empty_dir_is_all_none(tmp_path):
    numbers = gt.p4c_numbers(_logs(tmp_path))
    assert numbers == gt.P4cNumbers(None, None, None, None, None, None)


def test_p4c_numbers_sums_committed_blocks(tmp_path):
    numbers = gt.p4c_numbers(_logs(tmp_path, resources=_RESOURCES,
                                   summary=_SUMMARY_TWO_ROUNDS, metrics=_METRICS))
    assert (numbers.stage_depth, numbers.blocks, numbers.phv_containers) == (12, 6, 59)
    assert numbers.allocated


def test_the_script_still_exports_the_old_names():
    from scripts import p4_artifact_replay as R
    assert R._committed_blocks is gt.committed_blocks
    assert R.committed_table_stages is gt.committed_table_stages
    assert R.committed_stages_real is gt.committed_stages_real


_ARCHIVE = os.path.join("results", "compiler_calibration_pragmas_2026_09_29")


@pytest.mark.skipif(not os.path.isdir(_ARCHIVE), reason="gitignored evidence archive")
def test_phv_and_totals_match_the_pragmas_archive_manifest():
    with open(os.path.join(_ARCHIVE, "manifest.json"), encoding="utf-8") as handle:
        designs = json.load(handle)["designs"]
    for design in designs:
        logs = os.path.join(_ARCHIVE, "compiles", design["row_id"], "pipe", "logs")
        numbers = gt.p4c_numbers(logs)
        want = design["p4c_with_pragmas"]
        assert numbers.phv_containers == want["phv_containers"], design["row_id"]
        assert numbers.stage_depth == want["stages_real"], design["row_id"]
