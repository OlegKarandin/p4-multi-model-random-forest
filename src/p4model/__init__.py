"""A calibrated surrogate for p4c's Tofino backend.

Predicts TCAM block usage and pipeline stage depth for the table topology this
project's generator emits, without invoking the compiler. Imports stdlib only:
no sklearn, no numpy, no pandas, no CWD-relative file reads -- see
tests/test_p4model_guards.py, which enforces exactly that."""
from src.p4model.errors import CodewordTooLong, CrossbarKeyTooWide
# packing.py is NOT re-exported here (2026-09-21, TCAM block model rewrite
# Task 2): it imports version_block_delta and the offset-taking
# codeword_to_blocks, both retired from tables.py by this task, and Task 4
# owns packing.py's own repair (do not touch packing.py itself to fix this --
# see that task's report). Nothing in this codebase imported StagePlan,
# crossbar_stages_needed or fits_two_columns from this top-level package path
# (every real caller goes through `from src.p4model import packing` or
# `from src.p4model.packing import ...` directly, which still raises the same
# ImportError packing.py's own broken import always would), so dropping this
# line changes no real caller's behaviour -- it only stops packing.py's
# EXPECTED breakage from also taking down every other p4model submodule via
# this package's __init__.
from src.p4model.ranges import range_entry_count
from src.p4model.registers import (
    gated_block_interior_stages,
    readiness_levels_for,
    register_stage_schedule,
)
from src.p4model.names import normalise_feature_name
from src.p4model.tables import (
    range_matching_resource_usage,
    ternary_matching_resource_usage,
)
from src.p4model.usage import ResourceUsage, assemble_usage
