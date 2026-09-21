"""A calibrated surrogate for p4c's Tofino backend.

Predicts TCAM block usage and pipeline stage depth for the table topology this
project's generator emits, without invoking the compiler. Imports stdlib only:
no sklearn, no numpy, no pandas, no CWD-relative file reads -- see
tests/test_p4model_guards.py, which enforces exactly that."""
from src.p4model.errors import CodewordTooLong, CrossbarKeyTooWide
from src.p4model.packing import StagePlan, crossbar_stages_needed, fits_two_columns
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
