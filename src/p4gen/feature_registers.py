"""Moved to src/p4model/catalog.py. Kept as a re-export so build_p4_script,
evaluation and tests/test_feature_registers.py keep their existing import
paths. Explicit names, not `import *`."""
from src.p4model.catalog import (
    FEATURE_REGISTER_CATALOG,
    register_names_for,
    register_width_bits,
)
