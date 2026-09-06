"""A calibrated surrogate for p4c's Tofino backend.

Predicts TCAM block usage and pipeline stage depth for the table topology this
project's generator emits, without invoking the compiler. Imports stdlib only:
no sklearn, no numpy, no pandas, no CWD-relative file reads -- see
tests/test_p4model_guards.py, which enforces exactly that."""
