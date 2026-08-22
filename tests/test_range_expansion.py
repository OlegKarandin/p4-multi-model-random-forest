from p4 import range_expansion
from src.p4gen import evaluation as ev
import pytest


def test_range_entry_count_reproduces_10_300_worked_example():
    # reviews/cited_papers/tofino_results_2.odt.pdf slide 11: matching
    # [10,300] on a 16-bit field needs exactly 4 physical TCAM entries
    # under Tofino's nibble-based range decomposition (vs. 10 under a
    # naive bit-level decomposition). This is the exact algorithm
    # (expand_range() ported from bf-drivers), not an approximation.
    assert range_expansion.range_entry_count(10, 300) == 4


def test_range_entry_count_point_entry_is_one():
    assert range_expansion.range_entry_count(5, 5) == 1


def test_range_entry_count_full_16_bit_range_is_one():
    # [0,65535] is exactly the full 16-bit address space -- nibble-aligned
    # at every level -- so it costs exactly one physical row, same as any
    # other power-of-2-aligned block (see the [0,255] case below).
    assert range_expansion.range_entry_count(0, 65535) == 1


def test_range_entry_count_aligned_power_of_two_block_is_one():
    # A power-of-2-aligned range always costs exactly one physical row
    # (RM-8's own empirical finding, reviews/t12_tcam_model_experiment_plan.md
    # Section 11.3) -- confirmed here from the exact algorithm directly.
    assert range_expansion.range_entry_count(0, 255) == 1


def test_range_entry_count_rejects_hi_less_than_lo():
    with pytest.raises(ValueError):
        range_expansion.range_entry_count(10, 5)


def test_evaluation_range_entry_count_is_the_same_object_as_range_expansion():
    # evaluation.py and p4/deploy_table_entries.py both need this function
    # (the latter from inside bfshell's sklearn-less embedded Python), and
    # used to carry independent ~40-line copies that could silently drift.
    # Asserting `is`, not just equal behaviour, proves evaluation.py now
    # imports the one shared implementation rather than keeping its own copy.
    assert ev.range_entry_count is range_expansion.range_entry_count
