"""Behaviour of p4model/packing.py: the column geometry and the stage packer
(`crossbar_stages_needed`)."""
from p4model import packing


def test_a_table_wider_than_one_column_never_fits():
    # A table chains its blocks down ONE 12-row column (scripts/
    # tcam_column_sweep.py, Mechanism C), so a 13-block shard fits no stage;
    # _stage_shards splits such tables before they reach fits_two_columns.
    assert not packing.fits_two_columns([13])
    assert packing.fits_two_columns([12])
