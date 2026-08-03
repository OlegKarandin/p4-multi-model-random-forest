"""Central config for this project's opt-in P4-generation parameters.

Scope deliberately narrow (confirmed with the project owner, 2026-08-03):
only validate_on_hardware/hardware_output_dir, use_default_action_discount,
and match_type live here. main.py's model/training constants and this
project's hardware/toolchain constants do NOT belong in this file -- see
this plan's Task 4 description for why each was excluded.
"""
from dataclasses import dataclass
from typing import Optional


@dataclass
class P4GenConfig:
    validate_on_hardware: bool = False
    hardware_output_dir: Optional[str] = None
    use_default_action_discount: bool = False
    match_type: str = 'ternary'

    def __post_init__(self):
        if self.match_type not in ('ternary', 'exact'):
            raise ValueError(
                "match_type must be 'ternary' or 'exact', got {!r}".format(self.match_type))
        if self.validate_on_hardware and not self.hardware_output_dir:
            raise ValueError(
                "hardware_output_dir must be set when validate_on_hardware=True")

    @classmethod
    def from_dict(cls, d: dict) -> "P4GenConfig":
        return cls(**d)
