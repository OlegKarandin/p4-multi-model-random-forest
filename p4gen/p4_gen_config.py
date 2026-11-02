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
    # match_type='exact' (Planter RF_EB-style exact-match/SRAM code/decision
    # tables) is DEFERRED, not abandoned: it does not yet produce a loadable
    # program, because enumerating this project's real feature intervals into
    # concrete exact-match entries comes out to ~1.3x10**34 entries -- no real
    # switch's SRAM could hold that. See reviews/todo.md:343-349 (2026-08-03
    # decision to defer) and .superpowers/plans/2026-08-03-p4-generator-fixes-
    # and-config.md Task 2 (guarded Cartesian-product enumeration, recorded
    # there as on hold pending this decision).
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
