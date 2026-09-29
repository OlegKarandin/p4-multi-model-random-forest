"""The generator's code_* pins, recomputed from each archived program's key
structure, must equal the pins injected into the 73-design evidence archive
(results/compiler_calibration_pragmas_2026_09_29/manifest.json). Spec
2026-09-29 Sec 6 step 2."""
import os

import pytest

from scripts import verify_pragma_generation as vpg

pytestmark = pytest.mark.skipif(
    not os.path.isfile(os.path.join(vpg.ARCHIVE, "manifest.json")),
    reason="needs results/compiler_calibration_pragmas_2026_09_29 (gitignored)")


def test_generator_pins_equal_the_injected_pins_on_all_73_designs():
    assert vpg.check_archive(vpg.ARCHIVE) == []
