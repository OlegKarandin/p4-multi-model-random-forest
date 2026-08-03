import pytest
import p4_compile as pc


@pytest.mark.slow
def test_const_default_action_compiles_cleanly(tmp_path):
    """The const default_action = <action>(<literal>); construct Task 1's
    generate_P4_tables_and_apply now emits has never been through the real
    Tofino compiler -- only asserted as a Python string in fast tests. This
    compiles a minimal standalone program using the same construct and
    confirms 0 errors."""
    result = pc.compile_p4(
        "resources/default_action_validation_template.p4", str(tmp_path / "logs"))
    assert result.errors == 0, f"const default_action construct failed to compile: check logs"
