import pytest
import p4_compile as pc


@pytest.mark.slow
def test_const_default_action_compiles_cleanly(tmp_path):
    """The const default_action = <action>(<literal>); construct compiles
    cleanly (0 errors) -- kept as the standing record of that fact.

    NOTE: generate_P4_tables_and_apply no longer emits this construct. A
    later real-compile A/B showed that although it compiles cleanly, it
    costs +1 real pipeline stage per table it is added to (+2 across all
    four classification tables, 9 -> 11) purely as a compiler
    placement artifact -- the dependency graph's own critical path length
    was unchanged -- so the default class is now installed by the control
    plane at deploy time instead (see build_p4_script.get_table_entries'
    is_default_action records and p4/deploy_table_entries.py). This test
    therefore covers resources/default_action_validation_template.p4 only,
    not any generated program."""
    result = pc.compile_p4(
        "resources/default_action_validation_template.p4", str(tmp_path / "logs"))
    assert result.errors == 0, f"const default_action construct failed to compile: check logs"
