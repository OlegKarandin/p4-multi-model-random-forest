from src.p4gen import p4_gen_config as cfg


def test_p4_gen_config_defaults_match_pre_existing_behavior():
    """Every field's default must reproduce today's behavior exactly when
    no config is loaded -- this config file introduces zero new default
    behavior, only a single place to set the three flags together."""
    c = cfg.P4GenConfig()
    assert c.validate_on_hardware is False
    assert c.hardware_output_dir is None
    assert c.use_default_action_discount is False
    assert c.match_type == 'ternary'


def test_p4_gen_config_from_dict_overrides_defaults():
    c = cfg.P4GenConfig.from_dict({
        "validate_on_hardware": True,
        "hardware_output_dir": "p4/hw_validation/",
        "use_default_action_discount": True,
        "match_type": "exact",
    })
    assert c.validate_on_hardware is True
    assert c.hardware_output_dir == "p4/hw_validation/"
    assert c.use_default_action_discount is True
    assert c.match_type == "exact"


def test_p4_gen_config_rejects_unknown_match_type():
    import pytest
    with pytest.raises(ValueError, match="match_type"):
        cfg.P4GenConfig(match_type="bogus")


def test_p4_gen_config_requires_output_dir_when_validation_enabled():
    import pytest
    with pytest.raises(ValueError, match="hardware_output_dir"):
        cfg.P4GenConfig(validate_on_hardware=True, hardware_output_dir=None)
