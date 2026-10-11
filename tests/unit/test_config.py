import copy

import pytest
import yaml

from beacon.config import ConfigError, load_config, parse_config, require_run_ready


def test_example_config_is_valid(raw_config):
    cfg = parse_config(raw_config)
    assert cfg.privacy.blur_kernel % 2 == 1
    assert cfg.lock.mode == "dry_run"


def test_load_config_from_file(tmp_path, raw_config):
    p = tmp_path / "c.yaml"
    p.write_text(yaml.safe_dump(raw_config))
    assert load_config(p).camera.index == 0


def test_missing_file_message(tmp_path):
    with pytest.raises(ConfigError, match="config.example.yaml"):
        load_config(tmp_path / "nope.yaml")


def test_invalid_yaml(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("camera: [unclosed")
    with pytest.raises(ConfigError, match="invalid YAML"):
        load_config(p)


def _bad(raw, sec, key, value):
    raw = copy.deepcopy(raw)
    raw[sec][key] = value
    return raw


@pytest.mark.parametrize(
    "sec,key,value,msg",
    [
        ("privacy", "blur_kernel", 30, "odd"),
        ("privacy", "blur_kernel", -3, ">= 3"),
        ("privacy", "blur_kernel", 31.0, "integer"),
        ("privacy", "blur_sigma", 0, "> 0"),
        ("privacy", "blur_sigma", -1.0, "> 0"),
        ("privacy", "blur_sigma", "6", "number"),
        ("privacy", "save_debug_frames", True, "must stay false"),
        ("camera", "index", True, "integer"),
        ("camera", "read_timeout_s", 0, "> 0"),
        ("trust", "lock_below", 0.6, "lock_below"),
        ("trust", "lock_below", 0.9, "lock_below"),
        ("trust", "trusted_above", 0.5, "hysteresis"),
        ("trust", "trusted_above", 1.0, "unreachable"),
        ("trust", "multi_face_policy", "ignore", "one of"),
        ("trust", "tau_rise_s", 0, "> 0"),
        ("lock", "mode", "yes", "one of"),
        ("lock", "require_cli_flag", False, "must be true"),
        ("identity", "similarity_threshold", 1.5, "<= 1.0"),
        ("identity", "min_count_face_px", 500, "min_count_face_px"),
        ("liveness", "spoof_lock_count", 0, ">= 1"),
        ("models", "detector", "opencv", "one of"),
        ("enrollment", "collapse_similarity", 0.1, "min_template_similarity"),
    ],
)
def test_invalid_values_rejected(raw_config, sec, key, value, msg):
    with pytest.raises(ConfigError, match=msg) as ei:
        parse_config(_bad(raw_config, sec, key, value))
    assert f"{sec}" in str(ei.value)


def test_nan_rejected(raw_config):
    with pytest.raises(ConfigError, match="finite"):
        parse_config(_bad(raw_config, "trust", "initial", float("nan")))


def test_missing_field_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    del raw["trust"]["tau_absent_s"]
    with pytest.raises(ConfigError, match="trust.tau_absent_s: required"):
        parse_config(raw)


def test_missing_section_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    del raw["lock"]
    with pytest.raises(ConfigError, match="lock: required section"):
        parse_config(raw)


def test_unknown_key_rejected(raw_config):
    raw = _bad(raw_config, "trust", "tau_typo", 1.0)
    with pytest.raises(ConfigError, match="unknown key"):
        parse_config(raw)


def test_unknown_section_rejected(raw_config):
    raw = copy.deepcopy(raw_config)
    raw["extras"] = {}
    with pytest.raises(ConfigError, match="unknown section"):
        parse_config(raw)


def test_null_threshold_refuses_run(raw_config):
    cfg = parse_config(_bad(raw_config, "identity", "similarity_threshold", None))
    with pytest.raises(ConfigError, match="must be calibrated"):
        require_run_ready(cfg)


def test_threshold_present_allows_run(config):
    assert require_run_ready(config) == pytest.approx(0.40)
