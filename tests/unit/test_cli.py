"""CLI paths that must refuse or clean up without loading any model."""

import numpy as np
import pytest
import yaml

from beacon.__main__ import main
from beacon.enrollment import build_profile, save_profile

from ..conftest import USER_EMB


@pytest.fixture
def cfg_file(tmp_path, raw_config):
    raw = dict(raw_config)
    raw["storage"] = {"profiles_dir": str(tmp_path / "profiles"), "results_dir": str(tmp_path / "results")}
    raw["logging"] = {"level": "INFO", "file": str(tmp_path / "logs" / "beacon.log")}
    p = tmp_path / "config.yaml"
    p.write_text(yaml.safe_dump(raw))
    return p, raw


def _profile(config, tmp_path):
    rng = np.random.default_rng(0)
    p = build_profile("user_a", [USER_EMB + rng.normal(0, 0.05, 16) for _ in range(20)], config)
    save_profile(p, tmp_path / "profiles")


def test_run_refuses_null_threshold(tmp_path, cfg_file, capsys):
    path, raw = cfg_file
    raw["identity"]["similarity_threshold"] = None
    path.write_text(yaml.safe_dump(raw))
    assert main(["--config", str(path), "run", "--user", "user_a"]) == 2
    assert "must be calibrated" in capsys.readouterr().err


def test_run_refuses_without_profile(cfg_file, capsys):
    path, _ = cfg_file
    assert main(["--config", str(path), "run", "--user", "user_a"]) == 1
    assert "no enrolled profile" in capsys.readouterr().err


def test_run_refuses_profile_with_other_blur(tmp_path, cfg_file, config, capsys):
    path, raw = cfg_file
    _profile(config, tmp_path)
    raw["privacy"]["blur_sigma"] = 5.0
    path.write_text(yaml.safe_dump(raw))
    assert main(["--config", str(path), "run", "--user", "user_a"]) == 1
    assert "different blur" in capsys.readouterr().err


def test_delete_then_run_refuses(tmp_path, cfg_file, config, capsys):
    path, _ = cfg_file
    _profile(config, tmp_path)
    assert main(["--config", str(path), "show-profile", "--user", "user_a"]) == 0
    assert '"n_samples": 20' in capsys.readouterr().out
    assert main(["--config", str(path), "delete-profile", "--user", "user_a", "--yes"]) == 0
    assert not (tmp_path / "profiles" / "user_a.profile").exists()
    assert main(["--config", str(path), "run", "--user", "user_a"]) == 1
    assert "no enrolled profile" in capsys.readouterr().err


def test_delete_asks_for_confirmation(tmp_path, cfg_file, config, monkeypatch):
    path, _ = cfg_file
    _profile(config, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    assert main(["--config", str(path), "delete-profile", "--user", "user_a"]) == 1
    assert (tmp_path / "profiles" / "user_a.profile").exists()


def test_reset_deletes_only_app_data(tmp_path, cfg_file, config):
    path, _ = cfg_file
    _profile(config, tmp_path)
    (tmp_path / "logs").mkdir()
    (tmp_path / "logs" / "beacon.log").write_text("x")
    (tmp_path / "logs" / "unrelated.txt").write_text("keep me")
    (tmp_path / "results" / "raw").mkdir(parents=True)
    (tmp_path / "results" / "raw" / "t.csv").write_text("x")
    (tmp_path / "results" / "metrics.json").write_text("{}")
    (tmp_path / "profiles" / "notes.txt").write_text("keep me")
    assert main(["--config", str(path), "reset", "--all", "--yes"]) == 0
    assert not (tmp_path / "profiles" / "user_a.profile").exists()
    assert not (tmp_path / "logs" / "beacon.log").exists()
    assert not (tmp_path / "results" / "raw").exists()
    assert (tmp_path / "logs" / "unrelated.txt").exists()
    assert (tmp_path / "profiles" / "notes.txt").exists()
    assert (tmp_path / "results" / "metrics.json").exists()


def test_reset_requires_confirmation(tmp_path, cfg_file, config, monkeypatch):
    path, _ = cfg_file
    _profile(config, tmp_path)
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert main(["--config", str(path), "reset", "--all"]) == 1
    assert (tmp_path / "profiles" / "user_a.profile").exists()


def test_enroll_requires_consent(tmp_path, cfg_file, monkeypatch, capsys):
    path, _ = cfg_file
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    assert main(["--config", str(path), "enroll", "--user", "user_a"]) == 1
    assert "nothing was captured" in capsys.readouterr().out


def test_bad_config_is_reported(tmp_path, capsys):
    p = tmp_path / "c.yaml"
    p.write_text("camera: {}\n")
    assert main(["--config", str(p), "run", "--user", "u"]) == 2
    assert "configuration error" in capsys.readouterr().err


def test_enroll_accepts_preview_flag(tmp_path, cfg_file, monkeypatch, capsys):
    path, _ = cfg_file
    monkeypatch.setattr("builtins.input", lambda *a: "no")
    assert main(["--config", str(path), "enroll", "--user", "user_a", "--preview"]) == 1
    assert "nothing was captured" in capsys.readouterr().out

