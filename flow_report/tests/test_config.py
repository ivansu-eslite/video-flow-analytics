"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法 →
直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓報表以非預期的
口徑產出而無人察覺。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`_get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

import os
from pathlib import Path
from unittest import mock

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from flow_report.models.config import (
    AppConfig,
    _get_toml_path,
    find_project_root,
    load_config,
)


def _config_class(toml_path) -> type[AppConfig]:
    """建立一個讀指定 toml 的 AppConfig 子類別。"""

    class _ScopedConfig(AppConfig):
        model_config = SettingsConfigDict(
            toml_file=str(toml_path),
            env_nested_delimiter="__",
        )

    return _ScopedConfig


def test_find_project_root_locates_package_root():
    """find_project_root 取代寫死的 parents[N]，須能定位到含 pyproject.toml 的套件根。"""
    root = find_project_root(Path(__file__).resolve())
    assert root is not None
    assert (root / "pyproject.toml").exists()


def test_get_toml_path_points_to_existing_config():
    path = _get_toml_path()
    assert path is not None
    assert path.endswith("config.toml")


def test_uses_defaults_when_toml_missing(tmp_path):
    """找不到設定檔時以預設值啟動，而非中止。"""
    config = _config_class(tmp_path / "nope.toml")()

    assert config.report.period_minutes == 60
    assert config.report.metric == "entries"
    assert config.input.bucket_minutes == 60


def test_reads_values_from_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[input]\nbucket_dir = "bucket_x"\ndate = 2026-05-01\nbucket_minutes = 30\n'
        '[report]\nperiod_minutes = 90\nmetric = "unique_visitors"\n',
        encoding="utf-8",
    )

    config = _config_class(toml)()

    assert config.input.bucket_dir == "bucket_x"
    assert config.input.bucket_minutes == 30
    assert config.report.period_minutes == 90
    assert config.report.metric == "unique_visitors"


def test_legacy_zone_section_reports_where_bucket_minutes_moved(tmp_path):
    """沿用舊 `[zone] bucket_minutes` 的設定檔要報出新位置。

    `extra="forbid"` 本來就會擋下 `[zone]`，但只說「不允許額外欄位」；沿用舊設定
    的人需要知道該把這個數值搬到哪裡，否則只能去翻 source。
    """
    toml = tmp_path / "config.toml"
    toml.write_text("[zone]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="bucket_minutes 改放 \\[input\\]"):
        _config_class(toml)()


def test_legacy_zone_env_var_warns_but_still_loads(tmp_path, capsys):
    """`ZONE__BUCKET_MINUTES` 只警告，不能讓本包載入失敗。

    這個變數對本包是靜默忽略的覆寫（pydantic-settings 的 env source 只查已知欄位名，
    它不會進到模型、也不會被 `extra="forbid"` 擋下），值得提醒；但它同時是
    `zone_mapping` 的合法設定，四包共用一份環境設定是常態，拋錯等於讓別的套件的正常
    設定把報表階段打死——而且本模組在載入時就 `load_config()`，連 import 都會失敗。
    """
    toml = tmp_path / "config.toml"
    toml.write_text("[input]\nbucket_minutes = 60\n", encoding="utf-8")

    with mock.patch.dict(os.environ, {"ZONE__BUCKET_MINUTES": "30"}):
        config = _config_class(toml)()

    out = capsys.readouterr().out
    assert config.input.bucket_minutes == 60
    assert "WARNING" in out
    assert "INPUT__BUCKET_MINUTES" in out


def test_invalid_value_in_toml_raises_instead_of_silently_defaulting(tmp_path):
    """設定檔存在但值不合法時必須報錯——靜默套預設值會讓報表口徑被悄悄改掉。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[report]\nperiod_minutes = 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_invalid_metric_in_toml_raises(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text('[report]\nmetric = "bogus"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_unknown_field_in_nested_section_raises(tmp_path):
    """巢狀區塊內欄位名打錯要報錯：period_minutes 少一個底線寫成 periodminutes，不可被
    靜默忽略而退回預設（巢狀 model 的 extra="forbid"，非僅頂層區塊名）。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[report]\nperiodminutes = 90\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_load_config_warns_when_toml_missing(monkeypatch, capsys):
    """找不到設定檔時要留下警告，不可靜默啟動。"""
    monkeypatch.setattr(
        "flow_report.models.config._get_toml_path", lambda: "/nonexistent/config.toml"
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
