"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法或頂層
區塊名未知 → 直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓人流
統計以非預期的口徑產出而無人察覺（例如把 `[zone]` 拼成 `[zones]`，去抖與時段
粒度整段悄悄退回預設）。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`_get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from zone_mapping.models.config import (
    AppConfig,
    _get_toml_path,
    find_project_root,
    load_config,
)

# 設定來源含環境變數，且欄位名未加前綴：執行環境剛好有這些變數時會蓋掉 toml 的值，
# 讓測試結果取決於誰的機器在跑。逐一清掉，測的才是「從這份 toml 載入」的行為。
_ENV_OVERRIDES = ("INPUT", "ZONE", "INPUT__BUCKET_DIR", "INPUT__DATE") + (
    "ZONE__BUCKET_MINUTES",
    "ZONE__ENTRY_DEBOUNCE_FRAMES",
)


@pytest.fixture(autouse=True)
def _clear_config_env(monkeypatch):
    for name in _ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def _config_class(toml_path) -> type[AppConfig]:
    """建立一個讀指定 toml 的 AppConfig 子類別。"""

    class _ScopedConfig(AppConfig):
        # extra="forbid" 明寫出來：test_unknown_top_level_section_raises 靠的就是它，
        # 繼承自父類別的話，日後有人改父類別會讓那支測試無聲失去守護對象。
        model_config = SettingsConfigDict(
            toml_file=str(toml_path),
            env_nested_delimiter="__",
            extra="forbid",
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

    assert config.zone.bucket_minutes == 60
    assert config.zone.entry_debounce_frames == 1
    assert config.input.bucket_dir == "bucket_name"


def test_reads_values_from_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[input]\nbucket_dir = "bucket_x"\ndate = 2026-05-01\n'
        "[zone]\nbucket_minutes = 30\nentry_debounce_frames = 3\n",
        encoding="utf-8",
    )

    config = _config_class(toml)()

    assert config.input.bucket_dir == "bucket_x"
    assert config.zone.bucket_minutes == 30
    assert config.zone.entry_debounce_frames == 3


def test_invalid_value_in_toml_raises_instead_of_silently_defaulting(tmp_path):
    """設定檔存在但值不合法時必須報錯——靜默套預設值會讓統計口徑被悄悄改掉。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[zone]\nbucket_minutes = 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_unknown_top_level_section_raises(tmp_path):
    """區塊名打錯（如 [zones]）要報錯，不可被靜默忽略而套用預設值。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[zones]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_load_config_warns_when_toml_missing(monkeypatch, capsys):
    """找不到設定檔時要留下警告，不可靜默啟動。"""
    monkeypatch.setattr(
        "zone_mapping.models.config._get_toml_path", lambda: "/nonexistent/config.toml"
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
