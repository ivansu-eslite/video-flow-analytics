"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法或頂層
區塊名未知 → 直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓推理
以非預期的口徑產出而無人察覺（例如把 `[model]` 拼成 `[models]`，classes 過濾整段
悄悄退回預設）。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`_get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict

from video_analyze.models.config import (
    AppConfig,
    ModelConfig,
    TrackerConfig,
    _get_toml_path,
    find_project_root,
    load_config,
)

# 設定來源含環境變數，且欄位名未加前綴：執行環境剛好有這些變數時會蓋掉 toml 的值，
# 讓測試結果取決於誰的機器在跑。逐一清掉，測的才是「從這份 toml 載入」的行為。
_ENV_OVERRIDES = (
    "TRACKER",
    "MODEL",
    "OUTPUT",
    "INPUT",
    "MODEL__CLASSES",
    "MODEL__BATCH",
    "MODEL__MODEL_PATH",
    "TRACKER__TRACK_BUFFER",
    "INPUT__BUCKET_DIR",
    "INPUT__DATE",
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


def test_model_config_classes_defaults_to_fbody():
    assert ModelConfig().classes == [2]


def test_model_config_classes_rejects_empty_list():
    with pytest.raises(ValidationError):
        ModelConfig(classes=[])


def test_tracker_config_defaults_unaffected():
    assert TrackerConfig().track_buffer == 30


def test_find_project_root_locates_package_root():
    """find_project_root 取代寫死的 parents[N]，須能定位到含 pyproject.toml 的套件根。"""
    root = find_project_root(Path(__file__).resolve())
    assert root is not None
    assert (root / "pyproject.toml").exists()


def test_get_toml_path_points_to_existing_config():
    path = _get_toml_path()
    assert path is not None
    assert path.endswith("config.toml")
    assert Path(path).exists()


def test_uses_defaults_when_toml_missing(tmp_path):
    """找不到設定檔時以預設值啟動，而非中止。"""
    config = _config_class(tmp_path / "nope.toml")()

    assert config.model.classes == [2]
    assert config.tracker.track_buffer == 30
    assert config.input.bucket_dir == "bucket_name"
    assert config.output.save_video is False


def test_reads_values_from_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[model]\nmodel_path = "some_model.pt"\nbatch = 4\nclasses = [0, 2]\n'
        "[input]\nbucket_dir = \"bucket_x\"\ndate = 2026-05-01\n",
        encoding="utf-8",
    )

    config = _config_class(toml)()

    assert config.model.classes == [0, 2]
    assert config.model.batch == 4
    assert config.input.bucket_dir == "bucket_x"


def test_invalid_value_in_toml_raises_instead_of_silently_defaulting(tmp_path):
    """設定檔存在但值不合法時必須報錯——靜默套預設值會讓推理口徑被悄悄改掉。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[model]\nclasses = []\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_unknown_top_level_section_raises(tmp_path):
    """區塊名打錯（如 [models]）要報錯，不可被靜默忽略而套用預設值。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[models]\nbatch = 4\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_load_config_warns_when_toml_missing(monkeypatch, capsys):
    """找不到設定檔時要留下警告，不可靜默啟動。"""
    monkeypatch.setattr(
        "video_analyze.models.config._get_toml_path",
        lambda: "/nonexistent/config.toml",
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
