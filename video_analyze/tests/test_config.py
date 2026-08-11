"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法或頂層
區塊名未知 → 直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓推理
以非預期的口徑產出而無人察覺（例如把 `[model]` 拼成 `[models]`，classes 過濾整段
悄悄退回預設）。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

from pathlib import Path

import pytest
import vfa_config
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict
from vfa_config import get_toml_path

from video_analyze.models import config as config_module
from video_analyze.models.config import (
    AppConfig,
    FootPointConfig,
    ModelConfig,
    TrackerConfig,
    load_config,
)

# 設定來源含環境變數，且欄位名未加前綴：執行環境剛好有這些變數時會蓋掉 toml 的值，
# 讓測試結果取決於誰的機器在跑。逐一清掉，測的才是「從這份 toml 載入」的行為。
_ENV_OVERRIDES = (
    "TRACKER",
    "MODEL",
    "OUTPUT",
    "INPUT",
    "FOOT_POINT",
    "FOOT_POINT__METHOD",
    "MODEL__CLASSES",
    "MODEL__BATCH",
    "MODEL__MODEL_PATH",
    "TRACKER__TRACK_BUFFER",
    "INPUT__BUCKET_DIR",
    "INPUT__DATE",
    "INPUT__CAMERA_IDS",
    "INPUT__BUCKET_MINUTES",
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


def test_model_config_classes_defaults_to_head_and_fbody():
    """head 也要偵測：它不進 tracker，只供落腳點推算（見 ADR-009）。"""
    assert ModelConfig().classes == [0, 2]


def test_model_config_classes_rejects_empty_list():
    with pytest.raises(ValidationError):
        ModelConfig(classes=[])


def test_tracker_config_defaults_unaffected():
    assert TrackerConfig().track_buffer == 30


def test_config_sections_match_shared_contract():
    """釘住本包宣告了哪些頂層區塊、`input` 來自共用 lib（見 ADR-008）。

    頂層區塊名等於一段全域的環境變數命名空間（`env_nested_delimiter="__"`），所以
    新增區塊會讓這支變紅，逼使用者確認該名稱在其他包沒被用過；把 `input` 改回本地
    定義同樣變紅——那正是 `INPUT__*` 撞名的來源。
    """
    assert set(AppConfig.model_fields) == {
        "tracker",
        "model",
        "foot_point",
        "output",
        "input",
    }
    assert AppConfig.model_fields["input"].annotation is vfa_config.InputConfig


@pytest.mark.parametrize(
    ("env_name", "env_value"),
    [
        ("INPUT__BUCKET_DIR", "bucket_x"),
        ("INPUT__DATE", "2026-05-01"),
        ("INPUT__CAMERA_IDS", '["cam1"]'),
        ("INPUT__BUCKET_MINUTES", "30"),
    ],
)
def test_shared_input_env_vars_do_not_break_this_package(
    env_name, env_value, monkeypatch, tmp_path
):
    """任一 `INPUT__*` 都不能讓本包載入失敗，包含本包不讀的欄位。

    四包共用一份環境設定執行是常態，而本模組在載入時就 `load_config()`；`[input]`
    少列一個別包有的欄位，設了該變數的環境會讓本包連 import 都失敗（`bucket_minutes`
    只有另外三包讀，過去正是這樣打死本包）。
    """
    monkeypatch.setenv(env_name, env_value)
    toml = tmp_path / "config.toml"
    toml.write_text("[model]\nbatch = 4\n", encoding="utf-8")

    config = _config_class(toml)()

    assert config.model.batch == 4


def test_get_toml_path_points_to_this_packages_config():
    """本包的 config 模組要定位到**自己**的 `config.toml`，不是共用 lib 的目錄。

    `get_toml_path` 抽進 `vfa_config` 後靠呼叫端傳入 `__file__`；傳錯（例如共用 lib
    自己的檔案）會讓四包都去讀不存在的 `libs/vfa_config/config.toml`，全部靜默退回
    預設參數。
    """
    path = get_toml_path(config_module.__file__)

    assert path is not None
    assert Path(path).exists()
    assert Path(path) == Path(__file__).resolve().parents[1] / "config.toml"


def test_uses_defaults_when_toml_missing(tmp_path):
    """找不到設定檔時以預設值啟動，而非中止。"""
    config = _config_class(tmp_path / "nope.toml")()

    assert config.model.classes == [0, 2]
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


def test_foot_point_method_defaults_to_head():
    assert FootPointConfig().method == "head"


def test_foot_point_method_rejects_unknown_value():
    with pytest.raises(ValidationError):
        FootPointConfig(method="pose")


def test_head_method_without_head_class_is_rejected(tmp_path):
    """`method="head"` 卻沒偵測 head，每列都會退回框底邊中點——改動靜默失效，須擋下。"""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[model]\nclasses = [2]\n[foot_point]\nmethod = \"head\"\n", encoding="utf-8"
    )

    with pytest.raises(ValidationError, match="head"):
        _config_class(toml)()


def test_bbox_bottom_method_without_head_class_is_allowed(tmp_path):
    """切回舊算法時不需要 head，classes 只留 fbody 是合法組合。"""
    toml = tmp_path / "config.toml"
    toml.write_text(
        "[model]\nclasses = [2]\n[foot_point]\nmethod = \"bbox_bottom\"\n",
        encoding="utf-8",
    )

    config = _config_class(toml)()

    assert config.foot_point.method == "bbox_bottom"
    assert config.model.classes == [2]


def test_classes_without_fbody_is_rejected(tmp_path):
    """少了追蹤目標，整天的 parquet 會是空的，卻不會有任何錯誤訊息。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[model]\nclasses = [0]\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="fbody"):
        _config_class(toml)()


def test_unknown_top_level_section_raises(tmp_path):
    """區塊名打錯（如 [models]）要報錯，不可被靜默忽略而套用預設值。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[models]\nbatch = 4\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_unknown_field_in_nested_section_raises(tmp_path):
    """巢狀區塊內欄位名打錯要報錯（如 [model] 多一個未知欄位），不可被靜默忽略而
    套用預設值（巢狀 model 的 extra="forbid"，非僅頂層區塊名）。"""
    toml = tmp_path / "config.toml"
    toml.write_text('[model]\nmodelpath = "x.pt"\n', encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_load_config_warns_when_toml_missing(monkeypatch, capsys):
    """找不到設定檔時要留下警告，不可靜默啟動。"""
    monkeypatch.setattr(
        "video_analyze.models.config.get_toml_path",
        lambda _caller_file: "/nonexistent/config.toml",
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
