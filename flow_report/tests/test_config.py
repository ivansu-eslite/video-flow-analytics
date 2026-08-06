"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法 →
直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓報表以非預期的
口徑產出而無人察覺。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

from pathlib import Path

import pytest
import vfa_config
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict
from vfa_config import get_toml_path

from flow_report.models import config as config_module
from flow_report.models.config import AppConfig, load_config

# 設定來源含環境變數，且欄位名未加前綴：執行環境剛好有這些變數時會蓋掉 toml 的值，
# 讓測試結果取決於誰的機器在跑。逐一清掉，測的才是「從這份 toml 載入」的行為。
_ENV_OVERRIDES = (
    "INPUT",
    "REPORT",
    "INPUT__BUCKET_DIR",
    "INPUT__DATE",
    "INPUT__CAMERA_IDS",
    "INPUT__BUCKET_MINUTES",
    "REPORT__PERIOD_MINUTES",
    "REPORT__METRIC",
    "REPORT__ON_DUPLICATE_DATE",
)


@pytest.fixture(autouse=True)
def _clear_config_env(monkeypatch):
    for name in _ENV_OVERRIDES:
        monkeypatch.delenv(name, raising=False)


def _config_class(toml_path) -> type[AppConfig]:
    """建立一個讀指定 toml 的 AppConfig 子類別。"""

    class _ScopedConfig(AppConfig):
        model_config = SettingsConfigDict(
            toml_file=str(toml_path),
            env_nested_delimiter="__",
        )

    return _ScopedConfig


def test_config_sections_match_shared_contract():
    """釘住本包宣告了哪些頂層區塊、`input` 來自共用 lib（見 ADR-008）。

    頂層區塊名等於一段全域的環境變數命名空間（`env_nested_delimiter="__"`），所以
    新增區塊會讓這支變紅，逼使用者確認該名稱在其他包沒被用過；把 `input` 改回本地
    定義同樣變紅——那正是 `INPUT__*` 撞名的來源。
    """
    assert set(AppConfig.model_fields) == {"input", "report"}
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
    少列一個別包有的欄位，設了該變數的環境會讓本包連 import 都失敗（`camera_ids`
    只有 `video_analyze` 讀，過去正是這樣打死本包）。
    """
    monkeypatch.setenv(env_name, env_value)
    toml = tmp_path / "config.toml"
    toml.write_text("[report]\nperiod_minutes = 90\n", encoding="utf-8")

    config = _config_class(toml)()

    assert config.report.period_minutes == 90


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


def test_legacy_zone_env_var_does_not_break_this_package(monkeypatch, tmp_path):
    """`ZONE__BUCKET_MINUTES` 不能讓本包載入失敗。

    本包沒有 `[zone]` 區塊，pydantic-settings 的 env source 只查已知欄位名，這個變數
    不會進到模型、也不會被 `extra="forbid"` 擋下。本模組在載入時就 `load_config()`，
    在這裡拋錯等於連 import 都失敗；提醒的責任在 `zone_mapping`（該包的搬家提示會
    擋下它），本包不再重複警告一份。
    """
    monkeypatch.setenv("ZONE__BUCKET_MINUTES", "30")
    toml = tmp_path / "config.toml"
    toml.write_text("[input]\nbucket_minutes = 60\n", encoding="utf-8")

    config = _config_class(toml)()

    assert config.input.bucket_minutes == 60


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
        "flow_report.models.config.get_toml_path",
        lambda _caller_file: "/nonexistent/config.toml",
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
