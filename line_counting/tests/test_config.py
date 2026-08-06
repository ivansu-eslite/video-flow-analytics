"""設定載入語義的回歸測試。

重點守護「找不到 `config.toml` → 警告並以預設值啟動；檔案存在但值不合法或頂層
區塊名未知 → 直接報錯」這條 fail-loud 語義：參數錯了卻靜默套用預設值，會讓進出
人數統計以非預期的口徑產出而無人察覺（例如把 `[line]` 拼成 `[lines]`，去抖的線段區域寬度與
時段粒度整段悄悄退回預設）。

`AppConfig.model_config` 的 `toml_file` 在 class 定義時就求值，事後 monkeypatch
`get_toml_path` 不會改變它，故這裡改用指定 `toml_file` 的子類別來測實際載入行為。
"""

import inspect
from pathlib import Path

import pytest
import vfa_config
from pydantic import ValidationError
from pydantic_settings import SettingsConfigDict
from vfa_config import get_toml_path

from line_counting.config.constants import DEFAULT_CROSSING_BAND_PX_1080P
from line_counting.models import config as config_module
from line_counting.models.config import AppConfig, LineConfig, load_config
from line_counting.services.line_map import count_lines_daily

# 設定來源含環境變數，且欄位名未加前綴：執行環境剛好有這些變數時會蓋掉 toml 的值，
# 讓測試結果取決於誰的機器在跑。逐一清掉，測的才是「從這份 toml 載入」的行為。
# 舊名 LINE__CROSSING_BAND_PX 與搬家的 LINE__BUCKET_MINUTES 一併清掉：兩者都不再是
# 有效欄位、且會觸發改名／搬家錯誤，執行環境剛好留著會讓其他測試整批報錯。
_ENV_OVERRIDES = ("INPUT", "LINE") + (
    "INPUT__BUCKET_DIR",
    "INPUT__DATE",
    "INPUT__CAMERA_IDS",
    "INPUT__BUCKET_MINUTES",
    "LINE__BUCKET_MINUTES",
    "LINE__CROSSING_BAND_PX_1080P",
    "LINE__CROSSING_BAND_PX",
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


def test_config_sections_match_shared_contract():
    """釘住本包宣告了哪些頂層區塊、`input` 來自共用 lib（見 ADR-008）。

    頂層區塊名等於一段全域的環境變數命名空間（`env_nested_delimiter="__"`），所以
    新增區塊會讓這支變紅，逼使用者確認該名稱在其他包沒被用過；把 `input` 改回本地
    定義同樣變紅——那正是 `INPUT__*` 撞名的來源。
    """
    assert set(AppConfig.model_fields) == {"input", "line"}
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
    只有 `video_analyze` 讀，過去正是這樣打死另外三包）。
    """
    monkeypatch.setenv(env_name, env_value)
    toml = tmp_path / "config.toml"
    toml.write_text("[line]\ncrossing_band_px_1080p = 5\n", encoding="utf-8")

    config = _config_class(toml)()

    assert config.line.crossing_band_px_1080p == 5


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


def test_default_crossing_band_agrees_across_all_three_places():
    """版控的 `config.toml`、`LineConfig` 與 `count_lines_daily` 簽名的預設值要一致。

    這個值是實測調出來的（見 README），不是「沒設定時的中性值」；任兩處分岔都會讓同
    一份程式用不同的去抖尺度統計，而且不會有任何錯誤訊息——沒有 `config.toml` 的環境
    吃模型預設值，直接呼叫 `count_lines_daily`（README 說明的正式進入點）吃的是簽名
    預設值。後兩者已共用同一個常數，`config.toml` 無法引用常數，由本測試鎖住。
    （`bucket_dir` 一類的欄位刻意不受此約束。）
    """
    assert AppConfig().line.crossing_band_px_1080p == DEFAULT_CROSSING_BAND_PX_1080P
    assert LineConfig().crossing_band_px_1080p == DEFAULT_CROSSING_BAND_PX_1080P
    signature_default = inspect.signature(count_lines_daily).parameters[
        "crossing_band_px_1080p"
    ].default
    assert signature_default == DEFAULT_CROSSING_BAND_PX_1080P


def test_uses_defaults_when_toml_missing(tmp_path):
    """找不到設定檔時以預設值啟動，而非中止。"""
    config = _config_class(tmp_path / "nope.toml")()

    assert config.input.bucket_minutes == 60
    assert config.line.crossing_band_px_1080p == 25
    assert config.input.bucket_dir == "bucket_name"


def test_reads_values_from_toml(tmp_path):
    toml = tmp_path / "config.toml"
    toml.write_text(
        '[input]\nbucket_dir = "bucket_x"\ndate = 2026-05-01\nbucket_minutes = 30\n'
        "[line]\ncrossing_band_px_1080p = 5\n",
        encoding="utf-8",
    )

    config = _config_class(toml)()

    assert config.input.bucket_dir == "bucket_x"
    assert config.input.bucket_minutes == 30
    assert config.line.crossing_band_px_1080p == 5


def test_invalid_value_in_toml_raises_instead_of_silently_defaulting(tmp_path):
    """設定檔存在但值不合法時必須報錯——靜默套預設值會讓統計口徑被悄悄改掉。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[input]\nbucket_minutes = 0\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_moved_bucket_minutes_reports_new_location(tmp_path):
    """沿用舊 `[line] bucket_minutes` 的設定檔要報出新位置。

    `extra="forbid"` 本來就會擋下，但只說「不允許額外欄位」；沿用舊設定的人需要知道
    該把這個數值搬到哪裡，否則只能去翻 source。
    """
    toml = tmp_path / "config.toml"
    toml.write_text("[line]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError, match=r"bucket_minutes 已移到 \[input\]"):
        _config_class(toml)()


def test_moved_bucket_minutes_from_env_reports_new_location(monkeypatch, tmp_path):
    """環境變數路徑同樣要報出新位置——`LINE__BUCKET_MINUTES` 過去是本包的合法設定。"""
    monkeypatch.setenv("LINE__BUCKET_MINUTES", "30")
    toml = tmp_path / "config.toml"
    toml.write_text("[input]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="INPUT__BUCKET_MINUTES"):
        _config_class(toml)()


def test_negative_crossing_band_px_1080p_raises(tmp_path):
    """crossing_band_px_1080p 有 ge=0 約束：負寬度不合法，須報錯而非靜默套用。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[line]\ncrossing_band_px_1080p = -1\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_renamed_crossing_band_px_raises_actionable_error(tmp_path):
    """沿用舊參數名 `crossing_band_px` 要報出「已改名且語義改成 1080p 基準值」。

    `extra="forbid"` 本來就會擋下它，但訊息只說欄位未知；沿用舊設定的人需要知道
    數值語義也變了（4K 攝影機上實際的線段區域寬度變兩倍），才知道該不該改數字。
    """
    toml = tmp_path / "config.toml"
    toml.write_text("[line]\ncrossing_band_px = 5\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="crossing_band_px_1080p"):
        _config_class(toml)()


def test_renamed_crossing_band_px_from_env_raises(monkeypatch, tmp_path):
    """環境變數路徑同樣要報出改名訊息——覆寫來源不只 toml 一條。"""
    monkeypatch.setenv("LINE__CROSSING_BAND_PX", "5")
    toml = tmp_path / "config.toml"
    toml.write_text("[input]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError, match="crossing_band_px_1080p"):
        _config_class(toml)()


def test_unknown_top_level_section_raises(tmp_path):
    """區塊名打錯（如 [lines]）要報錯，不可被靜默忽略而套用預設值。"""
    toml = tmp_path / "config.toml"
    toml.write_text("[lines]\nbucket_minutes = 30\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_unknown_field_in_nested_section_raises(tmp_path):
    """巢狀欄位名打錯要報錯：crossing_band_px_1080p 拼成 crossing_band_1080p，不可被
    靜默忽略而讓線段區域寬度退回預設值（巢狀 model 的 extra="forbid"，非僅頂層區塊名）。

    錯字刻意不用舊參數名 `crossing_band_px`：那會被改名偵測的 before validator 接手，
    這支測試就會在 extra="forbid" 失效時依然通過，守護目標無聲消失。
    """
    toml = tmp_path / "config.toml"
    toml.write_text("[line]\ncrossing_band_1080p = 5\n", encoding="utf-8")

    with pytest.raises(ValidationError):
        _config_class(toml)()


def test_load_config_warns_when_toml_missing(monkeypatch, capsys):
    """找不到設定檔時要留下警告，不可靜默啟動。"""
    monkeypatch.setattr(
        "line_counting.models.config.get_toml_path",
        lambda _caller_file: "/nonexistent/config.toml",
    )

    load_config()

    out = capsys.readouterr().out
    assert "找不到 config.toml" in out
    assert '"severity": "WARNING"' in out
