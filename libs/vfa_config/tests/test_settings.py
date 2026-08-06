"""共用 `[input]` 區塊與 `config.toml` 定位的回歸測試。

重點守護兩件事：`InputConfig` 是四包欄位的**聯集**（少一個欄位就會讓設了對應
`INPUT__*` 環境變數的環境把不讀該欄位的套件打死，見 ADR-008）；`get_toml_path` 定位
的是**呼叫端**的套件根，不是本 lib 的。
"""

from pathlib import Path

import pytest
from pydantic import ValidationError

from vfa_config import InputConfig, find_project_root, get_toml_path


def test_input_config_covers_all_four_packages_fields():
    """`[input]` 的欄位必須是四包需求的聯集，逐一釘住。

    刪掉其中任一欄（例如「`flow_report` 用不到 `camera_ids`」）就會讓設了
    `INPUT__CAMERA_IDS` 的環境把該包打死在 import 期。這支測試是那個回歸的門，
    新增欄位時一併更新，並確認該名稱在四包沒有語義衝突。
    """
    assert set(InputConfig.model_fields) == {
        "bucket_dir",
        "date",
        "camera_ids",
        "bucket_minutes",
    }


def test_input_config_accepts_every_field_regardless_of_consumer():
    """不讀某欄位的套件也要能接受它——共用區塊的重點就在這裡。"""
    config = InputConfig(
        bucket_dir="bucket_x",
        date="2026-05-01",
        camera_ids=["cam1"],
        bucket_minutes=30,
    )

    assert config.bucket_dir == "bucket_x"
    assert config.camera_ids == ["cam1"]
    assert config.bucket_minutes == 30


def test_input_config_defaults():
    config = InputConfig()

    assert config.bucket_dir == "bucket_name"
    assert config.date is None
    assert config.camera_ids == []
    assert config.bucket_minutes == 60


def test_unknown_field_still_rejected():
    """聯集不等於放行任意欄位：拼錯的欄位名仍要報錯，不可靜默退回預設值。"""
    with pytest.raises(ValidationError):
        InputConfig(bucket_dirs="bucket_x")


def test_bucket_minutes_rejects_non_positive():
    """`ge=1`：0 分鐘的時段粒度不合法，須報錯而非靜默套用。"""
    with pytest.raises(ValidationError):
        InputConfig(bucket_minutes=0)


def test_find_project_root_locates_package_root():
    """find_project_root 取代寫死的 parents[N]，須能定位到含 pyproject.toml 的套件根。"""
    root = find_project_root(Path(__file__).resolve())
    assert root is not None
    assert (root / "pyproject.toml").exists()


def test_find_project_root_returns_none_when_no_pyproject(tmp_path):
    """向上找不到 pyproject.toml 時回 None，由呼叫端決定退路。"""
    assert find_project_root(tmp_path / "a" / "b" / "c.py") is None


def test_get_toml_path_resolves_relative_to_caller_not_this_lib():
    """路徑要由傳入的 `caller_file` 決定，不是本 lib 的位置。

    抽成共用函式的關鍵風險：實作若還是用本檔的 `__file__`，四包都會去讀
    `libs/vfa_config/config.toml`（不存在），全部靜默退回預設參數。
    """
    lib_root = find_project_root(Path(__file__).resolve())
    assert lib_root is not None

    path = get_toml_path(__file__)
    assert path == str(lib_root / "config.toml")

    # 換一個位於別處的呼叫端檔案，結果要跟著換，而不是黏在本 lib 上。
    repo_root = find_project_root(lib_root.resolve())
    assert repo_root is not None
    other_caller = repo_root / "flow_report" / "src" / "flow_report" / "models" / "config.py"
    assert get_toml_path(other_caller) == str(repo_root / "flow_report" / "config.toml")


def test_get_toml_path_falls_back_to_cwd_when_no_project_root(tmp_path, monkeypatch):
    """容器環境下 pyproject.toml 可能未一併複製，此時以 cwd 的 config.toml 為準。"""
    # 直接 patch find_project_root 而非靠 tmp_path 的祖先剛好沒有 pyproject.toml：
    # tmp_path 位於 /tmp 底下，祖先目錄不受本測試控制。
    monkeypatch.setattr("vfa_config.settings.find_project_root", lambda _: None)
    (tmp_path / "config.toml").write_text("[input]\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert get_toml_path(tmp_path / "nowhere" / "config.py") == str(tmp_path / "config.toml")


def test_get_toml_path_returns_none_when_nothing_found(tmp_path, monkeypatch):
    """兩條路都找不到時回 None，讓呼叫端警告後以預設值啟動。"""
    monkeypatch.setattr("vfa_config.settings.find_project_root", lambda _: None)
    monkeypatch.chdir(tmp_path)

    assert get_toml_path(tmp_path / "nowhere" / "config.py") is None
