import pytest

from video_analyze.models.config import settings
from video_analyze.services.detector import (
    YOLODetector,
    _basename,
    _log_model_metadata,
    _validate_classes,
    _validate_imgsz,
)
from video_analyze.services.letterbox import INFER_HEIGHT, INFER_WIDTH


class _FakeModel:
    """只帶 `_log_model_metadata`／`_validate_classes` 會用到的屬性，不真載權重。"""

    def __init__(self, names=None, ckpt=None):
        self.names = names
        self.ckpt = ckpt


def test_basename_strips_absolute_path():
    assert _basename("/home/trainer/runs/yolo26m_baseline/weights/best.pt") == (
        "best.pt"
    )


def test_basename_returns_non_string_unchanged():
    assert _basename(None) is None
    assert _basename(123) == 123


def test_basename_returns_empty_string_unchanged():
    assert _basename("") == ""


def test_log_model_metadata_with_full_ckpt(capsys):
    model = _FakeModel(
        names={0: "head", 1: "vbody", 2: "fbody"},
        ckpt={
            "train_args": {
                "model": "/home/trainer/runs/yolo26m_baseline/weights/best.pt",
                "data": "/home/trainer/datasets/crowdhuman/data.yaml",
            },
            "version": "8.4.90",
            "date": "2026-07-14",
            "train_metrics": {"mAP50": 0.806, "recall": 0.72, "precision": 0.847},
        },
    )

    _log_model_metadata(model)

    out = capsys.readouterr().out
    assert "fbody" in out
    assert "best.pt" in out
    assert "data.yaml" in out
    assert "8.4.90" in out
    assert "2026-07-14" in out
    assert "0.806" in out


def test_log_model_metadata_missing_train_args_logs_available_fields(capsys):
    model = _FakeModel(
        names={2: "fbody"},
        ckpt={"version": "8.4.90"},
    )

    _log_model_metadata(model)

    assert "8.4.90" in capsys.readouterr().out


def test_log_model_metadata_non_dict_ckpt_warns_without_raising(capsys):
    model = _FakeModel(names={2: "fbody"}, ckpt=None)

    _log_model_metadata(model)

    assert '"severity": "WARNING"' in capsys.readouterr().out


def test_log_model_metadata_missing_names_attr_does_not_raise(capsys):
    class _NoNamesModel:
        ckpt = {"version": "8.4.90"}

    _log_model_metadata(_NoNamesModel())

    assert "8.4.90" in capsys.readouterr().out


def test_log_model_metadata_exception_during_read_warns_without_raising(capsys):
    class _RaisingNamesModel:
        ckpt = {"version": "8.4.90"}

        @property
        def names(self):
            raise RuntimeError("boom")

    _log_model_metadata(_RaisingNamesModel())  # 不拋例外，只 warning

    assert '"severity": "WARNING"' in capsys.readouterr().out


def test_validate_classes_passes_when_all_ids_present(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2])
    model = _FakeModel(names={0: "head", 1: "vbody", 2: "fbody"})

    _validate_classes(model)  # 不應拋例外


def test_validate_classes_raises_when_id_missing_from_model_names(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2, 5])
    model = _FakeModel(names={0: "head", 1: "vbody", 2: "fbody"})

    with pytest.raises(ValueError, match=r"\[5\]"):
        _validate_classes(model)


def test_validate_classes_skips_when_model_names_unavailable(monkeypatch):
    monkeypatch.setattr(settings.model, "classes", [2])
    model = _FakeModel(names=None)

    _validate_classes(model)  # names 缺失時無法驗證，略過而非拋例外


def test_yolo_detector_raises_when_model_path_missing(monkeypatch, tmp_path):
    # 檔案不存在時必須直接 fail loud，不可讓 ultralytics 自行 fallback 下載到
    # 別的模型（那樣 _validate_classes 也擋不住，見同函式的說明）。
    monkeypatch.setattr(settings.model, "model_path", str(tmp_path / "missing.pt"))

    with pytest.raises(FileNotFoundError):
        YOLODetector()


class _FakeModelWithOverrides:
    """只帶 `_validate_imgsz` 會用到的 `overrides`。"""

    def __init__(self, overrides):
        self.overrides = overrides


def test_validate_imgsz_accepts_a_matching_weight():
    _validate_imgsz(_FakeModelWithOverrides({"imgsz": INFER_WIDTH}))
    _validate_imgsz(_FakeModelWithOverrides({"imgsz": [INFER_HEIGHT, INFER_WIDTH]}))


def test_validate_imgsz_rejects_a_weight_trained_at_another_size():
    """換上以別的 imgsz 訓練的權重要當場擋下。

    讀取端已把影格縮成 `INFER_WIDTH` × `INFER_HEIGHT`，`predict` 會再照權重的 imgsz
    縮放一次——先縮小再放大回去，細節回不來、召回率靜默下降，而輸出的欄位、座標、
    列數全部正常。
    """
    with pytest.raises(ValueError, match="imgsz"):
        _validate_imgsz(_FakeModelWithOverrides({"imgsz": 1280}))


def test_validate_imgsz_only_warns_when_the_weight_has_no_imgsz(capsys):
    """metadata 缺漏不是已知的錯誤組合，記警告就好，不擋整天的分析。"""
    _validate_imgsz(_FakeModelWithOverrides({}))

    assert "imgsz" in capsys.readouterr().out
