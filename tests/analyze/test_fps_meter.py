import pytest

from video_flow_analytics.analyze.fps_meter import FpsMeter


def test_per_camera_and_overall_throughput():
    meter = FpsMeter()
    for _ in range(10):
        meter.record("cam_a")
    for _ in range(30):
        meter.record("cam_b")

    summary = meter.summary(elapsed_seconds=2.0)

    assert summary.total_frames == 40
    assert summary.per_camera_frames == {"cam_a": 10, "cam_b": 30}
    assert summary.overall_fps == pytest.approx(20.0)
    assert summary.per_camera_fps["cam_a"] == pytest.approx(5.0)
    assert summary.per_camera_fps["cam_b"] == pytest.approx(15.0)


def test_per_camera_fps_sums_to_overall_fps():
    # 逐路 FPS 用同一段 wall-clock 當分母，語意為「該路對整體吞吐的貢獻」，
    # 因此相加必須等於整體 FPS——若哪天分母改成各路自己的區間，這個不變量會壞掉。
    meter = FpsMeter()
    for _ in range(7):
        meter.record("cam_a")
    for _ in range(13):
        meter.record("cam_b")

    summary = meter.summary(elapsed_seconds=4.0)

    assert sum(summary.per_camera_fps.values()) == pytest.approx(summary.overall_fps)


def test_detection_and_tracking_stage_fps():
    # 階段 FPS = 總格數 / 該階段累計耗時，代表「只跑該階段」的吞吐。
    meter = FpsMeter()
    for _ in range(40):
        meter.record("cam_a")
    # 兩批偵測共 1.0 秒 → 40 / 1.0 = 40 fps
    meter.add_detection_time(0.5)
    meter.add_detection_time(0.5)
    # 40 格追蹤共 0.4 秒 → 40 / 0.4 = 100 fps
    for _ in range(40):
        meter.add_tracking_time(0.01)

    summary = meter.summary(elapsed_seconds=2.0)

    assert summary.detection_fps == pytest.approx(40.0)
    assert summary.tracking_fps == pytest.approx(100.0)


def test_zero_elapsed_yields_zero_fps_without_error():
    # 防呆：分母為 0 不可拋 ZeroDivisionError，回 0.0 讓總結仍印得出來。
    meter = FpsMeter()
    meter.record("cam_a")

    summary = meter.summary(elapsed_seconds=0.0)

    assert summary.overall_fps == 0.0
    assert summary.per_camera_fps["cam_a"] == 0.0


def test_no_stage_time_yields_zero_stage_fps():
    # 未累計任何階段耗時（例如都還沒跑）時，階段 FPS 分母為 0，回 0.0。
    meter = FpsMeter()
    meter.record("cam_a")

    summary = meter.summary(elapsed_seconds=1.0)

    assert summary.detection_fps == 0.0
    assert summary.tracking_fps == 0.0
