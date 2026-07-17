import tomllib

from video_analyze.config import AppConfig, ModelConfig, TrackerConfig


def test_model_config_classes_defaults_to_fbody():
    assert ModelConfig().classes == [2]


def test_model_config_classes_overridable_from_toml():
    toml_data = """
    [tracker]
    track_high_thresh = 0.5
    track_low_thresh = 0.1
    new_track_thresh = 0.6
    track_buffer = 30
    match_thresh = 0.8
    fuse_score = true
    gmc_method = "none"

    [model]
    model_path = "some_model.pt"
    batch = 4
    classes = [0, 2]
    """
    config_data = tomllib.loads(toml_data)
    config = AppConfig(**config_data)

    assert config.model.classes == [0, 2]


def test_tracker_config_defaults_unaffected():
    assert TrackerConfig().track_buffer == 30
