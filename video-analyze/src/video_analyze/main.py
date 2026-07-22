import dataclasses
import sys

from vfa_observability import StructuredLogger

from video_analyze.models.config import settings
from video_analyze.services.pipeline import analyze_daily

logger = StructuredLogger(component="main")


def main() -> None:
    """CLI 進入點：從 `config.toml` 取參數後呼叫 `analyze_daily`。

    Raises:
        ValueError: `config.toml` 的 `[input].date` 未設定。
    """
    if settings.input.date is None:
        raise ValueError("config.toml 的 [input].date 未設定，請指定要分析的日期。")
    try:
        result = analyze_daily(
            date=settings.input.date,
            bucket_dir=settings.input.bucket_dir,
            camera_ids=settings.input.camera_ids,
        )
    except KeyboardInterrupt:
        sys.exit(130)
    logger.info("當日所有影片皆已處理完畢")
    logger.info("分析結果", result=dataclasses.asdict(result))


if __name__ == "__main__":
    main()
