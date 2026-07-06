"""單一 CLI 進入點，把三個階段收斂成同一支程式的三個子命令。

- analyze：偵測/追蹤階段（重、GPU、多進程），輸出 tracking_results.parquet 與標註影片。
- zone-map：zone 人流統計階段（輕、純運算），讀上一階段的 parquet 輸出 zone_counts。
- report：Excel 人流報表階段（輕、純運算），彙總 zone_counts.parquet 成跨日累加報表。

三個階段各對應一個獨立可呼叫的函式進入點，刻意維持各自獨立可跑：調 camera_registry.yaml
內的 zone 定義後只重跑 zone-map，不必重跑昂貴的偵測階段；調報表參數後也只重跑 report。

各分支才 lazy import 對應模組，讓 zone-map 和 report 不必載入 torch/ultralytics。
"""

import argparse


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="video-flow-analytics",
        description="多路離線影片流分析：偵測追蹤與 zone 人流統計",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser(
        "analyze",
        help="讀影片跑 YOLO+ByteTrack，輸出 tracking_results.parquet 與標註影片",
    )
    subparsers.add_parser(
        "zone-map",
        help="讀 tracking_results.parquet 套 zone 幾何，輸出每時段每區域人流",
    )
    subparsers.add_parser(
        "report",
        help="讀 zone_counts.parquet 彙總成跨日累加的 Excel 人流報表",
    )
    args = parser.parse_args()

    if args.command == "analyze":
        from video_flow_analytics.analyze.pipeline import run_analyze

        run_analyze()
    elif args.command == "zone-map":
        from video_flow_analytics.zone_mapping.pipeline import run_zone_map

        run_zone_map()
    elif args.command == "report":
        from video_flow_analytics.report.pipeline import run_report

        run_report()


if __name__ == "__main__":
    main()
