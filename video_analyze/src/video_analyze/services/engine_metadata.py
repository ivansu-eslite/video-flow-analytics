"""TensorRT 引擎自帶 metadata 的注入格式、讀取，以及對當下環境的驗證。

**建置端與載入端共用這一份**：`tools/build_engine.py` 用它組出要注入的欄位，
`services/detector.py` 用它把欄位讀回來比對。分成兩份寫的話，哪天有人只改其中一邊，
比對會靜默地全數通過（欄位名對不上就走「缺欄位」那條路），而那正是這些欄位存在的
理由——引擎是二進位產物，跑錯一顆不會有任何症狀。

metadata 的容器格式不是我們定的：ultralytics 的 exporter 把 `json.dumps(metadata)`
的長度以 4 bytes little-endian 寫在引擎檔開頭，接著是 JSON 本體，然後才是序列化的
引擎（`utils/export/engine.py` 的 `onnx2engine`）。`nn/backends/tensorrt.py` 讀回來
之後會把**每一個 key `setattr` 到 backend 上**，所以注入的欄位全部收在單一的
`VFA_METADATA_KEY` 底下，而不是攤成數個頂層 key——攤開的話要逐一避開
`fp16`／`dynamic`／`model`／`context`／`bindings`／`output_names` 這些 backend
自己的屬性（其中前幾個是在 `apply_metadata` **之後**才賦值的，注入的值會被無聲蓋掉，
比對因此會拿到 backend 的值而不是引擎的值）。

本模組讀檔頭而不透過 `AutoBackend`：驗證要在 deserialize **之前**發生（不合格的引擎
不該先吃掉幾秒與數 GB 顯存），而且取 `names` 不必為此多載一次引擎。
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from vfa_observability import StructuredLogger

logger = StructuredLogger(component="engine_metadata")

# 注入欄位的容器 key。改名會讓舊引擎在載入時走到「缺欄位」那條路而被擋下（不是靜默
# 通過），所以改名是安全的，但既有引擎全部要重建。
VFA_METADATA_KEY = "vfa"

# 注入欄位的版本號。欄位語義改變（不只是新增）時要進位，載入端才擋得掉舊引擎。
VFA_METADATA_SCHEMA = 1

# ultralytics 寫在引擎檔開頭的 metadata 長度欄位寬度（bytes，little-endian、signed）。
_META_LEN_BYTES = 4

# 檔頭長度的上限，用來擋下「這根本不是 ultralytics 匯出的引擎」——裸 TensorRT 引擎
# 的前 4 bytes 是引擎資料本身，解讀成長度會得到一個荒謬的值，接著才是 JSON 解析失敗。
# 先擋長度，錯誤訊息才指得出真正的原因。
_META_LEN_MAX = 1 << 20


@dataclass(frozen=True)
class GpuEnvironment:
    """引擎的建置環境／載入環境中，會決定引擎能不能用的那幾項。

    Attributes:
        compute_capability: GPU 的 compute capability（如 `"7.5"`）。**不是型號字串**：
            TensorRT 的 `HardwareCompatibilityLevel` 只有 `NONE`／`AMPERE_PLUS`／
            `SAME_COMPUTE_CAPABILITY`，引擎的實際約束是 SM 而不是型號。
        gpu_name: GPU 型號字串，只供人閱讀與追溯，不參與比對。
        tensorrt: TensorRT 版本（如 `"10.13.3.9"`）。
        cuda_major: 建置／載入端 torch 所綁的 CUDA 大版本（如 `"12"`）。
        driver: 顯示卡驅動版本字串；取不到時為 `None`。
    """

    compute_capability: str
    gpu_name: str
    tensorrt: str
    cuda_major: str
    driver: str | None


def current_gpu_environment() -> GpuEnvironment:
    """取當下進程看到的 GPU 環境。

    Returns:
        當下的 `GpuEnvironment`。

    Raises:
        RuntimeError: 沒有可用的 CUDA 裝置。引擎綁 GPU，這裡不做 CPU fallback
            （見 `services/detector.py`）。
    """
    import tensorrt as trt
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(
            "沒有可用的 CUDA 裝置。正式推論路徑是 TensorRT 引擎（見 ADR-011），"
            "引擎綁 GPU，CPU 上不是「比較慢」而是一定失敗，故不提供 fallback。"
        )
    props = torch.cuda.get_device_properties(0)
    return GpuEnvironment(
        compute_capability=f"{props.major}.{props.minor}",
        gpu_name=props.name,
        tensorrt=trt.__version__,
        cuda_major=str(torch.version.cuda).split(".")[0],
        driver=_driver_version(),
    )


def _driver_version() -> str | None:
    """讀顯示卡驅動版本；讀不到回 `None`。

    走 `pynvml`（`nvidia-ml-py`）——它是 `ultralytics` 的直接依賴，凡是跑得動本套件的
    環境都有。仍然包在 try 內：驅動版本是**只記錄不擋**的欄位（見
    `validate_engine_metadata`），為了取一個不參與判定的值而讓整天的分析起不來，
    代價方向不對。
    """
    try:
        import pynvml

        pynvml.nvmlInit()
        version = pynvml.nvmlSystemGetDriverVersion()
    except Exception as exc:
        logger.warning("讀不到顯示卡驅動版本，該欄位以 null 記錄。", error=str(exc))
        return None
    return version.decode() if isinstance(version, bytes) else str(version)


def sha256_of(path: Path) -> str:
    """算檔案內容的 SHA-256（十六進位小寫）。"""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_vfa_metadata(
    weights_path: Path,
    environment: GpuEnvironment,
    train_info: dict,
) -> dict:
    """組出要注入引擎的 `vfa` 欄位。

    Args:
        weights_path: 來源 `.pt` 權重檔。
        environment: 建置機的 GPU 環境。
        train_info: 從 `.pt` 的 ckpt 抄出來的訓練追溯資訊（見
            `tools/build_engine.py` 的 `read_train_info`）。

    Returns:
        可直接塞進 `exporter.metadata[VFA_METADATA_KEY]` 的 dict（全部欄位皆可
        `json.dumps`）。
    """
    return {
        "schema": VFA_METADATA_SCHEMA,
        "source_weights": {
            "name": weights_path.name,
            "sha256": sha256_of(weights_path),
        },
        "compute_capability": environment.compute_capability,
        "gpu_name": environment.gpu_name,
        "tensorrt": environment.tensorrt,
        "cuda_major": environment.cuda_major,
        "driver": environment.driver,
        "train": train_info,
    }


def read_engine_metadata(engine_path: Path) -> dict:
    """從引擎檔開頭讀回 ultralytics 寫入的 metadata。

    Args:
        engine_path: `.engine` 檔路徑。

    Returns:
        metadata dict。`names` 的 key 會還原成 int——JSON 沒有整數 key，
        `json.dumps({0: "head"})` 會寫成 `{"0": "head"}`，直接拿去跟
        `settings.model.classes` 的 int 比對會全數落空（而且是「查無此 id」那個
        方向，剛好會誤判成設定錯誤）。

    Raises:
        ValueError: 檔頭不是 ultralytics 的 metadata 格式（長度欄位荒謬或 JSON
            解不開），代表這顆引擎不是經由本 repo 的建置工具產出的。
    """
    with open(engine_path, "rb") as handle:
        raw_len = handle.read(_META_LEN_BYTES)
        meta_len = int.from_bytes(raw_len, byteorder="little", signed=True)
        if not 0 < meta_len <= _META_LEN_MAX:
            raise ValueError(
                f"{engine_path} 的檔頭沒有 ultralytics metadata（長度欄位讀到 "
                f"{meta_len}）。裸的 TensorRT 引擎不帶這段檔頭，請改用 "
                "tools/build_engine.py 產出的引擎。"
            )
        try:
            metadata = json.loads(handle.read(meta_len).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError(
                f"{engine_path} 的 metadata 解析失敗：{exc}。這顆引擎不是 "
                "tools/build_engine.py 產出的。"
            ) from exc
    names = metadata.get("names")
    if isinstance(names, dict):
        metadata["names"] = {
            int(k) if isinstance(k, str) and k.lstrip("-").isdigit() else k: v
            for k, v in names.items()
        }
    return metadata


def validate_engine_metadata(
    metadata: dict,
    environment: GpuEnvironment,
    expected_weights_sha256: str,
) -> None:
    """比對引擎自帶的 metadata 與當下環境；不符即中止，訊息指出是哪一項。

    **每一項各自拋出、訊息各自寫**，不合併成一句「metadata 與環境不符」：這幾項的
    處置方式完全不同——SM 不符要換一顆引擎（且只能在目標卡上重建），TensorRT 版本不符
    要對齊映像檔，權重 hash 不符是拿錯了引擎。合成一句話會讓看 log 的人得自己重跑一次
    才知道要修哪裡。

    **驅動版本只記錄、不擋**：引擎的硬約束來自 TensorRT 本身（SM 與 TRT 版本），驅動
    不在其中；而建置機（GCE T4 VM）與正式節點（Vertex CustomJob 的容器）本來就不保證
    同一份映像檔與驅動，把它做成致命條件等於堵死唯一可行的建置路徑。不符時記 warning，
    留給人判斷。

    Args:
        metadata: `read_engine_metadata` 讀回的 metadata。
        environment: 當下的 GPU 環境（`current_gpu_environment`）。
        expected_weights_sha256: 設定檔釘住的來源權重 SHA-256；空字串代表沒釘，
            此時只記錄不比對（並 warning，讓「沒釘」這件事在 log 上看得見）。

    Raises:
        ValueError: metadata 缺少 `vfa` 區塊、schema 版本不符，或權重身分／
            compute capability／TensorRT 版本／CUDA 大版本任一項與當下環境不符。
    """
    vfa = metadata.get(VFA_METADATA_KEY)
    if not isinstance(vfa, dict):
        raise ValueError(
            f"引擎 metadata 沒有 `{VFA_METADATA_KEY}` 區塊，無法確認它是為哪一張卡、"
            "哪一份權重、哪一版 TensorRT 建的。請改用 tools/build_engine.py 重建。"
        )
    schema = vfa.get("schema")
    if schema != VFA_METADATA_SCHEMA:
        raise ValueError(
            f"引擎 metadata 的 schema 是 {schema}，本版預期 {VFA_METADATA_SCHEMA}。"
            "注入欄位的語義已改變，這顆引擎要用當前的 tools/build_engine.py 重建。"
        )

    engine_cc = vfa.get("compute_capability")
    if engine_cc != environment.compute_capability:
        raise ValueError(
            f"引擎的 compute capability 是 {engine_cc}（建置機 "
            f"{vfa.get('gpu_name')}），當下的 GPU 是 "
            f"{environment.gpu_name}（{environment.compute_capability}）。"
            "TensorRT 引擎不可跨架構重用——`IBuilderConfig` 沒有指定 target SM 的 API，"
            "`HardwareCompatibilityLevel` 也只有 NONE／AMPERE_PLUS／"
            "SAME_COMPUTE_CAPABILITY（AMPERE_PLUS 不含 Turing）。請在這張卡上重建引擎。"
        )

    engine_trt = vfa.get("tensorrt")
    if engine_trt != environment.tensorrt:
        raise ValueError(
            f"引擎是用 TensorRT {engine_trt} 建的，當下環境是 "
            f"{environment.tensorrt}。跨版的引擎 deserialize 不保證成功，成功了也不"
            "保證數值一致。升版順序是有向的：先在目標卡上用新版重建引擎並發布，"
            "才能滾動映像檔。"
        )

    engine_cuda = vfa.get("cuda_major")
    if engine_cuda != environment.cuda_major:
        raise ValueError(
            f"引擎的建置環境 CUDA 大版本是 {engine_cuda}，當下是 "
            f"{environment.cuda_major}。`tensorrt-cu12` 與 `tensorrt-cu13` 是不同的"
            "套件，兩者的 runtime 不可互換。"
        )

    engine_weights = vfa.get("source_weights") or {}
    engine_sha = engine_weights.get("sha256")
    if expected_weights_sha256:
        if engine_sha != expected_weights_sha256:
            raise ValueError(
                f"引擎的來源權重是 {engine_weights.get('name')}"
                f"（sha256 {engine_sha}），設定檔 [model].source_weights_sha256 "
                f"釘的是 {expected_weights_sha256}。載入的不是預期的那顆引擎——"
                "類別 id 可能剛好都存在而語義不同，`classes` 過濾不會有任何訊號。"
            )
    else:
        logger.warning(
            "[model].source_weights_sha256 未設定，來源權重身分只記錄不比對。",
            engine_source_weights=engine_weights.get("name"),
            engine_source_sha256=engine_sha,
        )

    if vfa.get("driver") != environment.driver:
        # 只記錄不擋，理由見本函式 docstring
        logger.warning(
            "引擎建置機與當下環境的顯示卡驅動版本不同（不影響引擎有效性，僅供追溯）。",
            build_driver=vfa.get("driver"),
            current_driver=environment.driver,
        )
