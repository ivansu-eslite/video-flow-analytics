"""讓 `tests/` import 得到 `tools/` 底下的工具。

`tools/` 在 `src/` 之外、刻意不是 package（不隨 wheel 出貨，見 ADR-011），pytest 依
rootdir 插進 `sys.path` 的是套件本身，搆不到它。**這是本 repo 第一次讓測試碰 `tools/`**，
機制寫在這裡。

放 conftest 而不是在測試檔內插 `sys.path`：ruff 的 `E` 含 E402、`I` 是 isort，測試檔內
先插 path 再 import 會讓每行 import 都要掛 `# noqa` 並與 isort 打架，而現有測試檔全是
乾淨的頂層 import。放這裡也讓下一支工具的測試不必各自再推導一次路徑。
"""

import sys
from pathlib import Path

_TOOLS_DIR = Path(__file__).resolve().parent.parent / "tools"
if str(_TOOLS_DIR) not in sys.path:
    sys.path.insert(0, str(_TOOLS_DIR))
