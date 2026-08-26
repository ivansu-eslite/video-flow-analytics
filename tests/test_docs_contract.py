"""文件契約測試：把 CLAUDE.md／README 裡可驗證的斷言釘成測試。

散文寫下的數量與路徑會隨程式碼漂移，而漂移本身沒有訊號——要有人剛好讀到那句話、
又剛好記得實際值，才會發現不一致。這支測試把「可以用 ls 算出來的事實」交給 CI，
漂了就紅燈，不依賴誰記得同步。

新增這類斷言的判準：該事實能由 repo 現況機械算出（檔數、路徑、編號），而非設計理由。
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_DIR = REPO_ROOT / "docs" / "adr"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ROOT_README = REPO_ROOT / "README.md"

# CLAUDE.md 常用指令段列出的順序，測試檔數斷言依此對位
PACKAGES = ["video_analyze", "zone_mapping", "line_counting", "flow_report"]
LIBS = ["vfa_registry", "vfa_observability", "vfa_config"]

DOC_FILES = [
    CLAUDE_MD,
    ROOT_README,
    *(REPO_ROOT / pkg / "README.md" for pkg in PACKAGES),
    *(REPO_ROOT / "libs" / lib / "README.md" for lib in LIBS),
]

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")


def _adr_files() -> list[str]:
    """所有 ADR 的 repo 相對路徑（docs/adr/<子目錄>/NNN-*.md）。"""
    return sorted(p.relative_to(REPO_ROOT).as_posix() for p in ADR_DIR.glob("*/*.md"))


def _test_file_count(pkg_dir: Path) -> int:
    return len(list((pkg_dir / "tests").glob("test_*.py")))


def test_readme_adr_index_covers_every_adr_file():
    """README 的架構決策紀錄是 ADR 的索引，新增 ADR 沒登錄就等於找不到。"""
    readme = ROOT_README.read_text(encoding="utf-8")
    missing = [adr for adr in _adr_files() if adr not in readme]
    assert not missing, f"這些 ADR 檔不在 README 的索引表裡：{missing}"


def test_readme_adr_index_has_no_dangling_entry():
    """索引指向已搬走或改名的 ADR 時，讀者會撞到死連結。"""
    readme = ROOT_README.read_text(encoding="utf-8")
    linked = sorted(set(re.findall(r"\((docs/adr/[^)]+\.md)\)", readme)))
    dangling = [link for link in linked if not (REPO_ROOT / link).exists()]
    assert not dangling, f"README 索引指向不存在的 ADR：{dangling}"


def test_claude_md_package_test_counts_match_reality():
    """CLAUDE.md 常用指令段寫的四包測試檔數。"""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    match = re.search(r"四包各 ([\d／]+) 支測試檔", text)
    assert match, "CLAUDE.md 找不到「四包各 N／N／N／N 支測試檔」的字樣"
    claimed = [int(n) for n in match.group(1).split("／")]
    assert len(claimed) == len(PACKAGES), f"宣稱的數字有 {len(claimed)} 個，四包應為 4 個"
    actual = [_test_file_count(REPO_ROOT / pkg) for pkg in PACKAGES]
    assert claimed == actual, f"CLAUDE.md 宣稱 {claimed}，實際 {dict(zip(PACKAGES, actual))}"


@pytest.mark.parametrize("lib", LIBS)
def test_claude_md_lib_test_counts_match_reality(lib: str):
    """CLAUDE.md 常用指令段在每個共用 lib 那行註記的測試檔數。"""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    match = re.search(rf"libs/{lib} pytest.*?（(\d+) 支）", text)
    assert match, f"CLAUDE.md 的 libs/{lib} 那行找不到「（N 支）」註記"
    actual = _test_file_count(REPO_ROOT / "libs" / lib)
    assert int(match.group(1)) == actual, (
        f"CLAUDE.md 宣稱 libs/{lib} 有 {match.group(1)} 支測試，實際 {actual} 支"
    )


@pytest.mark.parametrize("doc", DOC_FILES, ids=lambda p: p.relative_to(REPO_ROOT).as_posix())
def test_markdown_links_point_to_existing_paths(doc: Path):
    """檔案搬移後只改了程式碼、沒改文件連結，讀者會被指到不存在的路徑。"""
    if not doc.exists():
        pytest.skip(f"{doc.relative_to(REPO_ROOT)} 不存在")
    broken = []
    for target in LINK_RE.findall(doc.read_text(encoding="utf-8")):
        if target.startswith(("http://", "https://", "mailto:", "#", "~")):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (doc.parent / path).exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(REPO_ROOT)} 的連結指向不存在的路徑：{broken}"


def test_adr_numbers_are_a_gapless_global_sequence():
    """CLAUDE.md：編號是全域流水號、與子目錄無關，新增一律取下一號。

    編號被當成穩定識別碼在用（ADR 互相修訂、issue 與 PR 直接引用），重號或跳號
    會讓這些引用指錯對象。
    """
    numbers = sorted(int(Path(adr).name[:3]) for adr in _adr_files())
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"ADR 編號重複：{duplicates}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR 編號不連續：{numbers}（應為 1..{len(numbers)}）"
    )
