"""文件契約測試：把 CLAUDE.md／README 裡可驗證的斷言釘成測試。

散文寫下的數量與路徑會隨程式碼漂移，而漂移本身沒有訊號——要有人剛好讀到那句話、
又剛好記得實際值，才會發現不一致。這支測試把「可以用 ls 算出來的事實」交給 CI，
漂了就紅燈，不依賴誰記得同步。

新增這類斷言的判準：該事實能由 repo 現況機械算出（檔數、路徑、編號），而非設計理由。

執行方式見 CLAUDE.md 常用指令段——本測試只用標準庫與 pytest，刻意不經 workspace
解析執行，避免為了跑文件檢查而裝上 video_analyze 的 torch 依賴子樹。
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
ADR_FILENAME_RE = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")


def _adr_files() -> list[Path]:
    """所有 ADR 檔。故意用寬鬆的 glob，放錯位置或命名不合規的檔要被下面的測試抓到。"""
    return sorted(ADR_DIR.glob("**/*.md"))


def _adr_numbers() -> list[int]:
    return sorted(
        int(m.group(1)) for p in _adr_files() if (m := ADR_FILENAME_RE.match(p.name))
    )


def _readme_adr_index_rows() -> list[str]:
    """README「架構決策紀錄」章的索引表列。

    只認表格列，不認整份 README——ADR 連結在別的段落也會出現（本檔多處內嵌引用），
    拿整份文字做子字串比對的話，漏加表格列這種最自然的漏法會驗不出來。
    """
    readme = ROOT_README.read_text(encoding="utf-8")
    section = re.search(r"^## 架構決策紀錄$(.*?)^## ", readme, re.MULTILINE | re.DOTALL)
    assert section, "README 找不到「## 架構決策紀錄」章，索引搬家了就要一併改這支測試"
    return [line for line in section.group(1).splitlines() if line.lstrip().startswith("|")]


def _test_file_count(pkg_dir: Path) -> int:
    return len(list((pkg_dir / "tests").glob("test_*.py")))


def test_adr_files_follow_naming_and_placement():
    """ADR 一律放在 docs/adr/<子目錄>/NNN-描述.md。

    編號與位置是後面兩支測試的前提：命名不合規的檔會讓編號抽取失效，
    直接放在 docs/adr/ 底下的檔會逃過依子目錄分類的索引。
    """
    offenders = []
    for path in _adr_files():
        rel = path.relative_to(REPO_ROOT).as_posix()
        if path.parent == ADR_DIR:
            offenders.append(f"{rel}（未放進子目錄）")
        elif not ADR_FILENAME_RE.match(path.name):
            offenders.append(f"{rel}（檔名不符 NNN-描述.md）")
    assert not offenders, f"ADR 命名或位置不合規：{offenders}"


def test_readme_adr_index_covers_every_adr_file():
    """README 的架構決策紀錄是 ADR 的索引，新增 ADR 沒登錄就等於找不到。"""
    rows = "\n".join(_readme_adr_index_rows())
    missing = [
        p.relative_to(REPO_ROOT).as_posix()
        for p in _adr_files()
        if p.relative_to(REPO_ROOT).as_posix() not in rows
    ]
    assert not missing, f"這些 ADR 檔不在 README 的索引表裡：{missing}"


def test_adr_numbers_are_a_gapless_global_sequence():
    """CLAUDE.md：編號是全域流水號、與子目錄無關，新增一律取下一號。

    編號被當成穩定識別碼在用（ADR 互相修訂、issue 與 PR 直接引用），重號會讓這些
    引用指向兩份文件。連號斷言的前提是**ADR 不刪除**——被取代的改 Status 為
    Superseded 並保留檔案（見 README 取號規則），所以正常演進不會產生空號。
    """
    numbers = _adr_numbers()
    duplicates = sorted({n for n in numbers if numbers.count(n) > 1})
    assert not duplicates, f"ADR 編號重複：{duplicates}"
    assert numbers == list(range(1, len(numbers) + 1)), (
        f"ADR 編號不連續：{numbers}（應為 1..{len(numbers)}）。"
        "若因刪除 ADR 而跳號，正確處置是保留檔案並改 Status，不是重新編號。"
    )


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
    """檔案搬移後只改了程式碼、沒改文件連結，讀者會被指到不存在的路徑。

    缺檔一律當失敗而非 skip：DOC_FILES 是這道護欄的涵蓋範圍，文件改名或搬走時
    要有訊號，靜默少驗一份就是護欄失效。
    """
    assert doc.exists(), (
        f"{doc.relative_to(REPO_ROOT)} 不存在。文件若已改名或搬走，"
        "請一併更新本測試的 DOC_FILES，不要讓涵蓋範圍靜默縮小"
    )
    broken = []
    for target in LINK_RE.findall(doc.read_text(encoding="utf-8")):
        # 外部 URL 不驗連通性；~ 開頭是本機個人筆記，在別人的機器上不存在
        if target.startswith(("http://", "https://", "mailto:", "#", "~")):
            continue
        path = target.split("#", 1)[0]
        if not path:
            continue
        if not (doc.parent / path).exists():
            broken.append(target)
    assert not broken, f"{doc.relative_to(REPO_ROOT)} 的連結指向不存在的路徑：{broken}"
