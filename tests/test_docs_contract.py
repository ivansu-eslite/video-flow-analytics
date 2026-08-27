"""文件契約測試：把 CLAUDE.md／README 裡可驗證的斷言釘成測試。

散文寫下的數量與路徑會隨程式碼漂移，而漂移本身沒有訊號——要有人剛好讀到那句話、
又剛好記得實際值，才會發現不一致。這支測試把「可以用 ls 算出來的事實」交給測試，
漂了就紅燈，不依賴誰記得同步。

新增這類斷言的判準：該事實能由 repo 現況機械算出（檔數、路徑、編號），而非設計理由。

涵蓋範圍本身也由 repo 推導而非硬編碼——套件清單來自 CLAUDE.md 那行 `<pkg> = ...`、
lib 清單來自 libs/ 目錄、兩者合起來要等於 workspace members，新增成員沒被納入檢查
時會紅燈。硬編碼的話，新增一個 lib 只會讓它靜默落在護欄外。

同一個立場也套用到 CI 設定：.github/workflows/ci.yml 的成員清單是人工維護的，
這裡把它釘回 workspace members——否則新增成員時本檔的文件斷言會因 libs/* glob
自動納入而通過，該成員的 lint 與測試卻靜默不執行。

執行方式見 CLAUDE.md 常用指令段——本測試刻意不經 workspace 解析執行，避免為了跑
文件檢查而裝上 video_analyze 的 torch 依賴子樹；相依只有 pytest 與 pyyaml（後者用來
解析 ci.yml，理由見 _ci_covered_members），兩者都以 `--with` 帶進去。
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
ADR_DIR = REPO_ROOT / "docs" / "adr"
CLAUDE_MD = REPO_ROOT / "CLAUDE.md"
ROOT_README = REPO_ROOT / "README.md"
CI_WORKFLOW = REPO_ROOT / ".github" / "workflows" / "ci.yml"

LINK_RE = re.compile(r"\[[^\]]*\]\(([^)\s]+)\)")
ADR_FILENAME_RE = re.compile(r"^(\d{3})-[a-z0-9-]+\.md$")
ADR_INDEX_ROW_RE = re.compile(r"\[(\d+)\]\((docs/adr/[^)]+/(\d{3})-[^)]+\.md)\)")
CI_DIRECTORY_RE = re.compile(r"uv run --directory (\S+)")


def _packages_from_claude_md() -> list[str]:
    """CLAUDE.md lint 那行的 `<pkg> = a / b / c`。

    順序有意義：測試檔數斷言寫成 `四包各 15／3／3／3`，兩處要對得上。從文件解析
    而非硬編碼，順序本身才有護欄——硬編碼的話，日後有人重排文件裡的順序，
    數字會靜默對錯套件。
    """
    match = re.search(r"<pkg> = ([\w /]+)", CLAUDE_MD.read_text(encoding="utf-8"))
    if not match:
        raise RuntimeError("CLAUDE.md 找不到 `<pkg> = ...` 那行，措辭改了就要一併改本測試")
    return [name.strip() for name in match.group(1).split("/")]


def _libs() -> list[str]:
    return sorted(p.name for p in (REPO_ROOT / "libs").iterdir() if p.is_dir())


def _workspace_members() -> list[str]:
    """根 pyproject.toml 宣告的 workspace 成員，glob 展開成實際目錄。"""
    data = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    members: list[str] = []
    for pattern in data["tool"]["uv"]["workspace"]["members"]:
        if "*" in pattern:
            members.extend(
                sorted(
                    p.relative_to(REPO_ROOT).as_posix()
                    for p in REPO_ROOT.glob(pattern)
                    if p.is_dir()
                )
            )
        else:
            members.append(pattern)
    return members


PACKAGES = _packages_from_claude_md()
LIBS = _libs()
DOC_FILES = [
    CLAUDE_MD,
    ROOT_README,
    *(REPO_ROOT / pkg / "README.md" for pkg in PACKAGES),
    *(REPO_ROOT / "libs" / lib / "README.md" for lib in LIBS),
]


def _matrix_pkgs(job: dict) -> set[str]:
    """一個 job 的 matrix 會展開出哪些 `pkg` 值（`include` 算進來、`exclude` 扣掉）。

    `pkg` 不是清單就當沒有（`${{ fromJSON(...) }}` 這種動態寫法會是字串，逐字元
    迭代出來的單字元集合比誤報還難解讀）。`exclude` 只認單鍵的條目：GitHub 的
    語義是「列出的鍵全部吻合才排除」，多維 matrix 的 `{pkg: x, os: y}` 排掉的只是
    其中一種組合，`x` 仍會在別的 os 上跑，扣掉就成了誤報。
    """
    matrix = job.get("strategy", {}).get("matrix", {})
    if not isinstance(matrix, dict):  # matrix 整個是動態運算式
        return set()
    raw = matrix.get("pkg", [])
    pkgs = {str(p) for p in raw} if isinstance(raw, list) else set()
    pkgs |= {
        str(e["pkg"])
        for e in matrix.get("include", [])
        if isinstance(e, dict) and "pkg" in e
    }
    return pkgs - {
        str(e["pkg"])
        for e in matrix.get("exclude", [])
        if isinstance(e, dict) and set(e) == {"pkg"}
    }


def _ci_covered_members() -> set[str]:
    """ci.yml 實際會執行到的 workspace 成員。

    兩個來源合起來：各 job 的 matrix `pkg` 值，以及 step 的 `run` 裡寫死目標的
    `uv run --directory <成員>`（`${{ matrix.pkg }}` 這種佔位符濾掉）。第二個來源
    讓「某個成員從 matrix 拆成獨立 job」不必改本測試，第一個來源則涵蓋反方向。

    **用 YAML 解析而非掃文字**：改動初版用正則掃整份檔案，review 實測出十種合法的
    ci.yml 寫法會給錯答案——行尾註解、`include:`、清單項帶引號、indentless sequence、
    檔尾無換行都會誤紅，而被 `#` 註解掉的 job 更是靜默算成「有涵蓋」。這些全是
    YAML 語法層的事，交給 parser 才不會每加一種寫法就多一個破口。

    仍擋不到的是被 `if:` 條件停用的 job：條件要在 GitHub 的執行期才求得出值。
    見本檔測試的已知缺口。
    """
    workflow = yaml.safe_load(CI_WORKFLOW.read_text(encoding="utf-8"))
    jobs = workflow.get("jobs") if isinstance(workflow, dict) else None
    if not jobs:
        raise RuntimeError(
            f"{CI_WORKFLOW.relative_to(REPO_ROOT)} 解析不出任何 job。"
            "檔案搬走或結構改寫時要一併改本測試——算出空集合的話，這道護欄會靜默失效"
        )
    covered: set[str] = set()
    for job in jobs.values():
        if not isinstance(job, dict):  # job 主體被註解掉、只剩名字
            continue
        covered |= _matrix_pkgs(job)
        for step in job.get("steps", []):
            # 一個 step 可能是 `run: |` 多行區塊，逐個 match 取，不能只取第一個
            for raw in CI_DIRECTORY_RE.findall(str(step.get("run", ""))):
                target = raw.strip("\"'")
                if not target.startswith("${{"):
                    covered.add(target)
    return covered


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
    """遞迴計數，與 pytest 的收集方式一致——放進子目錄的測試檔也要算進去。"""
    return len(list((pkg_dir / "tests").rglob("test_*.py")))


def test_coverage_matches_workspace_members():
    """本測試的涵蓋範圍要等於 workspace 成員，新增成員不能靜默落在護欄外。

    `libs/*` 是 glob，新增一個 lib 會自動成為成員；涵蓋範圍若是硬編碼，該 lib 的
    README 連結永遠不會被驗。這與下方對缺檔採 fail loud 的立場是同一件事。
    """
    covered = {*PACKAGES, *(f"libs/{lib}" for lib in LIBS)}
    assert covered == set(_workspace_members()), (
        f"涵蓋範圍 {sorted(covered)} 與 workspace 成員 {_workspace_members()} 不一致"
    )


def test_ci_runs_every_workspace_member():
    """每個 workspace 成員都要被 CI 執行到，新增成員不能靜默落在 CI 之外。

    這是上一支斷言的另一半。`libs/*` 是 glob，新增一個 lib 時上一支會自動納入而
    通過；ci.yml 的清單卻是人工維護的，沒有這道斷言的話該成員的 lint 與測試靜默
    不執行——測試壞掉不會有訊號，正是本檔要消除的那種漂移。

    **只檢查單向（成員 ⊆ CI 涵蓋）**，不要求兩者相等：CI 跑了成員以外的目錄
    （例如 `uv run --directory .`）不是漂移，拿相等去驗只會製造與本斷言目的無關的
    紅燈。反方向——成員被刪掉但 ci.yml 還列著——本來就會讓該 job 以「目錄不存在」
    失敗，不需要這裡再驗一次。

    只驗「成員有沒有被 CI 執行到」，不驗各 job 實際跑了哪些指令：matrix job 的指令
    目標是佔位符，要把指令與目標配對得建模 GitHub Actions 的展開規則。兩種情況因此
    算「有涵蓋」而不在範圍內：列入了但只跑 ruff 沒跑 pytest、job 被 `if:` 條件停用
    （條件要在 GitHub 的執行期才求得出值）。
    """
    missing = set(_workspace_members()) - _ci_covered_members()
    assert not missing, (
        f"這些 workspace 成員沒被 CI 執行到：{sorted(missing)}。"
        "新增成員時 .github/workflows/ci.yml 要一併加上，否則它的 lint 與測試靜默不跑"
    )


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


def test_readme_adr_index_labels_match_linked_files():
    """索引表顯示的編號要與它連到的檔案編號相符。

    表格列是複製上一列改出來的，改了連結忘了改標籤不會有任何訊號——而編號正是
    被 issue 與 PR 直接引用的那個識別碼。
    """
    mismatched = [
        f"標籤 [{m.group(1)}] 連到 {m.group(2)}"
        for row in _readme_adr_index_rows()
        if (m := ADR_INDEX_ROW_RE.search(row)) and m.group(1).zfill(3) != m.group(3)
    ]
    assert not mismatched, f"README 索引表的編號標籤與連結對不上：{mismatched}"


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
    """CLAUDE.md 常用指令段寫的四包測試檔數，依 `<pkg> = ...` 的順序對位。"""
    text = CLAUDE_MD.read_text(encoding="utf-8")
    match = re.search(r"四包各 ([\d／]+) 支測試檔", text)
    assert match, "CLAUDE.md 找不到「四包各 N／N／N／N 支測試檔」的字樣"
    claimed = [int(n) for n in match.group(1).split("／")]
    assert len(claimed) == len(PACKAGES), (
        f"宣稱的數字有 {len(claimed)} 個，`<pkg> = ...` 列了 {len(PACKAGES)} 個套件"
    )
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
        "請一併更新本測試的涵蓋範圍，不要讓它靜默縮小"
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
