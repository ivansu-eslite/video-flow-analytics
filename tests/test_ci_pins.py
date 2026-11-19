"""CI 釘版測試：外部工具要釘 commit 指紋，uv 指令要鎖 lockfile。

為什麼要擋：
- `uses: actions/checkout@v7` 的 `v7` 是作者隨時能改指到別份程式的標籤，
  標籤被改指或帳號被冒用時，CI 會在沒有任何訊號下跑到別人的程式。釘成 40 位
  commit 指紋就換不掉。
- `uv run` 不帶 `--locked` 時，lockfile 跟 pyproject 對不上，uv 會自己重算版本
  照跑、照樣綠，CI 實際跑的版本就不是 `uv.lock` 記的那份。`--no-project`／`--with`
  臨時裝的套件根本不在 `uv.lock` 裡，每次都是當下最新版。

判法：不去猜一條 shell 指令從哪開始（`if`、`$(…)`、反引號、多行都有各自的寫法），
而是把每個 step 的 `run` 整段切成字，只要出現 `uv run`／`uv sync` 就檢查
它後面緊接 `--locked`，出現在哪都算。

保護不到（要寫在這裡，不要讓人以為有擋）：
- `${{ }}` 運算式展開後的內容。
- `bash -c '…'` 或腳本檔裡的 uv 指令。
- `.github/actions/` 底下自訂的 action（目前沒有）。
- 版號註解（`# v7.0.1`）跟指紋是不是同一版：要上網查才知道，靠寫的當下用
  `gh api repos/<owner>/<repo>/git/ref/tags/<tag>` 查一次，之後由 Dependabot 一起改。
- setup-uv 裝的 uv 本身是哪一版：指紋只管 action 的程式，不管它執行時再下載的東西。
- shlex 把單字中間的 `#` 當成註解開頭：`echo a#b; uv run x` 的 `uv run x` 會被略過。

只用 pytest、pyyaml、標準庫；跟 test_docs_contract.py 同一套跑法
（`uv run --locked --only-group docs-contract pytest tests/`）。
"""

from __future__ import annotations

import re
import shlex
import textwrap
from pathlib import Path
from typing import NamedTuple

import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
WORKFLOW_DIR = REPO_ROOT / ".github" / "workflows"
CI_WORKFLOW = WORKFLOW_DIR / "ci.yml"
DEPENDABOT = REPO_ROOT / ".github" / "dependabot.yml"
BEFORE_PINS = REPO_ROOT / "tests" / "fixtures" / "ci_before_pins.yml"

PUNCTUATION = "();<>|&\n`"
SHA40_RE = re.compile(r"[0-9a-f]{40}")
ACTION_RE = re.compile(r"[\w.-]+/[\w.-]+(?:/[\w./-]+)?@([0-9a-f]{40})")
DOCKER_RE = re.compile(r"docker://\S+@sha256:[0-9a-f]{64}")
VERSION_COMMENT_RE = re.compile(r"v\d+\.\d+\.\d+")
USES_RE = re.compile(r"(?:^|[\s{,])uses:(?:\s+(.*))?$")
HIDDEN_COMMAND_RE = re.compile(r"\buv\s+(?:run|sync|tool|pip)\b|\buvx\b|\bpip3?\s+install\b")
LOCK_DISABLING_ENV = ("UV_NO_SYNC", "UV_FROZEN")
BYPASS = "沒經過 uv.lock，改用 uv run --locked"
# 沒有 --locked 檢查、或會讓檢查失效的旗標，各自對應一句改法
UNLOCKED_FLAGS = {
    "--no-project": "--no-project 讓 uv 不讀 uv.lock，改用根 pyproject 的相依群組（--only-group）",
    "--frozen": "--frozen 不檢查 lockfile 過期沒，改用 --locked",
    "--no-sync": "--no-sync 讓 --locked 不檢查就照跑，拿掉",
}


class Violation(NamedTuple):
    where: str
    message: str

    def __str__(self) -> str:
        return f"  {self.where}：{self.message}"


# ---------------------------------------------------------------- uses

def uses_from_text(text: str) -> list[tuple[int, str, str]]:
    """逐行找 `uses:`，回傳（行號, 值, 行尾註解）。整行註解不算。"""
    found = []
    for lineno, line in enumerate(text.splitlines(), start=1):
        if line.lstrip().startswith("#"):
            continue
        comment = ""
        match = re.search(r"\s+#\s*(.*)$", line)
        if match:
            comment = match.group(1).strip()
            line = line[: match.start()]
        match = USES_RE.search(line.rstrip())
        if not match:
            continue
        value = (match.group(1) or "").strip()
        if "{" in line[: match.start() + 1] or "{" in line.split("uses:")[0]:
            value = re.split(r"[,}]", value)[0].strip()
        found.append((lineno, value, comment))
    return found


def uses_from_yaml(text: str) -> list[str]:
    """走 `jobs.*.uses` 與 `jobs.*.steps[*].uses`，回傳值。"""
    values = []
    for job in _jobs(text).values():
        if "uses" in job:
            values.append(str(job["uses"]))
        for step in job.get("steps") or []:
            if isinstance(step, dict) and "uses" in step:
                values.append(str(step["uses"]))
    return values


def uses_violations(name: str, text: str) -> list[Violation]:
    from_text = uses_from_text(text)
    from_yaml = uses_from_yaml(text)
    problems = []
    plain = [value for _, value, _ in from_text if not value.startswith(("'", '"', ">", "|"))]
    if len(from_text) != len(from_yaml) or any(v not in from_yaml for v in plain):
        problems.append(
            Violation(
                name,
                f"文字掃描找到 {len(from_text)} 筆 uses、yaml 解析找到 {len(from_yaml)} 筆，"
                "其中一邊漏看或寫法太特殊，改成一行一個 `uses: owner/repo@指紋  # vX.Y.Z`",
            )
        )
    for lineno, value, comment in from_text:
        reason = _uses_problem(value, comment)
        if reason:
            problems.append(Violation(f"{name}:{lineno}", f"{value or '（空）'} {reason}"))
    return problems


def _uses_problem(value: str, comment: str) -> str | None:
    fix = "寫成 owner/repo@<40 位小寫指紋>  # vX.Y.Z"
    if value.startswith("./"):
        return None
    if value.startswith(("'", '"')):
        return f"用引號包起來，掃描判不出來；{fix}"
    if value.startswith((">", "|")):
        return f"用折行寫法，掃描判不出來；{fix}"
    if value.startswith("docker://"):
        if DOCKER_RE.fullmatch(value):
            return None
        return "沒釘到映像檔指紋；寫成 docker://名稱@sha256:<64 位小寫 hex>"
    if not ACTION_RE.fullmatch(value):
        return f"沒釘到 commit 指紋；{fix}"
    if not VERSION_COMMENT_RE.fullmatch(comment):
        return f"缺行尾版號註解（現在是「{comment}」）；{fix}"
    return None


# ---------------------------------------------------------------- uv 指令

def _jobs(text: str) -> dict[str, dict]:
    data = yaml.safe_load(text)
    jobs = data.get("jobs") if isinstance(data, dict) else None
    return {k: v for k, v in (jobs or {}).items() if isinstance(v, dict)}


def _split_words(run: str) -> list[str]:
    lex = shlex.shlex(run.replace("\\\n", " "), posix=True, punctuation_chars=PUNCTUATION)
    lex.whitespace = " \t\r"
    lex.whitespace_split = True
    return list(lex)


def run_tokens(name: str, text: str) -> list[tuple[str, list[str] | None]]:
    """每個 step 的 `run` 切成字。位置：檔名 job 名 第 N 步（N 從 1 數，uses 步驟也算）。
    切不開的回傳 None。"""
    result = []
    for job_id, job in _jobs(text).items():
        for number, step in enumerate(job.get("steps") or [], start=1):
            if not isinstance(step, dict) or "run" not in step:
                continue
            where = f"{name} {job_id} 第 {number} 步"
            try:
                result.append((where, _split_words(str(step["run"]))))
            except ValueError:
                result.append((where, None))
    return result


def _is_boundary(word: str) -> bool:
    return bool(word) and set(word) <= set(PUNCTUATION)


def _end_of_command(words: list[str], start: int) -> int:
    for index in range(start, len(words)):
        if _is_boundary(words[index]):
            return index
    return len(words)


def _unlocked_flag_messages(args: list[str]) -> list[str]:
    messages = []
    for word in args:
        if word in UNLOCKED_FLAGS:
            messages.append(UNLOCKED_FLAGS[word])
        elif word == "--with" or word.startswith(("--with=", "--with-")):
            flag = word.split("=")[0]
            messages.append(f"{flag} 臨時裝的套件不在 uv.lock，改放根 pyproject 的相依群組")
        elif word.startswith("-w"):
            messages.append(f"{word} 是 --with 的短寫，臨時裝的套件不在 uv.lock")
    return messages


def uv_command_count(text: str) -> int:
    """掃到幾個 `uv run`／`uv sync`，給「掃描範圍不能變空」用。"""
    count = 0
    for _, words in run_tokens("", text):
        for i in range(len(words or []) - 1):
            if words[i] == "uv" and words[i + 1] in ("run", "sync"):
                count += 1
    return count


def _env_violations(name: str, text: str) -> list[Violation]:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        return []
    layers = [("workflow", data.get("env"))]
    for job_id, job in _jobs(text).items():
        layers.append((f"job {job_id}", job.get("env")))
        for number, step in enumerate(job.get("steps") or [], start=1):
            if isinstance(step, dict):
                layers.append((f"job {job_id} 第 {number} 步", step.get("env")))
    problems = []
    for layer, env in layers:
        for key in LOCK_DISABLING_ENV:
            if isinstance(env, dict) and key in env:
                problems.append(
                    Violation(
                        f"{name} env（{layer}）",
                        f"{key} 會讓 --locked 不檢查就照跑，拿掉",
                    )
                )
    return problems


def uv_violations(name: str, text: str) -> list[Violation]:
    problems = _env_violations(name, text)
    for where, words in run_tokens(name, text):
        if words is None:
            problems.append(Violation(where, "解析不了（引號沒配對？），拆成單純的指令"))
            continue
        for i, word in enumerate(words):
            following = words[i + 1 : i + 3]
            if word == "uv" and following[:1] in (["run"], ["sync"]):
                if following[1:2] != ["--locked"]:
                    problems.append(Violation(where, f"uv {following[0]} 後面要緊接 --locked"))
                end = _end_of_command(words, i + 2)
                problems += [
                    Violation(where, message)
                    for message in _unlocked_flag_messages(words[i + 2 : end])
                ]
            elif word == "uv" and following[:1] and following[0].startswith("-"):
                problems.append(
                    Violation(where, "uv 的選項寫在子指令後面：uv run --locked --directory x …")
                )
            elif word == "uvx" or (word == "uv" and following[:1] in (["tool"], ["pip"])):
                used = " ".join([word, *following[:1]]) if word == "uv" else word
                problems.append(Violation(where, f"{used} {BYPASS}"))
            elif word in ("pip", "pip3") and following[:1] == ["install"]:
                problems.append(Violation(where, f"{word} install {BYPASS}"))
            elif word == "-m" and following[:1] == ["pip"]:
                problems.append(Violation(where, f"python -m pip {BYPASS}"))
            elif any(key + "=" in word for key in LOCK_DISABLING_ENV):
                problems.append(Violation(where, f"{word} 會讓 --locked 不檢查就照跑，拿掉"))
            elif (
                any(ch.isspace() for ch in word)
                and HIDDEN_COMMAND_RE.search(word)
            ):
                problems.append(
                    Violation(where, "藏在引號裡的指令判不出有沒有鎖，拿到引號外面寫")
                )
    return problems


# ---------------------------------------------------------------- 測試用素材

def _workflows() -> list[Path]:
    return sorted([*WORKFLOW_DIR.glob("*.yml"), *WORKFLOW_DIR.glob("*.yaml")])


def _report(problems: list[Violation]) -> str:
    return "\n" + "\n".join(str(p) for p in problems)


def _step(step: str) -> str:
    """把一個 step 的內容包成最小的 workflow（第一行接在 `- ` 後面）。"""
    first, *rest = step.split("\n")
    lines = ["      - " + first, *("        " + line for line in rest)]
    return "jobs:\n  j:\n    runs-on: ubuntu-latest\n    steps:\n" + "\n".join(lines) + "\n"


def _run(script: str) -> str:
    return _step("run: |\n" + textwrap.indent(script, "  "))


SHA = "3d3c42e5aac5ba805825da76410c181273ba90b1"
DIGEST = "a" * 64

# ---------------------------------------------------------------- 真實檔案

def test_all_workflow_uses_are_pinned():
    problems = []
    for path in _workflows():
        problems += uses_violations(path.name, path.read_text(encoding="utf-8"))
    header = "這些外部工具沒有釘到 commit 指紋（寫成 owner/repo@<40 位指紋>  # vX.Y.Z）："
    assert not problems, header + _report(problems)


def test_all_uv_commands_are_locked():
    problems = []
    for path in _workflows():
        problems += uv_violations(path.name, path.read_text(encoding="utf-8"))
    assert not problems, "這些 uv 指令沒鎖住版本：" + _report(problems)


def test_scan_covers_ci_workflow():
    """掃描範圍不能靜默變空：ci.yml 改名、結構換了、glob 掃不到時要紅。"""
    assert CI_WORKFLOW in _workflows(), "workflow glob 掃不到 ci.yml"
    text = CI_WORKFLOW.read_text(encoding="utf-8")
    assert uses_from_text(text), "ci.yml 掃不到任何 uses，掃描方式失效了"
    assert uv_command_count(text), "ci.yml 掃不到任何 uv run／uv sync，掃描方式失效了"


def test_dependabot_keeps_actions_up_to_date():
    """指紋釘住後不會自己跟新版，要有 Dependabot 提升版 PR，不然永遠停在今天。"""
    assert DEPENDABOT.exists(), ".github/dependabot.yml 不見了，被釘住的 action 不會再有人升版"
    config = yaml.safe_load(DEPENDABOT.read_text(encoding="utf-8"))
    ecosystems = [u.get("package-ecosystem") for u in config.get("updates", [])]
    assert "github-actions" in ecosystems, "dependabot.yml 沒有 github-actions 這一項"


def test_flags_every_spot_in_ci_before_pins():
    """凍結的舊 ci.yml（21dc2cf 原封不動）要被點名恰好 12 個位置：6 處沒釘、6 個 step 沒鎖。"""
    text = BEFORE_PINS.read_text(encoding="utf-8")
    name = BEFORE_PINS.name
    flagged = {v.where for v in [*uses_violations(name, text), *uv_violations(name, text)]}
    expected = {f"{name}:{line}" for line in (21, 22, 45, 46, 60, 61)} | {
        f"{name} {job} 第 {step} 步"
        for job in ("docs-contract", "lint-test", "video-analyze")
        for step in (3, 4)
    }
    assert flagged == expected


def test_text_scan_and_yaml_parse_agree_on_workflows():
    for path in _workflows():
        text = path.read_text(encoding="utf-8")
        assert [v for _, v, _ in uses_from_text(text)] == uses_from_yaml(text), path.name


# ---------------------------------------------------------------- uses 案例

BAD_USES = {
    "tag v7": "uses: actions/checkout@v7  # v7.0.1",
    "branch": "uses: actions/checkout@main",
    "uppercase sha": f"uses: actions/checkout@{SHA.upper()}  # v7.0.1",
    "short sha": "uses: actions/checkout@3d3c42e  # v7.0.1",
    "no comment": f"uses: actions/checkout@{SHA}",
    "comment v7": f"uses: actions/checkout@{SHA}  # v7",
    "comment v7.0": f"uses: actions/checkout@{SHA}  # v7.0",
    "quoted": f'uses: "actions/checkout@{SHA}"  # v7.0.1',
    "flow style": "{uses: actions/checkout@v7}",
    "folded": f"uses: >-\n  actions/checkout@{SHA}",
    "docker tag": "uses: docker://alpine:1.0",
    "docker wrong digest": f"uses: docker://alpine@sha1:{DIGEST}",
}
GOOD_USES = {
    "pinned": f"uses: actions/checkout@{SHA}  # v7.0.1",
    "subpath": f"uses: owner/repo/sub/path@{SHA}  # v1.2.3",
    "local action": "uses: ./.github/actions/x",
    "commented out": "run: echo hi\n# uses: actions/checkout@v7",
    "docker digest": f"uses: docker://alpine@sha256:{DIGEST}",
    "flow pinned": f"{{uses: actions/checkout@{SHA}}}  # v7.0.1",
}


@pytest.mark.parametrize("step", BAD_USES.values(), ids=BAD_USES.keys())
def test_uses_violations_flag_unpinned(step):
    assert uses_violations("t.yml", _step(step))


@pytest.mark.parametrize("step", GOOD_USES.values(), ids=GOOD_USES.keys())
def test_uses_violations_allow_pinned(step):
    assert uses_violations("t.yml", _step(step)) == []


# ---------------------------------------------------------------- uv 案例

BAD_UV = {
    "locked after command": "uv run --directory x --locked pytest",
    "no locked": "uv run pytest",
    "sync no locked": "uv sync --package x",
    "frozen": "uv run --locked --frozen pytest",
    "frozen instead": "uv run --frozen pytest",
    "no-project": "uv run --locked --no-project pytest",
    "with": "uv run --locked --with x pytest",
    "with equals": "uv run --locked --with=x pytest",
    "with-requirements": "uv run --locked --with-requirements r.txt pytest",
    "short with": "uv run --locked -w x pytest",
    "no-sync": "uv run --locked --no-sync pytest",
    "env prefix no-sync": "UV_NO_SYNC=1 uv run --locked x",
    "env prefix frozen": "UV_FROZEN=1 uv run --locked x",
    "uvx": "uvx ruff check .",
    "uv tool": "uv tool run ruff",
    "uv pip": "uv pip install x",
    "pip install": "pip install x",
    "python -m pip": "python -m pip install x",
    "option before subcommand": "uv --directory x run pytest",
    "quiet before sync": "uv -q sync",
    "second command unlocked": "uv run --locked a && uv run b",
    "second line unlocked": "uv run --locked a\nuv run b",
    "assignment": "out=$(uv run x)",
    "backtick": "out=`uv run x`",
    "if": "if uv run x; then echo ok; fi",
    "env prefix": "FOO=1 uv run x",
    "quoted subshell": 'echo "$(uv run x)"',
    "continuation not locked": "uv run \\\n  --directory x pytest",
    "unparsable": 'echo "unterminated',
}
GOOD_UV = {
    "locked": "uv run --locked --directory x pytest",
    "sync locked": "uv sync --locked --package x",
    "continuation locked": "uv run --locked \\\n  --directory x pytest",
    "chained": "uv run --locked a && uv run --locked b",
    "lock check": "uv lock --check",
    "comment": "# uv run x\necho hi",
    "subshell locked": "out=$(uv run --locked x)",
    "ci-gate jq": (
        "bad=$(jq -r 'to_entries[] | select(.value.result != \"success\") "
        '| "\\(.key)：\\(.value.result)"\' <<<"$NEEDS")\n'
        'if [ -n "$bad" ]; then printf \'沒過的關：\\n%s\\n\' "$bad"; exit 1; fi\n'
        'echo "$(jq length <<<"$NEEDS") 關都過"'
    ),
}


@pytest.mark.parametrize("script", BAD_UV.values(), ids=BAD_UV.keys())
def test_uv_violations_flag_unlocked(script):
    assert uv_violations("t.yml", _run(script))


@pytest.mark.parametrize("script", GOOD_UV.values(), ids=GOOD_UV.keys())
def test_uv_violations_allow_locked(script):
    assert uv_violations("t.yml", _run(script)) == []


ENV_CASES = {
    "workflow": "env:\n  UV_NO_SYNC: '1'\n" + _run("uv run --locked x"),
    "job": _run("uv run --locked x").replace(
        "    runs-on", "    env:\n      UV_FROZEN: '1'\n    runs-on"
    ),
    "step": _step("env:\n  UV_NO_SYNC: '1'\nrun: uv run --locked x"),
}


@pytest.mark.parametrize("text", ENV_CASES.values(), ids=ENV_CASES.keys())
def test_uv_violations_flag_lock_disabling_env(text):
    problems = uv_violations("t.yml", text)
    assert problems and all("env" in p.where for p in problems)
