"""Agents, Animated · Episode 14 · The File System as Memory (Module 3 · Memory & Knowledge)

Why files are an agent's most durable working memory. Running example: a multi-session refactor of
tinyshop (every price moves from floats to integer cents, across five files). The runs are not drawn by
hand: the scene adds write_file and two system-prompt lines to the reference harness, runs the reference
agent_loop against scripted stand-ins for the model in two sessions over one in-memory "disk", and every
number on screen comes from those runs or from the cited sources (asserted). Session one ends when its
context passes the window (the scripted model refuses the call, as an API would); session two starts
from an empty messages list and learns everything about session one from NOTES.md and progress.md.

Arc: hook (a refactor too big for one session) · context is RAM, files are disk · what to write (Li's
"what gets lost first" → NOTES.md for the why, progress.md for the where; write_file + two prompt
lines) · session one (plan, notes, a failed test and its trap, ticks, the window fills, the context is
gone, the files stay) · session two (reads both files, resumes, finishes) · Anthropic's long-running
pattern · large outputs spilled to disk (Episode 11) · knowledge as a directory tree (OpenViking L0/L1/
L2) · plain Markdown memory (OpenClaw) · knowledge changes as reviewed diffs · Learn Claude Code's
state folders · recap.

Sources (facts dated October 2026):
- Li, AI Agents in Depth, ch. 2 (context compression): what gets lost first in compression is "early
  architectural decisions, the reasons behind constraints, and failed paths" (keep these in files).
- Anthropic, "Effective harnesses for long-running agents" (engineering blog, Nov 26, 2025): an
  initializer agent sets up the environment and a feature list in JSON (feature_list.json, each feature
  "passes": false at first; JSON because the model "is less likely to inappropriately change or overwrite
  JSON files compared to Markdown files"), a progress file and an initial git commit; each later coding
  session reads the progress file and the git log, works on one feature at a time, tests, commits. Two
  failure patterns: doing too much at once and "running out of context in the middle of its
  implementation"; and, later in a project, declaring the job done. Li, ch. 1, summarises it as an
  Initialization Agent plus an Execution Agent making "incremental progress each session and leaving
  clear handover artifacts"; failure modes "running out of context and declaring the task done
  prematurely".
- Learn Claude Code, s08 (learn.shareai.run/en/s08/): tool results over 200,000 characters (the code's
  max_bytes=200_000; prose "200KB") are persisted to .task_outputs/tool-results/ and the context keeps
  a <persisted-output> marker with a 2,000-character preview; transcripts go to .transcripts/ (JSONL).
  The marker's exact format is not shown there: ours is illustrative.
- Li, ch. 3 (the filesystem paradigm): OpenViking organises knowledge as a virtual file system under
  viking:// (resources/, user/memories/, agent/ with skills/ and memories/); every entry has L0 (a
  one-sentence summary, about 100 tokens), L1 (an overview, about 2,000 tokens), L2 (the full text,
  loaded on demand). How knowledge should be updated: a Proposer Agent "proposes the smallest complete
  diff on a working branch"; a Reviewer Agent, preferably from a different model family, checks it
  against "the raw evidence"; CI "checks formatting, links, metadata, and permission labels"; merge only
  after approval; indexes are rebuilt from the merged version; "The index is therefore a reproducible
  derivative, while the reviewed knowledge in Git is the source of truth."
- Li, ch. 5 (from Manus to OpenClaw): OpenClaw keeps long-term memory in MEMORY.md plus "Markdown logs
  archived by date", Markdown over a vector database; users can edit it ("if the Agent misremembers
  something, just delete that line"); "version control and rollback via Git"; "the file system is the
  hub of information flow".
- Learn Claude Code (per our research notes, s20 and the sessions that introduce each folder): the final
  harness keeps its state in .tasks/, .memory/, skills/, .transcripts/, .task_outputs/tool-results/,
  .mailboxes/, .worktrees/ and .scheduled_tasks.json.

Illustrative (for teaching): tinyshop's files and their contents, the refactor's five steps, Otto's
choices in both sessions, the text of NOTES.md and progress.md, the token sizes that fill session one's
window (the meter shows fractions only), the order log, the tool-call ids, OpenClaw's example memory
lines and log dates, the git hashes, the knowledge diff and its evidence. Computed and asserted below:
int(17.99 * 100) == 1798, the files' byte counts, the log's length, the preview length, the test counts,
the ticks.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

HERE = Path(__file__).resolve().read_text()
REF = (SERIES / "harness_reference.py").read_text()

# ---------------------------------------------------------------------------- tinyshop, simulated

TASK = "Switch tinyshop to integer cents."
CONTINUE = "Continue the cents refactor."
STEPS = ["cart.py", "checkout.py", "discounts.py", "receipt.py", "orders.py"]
OTHER_TESTS = 12  # tinyshop's other tests, all passing (as in Episodes 1-2 and 9)

CART0 = "def total(prices):\n    return sum(prices)\n"
CART_INT = "def to_cents(price):\n    return int(price * 100)\n\n\ndef total(prices):\n    return sum(to_cents(p) for p in prices)\n"
DISK = {  # the repo on disk (in memory here)
    "cart.py": CART0,
    "checkout.py": "SHIPPING = 4.99\n",
    "discounts.py": 'def apply_discount(amount, code):\n    if code == "SAVE10":\n        return round(amount * 0.9, 2)\n    return amount\n',
    "receipt.py": 'def line(amount):\n    return f"${amount:.2f}"\n',
    "orders.py": 'def log(order_id, total):\n    return f"order {order_id} total={total}"\n',
    "payments.py": "# Vendored from PayCo. Do not modify.\n",
    "tests/test_money.py": "def test_total():\n    assert total([17.99]) == 17.99\n",
}
EDITS = {  # each remaining step's change: (old_text, new_text)
    "checkout.py": ("SHIPPING = 4.99", "SHIPPING = 499"),
    "discounts.py": ("round(amount * 0.9, 2)", "round(amount * 0.9)"),
    "receipt.py": ("{amount:.2f}", "{amount / 100:.2f}"),
    "orders.py": ("total={total}", "total_cents={total}"),
}

# Li's three things that get lost first, as Otto writes them down
NOTE_DECISION = "- money is int cents"
NOTE_CONSTRAINT = "- leave payments.py: vendored"
NOTE_FAILED = "- tried int(): lost a cent"
NOTES0 = NOTE_DECISION + "\n" + NOTE_CONSTRAINT + "\n"
PROGRESS0 = "".join(f"- [ ] {s}\n" for s in STEPS)
TRAP = "int(17.99 * 100) == 1798"
assert 17.99 * 100 != 1799 and int(17.99 * 100) == 1798 and round(17.99 * 100) == 1799  # the trap is real Python


def tick(step):
    return ("progress.md", f"- [ ] {step}", f"- [x] {step}")


def pytest_q():
    """tinyshop's test run: the money test runs the real cart.py on disk against the test file's value."""
    ns = {}
    exec(DISK["cart.py"], ns)
    want = float(re.search(r"== ([\d.]+)", DISK["tests/test_money.py"]).group(1))
    failed = int(ns["total"]([17.99]) != want)
    passed = OTHER_TESTS + 1 - failed
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


def run_bash(command):
    if command == "pytest -q":
        return pytest_q()
    if command == "ls":
        return "\n".join(sorted({p.split("/")[0] for p in DISK}))
    raise ValueError(command)


def read_file(path):
    return DISK[path]


def edit_file(path, old_text, new_text):
    assert DISK[path].count(old_text) == 1, (path, old_text)
    DISK[path] = DISK[path].replace(old_text, new_text, 1)
    return f"edited {path}"


# ---------------------------------------------------------------------------- the harness (reference names)

SYSTEM = re.search(r'^SYSTEM = "(.*)"$', REF, re.M).group(1)
assert SYSTEM.startswith("You are Otto")
assert 'TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}' in REF
assert "def agent_loop(messages):" in REF and 'results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})' in REF
TOOLS = [{"name": "bash"}, {"name": "read_file"}, {"name": "edit_file"}]  # schemas abridged (Episode 3)
TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}
WRITE_FILE = {
    "name": "write_file",
    "description": "Create or overwrite a text file.",
    "input_schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                     "required": ["path", "content"]},
}

# ---- what Episode 14 adds (on screen; the handler writes to the in-memory disk here) ----
def write_file(path, content):  # noqa: E302
    DISK[path] = content  # on screen: open(path, "w").write(content), as harness_reference's edit_file
    return f"wrote {path}"

TOOLS.append(WRITE_FILE)  # noqa: E305
TOOL_HANDLERS["write_file"] = write_file
SYSTEM += "\nKeep NOTES.md and progress.md current."
SYSTEM += "\nAt the start, read both."
# ---- end of what Episode 14 adds ----

ADD_SRC = '''def write_file(path, content):
    open(path, "w").write(content)
    return f"wrote {path}"

TOOLS.append(WRITE_FILE)
TOOL_HANDLERS["write_file"] = write_file
SYSTEM += "\\nKeep NOTES.md and progress.md current."
SYSTEM += "\\nAt the start, read both."'''
_added = HERE.split("# ---- what Episode 14 adds")[1].split("# ---- end of what Episode 14 adds")[0]
assert all(line in _added for line in ADD_SRC.split("\n")[3:] if line)
assert 'return f"wrote {path}"' in _added and len(ADD_SRC.split("\n")) <= 10


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


# ---------------------------------------------------------------------------- the model, scripted

WINDOW = 200_000  # the context window, in tokens (illustrative; the meter shows fractions only)
READ_TOKENS = {"payments.py": 86_000, "cart.py": 19_000, "checkout.py": 31_000, "discounts.py": 62_000,
               "receipt.py": 12_000, "orders.py": 9_000}  # big modules (illustrative)


class ContextFull(Exception):
    """What the API answers when the prompt no longer fits the window."""


def result_tokens(call, content):
    if call.name == "read_file":
        return READ_TOKENS.get(call.input["path"], len(content) // 4 + 5)
    if call.name == "bash" and call.input["command"] == "pytest -q":
        return 2_600 if "failed" in content else 300  # a failure prints its traceback
    return 15


def context_tokens(messages):
    total, calls = 0, {}
    for msg in messages:
        if isinstance(msg["content"], str):
            total += 20
            continue
        for block in msg["content"]:
            if isinstance(block, dict):  # a tool_result
                total += result_tokens(calls[block["tool_use_id"]], block["content"])
            elif block.type == "tool_use":
                calls[block.id] = block
                total += 60 + len(block.input.get("content", "")) // 4
            else:
                total += 30
    return total


IDS = iter(range(1, 100))


class ScriptedModel:
    """Stands in for client.messages: follows a plan, and refuses a prompt larger than the window."""

    def __init__(self, plan, final=None):
        self.plan, self.final, self.calls, self.sent = plan, final, 0, []

    def create(self, system, messages, tools):
        used = context_tokens(messages)
        self.sent.append((system, used, [t["name"] for t in tools]))
        if used > WINDOW:
            raise ContextFull(used)
        step = self.plan[self.calls]
        self.calls += 1
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=self.final)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{next(IDS):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=None)


def agent_loop(messages):  # harness_reference.agent_loop, unchanged
    while True:
        response = client.messages.create(system=SYSTEM, messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


def edit(path, old, new):
    return ("edit_file", {"path": path, "old_text": old, "new_text": new})


def edit_tick(step):
    path, old, new = tick(step)
    return edit(path, old, new)


# session one (Otto's choices are illustrative): look around, write the plan and the notes, then work
PLAN1 = [
    ("bash", {"command": "ls"}),
    ("read_file", {"path": "payments.py"}),
    ("write_file", {"path": "progress.md", "content": PROGRESS0}),
    ("write_file", {"path": "NOTES.md", "content": NOTES0}),
    ("read_file", {"path": "cart.py"}),
    edit("cart.py", CART0, CART_INT),
    edit("tests/test_money.py", "== 17.99", "== 1799"),
    ("bash", {"command": "pytest -q"}),
    edit("cart.py", "int(price * 100)", "round(price * 100)"),
    ("bash", {"command": "pytest -q"}),
    edit("NOTES.md", NOTES0, NOTES0 + NOTE_FAILED + "\n"),
    edit_tick("cart.py"),
    ("read_file", {"path": "checkout.py"}),
    edit("checkout.py", *EDITS["checkout.py"]),
    ("bash", {"command": "pytest -q"}),
    edit_tick("checkout.py"),
    ("read_file", {"path": "discounts.py"}),
    edit("discounts.py", *EDITS["discounts.py"]),  # never sent: the window is full before this call
]
MESSAGES1 = [{"role": "user", "content": TASK}]
client.messages = S1 = ScriptedModel(PLAN1)
try:
    agent_loop(MESSAGES1)
    ENDED = False
except ContextFull:
    ENDED = True
AFTER1 = dict(DISK)

# session two: an empty context; the plan follows the system prompt (read both files first), then
# the first unticked step in progress.md, the way NOTES.md says (round, never int; payments.py untouched)
FINAL = "All five files use cents. 13 passed."
PLAN2 = [
    ("read_file", {"path": "progress.md"}),
    ("read_file", {"path": "NOTES.md"}),
    ("read_file", {"path": "discounts.py"}),
    edit("discounts.py", *EDITS["discounts.py"]),
    ("bash", {"command": "pytest -q"}),
    edit_tick("discounts.py"),
    edit("receipt.py", *EDITS["receipt.py"]),
    edit_tick("receipt.py"),
    edit("orders.py", *EDITS["orders.py"]),
    ("bash", {"command": "pytest -q"}),
    edit_tick("orders.py"),
    None,
]
MESSAGES2 = [{"role": "user", "content": CONTINUE}]
client.messages = S2 = ScriptedModel(PLAN2, FINAL)
ANSWER = agent_loop(MESSAGES2)
AFTER2 = dict(DISK)


def calls_of(messages):
    return [b for msg in messages if msg["role"] == "assistant" for b in msg["content"] if getattr(b, "type", "") == "tool_use"]


def results_of(messages):
    return [b["content"] for msg in messages if msg["role"] == "user" and not isinstance(msg["content"], str) for b in msg["content"]]


CALLS1, RESULTS1 = calls_of(MESSAGES1), results_of(MESSAGES1)
CALLS2, RESULTS2 = calls_of(MESSAGES2), results_of(MESSAGES2)


def unticked(progress):
    return [line[6:] for line in progress.splitlines() if line.startswith("- [ ] ")]


def ticked(progress):
    return [line[6:] for line in progress.splitlines() if line.startswith("- [x] ")]


# session one: the window filled after the discounts read; the files hold the plan and the notes
assert ENDED and S1.calls == 17 and len(MESSAGES1) == 35 and S1.sent[-1][1] > WINDOW >= S1.sent[-2][1]
assert len(STEPS) == 5  # "five files"
assert RESULTS1[7] == "1 failed, 12 passed" and RESULTS1[9] == f"{OTHER_TESTS + 1} passed"  # the trap, then the fix
assert ticked(AFTER1["progress.md"]) == ["cart.py", "checkout.py"] and len(unticked(AFTER1["progress.md"])) == 3  # "two of five"
assert AFTER1["NOTES.md"].splitlines() == [NOTE_DECISION, NOTE_CONSTRAINT, NOTE_FAILED]
assert all(s[0] == SYSTEM for s in S1.sent + S2.sent) and all("write_file" in s[2] for s in S1.sent + S2.sent)
# session two knew nothing but its task, then read both files from disk
assert "1 failed" not in repr(MESSAGES2) and all(c.input.get("path") != "payments.py" for c in CALLS2)
assert [c.input["path"] for c in CALLS2[:2]] == ["progress.md", "NOTES.md"] and "read both" in SYSTEM
assert RESULTS2[0] == AFTER1["progress.md"] and RESULTS2[1] == AFTER1["NOTES.md"]
NEXT = unticked(RESULTS2[0])[0]
assert NEXT == "discounts.py" and CALLS2[3].input["path"] == NEXT and CALLS2[3].name == "edit_file"
assert "round(" in CALLS2[3].input["new_text"] and "int(" not in CALLS2[3].input["new_text"] and "int()" in RESULTS2[1]
FINAL_TESTS = RESULTS2[9]
assert ticked(AFTER2["progress.md"]) == STEPS and FINAL_TESTS == RESULTS2[4] == f"{OTHER_TESTS + 1} passed"  # "thirteen"
assert ANSWER.content[0].text == FINAL and max(s[1] for s in S2.sent) < WINDOW / 2
BYTES = {name: len(AFTER1[name].encode()) for name in ("progress.md", "NOTES.md")}
TOTAL_BYTES = sum(BYTES.values())
assert 100 < TOTAL_BYTES < 200  # "less than two hundred bytes"


def frac(messages, k):
    """The meter after the first k messages (fraction of the window)."""
    return min(1.0, context_tokens(messages[:k]) / WINDOW)


# ---------------------------------------------------------------------------- the order log (Episode 11's spill)

LIMIT = 200_000  # characters (learn-claude-code s08)
PREVIEW = 2_000  # characters kept in the context
LOG = "".join(
    f"2026-10-{1 + i // 250:02d} {9 + i % 250 // 25:02d}:{i % 25 * 2:02d}:00 order {i + 1} total={(i * 37 % 9000 + 99) / 100:.2f}"
    f"{' ERROR declined' if (i + 1) % 1000 == 0 else ' OK'}\n"
    for i in range(6_800)
)
LOG_CHARS = len(LOG)
LOG_LINES = LOG.count("\n")
ERRORS = sum(" ERROR " in line for line in LOG.splitlines())
SAVED = ".task_outputs/tool-results/toolu_31.txt"
PREVIEW_TEXT = LOG[:PREVIEW]
assert 300_000 < LOG_CHARS < 400_000 and LOG_CHARS > LIMIT  # "over three hundred thousand", over the limit
assert len(PREVIEW_TEXT) == PREVIEW and ERRORS == 6

# ---------------------------------------------------------------------------- OpenViking (Li, ch. 3)

L0_TOKENS, L1_TOKENS = 100, 2_000

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"
ROW_H = 0.48
MD_T2C = {r"\[x\]": MINT, r"\[ \]": FAINT, r"^- ": MUTED, r"◂ next": SUN}
JSON_T2C = {r"\"[a-z]+\"(?=:)": AZURE, r"(?<=: )\"[^\"]*\"": MINT, r"\bfalse\b": ROSE, r"\btrue\b": MINT}
DIFF_T2C = {r"^-[^\n]*": ROSE, r"^\+[^\n]*": MINT}
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep14")


def cite(s):
    return text(s, 20, MUTED)


def fmt(n):
    return f"{n:,}"


def baseline_of(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def baseline(words):
    return baseline_of(words, [c for c in words.text if not c.isspace()])


def sit(words, y):
    """Put a Text's baseline at y."""
    words.shift((y - baseline(words)) * m.UP)
    return words


def first_line(words):
    """The glyphs of a multi-line Text's first line, and its characters."""
    line = words.text.split("\n")[0]
    chars = [c for c in line if not c.isspace()]
    return words[: len(chars)], chars


def line_glyphs(words, i):
    lines = words.text.split("\n")
    start = sum(sum(not c.isspace() for c in l) for l in lines[:i])
    n = sum(not c.isspace() for c in lines[i])
    return words[start:start + n]


def row(role, body, width, accent=None):
    """One message on one line: a stripe in the block's colour, the role, the content (as Episode 9)."""
    accent = accent or ROLES[role][0]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0, fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    if body.width > width - 2.13:
        raise ValueError(f"row body too wide ({body.text!r}: {body.width:.2f} > {width - 2.13:.2f})")
    body.move_to(box).align_to(box, m.LEFT).shift(1.98 * m.RIGHT)
    base = box.get_y() - 0.09
    sit(label, base)
    sit(body, base)
    return m.VGroup(box, stripe, label, body)


def tall(role, body, width, accent=None):
    """A message whose content has several lines: the role beside the first line."""
    accent = accent or ROLES[role][0]
    if body.width > width - 2.13:
        raise ValueError(f"tall body too wide ({body.width:.2f} > {width - 2.13:.2f})")
    box = card(width, body.height + 0.34, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=box.height - 0.16, stroke_width=0, fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    body.align_to(box, m.UP).shift(0.17 * m.DOWN).align_to(box, m.LEFT).shift(1.98 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    label.align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    sit(label, baseline_of(*first_line(body)))
    return m.VGroup(box, stripe, label, body)


def stack(role, body, width, accent=None):
    """A message with its role above a wide body (kit.message, with ready-made words)."""
    color = ROLES[role][0]
    label = code(role, 22, color, weight=m.BOLD)
    inner = m.VGroup(label, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    if body.width > width - 0.5:
        raise ValueError(f"stacked body too wide ({body.width:.2f} > {width - 0.5:.2f})")
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0, fill_color=accent or color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def short(out, room=28):
    """A tool result on one row: a listing joined, else the first line, cut to fit."""
    lines = out.strip().splitlines()
    if all(" " not in l for l in lines) and len(lines) > 1:  # a listing (ls)
        first, more = " ".join(lines), True
    else:
        first, more = lines[0], len(lines) > 1
    if len(first) <= room:
        return first + (" …" if more else "")
    if "(" in first[:room]:
        return first[:first.index("(") + 1] + "…)"
    cut = first[:room]
    if " " in cut[8:]:
        cut = cut[:cut.rindex(" ")]
    return cut.rstrip(" ,(:") + " …"


def call_words(block):
    a = block.input
    name = block.name
    if name in ("write_file", "edit_file"):
        src = f'{name}("{a["path"].split("/")[-1]}", …)'
    elif name == "read_file":
        src = f'read_file("{a["path"]}")'
    else:
        src = f'bash("{a["command"]}")'
    return code(src, 20, SUN)


def result_words(content):
    color = ROSE if "failed" in content else MINT
    return code(short(content), 20, color)


def md(s, size=20):
    return code(s, size, INK, t2c=MD_T2C)


def mono_chip(word, color, fill, size=24, height=0.66, pad=0.5):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def word_chip(word, color, fill, size=24, height=0.66, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def strike(words, color=ROSE):
    return m.Line(words.get_left() + 0.08 * m.LEFT, words.get_right() + 0.08 * m.RIGHT).set_stroke(color, 3).move_to(words)


class Slot:
    """One place on screen whose content is swapped (a caption, a title)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15):
        on_screen = {id(x) for x in self.scene.get_mobject_family_members()}
        anims = []
        if self.current is not None:
            out = m.FadeOut(self.current, shift=shift * m.UP)
            if id(self.current) not in on_screen:
                out.introducer = True
            anims.append(out)
        if new is not None:
            anims.append(m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


class FileCard:
    """A file on disk drawn as a card: its name, then one Text per line (so lines can change alone).
    Lines are separate top-level mobjects; the frame is box + rule + name."""

    def __init__(self, name, lines, center, width, slots=None, size=20, line_h=0.36, accent=ORCHID, t2c=None,
                 stroke=EDGE):
        self.size, self.t2c, self.line_h = size, (MD_T2C if t2c is None else t2c), line_h
        n = slots or len(lines)
        title_h = 0.52
        self.box = card(width, title_h + n * line_h + 0.26, stroke, CARD, radius=0.16, stroke=2).move_to(center)
        self.name = code(name, 20, accent, weight=m.BOLD)
        self.name.align_to(self.box, m.LEFT).shift(0.3 * m.RIGHT)
        sit(self.name, self.box.get_top()[1] - 0.34)
        y = self.box.get_top()[1] - title_h
        self.rule = m.Line([self.box.get_left()[0] + 0.12, y, 0], [self.box.get_right()[0] - 0.12, y, 0]).set_stroke(EDGE, 1.5)
        self.left = self.box.get_left()[0] + 0.32
        self.top_base = y - 0.1 - 0.62 * line_h
        self.lines = [self.make(s, i) for i, s in enumerate(lines)]
        self.frame_g = m.VGroup(self.box, self.rule, self.name)

    def base(self, i):
        return self.top_base - i * self.line_h

    def make(self, s, i):
        words = code(s, self.size, INK, t2c=self.t2c)
        words.align_to([self.left, 0, 0], m.LEFT)
        return sit(words, self.base(i))

    def frame(self):
        return self.frame_g

    def outro(self):
        return m.AnimationGroup(m.FadeOut(self.frame_g), *[m.FadeOut(l) for l in self.lines])

    def everything(self):
        return m.VGroup(self.box, self.rule, self.name, *self.lines)

    def intro(self, source=None, lag=0.12):
        if source is None:
            first = m.FadeIn(self.frame(), shift=0.15 * m.UP)
        else:
            first = m.FadeIn(self.frame(), target_position=source, scale=0.5)
        return m.Succession(first, m.LaggedStart(*[m.FadeIn(l, shift=0.1 * m.RIGHT) for l in self.lines], lag_ratio=lag))

    def put(self, i, s, source=None):
        """Change (or add) line i: the old line fades, the new one fades in."""
        new = self.make(s, i)
        anims = []
        if i < len(self.lines):
            anims.append(m.FadeOut(self.lines[i]))
            self.lines[i] = new
        else:
            self.lines.append(new)
        anims.append(m.FadeIn(new, target_position=source) if source is not None else m.FadeIn(new, shift=0.1 * m.RIGHT))
        return m.AnimationGroup(*anims)

    def right_note(self, words):
        """A small note at the right end of the title row (e.g. its size in bytes)."""
        words.align_to(self.box, m.RIGHT).shift(0.3 * m.LEFT)
        return sit(words, baseline(self.name))


class Column(ContextColumn):
    """The kit's ContextColumn; push() may fly the new message in from `source`; wipe() empties it."""

    def push(self, msg, run_time=0.9, source=None):
        room = self.top - self.bottom
        total = sum(item.height + self.gap for item, _ in self.slots) + msg.height
        leaving = []
        while self.slots and total > room + 1e-6:
            item, _ = self.slots.pop(0)
            leaving.append(item)
            total -= item.height + self.gap
        anims = []
        if leaving:
            rise = (self.top - self.slots[0][1]) if self.slots else room
            anims += [m.FadeOut(item, shift=rise * m.UP) for item in leaving]
            y = self.top
            for slot in self.slots:
                item = slot[0]
                anims.append(item.animate.move_to([item.get_x(), y - item.height / 2, 0]))
                slot[1] = y
                y -= item.height + self.gap
        y = (self.slots[-1][1] - self.slots[-1][0].height - self.gap) if self.slots else self.top
        msg.move_to([self.panel.get_x(), 0, 0]).align_to([0, y, 0], m.UP)
        anims.append(m.FadeIn(msg, shift=0.25 * m.UP) if source is None else m.FadeIn(msg, target_position=source))
        self.slots.append([msg, y])
        self.items = [item for item, _ in self.slots]
        return m.AnimationGroup(*anims, run_time=run_time)

    def wipe(self, run_time=1.4):
        gone = list(self.items)
        self.slots, self.items = [], []
        return m.LaggedStart(*[m.FadeOut(item, shift=0.35 * m.DOWN, scale=0.9) for item in reversed(gone)], lag_ratio=0.05, run_time=run_time)


# ---------------------------------------------------------------------------- the episode


class Ep14FileSystemAsMemory(Episode):
    number = 14

    def lesson(self):
        self.hook()
        self.ram_and_disk()
        self.what_to_write()
        self.session_one()
        self.session_two()
        self.long_running()
        self.spill()
        self.knowledge_tree()
        self.plain_files()
        self.reviewed_diffs()
        self.all_state()
        self.recap()

    # helpers

    def speak(self, script, *animations):
        """say(), with every animation of an object not yet on screen marked as an introducer: a row
        pushed and then scrolled (or faded) in the same line must not show before its cue (Episode 9)."""
        on_screen = {id(x) for x in self.get_mobject_family_members()}

        def leaves(a):
            subs = getattr(a, "animations", None)
            if subs:
                for sub in subs:
                    yield from leaves(sub)
            else:
                yield a

        for a in animations:
            for leaf in leaves(a):
                mob = getattr(leaf, "mobject", None)
                if mob is not None and not leaf.is_introducer() and id(mob) not in on_screen:
                    leaf.introducer = True
        self.say(script, *animations)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Otto (the series' agent) is asked to move tinyshop's money from floats to integer cents ({TASK!r}) across {len(STEPS)} files: {', '.join(STEPS)}. Too much for one session: halfway through the context fills and the session ends; the next session starts empty. The task and files are illustrative. Facts as of October 2026.")
        bot = otto(1.1).move_to([-5.3, 1.6, 0])
        person = icon("user", 0.95).move_to([5.75, 1.3, 0])
        ask = bubble(text(TASK, 28, INK), AZURE, AZURE_FILL, tail="right")
        ask.move_to([2.4, 1.95, 0])
        ask.shift((person.get_left()[0] - 0.15 - ask[0].get_vertices()[2][0]) * m.RIGHT)

        files = m.VGroup(*[chip(s, EDGE, CARD, width=code(s, 22).width + 0.55, height=0.66, size=26, mono=True) for s in STEPS])
        files.arrange(m.RIGHT, buff=0.32).move_to([0.25, -0.35, 0])
        five = text(f"{len(STEPS)} files", 24, MUTED).next_to(files, m.DOWN, buff=0.18)
        checks = m.VGroup(*[text("✓", 30, MINT, weight=m.BOLD).next_to(f, m.UP, buff=0.12) for f in files[:2]])
        meter = Meter(8.6, 0.0, SUN, 0.26).move_to([0.95, -1.75, 0])
        m_lab = code("context", 20, MUTED).next_to(meter, m.LEFT, buff=0.3)
        ended = pill("session ends", ROSE, ROSE_FILL, 22).move_to([0.95, -2.4, 0])
        quiz = title_text("?", 64, SUN).next_to(bot, m.RIGHT, buff=0.3).shift(0.3 * m.UP)
        question = title_text("How does it pick up where it left off?", 42, INK).move_to([0, -3.1, 0])

        self.speak(
            "[Say you ask Otto for a big refactor]: [every price in tinyshop moves from floats to whole cents], [across five files.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(person, shift=0.2 * m.UP), lag_ratio=0.3),
            m.FadeIn(ask, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(five), m.LaggedStart(*[m.FadeIn(f, shift=0.2 * m.UP) for f in files], lag_ratio=0.15)),
        )
        self.speak(
            "[That's too much for one session.] [Halfway through, the context fills up], [and the session ends.]",
            m.AnimationGroup(m.Succession(m.AnimationGroup(m.FadeIn(meter), m.FadeIn(m_lab), run_time=0.4), meter.fill_to(0.45)), m.FadeIn(checks[0], scale=0.6)),
            m.AnimationGroup(meter.fill_to(1.0), m.FadeIn(checks[1], scale=0.6)),
            m.FadeIn(ended, scale=1.1),
        )
        self.speak(
            "[Tomorrow, a fresh Otto starts with an empty context.] [So how does it pick up where it left off?]",
            m.AnimationGroup(meter.fill_to(0.0), *[m.FadeOut(c) for c in checks], m.FadeOut(ended), m.FadeIn(quiz, scale=0.6)),
            m.Write(question),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ RAM and disk

    def ram_and_disk(self):
        self.next_section("1 · RAM and disk", notes="The analogy: the context window is the agent's RAM (everything the model sees, fast, but bounded by a token budget and gone when the session ends or is compacted, Episode 11); files are its disk (one tool call away, any size, they survive compaction, crashes and new sessions). Memory across conversations is Episode 12. Facts as of October 2026.")
        panels = []
        spec = [("ep14-ram", "Context", "= RAM", SUN, ["fast: right in front of the model", "small: a fixed token budget", "wiped when the session ends"]),
                ("ep14-disk", "Files", "= disk", ORCHID, ["one tool call away", "as big as you need", "survives compaction, new sessions"])]
        for k, (pic, name, eq, color, bullets) in enumerate(spec):
            box = card(6.25, 4.75, color, CARD, radius=0.22, stroke=2.5).move_to([(-1) ** (k + 1) * 3.4, -0.2, 0])
            ic = icon(pic, 0.5 if k == 0 else 0.7)
            title = title_text(name, 34, INK)
            eqw = title_text(eq, 34, color)
            head = m.VGroup(ic, title, eqw).arrange(m.RIGHT, buff=0.3)
            eqw.align_to(title, m.DOWN)
            head.move_to(box).align_to(box, m.UP).shift(0.45 * m.DOWN).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
            rows = m.VGroup(*[text(b, 24, INK) for b in bullets]).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
            rows.next_to(head, m.DOWN, buff=0.45).align_to(head, m.LEFT)
            panels.append((box, head, rows))
        (lbox, lhead, lrows), (rbox, rhead, rrows) = panels
        rhead.shift((lhead[1].get_y() - rhead[1].get_y()) * m.UP)
        rrows.align_to(lrows, m.UP)
        lrows[2].set_color(ROSE)
        rrows[2].set_color(MINT)
        gauge = Meter(4.1, 0.0, SUN, 0.24).move_to([0, lbox.get_bottom()[1] + 0.7, 0]).align_to(lrows, m.LEFT)
        wiped = code("wiped", 20, ROSE, weight=m.BOLD).next_to(gauge, m.RIGHT, buff=0.3)
        docs = m.VGroup(*[mono_chip(s, ORCHID, ORCHID_FILL, size=22) for s in ("NOTES.md", "progress.md")]).arrange(m.RIGHT, buff=0.35)
        docs.move_to([0, rbox.get_bottom()[1] + 0.7, 0]).align_to(rrows, m.LEFT)
        stays = code("kept", 20, MINT, weight=m.BOLD).next_to(docs, m.RIGHT, buff=0.3)
        statement = title_text("Files: an agent's most durable working memory.", 34, INK).move_to([0, -3.25, 0])

        self.speak(
            "[Think of the context window as RAM]: [fast, and right in front of the model], [but small], [and wiped when the session ends.]",
            m.AnimationGroup(self.new_header("01 · ram and disk", "Context is RAM, files are disk"), m.FadeIn(lbox), m.FadeIn(lhead, shift=0.15 * m.UP)),
            m.FadeIn(lrows[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(lrows[1], shift=0.1 * m.UP), m.Succession(m.FadeIn(gauge, run_time=0.3), gauge.fill_to(0.85))),
            m.AnimationGroup(m.FadeIn(lrows[2], shift=0.1 * m.UP), gauge.fill_to(0.0), m.FadeIn(wiped)),
        )
        self.speak(
            "[Files are the disk]: [one tool call away], [as big as you need], [and they survive compaction, crashes, and new sessions.]",
            m.AnimationGroup(m.FadeIn(rbox), m.FadeIn(rhead, shift=0.15 * m.UP), m.FadeIn(docs, shift=0.1 * m.UP)),
            m.FadeIn(rrows[0], shift=0.1 * m.UP),
            m.FadeIn(rrows[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(rrows[2], shift=0.1 * m.UP), m.FadeIn(stays), m.Circumscribe(docs, color=MINT, buff=0.08)),
        )
        self.speak(
            "[That makes files an agent's most durable working memory.]",
            m.Write(statement),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what to write

    def what_to_write(self):
        self.next_section("2 · What to write", notes=f"Li, AI Agents in Depth, ch. 2 (compression): what gets lost first is 'early architectural decisions, the reasons behind constraints, and failed paths' — keep them in files. Otto's NOTES.md (the why) and progress.md (the where) as they stand at the end of session one (asserted against the run): {AFTER1['NOTES.md'].splitlines()} / {AFTER1['progress.md'].splitlines()}. What Episode 14 adds to harness_reference.py, in its style: a write_file tool and two system-prompt lines; agent_loop is unchanged. Facts as of October 2026.")
        # left: a long session folded into a summary, and what falls out of it
        stripes = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.04, width=2.0, height=0.13, stroke_width=0,
                               fill_color=[AZURE, SUN, MINT, SUN, MINT, SUN, MINT, CORAL][i], fill_opacity=0.9)
            for i in range(8)
        ]).arrange(m.DOWN, buff=0.09)
        s_box = card(2.6, stripes.height + 0.5, EDGE, PANEL, radius=0.16, stroke=2)
        session = m.VGroup(s_box, stripes.move_to(s_box)).move_to([-5.15, 1.3, 0])
        s_cap = code("a long session", 20, MUTED).next_to(session, m.DOWN, buff=0.12)
        summ = card(2.6, 1.15, EDGE, CARD, radius=0.16, stroke=2).move_to([-1.85, 1.3, 0])
        summ_words = m.VGroup(code("summary", 20, INK, weight=m.BOLD), *[
            m.RoundedRectangle(corner_radius=0.03, width=w, height=0.09, stroke_width=0, fill_color=FAINT, fill_opacity=1) for w in (1.8, 1.4)
        ]).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT).move_to(summ)
        fold = arrow(session.get_right(), summ.get_left(), MUTED, buff=0.15)
        lost_cap = text("lost first", 24, ROSE, weight=m.BOLD).move_to([-3.6, -0.85, 0]).align_to([-6.55, 0, 0], m.LEFT)
        lost = m.VGroup(*[word_chip(s, ROSE, ROSE_FILL, size=24) for s in ("early decisions", "reasons behind constraints", "failed paths")])
        lost.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).next_to(lost_cap, m.DOWN, buff=0.22).align_to(lost_cap, m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 2").next_to(lost_cap, m.RIGHT, buff=0.45)
        sit(src, baseline(lost_cap))

        # right: the two files, as session one left them
        notes_lines = AFTER1["NOTES.md"].splitlines()
        prog_lines = AFTER1["progress.md"].splitlines()
        notes = FileCard("NOTES.md", notes_lines, center=(3.65, 1.4, 0), width=5.7, size=22, line_h=0.42)
        prog = FileCard("progress.md", prog_lines, center=(3.65, 0, 0), width=5.7, size=22, line_h=0.42)
        gap = 0.3
        prog_shift = (notes.box.get_bottom()[1] - gap - prog.box.get_top()[1])
        prog = FileCard("progress.md", prog_lines, center=(3.65, prog_shift, 0), width=5.7, size=22, line_h=0.42)
        why = text("the why", 24, ORCHID, weight=m.SEMIBOLD)
        notes.right_note(why)
        where = text("the where", 24, ORCHID, weight=m.SEMIBOLD)
        prog.right_note(where)
        k_next = prog_lines.index(f"- [ ] {NEXT}")
        nxt = code("◂ next", 20, SUN, weight=m.BOLD).next_to(prog.lines[k_next], m.RIGHT, buff=0.35)
        done_lines = m.VGroup(*[prog.lines[i] for i, l in enumerate(prog_lines) if l.startswith("- [x]")])

        # the harness: one tool, two prompt lines
        cc = code_block(ADD_SRC, size=22, title="harness · what this episode adds")
        cc.move_to([0, 0.35, 0])
        body = cc[1]
        src_lines = ADD_SRC.split("\n")

        def bar(a, b):
            ga, gb = line_glyphs(body, a), line_glyphs(body, b)
            top = baseline_of(ga, [c for c in src_lines[a] if not c.isspace()]) + 0.3
            bot = baseline_of(gb, [c for c in src_lines[b] if not c.isspace()]) - 0.12
            left, right = cc[0].get_left()[0] + 0.12, cc[0].get_right()[0] - 0.1
            rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=SUN, fill_opacity=0.14)
            mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=SUN, fill_opacity=1)
            rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
            mark.move_to([left + 0.025, (top + bot) / 2, 0])
            return m.VGroup(rect, mark).set_z_index(1)

        body.set_z_index(2)
        bar_tool = bar(0, 5)
        bar_sys = bar(6, 7)
        same = pill("agent_loop · unchanged", MINT, MINT_FILL, 22, mono=True).next_to(cc, m.DOWN, buff=0.4)

        self.speak(
            "[When a long session is summarised, or simply ends], [some things get lost first.]",
            m.AnimationGroup(self.new_header("02 · what to write", "The why and the where"), m.FadeIn(session), m.FadeIn(s_cap),
                             m.GrowArrow(fold), m.FadeIn(summ), m.FadeIn(summ_words), lag_ratio=0.15),
            m.FadeIn(lost_cap, shift=0.1 * m.DOWN),
        )
        self.speak(
            "[AI Agents in Depth names three]: [early decisions], [the reasons behind constraints], [and the paths that failed.]",
            m.FadeIn(src),
            *[m.FadeIn(c, target_position=summ, scale=0.6) for c in lost],
        )
        self.speak(
            "[So Otto writes them into its notes file]: [the why behind the work.]",
            m.Succession(m.FadeIn(notes.frame(), shift=0.15 * m.UP),
                         m.AnimationGroup(*[m.FadeIn(l, target_position=c) for l, c in zip(notes.lines, lost)], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(why, shift=0.1 * m.LEFT), *[c.animate.set_opacity(0.35) for c in lost]),
        )
        self.speak(
            "[And progress.md keeps the where]: [every step], [which ones are done], [and what comes next.]",
            m.AnimationGroup(m.FadeIn(prog.frame(), shift=0.15 * m.UP), m.FadeIn(where, shift=0.1 * m.LEFT)),
            m.LaggedStart(*[m.FadeIn(l, shift=0.1 * m.RIGHT) for l in prog.lines], lag_ratio=0.15),
            m.Circumscribe(done_lines, color=MINT, buff=0.08),
            m.FadeIn(nxt, shift=0.15 * m.LEFT),
        )
        everything = [session, s_cap, summ, summ_words, fold, lost_cap, *lost, src, why, where, nxt]
        self.speak(
            "[In the harness, that's one small tool], [and two lines of system prompt]: [keep the files current, and read them first.]",
            m.Succession(m.AnimationGroup(*[m.FadeOut(x) for x in everything], notes.outro(), prog.outro()), m.AnimationGroup(m.FadeIn(cc, shift=0.15 * m.UP), m.FadeIn(bar_tool))),
            m.AnimationGroup(m.Transform(bar_tool, bar_sys), m.FadeIn(same, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(line_glyphs(body, 6), color=ORCHID, buff=0.06), m.Circumscribe(line_glyphs(body, 7), color=ORCHID, buff=0.06), lag_ratio=0.5),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the stage for both sessions

    def build_stage(self):
        st = NS()
        top, bot = 1.95, -3.5
        st.ctx = Column(center=(-3.15, (top + bot) / 2, 0), width=7.0, height=top - bot, title="", gap=0.1)
        st.meter = Meter(2.3, 0.0, SUN, 0.16)
        st.meter.move_to([st.ctx.panel.get_right()[0] - 1.15, top + 0.27, 0])
        st.title = Slot(self)
        st.title_y = top + 0.27
        st.disk = card(6.0, top - bot, EDGE, PANEL, radius=0.2, stroke=2).move_to([3.65, (top + bot) / 2, 0])
        disk_icon = icon("ep14-disk", 0.34)
        disk_t = code("on disk · tinyshop/", 20, ORCHID, weight=m.BOLD)
        st.disk_title = m.VGroup(disk_icon, disk_t).arrange(m.RIGHT, buff=0.15)
        st.disk_title.align_to(st.disk, m.LEFT).shift(0.05 * m.RIGHT)
        sit(disk_t, st.title_y - 0.07)
        disk_icon.set_y(disk_t.get_y())
        st.note = Slot(self)
        w = 5.5
        st.prog = FileCard("progress.md", [], center=(3.65, 0, 0), width=w, slots=5, size=20, line_h=0.36)
        st.prog = FileCard("progress.md", [], center=(3.65, top - 0.25 - st.prog.box.height / 2, 0), width=w, slots=5, size=20, line_h=0.36)
        st.notes = FileCard("NOTES.md", [], center=(3.65, 0, 0), width=w, slots=3, size=20, line_h=0.36)
        st.notes = FileCard("NOTES.md", [], center=(3.65, st.prog.box.get_bottom()[1] - 0.3 - st.notes.box.height / 2, 0), width=w, slots=3, size=20, line_h=0.36)
        st.width = st.ctx.width - 0.36
        return st

    def title_words(self, st, s, color=MUTED):
        words = code(s, 20, color, weight=m.BOLD if color != MUTED else m.NORMAL)
        words.align_to(st.ctx.panel, m.LEFT).shift(0.05 * m.RIGHT)
        return sit(words, st.title_y - 0.07)

    def note_words(self, st, s, color):
        words = code(s, 20, color, weight=m.BOLD)
        words.align_to(st.disk, m.RIGHT).shift(0.05 * m.LEFT)
        return sit(words, st.title_y - 0.07)

    def msg_rows(self, st, messages):
        rows = []
        for msg in messages:
            content = msg["content"]
            if isinstance(content, str):
                rows.append(row("user", text(content, 20, INK), st.width))
                continue
            block = content[0]
            if isinstance(block, dict):
                rows.append(row("user", result_words(block["content"]), st.width, MINT))
            elif block.type == "text":
                rows.append(row("assistant", text(block.text, 20, INK), st.width))
            else:
                rows.append(row("assistant", call_words(block), st.width, SUN))
        return rows

    def pushes(self, st, rows, messages, k0, run_time=0.42, extra=None):
        """Push rows[k0:k0+n] one after another, the meter following the context (messages[:k+1])."""
        extra = extra or {}
        steps = []
        for k in range(k0, k0 + len(rows)):
            parts = [st.ctx.push(rows[k - k0], run_time=run_time), st.meter.fill_to(frac(messages, k + 1), run_time=run_time)]
            parts += [e() if callable(e) else e for e in extra.get(k, [])]  # callables are built once the row is placed
            steps.append(m.AnimationGroup(*parts))
        return m.Succession(*steps)

    # ------------------------------------------------------------------ session one

    def session_one(self):
        self.next_section("3 · Session one", notes=f"The run (harness_reference.agent_loop, unchanged, against a scripted model): Otto lists the repo, reads payments.py (vendored), writes progress.md ({BYTES['progress.md']} bytes) and NOTES.md, converts cart.py with int(price * 100) — pytest -q says {RESULTS1[7]!r} because {TRAP} (real Python) — fixes it with round(), notes the failed path, ticks cart.py, converts checkout.py, ticks it, reads discounts.py. The next call no longer fits the window (the scripted model refuses it, as an API would): session one ends after {S1.calls} calls and {len(MESSAGES1)} messages. Token sizes and the window are illustrative (the meter shows fractions); every message, file and test result is the run's (asserted). Facts as of October 2026.")
        st = self.st = self.build_stage()
        rows = self.msg_rows(st, MESSAGES1)
        call_k = lambda c: 1 + 2 * c  # noqa: E731  (message index of call c; its result follows)
        fail_row = rows[call_k(7) + 1]
        prog_src, notes_src = rows[call_k(2)], rows[call_k(3)]
        st.prog.lines = [st.prog.make(l, i) for i, l in enumerate(PROGRESS0.splitlines())]
        st.notes.lines = [st.notes.make(l, i) for i, l in enumerate(NOTES0.splitlines())]
        trap = code(TRAP, 20, ROSE).align_to([st.notes.left, 0, 0], m.LEFT)
        sit(trap, st.notes.base(2))
        title1 = self.title_words(st, "Otto's context · session 1")
        full = self.title_words(st, "session 1 · context full", ROSE)
        kept = self.note_words(st, "still on disk", MINT)

        def P(c0, c1, **kw):  # the rows of calls c0..c1 (each call and its result), one after another
            a, b = call_k(c0), call_k(c1) + 2
            return self.pushes(st, rows[a:b], MESSAGES1, a, **kw)

        intro = m.AnimationGroup(self.new_header("03 · session one", "Writing it down"), m.FadeIn(st.ctx.panel), st.title.to(title1),
                                 m.FadeIn(st.meter), m.FadeIn(st.disk), m.FadeIn(st.disk_title))
        self.speak(
            "[Session one.] [Otto looks around the repo, writes its plan to progress.md, and its first decisions to NOTES.md.]",
            intro,
            self.pushes(st, rows[:call_k(3) + 2], MESSAGES1, 0, extra={
                call_k(2): [lambda: st.prog.intro(source=prog_src)],
                call_k(3): [lambda: st.notes.intro(source=notes_src)],
            }),
        )
        self.speak(
            "[Then it gets to work]: reading, editing, running the tests, [while the context fills, turn by turn.]",
            P(4, 7),
            m.Circumscribe(st.meter, color=SUN, buff=0.08),
        )
        self.speak(
            "[A test fails]: [converting a price to cents loses one cent.] [Otto fixes it, and writes the trap into its notes.]",
            m.Circumscribe(fail_row, color=ROSE, buff=0.05),
            m.FadeIn(trap, shift=0.1 * m.UP),
            P(8, 10, extra={call_k(10): [m.FadeOut(trap), lambda: st.notes.put(2, NOTE_FAILED, source=rows[call_k(10)])]}),
        )
        self.speak(
            "[Each finished file gets a tick in progress.md: first the cart, then checkout.]",
            P(11, 15, run_time=0.36, extra={
                call_k(11) + 1: [lambda: st.prog.put(0, "- [x] cart.py")],
                call_k(15) + 1: [lambda: st.prog.put(1, "- [x] checkout.py")],
            }),
        )
        self.speak(
            "[Then the window is full.] [The session ends], [and everything in the context is gone.]",
            P(16, 16, run_time=0.55),
            st.title.to(full),
            m.AnimationGroup(st.ctx.wipe(), st.meter.fill_to(0.0, run_time=1.4)),
        )
        self.speak(
            "[But the two files are still on disk.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(st.prog.box, st.notes.box), color=MINT, buff=0.08), st.note.to(kept)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ session two

    def session_two(self):
        self.next_section("4 · Session two", notes=f"Session two starts from an empty messages list with {CONTINUE!r}. Following its system prompt ('At the start, read both.'), Otto reads progress.md (2 of 5 ticked; next: {NEXT}) and NOTES.md (whole cents; leave payments.py; the int() trap) — {TOTAL_BYTES} bytes in all ({BYTES['progress.md']} + {BYTES['NOTES.md']}, computed). It converts discounts.py with round() (asserted: not int()), never touches payments.py, ticks every step, and ends with {FINAL_TESTS!r}. Session two peaks at {max(s[1] for s in S2.sent) / WINDOW:.0%} of the illustrative window. Facts as of October 2026.")
        st = self.st
        title2 = self.title_words(st, "Otto's context · session 2")
        rows = self.msg_rows(st, MESSAGES2)
        # the two file reads come back as tall rows that look like the files
        rows[2] = tall("user", md(RESULTS2[0].rstrip()), st.width, MINT)
        rows[4] = tall("user", md(RESULTS2[1].rstrip()), st.width, MINT)
        k_next = RESULTS2[0].splitlines().index(f"- [ ] {NEXT}")
        nxt = code("◂ next", 20, SUN, weight=m.BOLD).next_to(st.prog.lines[k_next], m.RIGHT, buff=0.3)
        size_p = st.prog.right_note(code(f"{BYTES['progress.md']} bytes", 20, MUTED))
        size_n = st.notes.right_note(code(f"{BYTES['NOTES.md']} bytes", 20, MUTED))
        total = self.note_words(st, f"= {TOTAL_BYTES} bytes", ORCHID)
        done_note = self.note_words(st, f"{FINAL_TESTS} ✓", MINT)
        call_k = lambda c: 1 + 2 * c  # noqa: E731

        def P(c0, c1, **kw):
            a, b = call_k(c0), call_k(c1) + 2
            return self.pushes(st, rows[a:b], MESSAGES2, a, **kw)

        def lines_of(r, idx):
            return m.VGroup(*[line_glyphs(r[3], i) for i in idx])

        def ticks(c):  # the card change that goes with tick call c
            new = CALLS2[c].input["new_text"]
            return lambda: st.prog.put(STEPS.index(new[6:]), new)

        prog_row, notes_row = rows[2], rows[4]
        ticked_idx = [i for i, l in enumerate(RESULTS2[0].splitlines()) if l.startswith("- [x]")]
        assert CALLS2[5].input["new_text"] == f"- [x] {NEXT}" and STEPS.index(NEXT) == k_next
        self.speak(
            "[Session two starts with an empty context], [and a short task: carry on.]",
            m.AnimationGroup(self.new_header("04 · session two", "Picking up from the files"), st.title.to(title2)),
            self.pushes(st, rows[:1], MESSAGES2, 0),
        )
        self.speak(
            "[First, Otto reads progress.md]: [two of five are done], [and discounts are next.]",
            m.Succession(self.pushes(st, rows[1:2], MESSAGES2, 1),
                         m.AnimationGroup(st.ctx.push(prog_row, run_time=0.8, source=st.prog.box), st.meter.fill_to(frac(MESSAGES2, 3), run_time=0.8))),
            m.Circumscribe(lines_of(prog_row, ticked_idx), color=MINT, buff=0.06),
            m.AnimationGroup(m.Circumscribe(line_glyphs(prog_row[3], k_next), color=SUN, buff=0.06), m.FadeIn(nxt, shift=0.15 * m.LEFT)),
        )
        self.speak(
            "[Then NOTES.md brings back the why]: [whole cents], [leave the payments file alone], [and the rounding trap.]",
            m.Succession(self.pushes(st, rows[3:4], MESSAGES2, 3),
                         m.AnimationGroup(st.ctx.push(notes_row, run_time=0.8, source=st.notes.box), st.meter.fill_to(frac(MESSAGES2, 5), run_time=0.8))),
            m.Circumscribe(line_glyphs(notes_row[3], 0), color=ORCHID, buff=0.06),
            m.Circumscribe(line_glyphs(notes_row[3], 1), color=ORCHID, buff=0.06),
            m.Circumscribe(line_glyphs(notes_row[3], 2), color=ORCHID, buff=0.06),
        )
        self.speak(
            "[Less than two hundred bytes], [and Otto carries on, right where it stopped.]",
            m.AnimationGroup(st.note.to(total), m.FadeIn(size_p), m.FadeIn(size_n)),
            P(2, 5, run_time=0.32, extra={call_k(5) + 1: [ticks(5), m.FadeOut(nxt)]}),
        )
        self.speak(
            "[By the end of session two, every box is ticked], [and all thirteen tests pass.]",
            self.pushes(st, rows[call_k(6):], MESSAGES2, call_k(6), run_time=0.3,
                        extra={call_k(7) + 1: [ticks(7)], call_k(10) + 1: [ticks(10)]}),
            m.AnimationGroup(st.note.to(done_note), m.Circumscribe(st.prog.box, color=MINT, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ long-running agents

    def long_running(self):
        self.next_section("5 · Long-running agents", notes="Anthropic, 'Effective harnesses for long-running agents' (Nov 2025): an initializer agent writes a feature list in JSON (feature_list.json; every feature 'passes': false at first — JSON because the model 'is less likely to inappropriately change or overwrite JSON files compared to Markdown files'), a progress file and a first git commit; every later session reads the progress file and git log, works on one feature, tests it, commits. Failure patterns: doing too much at once and 'running out of context in the middle of its implementation'; declaring the job done. Li, AI Agents in Depth, ch. 1: 'incremental progress each session and leaving clear handover artifacts'; failure modes 'running out of context and declaring the task done prematurely'. Verified 'done': Episode 22. tinyshop's task names are illustrative. Facts as of October 2026.")
        names = [("initializer", ORCHID, ORCHID_FILL), ("session 1", CORAL, CORAL_FILL), ("session 2", CORAL, CORAL_FILL), ("session 3", CORAL, CORAL_FILL)]
        nodes = m.VGroup()
        for i, (label, color, fill) in enumerate(names):
            box = card(2.35, 0.9, color, fill, radius=0.18, stroke=2.5)
            nodes.add(m.VGroup(box, text(label, 26, INK, weight=m.BOLD).move_to(box)))
        nodes.arrange(m.RIGHT, buff=0.6).move_to([-0.55, 1.75, 0])
        more = text("until\ndone", 22, MUTED).next_to(nodes, m.RIGHT, buff=0.45)
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(nodes, nodes[1:])])
        link_more = arrow(nodes[-1].get_right(), more.get_left(), MUTED, buff=0.08)
        commits = m.VGroup(*[pill("git commit", MINT, MINT_FILL, 20, mono=True).next_to(n, m.DOWN, buff=0.25) for n in nodes[1:]])
        src_a = cite("Anthropic, 2025").to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)

        tasks = ["cart", "checkout", "discounts"]
        jline = lambda t, ok: f'{{"task": "{t}", "passes": {"true" if ok else "false"}}},'  # noqa: E731
        feats = FileCard("feature_list.json", [jline(t, False) for t in tasks], center=(-3.35, -0.85, 0), width=6.5,
                         size=20, line_h=0.4, accent=SUN, t2c=JSON_T2C)
        assert json.loads("[" + "".join(jline(t, False) for t in tasks).rstrip(",") + "]")[0]["passes"] is False
        fails = [feats.lines[i][-7:-2] for i in range(3)]  # the "false" glyphs (before "},")

        quote = text("“…incremental progress each session and\nleaving clear handover artifacts.”", 26, INK)
        quote.move_to([3.5, -0.55, 0])
        q_src = cite("Li, AI Agents in Depth · ch. 1").next_to(quote, m.DOWN, buff=0.15).align_to(quote, m.RIGHT)
        fcards = m.VGroup()
        for bad, good in (("running out of context mid-task", "fix: one task per session"), ("declaring the job done too early", "fix: done means a test passes")):
            box = card(6.45, 1.2, ROSE, CARD, radius=0.16, stroke=2)
            words = m.VGroup(text(bad, 24, ROSE, weight=m.BOLD), text(good, 22, MINT, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            fcards.add(m.VGroup(box, words))
        fcards.arrange(m.RIGHT, buff=0.3).move_to([0, -2.75, 0])
        ep22 = cite("Episode 22").move_to(fcards[1][0]).align_to(fcards[1][0], m.RIGHT).shift(0.3 * m.LEFT)
        sit(ep22, baseline(fcards[1][1][1]))

        self.speak(
            "[Anthropic describes the same pattern for long-running agents], [working across many sessions.]",
            m.AnimationGroup(self.new_header("05 · long-running", "One task per session"), m.FadeIn(src_a)),
            m.LaggedStart(*[m.FadeIn(n, shift=0.15 * m.RIGHT) for n in nodes], *[m.GrowArrow(a) for a in links], m.GrowArrow(link_more), m.FadeIn(more), lag_ratio=0.12),
        )
        self.speak(
            "[First, an initializer agent writes the full task list as JSON], [every item marked as failing.]",
            m.AnimationGroup(m.Circumscribe(nodes[0], color=ORCHID, buff=0.06), feats.intro(source=nodes[0])),
            m.AnimationGroup(*[m.Circumscribe(f, color=ROSE, buff=0.05) for f in fails], lag_ratio=0.2),
        )
        self.speak(
            "[Then each session reads the progress file and the git log], [does one task and tests it], [then commits.]",
            m.AnimationGroup(pulse(links[0], CORAL), m.Circumscribe(nodes[1], color=CORAL, buff=0.06)),
            feats.put(0, jline(tasks[0], True)),
            m.AnimationGroup(m.FadeIn(commits[0], shift=0.1 * m.DOWN),
                             m.Succession(m.Wait(0.4), m.AnimationGroup(pulse(links[1], CORAL), m.FadeIn(commits[1], shift=0.1 * m.DOWN), feats.put(1, jline(tasks[1], True)))),
                             m.Succession(m.Wait(0.9), m.AnimationGroup(pulse(links[2], CORAL), m.FadeIn(commits[2], shift=0.1 * m.DOWN), feats.put(2, jline(tasks[2], True))))),
        )
        self.speak(
            "[AI Agents in Depth sums it up]: [incremental progress each session], [leaving clear handover artifacts.]",
            m.FadeIn(q_src),
            m.FadeIn(quote, shift=0.1 * m.UP),
            m.Circumscribe(commits, color=MINT, buff=0.06),
        )
        self.speak(
            "[It guards against two failures]: [running out of context mid-task], [and declaring the job done too early.]",
            m.Circumscribe(quote, color=MUTED, buff=0.08),
            m.FadeIn(fcards[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(fcards[1], shift=0.15 * m.UP), m.FadeIn(ep22)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ spill to disk

    def spill(self):
        self.next_section("6 · Spill to disk", notes=f"Learn Claude Code, s08 (and Episode 11): a tool result over {fmt(LIMIT)} characters is persisted to .task_outputs/tool-results/ and the context keeps a <persisted-output> marker with a {fmt(PREVIEW)}-character preview. tinyshop's order log is generated here: {fmt(LOG_LINES)} lines, {fmt(LOG_CHARS)} characters, {ERRORS} ERROR lines (computed). The marker's layout and the tool-call id are illustrative (s08 does not show the format). Facts as of October 2026.")
        call = mono_chip('bash("cat orders.log")', SUN, SUN_FILL, size=24).move_to([-3.45, 1.85, 0])
        big = card(6.3, 1.5, ROSE, ROSE_FILL, radius=0.18, stroke=2.5).move_to([-3.45, 0.35, 0])
        big_n = title_text(f"{fmt(LOG_CHARS)} characters", 40, ROSE)
        big_sub = text(f"orders.log · {fmt(LOG_LINES)} lines", 22, MUTED)
        m.VGroup(big_n, big_sub).arrange(m.DOWN, buff=0.12).move_to(big)
        a1 = arrow(call.get_bottom(), big.get_top(), MUTED, buff=0.08)
        rule = text(f"over {fmt(LIMIT)}: the harness saves it", 24, INK).next_to(big, m.DOWN, buff=0.3)
        r_src = cite("learn-claude-code · s08").next_to(rule, m.DOWN, buff=0.1)
        saved = FileCard(SAVED, [f"{fmt(LOG_CHARS)} characters, in full"], center=(-3.45, -2.55, 0), width=6.5, size=20, line_h=0.38,
                         accent=ORCHID, t2c={r"^\d[\d,]*": ROSE})

        col = Column(center=(3.5, -0.75, 0), width=6.4, height=5.5, title="Otto's context", gap=0.12)
        w = col.width - 0.36
        u_call = row("assistant", code('bash("cat orders.log")', 20, SUN), w, SUN)
        p_lines = ["<persisted-output>", "saved: …/tool-results/toolu_31.txt", f"first {fmt(PREVIEW)} characters:",
                   PREVIEW_TEXT.splitlines()[0][:29] + "…", "</persisted-output>"]
        marker = stack("user", code("\n".join(p_lines), 20, MINT, t2c={r"</?persisted-output>": SUN, r"^first[^\n]*": MUTED}), w, MINT)
        g_call = stack("assistant", code('bash("grep -c ERROR …/toolu_31.txt")', 20, SUN), w, SUN)
        g_res = row("user", code(str(ERRORS), 20, MINT), w, MINT)
        ep11 = cite("Episode 11 · Compaction").next_to(col.panel, m.DOWN, buff=0.1).align_to(col.panel, m.RIGHT)

        self.speak(
            "[Files also catch what's too big for the context.] [Say Otto prints tinyshop's order log]: [over three hundred thousand characters.]",
            m.AnimationGroup(self.new_header("06 · spill to disk", "Too big for the context"), m.FadeIn(col.panel), m.FadeIn(col.title)),
            m.AnimationGroup(m.FadeIn(call, shift=0.15 * m.DOWN), col.push(u_call, source=call)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(big, shift=0.15 * m.DOWN), m.FadeIn(big_n), m.FadeIn(big_sub)),
        )
        self.speak(
            "[That's over the two hundred thousand limit], [so the harness writes it to disk], [and the context gets a two-thousand-character preview, plus the path.]",
            m.AnimationGroup(m.FadeIn(rule, shift=0.1 * m.UP), m.FadeIn(r_src)),
            m.AnimationGroup(saved.intro(source=big), big.animate.set_fill(ROSE_FILL, opacity=0.4)),
            col.push(marker, run_time=1.0, source=rule),
        )
        self.speak(
            "[When Otto needs a detail], [it searches the saved file with grep], [as episode eleven shows.]",
            m.AnimationGroup(col.push(g_call), m.Circumscribe(saved.box, color=ORCHID, buff=0.06)),
            col.push(g_res, source=saved.box),
            m.FadeIn(ep11, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ knowledge as a directory tree

    def knowledge_tree(self):
        self.next_section("7 · Knowledge as a tree", notes=f"Li, AI Agents in Depth, ch. 3 (the filesystem paradigm): OpenViking organises an agent's knowledge as a virtual file system under viking:// — resources/ (external documents, code, web pages), user/memories/ (preferences, habits), agent/ (skills/, memories/: experience). Each entry has three levels: L0, a one-sentence summary of about {L0_TOKENS} tokens; L1, an overview of about {fmt(L1_TOKENS)} tokens; L2, the full text, loaded on demand. The same progressive disclosure as skills (Episode 10). The tinyshop entries are illustrative. Facts as of October 2026.")
        root = code("viking://", 22, ORCHID, weight=m.BOLD)
        entries = ["resources/", "resources/tinyshop-docs/", "user/", "user/memories/", "agent/", "agent/skills/", "agent/memories/"]
        tree = file_tree(entries, size=22)
        indent = 1.8 * code("M", 22).width
        tree.next_to(root, m.DOWN, buff=0.15, aligned_edge=m.LEFT).shift(indent * m.RIGHT)
        group = m.VGroup(root, tree).move_to([-4.0, 0.3, 0]).align_to([-6.5, 0, 0], m.LEFT)
        notes = [("documents, code, web", 0), ("the user's preferences", 3), ("skills and experience", 5)]
        annots = m.VGroup()
        for s, i in notes:
            a = text(s, 22, MUTED)
            a.move_to([0, tree[i].get_y(), 0]).align_to([-3.55, 0, 0], m.LEFT)
            sit(a, baseline(tree[i]))
            annots.add(a)
        src = cite("Li, AI Agents in Depth · ch. 3").next_to(group, m.DOWN, buff=0.45).align_to(group, m.LEFT)

        spec = [("L0", "summary", f"≈{L0_TOKENS}", "tokens", 1.65), ("L1", "overview", f"≈{fmt(L1_TOKENS)}", "tokens", 2.3), ("L2", "full text", "", "on demand", 2.95)]
        levels = m.VGroup()
        for name, what, n, unit, h in spec:
            box = card(1.9, h, ORCHID, ORCHID_FILL, radius=0.18, stroke=2.5)
            parts = [title_text(name, 34, INK), text(what, 22, INK)]
            parts += [code(n, 22, SUN, weight=m.BOLD), text(unit, 20, MUTED)] if n else [text(unit, 22, SUN, weight=m.SEMIBOLD)]
            words = m.VGroup(*parts).arrange(m.DOWN, buff=0.1)
            words.move_to(box).align_to(box, m.UP).shift(0.2 * m.DOWN)
            room = words.get_bottom()[1] - box.get_bottom()[1] - 0.25
            n_bars = int(room // 0.22) if room > 0.3 else 0
            bars = m.VGroup(*[m.RoundedRectangle(corner_radius=0.03, width=1.3 - 0.35 * (j % 3 == 2), height=0.08, stroke_width=0,
                                                 fill_color=ORCHID, fill_opacity=0.35) for j in range(n_bars)]).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
            if n_bars:
                bars.next_to(words, m.DOWN, buff=0.22).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            levels.add(m.VGroup(box, words, bars))
        levels.arrange(m.RIGHT, buff=0.32, aligned_edge=m.DOWN).move_to([3.55, -0.7, 0])
        hop_y = levels.get_bottom()[1] + 0.45
        hops = m.VGroup(*[arrow([a[0].get_right()[0], hop_y, 0], [b[0].get_left()[0], hop_y, 0], MUTED, buff=0.02) for a, b in zip(levels, levels[1:])])
        entry = code("tinyshop-docs/", 20, MUTED).next_to(levels, m.UP, buff=0.3).align_to(levels, m.LEFT)
        skim = text("skim the summaries, open only what you need", 24, INK).move_to([3.55, -2.85, 0])
        ep10 = cite("progressive disclosure · Episode 10").next_to(skim, m.DOWN, buff=0.1)

        self.speak(
            "[Scale that up], [and a whole knowledge base becomes a directory tree.]",
            self.new_header("07 · knowledge tree", "Knowledge as a directory tree"),
            m.AnimationGroup(m.FadeIn(root, shift=0.1 * m.DOWN), m.FadeIn(src)),
        )
        self.speak(
            "[AI Agents in Depth describes OpenViking]: [outside resources], [the user's memories], [and the agent's own skills and experience.]",
            m.Circumscribe(root, color=ORCHID, buff=0.06),
            m.AnimationGroup(m.FadeIn(tree[0:2], shift=0.1 * m.RIGHT), m.FadeIn(annots[0])),
            m.AnimationGroup(m.FadeIn(tree[2:4], shift=0.1 * m.RIGHT), m.FadeIn(annots[1])),
            m.AnimationGroup(m.FadeIn(tree[4:7], shift=0.1 * m.RIGHT), m.FadeIn(annots[2])),
        )
        self.speak(
            "[Every entry comes in three sizes]: [a summary of about a hundred tokens], [an overview of about two thousand], [and the full text, on demand.]",
            m.AnimationGroup(m.FadeIn(entry, shift=0.1 * m.UP), m.Circumscribe(tree[1], color=SUN, buff=0.05)),
            m.FadeIn(levels[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(hops[0]), m.FadeIn(levels[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(hops[1]), m.FadeIn(levels[2], shift=0.2 * m.UP)),
        )
        self.speak(
            "[Progressive disclosure again, as with skills in episode ten.]",
            m.AnimationGroup(m.FadeIn(skim, shift=0.1 * m.UP), m.FadeIn(ep10), m.Circumscribe(levels[0], color=SUN, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ plain files

    def plain_files(self):
        self.next_section("8 · Plain files", notes="Li, AI Agents in Depth, ch. 5 (from Manus to OpenClaw): OpenClaw keeps long-term memory in MEMORY.md (high-level facts and preferences) plus Markdown logs archived by date, Markdown over a vector database; users can read and edit it — 'if the Agent misremembers something, just delete that line' — with version control and rollback via Git. How memories are written and chosen: Episode 12. The memory lines, log dates and git hashes are illustrative. Facts as of October 2026.")
        tree = file_tree(["workspace/", "workspace/MEMORY.md", "workspace/memory/", "workspace/memory/2026-10-20.md",
                          "workspace/memory/2026-10-21.md", "workspace/memory/2026-10-22.md"], size=22, highlight=("MEMORY.md",))
        tree.move_to([-4.3, 0.85, 0]).align_to([-6.5, 0, 0], m.LEFT)
        db = icon(recolored("database", MUTED, CARD_HI, prefix="ep14"), 0.95).move_to([-5.6, -2.0, 0])
        cross = m.VGroup(m.Line(db.get_corner(m.UL), db.get_corner(m.DR)), m.Line(db.get_corner(m.DL), db.get_corner(m.UR))).set_stroke(ROSE, 6)
        no_db = text("no vector database", 26, ROSE, weight=m.SEMIBOLD).next_to(db, m.RIGHT, buff=0.35)
        src = cite("Li, AI Agents in Depth · ch. 5").next_to(no_db, m.DOWN, buff=0.2).align_to(no_db, m.LEFT)
        mem_lines = ["- Prefers tabs, not spaces.", "- Test command: pytest -q", "- Prefers spaces."]
        mem = FileCard("MEMORY.md", mem_lines, center=(3.35, 1.0, 0), width=6.2, size=22, line_h=0.44, accent=SUN)
        wrong = mem.lines[2]
        cut = strike(wrong)
        term = terminal(["$ git log --oneline MEMORY.md", "3f2a91c remove a wrong line", "b81c07e add the test command"], width=6.2, size=20, title="git")
        term.move_to([3.35, -1.85, 0])
        ep12 = cite("how memories are written and picked · Episode 12").move_to([0, -3.45, 0])

        self.speak(
            "[OpenClaw keeps its memory as plain files too]: [a MEMORY.md], [plus Markdown log files, archived by date], [instead of a vector database.]",
            m.AnimationGroup(self.new_header("08 · plain files", "Markdown, not a vector database"), m.FadeIn(src), m.FadeIn(tree[0])),
            m.AnimationGroup(m.FadeIn(tree[1], shift=0.1 * m.RIGHT), mem.intro(source=tree[1])),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in tree[2:]], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(db), m.Create(cross), m.FadeIn(no_db, shift=0.1 * m.LEFT), m.FadeIn(ep12)),
        )
        self.speak(
            "[If it remembers something wrong], [you just delete that line], [and Git keeps every version.]",
            m.AnimationGroup(wrong.animate.set_color(ROSE), m.Circumscribe(wrong, color=ROSE, buff=0.06)),
            m.Succession(m.Create(cut), m.AnimationGroup(m.FadeOut(wrong, shift=0.2 * m.RIGHT), m.FadeOut(cut, shift=0.2 * m.RIGHT))),
            m.FadeIn(term, shift=0.15 * m.UP),
        )
        mem.lines = mem.lines[:2]
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reviewed diffs

    def reviewed_diffs(self):
        self.next_section("9 · Reviewed diffs", notes="Li, AI Agents in Depth, ch. 3 (how knowledge should be updated): a Proposer Agent 'proposes the smallest complete diff on a working branch'; a Reviewer Agent — preferably a model of similar capability from a different family — audits it against 'the raw evidence'; CI 'first checks formatting, links, metadata, and permission labels'; a PR merges only after explicit approval; chunks, summaries and vector indexes are rebuilt from the merged version. 'The index is therefore a reproducible derivative, while the reviewed knowledge in Git is the source of truth.' The tinyshop diff and its evidence are illustrative (the evidence matches this episode's run). Facts as of October 2026.")
        spec = [("proposer", "smallest diff", otto(0.42)), ("reviewer", "other family", otto(0.42, AZURE, AZURE_FILL)),
                ("checks", "format, links", icon(GEAR_ICON, 0.5)), ("merge", "after approval", icon("check", 0.5)),
                ("index", "rebuilt", icon("search", 0.5))]
        nodes = m.VGroup()
        for label, sub, pic in spec:
            box = card(2.3, 1.75, EDGE, CARD, radius=0.18, stroke=2)
            words = m.VGroup(pic, text(label, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1).move_to(box)
            nodes.add(m.VGroup(box, words))
        nodes.arrange(m.RIGHT, buff=0.42).move_to([0, 1.55, 0])
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.05) for a, b in zip(nodes, nodes[1:])])
        diff_lines = ["- Prices are floats, in dollars.", "+ Prices are int cents.", "+ Convert with round(), not int()."]
        diff = FileCard("knowledge/money.md", diff_lines, center=(-3.25, -1.35, 0), width=6.2, size=20, line_h=0.4, accent=CORAL, t2c=DIFF_T2C)
        branch = code("on a branch", 20, CORAL, weight=m.BOLD)
        diff.right_note(branch)
        cart_after = AFTER2["cart.py"].splitlines()[1].strip()
        ev_lines = [f"cart.py: {cart_after}", f"pytest -q: {FINAL_TESTS}"]
        assert cart_after == "return round(price * 100)"
        ev = FileCard("raw evidence", ev_lines, center=(3.45, -1.15, 0), width=5.9, size=20, line_h=0.4, accent=AZURE,
                      t2c={r"^[a-z. -]+:": MUTED})
        ok = pill("approved", MINT, MINT_FILL, 22).next_to(ev.box, m.DOWN, buff=0.2).align_to(ev.box, m.RIGHT)
        quote = text("“The reviewed knowledge in Git is the source of truth.”", 28, INK).move_to([-0.7, -3.3, 0])
        q_src = cite("Li · ch. 3").next_to(quote, m.RIGHT, buff=0.3).align_to(quote, m.DOWN)

        self.speak(
            "[When shared knowledge changes], [treat it like code.]",
            self.new_header("09 · reviewed diffs", "Change knowledge like code"),
            m.LaggedStart(*[m.FadeIn(n, shift=0.15 * m.UP) for n in nodes], *[m.GrowArrow(a) for a in links], lag_ratio=0.1),
        )
        self.speak(
            "[A proposer agent drafts the smallest complete diff], [on a branch.]",
            m.AnimationGroup(nodes[0][0].animate.set_stroke(CORAL), diff.intro(source=nodes[0])),
            m.FadeIn(branch, shift=0.1 * m.LEFT),
        )
        self.speak(
            "[A reviewer agent from a different model family] [checks it against the raw evidence.]",
            m.AnimationGroup(pulse(links[0], CORAL), nodes[1][0].animate.set_stroke(AZURE)),
            m.AnimationGroup(ev.intro(source=nodes[1]), m.Succession(m.Wait(0.8), m.FadeIn(ok, scale=1.1))),
        )
        self.speak(
            "[Automated checks run], [the change merges], [and the search index is rebuilt from it.]",
            m.AnimationGroup(pulse(links[1], AZURE), nodes[2][0].animate.set_stroke(SUN)),
            m.AnimationGroup(pulse(links[2], SUN), nodes[3][0].animate.set_stroke(MINT)),
            m.AnimationGroup(pulse(links[3], MINT), nodes[4][0].animate.set_stroke(ORCHID), m.Circumscribe(nodes[4], color=ORCHID, buff=0.06)),
        )
        self.speak(
            "[In the book's words]: [the reviewed knowledge in Git is the source of truth.]",
            m.FadeIn(q_src),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ all state on disk

    def all_state(self):
        self.next_section("10 · State on disk", notes="Learn Claude Code's final harness (s20, per our research notes; each folder comes from the session that introduces it) keeps its lasting state (the todo plan stays in memory) in .tasks/ (task graph, Episode 22), .memory/ (Episode 12), skills/ (Episode 10), .transcripts/ and .task_outputs/tool-results/ (compaction, Episode 11), .mailboxes/ (agent teams, Episode 24), .worktrees/ (Episode 25) and .scheduled_tasks.json (cron, Episode 23); NOTES.md and progress.md are this episode's. Li, AI Agents in Depth, ch. 5: 'the file system is the hub of information flow.' Facts as of October 2026.")
        rows_spec = [(".tasks/", "task graph", 22), (".memory/", "memories", 12), ("skills/", "skills", 10),
                     (".transcripts/", "transcripts", 11), (".task_outputs/", "big outputs", 11), (".mailboxes/", "teammates' mail", 24),
                     (".worktrees/", "worktrees", 25), (".scheduled_tasks.json", "schedules", 23), ("NOTES.md, progress.md", "handover", 14)]
        tree_rows, labels, pills_ = m.VGroup(), m.VGroup(), m.VGroup()
        y0, dy = 1.62, 0.48
        for i, (name, what, ep) in enumerate(rows_spec):
            color = AZURE if name.endswith("/") else (ORCHID if ep == 14 else INK)
            r = code(("▸ " if name.endswith("/") else "· ") + name, 22, color)
            r.align_to([-5.0, 0, 0], m.LEFT)
            sit(r, y0 - i * dy)
            lab = text(what, 24, INK if ep != 14 else ORCHID)
            lab.align_to([0.0, 0, 0], m.LEFT)
            sit(lab, y0 - i * dy)
            accent, fill = MODULES[EPISODES[ep][1]][1:]
            pl = pill(f"ep {ep}", accent, fill, 20, mono=True)
            pl.move_to([4.0, 0, 0])
            pl.set_y(r.get_y())
            tree_rows.add(r)
            labels.add(lab)
            pills_.add(pl)
        root = code("tinyshop/", 22, AZURE, weight=m.BOLD).align_to([-5.5, 0, 0], m.LEFT)
        sit(root, y0 + dy)
        quote = text("“The file system is the hub of information flow.”", 30, INK).move_to([0, -3.15, 0])
        q_src = cite("Li, AI Agents in Depth · ch. 5").next_to(quote, m.DOWN, buff=0.1)
        src = cite("learn-claude-code · s20").to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)
        groups = [[0], [1], [2], [3, 4], [5], [6], [7]]

        def show(idx):
            return m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(labels[i], shift=0.1 * m.RIGHT), m.FadeIn(pills_[i], shift=0.1 * m.LEFT)) for i in idx])

        self.speak(
            "[Learn Claude Code's final harness keeps its lasting state in folders.]",
            m.AnimationGroup(self.new_header("10 · state on disk", "The file system is the hub"), m.FadeIn(src), m.FadeIn(root),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in tree_rows], lag_ratio=0.12)),
        )
        self.speak(
            "[Tasks], [memories], [skills], [transcripts and big outputs], [mailboxes], [worktrees], [and schedules]: [each one has its own episode.]",
            *[show(g) for g in groups],
            m.AnimationGroup(show([8]), m.Circumscribe(m.VGroup(tree_rows[8], pills_[8]), color=ORCHID, buff=0.08)),
        )
        self.speak(
            "[As AI Agents in Depth puts it]: [the file system is the hub of information flow.]",
            m.FadeIn(q_src),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Write (NOTES.md for the why, progress.md for the where), forget (the context is wiped: compaction, a full window, a new session), resume (read the files first). Context is RAM, files are disk.")
        halo = glow(ORCHID, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Write", ORCHID), ("Forget", ROSE), ("Resume", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("NOTES.md · progress.md", "the context is wiped", "read the files first")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[1][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = on_baseline(m.VGroup(text("Context is RAM, files are disk:", 30, MUTED), text("write down what the next session needs.", 30, MUTED)).arrange(m.RIGHT, buff=0.22))
        tagline.next_to(row_, m.DOWN, buff=0.7)

        self.speak(
            "[To sum up]: [write], [forget], [resume.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.speak("[Context is RAM, files are disk]: [so write down what the next session needs.]",
                   m.FadeIn(tagline[0], shift=0.1 * m.UP), m.FadeIn(tagline[1], shift=0.1 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
