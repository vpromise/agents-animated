"""Agents, Animated · Episode 11 · Compaction (Module 2 · Context Engineering)

What to do when the context fills up. Running example: a long Otto session on tinyshop ("Add discount
codes to tinyshop, with tests."), 32 turns of reading, testing and editing, until the token meter
crosses the threshold. The session is not drawn by hand: the scene builds it as real Anthropic-style
messages, runs Learn Claude Code's four compaction layers on it (written below in the harness's style),
and every number on screen comes from that run (asserted below).

Sections: hook (the meter climbing over 32 turns and crossing a line) · the threshold (Claude Code's
window − max output − 13,000, as Learn Claude Code reads it; Li's "well before full", context
anxiety) · the summary (transcript to disk, one model call, five things kept, the history replaced,
Otto's last two turns kept) · does it pay off (Li's Experiment 2-10) · what a summary loses (early
decisions, reasons behind constraints, failed paths → files, Episode 14) · the 3-failure breaker ·
cheap first (the four layers in run order) · layer 1, the tool-result budget (800 KB log → disk,
marker + 2,000-character preview) · layer 2, snip (keep 3 + 47, never split a call from its result)
· layer 3, micro-compact (keep the last 3 results, older ones over 120 characters → one line) · the
catch (in-place edits break the prompt cache: compress in batches) · recap: watch, shrink, keep.

Sources (facts dated October 2026):
- Learn Claude Code, s08 "Context Compact" (learn.shareai.run/en/s08/): four layers, "cheap first,
  expensive last"; per turn tool_result_budget → snip_compact → micro_compact → (if
  estimate_token_count(messages) > THRESHOLD) compact_history. Teaching constants: budget 200,000
  bytes for the last message's tool_results, the largest persisted to .task_outputs/tool-results/
  with a <persisted-output> marker and a 2,000-character preview; snip over 50 messages keeps the
  first 3 and last 47 with "[snipped N messages from conversation middle]", never separating a
  tool_use from its tool_result; micro keeps the 3 most recent tool_results, older ones over 120
  characters become "[Earlier tool result compacted. Re-run if needed.]"; compact_history saves a
  JSONL transcript to .transcripts/, an LLM summary (goals, findings, modified files, remaining work,
  user constraints) replaces the history ("[Compacted]"); breaker after 3 consecutive failures. Budget
  must run before micro ("budget has to persist the full content first"). "Dive into CC source" (the
  site authors' reading of Claude Code): threshold = contextWindow − maxOutputTokens − 13,000
  (AUTOCOMPACT_BUFFER_TOKENS), MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3.
- Li Bojie, AI Agents in Depth, ch. 2 ("Why Compression Is Needed"; Experiment 2-10; "Design
  Principles"; "Compression and KV Cache"): compress "well before the window is close to full",
  "context anxiety"; no compression: "by the fifth iteration, the cumulative context exceeded the
  128K limit"; context-aware (query-aware) compression "requires only 7 iterations and 40,157 tokens",
  "reduces token usage by over 75%" (the book names no baseline for the 75%); adaptive trigger at 80%;
  compression most easily loses "early architectural decisions, the reasons behind constraints, and
  failed paths"; "truncating old tool results in place" breaks the cache; "perform batch compression
  when the context approaches the threshold, rather than compressing every round".
- Li, ch. 5 ("Failure and Error Recovery"): one session failed "over three thousand times in a row",
  futile retries "wasted about 250,000 API calls per day worldwide"; "Three is the empirical
  inflection point" (the book's study of Claude Code; no external citation).
- harness_reference.py: agent_loop, messages, client.messages.create(..., max_tokens=8000), tool_use /
  tool_result blocks with tool_use_id. Episode 6: the 3,000-token static prefix (1,800 + 1,200), the
  200,000-token window (for example).

Illustrative (said "for example"/"say" or labelled): the session (files, results and every token
count), the 200,000-token window, 4 characters per token, the 800 KB log, the summary's text and its
1,500 tokens, the three lost details, the marker's exact layout (simplified), the cache strip and the
timelines. Otto's compact_history also keeps its last two turns (s08's keeps only the summary; its
reactive path keeps the last five messages).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import copy
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep11"

# ---------------------------------------------------------------------------- the numbers

WINDOW = 200_000  # a context window, for example (as in Episode 6)
MAX_TOKENS = 8_000  # harness_reference.py: client.messages.create(..., max_tokens=8000)
BUFFER = 13_000  # Claude Code's autocompact buffer, as Learn Claude Code reads its source
THRESHOLD = WINDOW - MAX_TOKENS - BUFFER
EARLY = 0.8  # Li's adaptive trigger in Experiment 2-10
CHARS_PER_TOKEN = 4  # a rough estimate from character count, for example
SYSTEM_TOKENS, TOOLS_TOKENS = 1_800, 1_200  # Episode 6's static prefix
PREFIX_TOKENS = SYSTEM_TOKENS + TOOLS_TOKENS

# Learn Claude Code s08's teaching constants
MAX_BYTES = 200_000
PREVIEW_CHARS = 2_000
MAX_MESSAGES, KEEP_HEAD, KEEP_TAIL = 50, 3, 47
KEEP_RECENT_TOOL_RESULTS = 3
MICRO_MIN_CHARS = 120
PLACEHOLDER = "[Earlier tool result compacted. Re-run if needed.]"
MAX_FAILURES = 3
# Otto's own choices (illustrative)
KEEP_TURNS = 2
SUMMARY_TOKENS = 1_500

TASK = "Add discount codes to tinyshop, with tests."

# Otto's session (illustrative): (tokens Otto writes, tool, argument, what came back, its tokens)
TURNS = [
    (60, "bash", "pytest -q", "13 passed", 30),
    (50, "read_file", "checkout.py", "class Checkout: …", 2_400),
    (50, "read_file", "cart.py", "def total(items): …", 600),
    (70, "bash", "grep -rn price .", "41 matches", 1_900),
    (60, "read_file", "docs/discounts.md", "# Discount rules …", 3_200),
    (60, "read_file", "orders.json", '[{"id": 1, …', 30_000),
    (90, "read_file", "pricing.py", "TAX = 0.2 …", 1_700),
    (240, "edit_file", "pricing.py", "edited pricing.py", 10),
    (40, "bash", "pytest -q", "2 failed, 11 passed", 2_600),
    (60, "read_file", "tests/test_checkout.py", "def test_pay(): …", 2_800),
    (220, "edit_file", "checkout.py", "edited checkout.py", 10),
    (40, "bash", "pytest -q", "1 failed, 12 passed", 1_800),
    (50, "bash", "cat logs/app.log", "INFO start …", 38_000),
    (180, "edit_file", "pricing.py", "edited pricing.py", 10),
    (40, "bash", "pytest -q", "13 passed", 30),
    (60, "bash", "git log -p", "commit 4f2a …", 24_000),
    (60, "read_file", "docs/api.md", "# tinyshop API …", 12_000),
    (300, "edit_file", "tests/test_checkout.py", "edited tests/test_checkout.py", 10),
    (40, "bash", "pytest -q", "3 failed, 13 passed", 3_400),
    (60, "read_file", "coupons.py", "class Coupon: …", 4_200),
    (200, "edit_file", "coupons.py", "edited coupons.py", 10),
    (40, "bash", "pytest -q", "1 failed, 15 passed", 1_900),
    (60, "bash", "grep -rn expires .", "17 matches", 1_200),
    (60, "read_file", "codes.csv", "code,percent,expires …", 14_000),
    (180, "edit_file", "coupons.py", "edited coupons.py", 10),
    (40, "bash", "pytest -q", "16 passed", 30),
    (60, "read_file", "refunds.py", "def refund(order): …", 7_500),
    (60, "bash", "pytest -q tests/", "2 failed, 16 passed", 4_100),
    (60, "read_file", "tests/test_refunds.py", "def test_refund(): …", 3_600),
    (220, "edit_file", "refunds.py", "edited refunds.py", 10),
    (40, "bash", "pytest -q", "18 passed", 30),
    (60, "read_file", "payments.md", "# Payments guide …", 14_000),
]


def call_text(name, arg):
    return f'{name}("{arg}")'


def build_session():
    """The messages list after every turn, as the harness keeps it (Anthropic-style blocks). Content
    is padded to the turn's token count (at CHARS_PER_TOKEN) so estimates are exact."""
    messages = [{"role": "user", "content": TASK}]
    for k, (said, name, arg, shown, tokens) in enumerate(TURNS, 1):
        tid = f"toolu_{k:02d}"
        args = {"command" if name == "bash" else "path": arg}
        pad = said * CHARS_PER_TOKEN - len(name) - len(json.dumps(args))
        assert pad >= 0, k
        messages.append({"role": "assistant", "content": [
            {"type": "text", "text": "·" * pad},
            {"type": "tool_use", "id": tid, "name": name, "input": args},
        ]})
        body = shown + " " + "x" * (tokens * CHARS_PER_TOKEN - len(shown) - 1)
        messages.append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": tid, "content": body}]})
    return messages


def chars_of(msg):
    content = msg["content"]
    if isinstance(content, str):
        return len(content)
    n = 0
    for block in content:
        if block["type"] == "text":
            n += len(block["text"])
        elif block["type"] == "tool_use":
            n += len(block["name"]) + len(json.dumps(block["input"]))
        else:
            n += len(block["content"])
    return n


def tokens_of(msg):
    return math.ceil(chars_of(msg) / CHARS_PER_TOKEN)


def estimate_token_count(messages):
    """The whole context: the static prefix plus every message (an estimate from characters)."""
    return PREFIX_TOKENS + sum(tokens_of(x) for x in messages)


def is_tool_result(msg):
    c = msg["content"]
    return msg["role"] == "user" and isinstance(c, list) and c[0]["type"] == "tool_result"


def results_of(messages):
    return [b for x in messages if isinstance(x["content"], list) for b in x["content"] if b["type"] == "tool_result"]


# ---- the four layers, in the harness's style (Learn Claude Code s08's rules)

DISK = {}  # what the harness wrote to disk


def tool_result_budget(messages):
    """Layer 1: if the latest results exceed the budget, persist the largest to disk, keep a marker."""
    blocks = results_of(messages[-1:])
    while sum(len(b["content"]) for b in blocks) > MAX_BYTES:
        big = max(blocks, key=lambda b: len(b["content"]))
        path = f".task_outputs/tool-results/{big['tool_use_id']}.txt"
        DISK[path] = big["content"]
        big["content"] = f"<persisted-output>\nsaved: {path}\n{big['content'][:PREVIEW_CHARS]}\n</persisted-output>"
    return messages


def snip_compact(messages):
    """Layer 2: over 50 messages keep the first 3 and the last 47; never orphan a tool_result."""
    if len(messages) <= MAX_MESSAGES:
        return messages
    start = len(messages) - KEEP_TAIL
    while is_tool_result(messages[start]):  # its tool_use would be cut: move the cut by one
        start += 1
    marker = {"role": "user", "content": f"[snipped {start - KEEP_HEAD} messages from conversation middle]"}
    return messages[:KEEP_HEAD] + [marker] + messages[start:]


def micro_compact(messages):
    """Layer 3: keep the last 3 tool results; older ones over 120 characters become one line."""
    for block in results_of(messages)[:-KEEP_RECENT_TOOL_RESULTS]:
        if len(block["content"]) > MICRO_MIN_CHARS:
            block["content"] = PLACEHOLDER
    return messages


SUMMARY = {  # what Otto's summary keeps (s08: goals, findings, modified files, remaining work, user constraints)
    "goal": "discount codes, with tests",
    "found": "round once, at the end",
    "files": "coupons.py, refunds.py, …",
    "left": "expired codes, docs",
    "user": "keep cart.total() as it is",
}
LOST = [  # what the history said and the summary does not (Li: the three things lost first)
    ("turn 8", "decided: codes live in coupons.py"),
    ("turn 13", "why: the mobile app calls total()"),
    ("turn 9", "failed: rounding per item"),
]


def summarize_history(messages):
    """The one model call (scripted here): the five things, padded to the summary's size."""
    text_ = "\n".join(f"{k}: {v}" for k, v in SUMMARY.items())
    return text_ + "\n" + "·" * (SUMMARY_TOKENS * CHARS_PER_TOKEN - len("[Compacted]\n\n") - len(text_) - 1)


def compact_history(messages):
    """Layer 4: save the transcript, summarize, replace the history (Otto keeps its last two turns)."""
    DISK[".transcripts/turn-32.jsonl"] = "\n".join(json.dumps(x) for x in messages)
    summary = summarize_history(messages)
    return [{"role": "user", "content": f"[Compacted]\n\n{summary}"}] + messages[-2 * KEEP_TURNS:]


# ---- the run

SESSION = build_session()
RUN = [estimate_token_count(SESSION[: 1 + 2 * k]) for k in range(len(TURNS) + 1)]
CROSS = next(k for k, t in enumerate(RUN) if t > THRESHOLD)
TOTAL = RUN[-1]
N_MSG = len(SESSION)

BUDGETED = tool_result_budget(copy.deepcopy(SESSION))  # nothing over 200 KB at turn 32
SNIPPED = snip_compact(copy.deepcopy(BUDGETED))
SNIP_N = len(SESSION) - (len(SNIPPED) - 1)
SNIP_START = SNIP_N + KEEP_HEAD  # first kept message of the tail
SNIP_TOKENS = estimate_token_count(SNIPPED)
MICRO = micro_compact(copy.deepcopy(SNIPPED))
MICRO_TOKENS = estimate_token_count(MICRO)
SHRUNK = [i for i, (a, b) in enumerate(zip(SNIPPED, MICRO)) if chars_of(a) != chars_of(b)]
FOLDED = compact_history(copy.deepcopy(SESSION))
FOLDED_TOKENS = estimate_token_count(FOLDED)

# the replay: the same 32 turns with the three cheap layers at the top of every turn (notes only)
_live, PEAK = [SESSION[0]], 0
for _k in range(1, len(TURNS) + 1):
    _live = micro_compact(snip_compact(tool_result_budget(_live + copy.deepcopy(SESSION[2 * _k - 1: 2 * _k + 1]))))
    PEAK = max(PEAK, estimate_token_count(_live))

# layer 1's example: say Otto prints a long test log
LOG_HEAD = "FAILED tests/test_coupons.py::test_expired"
LOG_CHARS = 800_000
_log = LOG_HEAD + "\n" + "." * (LOG_CHARS - len(LOG_HEAD) - 1)
LOG_MSG = {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "toolu_33", "content": _log}]}
LOG_TOKENS = tokens_of(LOG_MSG)
MARKED = tool_result_budget([copy.deepcopy(LOG_MSG)])
MARKER_TOKENS = tokens_of(MARKED[0])
LOG_PATH = ".task_outputs/tool-results/toolu_33.txt"

# Li, AI Agents in Depth, ch. 2, Experiment 2-10 (cited)
EXP_CAP, EXP_FAIL_AT, EXP_ITERS, EXP_TOKENS = 128_000, 5, 7, 40_157
# the cache strip and the two schedules (illustrative)
STRIP = ["system", "tools", "user"] + ["tool_use", "tool_result"] * 5 + ["tool_use"]
EDITED = 4  # the first old tool result, edited in place
TICKS = 16
EVERY_TURN = list(range(4, TICKS))  # a little compaction on every turn once results pile up
BATCHED = [13]  # one batch, as the context nears the line

assert THRESHOLD == 179_000  # "a hundred and seventy-nine thousand"
assert MAX_TOKENS == 8_000 and BUFFER == 13_000 and WINDOW == 200_000
assert N_MSG == 65 and len(TURNS) == 32 and CROSS == 32  # "turn thirty-two", "sixty-five messages"
assert RUN[31] <= THRESHOLD < RUN[32] == TOTAL
assert round(TOTAL, -3) == 181_000 and TOTAL / WINDOW > 0.9  # "a hundred and eighty-one thousand"; meter turns ROSE
assert TURNS[-1][4] == 14_000  # "fourteen thousand tokens"
assert BUDGETED == SESSION  # layer 1 finds nothing over the budget at turn 32
assert len(SNIPPED) == MAX_MESSAGES and SNIP_N == 16  # "the sixteen messages in the middle"
assert SNIP_START == N_MSG - KEEP_TAIL + 1  # "the cut moves by one message"
assert is_tool_result(SESSION[N_MSG - KEEP_TAIL]) and not is_tool_result(SESSION[SNIP_START])
assert round(SNIP_TOKENS, -3) == 138_000  # "a hundred and thirty-eight thousand tokens"
assert MICRO_TOKENS < 20_000  # "under twenty thousand"
assert all(is_tool_result(SNIPPED[i]) for i in SHRUNK) and len(SHRUNK) == 13
assert tokens_of(FOLDED[0]) == SUMMARY_TOKENS and len(FOLDED) == 1 + 2 * KEEP_TURNS
assert FOLDED_TOKENS < 19_000  # "under nineteen thousand"
assert abs(MICRO_TOKENS / FOLDED_TOKENS - 1) < 0.1  # "about what the summary left"
assert LOG_CHARS == 4 * MAX_BYTES and LOG_TOKENS == WINDOW  # "eight hundred kilobytes": "the whole window"
assert DISK[LOG_PATH] == _log and PREVIEW_CHARS == 2_000 and MARKER_TOKENS < 600
assert MARKED[0]["content"][0]["content"].count(".") > PREVIEW_CHARS - len(LOG_HEAD) - 2  # the preview is in the marker
assert round(EXP_TOKENS, -4) == 40_000  # "about forty thousand tokens"
assert len(EVERY_TURN) == 12 and len(BATCHED) == 1
assert PEAK / WINDOW < 0.25  # replayed with the cheap layers every turn, the meter stays under a quarter

# ---------------------------------------------------------------------------- look

LLM = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
ACCENT = {"system": ORCHID, "tools": ORCHID, "user": AZURE, "tool_use": SUN, "tool_result": MINT,
          "summary": AZURE, "snip": MUTED}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPRSTUVWXYZ0123456789")
PANEL_FILL = "#18150F"
LEFT_X = -6.75

PANEL_C, PANEL_W, PANEL_H = (3.55, -0.95, 0), 6.0, 4.6
PANEL_TOP = PANEL_C[1] + PANEL_H / 2
PANEL_LEFT = PANEL_C[0] - PANEL_W / 2
GAUGE_Y = 2.12
ROW_H, ROW_GAP = 0.46, 0.08
ROW_W = PANEL_W - 0.36

MAP_ROWS = 2 + N_MSG
MAP_TOP = PANEL_TOP - 0.15
MAP_PITCH = (PANEL_H - 0.3) / MAP_ROWS
BAR_H = MAP_PITCH * 0.7
BAR_X = PANEL_LEFT + 0.22
BAR_MAX = PANEL_W - 0.44
BIGGEST = max(tokens_of(x) for x in SESSION)
MIN_W = 0.06


def cite(s):
    return text(s, 20, MUTED)


def sit(words, base):
    """Put a Text's first flat-bottomed glyph on a baseline."""
    k = 0
    for ch in words.text:
        if ch.isspace():
            continue
        if ch in FLAT:
            break
        k += 1
    k = min(k, len(words) - 1)
    return words.shift((base - words[k].get_bottom()[1]) * m.UP)


def row(kind, body, tokens, width=ROW_W):
    """One message on one line: a stripe in its role's colour, the content, its tokens."""
    accent = ACCENT[kind]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    if kind in ("system", "tools", "user"):
        words = text(body, 20, INK)
    elif kind == "summary":
        words = code(body, 20, INK)
    else:
        words = code(body, 20, accent)
    count = code(f"{tokens:,}", 20, MUTED)
    words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    count.move_to(box).align_to(box, m.RIGHT).shift(0.2 * m.LEFT)
    base = box.get_y() - 0.09
    sit(words, base)
    sit(count, base)
    assert words.get_right()[0] < count.get_left()[0] - 0.15, body
    return m.VGroup(box, stripe, words, count)


def turn_rows(k):
    """The two rows of turn k (1-based): Otto's call and the result."""
    said, name, arg, shown, tokens = TURNS[k - 1]
    return row("tool_use", call_text(name, arg), said), row("tool_result", shown, tokens)


def kind_of(msg):
    c = msg["content"]
    if isinstance(c, str):
        return "snip" if c.startswith("[snipped") else ("summary" if c.startswith("[Compacted]") else "user")
    return "tool_use" if msg["role"] == "assistant" else "tool_result"


def entries(messages):
    return [("system", SYSTEM_TOKENS), ("tools", TOOLS_TOKENS)] + [(kind_of(x), tokens_of(x)) for x in messages]


def map_y(i):
    return MAP_TOP - MAP_PITCH * (i + 0.5)


def bar(kind, tokens, i):
    """One message of the zoomed-out map: length ∝ tokens."""
    w = max(MIN_W, BAR_MAX * tokens / BIGGEST)
    b = m.RoundedRectangle(corner_radius=min(0.02, BAR_H / 2), width=w, height=BAR_H, stroke_width=0,
                           fill_color=ACCENT[kind], fill_opacity=1)
    return b.move_to([BAR_X + w / 2, map_y(i), 0])


def context_map(items):
    return [bar(kind, tok, i) for i, (kind, tok) in enumerate(items)]


def map_panel(title="context window"):
    panel = card(PANEL_W, PANEL_H, EDGE, PANEL_FILL, radius=0.2, stroke=2).move_to(PANEL_C).set_z_index(-1)
    cap = code(title, 20, MUTED).next_to(panel, m.UP, buff=0.12).align_to(panel, m.LEFT)
    return panel, cap


def model_card(at, width=2.1, height=1.15):
    box = card(width, height, CORAL, CORAL_FILL)
    body = m.VGroup(icon(LLM, 0.62), text("model", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.22).move_to(box)
    return m.VGroup(box, body).move_to(at)


def send(group, target):
    """A miniature copy of `group` flying into `target` and vanishing there (one cue)."""
    p = group.copy().scale(0.3).move_to(target.get_center()).set_z_index(5)
    out = m.FadeOut(p, scale=0.3, run_time=0.35)
    out.introducer = True  # p is introduced by the FadeIn just before, in the same line
    return m.Succession(m.FadeIn(p, target_position=group.get_center(), run_time=0.8), out)


def bracket(top, bottom, x, color, label, size=22):
    """A thin bracket left of the panel, its label on the left."""
    spine = m.Line([x, top, 0], [x, bottom, 0]).set_stroke(color, 3)
    ticks = m.VGroup(*[m.Line([x, y, 0], [x + 0.1, y, 0]).set_stroke(color, 3) for y in (top, bottom)])
    words = text(label, size, color, weight=m.SEMIBOLD).next_to(spine, m.LEFT, buff=0.14)
    return m.VGroup(spine, ticks, words)


def cross(at, size=0.26, color=ROSE):
    a, b = np.array([size / 2, size / 2, 0]), np.array([size / 2, -size / 2, 0])
    at = np.array(at, dtype=float)
    return m.VGroup(m.Line(at - a, at + a), m.Line(at - b, at + b)).set_stroke(color, 5)


def files_card(caption, entries_, highlight=(), width=3.9):
    tree = file_tree(entries_, size=20, highlight=highlight)
    cap = code(caption, 20, MUTED)
    inner = m.VGroup(cap, tree).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    box = card(max(width, inner.width + 0.6), inner.height + 0.5, EDGE, CARD, radius=0.16, stroke=2)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, inner)


def summary_card(width=6.6):
    """Otto's summary: the five things, one line each."""
    cap = m.VGroup(code("[Compacted]", 20, AZURE, weight=m.BOLD), text("written by the model", 20, MUTED)).arrange(m.RIGHT, buff=0.3)
    keys = [code(k, 22, MUTED) for k in SUMMARY]
    vals = [code(v, 20, INK) if k == "files" else text(v, 22, INK) for k, v in SUMMARY.items()]
    key_w = max(k.width for k in keys)
    lines = m.VGroup()
    for k, v in zip(keys, vals):
        v.next_to(k, m.RIGHT, buff=key_w - k.width + 0.3)
        sit(v, k[0].get_bottom()[1])
        lines.add(m.VGroup(k, v))
    lines.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
    inner = m.VGroup(cap, lines).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
    box = card(width, inner.height + 0.6, AZURE, CARD, radius=0.16, stroke=2)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    assert inner.get_right()[0] < box.get_right()[0] - 0.2
    return m.VGroup(box, cap, lines)


class Slot:
    """One place on screen whose content is swapped (a counter, a status line)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15):
        on_screen = {id(x) for x in self.scene.get_mobject_family_members()}
        anims = []
        if self.current is not None:
            out = m.FadeOut(self.current, shift=shift * m.UP)
            if id(self.current) not in on_screen:  # introduced earlier in this same line
                out.introducer = True
            anims.append(out)
        if new is not None:
            anims.append(m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


class Gauge:
    """The token meter above the context panel, with the threshold marked and a counter."""

    def __init__(self, scene, tokens=0):
        self.meter = Meter(PANEL_W, tokens / WINDOW, SUN, 0.24).move_to([PANEL_C[0], GAUGE_Y, 0])
        x = self.meter.track.get_left()[0] + PANEL_W * THRESHOLD / WINDOW
        self.tick = m.Line([x, GAUGE_Y - 0.21, 0], [x, GAUGE_Y + 0.21, 0]).set_stroke(INK, 4).set_z_index(3)
        self.label = code("threshold", 20, MUTED).next_to(self.tick, m.UP, buff=0.06)
        self.frame = m.VGroup(self.meter, self.tick, self.label)
        self.counter = Slot(scene)
        self.base = PANEL_TOP + 0.12

    def count(self, tokens):
        words = code(f"{tokens:,} tokens", 22, ROSE if tokens > THRESHOLD else INK)
        words.align_to([PANEL_C[0] + PANEL_W / 2, 0, 0], m.RIGHT)
        return sit(words, self.base)

    def to(self, tokens, **kw):
        return m.AnimationGroup(self.meter.fill_to(tokens / WINDOW), self.counter.to(self.count(tokens)), **kw)


def status(turn):
    t = title_text(f"turn {turn}", 44, INK)
    n = 1 + 2 * turn
    sub = text(f"{n} message{'' if n == 1 else 's'}", 24, MUTED)
    return m.VGroup(t, sub).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT).move_to([-3.0, -0.35, 0], aligned_edge=m.LEFT)


def reset_column(col, rows):
    """Lay rows out from the top of a ContextColumn and make them its stack (a fast-forward)."""
    y = col.top
    col.slots = []
    for r in rows:
        r.move_to([col.panel.get_x(), 0, 0]).align_to([0, y, 0], m.UP)
        col.slots.append([r, y])
        y -= r.height + col.gap
    col.items = list(rows)


def line_bar(block, keys, color, opacity=0.16):
    """A highlight across the code block on the line(s) holding `keys`."""
    ys = [glyphs(block[1], k).get_center()[1] for k in keys]
    left, right = block[0].get_left()[0] + 0.12, block[0].get_right()[0] - 0.12
    h = max(ys) - min(ys) + 0.36
    cy = (max(ys) + min(ys)) / 2
    rect = m.Rectangle(width=right - left, height=h, stroke_width=0, fill_color=color, fill_opacity=opacity).move_to([(left + right) / 2, cy, 0])
    mark = m.Rectangle(width=0.05, height=h, stroke_width=0, fill_color=color, fill_opacity=1).move_to([left + 0.025, cy, 0])
    return m.VGroup(rect, mark).set_z_index(1)


# ---------------------------------------------------------------------------- the episode


class Ep11Compaction(Episode):
    number = 11

    def lesson(self):
        self.hook()
        self.threshold()
        self.summarize()
        self.pays_off()
        self.lost()
        self.breaker()
        self.cheap_first()
        self.budget()
        self.snip()
        self.micro()
        self.catch()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"A long session (illustrative): Otto adds discount codes to tinyshop. Each turn appends Otto's call and the tool result to the context (the append-only loop of Episode 2). The static prefix is Episode 6's 3,000 tokens; the window is {WINDOW:,} tokens (for example); tokens are estimated at {CHARS_PER_TOKEN} characters each. Turn 31: {RUN[31]:,} tokens; turn 32 reads payments.md ({TURNS[-1][4]:,} tokens) and the context reaches {TOTAL:,} tokens in {N_MSG} messages, past the {THRESHOLD:,}-token threshold (computed from the session).")
        col = ContextColumn(center=PANEL_C, width=PANEL_W, height=PANEL_H, title="context window", gap=ROW_GAP)
        col.panel.set_z_index(-1)
        gauge = Gauge(self)
        bot = otto(1.25).move_to([-5.3, -0.3, 0])
        ask = message("user", TASK, width=6.3, size=22).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.75, 0], m.UP)
        state = Slot(self)
        sys_row = row("system", "You are Otto, a coding agent …", SYSTEM_TOKENS)
        tools_row = row("tools", "bash, read_file, edit_file", TOOLS_TOKENS)
        task_row = row("user", "Add discount codes …", tokens_of(SESSION[0]))
        r1, r2, r3 = turn_rows(1), turn_rows(2), turn_rows(3)
        late = [*turn_rows(30), *turn_rows(31)]
        r32 = turn_rows(32)

        self.say(
            "[In episode six, we took Otto's context apart.] [Today it has a long job]: [add discount codes to tinyshop, with tests.]",
            m.AnimationGroup(m.FadeIn(col.frame), m.FadeIn(gauge.frame), m.FadeIn(bot, shift=0.2 * m.UP),
                             col.push(sys_row), col.push(tools_row), gauge.to(PREFIX_TOKENS)),
            m.FadeIn(ask, shift=0.2 * m.DOWN),
            m.AnimationGroup(col.push(task_row), gauge.to(RUN[0]), state.to(status(0))),
        )
        self.say(
            "[It runs the tests], [reads the code], [and every call and every result stays in the context.]",
            m.AnimationGroup(m.Succession(col.push(r1[0], 0.6), col.push(r1[1], 0.6)), gauge.to(RUN[1]), state.to(status(1))),
            m.AnimationGroup(m.Succession(*[col.push(r, 0.5) for r in (*r2, *r3)]), gauge.to(RUN[3]), state.to(status(3))),
            m.Circumscribe(col.panel, color=MINT, buff=0.04),
        )
        old = list(col.items)
        reset_column(col, late)
        self.say(
            "[The session runs on], [turn after turn], [and the meter keeps climbing.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(*old), shift=0.4 * m.UP),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.3 * m.UP) for r in late], lag_ratio=0.15)),
            m.Succession(*[m.AnimationGroup(state.to(status(t)), gauge.to(RUN[t]), run_time=0.7) for t in (12, 20, 26, 31)]),
            m.Circumscribe(gauge.meter, color=SUN, buff=0.08),
        )
        self.say(
            "[At turn thirty-two], [Otto reads the payments guide], [fourteen thousand tokens], [and the meter crosses a line.]",
            state.to(status(32)),
            col.push(r32[0]),
            m.AnimationGroup(col.push(r32[1]), m.Circumscribe(r32[1][3], color=MINT, buff=0.06)),
            m.AnimationGroup(gauge.to(TOTAL, run_time=1.2), m.Circumscribe(gauge.label, color=ROSE, buff=0.06)),
        )
        self.say(
            "[A hundred and eighty-one thousand tokens.] [Today: what a harness does when the context fills up.]",
            m.Circumscribe(gauge.counter.current, color=ROSE, buff=0.08),
            m.Wiggle(bot),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the threshold

    def threshold(self):
        self.next_section("1 · The threshold", notes=f"Learn Claude Code s08, 'Dive into CC source' (the site authors' reading of Claude Code): auto-compact threshold = contextWindow − maxOutputTokens − 13,000 (AUTOCOMPACT_BUFFER_TOKENS). With a {WINDOW:,}-token window (for example) and harness_reference's max_tokens={MAX_TOKENS}: {THRESHOLD:,} (computed). Otto's context: {TOTAL:,}. Li, AI Agents in Depth, ch. 2: 'Compressing the context well before the window is close to full may improve the quality of the model's decisions'; 'context anxiety': a model starts wrapping up before the task is done because it believes its window is nearly full. Li's Experiment 2-10 triggers its adaptive compression at 80% (shown as 'earlier · 80%', an example). Facts as of October 2026.")
        x0, x1 = -6.0, 6.0
        w = x1 - x0
        full_y, zoom_y = 1.5, -0.25
        lo = 150_000
        fx = lambda t: x0 + w * t / WINDOW  # noqa: E731
        zx = lambda t: x0 + w * (t - lo) / (WINDOW - lo)  # noqa: E731

        def span(a, b, y, h, color, opacity=1.0, radius=0.0):
            r = m.RoundedRectangle(corner_radius=radius, width=b - a, height=h, stroke_width=0, fill_color=color, fill_opacity=opacity)
            return r.move_to([(a + b) / 2, y, 0])

        full_track = card(w, 0.36, EDGE, CARD, radius=0.18, stroke=2).move_to([0, full_y, 0])
        full_fill = span(x0, fx(TOTAL), full_y, 0.36, ROSE, radius=0.18)
        full_cap = code(f"context window · {WINDOW:,} tokens", 20, MUTED).next_to(full_track, m.UP, buff=0.14).align_to(full_track, m.LEFT)
        full_tick = m.Line([fx(THRESHOLD), full_y - 0.26, 0], [fx(THRESHOLD), full_y + 0.26, 0]).set_stroke(INK, 3).set_z_index(3)
        zoom_track = card(w, 0.6, EDGE, CARD, radius=0.1, stroke=2).move_to([0, zoom_y, 0])
        zoom_fill = span(x0, zx(TOTAL), zoom_y, 0.6, ROSE, radius=0.1)
        links = m.VGroup(
            m.DashedLine([fx(lo), full_y - 0.2, 0], [x0, zoom_y + 0.32, 0], dash_length=0.08),
            m.DashedLine([x1, full_y - 0.2, 0], [x1, zoom_y + 0.32, 0], dash_length=0.08),
        ).set_stroke(FAINT, 2)
        lab_y = zoom_y - 0.66
        lo_lab = code(f"{lo:,}", 20, MUTED).move_to([x0, lab_y, 0], aligned_edge=m.LEFT)
        hi_lab = code(f"{WINDOW:,}", 20, MUTED).move_to([x1, lab_y, 0], aligned_edge=m.RIGHT)
        zone_out = span(zx(WINDOW - MAX_TOKENS), x1, zoom_y, 0.6, CORAL, 0.35)
        zone_buf = span(zx(THRESHOLD), zx(WINDOW - MAX_TOKENS), zoom_y, 0.6, MUTED, 0.22)
        out_lab = code(f"reply {MAX_TOKENS:,}", 20, INK).move_to(zone_out)
        buf_lab = code(f"buffer {BUFFER:,}", 20, INK).move_to(zone_buf).shift(0.2 * m.RIGHT)
        zone_out.set_z_index(1)
        zone_buf.set_z_index(1)
        out_lab.set_z_index(2)
        buf_lab.set_z_index(2)
        tick = m.Line([zx(THRESHOLD), zoom_y - 0.42, 0], [zx(THRESHOLD), zoom_y + 0.42, 0]).set_stroke(INK, 4).set_z_index(3)
        tick_lab = code(f"threshold · {THRESHOLD:,}", 22, INK, weight=m.BOLD).next_to(tick, m.UP, buff=0.1)
        otto_lab = code(f"Otto · {TOTAL:,}", 20, ROSE).move_to([zx(TOTAL) + 0.25, lab_y, 0])
        early = m.DashedLine([zx(EARLY * WINDOW), zoom_y - 0.42, 0], [zx(EARLY * WINDOW), zoom_y + 0.42, 0], dash_length=0.07).set_stroke(SUN, 3).set_z_index(3)
        early_lab = code(f"earlier · {EARLY:.0%}", 20, SUN).next_to(early, m.RIGHT, buff=0.1).set_y(lab_y)
        assert lo_lab.get_right()[0] < early.get_left()[0] and early_lab.get_right()[0] < otto_lab.get_left()[0] - 0.2
        assert buf_lab.get_left()[0] > zoom_fill.get_right()[0] + 0.1 and out_lab.width < zone_out.width - 0.1

        formula_src = "threshold = window − max output − 13,000"
        formula = title_text(formula_src, 34, INK)
        paint(formula, formula_src, {"window": SUN, "max output": CORAL, "13,000": MUTED, "[=−]": MUTED})
        formula.move_to([0, -1.75, 0])
        nums_src = f"{WINDOW:,} − {MAX_TOKENS:,} − {BUFFER:,} = {THRESHOLD:,}"
        nums = code(nums_src, 26, INK, t2c={f"{WINDOW:,}": SUN, f"{MAX_TOKENS:,}": CORAL, f"= {THRESHOLD:,}": INK, "−": MUTED})
        nums.move_to([0, -2.45, 0])
        src_a = cite("learn-claude-code · s08, reading Claude Code's source").move_to([0, -3.2, 0])
        f_head = glyphs(formula, "threshold = window")
        f_out = glyphs(formula, "− max output")
        f_buf = glyphs(formula, "− 13,000")
        n_win = glyphs(nums, f"{WINDOW:,}")
        n_out = glyphs(nums, f"− {MAX_TOKENS:,}")
        n_rest = glyphs(nums, f"− {BUFFER:,} = {THRESHOLD:,}")
        anxiety = title_text("context anxiety", 40, SUN).move_to([0, -1.75, 0])
        anx_sub = text("a model wraps up early when it thinks its window is nearly full", 24, MUTED).move_to([0, -2.45, 0])
        src_b = cite("Li, AI Agents in Depth · ch. 2").move_to([0, -3.2, 0])

        self.say(
            "[When should the harness step in?] [Not at the very top of the window], [but at a line below it.]",
            m.AnimationGroup(self.new_header("01 · threshold", "Where to draw the line"),
                             m.FadeIn(full_track), m.FadeIn(full_fill), m.FadeIn(full_cap)),
            m.AnimationGroup(m.Create(links), m.FadeIn(zoom_track), m.FadeIn(zoom_fill), m.FadeIn(lo_lab), m.FadeIn(hi_lab)),
            m.AnimationGroup(m.GrowFromCenter(tick), m.GrowFromCenter(full_tick), m.FadeIn(tick_lab, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[In Claude Code, as Learn Claude Code reads its source], [the line is the window], [minus the reply's maximum length], [minus a thirteen thousand token buffer.]",
            m.FadeIn(src_a),
            m.FadeIn(f_head, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(f_out, shift=0.15 * m.UP), m.FadeIn(zone_out), m.FadeIn(out_lab)),
            m.AnimationGroup(m.FadeIn(f_buf, shift=0.15 * m.UP), m.FadeIn(zone_buf), m.FadeIn(buf_lab)),
        )
        self.say(
            "[For a two hundred thousand token window], [and Otto's eight thousand token replies], [the line sits at a hundred and seventy-nine thousand.]",
            m.FadeIn(n_win, shift=0.1 * m.UP),
            m.FadeIn(n_out, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(n_rest, shift=0.1 * m.UP), m.Circumscribe(tick_lab, color=SUN, buff=0.08)),
        )
        self.say(
            "[Otto's context is past it], [so the harness has to make room.]",
            m.AnimationGroup(m.FadeIn(otto_lab, shift=0.1 * m.UP), m.Circumscribe(zoom_fill, color=ROSE, buff=0.05)),
            m.Circumscribe(zoom_track, color=ROSE, buff=0.08),
        )
        self.say(
            "[The book AI Agents in Depth argues for acting earlier]: [near a full window, models get context anxiety], [and wrap up too soon.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (f_head, f_out, f_buf, n_win, n_out, n_rest, src_a)],
                             m.Create(early), m.FadeIn(early_lab), m.FadeIn(src_b)),
            m.FadeIn(anxiety, shift=0.15 * m.UP),
            m.FadeIn(anx_sub),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the summary

    def summarize(self):
        self.next_section("2 · Summarize", notes=f"Learn Claude Code s08, layer 4 (compact_history): save the full transcript as JSONL in .transcripts/, then one model call (summarize_history) writes a summary keeping goals, findings, modified files, remaining work and user constraints; the history is replaced by one user message '[Compacted]' + summary. Otto's version also keeps its last {KEEP_TURNS} turns verbatim (s08's keeps only the summary; its reactive path keeps the last five messages; Claude Code re-attaches up to five recent files). System prompt and tools are separate parameters of client.messages.create, so they never change. Computed: {TOTAL:,} → {FOLDED_TOKENS:,} tokens (prefix {PREFIX_TOKENS:,} + summary {SUMMARY_TOKENS:,} (illustrative) + the last two turns). The summary's text is illustrative.")
        panel, cap = map_panel()
        gauge = Gauge(self, TOTAL)
        bars = context_map(entries(SESSION))
        disk = files_card("on disk", [".transcripts/", ".transcripts/turn-32.jsonl"], highlight=("turn-32.jsonl",), width=3.9)
        disk.move_to([0, 0, 0]).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.3, 0], m.UP)
        model = model_card([-1.2, disk.get_y(), 0])
        summ = summary_card(6.6).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 0.75, 0], m.UP)
        summ.set_z_index(2)
        lines = summ[2]

        # after the fold: the prefix, the summary, the last two turns, as rows
        after = [row("system", "You are Otto, a coding agent …", SYSTEM_TOKENS), row("tools", "bash, read_file, edit_file", TOOLS_TOKENS),
                 row("summary", "[Compacted] summary", SUMMARY_TOKENS), *turn_rows(31), *turn_rows(32)]
        y = PANEL_TOP - 0.18
        for r in after:
            r.move_to([PANEL_C[0], y - ROW_H / 2, 0])
            y -= ROW_H + ROW_GAP
        keep_n = 2 * KEEP_TURNS
        old_bars = m.VGroup(*bars[2:-keep_n])
        last_box = m.VGroup(*after[3:])

        self.say(
            "[The best-known answer is a summary.] [First, the harness saves the full transcript to disk.]",
            m.AnimationGroup(self.new_header("02 · summarize", "Fold the history into a summary"), m.FadeIn(panel), m.FadeIn(cap),
                             m.FadeIn(gauge.frame), gauge.counter.to(gauge.count(TOTAL)),
                             m.LaggedStart(*[m.FadeIn(b) for b in bars], lag_ratio=0.02, run_time=1.4)),
            m.Succession(m.FadeIn(disk, shift=0.15 * m.RIGHT), send(m.VGroup(*bars), disk)),
        )
        self.say(
            "[Then one model call reads the history], [and writes down what matters]: [the goals], [the findings], [the files changed], [the work left], [and the user's constraints.]",
            m.Succession(m.FadeIn(model, shift=0.15 * m.LEFT), send(m.VGroup(*bars), model)),
            m.FadeIn(m.VGroup(summ[0], summ[1]), shift=0.15 * m.UP),
            *[m.FadeIn(ln, shift=0.12 * m.RIGHT) for ln in lines],
        )
        fold = [m.FadeOut(b, target_position=after[2].get_center(), scale=0.2) for b in old_bars]
        fold += [m.FadeIn(after[2], target_position=old_bars.get_center())]
        fold += [m.FadeOut(bars[i]) for i in (0, 1)] + [m.FadeOut(bars[-keep_n + j]) for j in range(keep_n)]
        fold += [m.FadeIn(after[i], shift=0.1 * m.UP) for i in (0, 1, *range(3, 3 + keep_n))]
        self.say(
            "[That summary replaces the old history.] [Otto also keeps its last two turns, word for word.]",
            m.AnimationGroup(*fold, run_time=1.4),
            m.Circumscribe(last_box, color=MINT, buff=0.06),
        )
        self.say(
            "[The system prompt and tools never change], [and a hundred and eighty-one thousand tokens become under nineteen thousand.]",
            m.Circumscribe(m.VGroup(after[0], after[1]), color=ORCHID, buff=0.06),
            gauge.to(FOLDED_TOKENS, run_time=1.2),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ does it pay off

    def pays_off(self):
        self.next_section("3 · Does it pay off?", notes=f"Li, AI Agents in Depth, ch. 2, Experiment 2-10 (a research agent tracking OpenAI's co-founders, context capped at 128K): with no compression, 'by the fifth iteration, the cumulative context exceeded the 128K limit' and the task failed; context-aware (query-aware) compression 'requires only {EXP_ITERS} iterations and {EXP_TOKENS:,} tokens' and the book says it 'reduces token usage by over 75%' (no baseline named). Numbers as the book states them (the author's own experiment).")
        panels = m.VGroup(*[card(6.3, 4.5, EDGE, PANEL_FILL, radius=0.2, stroke=2) for _ in range(2)])
        panels.arrange(m.RIGHT, buff=0.5).move_to([0, -0.35, 0])
        titles = m.VGroup(title_text("no compression", 30, ROSE), title_text("query-aware compression", 30, MINT))
        for t, p in zip(titles, panels):
            t.move_to(p.get_corner(m.UL) + [0.35, -0.45, 0], aligned_edge=m.LEFT)

        def dots(n, color, p):
            g = m.VGroup(*[m.Circle(radius=0.17).set_stroke(color, 3).set_fill(color, 0.25) for _ in range(n)]).arrange(m.RIGHT, buff=0.32)
            g.move_to([p.get_x(), 0.75, 0])
            nums = m.VGroup(*[code(str(i + 1), 20, MUTED).next_to(d, m.DOWN, buff=0.12) for i, d in enumerate(g)])
            return g, nums

        ld, ln = dots(EXP_FAIL_AT, ROSE, panels[0])
        rd, rn = dots(EXP_ITERS, MINT, panels[1])
        it_l = code("iteration", 20, MUTED).next_to(ln, m.DOWN, buff=0.1)
        it_r = code("iteration", 20, MUTED).next_to(rn, m.DOWN, buff=0.1)
        meter = Meter(5.0, 0.0, SUN, 0.22).move_to([panels[0].get_x(), -0.85, 0])
        m_cap = code(f"context · {EXP_CAP // 1000}K cap", 20, MUTED).next_to(meter, m.UP, buff=0.14).align_to(meter, m.LEFT)
        fail = pill(f"over {EXP_CAP // 1000}K at iteration {EXP_FAIL_AT}: failed", ROSE, ROSE_FILL, 20, mono=True).move_to([panels[0].get_x(), -1.85, 0])
        big = title_text(f"{EXP_TOKENS:,}", 56, MINT).move_to([panels[1].get_x(), -0.75, 0])
        big_sub = text("tokens in all", 24, MUTED).next_to(big, m.DOWN, buff=0.14)
        done = pill(f"{EXP_ITERS} iterations: done", MINT, MINT_FILL, 20, mono=True).move_to([panels[1].get_x(), -1.85, 0])
        quote = text("“reduces token usage by over 75%”  ·  Li, AI Agents in Depth · ch. 2, Experiment 2-10", 20, MUTED).move_to([0, -3.15, 0])

        self.say(
            "[Does it pay off? In an experiment from AI Agents in Depth], [a research agent without compression ran out of room at iteration five], [and failed.]",
            m.AnimationGroup(self.new_header("03 · does it pay off?", "A test from the book"), m.FadeIn(panels), m.FadeIn(titles), m.FadeIn(quote)),
            m.AnimationGroup(m.FadeIn(m_cap), m.FadeIn(meter), m.LaggedStart(*[m.FadeIn(m.VGroup(d, n), scale=0.6) for d, n in zip(ld, ln)], lag_ratio=0.3),
                             m.FadeIn(it_l), meter.fill_to(1.0, run_time=2.0)),
            m.FadeIn(fail, shift=0.1 * m.UP),
        )
        self.say(
            "[With compression that keeps what the question needs], [it finished in seven iterations], [using about forty thousand tokens.]",
            m.Circumscribe(titles[1], color=MINT, buff=0.08),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(m.VGroup(d, n), scale=0.6) for d, n in zip(rd, rn)], lag_ratio=0.25), m.FadeIn(it_r), m.FadeIn(done, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(big, shift=0.15 * m.UP), m.FadeIn(big_sub)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ what a summary loses

    def lost(self):
        self.next_section("4 · What gets lost", notes="A summary costs a model call and is lossy. Li, AI Agents in Depth, ch. 2 (Design Principles): compression most easily loses 'early architectural decisions, the reasons behind constraints, and failed paths'; save progress in documents rather than relying on the execution history. The three details are illustrative and match Otto's session (turn 9's test run failed two tests). Episode 14 (The File System as Memory) shows NOTES.md and progress files.")
        summ = summary_card(6.2).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.3, 0], m.UP)
        cost = pill("one model call", CORAL, CORAL_FILL, 22)
        lossy = pill("loses detail", ROSE, ROSE_FILL, 22)
        pills = m.VGroup(cost, lossy).arrange(m.RIGHT, buff=0.3).next_to(summ, m.DOWN, buff=0.35).align_to(summ, m.LEFT)
        head = text("in the history, not in the summary", 22, ROSE).move_to([0.2, 2.12, 0], aligned_edge=m.LEFT)
        items = m.VGroup()
        for tag, words in LOST:
            box = card(6.5, 0.62, ROSE, CARD, radius=0.12, stroke=1.5)
            t = code(tag, 20, MUTED).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            wds = text(words, 22, INK).move_to(box).align_to(box, m.LEFT).shift(1.45 * m.RIGHT)
            sit(wds, t[0].get_bottom()[1])
            assert t.get_right()[0] < wds.get_left()[0] - 0.1 and wds.get_right()[0] < box.get_right()[0] - 0.15, words
            items.add(m.VGroup(box, t, wds))
        items.arrange(m.DOWN, buff=0.16).next_to(head, m.DOWN, buff=0.22).align_to(head, m.LEFT)
        notes_src = "- codes live in coupons.py\n- cart.total(): the mobile app uses it\n- rounding per item: 2 tests failed"
        notes_body = code(notes_src, 20, INK)
        notes_cap = code("NOTES.md", 20, SUN, weight=m.BOLD)
        notes_box = card(6.45, notes_body.height + notes_cap.height + 0.85, MINT, CARD, radius=0.16, stroke=2)
        notes_cap.align_to(notes_box, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        notes_body.next_to(notes_cap, m.DOWN, buff=0.25).align_to(notes_cap, m.LEFT)
        notes = m.VGroup(notes_box, notes_cap, notes_body).next_to(items, m.DOWN, buff=0.45).align_to(items, m.LEFT)
        ep14 = pill("ep 14 · The File System as Memory", ORCHID, ORCHID_FILL, 20, mono=True).next_to(pills, m.DOWN, buff=0.45).align_to(pills, m.LEFT)
        user_line = summ[2][4]

        self.say(
            "[But a summary is the expensive layer]: [it costs a model call], [and it loses detail.]",
            m.AnimationGroup(self.new_header("04 · what gets lost", "What a summary forgets"), m.FadeIn(summ, shift=0.15 * m.UP)),
            m.FadeIn(cost, shift=0.1 * m.UP),
            m.FadeIn(lossy, shift=0.1 * m.UP),
        )
        self.say(
            "[What goes first? The book names three things]: [early decisions], [the reasons behind constraints], [and paths that already failed.]",
            m.FadeIn(head),
            m.FadeIn(items[0], shift=0.15 * m.LEFT),
            m.AnimationGroup(m.FadeIn(items[1], shift=0.15 * m.LEFT), m.Circumscribe(user_line, color=ROSE, buff=0.06)),
            m.FadeIn(items[2], shift=0.15 * m.LEFT),
        )
        self.say(
            "[Keep those in files, where no summary can lose them]: [episode fourteen shows how.]",
            m.AnimationGroup(m.FadeIn(notes_box), m.FadeIn(notes_cap),
                             m.LaggedStart(*[m.FadeIn(ln, target_position=it) for ln, it in zip(notes_body_lines(notes_body), items)], lag_ratio=0.3),
                             *[it[0].animate.set_stroke(MINT, 2) for it in items]),
            m.FadeIn(ep14, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the breaker

    def breaker(self):
        self.next_section("5 · The breaker", notes=f"Learn Claude Code s08: after {MAX_FAILURES} consecutive failures compact_history stops retrying; Claude Code's MAX_CONSECUTIVE_AUTOCOMPACT_FAILURES = 3 (the site's reading of the source). Li, AI Agents in Depth, ch. 5 (Failure and Error Recovery, 'every recovery path needs a ceiling'): one session failed over three thousand times in a row; futile retries wasted about 250,000 API calls per day worldwide; 'three is the empirical inflection point' (the book's study of Claude Code).")
        tries = m.VGroup()
        for i in range(MAX_FAILURES):
            box = card(2.35, 0.95, ROSE, CARD, radius=0.16, stroke=2)
            words = m.VGroup(code("compact", 20, INK), text(f"try {i + 1}", 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            words.move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            mark = cross(box.get_right() + [-0.42, 0, 0])
            tries.add(m.VGroup(box, words, mark))
        tries.arrange(m.RIGHT, buff=0.28)
        stop = pill("breaker: stop trying", ROSE, ROSE_FILL, 24)
        row_ = m.VGroup(tries, stop).arrange(m.RIGHT, buff=0.9).move_to([0, 1.35, 0])
        link = arrow(tries.get_right(), stop.get_left(), ROSE, buff=0.12)
        src_a = cite(f"Claude Code: {MAX_FAILURES} failures in a row  ·  learn-claude-code · s08").next_to(row_, m.DOWN, buff=0.4)
        stats = m.VGroup()
        for big_, sub_ in (("3,000+", "failures in a row, one session"), ("250,000", "wasted API calls a day, worldwide")):
            stats.add(m.VGroup(title_text(big_, 64, ROSE), text(sub_, 24, MUTED)).arrange(m.DOWN, buff=0.18))
        stats.arrange(m.RIGHT, buff=1.4).move_to([0, -1.45, 0])
        about = text("about", 22, MUTED).next_to(stats[1][0], m.LEFT, buff=0.2).align_to(stats[1][0], m.DOWN)
        src_b = cite("Li, AI Agents in Depth · ch. 5").move_to([0, -3.2, 0])

        self.say(
            "[And the summary call itself can fail.] [Claude Code stops after three failures in a row]: [a circuit breaker.]",
            m.AnimationGroup(self.new_header("05 · breaker", "Three strikes"), m.FadeIn(tries[0], shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.LaggedStart(m.FadeIn(tries[1], shift=0.15 * m.RIGHT), m.FadeIn(tries[2], shift=0.15 * m.RIGHT), lag_ratio=0.5), m.FadeIn(src_a)),
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(stop, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[The book tells why]: [one session once failed over three thousand times in a row.]",
            m.FadeIn(src_b),
            m.FadeIn(stats[0], shift=0.15 * m.UP),
        )
        self.say(
            "[Worldwide, retries like that wasted about two hundred and fifty thousand API calls a day.]",
            m.AnimationGroup(m.FadeIn(stats[1], shift=0.15 * m.UP), m.FadeIn(about)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ cheap first

    def cheap_first(self):
        self.next_section("6 · Cheap first", notes="Learn Claude Code s08: four layers, 'cheap first, expensive last'. Each iteration of agent_loop: messages[:] = tool_result_budget(messages); snip_compact; micro_compact; if estimate_token_count(messages) > THRESHOLD: compact_history; then the API call (s08's names; the call simplified as in harness_reference.py). The first three are plain code; only the summary calls the model.")
        src = (
            "def agent_loop(messages):\n"
            "    while True:\n"
            "        messages[:] = tool_result_budget(messages)\n"
            "        messages[:] = snip_compact(messages)\n"
            "        messages[:] = micro_compact(messages)\n"
            "        if estimate_token_count(messages) > THRESHOLD:\n"
            "            messages[:] = compact_history(messages)\n"
            "        response = client.messages.create(...)"
        )
        block = code_block(src, size=20, title="agent_loop · learn-claude-code s08")
        block.move_to([0, 0, 0]).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.3, 0], m.UP)
        block[1].set_z_index(2)
        block[2].set_z_index(2)
        bars = [line_bar(block, ["tool_result_budget"], SUN), line_bar(block, ["snip_compact"], SUN),
                line_bar(block, ["micro_compact"], SUN), line_bar(block, ["estimate_token_count", "compact_history"], CORAL)]
        quote = m.VGroup(title_text("cheap first,", 34, SUN), title_text("expensive last", 34, CORAL)).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT)
        quote.next_to(block, m.RIGHT, buff=0.5).align_to(block, m.UP).shift(0.35 * m.DOWN)
        q_src = cite("learn-claude-code · s08").next_to(quote, m.DOWN, buff=0.25).align_to(quote, m.LEFT)
        assert quote.get_right()[0] < 6.9
        ladder = m.VGroup()
        for i, (name, desc, cost, color) in enumerate([
            ("budget", "big results to disk", "free: plain code", MINT),
            ("snip", "cut the middle", "free: plain code", MINT),
            ("micro-compact", "shrink old results", "free: plain code", MINT),
            ("summary", "fold the history", "one model call", CORAL),
        ]):
            box = card(3.2, 1.75, color if i == 3 else EDGE, CARD, radius=0.16, stroke=2)
            body = m.VGroup(code(f"layer {i + 1}", 20, MUTED), title_text(name, 26, INK), text(desc, 20, MUTED),
                            text(cost, 20, color, weight=m.SEMIBOLD))
            body.arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            assert body.width < 3.2 - 0.4, name
            ladder.add(m.VGroup(box, body))
        ladder.arrange(m.RIGHT, buff=0.2).move_to([0, -2.5, 0])

        self.say(
            "[So Learn Claude Code puts three cheap layers in front of the summary]: [cheap first, expensive last.]",
            m.AnimationGroup(self.new_header("06 · cheap first", "Four layers, cheap first"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in ladder], lag_ratio=0.25)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.15 * m.LEFT), m.FadeIn(q_src)),
        )
        self.say(
            "[Each turn, before the call, they run one by one]: [a budget for tool results], [snip], [micro-compact], [and the summary, only if the context is still over the line.]",
            m.FadeIn(block, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(bars[0]), m.Circumscribe(ladder[0], color=SUN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(bars[1]), m.Circumscribe(ladder[1], color=SUN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(bars[2]), m.Circumscribe(ladder[2], color=SUN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(bars[3]), m.Circumscribe(ladder[3], color=CORAL, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ layer 1: the budget

    def budget(self):
        self.next_section("7 · Layer 1: the budget", notes=f"Learn Claude Code s08, tool_result_budget (teaching rule): if the tool_result blocks of the last user message exceed {MAX_BYTES:,} bytes, persist the largest (largest first) to .task_outputs/tool-results/ and keep a <persisted-output> marker plus a {PREVIEW_CHARS:,}-character preview (Claude Code: 200,000 characters). Example (illustrative): an {LOG_CHARS // 1000} KB test log = {LOG_TOKENS:,} tokens at {CHARS_PER_TOKEN} characters a token, the whole {WINDOW:,}-token window; after the budget the message is {MARKER_TOKENS:,} tokens (computed). The marker's layout on screen is simplified.")
        term = terminal(["$ cat logs/test.log", "! " + LOG_HEAD, f"… {LOG_CHARS:,} characters …"], width=6.6, size=20, title="tinyshop")
        term.align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.3, 0], m.UP)
        result = json_card({"type": "tool_result", "tool_use_id": "toolu_33", "content": f"FAILED tests/… ({LOG_CHARS // 1000} KB)"}, size=20, accent=MINT, title="tool_result")
        result.align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, -0.35, 0], m.UP)
        mark_src = f"<persisted-output>\nsaved: tool-results/toolu_33.txt\n{LOG_HEAD[:30]} …\n… first {PREVIEW_CHARS:,} characters …\n</persisted-output>"
        mark_body = code(mark_src, 20, INK, t2c={r"</?persisted-output>": MUTED, r"saved: .*": SUN, r"… first .*": MUTED})
        mark_box = card(mark_body.width + 0.6, mark_body.height + 0.55, MINT, CARD, radius=0.16, stroke=2)
        mark_body.move_to(mark_box)
        marker = m.VGroup(mark_box, mark_body).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, -0.35, 0], m.UP)
        assert marker.get_right()[0] < 0.4

        meter = Meter(5.6, 0.0, SUN, 0.24).move_to([3.75, 1.5, 0])
        m_cap = code("if it went in", 20, MUTED).next_to(meter, m.UP, buff=0.14).align_to(meter, m.LEFT)
        value = Slot(self)
        val_at = lambda s, c: code(s, 22, c).next_to(meter, m.UP, buff=0.12).align_to(meter, m.RIGHT)  # noqa: E731
        whole = text("the whole window", 24, ROSE, weight=m.SEMIBOLD).next_to(meter, m.DOWN, buff=0.2).align_to(meter, m.RIGHT)
        disk = files_card("on disk", [".task_outputs/", ".task_outputs/tool-results/", ".task_outputs/tool-results/toolu_33.txt"],
                          highlight=("toolu_33.txt",), width=4.4)
        disk.move_to([3.75, -1.1, 0])
        size_lab = code(f"{LOG_CHARS // 1000} KB", 20, MUTED).next_to(disk, m.DOWN, buff=0.15).align_to(disk, m.RIGHT)
        file_row = disk[1][1][2]
        reread = code(f'read_file("{LOG_PATH.split("/")[-1]}")', 20, SUN).next_to(size_lab, m.DOWN, buff=0.25).align_to(disk, m.RIGHT)
        rule = cite(f"over {MAX_BYTES // 1000} KB: the largest result goes to disk  ·  learn-claude-code · s08").move_to([0, -3.35, 0])

        self.say(
            "[Layer one is a budget for tool results.] [Say Otto prints a long test log]: [eight hundred kilobytes.]",
            m.AnimationGroup(self.new_header("07 · layer 1", "Spill big results to disk"), m.FadeIn(rule)),
            m.AnimationGroup(m.FadeIn(term[:4]), m.AddTextLetterByLetter(term[4][0], time_per_char=0.04), m.FadeIn(term[4][1:], run_time=1.2)),
            m.FadeIn(result, shift=0.2 * m.UP),
        )
        self.say(
            "[On its own, that's about two hundred thousand tokens]: [the whole window.]",
            m.AnimationGroup(m.FadeIn(meter), m.FadeIn(m_cap), meter.fill_to(LOG_TOKENS / WINDOW, run_time=1.4), value.to(val_at(f"≈ {LOG_TOKENS:,} tokens", ROSE))),
            m.AnimationGroup(m.FadeIn(whole, shift=0.1 * m.UP), m.Circumscribe(meter, color=ROSE, buff=0.08)),
        )
        self.say(
            "[Over two hundred kilobytes, the harness saves the largest result to disk], [and keeps a marker with a two thousand character preview.]",
            m.AnimationGroup(m.FadeIn(disk, shift=0.15 * m.LEFT), m.FadeIn(size_lab), send(result, file_row)),
            m.AnimationGroup(m.FadeTransform(result, marker), meter.fill_to(MARKER_TOKENS / WINDOW), value.to(val_at(f"+ {MARKER_TOKENS:,} tokens", MINT)),
                             m.FadeOut(whole), m.FadeOut(m_cap)),
        )
        self.say(
            "[Nothing is lost]: [the full log is one file read away.]",
            m.Circumscribe(file_row, color=SUN, buff=0.06),
            m.FadeIn(reread, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ layer 2: snip

    def snip(self):
        self.next_section("8 · Layer 2: snip", notes=f"Back to turn 32 before any summary (the same session). Layer 1 finds no result over the budget there (asserted). Learn Claude Code s08, snip_compact (teaching rule): over {MAX_MESSAGES} messages keep the first {KEEP_HEAD} and the last {KEEP_TAIL}, insert '[snipped N messages from conversation middle]', and never separate an assistant tool_use from the user tool_result after it. Otto's {N_MSG} messages: the 47th-from-last is turn 9's result, whose call would be cut, so the cut moves by one: {SNIP_N} messages snipped, {len(SNIPPED)} left (with the marker); {TOTAL:,} → {SNIP_TOKENS:,} tokens (computed).")
        panel, cap = map_panel()
        gauge = Gauge(self, TOTAL)
        bars = context_map(entries(SESSION))
        rewind = pill("back to turn 32 · before the summary", SUN, SUN_FILL, 20, mono=True).move_to([0, 2.25, 0]).align_to([LEFT_X, 0, 0], m.LEFT)
        count = Slot(self)
        count_at = lambda n: m.VGroup(title_text(f"{n} messages", 44, INK), code(f"limit {MAX_MESSAGES}: keep {KEEP_HEAD} + {KEEP_TAIL}", 22, MUTED)).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([LEFT_X, 0.95, 0], aligned_edge=m.LEFT)  # noqa: E731
        bx = PANEL_LEFT - 0.14
        head_br = bracket(map_y(2) + MAP_PITCH / 2, map_y(2 + KEEP_HEAD - 1) - MAP_PITCH / 2, bx, MINT, f"first {KEEP_HEAD}")
        t0 = 2 + N_MSG - KEEP_TAIL
        tail_br = bracket(map_y(t0) + MAP_PITCH / 2, map_y(MAP_ROWS - 1) - MAP_PITCH / 2, bx, MINT, f"last {KEEP_TAIL}")
        tail_br2 = bracket(map_y(t0 + 1) + MAP_PITCH / 2, map_y(MAP_ROWS - 1) - MAP_PITCH / 2, bx, MINT, f"last {KEEP_TAIL - 1}")
        pair = m.SurroundingRectangle(m.VGroup(bars[t0 - 1], bars[t0]), color=ROSE, buff=0.05, stroke_width=2.5)
        pair_lab = text("call + result stay together", 22, ROSE).move_to([PANEL_LEFT + 1.0, pair.get_y(), 0], aligned_edge=m.LEFT)
        pair_link = m.DashedLine([pair.get_right()[0] + 0.05, pair.get_y(), 0], pair_lab.get_left() + [-0.1, 0, 0], dash_length=0.07).set_stroke(ROSE, 2)
        assert pair_link.get_length() > 0.15 and pair_lab.get_right()[0] < PANEL_LEFT + PANEL_W - 0.2

        # the cut
        mid = [bars[i] for i in range(2 + KEEP_HEAD, 2 + SNIP_START)]
        snip_bar = bar("snip", tokens_of(SNIPPED[KEEP_HEAD]), 2 + KEEP_HEAD)
        rise = (SNIP_N - 1) * MAP_PITCH
        tail = [bars[i] for i in range(2 + SNIP_START, MAP_ROWS)]
        marker = code(f"[snipped {SNIP_N} messages\n from conversation middle]", 20, MUTED).move_to([-3.1, -1.5, 0])
        m_link = m.DashedLine(marker.get_right() + [0.1, 0, 0], [snip_bar.get_left()[0] - 0.06, snip_bar.get_y(), 0], dash_length=0.07).set_stroke(MUTED, 2)
        self.live = [bars[0], bars[1]] + [bars[2 + i] for i in range(KEEP_HEAD)] + [snip_bar] + tail  # map rows after the cut
        self.map = (panel, cap, gauge)

        self.say(
            "[Now back to turn thirty-two, before the summary.] [This time, the cheap layers go first.]",
            m.AnimationGroup(self.new_header("08 · layer 2", "Snip the middle"), m.FadeIn(panel), m.FadeIn(cap), m.FadeIn(gauge.frame),
                             gauge.counter.to(gauge.count(TOTAL)), m.FadeIn(rewind, shift=0.1 * m.DOWN),
                             m.LaggedStart(*[m.FadeIn(b) for b in bars], lag_ratio=0.02, run_time=1.4)),
            m.Circumscribe(gauge.meter, color=SUN, buff=0.08),
        )
        self.say(
            "[Layer two is snip.] [Over fifty messages], [it keeps the first three], [and the last forty-seven.]",
            m.Wait(0.1),
            count.to(count_at(N_MSG)),
            m.FadeIn(head_br, shift=0.1 * m.RIGHT),
            m.FadeIn(tail_br, shift=0.1 * m.RIGHT),
        )
        self.say(
            "[One rule: never split a tool call from its result.] [So here the cut moves by one message.]",
            m.AnimationGroup(m.Create(pair), m.FadeIn(pair_lab), m.Create(pair_link)),
            m.Transform(tail_br, tail_br2),
        )
        self.say(
            "[The sixteen messages in the middle are cut], [leaving one line that says so]: [a hundred and thirty-eight thousand tokens.]",
            m.AnimationGroup(*[m.FadeOut(b, shift=0.2 * m.RIGHT) for b in mid], m.FadeOut(pair), m.FadeOut(pair_lab), m.FadeOut(pair_link),
                             m.FadeOut(head_br), m.FadeOut(tail_br), run_time=0.9),
            m.AnimationGroup(m.FadeIn(snip_bar), *[b.animate.shift(rise * m.UP) for b in tail], m.FadeIn(marker), m.Create(m_link), run_time=1.0),
            m.AnimationGroup(gauge.to(SNIP_TOKENS), count.to(count_at(len(SNIPPED)))),
        )
        self.wait(0.3)
        self.keep = [panel, cap, gauge.frame, gauge.counter.current, *self.live]

    # ------------------------------------------------------------------ layer 3: micro-compact

    def micro(self):
        self.next_section("9 · Layer 3: micro-compact", notes=f"Learn Claude Code s08, micro_compact (teaching rule): keep the {KEEP_RECENT_TOOL_RESULTS} most recent tool_results; older ones longer than {MICRO_MIN_CHARS} characters become '{PLACEHOLDER}'. On Otto's snipped context: {len(SHRUNK)} results shrink; {SNIP_TOKENS:,} → {MICRO_TOKENS:,} tokens, against {FOLDED_TOKENS:,} after the summary (computed). Order matters: s08 runs the budget before micro because micro replaces large results with placeholders and the budget has to persist the full content first. Replayed with the three cheap layers at the top of every turn, the same 32 turns peak at {PEAK:,} tokens ({100 * PEAK / WINDOW:.0f}% of the window; computed, not narrated).")
        self.clear(*self.keep)
        panel, cap, gauge = self.map
        live = self.live
        shrink = [m.Transform(live[2 + i], bar("tool_result", tokens_of(MICRO[i]), 2 + i)) for i in SHRUNK]
        results_idx = [i for i, x in enumerate(SNIPPED) if is_tool_result(x)]
        recent = m.VGroup(*[live[2 + i] for i in results_idx[-KEEP_RECENT_TOOL_RESULTS:]])
        recent_box = m.SurroundingRectangle(recent, color=MINT, buff=0.06, stroke_width=2.5)
        big_old = [live[2 + i] for i in SHRUNK if tokens_of(SNIPPED[i]) >= 4_000]

        def rcard(tid, what, body, tokens, w=6.6):
            box = card(w, 1.4, EDGE, CARD, radius=0.14, stroke=1.5)
            stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=1.2, stroke_width=0, fill_color=MINT, fill_opacity=1)
            stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
            head = code(f"{tid} · {what}", 20, MUTED).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.18 * m.DOWN)
            cnt = code(f"{tokens:,}", 20, MUTED).align_to(box, m.UR).shift(0.25 * m.LEFT + 0.18 * m.DOWN)
            sit(cnt, head[0].get_bottom()[1])
            words = code(body, 20, MINT).next_to(head, m.DOWN, buff=0.16).align_to(head, m.LEFT)
            return m.VGroup(box, stripe, head, cnt, words)

        def tool_of(k):
            return TURNS[k - 1]

        a = rcard("toolu_13", tool_of(13)[2], tool_of(13)[3], tool_of(13)[4])
        b = rcard("toolu_16", tool_of(16)[2], tool_of(16)[3], tool_of(16)[4])
        c = rcard("toolu_32", "read " + tool_of(32)[2], tool_of(32)[3], tool_of(32)[4])
        cards = m.VGroup(a, b, c).arrange(m.DOWN, buff=0.22).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.35, 0], m.UP)
        kept = pill("recent: kept", MINT, MINT_FILL, 20).align_to(c[0], m.DR).shift(0.2 * m.LEFT + 0.15 * m.UP)
        ph = [code("[Earlier tool result compacted.\n Re-run if needed.]", 20, MUTED).next_to(x[2], m.DOWN, buff=0.16).align_to(x[2], m.LEFT) for x in (a, b)]
        ph_cnt = [code(f"{tokens_of({'role': 'user', 'content': PLACEHOLDER}):,}", 20, MUTED).move_to(x[3]).align_to(x[3], m.RIGHT) for x in (a, b)]
        rerun = code('bash("cat logs/app.log")', 20, SUN)
        rerun_lab = text("re-run:", 24, MUTED)
        rr = m.VGroup(rerun_lab, rerun).arrange(m.RIGHT, buff=0.2).next_to(cards, m.DOWN, buff=0.3).align_to(cards, m.LEFT)
        sit(rerun_lab, rerun[0].get_bottom()[1])
        s1 = pill("budget saves it", SUN, SUN_FILL, 20)
        s2 = pill("micro replaces it", MINT, MINT_FILL, 20)
        strip = m.VGroup(s1, s2).arrange(m.RIGHT, buff=0.8).next_to(cards, m.DOWN, buff=0.35).align_to(cards, m.LEFT)
        s_arrow = arrow(s1.get_right(), s2.get_left(), MUTED, buff=0.1)
        stat1 = m.VGroup(title_text(f"{MICRO_TOKENS:,} tokens", 52, MINT), text("three cheap layers · no model call", 24, MUTED)).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        stat2 = m.VGroup(title_text(f"{FOLDED_TOKENS:,} tokens", 52, AZURE), text("the summary · one model call", 24, MUTED)).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        stats = m.VGroup(stat1, stat2).arrange(m.DOWN, buff=0.6, aligned_edge=m.LEFT).align_to([LEFT_X, 0, 0], m.LEFT).set_y(0.1)

        self.say(
            "[Layer three is micro-compact.] [Old tool results are most of what's left], [and Otto rarely needs them again.]",
            self.new_header("09 · layer 3", "Shrink old results"),
            m.LaggedStart(*[m.Circumscribe(x, color=MINT, buff=0.03) for x in big_old], lag_ratio=0.2),
            m.AnimationGroup(m.FadeIn(a, shift=0.15 * m.RIGHT), m.FadeIn(b, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[The three most recent results stay whole.] [Every older one over a hundred and twenty characters], [becomes one short line.]",
            m.AnimationGroup(m.FadeIn(c, shift=0.15 * m.RIGHT), m.FadeIn(kept), m.Create(recent_box)),
            m.AnimationGroup(m.Circumscribe(a[3], color=MINT, buff=0.06), m.Circumscribe(b[3], color=MINT, buff=0.06)),
            m.AnimationGroup(m.LaggedStart(*shrink, lag_ratio=0.08, run_time=1.6), gauge.to(MICRO_TOKENS, run_time=1.6),
                             *[m.FadeTransform(x[4], p) for x, p in zip((a, b), ph)],
                             *[m.FadeTransform(x[3], n) for x, n in zip((a, b), ph_cnt)]),
        )
        for x in (a, b):  # their old body and count were faded out: drop them (a later fade would bring them back)
            x.remove(x[4], x[3])
        self.say(
            "[Earlier tool result compacted. Re-run if needed.] [If Otto needs that output again], [it can simply run the tool again.]",
            m.Circumscribe(ph[0], color=MUTED, buff=0.06),
            m.FadeIn(rerun_lab),
            m.FadeIn(rerun, shift=0.1 * m.UP),
        )
        self.say(
            "[That's why the budget runs first]: [a big result is saved to disk], [before micro-compact replaces it.]",
            m.AnimationGroup(m.FadeOut(rr), m.FadeIn(s1, shift=0.1 * m.UP)),
            m.Circumscribe(s1, color=SUN, buff=0.06),
            m.AnimationGroup(m.GrowArrow(s_arrow), m.FadeIn(s2, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[After three cheap layers], [Otto's context is under twenty thousand tokens], [about what the summary left], [without a single model call.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (a, b, c, kept, s1, s2, s_arrow, recent_box, *ph, *ph_cnt)]),
            m.AnimationGroup(m.FadeIn(stat1, shift=0.15 * m.UP), m.Circumscribe(gauge.counter.current, color=MINT, buff=0.08)),
            m.FadeIn(stat2, shift=0.15 * m.UP),
            m.Circumscribe(stat1[1], color=MINT, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the catch: the cache

    def catch(self):
        self.next_section("10 · The catch", notes=f"Tension between the two references. Learn Claude Code s08's snip and micro layers edit old messages in place (Claude Code's cached micro path uses the API's cache_edits, per the site). Li, AI Agents in Depth, ch. 2: 'truncating old tool results in place' is a common pitfall that invalidates the prompt cache; 'it is best to perform batch compression when the context approaches the threshold, rather than compressing every round'. Episode 7: a change early in the prompt invalidates the cache for everything after it. The strip and the two schedules ({len(EVERY_TURN)} cache breaks vs {len(BATCHED)}) are illustrative.")
        size, gap = 0.62, 0.14
        blocks = m.VGroup(*[m.RoundedRectangle(corner_radius=0.1, width=size, height=size, stroke_width=0,
                                                 fill_color=ACCENT[k], fill_opacity=1) for k in STRIP])
        blocks.arrange(m.RIGHT, buff=gap).move_to([0, 1.05, 0])
        top = blocks.get_top()[1] + 0.25
        x_a, x_e, x_b = blocks.get_left()[0], blocks[EDITED].get_left()[0] - gap / 2, blocks.get_right()[0]

        def span_line(a, b, color):
            line = m.Line([a, top, 0], [b, top, 0]).set_stroke(color, 4)
            ends = m.VGroup(*[m.Line([x, top, 0], [x, top - 0.15, 0]).set_stroke(color, 4) for x in (a, b)])
            return m.VGroup(line, ends)

        cached = span_line(x_a, x_b, SUN)
        cached_lab = text("cached from the last call", 22, SUN).next_to(cached, m.UP, buff=0.12)
        still = span_line(x_a, x_e - 0.04, SUN)
        again = span_line(x_e + 0.04, x_b, ROSE)
        still_lab = text("still cached", 22, SUN).next_to(still, m.UP, buff=0.12)
        again_lab = text("processed again", 22, ROSE).next_to(again, m.UP, buff=0.12)
        edited = m.RoundedRectangle(corner_radius=0.05, width=size, height=0.18, stroke_width=0, fill_color=MINT, fill_opacity=1)
        edited.move_to(blocks[EDITED]).align_to(blocks[EDITED], m.DOWN)
        edit_lab = code("edited in place", 20, ROSE).next_to(blocks[EDITED], m.DOWN, buff=0.2)
        ep07 = pill("ep 07 · KV Cache & Prompt Caching", SUN, SUN_FILL, 20, mono=True).next_to(edit_lab, m.DOWN, buff=0.3).align_to(blocks, m.RIGHT)

        def timeline(label, marks, y):
            lab = text(label, 24, INK, weight=m.SEMIBOLD).move_to([LEFT_X, y, 0], aligned_edge=m.LEFT)
            xs = [-1.4 + i * 0.52 for i in range(TICKS)]
            ticks = m.VGroup(*[m.Line([x, y - 0.18, 0], [x, y + 0.18, 0]).set_stroke(FAINT, 3) for x in xs])
            dots = m.VGroup(*[m.Dot([xs[i], y + 0.38, 0], radius=0.09, color=ROSE) for i in marks])
            n = code(f"{len(marks)} cache {'break' if len(marks) == 1 else 'breaks'}", 20, ROSE).next_to(ticks, m.DOWN, buff=0.15).align_to(ticks, m.RIGHT)
            assert lab.get_right()[0] < xs[0] - 0.3
            return lab, ticks, dots, n

        b_lab, b_ticks, b_dots, b_n = timeline("in batches, near the line", BATCHED, -1.3)
        e_lab, e_ticks, e_dots, e_n = timeline("a little every turn", EVERY_TURN, -2.65)
        line_mark = m.DashedLine([b_dots[0].get_x(), -1.45, 0], [b_dots[0].get_x(), -0.7, 0], dash_length=0.06).set_stroke(SUN, 2)
        src_b = cite("Li, AI Agents in Depth · ch. 2").move_to([LEFT_X, -3.4, 0], aligned_edge=m.LEFT)

        self.say(
            "[There's a catch.] [Snip and micro-compact edit old messages in place.]",
            m.AnimationGroup(self.new_header("10 · the catch", "Compaction and the cache"),
                             m.LaggedStart(*[m.FadeIn(x, shift=0.15 * m.UP) for x in blocks], lag_ratio=0.06), m.FadeIn(cached), m.FadeIn(cached_lab)),
            m.AnimationGroup(m.Transform(blocks[EDITED], edited), m.FadeIn(edit_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[And as episode seven showed], [one early edit breaks the prompt cache for everything after it.]",
            m.FadeIn(ep07, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(cached), m.FadeOut(cached_lab), m.FadeIn(still), m.FadeIn(still_lab), m.FadeIn(again), m.FadeIn(again_lab),
                             *[x.animate.set_opacity(0.45) for x in blocks[EDITED + 1:]]),
        )
        self.say(
            "[So the book's advice is to compress in batches], [as the context nears the line], [not a little every turn.]",
            m.AnimationGroup(m.FadeIn(b_lab), m.FadeIn(b_ticks), m.FadeIn(src_b)),
            m.AnimationGroup(m.Create(line_mark), m.FadeIn(b_dots, scale=0.5), m.FadeIn(b_n)),
            m.AnimationGroup(m.FadeIn(e_lab), m.FadeIn(e_ticks), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in e_dots], lag_ratio=0.15), m.FadeIn(e_n)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Watch (a threshold below the top of the window: Claude Code's window − max output − 13,000; Li: act earlier), shrink (cheap first: budget, snip, micro-compact; the summary last, with a 3-failure breaker), keep (what the work depends on: the summary's five things, decisions and reasons in files).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Watch", SUN), ("Shrink", MINT), ("Keep", AZURE))])
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("a line below the top", "cheap first, expensive last", "what the work depends on")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = min(w.get_bottom()[1] for w in words) - 0.45
        for s in subs:
            s.set_y(line_y)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Make room cheaply, and keep what the work depends on in files.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [watch the line], [shrink cheap first], [and keep what matters.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Make room cheaply, and keep what the work depends on in files.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)


def notes_body_lines(body):
    """The glyph groups of each line of a multi-line Text (one VGroup per line)."""
    out, k = [], 0
    for line in body.text.split("\n"):
        n = sum(not c.isspace() for c in line)
        out.append(body[k: k + n])
        k += n
    return out
