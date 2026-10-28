"""Agents, Animated · Episode 32 · Running Agents in Production (Module 6 · Beyond the Loop)

What it takes to run agents for real users. Running example: say Otto now serves tinyshop's whole team
(eight developers, forty tasks a day, some overnight). Six questions that a laptop never asked, one
section each: what does a task cost, how long does it take, what happened inside it, what if the server
crashes halfway, what stops a runaway, and did yesterday's change make Otto worse?

Nothing on screen is drawn by hand: the scene computes the cost ledger of a 20-turn task with and without
prompt caching, a latency budget, runs a traced copy of the reference agent_loop (a stand-in tracer with
OpenTelemetry's method names, a scripted model on a simulated clock) to get the waterfall, runs a logged
agent_loop that crashes at turn twelve and resumes from its append-only log (the record / resume code on
screen is exec'd), runs the budget check on screen against a model that never stops, and computes the
eval comparison. Every number on screen comes from those runs and is asserted below.

Arc: hook (one developer, one laptop → a team; six questions) · 1 cost per task (re-sent history, 250,000
vs 44,800 full-price tokens, ×3.6 for twice the turns) · 2 latency budget (+2 s a round = +40 s; 2,000
thinking tokens at 50 tokens/s = 40 s) · 3 traces and spans (the waterfall; span attributes; the slowest
span; OpenTelemetry + OpenInference; versions on every trace) · 4 a crash, resumed (append-only log;
start over vs resume; the interrupted tool call; idle watchdog; neutral format) · 5 budgets and rate limits
(turn and token budget; a shared rate limit; a runaway's own key) · 6 evals in CI (prompt change → eval set
→ compare → gate; production traces become regression tests) · 7 production is a loop · recap (Measure ·
Limit · Resume).

Colours (series roles): model CORAL, tool call SUN, tool result MINT, user/team AZURE, system prompt
ORCHID; full-price tokens SUN, cache reads MINT (as in Ep 7); bad ROSE, good MINT; headers LIME (Module 6).

Facts (checked October 2026):
- Li Bojie, AI Agents in Depth, ch. 7 (bojieli.github.io/ai-agent-book/book-en/chapter7/): re-sent history
  ("Round 1 sends 1000 tokens, Round 2 sends 2000 tokens, Round 3 sends 3000 tokens", 6,000 instead of
  3,000); "at 50 tokens/s, a model producing 2000 thinking tokens spends 40 seconds just thinking"; Agent
  Observability: traces ("Agent observability is built on the foundation of traces"; one trace per task
  execution), "each LLM call, each tool call, and each retrieval is a span", "OpenTelemetry is the
  general-purpose distributed tracing standard, while specifications like OpenInference define
  LLM-specific semantic conventions on top of it"; production traces → "extract failed and suspicious
  cases", anonymize, "distill them into new test cases and regression tests"; "the failure patterns
  exposed in production today become the regression tests guarding the baseline tomorrow"; "Every prompt
  change should run regression tests on the evaluation set—just as code changes require CI"; prompt version
  management ("each version is associated with runtime performance data").
- Li, ch. 2: cache reads cost "about one-tenth the price with Anthropic, DeepSeek, and GPT-5" (Ep 7).
- Li, ch. 1 (how to choose a model): "a 20-round Agent task that runs 2 seconds slower per round means an
  extra 40 seconds of waiting".
- Li, ch. 5 (failure and error recovery): a silent stall is "like a connected pipe that yields no water";
  "a production Agent needs an independent idle watchdog"; "a trajectory should not be stored in any single
  vendor's wire format, but kept in a neutral one" (handover to another model); in production, patch a
  missing tool_result with a placeholder (research notes §14).
- Li, ch. 10 (Failure Mode Five: Runaway Loops): "A runaway Agent may spawn thousands of sub-agents";
  "For highly autonomous Agents, use a dedicated API key as part of controlling token spending";
  "Explicit budgets, cancellation and stop conditions are required to keep execution bounded"; "set a step
  or token budget when starting a sub-agent"; "resource quotas, and rate limits".
- Li, ch. 1 (Design Patterns That Run Through the Book): append-only history — "What this buys is
  cacheability, replayability and auditability".
- OpenInference semantic conventions (github.com/Arize-ai/openinference, spec/semantic_conventions.md):
  openinference.span.kind (LLM, TOOL, RETRIEVER, ...), llm.model_name, llm.token_count.prompt,
  llm.token_count.completion. The tracer here is a stand-in with OpenTelemetry's method names
  (start_as_current_span, set_attribute); the model name is harness_reference.py's MODEL.
- Checkpoint / resume after a crash is not covered by either reference (research notes §26): the section
  shows the practical pattern that append-only logs make possible, conservatively.

Illustrative (for teaching, said "for example" or "say"): the team (8 developers × 5 tasks = 40 a day),
the task names, the 3,000-token prompt and 1,000 tokens a turn, the 2-minute budget, 4 s model + 1 s tools
per round, the waterfall's durations and token counts (TTFT 0.8 s + output at 50 tokens/s), the version
labels, the crash at turn 12 and the log's path, the budget (30 turns, 400,000 tokens), the shares of the
rate limit, the spawn tree, the eval numbers (73 vs 58 of 100 tasks; v15 takes 16 turns), task-0187.
Cost is counted in "full-price tokens": input only, cache reads ×0.1, cache writes at the normal input
price for simplicity (as in Ep 7); real prices vary by provider and model.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import io
import json
import re
import sys
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

REF = (SERIES / "harness_reference.py").read_text()
MODEL = re.search(r'^MODEL = "(.*?)"', REF, re.M).group(1)
assert MODEL == "claude-sonnet-4-5" and "def agent_loop(messages):" in REF

# ---------------------------------------------------------------------------- the team (illustrative)

DEVELOPERS, TASKS_EACH = 8, 5
TASKS_A_DAY = DEVELOPERS * TASKS_EACH
assert TASKS_A_DAY == 40  # "eight developers, forty tasks a day"

# ---------------------------------------------------------------------------- 1 · cost per task

PREFIX, PER_TURN, TURNS = 3000, 1000, 20  # illustrative token counts
CACHE_READ = 0.1  # Li ch. 2: cache reads about one tenth of the price


def sent(turns):
    """Input tokens each call sends: the whole history so far."""
    return [PREFIX + PER_TURN * k for k in range(turns)]


def new_tokens(turns):
    """Tokens no earlier call had sent (not in the cache)."""
    return [PREFIX] + [PER_TURN] * (turns - 1)


def billed(turns):
    """Full-price tokens with caching: new tokens ×1, re-sent tokens ×0.1."""
    total, new = sum(sent(turns)), sum(new_tokens(turns))
    return new + CACHE_READ * (total - new)


SENT = sent(TURNS)
NEW = new_tokens(TURNS)
READ = [s - n for s, n in zip(SENT, NEW)]
assert all(READ[k] == SENT[k - 1] for k in range(1, TURNS))  # each call re-sends the previous call's prompt
TOTAL, NEW_TOTAL, READ_TOTAL = sum(SENT), sum(NEW), sum(READ)
assert TOTAL == 250_000 and NEW_TOTAL == 22_000 and READ_TOTAL == 228_000  # "two hundred and fifty thousand … twenty-two thousand"
BILLED = billed(TURNS)
assert round(BILLED) == 44_800 and BILLED / TOTAL < 0.2  # "forty-four thousand eight hundred … under a fifth"
PCT = round(100 * BILLED / TOTAL)
assert PCT == 18
TOTAL_40 = sum(sent(40))
assert TOTAL_40 == 900_000 and TOTAL_40 / TOTAL == 3.6  # "forty turns … nine hundred thousand, three point six times"

# ---------------------------------------------------------------------------- 2 · latency

BUDGET_S, ROUNDS, MODEL_S, TOOL_S, SLOWER_S = 120, 20, 4, 1, 2  # budget and per-round times illustrative
ROUND_S = MODEL_S + TOOL_S
BASE_S = ROUNDS * ROUND_S
LATE_S = ROUNDS * (ROUND_S + SLOWER_S)
assert BASE_S == 100 <= BUDGET_S < LATE_S == 140 and LATE_S - BASE_S == 40  # Li ch. 1: "an extra 40 seconds"
THINK_TOKENS, TOK_PER_S = 2000, 50
THINK_S = THINK_TOKENS / TOK_PER_S
assert THINK_S == 40  # Li ch. 7: "spends 40 seconds just thinking"

# ---------------------------------------------------------------------------- 3 · a traced run

CLOCK = [0.0]  # simulated seconds


class Span:
    def __init__(self, name, parent):
        self.name, self.parent, self.attributes = name, parent, {}
        self.start, self.end = CLOCK[0], None

    def set_attribute(self, key, value):
        self.attributes[key] = value


class Tracer:
    """A stand-in for OpenTelemetry's tracer (same method names), on the simulated clock."""

    def __init__(self):
        self.spans, self.stack = [], []

    @contextmanager
    def start_as_current_span(self, name):
        span = Span(name, self.stack[-1] if self.stack else None)
        self.spans.append(span)
        self.stack.append(span)
        try:
            yield span
        finally:
            self.stack.pop()
            span.end = round(CLOCK[0], 3)


tracer = Tracer()
TTFT_S = 0.8  # time to first token (illustrative)
STEPS = [  # Otto's checkout fix, call by call: (tool, input, output tokens) — illustrative
    ("bash", {"command": "pytest -q"}, 60),
    ("read_file", {"path": "cart.py"}, 50),
    ("edit_file", {"path": "cart.py", "old_text": "range(len(items) - 1)", "new_text": "range(len(items))"}, 420),
    ("bash", {"command": "pytest -q"}, 40),
    (None, None, 90),
]
TOOL_TIME = {"pytest -q": 2.4, "read_file": 0.1, "edit_file": 0.1}
TEST_RUNS = ["1 failed, 12 passed", "13 passed"]  # tinyshop's running bug, before and after the fix


class TracedModel:
    def __init__(self):
        self.calls = 0

    def create(self, model, messages, **kw):
        tool, args, out = STEPS[self.calls]
        self.calls += 1
        CLOCK[0] += TTFT_S + out / TOK_PER_S
        usage = NS(input_tokens=PREFIX + PER_TURN * (self.calls - 1), output_tokens=out)
        if tool is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text="Fixed: total() counts every item.")], usage=usage)
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"toolu_0{self.calls}", name=tool, input=args)], usage=usage)


traced_client = NS(messages=TracedModel())


def traced_run_tool(block):
    key = block.input.get("command", block.name)
    CLOCK[0] += TOOL_TIME[key]
    return TEST_RUNS.pop(0) if key == "pytest -q" else "ok"


VERSIONS = {"model": MODEL, "prompt": "v14", "tools": "v6", "harness": "2.3.0"}  # labels illustrative


def traced_agent_loop(messages):
    with tracer.start_as_current_span("task") as task:
        for key, value in VERSIONS.items():
            task.set_attribute(f"version.{key}", value)
        while True:
            with tracer.start_as_current_span("llm") as span:
                response = traced_client.messages.create(model=MODEL, messages=messages)
                span.set_attribute("openinference.span.kind", "LLM")
                span.set_attribute("llm.model_name", MODEL)
                span.set_attribute("llm.token_count.prompt", response.usage.input_tokens)
                span.set_attribute("llm.token_count.completion", response.usage.output_tokens)
            messages.append({"role": "assistant", "content": response.content})
            if response.stop_reason != "tool_use":
                return response
            results = []
            for block in response.content:
                if block.type == "tool_use":
                    with tracer.start_as_current_span("tool") as span:
                        span.set_attribute("openinference.span.kind", "TOOL")
                        span.set_attribute("tool.name", block.name)
                        output = traced_run_tool(block)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
            messages.append({"role": "user", "content": results})


traced_agent_loop([{"role": "user", "content": "The checkout test fails. Fix it."}])
SPANS = tracer.spans
ROOT = SPANS[0]
CHILDREN = SPANS[1:]
assert ROOT.name == "task" and all(s.parent is ROOT for s in CHILDREN) and len(CHILDREN) == 9
assert [s.name for s in CHILDREN] == ["llm", "tool"] * 4 + ["llm"]
DUR = [round(s.end - s.start, 3) for s in CHILDREN]
assert DUR == [2.0, 2.4, 1.8, 0.1, 9.2, 0.1, 1.6, 2.4, 2.6]
assert ROOT.start == 0 and ROOT.end == 22.2  # "22.2 s"
SLOW = max(range(len(CHILDREN)), key=lambda i: DUR[i])
assert SLOW == 4 and CHILDREN[SLOW].name == "llm" and round(DUR[SLOW]) == 9  # "about nine seconds", the planning call
SLOW_SPAN = CHILDREN[SLOW]
assert SLOW_SPAN.attributes["llm.token_count.completion"] == 420 and SLOW_SPAN.attributes["llm.token_count.prompt"] == 5000
assert round(TTFT_S + 420 / TOK_PER_S, 3) == DUR[SLOW]
assert TEST_RUNS == []  # both test runs happened
ROW_NAMES = []
_calls = iter(range(1, 6))
for s in CHILDREN:
    if s.name == "llm":
        ROW_NAMES.append(f"llm · call {next(_calls)}")
    else:
        step = STEPS[len([r for r in ROW_NAMES if r.startswith("llm")]) - 1]
        ROW_NAMES.append(f"{s.attributes['tool.name']} · {step[1].get('command', step[1].get('path'))}")
assert ROW_NAMES[1] == "bash · pytest -q" and ROW_NAMES[3] == "read_file · cart.py" and ROW_NAMES[4] == "llm · call 3"
SPAN_ATTRS = {k: SLOW_SPAN.attributes[k] for k in ("openinference.span.kind", "llm.model_name", "llm.token_count.prompt", "llm.token_count.completion")}

# ---------------------------------------------------------------------------- 4 · a crash, resumed

DISK = {}  # an in-memory disk: the log survives the restart, the messages list does not
LOG_PATH = "runs/task-0042.jsonl"
CRASH_TURN = 12
INTERRUPTED = "interrupted by restart"


class LogFile:
    def __init__(self, path):
        self.path, self.pending = path, ""

    def write(self, s):
        self.pending += s

    def flush(self):
        DISK[self.path] = DISK.get(self.path, "") + self.pending
        self.pending = ""


def disk_open(path, mode="r"):
    return LogFile(path) if mode == "a" else io.StringIO(DISK[path])


RECORD_SRC = '''def record(messages, log, message):
    messages.append(message)
    log.write(json.dumps(message) + "\\n")
    log.flush()  # on disk before the next step

def resume(path):
    messages = [json.loads(s) for s in open(path)]
    return agent_loop(messages)'''


class Restart(Exception):
    pass


class LongModel:
    """A 20-turn task: a tool call each turn, then an answer."""

    def __init__(self):
        self.calls = 0

    def create(self, model, messages, **kw):
        self.calls += 1
        turn = sum(msg["role"] == "assistant" for msg in messages) + 1
        if turn < TURNS:
            return NS(stop_reason="tool_use", content=[{"type": "tool_use", "id": f"toolu_{turn:02d}", "name": "bash", "input": {"command": f"step {turn}"}}])
        return NS(stop_reason="end_turn", content=[{"type": "text", "text": "Done."}])


LOG_STATE = NS(crashed=False, lines_at_crash=None)


def logged_run_tool(block):
    if block["id"] == f"toolu_{CRASH_TURN:02d}" and not LOG_STATE.crashed:
        LOG_STATE.crashed = True
        LOG_STATE.lines_at_crash = DISK[LOG_PATH].count("\n")
        raise Restart()  # the server restarts while the tool runs
    return "ok"


def patch_interrupted(messages, log, record):
    """A tool call with no result (the run died while it ran) gets a placeholder result."""
    last = messages[-1]
    calls = [b for b in last["content"] if isinstance(b, dict) and b.get("type") == "tool_use"] if last["role"] == "assistant" else []
    if calls:
        record(messages, log, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": b["id"], "content": INTERRUPTED, "is_error": True} for b in calls]})


def logged_agent_loop(messages):
    log = LOG_NS["open"](LOG_PATH, "a")
    patch_interrupted(messages, log, LOG_NS["record"])
    while True:
        response = long_client.messages.create(model=MODEL, messages=messages)
        LOG_NS["record"](messages, log, {"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return messages
        results = []
        for block in response.content:
            if block["type"] == "tool_use":
                results.append({"type": "tool_result", "tool_use_id": block["id"], "content": logged_run_tool(block)})
        LOG_NS["record"](messages, log, {"role": "user", "content": results})


long_client = NS(messages=LongModel())
LOG_NS = {"json": json, "open": disk_open, "agent_loop": logged_agent_loop}
exec(RECORD_SRC, LOG_NS)  # the code on screen is the code that runs

_first = []
_log = disk_open(LOG_PATH, "a")
LOG_NS["record"](_first, _log, {"role": "user", "content": "Rename every price field to price_cents."})
try:
    logged_agent_loop(_first)
except Restart:
    pass  # the worker is gone, and its messages list with it
del _first
assert LOG_STATE.lines_at_crash == 24 == 1 + 2 * (CRASH_TURN - 1) + 1  # user + 11 finished turns + turn 12's call
CALLS_BEFORE = long_client.messages.calls
assert CALLS_BEFORE == CRASH_TURN
FINAL = LOG_NS["resume"](LOG_PATH)
LINES = [json.loads(s) for s in DISK[LOG_PATH].splitlines()]
assert FINAL == LINES and len(LINES) == 40  # the log and the rebuilt messages agree
assert long_client.messages.calls == TURNS  # 12 calls before the crash, 8 after: nothing paid twice
PLACEHOLDER = 1 + 2 * (CRASH_TURN - 1) + 1  # index of the placeholder message
assert LINES[PLACEHOLDER]["content"][0]["is_error"] and LINES[PLACEHOLDER]["content"][0]["content"] == INTERRUPTED
assert LINES[PLACEHOLDER]["content"][0]["tool_use_id"] == f"toolu_{CRASH_TURN:02d}"
ROLES_SEQ = ["user"] + ["assistant", "user"] * (TURNS - 1) + ["assistant"]
assert [msg["role"] for msg in LINES] == ROLES_SEQ
REDO_TURNS = CRASH_TURN - 1
REDO_TOKENS = sum(SENT[:REDO_TURNS])
REDO_S = REDO_TURNS * ROUND_S
assert REDO_TURNS == 11 and REDO_TOKENS == 88_000 and REDO_S == 55  # "eleven finished turns, paid for again"

# ---------------------------------------------------------------------------- 5 · budgets

BUDGET_SRC = '''def agent_loop(messages, budget):
    while True:
        if budget.turns == 0 or budget.tokens <= 0:
            return stop(messages, "budget used up")
        response = client.messages.create(...)
        budget.turns -= 1
        budget.tokens -= response.usage.input_tokens
        ...'''


class StuckModel:
    """A task that never finishes (it keeps calling tools)."""

    def __init__(self):
        self.calls = 0

    def create(self, *args, **kw):
        self.calls += 1
        return NS(stop_reason="tool_use", usage=NS(input_tokens=PREFIX + PER_TURN * (self.calls - 1)))


MAX_TURNS, MAX_TOKENS = 30, 400_000  # illustrative budget
_stuck = StuckModel()
BUDGET_NS = {"client": NS(messages=_stuck), "stop": lambda messages, reason: reason}
exec(BUDGET_SRC, BUDGET_NS)
_budget = NS(turns=MAX_TURNS, tokens=MAX_TOKENS)
assert BUDGET_NS["agent_loop"]([], _budget) == "budget used up"
USED_TURNS, USED_TOKENS = MAX_TURNS - _budget.turns, MAX_TOKENS - _budget.tokens
assert USED_TURNS == _stuck.calls == 26 and USED_TOKENS == sum(sent(26)) == 403_000  # "stopped after 26 turns"

TEAM_SHARE = 1 / DEVELOPERS  # the limiter's fair share
assert TEAM_SHARE == 0.125

# ---------------------------------------------------------------------------- 6 · evals in CI (illustrative)

EVAL_TASKS = 100  # the tinyshop eval set (Ep 29)
V14 = NS(passed=73, turns=TURNS)
V15 = NS(passed=58, turns=16)  # "test only at the end": fewer rounds
for v in (V14, V15):
    v.cost = round(billed(v.turns))
    v.time = v.turns * ROUND_S
assert V14.cost == BILLED == 44_800 and V15.cost == 33_000 and V14.time == 100 and V15.time == 80
assert V15.cost < V14.cost and V15.time < V14.time  # "cheaper and faster"
assert V14.passed - V15.passed == 15 > 9  # "fifteen fewer tasks", beyond Ep 29's ±9-point noise for 100 tasks

# ---------------------------------------------------------------------------- look

SERVER_ICON = "ep16-server"
KEY_TEAM = recolored("ep16-key", AZURE, AZURE_FILL, prefix="ep32")
KEY_AGENT = recolored("ep16-key", SUN, SUN_FILL, prefix="ep32")
ALARM = recolored("ep23-alarm", SUN, SUN_FILL, prefix="ep32")
GEAR = recolored("gear", MUTED, CARD_HI, prefix="ep32")
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def sit(words, base):
    """Put a Text on a baseline (by its flat-bottomed glyphs)."""
    return words.shift((base - baseline(words, words.text.replace(" ", "").replace("\n", ""))) * m.UP)


def ep_chip(label, ep, width=None):
    """A link to another episode: a neutral card, a label, "ep N" in that episode's module colour."""
    accent = MODULES[EPISODES[ep][1]][1]
    body = m.VGroup(text(label, 22, INK, weight=m.SEMIBOLD), code(f"ep {ep}", 20, accent, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
    sit(body[1], baseline(body[0], label.replace(" ", "")))
    box = card(width or body.width + 0.6, 0.66, EDGE, CARD, radius=0.16, stroke=2)
    return m.VGroup(box, body.move_to(box))


def lines_of(words, src):
    """The glyphs of each line of a multi-line Text made from `src`."""
    out, k = [], 0
    for line in src.split("\n"):
        n = sum(not c.isspace() for c in line)
        out.append(words[k:k + n])
        k += n
    return out


def part_of(words, src, key):
    """The glyphs of the first `key` in a Text made from `src`."""
    start = src.index(key)
    skip = sum(c.isspace() for c in src[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip:start - skip + width]


class Slot:
    """One place on screen whose content is swapped (a counter, a label)."""

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


def ledger_row(label, value, size=22, color=INK, value_color=SUN, pad=11, width=9):
    s = f"{label:<{pad}}{value:>{width}}"
    return code(s, size, color, t2c={r"[\d][\d,.]*": value_color})


def q_card(tag, question, w=4.0, h=1.4):
    box = card(w, h, EDGE, CARD, radius=0.18, stroke=2)
    t = code(tag, 20, LIME, weight=m.BOLD)
    q = text(question, 24, INK, weight=m.SEMIBOLD)
    m.VGroup(t, q).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18).move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, t, q)


def block(color, w=0.2, h=0.46, stroke=None):
    return m.RoundedRectangle(corner_radius=0.05, width=w, height=h, stroke_color=stroke or color,
                              stroke_width=2 if stroke else 0, fill_color=color, fill_opacity=1)


def mini_trace(width=2.5, color=ROSE):
    box = card(width, 1.15, color, PANEL, radius=0.14, stroke=2)
    rows = []
    spec = [(0.0, 1.0, LIME), (0.0, 0.18, CORAL), (0.18, 0.12, SUN), (0.3, 0.45, CORAL), (0.75, 0.2, SUN)]
    left = box.get_left()[0] + 0.25
    span_w = width - 0.5
    for i, (a, d, c) in enumerate(spec):
        r = m.Rectangle(width=max(d * span_w, 0.05), height=0.11, stroke_width=0, fill_color=c, fill_opacity=0.9 if i else 0.45)
        r.move_to([left + (a + d / 2) * span_w, box.get_top()[1] - 0.22 - 0.18 * i, 0])
        rows.append(r)
    return m.VGroup(box, *rows)


# ---------------------------------------------------------------------------- the episode


class Ep32RunningInProduction(Episode):
    number = 32

    def lesson(self):
        self.hook()
        self.cost()
        self.latency()
        self.tracing()
        self.crash()
        self.limits()
        self.evals()
        self.big_loop()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Until now (Episodes 1–31) Otto worked for one developer on one laptop. Say it now serves tinyshop's whole team: {DEVELOPERS} developers × {TASKS_EACH} tasks = {TASKS_A_DAY} tasks a day, some overnight (all illustrative). Six questions a laptop never asked, one section each: cost, latency, tracing, crash and resume, budgets and rate limits, evals in CI. Production operations are one of the gaps neither reference covers fully (research notes §26), so the episode stays conservative and practical. Facts dated October 2026.")
        dev = icon("user", 1.1).move_to([-4.3, 0.85, 0])
        dev_lab = text("one developer", 24, MUTED).next_to(dev, m.DOWN, buff=0.3)
        screen = card(3.1, 2.1, EDGE, PANEL, radius=0.14, stroke=2.5).move_to([1.3, 0.95, 0])
        base = m.RoundedRectangle(corner_radius=0.06, width=3.9, height=0.16, stroke_width=0, fill_color=EDGE, fill_opacity=1)
        base.next_to(screen, m.DOWN, buff=0.06)
        bot = otto(0.8).move_to(screen.get_center() + 0.08 * m.DOWN).set_z_index(3)
        lap_lab = text("one laptop", 24, MUTED).next_to(base, m.DOWN, buff=0.3)
        link = arrow(dev.get_right() + 0.15 * m.RIGHT, screen.get_left(), MUTED, buff=0.15)
        laptop = m.VGroup(screen, base)

        # the team
        server = card(3.9, 2.9, EDGE, PANEL, radius=0.2, stroke=2.5).move_to([4.0, 0.35, 0])
        rack = icon(SERVER_ICON, 0.5).move_to(server.get_corner(m.UL) + [0.45, -0.42, 0])
        s_lab = code("otto · server", 20, MUTED).next_to(rack, m.RIGHT, buff=0.2)
        bot_at = server.get_center() + 0.25 * m.DOWN
        team = m.VGroup(*[icon("user", 0.6) for _ in range(DEVELOPERS)]).arrange_in_grid(rows=4, cols=2, buff=(0.42, 0.32))
        team.move_to([-5.55, 0.45, 0])
        team_lab = text(f"{DEVELOPERS} developers", 24, MUTED).next_to(team, m.DOWN, buff=0.3)
        names = ["fix the checkout test", "add a discount code", "update the README", "review a pull request", "bump a dependency"]
        queue = m.VGroup(*[pill(n, AZURE, AZURE_FILL, 20) for n in names]).arrange(m.DOWN, buff=0.16).move_to([-1.55, 0.75, 0])
        per_day = code(f"{TASKS_A_DAY} tasks a day", 22, AZURE, weight=m.BOLD).next_to(queue, m.DOWN, buff=0.3)
        feed = arrow(queue.get_right() + 0.1 * m.RIGHT, server.get_left(), MUTED, buff=0.12)
        moon = icon("ep12-moon", 0.45)
        night = m.VGroup(moon, text("some run overnight", 22, ORCHID, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2)
        night.next_to(server, m.DOWN, buff=0.3)
        stage = m.VGroup(team, team_lab, queue, per_day, feed, server, rack, s_lab, night)

        qs = [("01 · cost", "What does a task cost?"), ("02 · latency", "How long does it take?"),
              ("03 · tracing", "What happened inside?"), ("04 · resume", "What if it crashes?"),
              ("05 · limits", "What stops a runaway?"), ("06 · evals in CI", "Did a change hurt?")]
        cards = [q_card(t, q, 4.1, 1.4) for t, q in qs]
        for i, c in enumerate(cards):
            c.move_to([(i % 3 - 1) * 4.4, -0.3 - (i // 3) * 1.7, 0])

        self.say(
            "[Until now, Otto has worked for one developer], [on one laptop.]",
            m.LaggedStart(m.FadeIn(m.VGroup(dev, dev_lab), shift=0.2 * m.UP), m.FadeIn(bot, shift=0.2 * m.UP), lag_ratio=0.4),
            m.AnimationGroup(m.FadeIn(laptop), m.GrowArrow(link), m.FadeIn(lap_lab)),
        )
        self.say(
            "[Now say it serves tinyshop's whole team]: [eight developers], [forty tasks a day], [some of them running overnight.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(dev, dev_lab, link, laptop, lap_lab)),
                bot.animate.move_to(bot_at),
                m.FadeIn(m.VGroup(server, rack, s_lab)),
            ),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(u, scale=0.8) for u in team], lag_ratio=0.12), m.FadeIn(team_lab)),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(q, target_position=team.get_right()) for q in queue], lag_ratio=0.2),
                m.FadeIn(per_day, shift=0.1 * m.UP), m.GrowArrow(feed),
            ),
            m.FadeIn(night, shift=0.15 * m.UP),
        )
        self.say(
            "[That raises new questions.] [What does a task cost], [how long does it take], [and what happened inside it?]",
            m.AnimationGroup(m.FadeOut(stage), bot.animate.scale(0.75).move_to([0, 1.55, 0])),
            m.FadeIn(cards[0], shift=0.2 * m.UP),
            m.FadeIn(cards[1], shift=0.2 * m.UP),
            m.FadeIn(cards[2], shift=0.2 * m.UP),
        )
        self.say(
            "[What if the server crashes halfway], [what stops a runaway], [and did yesterday's change make Otto worse?]",
            m.FadeIn(cards[3], shift=0.2 * m.UP),
            m.FadeIn(cards[4], shift=0.2 * m.UP),
            m.FadeIn(cards[5], shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · cost

    def cost(self):
        self.next_section("1 · Cost per task", notes=f"Illustrative token counts: the prompt starts at {PREFIX:,} tokens (system prompt + tools + task) and each turn adds {PER_TURN:,} (the model's tool call + the tool's result). Call k sends {PREFIX:,} + {PER_TURN:,}·(k−1): {SENT[0]:,} … {SENT[-1]:,}. Without caching, {TURNS} calls send {TOTAL:,} input tokens for {NEW_TOTAL:,} new ones (Li ch. 7: re-sent history, '1000 + 2000 + 3000 = 6000 instead of 3000'; Ep 6). The sum grows with the square of the turns: 40 turns send {TOTAL_40:,} (×{TOTAL_40 / TOTAL}). With prompt caching each call's re-sent part ({READ_TOTAL:,} in all) is a cache read at about a tenth of the price (Li ch. 2; Ep 7): {NEW_TOTAL:,} + 0.1 × {READ_TOTAL:,} = {BILLED:,.0f} full-price tokens ({PCT}%). Counted on the input side only, cache writes at the normal price for simplicity. Multiply by your provider's current price; no vendor prices are quoted.")
        ox, oy = -5.75, -2.65
        top_h = 4.3
        sy = top_h / max(SENT)
        step, bw = 0.32, 0.24
        xs = [ox + 0.32 + step * k for k in range(TURNS)]
        x_axis = m.Line([ox, oy, 0], [xs[-1] + 0.32, oy, 0]).set_stroke(FAINT, 2.5)
        y_axis = m.Line([ox, oy, 0], [ox, oy + top_h + 0.3, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup(*[m.Line([ox - 0.07, oy + v * sy, 0], [ox + 0.07, oy + v * sy, 0]).set_stroke(FAINT, 2.5) for v in (10_000, 20_000)])
        y_labs = m.VGroup(*[code(f"{v // 1000}k", 20, MUTED).next_to([ox, oy + v * sy, 0], m.LEFT, buff=0.14) for v in (10_000, 20_000)])
        x_labs = m.VGroup(*[code(str(k + 1), 20, MUTED).move_to([xs[k], oy - 0.27, 0]) for k in (0, 9, 19)])
        x_title = code("call", 20, MUTED).move_to([(xs[0] + xs[-1]) / 2, oy - 0.62, 0])
        title = code("tokens sent per call", 20, MUTED).move_to([-6.55, 2.3, 0], aligned_edge=m.LEFT)
        chart = m.VGroup(x_axis, y_axis, ticks, y_labs, x_labs, x_title, title)
        bars, lowers, uppers = [], [], []
        for k in range(TURNS):
            parts = []
            if READ[k]:
                low = m.Rectangle(width=bw, height=READ[k] * sy - 0.02, stroke_width=0, fill_color=SUN, fill_opacity=0.42)
                low.move_to([xs[k], oy + (READ[k] * sy - 0.02) / 2, 0])
                lowers.append(low)
                parts.append(low)
            up = m.Rectangle(width=bw, height=NEW[k] * sy, stroke_width=0, fill_color=SUN, fill_opacity=1)
            up.move_to([xs[k], oy + READ[k] * sy + NEW[k] * sy / 2, 0])
            uppers.append(up)
            parts.append(up)
            bars.append(m.VGroup(*parts))
        bar_group = m.VGroup(*bars)
        lab_3k = code(f"{PREFIX:,}", 20, SUN).move_to([xs[0] - bw / 2, -0.55, 0], aligned_edge=m.LEFT)
        arrow_3k = arrow([xs[0], lab_3k.get_bottom()[1] - 0.05, 0], uppers[0].get_top(), SUN, buff=0.08)
        add_lab = code(f"+{PER_TURN:,} a turn", 20, SUN).move_to([xs[5], oy + SENT[5] * sy + 0.95, 0])
        add_arrow = arrow(add_lab.get_bottom() + 0.05 * m.DOWN, uppers[5].get_top(), SUN, buff=0.08)
        area = m.Polygon([xs[0] - bw / 2, oy, 0], [xs[-1] + bw / 2, oy, 0], [xs[-1] + bw / 2, oy + SENT[-1] * sy, 0],
                         [xs[0] - bw / 2, oy + SENT[0] * sy, 0]).set_stroke(ROSE, 3).set_fill(ROSE, opacity=0)
        chip_slot = Slot(self)
        chip6 = ep_chip("re-sent history", 6).move_to([0.9, 2.3, 0], aligned_edge=m.RIGHT)
        chip7 = ep_chip("prompt caching", 7).move_to([0.9, 2.3, 0], aligned_edge=m.RIGHT)

        # the ledger
        lbox = card(5.45, 5.65, EDGE, PANEL, radius=0.2, stroke=2).move_to([3.85, -0.45, 0])
        lx = lbox.get_left()[0] + 0.35
        l_title = code("ledger · one task, for example", 20, MUTED).move_to([lx, 1.95, 0], aligned_edge=m.LEFT)
        r_sent = ledger_row("sent", f"{TOTAL:,}").move_to([lx, 1.4, 0], aligned_edge=m.LEFT)
        r_new = ledger_row("new", f"{NEW_TOTAL:,}").move_to([lx, 0.95, 0], aligned_edge=m.LEFT)
        r_40 = ledger_row("40 turns", f"{TOTAL_40:,}").move_to([lx, 0.5, 0], aligned_edge=m.LEFT)
        x36 = code(f"×{TOTAL_40 / TOTAL:g}", 22, ROSE, weight=m.BOLD).next_to(r_40, m.RIGHT, buff=0.3)
        sit(x36, baseline(r_40, "40turns900,000"))
        sep = m.Line([lx, 0.12, 0], [lbox.get_right()[0] - 0.35, 0.12, 0]).set_stroke(EDGE, 2)
        with_cache = text("with caching", 22, MINT, weight=m.BOLD).move_to([lx, -0.25, 0], aligned_edge=m.LEFT)
        r_c1 = ledger_row("new", f"{NEW_TOTAL:,}").move_to([lx, -0.7, 0], aligned_edge=m.LEFT)
        r_c1x = code("× 1", 22, MUTED).next_to(r_c1, m.RIGHT, buff=0.3)
        r_c2 = ledger_row("cache read", f"{READ_TOTAL:,}", value_color=MINT).move_to([lx, -1.15, 0], aligned_edge=m.LEFT)
        r_c2x = code(f"× {CACHE_READ:g}", 22, MINT).next_to(r_c2, m.RIGHT, buff=0.3)
        sit(r_c1x, baseline(r_c1, "new22,000"))
        sit(r_c2x, baseline(r_c2, "cacheread228,000"))
        r_c1x.align_to(r_c2x, m.LEFT)
        r_bill = ledger_row("billed", f"{BILLED:,.0f}", size=24, value_color=MINT).move_to([lx, -1.72, 0], aligned_edge=m.LEFT)
        r_bill[:6].set_color(INK)
        pct = pill(f"{PCT}%", MINT, MINT_FILL, 20, mono=True).next_to(r_bill, m.RIGHT, buff=0.3)
        final = code("cost per task = Σ calls × price", 20, INK, t2c={r"cost per task": LIME, r"Σ calls": SUN, r"price": AZURE})
        final.move_to([lx, -2.4, 0], aligned_edge=m.LEFT)
        l_src = cite("Li, AI Agents in Depth · ch. 2, ch. 7").move_to([lx, -3.0, 0], aligned_edge=m.LEFT)

        self.say(
            "[First, cost.] [Take one task of twenty turns]: [every call re-sends the whole history], [as in episode six.]",
            self.new_header("01 · cost", "What a task costs"),
            m.FadeIn(chart),
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars], lag_ratio=0.12),
            chip_slot.to(chip6),
        )
        self.say(
            "[For example], [the prompt starts at three thousand tokens], [and each turn adds a thousand more.]",
            m.FadeIn(m.VGroup(lbox, l_title), shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(lab_3k, shift=0.1 * m.UP), m.GrowArrow(arrow_3k), m.Indicate(bars[0], color=INK, scale_factor=1.0)),
            m.AnimationGroup(m.FadeIn(add_lab), m.GrowArrow(add_arrow),
                             m.LaggedStart(*[m.Indicate(u, color=INK, scale_factor=1.0) for u in uppers[1:]], lag_ratio=0.08)),
        )
        self.say(
            "[Add them up]: [twenty calls send two hundred and fifty thousand input tokens], [though only twenty-two thousand are new.]",
            m.Circumscribe(bar_group, color=MUTED, buff=0.1),
            m.FadeIn(r_sent, shift=0.1 * m.UP),
            m.FadeIn(r_new, shift=0.1 * m.UP),
        )
        self.say(
            "[The total grows with the square of the turns]: [forty turns would send nine hundred thousand], [three point six times as much.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(add_lab, add_arrow)), m.FadeIn(area)),
            m.FadeIn(r_40, shift=0.1 * m.UP),
            m.FadeIn(x36, scale=1.3),
        )
        self.say(
            "[Caching changes the bill.] [Everything a call re-sends is read from the cache], [at about a tenth of the price], [as episode seven showed.]",
            m.AnimationGroup(m.FadeOut(area), m.Create(sep), m.FadeIn(with_cache, shift=0.1 * m.UP)),
            m.LaggedStart(*[low.animate.set_fill(MINT, opacity=0.85) for low in lowers], lag_ratio=0.08),
            m.AnimationGroup(m.FadeIn(r_c2, shift=0.1 * m.UP), m.FadeIn(r_c2x, shift=0.1 * m.UP)),
            chip_slot.to(chip7),
        )
        self.say(
            "[New tokens at full price, repeated ones at a tenth]: [that's forty-four thousand eight hundred tokens' worth], [under a fifth of what it cost before.]",
            m.AnimationGroup(m.FadeIn(r_c1, shift=0.1 * m.UP), m.FadeIn(r_c1x, shift=0.1 * m.UP)),
            m.FadeIn(r_bill, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(pct, shift=0.1 * m.UP), m.FadeIn(l_src)),
        )
        self.say(
            "[So measure cost per task, not per call]: [add up every call the task made], [then multiply by your provider's price.]",
            m.FadeIn(final, shift=0.1 * m.UP),
            m.Circumscribe(bar_group, color=SUN, buff=0.1),
            m.Circumscribe(part_of(final, final.text, "price"), color=AZURE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · latency

    def latency(self):
        self.next_section("2 · Latency budget", notes=f"Illustrative: a {BUDGET_S} s budget for the {ROUNDS}-round task; each round {MODEL_S} s in the model and {TOOL_S} s in tools = {BASE_S} s. Li ch. 1 (how to choose a model): 'a 20-round Agent task that runs 2 seconds slower per round means an extra 40 seconds of waiting' → {LATE_S} s, over budget. Li ch. 7: 'at 50 tokens/s, a model producing 2000 thinking tokens spends 40 seconds just thinking'. Ways out: fewer rounds, tools called in parallel (Ep 3), less thinking on easy steps; caching also shortens time to first token (Ep 7).")
        x0, sx, y = -6.15, 0.082, 0.85
        bh = 0.52

        def seg(t0, dur, color, opacity=1.0, yy=y):
            return m.Rectangle(width=dur * sx, height=bh, stroke_color=BG, stroke_width=1.5, fill_color=color,
                               fill_opacity=opacity).move_to([x0 + (t0 + dur / 2) * sx, yy, 0])

        model_segs, tool_segs, extras, late_m, late_t = [], [], [], [], []
        for r in range(ROUNDS):
            t = r * ROUND_S
            model_segs.append(seg(t, MODEL_S, CORAL))
            tool_segs.append(seg(t + MODEL_S, TOOL_S, SUN))
            t2 = r * (ROUND_S + SLOWER_S)
            late_m.append(seg(t2, MODEL_S, CORAL).get_center())
            extras.append(seg(t2 + MODEL_S, SLOWER_S, ROSE))
            late_t.append(seg(t2 + MODEL_S + SLOWER_S, TOOL_S, SUN).get_center())
        axis_y = y - bh / 2 - 0.22
        axis = m.Line([x0, axis_y, 0], [x0 + 142 * sx, axis_y, 0]).set_stroke(FAINT, 2.5)
        tick_vals = list(range(0, 141, 20))
        tks = m.VGroup(*[m.Line([x0 + v * sx, axis_y - 0.07, 0], [x0 + v * sx, axis_y + 0.07, 0]).set_stroke(FAINT, 2.5) for v in tick_vals])
        tlabs = m.VGroup(*[code(f"{v} s" if v % 40 == 0 else str(v), 20, MUTED).move_to([x0 + v * sx, axis_y - 0.3, 0]) for v in tick_vals])
        bx = x0 + BUDGET_S * sx
        budget_line = m.DashedLine([bx, y + 0.62, 0], [bx, axis_y, 0], dash_length=0.1).set_stroke(SUN, 3)
        budget_lab = text("budget · 2 min", 22, SUN, weight=m.BOLD).next_to([bx, y + 0.62, 0], m.UP, buff=0.12)
        total_slot = Slot(self)
        t100 = code(f"{BASE_S} s", 22, INK, weight=m.BOLD).next_to([x0 + BASE_S * sx, y, 0], m.RIGHT, buff=0.18)
        t140 = code(f"{LATE_S} s", 22, ROSE, weight=m.BOLD).next_to([x0 + LATE_S * sx, y, 0], m.RIGHT, buff=0.18)
        over = m.Rectangle(width=(LATE_S - BUDGET_S) * sx, height=bh + 0.22, stroke_color=ROSE, stroke_width=3, fill_opacity=0)
        over.move_to([x0 + (BUDGET_S + LATE_S) / 2 * sx, y, 0])
        waiting = m.VGroup(icon("user", 0.55), text("a developer, waiting", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        waiting.move_to([x0, 2.25, 0], aligned_edge=m.LEFT)

        def key(color, label):
            sq = m.RoundedRectangle(corner_radius=0.04, width=0.26, height=0.26, stroke_width=0, fill_color=color, fill_opacity=1)
            return m.VGroup(sq, text(label, 22, INK)).arrange(m.RIGHT, buff=0.15)

        legend = m.VGroup(key(CORAL, f"model {MODEL_S} s"), key(SUN, f"tools {TOOL_S} s"), key(ROSE, f"+{SLOWER_S} s slower")).arrange(m.RIGHT, buff=0.45)
        legend.move_to([6.55, 2.25, 0], aligned_edge=m.RIGHT)
        src1 = cite("Li, AI Agents in Depth · ch. 1").move_to([6.55, axis_y - 0.72, 0], aligned_edge=m.RIGHT)
        # thinking
        ty = -1.45
        think_lab = text("thinking", 22, MUTED).move_to([x0, ty + 0.55, 0], aligned_edge=m.LEFT)
        think = seg(0, THINK_S, CORAL, 0.5, ty)
        formula = code(f"{THINK_TOKENS:,} tokens ÷ {TOK_PER_S} tokens/s = {THINK_S:g} s", 22, INK, t2c={r"\d[\d,]*": SUN})
        formula.next_to(think, m.RIGHT, buff=0.35)
        src2 = cite("Li, AI Agents in Depth · ch. 7").next_to(formula, m.DOWN, buff=0.16).align_to(formula, m.LEFT)
        fixes = m.VGroup(*[pill(s, MINT, MINT_FILL, 24) for s in ("fewer rounds", "tools in parallel", "less thinking on easy steps")])
        fixes.arrange(m.RIGHT, buff=0.4).move_to([0, -2.95, 0])
        timeline = m.VGroup(*model_segs, *tool_segs)

        self.say(
            "[Next, time.] [The developer waiting on Otto has a latency budget]: [say, two minutes for this task.]",
            self.new_header("02 · latency", "A latency budget"),
            m.AnimationGroup(m.FadeIn(waiting, shift=0.15 * m.UP), m.FadeIn(m.VGroup(axis, tks, tlabs))),
            m.AnimationGroup(m.Create(budget_line), m.FadeIn(budget_lab, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[For example, each round spends four seconds in the model and one in tools.] [Twenty rounds take a hundred seconds.]",
            m.FadeIn(legend[:2], shift=0.1 * m.UP),
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.GrowFromEdge(a, m.LEFT), m.GrowFromEdge(b, m.LEFT)) for a, b in zip(model_segs, tool_segs)], lag_ratio=0.15),
                total_slot.to(t100),
            ),
        )
        self.say(
            "[Now make each round two seconds slower.] [Over twenty rounds, that's forty seconds more], [and the budget is blown.]",
            m.FadeIn(legend[2], shift=0.1 * m.UP),
            m.AnimationGroup(
                *[s.animate.move_to(p) for s, p in zip(model_segs, late_m)],
                *[s.animate.move_to(p) for s, p in zip(tool_segs, late_t)],
                m.LaggedStart(*[m.FadeIn(e, scale=0.6) for e in extras], lag_ratio=0.05),
                total_slot.to(t140), m.FadeIn(src1),
            ),
            m.AnimationGroup(m.Create(over), m.Wiggle(budget_lab)),
        )
        self.say(
            "[Thinking takes time too]: [at fifty tokens a second], [two thousand thinking tokens take forty seconds.]",
            m.AnimationGroup(m.FadeIn(think_lab), m.GrowFromEdge(think, m.LEFT)),
            m.FadeIn(formula, shift=0.1 * m.LEFT),
            m.AnimationGroup(m.Circumscribe(part_of(formula, formula.text, f"= {THINK_S:g} s"), color=CORAL, buff=0.06), m.FadeIn(src2)),
        )
        self.say(
            "[So budget rounds as well as tokens]: [fewer rounds], [tools called in parallel], [and less thinking on easy steps.]",
            m.Circumscribe(m.VGroup(timeline, *extras), color=MUTED, buff=0.12),
            m.FadeIn(fixes[0], shift=0.15 * m.UP),
            m.FadeIn(fixes[1], shift=0.15 * m.UP),
            m.FadeIn(fixes[2], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · tracing

    def tracing(self):
        self.next_section("3 · Traces and spans", notes=f"Li ch. 7, Agent Observability: one trace per task; 'each LLM call, each tool call, and each retrieval is a span'; 'OpenTelemetry is the general-purpose distributed tracing standard, while specifications like OpenInference define LLM-specific semantic conventions on top of it'; prompt version management. The waterfall comes from a traced run of the reference loop (stand-in tracer with OpenTelemetry's method names, scripted model, simulated clock; durations illustrative: model calls = {TTFT_S} s to first token + output at {TOK_PER_S} tokens/s, tests {TOOL_TIME['pytest -q']} s): spans {list(zip(ROW_NAMES, DUR))}, task {ROOT.end} s. The slowest span is call 3, where Otto plans the edit ({SLOW_SPAN.attributes['llm.token_count.completion']} output tokens → {DUR[SLOW]} s). Span attributes are OpenInference names. Version labels (prompt v14, tools v6, harness 2.3.0) are illustrative; the model name is the reference harness's.")
        rows_y = [2.05 - 0.45 * i for i in range(10)]
        label_x, x0, sx = -6.55, -3.5, 0.17
        bh = 0.28

        def bar(t0, t1, color, yy, opacity=1.0):
            return m.Rectangle(width=max((t1 - t0) * sx, 0.05), height=bh, stroke_width=0, fill_color=color,
                               fill_opacity=opacity).move_to([x0 + (t0 + t1) / 2 * sx, yy, 0])

        root_lab = code("task · fix checkout", 20, LIME, weight=m.BOLD).move_to([label_x, rows_y[0], 0], aligned_edge=m.LEFT)
        root_bar = bar(ROOT.start, ROOT.end, LIME, rows_y[0], 0.45).set_stroke(LIME, 2)
        root_t = code(f"{ROOT.end:g} s", 20, LIME).next_to(root_bar, m.UP, buff=0.1).align_to(root_bar, m.RIGHT)
        labs, bars = [], []
        for i, (s, name) in enumerate(zip(CHILDREN, ROW_NAMES)):
            color = CORAL if s.name == "llm" else SUN
            labs.append(code(name, 20, color).move_to([label_x + 0.3, rows_y[i + 1], 0], aligned_edge=m.LEFT))
            bars.append(bar(s.start, s.end, color, rows_y[i + 1]))
        axis_y = rows_y[-1] - 0.38
        axis = m.Line([x0, axis_y, 0], [x0 + 24 * sx, axis_y, 0]).set_stroke(FAINT, 2.5)
        tks = m.VGroup(*[m.Line([x0 + v * sx, axis_y - 0.07, 0], [x0 + v * sx, axis_y + 0.07, 0]).set_stroke(FAINT, 2.5) for v in range(0, 21, 5)])
        tlabs = m.VGroup(*[code(f"{v} s", 20, MUTED).move_to([x0 + v * sx, axis_y - 0.3, 0]) for v in range(0, 21, 5)])
        clock_group = m.VGroup(axis, tks, tlabs)
        sweep = m.Line([x0, axis_y, 0], [x0 + ROOT.end * sx, axis_y, 0]).set_stroke(INK, 7)  # the clock running
        slow_t = code(f"{DUR[SLOW]:g} s", 20, CORAL, weight=m.BOLD).next_to(bars[SLOW], m.RIGHT, buff=0.15)
        band_w = 6.8
        band = m.Rectangle(width=band_w, height=0.42, stroke_width=0, fill_color=CORAL, fill_opacity=0.14)
        band.move_to([label_x - 0.1 + band_w / 2, rows_y[SLOW + 1], 0])
        root_band = m.Rectangle(width=band_w, height=0.42, stroke_width=0, fill_color=LIME, fill_opacity=0.12)
        root_band.move_to([label_x - 0.1 + band_w / 2, rows_y[0], 0])

        # right column
        rx = 6.6
        kinds = m.VGroup(*[pill(s, c, f, 22) for s, c, f in (("model call", CORAL, CORAL_FILL), ("tool call", SUN, SUN_FILL), ("retrieval", ORCHID, ORCHID_FILL))])
        kinds.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        kinds_title = text("a span is one…", 22, MUTED)
        kinds_box = m.VGroup(kinds_title, kinds).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to([1.6, 1.25, 0], aligned_edge=m.LEFT)
        span_card = json_card(SPAN_ATTRS, size=17, accent=CORAL, title=f"span · llm · call 3 · {DUR[SLOW]:g} s")
        span_card.move_to([rx, 2.35, 0], aligned_edge=m.UR)
        span_src = json.dumps(SPAN_ATTRS, indent=2)
        sc_lines = lines_of(span_card[1], span_src)
        cap = span_card[2]
        versions = json_card(VERSIONS, size=19, accent=LIME, title="trace · versions")
        versions.move_to([rx, 2.35, 0], aligned_edge=m.UR)
        v_lines = lines_of(versions[1], json.dumps(VERSIONS, indent=2))
        otel = pill("OpenTelemetry · traces and spans", INK, CARD_HI, 21)
        oinf = pill("OpenInference · LLM and tool fields", INK, CARD_HI, 21)
        std = m.VGroup(otel, oinf).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        std.move_to([rx, -1.15, 0], aligned_edge=m.RIGHT)
        std_src = cite("Li, AI Agents in Depth · ch. 7").next_to(std, m.DOWN, buff=0.18).align_to(std, m.LEFT)

        self.say(
            "[To see where the money and time went, record a trace.] [A trace is one task.] [A span is one model call, tool call, or retrieval.]",
            self.new_header("03 · tracing", "Traces and spans"),
            m.AnimationGroup(m.FadeIn(root_lab, shift=0.1 * m.RIGHT), m.GrowFromEdge(root_bar, m.LEFT), m.FadeIn(root_t)),
            m.LaggedStart(m.FadeIn(kinds_title), *[m.FadeIn(k, shift=0.1 * m.UP) for k in kinds], lag_ratio=0.3),
        )
        self.say(
            "[Here is Otto's checkout fix as a waterfall], [every span placed where it ran on the clock.]",
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(l, shift=0.1 * m.RIGHT), m.GrowFromEdge(b, m.LEFT)) for l, b in zip(labs, bars)], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(clock_group), m.ShowPassingFlash(sweep, time_width=0.5, run_time=1.8)),
        )
        self.say(
            "[Each span carries its numbers]: [the model, tokens in and out], [and how long it took.]",
            m.AnimationGroup(m.FadeOut(kinds_box), m.FadeIn(band), m.FadeIn(span_card, shift=0.15 * m.LEFT)),
            m.Circumscribe(m.VGroup(*sc_lines[2], *sc_lines[3], *sc_lines[4]), color=CORAL, buff=0.06),
            m.Circumscribe(part_of(cap, cap.text, f"{DUR[SLOW]:g} s"), color=CORAL, buff=0.06),
        )
        self.say(
            "[The slowest span stands out at once]: [the model call where Otto planned the fix], [about nine seconds of output.]",
            m.Circumscribe(bars[SLOW], color=ROSE, buff=0.08),
            m.AnimationGroup(m.Indicate(labs[SLOW], color=INK, scale_factor=1.1), m.FadeIn(slow_t, shift=0.1 * m.LEFT)),
            m.Circumscribe(sc_lines[4], color=ROSE, buff=0.06),
        )
        self.say(
            "[The common standard is OpenTelemetry], [and OpenInference adds the fields for model calls.]",
            m.FadeIn(otel, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(oinf, shift=0.15 * m.UP), m.FadeIn(std_src)),
        )
        self.say(
            "[Every trace also records versions]: [model, prompt, tools, and harness], so a change in behaviour points back to a change you made.",
            m.AnimationGroup(m.FadeOut(m.VGroup(span_card, band)), m.FadeIn(root_band), m.FadeIn(versions, shift=0.15 * m.LEFT)),
            m.LaggedStart(*[m.Circumscribe(v_lines[i], color=LIME, buff=0.05) for i in range(1, 5)], lag_ratio=0.35),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · crash and resume

    def crash(self):
        self.next_section("4 · A crash, resumed", notes=f"Neither reference covers checkpoint and resume after a crash (research notes §26), so this shows the practical pattern that an append-only history makes possible (Li ch. 1, design patterns: append-only 'buys … cacheability, replayability and auditability'). The scene runs it: record() appends each message to {LOG_PATH} and flushes; the server restarts during turn {CRASH_TURN}'s tool call (illustrative), leaving {LOG_STATE.lines_at_crash} lines on disk; resume() rebuilds the messages from the log and the loop carries on; a placeholder result marks the interrupted call ({INTERRUPTED!r}, is_error); the run ends with {len(LINES)} messages after {TURNS} model calls in all, none repeated. Starting over instead would redo {REDO_TURNS} turns: {REDO_TOKENS:,} input tokens and about {REDO_S} s (the cost and latency examples above). Side effects must be safe to repeat (idempotency, Ep 20). Li ch. 5: a stalled stream is 'like a connected pipe that yields no water' — 'a production Agent needs an independent idle watchdog'; 'a trajectory should not be stored in any single vendor's wire format, but kept in a neutral one'; in production, patch a missing tool_result with a placeholder.")
        bx0, bstep, bw, bh = -5.2, 0.275, 0.2, 0.46
        mem_y, disk_y = 1.92, 0.7
        colors = [AZURE if r == "user" and i == 0 else (CORAL if r == "assistant" else MINT) for i, r in enumerate(ROLES_SEQ)]

        def blk(i, yy):
            stroke = ROSE if i == PLACEHOLDER else None
            b = block(colors[i], bw, bh, stroke)
            if i == PLACEHOLDER:
                b.set_fill(ROSE_FILL)
            return b.move_to([bx0 + bstep * i, yy, 0])

        n0 = LOG_STATE.lines_at_crash
        mem = [blk(i, mem_y) for i in range(len(LINES))]
        disk = [blk(i, disk_y) for i in range(len(LINES))]
        mem2 = [blk(i, mem_y) for i in range(n0)]  # worker 2's rebuilt messages
        track_w = bstep * (len(LINES) - 1) + bw + 0.28
        track_left = bx0 - bw / 2 - 0.14
        tracks = [m.RoundedRectangle(corner_radius=0.1, width=track_w, height=bh + 0.24, stroke_color=EDGE, stroke_width=2,
                                     fill_color=PANEL, fill_opacity=1).move_to([track_left + track_w / 2, yy, 0]) for yy in (mem_y, disk_y)]
        ram = icon("ep14-ram", 0.28).next_to(tracks[0], m.LEFT, buff=0.22)
        hdd = icon("ep14-disk", 0.55).next_to(tracks[1], m.LEFT, buff=0.22)
        lab_slot = Slot(self)
        w1 = text("worker 1 · messages in memory", 20, MUTED).next_to(tracks[0], m.UP, buff=0.1).align_to(tracks[0], m.LEFT)
        w2 = text("worker 2 · messages in memory", 20, MINT).move_to(w1, aligned_edge=m.LEFT)
        d_lab = code(f"{LOG_PATH} · on disk", 20, MUTED).next_to(tracks[1], m.UP, buff=0.1).align_to(tracks[1], m.LEFT)
        counter = Slot(self)

        def turn_lab(s, color=INK):
            return code(s, 22, color, weight=m.BOLD).move_to([6.55, 2.52, 0], aligned_edge=m.RIGHT)

        restart = pill("server restart", ROSE, ROSE_FILL, 22).move_to([4.6, mem_y, 0])
        dashed = m.RoundedRectangle(corner_radius=0.05, width=bw, height=bh, stroke_color=ROSE, stroke_width=2.5, fill_opacity=0)
        dashed.move_to([bx0 + bstep * PLACEHOLDER, mem_y, 0])
        dashed_d = dashed.copy().move_to([bx0 + bstep * PLACEHOLDER, disk_y, 0])
        no_res = text("no tool_result", 22, ROSE, weight=m.SEMIBOLD).next_to(dashed_d, m.RIGHT, buff=0.3)

        # bottom left: the code; bottom right: the ledger, then the placeholder
        rec = code_block(RECORD_SRC, size=17, title="record and resume")
        rec.move_to([-6.6, 0.12, 0], aligned_edge=m.UL)
        rec_src = cite("Li, AI Agents in Depth · ch. 1: append-only → replayable").next_to(rec, m.DOWN, buff=0.12).align_to(rec, m.LEFT)
        lg_box = card(5.6, 2.55, ROSE, PANEL, radius=0.18, stroke=2).move_to([3.8, -1.45, 0])
        lg_title = text("without a checkpoint: start over", 22, ROSE, weight=m.BOLD).move_to(lg_box.get_corner(m.UL) + [0.3, -0.4, 0], aligned_edge=m.LEFT)
        lg_rows = m.VGroup(
            code(f"{REDO_TURNS} turns again", 22, INK, t2c={r"\d+": SUN}),
            code(f"{REDO_TOKENS:,} input tokens", 22, INK, t2c={r"[\d,]+": SUN}),
            code(f"about {REDO_S} s more waiting", 22, INK, t2c={r"\d+": SUN}),
        ).arrange(m.DOWN, buff=0.17, aligned_edge=m.LEFT).next_to(lg_title, m.DOWN, buff=0.25).align_to(lg_title, m.LEFT)
        ledger = m.VGroup(lg_box, lg_title, lg_rows)
        ph = tool_result(INTERRUPTED, size=18, error=True)
        ph.move_to([6.6, 0.12, 0], aligned_edge=m.UR)
        safe = ep_chip("safe to repeat", 20).next_to(ph, m.DOWN, buff=0.25).align_to(ph, m.RIGHT)

        # the guards
        def guard(title, desc, mark, src, w=6.3, h=2.75, color=SUN):
            box = card(w, h, color, PANEL, radius=0.18, stroke=2)
            head = m.VGroup(mark, title_text(title, 26, INK)).arrange(m.RIGHT, buff=0.25)
            head.move_to(box.get_corner(m.UL) + [0.3, -0.5, 0], aligned_edge=m.LEFT)
            body = text(desc, 22, MUTED).next_to(head, m.DOWN, buff=0.25).align_to(head, m.LEFT)
            s = cite(src).move_to(box.get_corner(m.DL) + [0.3, 0.35, 0], aligned_edge=m.LEFT)
            return m.VGroup(box, head, body, s)

        g1 = guard("Idle watchdog", "no new tokens for too long:\ncancel the stream, retry", icon(ALARM, 0.6),
                   "Li · ch. 5: “a connected pipe that yields no water”")
        g2 = guard("Neutral format", "the log is not tied to one vendor:\nresume on another provider", icon("doc", 0.6),
                   "Li · ch. 5: handover to another model", color=MINT)
        g1.move_to([-3.3, -1.6, 0])
        g2.move_to([3.3, -1.6, 0])
        gap_x = (bx0 + bstep * PLACEHOLDER)

        self.say(
            "[Now, a crash.] [As Otto works], [the harness appends every message to a log file on disk.]",
            m.AnimationGroup(self.new_header("04 · resume", "A crash, resumed"), m.FadeIn(m.VGroup(*tracks, ram, hdd, d_lab))),
            m.AnimationGroup(lab_slot.to(w1), m.LaggedStart(*[m.FadeIn(b, scale=0.6) for b in mem[:n0]], lag_ratio=0.12),
                             counter.to(turn_lab(f"turn {CRASH_TURN} / {TURNS}"))),
            m.LaggedStart(*[m.FadeIn(d, target_position=b) for d, b in zip(disk[:n0], mem[:n0])], lag_ratio=0.12),
        )
        self.say(
            "[At turn twelve of twenty, the server restarts], [and everything in memory is gone.]",
            m.AnimationGroup(m.FadeIn(restart, scale=1.2), m.Circumscribe(tracks[0], color=ROSE, buff=0.05)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeOut(b, shift=0.3 * m.DOWN, scale=0.5) for b in mem[:n0]], lag_ratio=0.04),
                             lab_slot.to(text("worker 1 · gone", 20, ROSE).move_to(w1, aligned_edge=m.LEFT))),
        )
        self.say(
            "[Without a checkpoint, the task starts over]: [eleven finished turns, paid for again.]",
            m.FadeIn(m.VGroup(lg_box, lg_title), shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in lg_rows], lag_ratio=0.3),
        )
        self.say(
            "[With the log, a fresh worker reads it back], [rebuilds the messages], [and picks up at turn twelve.]",
            m.AnimationGroup(m.FadeOut(ledger), m.FadeOut(restart), lab_slot.to(w2),
                             m.Circumscribe(tracks[1], color=MINT, buff=0.05)),
            m.LaggedStart(*[m.FadeIn(b, target_position=d) for b, d in zip(mem2, disk[:n0])], lag_ratio=0.06),
            counter.to(turn_lab(f"turn {CRASH_TURN} / {TURNS} · resumed", MINT)),
        )
        self.say(
            "[This works because the history is append-only]: [nothing is rewritten, so the log is the checkpoint.]",
            m.AnimationGroup(m.FadeIn(rec, shift=0.15 * m.UP), m.FadeIn(rec_src)),
            m.Circumscribe(m.VGroup(*disk[:n0]), color=MINT, buff=0.1),
        )
        self.say(
            "[One catch]: [turn twelve's tool call never got its result.]",
            m.Circumscribe(m.VGroup(mem2[-1], disk[n0 - 1]), color=SUN, buff=0.08),
            m.AnimationGroup(m.FadeIn(dashed), m.FadeIn(dashed_d), m.FadeIn(no_res, shift=0.1 * m.UP)),
        )
        rest_mem = mem[n0 + 1:]
        rest_disk = disk[n0 + 1:]
        self.say(
            "[The harness adds one saying it was interrupted], [so Otto checks before repeating it], [then carries on to the end.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(dashed, dashed_d, no_res)), m.FadeIn(mem[n0]), m.FadeIn(disk[n0], target_position=[gap_x, mem_y, 0]),
                             m.FadeIn(ph, shift=0.15 * m.LEFT)),
            m.FadeIn(safe, shift=0.1 * m.UP),
            m.AnimationGroup(
                m.LaggedStart(*[m.AnimationGroup(m.FadeIn(a, scale=0.6), m.FadeIn(b, target_position=a)) for a, b in zip(rest_mem, rest_disk)], lag_ratio=0.15),
                counter.to(turn_lab(f"turn {TURNS} / {TURNS} · done", MINT)),
            ),
        )
        self.say(
            "[Two more guards]: [a watchdog cancels a stream that stops sending tokens], [and a log in a neutral format can even resume on another provider.]",
            m.FadeOut(m.VGroup(rec, rec_src, ph, safe), shift=0.2 * m.DOWN),
            m.FadeIn(g1, shift=0.15 * m.UP),
            m.FadeIn(g2, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · limits

    def limits(self):
        self.next_section("5 · Budgets and rate limits", notes=f"Per-task budget (illustrative {MAX_TURNS} turns, {MAX_TOKENS:,} input tokens): the check on screen runs against a model that never finishes and stops it after {USED_TURNS} turns ({USED_TOKENS:,} tokens sent). Li ch. 10: 'set a step or token budget when starting a sub-agent, and stop once it is exceeded'; 'Explicit budgets, cancellation and stop conditions are required to keep execution bounded'; 'resource quotas, and rate limits'. A provider's rate limit (requests and tokens per minute) is shared by every task on the same key; a 429 is Ep 20's rate-limit error. The shares and the spawn tree are illustrative. Li ch. 10, Runaway Loops: 'A runaway Agent may spawn thousands of sub-agents'; 'For highly autonomous Agents, use a dedicated API key as part of controlling token spending'.")
        bud = code_block(BUDGET_SRC, size=18, title="a budget per task")
        bud.move_to([-6.6, 2.45, 0], aligned_edge=m.UL)
        b_lines = lines_of(bud[1], BUDGET_SRC)
        stuck = m.VGroup(otto(0.6), text("a task that\nnever finishes", 22, MUTED)).arrange(m.RIGHT, buff=0.3)
        stuck.move_to([bud.get_right()[0] + 0.6, 1.55, 0], aligned_edge=m.LEFT)
        mw, mleft = 5.8, -6.6
        m_turns = Meter(mw, 0.0, SUN, 0.24).move_to([mleft, -1.65, 0], aligned_edge=m.LEFT)
        m_tokens = Meter(mw, 0.0, SUN, 0.24).move_to([mleft, -2.85, 0], aligned_edge=m.LEFT)
        lt = text("turns", 22, INK).next_to(m_turns, m.UP, buff=0.15).align_to(m_turns, m.LEFT)
        ltk = text("input tokens", 22, INK).next_to(m_tokens, m.UP, buff=0.15).align_to(m_tokens, m.LEFT)
        vt = code(f"{USED_TURNS} / {MAX_TURNS}", 20, MUTED).next_to(m_turns, m.UP, buff=0.15).align_to(m_turns, m.RIGHT)
        vtk = code(f"{USED_TOKENS:,} / {MAX_TOKENS:,}", 20, ROSE).next_to(m_tokens, m.UP, buff=0.15).align_to(m_tokens, m.RIGHT)
        sit(vt, baseline(lt, "turns"))
        sit(vtk, baseline(ltk, "inputtokens"))
        stopped = pill(f"stopped after {USED_TURNS} turns: budget used up", ROSE, ROSE_FILL, 22).move_to([0.55, -2.05, 0], aligned_edge=m.LEFT)
        b_src = cite("Li, AI Agents in Depth · ch. 10").next_to(stopped, m.DOWN, buff=0.2).align_to(stopped, m.LEFT)
        screen1 = m.VGroup(bud, stuck, m_turns, m_tokens, lt, ltk, vt, vtk, stopped, b_src)

        # screen 2: a shared rate limit
        ys = [2.0 - 0.6 * i for i in range(DEVELOPERS)]
        busy = 2
        heads = m.VGroup(*[(otto(0.3, SUN, SUN_FILL) if i == busy else otto(0.3)).move_to([-6.2, yy, 0]) for i, yy in enumerate(ys)])
        heads_lab = text("8 tasks", 22, MUTED).next_to(heads, m.UP, buff=0.15)
        busy_lab = text("a busy refactor", 20, SUN).next_to(heads[busy], m.RIGHT, buff=0.2)
        api = card(4.6, 2.3, CORAL, CORAL_FILL, radius=0.2, stroke=2.5).move_to([4.15, -0.1, 0])
        api_t = text("model API · rate limit", 24, INK, weight=m.BOLD).move_to(api.get_top() + 0.42 * m.DOWN)
        api_s = text("tokens per minute, shared", 20, MUTED).next_to(api_t, m.DOWN, buff=0.1)
        share_w, share_y = 3.9, api.get_y() - 0.45

        def shares(fractions):
            segs, x = [], api.get_x() - share_w / 2
            for i, f in enumerate(fractions):
                w = share_w * f
                segs.append(m.Rectangle(width=w, height=0.42, stroke_color=CORAL_FILL, stroke_width=2,
                                        fill_color=SUN if i == busy else AZURE, fill_opacity=1).move_to([x + w / 2, share_y, 0]))
                x += w
            return segs

        even = [1 / DEVELOPERS] * DEVELOPERS
        greedy = [0.04] * DEVELOPERS
        greedy[busy] = 1 - 0.04 * (DEVELOPERS - 1)
        segs = shares(even)
        segs_greedy = shares(greedy)
        segs_fair = shares(even)
        api_in = api.get_left() + 0.1 * m.RIGHT
        starts = [(busy_lab if i == busy else h).get_right() + 0.12 * m.RIGHT for i, h in enumerate(heads)]
        direct = m.VGroup(*[m.Line(a, api_in).set_stroke(EDGE, 2) for a in starts])
        r429 = m.VGroup(*[pill("429", ROSE, ROSE_FILL, 20, mono=True).next_to(heads[i], m.RIGHT, buff=0.2) for i in range(DEVELOPERS) if i != busy])
        lim = card(2.0, 4.9, MINT, PANEL, radius=0.2, stroke=2.5).move_to([-2.3, -0.1, 0])
        lim_t = m.VGroup(icon(GEAR, 0.55), text("limiter", 22, MINT, weight=m.BOLD)).arrange(m.DOWN, buff=0.15).move_to(lim)
        lim_s = text("1/8 each", 20, MUTED).next_to(lim_t, m.DOWN, buff=0.15)
        to_lim = m.VGroup(*[m.Line(a, [lim.get_left()[0], a[1], 0]).set_stroke(EDGE, 2) for a in starts])
        lim_out = arrow(lim.get_right(), api.get_left(), MINT, buff=0.08)
        chip20 = ep_chip("rate limits", 20).move_to([4.15, -2.6, 0])

        # screen 3: a runaway
        root = otto(0.42).move_to([-3.6, 2.0, 0])
        root_lab = text("nightly agent", 20, MUTED).next_to(root, m.RIGHT, buff=0.2)
        gens, edges = [], []
        prev = [root.get_bottom()]
        for g, (n, yy) in enumerate(((2, 0.95), (4, 0.0), (8, -0.95), (16, -1.9))):
            spacing = 5.4 / n
            pts = [np.array([-3.6 + (i - (n - 1) / 2) * spacing, yy, 0]) for i in range(n)]
            dots = m.VGroup(*[m.Dot(p, radius=0.1 if g < 3 else 0.085, color=CORAL) for p in pts])
            lines = m.VGroup(*[m.Line(prev[i // 2] + 0.05 * m.DOWN, p + 0.1 * m.UP).set_stroke(EDGE, 1.8) for i, p in enumerate(pts)])
            gens.append(dots)
            edges.append(lines)
            prev = pts
        quote = text("“A runaway Agent may spawn\nthousands of sub-agents.”", 22, INK)
        q_src = cite("Li, AI Agents in Depth · ch. 10")
        q_box = m.VGroup(quote, q_src).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT).move_to([1.2, 2.0, 0], aligned_edge=m.LEFT)

        def key_card(name, label, y_, fill_):
            box = card(5.3, 1.55, EDGE, PANEL, radius=0.18, stroke=2).move_to([3.85, y_, 0])
            head = m.VGroup(icon(name, 0.42), text(label, 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25)
            head.move_to(box.get_corner(m.UL) + [0.3, -0.4, 0], aligned_edge=m.LEFT)
            meter = Meter(4.6, fill_, SUN if name == KEY_AGENT else AZURE, 0.22).move_to(box.get_corner(m.DL) + [0.35, 0.42, 0], aligned_edge=m.LEFT)
            return m.VGroup(box, head, meter), meter

        team_card, team_meter = key_card(KEY_TEAM, "team key", 0.35, 0.35)
        agent_card, agent_meter = key_card(KEY_AGENT, "nightly agent's own key", -1.55, 0.0)
        cap_lab = pill("limit reached · stopped", ROSE, ROSE_FILL, 20).next_to(agent_card, m.DOWN, buff=0.2).align_to(agent_card, m.RIGHT)
        safe_lab = text("unaffected", 22, MINT, weight=m.BOLD).move_to(team_card[0].get_corner(m.UR) + [-0.3, -0.4, 0], aligned_edge=m.RIGHT)

        self.say(
            "[Next, limits.] [Every task gets a budget of turns and tokens], [and stops cleanly when it runs out.]",
            self.new_header("05 · limits", "Budgets and rate limits"),
            m.AnimationGroup(m.FadeIn(bud, shift=0.15 * m.UP), m.FadeIn(stuck), m.FadeIn(m.VGroup(m_turns, m_tokens, lt, ltk))),
            m.AnimationGroup(
                m_turns.fill_to(USED_TURNS / MAX_TURNS, run_time=2.0), m_tokens.fill_to(1.0, run_time=2.0),
                m.FadeIn(vt), m.FadeIn(vtk),
                m.Succession(m.Wait(1.6), m.AnimationGroup(m.FadeIn(stopped, shift=0.1 * m.UP), m.FadeIn(b_src),
                                                             m.Circumscribe(m.VGroup(*b_lines[2], *b_lines[3]), color=ROSE, buff=0.05))),
            ),
        )
        self.say(
            "[The team's agents also share one rate limit at the provider.] [If one busy task grabs it all], [the others get rate limit errors, as in episode twenty.]",
            m.AnimationGroup(m.FadeOut(screen1), m.FadeIn(m.VGroup(heads, heads_lab, busy_lab, direct, api, api_t, api_s, *segs))),
            m.AnimationGroup(m.Circumscribe(busy_lab, color=SUN, buff=0.06), *[m.Transform(s, t) for s, t in zip(segs, segs_greedy)]),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(p, scale=0.7) for p in r429], lag_ratio=0.1), m.FadeIn(chip20, shift=0.1 * m.UP)),
        )
        self.say(
            "[So a limiter in the harness gives each task a fair share.]",
            m.AnimationGroup(m.FadeOut(direct), m.FadeOut(r429), m.FadeIn(lim), m.FadeIn(lim_t), m.FadeIn(lim_s), m.FadeIn(to_lim), m.GrowArrow(lim_out),
                             *[m.Transform(s, t) for s, t in zip(segs, segs_fair)]),
        )
        self.say(
            "[The worst case is a runaway]: [an agent spawning sub-agents in a loop.] [The book AI Agents in Depth warns it can spawn thousands.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(heads, heads_lab, busy_lab, lim, lim_t, lim_s, to_lim, lim_out, api, api_t, api_s, chip20, *segs)),
                             m.FadeIn(m.VGroup(root, root_lab), shift=0.15 * m.UP)),
            m.Succession(*[m.AnimationGroup(m.Create(e), m.FadeIn(g, scale=0.5), run_time=0.55) for e, g in zip(edges, gens)]),
            m.FadeIn(q_box, shift=0.15 * m.UP),
        )
        self.say(
            "[So give each autonomous agent its own API key and spending limit]: [the runaway hits its own ceiling], [not the team's.]",
            m.FadeIn(m.VGroup(team_card, agent_card), shift=0.15 * m.UP),
            m.AnimationGroup(agent_meter.fill_to(1.0, run_time=1.4), m.FadeIn(cap_lab, shift=0.1 * m.UP),
                             gens[3].animate.set_color(FAINT), edges[3].animate.set_stroke(FAINT)),
            m.AnimationGroup(m.Circumscribe(team_card, color=MINT, buff=0.06), m.FadeIn(safe_lab)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · evals in CI

    def evals(self):
        self.next_section("6 · Evals in CI", notes=f"Li ch. 7: 'Every prompt change should run regression tests on the evaluation set—just as code changes require CI'; 'extract failed and suspicious cases from production traces', anonymize, 'distill them into new test cases and regression tests'; 'the failure patterns exposed in production today become the regression tests guarding the baseline tomorrow'. The eval set is Ep 29's tinyshop set ({EVAL_TASKS} tasks). Illustrative comparison: v14 passes {V14.passed}, v15 ('test only at the end', {V15.turns} turns instead of {V14.turns}) passes {V15.passed}; cost per task {V14.cost:,} → {V15.cost:,} full-price tokens and time {V14.time} → {V15.time} s, computed with the cost and latency examples above. A drop of {V14.passed - V15.passed} points is well beyond Ep 29's ±9-point noise for {EVAL_TASKS} tasks. task-0187 is illustrative.")
        diff_src = "- Run the tests after every edit.\n+ Run the tests once, at the end."
        diff = code_block(diff_src, size=20, title="system_prompt.md · v14 → v15", accent=ORCHID)
        paint(diff[1], diff_src, {r"^-.*$": ROSE, r"^\+.*$": MINT})
        diff.move_to([-6.6, 2.4, 0], aligned_edge=m.UL)
        d_lines = lines_of(diff[1], diff_src)
        bot = otto(0.6).move_to([1.6, 1.75, 0])
        better = pill("better?", MINT, MINT_FILL, 22).next_to(bot, m.RIGHT, buff=0.35).shift(0.3 * m.UP)
        worse = pill("worse?", ROSE, ROSE_FILL, 22).next_to(bot, m.RIGHT, buff=0.35).shift(0.3 * m.DOWN)
        bw_group = m.VGroup(bot, better, worse)

        py = -0.3

        def stage(label, sub, color, fill, w=2.75):
            box = card(w, 1.05, color, fill, radius=0.18, stroke=2.5)
            words = m.VGroup(text(label, 22, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.06).move_to(box)
            return m.VGroup(box, words)

        st1 = stage("prompt change", "v15", ORCHID, ORCHID_FILL)
        st2 = stage("eval set", f"{EVAL_TASKS} tasks", MINT, MINT_FILL)
        st3 = stage("compare", "with v14", AZURE, AZURE_FILL)
        st4 = stage("merge?", "the gate", EDGE, CARD)
        pipe = m.VGroup(st1, st2, st3, st4).arrange(m.RIGHT, buff=0.62).move_to([0, py, 0])
        p_arrows = [arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(pipe, pipe[1:])]
        chip29 = code("ep 29", 20, MODULES[6][1], weight=m.BOLD).next_to(st2, m.UP, buff=0.12)
        n_slot = Slot(self, st2[1][1])
        n101 = text(f"{EVAL_TASKS + 1} tasks", 20, MINT, weight=m.BOLD).move_to(st2[1][1])
        blocked = m.VGroup(card(2.75, 1.05, ROSE, ROSE_FILL, radius=0.18, stroke=2.5),
                           m.VGroup(text("blocked", 22, INK, weight=m.BOLD), text("v15 stays out", 20, MUTED)).arrange(m.DOWN, buff=0.06)).move_to(st4)
        blocked[1].move_to(blocked[0])
        ci_src = cite("Li, AI Agents in Depth · ch. 7: “just as code changes require CI”").next_to(pipe, m.DOWN, buff=0.22).align_to(pipe, m.LEFT)

        # the comparison table
        ty0 = -1.7
        cols = [-6.1, -1.3, 0.9]

        def trow(label, a, b, y_, bcolor):
            return m.VGroup(text(label, 22, MUTED).move_to([cols[0], y_, 0], aligned_edge=m.LEFT),
                            code(a, 22, INK).move_to([cols[1], y_, 0], aligned_edge=m.LEFT),
                            code(b, 22, bcolor, weight=m.BOLD).move_to([cols[2], y_, 0], aligned_edge=m.LEFT))

        head_row = m.VGroup(code("v14", 22, MUTED, weight=m.BOLD).move_to([cols[1], ty0, 0], aligned_edge=m.LEFT),
                            code("v15", 22, MUTED, weight=m.BOLD).move_to([cols[2], ty0, 0], aligned_edge=m.LEFT))
        r_cost = trow("cost per task", f"{V14.cost:,}", f"{V15.cost:,}", ty0 - 0.48, MINT)
        r_time = trow("time per task", f"{V14.time} s", f"{V15.time} s", ty0 - 0.92, MINT)
        r_pass = trow("tasks passed", f"{V14.passed} / {EVAL_TASKS}", f"{V15.passed} / {EVAL_TASKS}", ty0 - 1.36, ROSE)
        for r in (r_cost, r_time, r_pass):
            for k in (1, 2):
                sit(r[k], baseline(r[0], r[0].text.replace(" ", "")))
        rule = m.Line([cols[0], ty0 - 0.24, 0], [3.2, ty0 - 0.24, 0]).set_stroke(EDGE, 2)
        ex_lab = pill("for example", MUTED, CARD, 20).move_to([5.1, ty0 - 0.7, 0])

        # production traces → regression tests
        tr = mini_trace(2.4).move_to([1.6, 1.75, 0])
        tr_lab = code("task-0187 · failed", 20, ROSE).next_to(tr, m.UP, buff=0.12)
        strip = pill("strip private data", INK, CARD_HI, 21).move_to([5.0, 1.75, 0])
        tr_arrow = arrow(tr.get_right(), strip.get_left(), MUTED, buff=0.1)
        case = pill("new regression test", MINT, MINT_FILL, 20).next_to(strip, m.DOWN, buff=0.3)
        reg_src = cite("Li · ch. 7").next_to(case, m.DOWN, buff=0.12).align_to(case, m.RIGHT)

        self.say(
            "[Last, change.] [Every edit to Otto's prompt can make it better or worse.]",
            m.AnimationGroup(self.new_header("06 · evals in CI", "Test every change"), m.FadeIn(diff, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(bot, shift=0.1 * m.UP), m.LaggedStart(m.FadeIn(better, shift=0.1 * m.LEFT), m.FadeIn(worse, shift=0.1 * m.LEFT), lag_ratio=0.4)),
        )
        self.say(
            "[So treat prompts like code]: [each change runs the eval set from episode twenty-nine], [just as code changes go through continuous integration.]",
            m.FadeIn(st1, shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowArrow(p_arrows[0]), m.FadeIn(st2, shift=0.15 * m.UP), m.FadeIn(chip29)),
            m.AnimationGroup(m.GrowArrow(p_arrows[1]), m.FadeIn(st3, shift=0.15 * m.UP), m.GrowArrow(p_arrows[2]), m.FadeIn(st4, shift=0.15 * m.UP), m.FadeIn(ci_src)),
        )
        self.say(
            "[For example, version fifteen tests only at the end.] [It's cheaper and faster], [but passes fifteen fewer tasks], [so the change is blocked.]",
            m.AnimationGroup(m.Circumscribe(d_lines[1], color=MINT, buff=0.06), m.FadeIn(m.VGroup(head_row, rule)), m.FadeIn(ex_lab)),
            m.LaggedStart(m.FadeIn(r_cost, shift=0.1 * m.UP), m.FadeIn(r_time, shift=0.1 * m.UP), lag_ratio=0.4),
            m.FadeIn(r_pass, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(st4), m.FadeIn(blocked, scale=1.1)),
        )
        self.say(
            "[And the eval set grows from production]: [a failed trace is stripped of private data], [and becomes tomorrow's regression test.]",
            m.AnimationGroup(m.FadeOut(bw_group), m.FadeIn(tr, shift=0.15 * m.UP), m.FadeIn(tr_lab)),
            m.AnimationGroup(m.GrowArrow(tr_arrow), m.FadeIn(strip, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(case, shift=0.1 * m.DOWN), m.FadeIn(reg_src), n_slot.to(n101), m.Circumscribe(st2, color=MINT, buff=0.06)),
        )
        st2[1].remove(st2[1][1])  # replaced by n101: keep it from coming back in the fade-out
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · the loop

    def big_loop(self):
        self.next_section("7 · Production is a loop", notes="Serve the team's tasks; trace every task (cost, time, spans, versions); turn failures into tests and run the eval set on every change; ship only what passes, versioned — then serve again.")
        ring = loop_ring(["serve", "trace", "test", "ship"], radius=1.75, colors=[AZURE, SUN, MINT, LIME], size=26, center=(0, -0.45, 0))
        nodes = ring[0]
        side = [
            text("40 tasks a day", 22, MUTED).next_to(nodes[0], m.UP, buff=0.2),
            text("cost · time · spans", 22, MUTED).next_to(nodes[1], m.RIGHT, buff=0.3),
            text("evals in CI", 22, MUTED).next_to(nodes[2], m.DOWN, buff=0.2),
            text("versioned, with budgets", 22, MUTED).next_to(nodes[3], m.LEFT, buff=0.3),
        ]
        self.say(
            "[Put together, production is a loop]: [serve the team], [trace every task], [test every change], [and ship only what passes.]",
            m.AnimationGroup(self.new_header("07 · together", "Production is a loop"), m.FadeIn(ring, scale=0.9)),
            m.AnimationGroup(m.Circumscribe(nodes[0], color=AZURE, buff=0.06), m.FadeIn(side[0])),
            m.AnimationGroup(m.Circumscribe(nodes[1], color=SUN, buff=0.06), m.FadeIn(side[1])),
            m.AnimationGroup(m.Circumscribe(nodes[2], color=MINT, buff=0.06), m.FadeIn(side[2])),
            m.AnimationGroup(m.Circumscribe(nodes[3], color=LIME, buff=0.06), m.FadeIn(side[3])),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Measure (cost per task, time per task, traces and spans, evals on every change), limit (budgets per task, a fair share of the rate limit, an own key for autonomous agents), resume (an append-only log as the checkpoint, placeholders for interrupted calls, an idle watchdog, a neutral format for another provider).")
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 68, c) for w, c in (("Measure", AZURE), ("Limit", SUN), ("Resume", MINT))])
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("cost · time · traces · evals", "budgets · rate limits · own keys", "logs · watchdogs · fallback")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.8
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
        tagline = text("Measure every task, limit every agent, and let every run resume.", 28, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [measure], [limit], [resume.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Measure every task, limit every agent, and let every run resume.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
