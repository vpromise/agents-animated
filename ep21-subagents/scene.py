"""Agents, Animated · Episode 21 · Subagents (Module 5 · Scaling Up)

Why delegating to a fresh context helps. Running example: Otto is asked to add gift cards to tinyshop's
checkout and first needs a map of every place the order total is computed or changed. Searching itself,
it would trace the call chain through 30 files in 60 rounds (Learn Claude Code s06's own numbers); instead
it calls the `task` tool and a subagent does the survey in a fresh context and sends back one short report.
The run is not drawn by hand: the scene runs the reference loop and the subagent handler it shows (the
same lines, checked with inspect) against a scripted stand-in for the model and an in-memory tinyshop,
and every number on screen comes from that run (asserted below).

Sections: hook (the survey done inline: 120 more messages, the task buried) · the idea (a new terminal,
s06's analogy) · the task tool (schema, spawn_subagent, child tools without task, final text only, Claude
Code's Agent tool and depth limit) · spawn (two context columns: the child's 122 messages vs Otto's two)
· isolation over compression (Li, ch. 2; compaction from Episode 11) · limits (self-contained descriptions,
side effects remain, permissions still apply) · forks (Claude Code's fork mode: inherit the prefix to hit
the prompt cache; five things must match) · the OS analogy (Li, ch. 10) · recap: spawn, isolate, report.

Sources (facts dated October 2026):
- Learn Claude Code, s06 "Subagent" (learn.shareai.run/en/s06/): a bug fix that "reads 30 files… 60
  rounds… messages list grows to 120 entries"; "open a new terminal… close the terminal, write the result
  into your notes, and return"; the task tool {description} → spawn_subagent: fresh
  messages=[{"role": "user", "content": description}], its own loop with a safety limit (s06: 30 rounds),
  no task tool in the child (no recursive spawning), only the final text returns; "Conversation context
  is discarded, but file system side effects… remain"; "Context isolation does not mean permission
  isolation". Dive into CC source (the site's reading of Claude Code, not verified by us): Normal / Fork /
  General-purpose modes; the fork builds a cache-friendly prefix: "The goal isn't isolation, but making
  the Anthropic API's prompt cache hit"; byte-identical system prompt, tools, model, message prefix,
  thinking config.
- Li Bojie, AI Agents in Depth, ch. 2 "Isolation over compression": searching directly brings "dozens of
  files and tens of thousands of tokens of raw code into the main context"; delegating, the main context
  gains "two messages: one task description and one conclusion" ("The function is handle_callback in
  src/payment/callbacks.py, with two other call sites"); the cost: "the task description must be
  self-contained"; compression is "a lossy, post-hoc remedy requiring extra LLM calls"; isolation leaves
  the main agent's KV-cache prefix unaffected; sub-agents that inherit the parent's context "must be
  byte-aligned" with it. Ch. 10, Table 10-2: fork → spawn_subagent, kill → cancel_subagent, ps →
  list_agents, exit code / wait() → the structured summary the sub-agent returns.
- Claude Code docs, sub-agents (code.claude.com/docs/en/sub-agents, read October 2026): the Task tool was
  renamed Agent (v2.1.63); "By default, a subagent can spawn subagents of its own, up to three layers
  below the main conversation" (configurable; at the limit the Agent tool is withheld); a fork "inherits
  the entire conversation so far" with "the same system prompt, tools, model, and message history".

Illustrative (said "in our example"/"for example" or labelled): the gift-card job, tinyshop's 30 files and
their sizes, the searches, the report's contents, 4 characters per token, our child's safety limit of 100
rounds (s06's teaching value is 30; ours fits the 60-round survey), the vague description, the edited
file, the request strips in the fork section.

Code shown: spawn_subagent and the two tool lists (9 lines). s06 inlines the child's loop; here the child calls
harness_reference's agent_loop (Episode 2) with the tool list and a safety limit as parameters (names kept).
Voice: the series card's line is spoken as "Sub-agents" (intro_voice): spelled as one word, the narrator's
aligner timed it early and its trim clipped the word's end (fixed seeds give the same take on every run).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import base64
import inspect
import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the run, computed

CHARS_PER_TOKEN = 4  # rule of thumb for English text and code (illustrative)


def tokens(s):
    return max(1, round(len(s) / CHARS_PER_TOKEN))


TASK = "Add gift cards to checkout."
DESCRIPTION = (
    "Survey tinyshop, read only: find every place the order total is computed or changed. "
    "Reply with file, function and one line each, and say where a gift card fits."
)
VAGUE = "Check the thing we discussed."

# tinyshop, grown up (illustrative): 30 files and their size in tokens
FILES = [
    ("cart.py", 640), ("checkout.py", 1380), ("discounts.py", 1220), ("pricing.py", 980),
    ("payments.py", 1460), ("coupons.py", 860), ("orders.py", 1240), ("receipt.py", 720),
    ("refunds.py", 1050), ("tax.py", 610), ("shipping.py", 930), ("currency.py", 540),
    ("money.py", 480), ("inventory.py", 1110), ("catalog.py", 1320), ("models.py", 1650),
    ("api.py", 1840), ("views.py", 1510), ("invoices.py", 990), ("emails.py", 700),
    ("tests/test_cart.py", 820), ("tests/test_checkout.py", 1270), ("tests/test_discounts.py", 1060),
    ("tests/test_coupons.py", 760), ("tests/test_pricing.py", 900), ("tests/test_payments.py", 1180),
    ("tests/test_orders.py", 940), ("tests/test_refunds.py", 830), ("tests/test_money.py", 520),
    ("tests/test_receipt.py", 610),
]
# the first searches, as the viewer sees them; the rest follow the same pattern
SHOWN = {
    "cart.py": ("total", "cart.py:12: def total(items):", "def total(items):"),
    "checkout.py": ("cart.total", "checkout.py:31: amount = cart.total()", "def checkout(cart, card):"),
    "discounts.py": ("apply_code", "discounts.py:8: def apply_code(total, code):", "def apply_code(total, code):"),
    "pricing.py": ("add_tax", "pricing.py:20: def add_tax(amount):", "def add_tax(amount):"),
}


def stem(path):
    return path.split("/")[-1][:-3]


def survey_step(i):
    """The i-th file of the trace: (search term, grep output line, the file's first line)."""
    path, _ = FILES[i]
    if path in SHOWN:
        return SHOWN[path]
    name = stem(path)
    head = f"def {name}():" if name.startswith("test_") else f"class {name.title().replace('_', '')}:"
    return name, f"{path}:{3 + 7 * i % 90}: {head}", head


TREE = {path: survey_step(i)[2] + "\n" + "#" * (size * CHARS_PER_TOKEN - len(survey_step(i)[2]) - 1) for i, (path, size) in enumerate(FILES)}
assert all(tokens(TREE[p]) == size for p, size in FILES)

REPORT_ROWS = [
    ("cart.py", "total()", "sums items"),
    ("discounts.py", "apply_code()", "minus code"),
    ("pricing.py", "add_tax()", "plus tax"),
    ("checkout.py", "charge()", "pays it"),
]
REPORT_HEAD = f"The total changes in {len(REPORT_ROWS)} places:"
REPORT_TAIL = "Add gift cards just before charge()."
REPORT = "\n".join([REPORT_HEAD] + [f"{f:<14}{fn:<14}{what}" for f, fn, what in REPORT_ROWS] + [REPORT_TAIL])


def run_bash(command):
    """tinyshop's shell, simulated: only the survey's searches."""
    term = command.split()[2]
    for i, (path, _) in enumerate(FILES):
        if survey_step(i)[0] == term:
            return survey_step(i)[1]
    raise ValueError(command)


def read_file(path):
    return TREE[path]


def edit_file(path, old_text, new_text):
    raise AssertionError("the survey is read only")


# The series' three tools (harness_reference.TOOLS, names and descriptions; schemas omitted here)
TOOLS = [
    {"name": "bash", "description": "Run a shell command in the repo and return its output."},
    {"name": "read_file", "description": "Read a text file and return its contents."},
    {"name": "edit_file", "description": "Replace old_text with new_text in a file."},
]
TASK_TOOL = {
    "name": "task",
    "description": "Hand a self-contained subtask to a subagent with a fresh context. Returns its final report.",
    "input_schema": {
        "type": "object",
        "properties": {"description": {"type": "string"}},
        "required": ["description"],
    },
}
PARENT_TOOLS = TOOLS + [TASK_TOOL]  # Otto can delegate
CHILD_TOOLS = TOOLS  # no task tool: a child cannot spawn children
CHILD_RUNS = []  # every child's messages list, kept so the scene can count them


# The model's decisions (illustrative): Otto delegates once; the child traces the call chain, file by
# file (search, then read), and then answers in words.
CHILD_PLAN = []
for _i, (_path, _) in enumerate(FILES):
    CHILD_PLAN.append(("bash", {"command": f"grep -rn {survey_step(_i)[0]} ."}))
    CHILD_PLAN.append(("read_file", {"path": _path}))


class ScriptedModel:
    """Stands in for client.messages: Otto when the task tool is offered, a child otherwise."""

    def __init__(self):
        self.sent = {"otto": [], "child": []}

    def create(self, messages, tools):
        who = "otto" if any(t["name"] == "task" for t in tools) else "child"
        self.sent[who].append(len(messages))
        k = len(self.sent[who])
        if who == "otto":
            step = ("task", {"description": DESCRIPTION}) if k == 1 else None
            final = "Got it: gift cards go just before charge()."
        else:
            step = CHILD_PLAN[k - 1] if k <= len(CHILD_PLAN) else None
            final = REPORT
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=final)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{who[0]}{k:02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=ScriptedModel())


def tool_results(response):
    """harness_reference.agent_loop's inner for-loop (run_tool over every tool_use block)."""
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output = TOOL_HANDLERS[block.name](**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
    return results


def agent_loop(messages, tools=PARENT_TOOLS, max_rounds=100):
    """harness_reference.agent_loop (Episode 2), with the tool list and a safety limit as parameters."""
    for _ in range(max_rounds):  # safety limit
        response = client.messages.create(messages=messages, tools=tools)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": tool_results(response)})
    raise RuntimeError("safety limit reached")


def spawn_subagent(description):
    messages = [{"role": "user",
                 "content": description}]
    CHILD_RUNS.append(messages)
    response = agent_loop(messages,
        tools=CHILD_TOOLS, max_rounds=100)
    return response.content[0].text


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file, "task": spawn_subagent}

SPAWN_SRC = '''PARENT_TOOLS = TOOLS + [TASK_TOOL]
CHILD_TOOLS = TOOLS  # no task tool

def spawn_subagent(description):
    messages = [{"role": "user",
                 "content": description}]
    response = agent_loop(messages,
        tools=CHILD_TOOLS, max_rounds=100)
    return response.content[0].text'''

MAIN = [{"role": "user", "content": TASK}]
ANSWER = agent_loop(MAIN)
CHILD = CHILD_RUNS[0]
SENT = client.messages.sent


def as_dict(block):
    return block if isinstance(block, dict) else {k: v for k, v in vars(block).items()}


def tokens_of(msg):
    """A message's size: its content as JSON, at four characters per token."""
    content = msg["content"]
    if isinstance(content, str):
        return tokens(content)
    return tokens(json.dumps([as_dict(b) for b in content]))


# the code on screen is the code that ran
_ran = inspect.getsource(spawn_subagent)
_shown = SPAWN_SRC.split("\n")
assert all(line.strip() in _ran for line in _shown[3:]) and len(_shown) <= 10
assert PARENT_TOOLS == TOOLS + [TASK_TOOL] and CHILD_TOOLS == TOOLS
# the child's run
ROUNDS = len(SENT["child"]) - 1  # model calls that asked for a tool
FILES_READ = sum(1 for msg in CHILD[1::2] if msg["content"][0].type == "tool_use" and msg["content"][0].name == "read_file")
assert len(FILES) == FILES_READ == 30 and ROUNDS == 60  # "thirty files, sixty rounds" (learn-claude-code s06)
assert len(CHILD) == 1 + 2 * ROUNDS + 1 == 122  # "one hundred and twenty-two messages"
assert CHILD[0]["content"] == DESCRIPTION and CHILD[-1]["content"][0].text == REPORT
assert len(SENT["child"]) <= 100  # inside the safety limit
assert not any(t["name"] == "task" for t in CHILD_TOOLS)  # no recursion
# searched inline, the same 120 messages would have landed in Otto's own list
SURVEY = CHILD[1:-1]
DIRECT = [MAIN[0]] + SURVEY
DIRECT_TOKENS = sum(tokens_of(msg) for msg in SURVEY)
assert len(SURVEY) == 120 and len(DIRECT) == 121  # "grown by a hundred and twenty entries"
assert round(DIRECT_TOKENS, -3) == 33_000  # "about thirty-three thousand tokens"
# delegated: two messages, the call and the report
assert MAIN[1]["content"][0].name == "task" and MAIN[2]["content"][0]["content"] == REPORT
assert SENT["otto"] == [1, 3] and len(MAIN) == 4
DELEGATED = MAIN[1:3]
DELEGATED_TOKENS = sum(tokens_of(msg) for msg in DELEGATED)
assert DELEGATED_TOKENS == 135  # "just a hundred and thirty-five tokens"
CHILD_TOKENS = sum(tokens_of(msg) for msg in CHILD)
RATIO = DIRECT_TOKENS / DELEGATED_TOKENS

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ACCENT = {"user": AZURE, "tool_use": SUN, "tool_result": MINT, "assistant": CORAL}
ROW_H = 0.46
LOCK = recolored("lock", MINT, MINT_FILL, prefix="ep21")


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


def leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from leaves(sub)
    else:
        yield animation


def fit(make, s, width):
    """make(s), shortened with an ellipsis until it is at most `width` wide."""
    words = make(s)
    if words.width <= width:
        return words
    n = max(4, int(len(s) * width / words.width))
    while True:
        words = make(s[:n].rstrip(" ,.:") + " …")
        if words.width <= width or n <= 4:
            return words
        n -= 1


def summary(msg):
    """(kind, body, tokens) of one message, as a row shows it."""
    content = msg["content"]
    if isinstance(content, str):
        return "user", content, tokens_of(msg)
    block = content[0]
    if isinstance(block, dict):
        return "tool_result", block["content"].split("\n")[0], tokens_of(msg)
    if block.type == "tool_use":
        arg = next(iter(block.input.values()))
        return "tool_use", f'{block.name}("{arg}")', tokens_of(msg)
    return "assistant", block.text.split("\n")[0], tokens_of(msg)


def row(kind, body, tok, width):
    """One message on one line: a stripe in its role's colour, the content, its size in tokens."""
    accent = ACCENT[kind]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    count = code(f"{tok:,}", 20, MUTED)
    room = width - 0.3 - count.width - 0.45
    if kind in ("user", "assistant"):
        words = fit(lambda s: text(s, 20, INK), body, room)
    else:
        words = fit(lambda s: code(s, 20, accent), body, room)
    words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    count.move_to(box).align_to(box, m.RIGHT).shift(0.2 * m.LEFT)
    base = box.get_y() - 0.09
    sit(words, base)
    sit(count, base)
    return m.VGroup(box, stripe, words, count)


def msg_row(msg, width):
    return row(*summary(msg), width)


def cross(at, size=0.26, color=ROSE, stroke=5):
    a, b = np.array([size / 2, size / 2, 0]), np.array([size / 2, -size / 2, 0])
    at = np.array(at, dtype=float)
    return m.VGroup(m.Line(at - a, at + a), m.Line(at - b, at + b)).set_stroke(color, stroke)


def tick(at, size=0.3, color=MINT, stroke=5):
    at = np.array(at, dtype=float)
    pts = [at + np.array(p) * size for p in ((-0.5, 0.0, 0), (-0.15, -0.38, 0), (0.55, 0.45, 0))]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, stroke)


class Slot:
    """One place on screen whose content is swapped (a counter, a status line)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15):
        anims = []
        if self.current is not None:
            anims.append(m.FadeOut(self.current, shift=shift * m.UP))
        if new is not None:
            anims.append(m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


class Column:
    """A ContextColumn with a label on the left and a message/token counter on the right."""

    def __init__(self, scene, center, width, height, label, base, below=False):
        self.col = ContextColumn(center=center, width=width, height=height, title="context", gap=0.08)
        self.panel = self.col.panel
        self.label = label.next_to(self.panel, m.UP, buff=0.16).align_to(self.panel, m.LEFT)
        self.base = base(self.label)
        self.frame = m.VGroup(self.panel, self.label)
        self.counter = Slot(scene)
        self.row_w = width - 0.36
        self.below = below
        if below:  # the counter sits under the panel, on its own baseline
            self.base = self.panel.get_bottom()[1] - 0.36

    def count(self, n, tok=None):
        s = f"{n} message{'s' * (n != 1)}" + (f" · {tok:,} tokens" if tok is not None else "")
        words = code(s, 20, MUTED, t2c={r"[\d,]+": SUN})
        words.align_to(self.panel, m.RIGHT)
        return sit(words, self.base)  # above the panel (beside the label) or below it

    def show(self, n, tok=None):
        return self.counter.to(self.count(n, tok))

    def push(self, r, run_time=0.9):
        return self.col.push(r, run_time=run_time), r

    def jump(self, rows):
        """Swap the visible stack for `rows` (a fast-forward): old rows fade, new ones stack from the top."""
        old = list(self.col.items)
        y = self.col.top
        self.col.slots = []
        for r in rows:
            r.move_to([self.panel.get_x(), 0, 0]).align_to([0, y, 0], m.UP)
            self.col.slots.append([r, y])
            y -= r.height + self.col.gap
        self.col.items = list(rows)
        return m.AnimationGroup(
            *[m.FadeOut(r, shift=0.4 * m.UP) for r in old],
            m.LaggedStart(*[m.FadeIn(r, shift=0.3 * m.UP) for r in rows], lag_ratio=0.12),
        )


class CodeCard:
    """A code_block with per-line glyph groups, baselines and a highlight bar."""

    def __init__(self, src, title, width=None):
        self.block = code_block(src, size=20, width=width, title=title)
        self.box, self.body = self.block[0], self.block[1]
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n
        self.body.set_z_index(2)

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=AZURE):
        b = a if b is None else b
        top = max(g.get_top()[1] for g in self.lines[a]) + 0.1
        bot = min(g.get_bottom()[1] for g in self.lines[b]) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


def tool_chips(names, width=1.95, struck=()):
    chips = m.VGroup()
    for n in names:
        color, fill = (AZURE, AZURE_FILL) if n == "task" else (SUN, SUN_FILL)
        if n in struck:
            color, fill = FAINT, CARD
        c = chip(n, color, fill, width=width, height=0.5, size=24, mono=True)
        if n in struck:
            c[1].set_color(FAINT)
        chips.add(c)
    return chips.arrange_in_grid(rows=2, cols=2, buff=(0.18, 0.12))


def tools_card(caption, names, struck=(), width=4.5):
    cap = code(caption, 20, MUTED)
    chips = tool_chips(names, struck=struck)
    inner = m.VGroup(cap, chips).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    box = card(width, inner.height + 0.42, EDGE, CARD, radius=0.16, stroke=2)
    inner.move_to(box)
    cap.align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    return m.VGroup(box, cap, chips)


def report_card(width):
    """The child's report, as Otto reads it."""
    head = text(REPORT_HEAD, 22, INK, weight=m.SEMIBOLD)
    rows = m.VGroup()
    for f, fn, what in REPORT_ROWS:
        rows.add(m.VGroup(code(f, 20, AZURE), code(fn, 20, SUN), text(what, 20, MUTED)))
    xs = [0, 2.2, 4.25]
    for r in rows:
        for part, x in zip(r, xs):
            part.move_to([x, 0, 0], aligned_edge=m.LEFT)
        base = r[0][0].get_bottom()[1]
        sit(r[1], base)
        sit(r[2], base)
    rows.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
    tail = text(REPORT_TAIL, 22, MINT, weight=m.SEMIBOLD)
    inner = m.VGroup(head, rows, tail).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
    cap = m.VGroup(code("report", 20, CORAL, weight=m.BOLD), text("all Otto sees of the child's work", 20, MUTED)).arrange(m.RIGHT, buff=0.3)
    sit(cap[1], cap[0][0].get_bottom()[1])
    body = m.VGroup(cap, inner).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
    box = card(width, body.height + 0.6, CORAL, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    assert body.get_right()[0] < box.get_right()[0] - 0.2
    return m.VGroup(box, body)


def description_card(words, width, accent, title):
    cap = code(title, 20, accent, weight=m.BOLD)
    body = m.VGroup(*[text(line, 22, INK) for line in words]).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
    inner = m.VGroup(cap, body).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
    box = card(max(width, inner.width + 0.75), inner.height + 0.55, accent, CARD, radius=0.16, stroke=2)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, inner)


def wrap(s, n):
    """Split s into lines of at most about n characters."""
    lines, cur = [], ""
    for w in s.split():
        if cur and len(cur) + 1 + len(w) > n:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    return lines + [cur]


def equals(at, color, slash=False):
    """An equals sign drawn with lines (a text "=" is too short for the layout check); slashed: not equal."""
    at = np.array(at, dtype=float)
    bars = [m.Line(at + [-0.17, dy, 0], at + [0.17, dy, 0]).set_stroke(color, 5) for dy in (0.07, -0.07)]
    if slash:
        bars.append(m.Line(at + [-0.1, -0.2, 0], at + [0.1, 0.2, 0]).set_stroke(color, 5))
    return m.VGroup(*bars)


def segment(label, width, color, fill, size=20, mono=False):
    box = card(width, 0.72, color, fill, radius=0.12, stroke=2)
    words = (code(label, size, INK) if mono else text(label, size, INK, weight=m.SEMIBOLD)).move_to(box)
    return m.VGroup(box, words)


def messages_segment(width):
    box = card(width, 0.72, EDGE, CARD, radius=0.12, stroke=2)
    words = text("messages", 20, INK, weight=m.SEMIBOLD)
    colors = [AZURE, SUN, MINT, SUN, MINT, CORAL, AZURE, SUN, MINT]
    bars = m.VGroup(*[m.RoundedRectangle(corner_radius=0.02, width=0.09, height=0.4, stroke_width=0, fill_color=c, fill_opacity=1) for c in colors])
    bars.arrange(m.RIGHT, buff=0.06)
    body = m.VGroup(words, bars).arrange(m.RIGHT, buff=0.25)
    box = card(max(width, body.width + 0.45), 0.72, EDGE, CARD, radius=0.12, stroke=2)
    body.move_to(box)
    return m.VGroup(box, body)


# ---------------------------------------------------------------------------- the episode

# Episode.say, plus: anything not yet on screen (a row pushed and scrolled in the same line, a group whose
# parts were faded in separately, a counter swapped twice) waits for its own cue instead of showing early.
# Compiled under ManimGX's package path, like the kit's say, so plays keep the episode's line numbers.
def intro_voice(text, spoken="Sub-agents."):
    """The kit's narrator for the series card's line only, reading the title as "Sub-agents". Spelled as one
    word, the narrator's aligner timed "Subagents" too early and its trim cut the word's end (the same take
    on every run: seeds are fixed). The speech keeps the script's text and words, so captions and cues match."""
    said = text.replace("Subagents.", spoken)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": said, "voice_ref": VOICE_REF, "exaggeration": 0.5, "cfg_weight": 0.3, "temperature": 0.8}).encode(),
        {"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=1800) as reply:
        out = json.load(reply)
    timed = out["words"]
    assert len(timed) == len(text.split())
    return m.Speech(base64.b64decode(out["audio"]), text=text, words=[Word(w, s, e) for w, (_, s, e) in zip(text.split(), timed)])


_SAY_SRC = """
def say(self, script, *animations, voice=None):
    if voice is None and "Episode twenty-one: Subagents." in script.replace("[", "").replace("]", "") and type(self).voice is narrator:
        voice = intro_voice  # see intro_voice
    on_screen = {id(x) for x in self.get_mobject_family_members()}
    for a in animations:
        for leaf in leaves(a):
            mob = getattr(leaf, "mobject", None)
            if mob is not None and not leaf.is_introducer() and id(mob) not in on_screen:
                leaf.introducer = True
    return episode_say(self, script, *animations, voice=voice)
"""
_SAY21 = {"leaves": leaves, "episode_say": Episode.say, "narrator": narrator, "intro_voice": intro_voice}
exec(compile(_SAY_SRC, str(Path(m.__file__).parent / "_ep21_say.py"), "exec"), _SAY21)


class Ep21Subagents(Episode):
    number = 21

    say = _SAY21["say"]

    def lesson(self):
        self.hook()
        self.idea()
        self.task_tool()
        self.spawn()
        self.isolation()
        self.limits()
        self.forks()
        self.analogy()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Otto is asked to add gift cards to tinyshop's checkout (illustrative). Before editing it needs a map of every place the order total changes; searching itself, it traces the call chain: grep, read the file, grep again. Learn Claude Code s06's example: a bug fix that 'reads 30 files… 60 rounds… messages list grows to 120 entries'. Our run (computed): {len(FILES)} files, {ROUNDS} rounds, {len(SURVEY)} new messages, {DIRECT_TOKENS:,} tokens at {CHARS_PER_TOKEN} characters per token (file sizes illustrative). The task is 1 of {len(DIRECT)} messages. Facts as of October 2026.")
        bot = otto(1.15).move_to([-4.75, 1.45, 0])
        col = Column(self, (2.45, -0.45, 0), 7.2, 5.9, code("Otto's messages", 20, MUTED), lambda lab: lab[0].get_bottom()[1])
        need = text("where does the total change?", 24, SUN).next_to(bot, m.DOWN, buff=0.45)
        status = Slot(self)

        def status_at(r, files):
            t = title_text(f"round {r}", 44, INK)
            sub = code(f"files read: {files}", 22, MUTED, t2c={r"\d+": SUN})
            return m.VGroup(t, sub).arrange(m.DOWN, buff=0.18).move_to([-4.75, -1.55, 0])

        src = cite("learn-claude-code · s06").move_to([-4.75, -3.1, 0])
        rows = {i: msg_row(DIRECT[i], col.row_w) for i in list(range(17)) + list(range(len(DIRECT) - 10, len(DIRECT)))}

        a0 = col.push(rows[0])[0]
        self.say(
            "[Otto has a new job]: [add gift cards to tinyshop's checkout.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(col.frame)),
            m.AnimationGroup(a0, col.show(1)),
        )
        self.say(
            "[Before it changes anything], [it needs a map]: [every place where the order total is computed or changed.]",
            m.Indicate(bot, color=CORAL, scale_factor=1.06),
            m.FadeIn(need, shift=0.15 * m.UP),
            m.Circumscribe(need, color=SUN, buff=0.08),
        )
        # built in timeline order: message i of the list arrives in round (i + 1) // 2, after i // 4 files
        b1 = m.AnimationGroup(status.to(status_at(1, 0)), col.push(rows[1], run_time=0.8)[0], col.show(2), run_time=0.8)
        b2 = m.AnimationGroup(col.push(rows[2], run_time=0.7)[0], col.show(3), run_time=0.7)
        b3 = m.AnimationGroup(status.to(status_at(2, 1)), m.Succession(col.push(rows[3], run_time=0.45)[0], col.push(rows[4], run_time=0.45)[0]), col.show(5), run_time=0.9)
        b4 = m.AnimationGroup(status.to(status_at(3, 1)), col.push(rows[5], run_time=0.6)[0], col.show(6), run_time=0.6)
        fast = []
        for i in range(6, 17):
            fast.append(m.AnimationGroup(col.push(rows[i], run_time=0.28)[0], col.show(i + 1), status.to(status_at((i + 1) // 2, i // 4), shift=0.08), run_time=0.28))
        self.say(
            "[So it traces the call chain]: [search for a function], [read the file it lives in], [search again], [and again.]",
            b1, b2, b3, b4, m.Succession(*fast),
        )
        tail = [rows[i] for i in range(len(DIRECT) - 10, len(DIRECT))]
        self.say(
            "[Thirty files and sixty rounds later], [its messages list has grown by a hundred and twenty entries.]",
            m.AnimationGroup(col.jump(tail), status.to(status_at(ROUNDS, FILES_READ)), col.show(len(DIRECT)), run_time=1.0),
            m.AnimationGroup(m.FadeIn(src), m.Circumscribe(col.panel, color=SUN, buff=0.06)),
        )
        # zoomed out: every message as a thin bar, length by size
        top, bottom = col.panel.get_top()[1] - 0.2, col.panel.get_bottom()[1] + 0.2
        pitch = (top - bottom) / len(DIRECT)
        left = col.panel.get_left()[0] + 0.25
        longest = max(tokens_of(msg) for msg in DIRECT)
        bars = m.VGroup()
        for i, msg in enumerate(DIRECT):
            kind = summary(msg)[0]
            w = max(0.1, (col.panel.width - 0.5) * tokens_of(msg) / longest)
            b = m.Rectangle(width=w, height=pitch * 0.6, stroke_width=0, fill_color=ACCENT[kind], fill_opacity=1)
            bars.add(b.move_to([left + w / 2, top - pitch * (i + 0.5), 0]))
        bars[0].set_fill(AZURE).stretch_to_fit_height(pitch * 0.9)
        self.say(
            "[In our example, that's about thirty-three thousand tokens of raw code], [carried along on every call from now on.]",
            m.AnimationGroup(
                *[m.FadeOut(r) for r in col.col.items],
                m.LaggedStart(*[m.FadeIn(b) for b in bars], lag_ratio=0.03, run_time=1.6),
                col.show(len(DIRECT), tokens_of_sum(DIRECT)),
            ),
            m.Circumscribe(col.counter.current, color=SUN, buff=0.08),
        )
        col.col.items, col.col.slots = [], []
        label = text("the job", 24, AZURE, weight=m.SEMIBOLD)
        label.move_to([col.panel.get_left()[0] - 0.95, bars[0].get_y(), 0])
        point = arrow(label.get_right(), bars[0].get_left() + 0.02 * m.LEFT, AZURE, buff=0.08)
        frac = text(f"1 of {len(DIRECT)} messages", 22, MUTED).next_to(label, m.UP, buff=0.16).align_to(label, m.RIGHT)
        self.say(
            "[And the job it was given]? [One thin line at the top], [buried under the search.]",
            m.AnimationGroup(m.FadeIn(label, shift=0.2 * m.RIGHT), m.GrowArrow(point)),
            m.AnimationGroup(m.Indicate(bars[0], color=AZURE, scale_factor=1.0), m.FadeIn(frac)),
            m.Circumscribe(bars[1:], color=MUTED, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the idea

    def idea(self):
        self.next_section("1 · The idea", notes="Learn Claude Code s06's analogy: tracing a call chain is like opening a new terminal; when you are done you close it, write the result into your notes, and return to the original terminal. That new terminal is a subagent: a fresh context (messages list) for one subtask. Terminal contents are illustrative.")
        main = terminal(["$ git switch -c gift-cards", "$ vim checkout.py"], width=6.2, title="main terminal")
        main.move_to([-3.35, 0.95, 0])
        q = text("where else does the total change?", 24, SUN).next_to(main, m.DOWN, buff=0.35)
        side = terminal(["$ grep -rn total .", "cart.py:12: def total(items):", "$ less discounts.py",
                         "$ grep -rn add_tax .", "… 30 files later"], width=5.9, title="new terminal")
        side.move_to([3.45, 0.35, 0])
        side_frame = m.VGroup(*side[:4])
        side_rows = list(side[4])
        notes = m.VGroup(code("notes.md", 20, MUTED), text("total: cart, discounts, pricing, checkout", 22, INK))
        notes.arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        nbox = card(main.width, notes.height + 0.5, MINT, CARD, radius=0.16, stroke=2)
        notes.move_to(nbox).align_to(nbox, m.LEFT).shift(0.35 * m.RIGHT)
        note = m.VGroup(nbox, notes).next_to(q, m.DOWN, buff=0.4)
        word = title_text("subagent", 60, AZURE)
        defn = text("a fresh context for one subtask", 28, INK)
        right = m.VGroup(word, defn).arrange(m.DOWN, buff=0.3).move_to([3.45, 0.15, 0])
        src = cite("learn-claude-code · s06").next_to(right, m.DOWN, buff=0.45)

        self.say(
            "[Learn Claude Code's fix starts with a habit you already know.] [You're working in one terminal], [and a side question comes up.]",
            self.new_header("01 · the idea", "Open a new terminal"),
            m.FadeIn(main, shift=0.2 * m.UP),
            m.FadeIn(q, shift=0.15 * m.UP),
        )
        self.say(
            "[So you open a new terminal], [dig through the files there], [close it], [and write the result into your notes.]",
            m.FadeIn(side_frame, target_position=q, scale=0.6),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in side_rows], lag_ratio=0.3, run_time=1.3),
            m.AnimationGroup(m.FadeOut(side_frame, scale=0.6, target_position=note), *[m.FadeOut(r, target_position=note) for r in side_rows]),
            m.FadeIn(note, shift=0.2 * m.UP),
        )
        self.say(
            "[Your main terminal never saw the mess.] [That new terminal is a subagent]: [a fresh context for one subtask.]",
            m.Circumscribe(main, color=MINT, buff=0.08),
            m.FadeIn(word, scale=0.85),
            m.AnimationGroup(m.FadeIn(defn, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the task tool

    def task_tool(self):
        self.next_section("2 · The task tool", notes="Learn Claude Code s06: the task tool {description} → spawn_subagent; the child starts with messages=[{'role': 'user', 'content': description}], runs its own loop with a safety limit (s06: 30 rounds; ours: 100, so the 60-round survey fits), gets the base tools but not task (no recursive spawning), and only its final text returns. The scene runs exactly this code. Claude Code docs (October 2026): the Task tool was renamed Agent (v2.1.63); by default a subagent may spawn its own, 'up to three layers below the main conversation', and at the limit the tool is withheld.")
        cc = CodeCard(SPAWN_SRC, None)
        cc.block.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, 2.45, 0], m.UP)
        rx0 = cc.box.get_right()[0] + 0.45
        rw = min(4.7, 6.6 - rx0)
        rcx = (rx0 + 6.6) / 2
        parent = tools_card("Otto's tools", ["bash", "read_file", "edit_file", "task"], width=rw)
        parent.move_to([rcx, 0, 0]).align_to([0, 2.45, 0], m.UP)
        task_chip = parent[2][3]
        parent_rest = m.VGroup(parent[0], parent[1], *parent[2][:3])
        schema = code('input: {"description": str}', 20, MUTED, t2c={r'"description"': AZURE})
        schema.next_to(parent, m.DOWN, buff=0.2).align_to(parent, m.LEFT).shift(0.15 * m.RIGHT)
        child = tools_card("the child's tools", ["bash", "read_file", "edit_file", "task"], struck=("task",), width=rw)
        child.next_to(schema, m.DOWN, buff=0.35).align_to(parent, m.LEFT)
        ghost = child[2][3]
        x = cross(ghost[0].get_right() + 0.32 * m.LEFT, 0.26, ROSE, 5)
        norec = pill("no recursion", ROSE, ROSE_FILL, 22).next_to(child, m.DOWN, buff=0.3)
        cc_note = text("Claude Code: the Agent tool, nested up to a depth limit", 22, INK)
        cc_note.next_to(cc.box, m.DOWN, buff=0.45).align_to(cc.box, m.LEFT)
        cc_src = cite("Claude Code docs · sub-agents, 2026").next_to(cc_note, m.DOWN, buff=0.14).align_to(cc_note, m.LEFT)

        bar = cc.bar(3)
        self.say(
            "[In code, delegating is just one more tool], [called task.]",
            m.AnimationGroup(self.new_header("02 · the task tool", "Delegation is one more tool"), m.FadeIn(parent_rest, shift=0.15 * m.UP)),
            m.FadeIn(task_chip, scale=0.7),
        )
        self.say(
            "[Its only input is a description of the subtask], [and its handler is spawn subagent.]",
            m.AnimationGroup(m.FadeIn(schema, shift=0.1 * m.UP), m.Circumscribe(task_chip, color=AZURE, buff=0.06)),
            m.AnimationGroup(m.FadeIn(cc.block, shift=0.2 * m.UP), m.FadeIn(bar)),
        )
        self.say(
            "[The handler starts a brand new messages list], [holding nothing but that description.]",
            m.Transform(bar, cc.bar(4, 5)),
            m.Circumscribe(cc.part(5, "description"), color=AZURE, buff=0.05),
        )
        self.say(
            "[Then it runs the same loop from episode two], [with a safety limit on rounds.]",
            m.AnimationGroup(m.Transform(bar, cc.bar(6, 7)), m.Circumscribe(cc.part(6, "agent_loop"), color=CORAL, buff=0.05)),
            m.Circumscribe(cc.part(7, "max_rounds=100"), color=SUN, buff=0.05),
        )
        self.say(
            "[The child gets Otto's tools, minus one]: [no task tool], [so it can't spawn children of its own.]",
            m.AnimationGroup(m.Transform(bar, cc.bar(0, 1)), m.FadeIn(child, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(x), m.Circumscribe(cc.part(1, "CHILD_TOOLS = TOOLS"), color=ROSE, buff=0.05)),
            m.FadeIn(norec, shift=0.1 * m.UP),
        )
        self.say(
            "[When the loop ends], [only the final text comes back.]",
            m.Transform(bar, cc.bar(8, color=MINT)),
            m.Circumscribe(cc.part(8, "response.content[0].text"), color=MINT, buff=0.05),
        )
        self.say(
            "[Claude Code calls this the Agent tool], [and lets subagents nest a few levels deep, up to a limit.]",
            m.AnimationGroup(m.FadeIn(cc_note, shift=0.1 * m.UP), m.FadeIn(cc_src)),
            m.Circumscribe(norec, color=MUTED, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ spawn

    def spawn(self):
        self.next_section("3 · Spawn", notes=f"The run, computed: Otto's list starts with the job; call 1 returns a task tool_use whose description is self-contained; the harness runs spawn_subagent: the child's list starts with that description alone, runs {ROUNDS} tool rounds ({FILES_READ} files read) and answers in words: {len(CHILD)} messages, {CHILD_TOKENS:,} tokens. Only the final text returns, as the task's tool_result ({tokens_of(MAIN[2])} tokens). Otto's list: {len(MAIN) - 1} messages before its next call; the call and the result together are {DELEGATED_TOKENS} tokens. The child's context is discarded. The report's contents are illustrative.")
        width, height, y = 6.2, 4.65, -0.75
        lab_l = m.VGroup(otto(0.32), title_text("Otto", 26, INK)).arrange(m.RIGHT, buff=0.22)
        lab_r = m.VGroup(otto(0.32, AZURE, AZURE_FILL), title_text("subagent", 26, AZURE)).arrange(m.RIGHT, buff=0.22)
        base = lambda lab: lab[1][0].get_bottom()[1]  # noqa: E731
        left = Column(self, (-3.4, y, 0), width, height, lab_l, base, below=True)
        right = Column(self, (3.4, y, 0), width, height, lab_r, base, below=True)
        rows_l = [msg_row(msg, left.row_w) for msg in MAIN[:3]]
        keep = list(range(17)) + list(range(len(CHILD) - 8, len(CHILD)))
        rows_r = {i: msg_row(CHILD[i], right.row_w) for i in keep}
        last = len(CHILD) - 1

        desc_lines = wrap(DESCRIPTION, 40)
        call = description_card(desc_lines, width, SUN, 'tool_use · task · "description":')
        call.move_to(right.panel).set_z_index(3)  # stays above the child's panel as it fades in

        a0 = left.push(rows_l[0])[0]
        self.say(
            "[Let's run it.] [Otto's context starts with the job]: [add gift cards to checkout.]",
            self.new_header("03 · spawn", "A fresh context, one report back"),
            m.FadeIn(left.frame),
            m.AnimationGroup(a0, left.show(1, tokens_of(MAIN[0]))),
        )
        a1 = left.push(rows_l[1])[0]
        self.say(
            "[Otto calls the task tool], [with a description that stands on its own.]",
            m.AnimationGroup(a1, left.show(2, tokens_of_sum(MAIN[:2]))),
            m.FadeIn(call, target_position=rows_l[1], scale=0.6),
        )
        r0 = right.push(rows_r[0])[0]
        self.say(
            "[The harness spawns a child.] [Its context holds exactly one message]: [that description.]",
            m.FadeIn(right.frame),
            m.AnimationGroup(m.FadeOut(call, target_position=right.col.slots[0][0], scale=0.4), r0),
            m.AnimationGroup(right.show(1, tokens_of(CHILD[0])), m.Circumscribe(rows_r[0], color=AZURE, buff=0.05)),
        )
        run = lambda i: tokens_of_sum(CHILD[: i + 1])  # noqa: E731
        # built in timeline order (the counter slot and the stack track what each step replaces)
        s1 = m.AnimationGroup(m.Succession(right.push(rows_r[1], run_time=0.45)[0], right.push(rows_r[2], run_time=0.45)[0]), right.show(3, run(2)), run_time=0.9)
        s2 = m.AnimationGroup(m.Succession(right.push(rows_r[3], run_time=0.4)[0], right.push(rows_r[4], run_time=0.4)[0]), right.show(5, run(4)), run_time=0.8)
        fast = []
        for i in range(5, 17):
            fast.append(m.AnimationGroup(right.push(rows_r[i], run_time=0.26)[0], right.show(i + 1, run(i)), run_time=0.26))
        self.say(
            "[The child searches and reads], [then searches again], [file after file.]",
            s1, s2, m.Succession(*fast),
        )
        tail = [rows_r[i] for i in range(len(CHILD) - 8, last)]
        self.say(
            "[Sixty rounds later], [all of it sits in the child's context], [and none of it in Otto's.]",
            m.AnimationGroup(right.jump(tail), right.show(last, run(last - 1)), run_time=1.0),
            m.Circumscribe(right.panel, color=SUN, buff=0.06),
            m.Circumscribe(left.panel, color=MINT, buff=0.06),
        )
        final = right.push(rows_r[last])[0]
        self.say(
            "[Then the child answers in words], [with a short report.]",
            m.AnimationGroup(final, right.show(len(CHILD), CHILD_TOKENS)),
            m.Circumscribe(rows_r[last], color=CORAL, buff=0.05),
        )
        back, ret = left.push(rows_l[2])
        self.say(
            "[Only that report goes back to Otto], [as the task tool's result.]",
            m.AnimationGroup(
                m.FadeIn(ret, target_position=rows_r[last], path_arc=0.4, run_time=1.1),
                left.show(3, sum(tokens_of(x) for x in MAIN[:3])),
            ),
            m.Circumscribe(ret, color=MINT, buff=0.05),
        )
        gone = pill("context discarded", ROSE, ROSE_FILL, 22).move_to(right.panel)
        old = list(right.col.items)
        self.say(
            "[The child's context], [all one hundred and twenty-two messages], [is thrown away.]",
            m.Circumscribe(right.panel, color=ROSE, buff=0.06),
            m.Indicate(right.counter.current, color=ROSE, scale_factor=1.08),
            m.AnimationGroup(*[m.FadeOut(r, shift=0.3 * m.DOWN, scale=0.9) for r in old], right.counter.to(None), m.FadeIn(gone, scale=0.8)),
        )
        right.col.items, right.col.slots = [], []
        grew = m.VGroup(rows_l[1], rows_l[2])
        plus = code(f"+{len(DELEGATED)} messages · +{DELEGATED_TOKENS} tokens", 22, MINT, t2c={r"[\d+]+": SUN})
        plus.next_to(rows_l[2], m.DOWN, buff=0.35).align_to(rows_l[2], m.LEFT)
        self.say(
            "[Otto's context grew by just two messages]: [the call, and the report.]",
            m.AnimationGroup(m.Circumscribe(grew, color=MINT, buff=0.06), m.FadeIn(plus, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Indicate(rows_l[1][2], color=SUN, scale_factor=1.05), m.Indicate(rows_l[2][2], color=MINT, scale_factor=1.05)),
        )
        rep = report_card(width)
        rep.move_to(right.panel)
        self.say(
            "[Here's the report.] [Four places, one line each], [and where gift cards fit in.]",
            m.AnimationGroup(m.FadeOut(gone, scale=0.8), m.FadeOut(right.panel), m.FadeIn(rep, target_position=rows_l[2], scale=0.5)),
            m.Circumscribe(rep[1][1][1], color=AZURE, buff=0.06),
            m.Circumscribe(rep[1][1][2], color=MINT, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ isolation over compression

    def isolation(self):
        self.next_section("4 · Isolation over compression", notes=f"Computed: searching inline adds {len(SURVEY)} messages and {DIRECT_TOKENS:,} tokens to Otto's list; delegating adds {len(DELEGATED)} messages and {DELEGATED_TOKENS} tokens (about {RATIO:.0f} times fewer). Li, AI Agents in Depth, ch. 2 'Isolation over compression': direct search brings 'dozens of files and tens of thousands of tokens of raw code into the main context'; delegated, the main context gains 'two messages: one task description and one conclusion' — 'The function is handle_callback in src/payment/callbacks.py, with two other call sites'; compression is 'a lossy, post-hoc remedy requiring extra LLM calls'; isolation keeps noise out from the start and leaves the main agent's KV-cache prefix unaffected. Episode 11: compaction.")
        x0, full = -3.75, 4.6
        names = [text("searched itself", 26, INK, weight=m.SEMIBOLD), text("delegated", 26, INK, weight=m.SEMIBOLD)]
        ys = [1.85, 0.95]
        for n, yy in zip(names, ys):
            n.move_to([-6.5, yy, 0], aligned_edge=m.LEFT)
        w1, w2 = full, max(0.08, full * DELEGATED_TOKENS / DIRECT_TOKENS)
        bar1 = m.RoundedRectangle(corner_radius=0.08, width=w1, height=0.42, stroke_width=0, fill_color=SUN, fill_opacity=1)
        bar2 = m.RoundedRectangle(corner_radius=0.03, width=w2, height=0.42, stroke_width=0, fill_color=MINT, fill_opacity=1)
        bar1.move_to([x0 + w1 / 2, ys[0], 0])
        bar2.move_to([x0 + w2 / 2, ys[1], 0])
        v1 = code(f"+{len(SURVEY)} messages · {DIRECT_TOKENS:,} tokens", 20, INK, t2c={r"[\d,+]+": SUN}).next_to(bar1, m.RIGHT, buff=0.25)
        v2 = code(f"+{len(DELEGATED)} messages · {DELEGATED_TOKENS} tokens", 20, INK, t2c={r"[\d,+]+": MINT}).next_to(bar2, m.RIGHT, buff=0.25)
        v2.align_to(v1, m.LEFT)
        for n, v in zip(names, (v1, v2)):
            sit(v, n[0].get_bottom()[1])
        caption = code("tokens added to Otto's context", 20, MUTED).move_to([x0, 2.45, 0], aligned_edge=m.LEFT)

        title = title_text("Isolation over compression", 56, AZURE).move_to([0, 0.0, 0])
        src = cite("Li, AI Agents in Depth · ch. 2").next_to(title, m.DOWN, buff=0.22)
        book = m.VGroup(
            text("the book's example, delegated:", 22, MUTED),
            code("The function is handle_callback in src/payment/callbacks.py …", 20, MINT),
        ).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        bbox = card(book.width + 0.7, book.height + 0.5, EDGE, CARD, radius=0.16, stroke=2)
        book = m.VGroup(bbox, book.move_to(bbox)).move_to([0, -2.3, 0])

        def side(head, accent, fill, lines, at):
            h = title_text(head, 28, accent)
            body = m.VGroup(*[text(s, 22, INK) for s in lines]).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            inner = m.VGroup(h, body).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
            box = card(6.0, inner.height + 0.55, accent, fill, radius=0.18, stroke=2)
            inner.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            return m.VGroup(box, inner).move_to(at)

        comp = side("compaction · ep 11", SUN, CARD, ["cleans up after the fact", "lossy", "extra model calls"], [-3.25, -2.3, 0])
        iso = side("isolation · subagent", AZURE, CARD, ["noise never gets in", "nothing to summarize later", "Otto's cached prefix untouched"], [3.25, -2.3, 0])

        self.say(
            "[Compare Otto's context both ways.] [Searching itself added a hundred and twenty messages], [about thirty-three thousand tokens.]",
            m.AnimationGroup(self.new_header("04 · isolation", "Keep the noise out"), m.FadeIn(caption)),
            m.FadeIn(names[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(bar1, m.LEFT), m.FadeIn(v1, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Delegating added two messages], [just a hundred and thirty-five tokens.]",
            m.FadeIn(names[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(bar2, m.LEFT), m.FadeIn(v2, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The book AI Agents in Depth calls this isolation over compression.]",
            m.AnimationGroup(m.FadeIn(title, scale=0.9), m.FadeIn(src)),
        )
        self.say(
            "[Its own example ends the same way]: [two messages, and one line naming the function and its file.]",
            m.FadeIn(book, shift=0.15 * m.UP),
            m.Circumscribe(book[1][1], color=MINT, buff=0.06),
        )
        self.say(
            "[Compaction, from episode eleven, cleans up after the fact]: [it's lossy], [and it costs extra model calls.]",
            m.AnimationGroup(m.FadeOut(book, shift=0.15 * m.DOWN), m.FadeIn(comp[0]), m.FadeIn(comp[1][0]), m.FadeIn(comp[1][1][0])),
            m.FadeIn(comp[1][1][1], shift=0.1 * m.UP),
            m.FadeIn(comp[1][1][2], shift=0.1 * m.UP),
        )
        self.say(
            "[Isolation keeps the noise out from the start], [and Otto's cached prefix is never touched.]",
            m.AnimationGroup(m.FadeIn(iso[0]), m.FadeIn(iso[1][0]), m.FadeIn(iso[1][1][0]), m.FadeIn(iso[1][1][1])),
            m.FadeIn(iso[1][1][2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ limits

    def limits(self):
        self.next_section("5 · Limits", notes="Li, ch. 2: the cost of isolation is that 'the task description must be self-contained and the goal must be clear', because the sub-agent does not see the main agent's context. Learn Claude Code s06: 'Conversation context is discarded, but file system side effects… remain'; sub-agent tool calls still pass the permission hooks — 'Context isolation does not mean permission isolation' (Episode 17: permissions). The vague description and the edited file are illustrative.")
        bad = description_card([VAGUE], 6.0, ROSE, "vague")
        good = description_card(wrap(DESCRIPTION.split(". ")[0] + ". Reply with file, function …", 40), 6.0, MINT, "self-contained")
        bad.move_to([-3.3, 0, 0]).align_to([0, 2.4, 0], m.UP)
        good.move_to([3.3, 0, 0]).align_to([0, 2.4, 0], m.UP)
        bad_x = cross(bad[0].get_corner(m.UR) + np.array([-0.4, -0.38, 0]), 0.28)
        good_t = tick(good[0].get_corner(m.UR) + np.array([-0.4, -0.38, 0]), 0.36)
        blind = text("the child never saw “the thing”", 22, MUTED).next_to(bad, m.DOWN, buff=0.2).align_to(bad, m.LEFT)

        # side effects: a child edits a file, its context goes, the edit stays
        tree = file_tree(["tinyshop/", "tinyshop/cart.py", "tinyshop/checkout.py", "tinyshop/pricing.py"], size=22)
        tbox = card(4.3, tree.height + 0.6, EDGE, CARD, radius=0.16, stroke=2)
        tree.move_to(tbox).align_to(tbox, m.LEFT).shift(0.3 * m.RIGHT)
        files = m.VGroup(tbox, tree).move_to([-4.35, -1.6, 0])
        kid = m.VGroup(otto(0.42, AZURE, AZURE_FILL), code("child", 20, AZURE)).arrange(m.DOWN, buff=0.1).move_to([-1.15, -1.3, 0])
        target = tree[2]
        edit = arrow(kid[0].get_left(), target.get_right() + 0.1 * m.RIGHT, SUN, buff=0.12)
        edited = code("edited", 20, SUN, weight=m.BOLD).next_to(target, m.RIGHT, buff=0.2)
        stays = text("the edit stays", 22, MINT).next_to(files, m.DOWN, buff=0.2)

        # permissions: the same checks
        gate = m.VGroup(icon(LOCK, 0.62), m.VGroup(text("same permission checks", 24, INK, weight=m.SEMIBOLD), text("as Otto's own · Episode 17", 22, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)).arrange(m.RIGHT, buff=0.3)
        gbox = card(gate.width + 0.8, gate.height + 0.6, MINT, CARD, radius=0.16, stroke=2)
        gate = m.VGroup(gbox, gate.move_to(gbox)).move_to([3.4, -1.35, 0])
        quote = m.VGroup(text("“Context isolation does not mean", 22, INK), text("permission isolation.”", 22, INK)).arrange(m.DOWN, buff=0.1)
        quote.next_to(gate, m.DOWN, buff=0.35)
        src = cite("learn-claude-code · s06").next_to(quote, m.DOWN, buff=0.15)

        self.say(
            "[Isolation has a price.] [The child sees none of Otto's conversation], [so the description must stand on its own.]",
            m.AnimationGroup(self.new_header("05 · limits", "What isolation doesn't cover"), m.FadeIn(bad, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Create(bad_x), m.FadeIn(blind)),
            m.AnimationGroup(m.FadeIn(good, shift=0.15 * m.UP), m.Create(good_t)),
        )
        self.say(
            "[And it isolates the conversation], [not the world.] [If a child edits a file], [the edit is still there after its context is gone.]",
            m.AnimationGroup(m.FadeIn(files, shift=0.15 * m.UP), m.FadeIn(kid, shift=0.15 * m.UP)),
            m.Circumscribe(files, color=MUTED, buff=0.06),
            m.AnimationGroup(m.GrowArrow(edit), target.animate.set_color(SUN), m.FadeIn(edited, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeOut(kid, scale=0.6), m.FadeOut(edit), m.FadeIn(stays, shift=0.1 * m.UP)),
        )
        self.say(
            "[Its tool calls pass the same permission checks as Otto's.] [Context isolation is not permission isolation.]",
            m.FadeIn(gate, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ forks

    def forks(self):
        self.next_section("6 · Forks", notes="Claude Code's fork mode (Learn Claude Code s06, reading CC's source; Claude Code docs: a fork 'inherits the entire conversation so far' with 'the same system prompt, tools, model, and message history'): the child inherits the parent's prefix; 'The goal isn't isolation, but making the Anthropic API's prompt cache hit'. Byte-identical on five things: system prompt, tools, model, message prefix, thinking config (s06; Li, ch. 2: sub-agents that inherit context 'must be byte-aligned with the parent Agent'). Episode 7: prompt caching. The request strips are a simplified picture (segment order is illustrative).")
        specs = [("model", CORAL, CORAL_FILL, 1.2), ("thinking", CORAL, CORAL_FILL, 1.45), ("system prompt", ORCHID, ORCHID_FILL, 1.95), ("tools", SUN, SUN_FILL, 1.05)]

        def strip():
            segs = [segment(lab, w, c, f) for lab, c, f, w in specs] + [messages_segment(2.6)]
            return m.VGroup(*segs).arrange(m.RIGHT, buff=0.1)

        x0 = -5.05
        top = strip()
        top.move_to([x0 + top.width / 2, 1.55, 0])
        low = strip()
        low.move_to([x0 + low.width / 2, -0.35, 0])
        task_seg = segment("fork's task", 1.75, AZURE, AZURE_FILL).next_to(low, m.RIGHT, buff=0.1)
        assert task_seg.get_right()[0] < 6.55
        lab_top = m.VGroup(otto(0.36), text("Otto", 24, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.1).move_to([-5.85, 1.55, 0])
        lab_low = m.VGroup(otto(0.36, AZURE, AZURE_FILL), text("fork", 24, AZURE, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.1).move_to([-5.85, -0.35, 0])
        eqs = m.VGroup(*[equals([a.get_x(), 0.6, 0], MINT) for a in top])
        span = m.VGroup(*low)
        hit_box = m.RoundedRectangle(corner_radius=0.18, width=span.width + 0.24, height=span.height + 0.24, stroke_color=MINT, stroke_width=3).move_to(span)
        hit = m.VGroup(text("cache hit", 26, MINT, weight=m.SEMIBOLD), pill("Episode 7 · prompt caching", SUN, SUN_FILL, 20)).arrange(m.RIGHT, buff=0.4)
        hit.next_to(hit_box, m.DOWN, buff=0.35).align_to(hit_box, m.LEFT)
        goal = text("the goal isn't isolation", 24, MUTED).align_to(task_seg, m.RIGHT)
        sit(goal, hit[0][0].get_bottom()[1])
        src = cite("learn-claude-code · s06, reading Claude Code's source").move_to([0, -3.35, 0])
        other = segment("model B", 1.2, ROSE, ROSE_FILL, size=20).move_to(low[0])
        neq = equals(eqs[0].get_center(), ROSE, slash=True)
        miss = text("cache miss: the whole prefix, at full price", 26, ROSE, weight=m.SEMIBOLD).move_to([0, -2.45, 0])

        self.say(
            "[Claude Code has a second kind of subagent]: [a fork.]",
            m.AnimationGroup(self.new_header("06 · forks", "Forks share the cache"), m.FadeIn(lab_top), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.RIGHT) for s in top], lag_ratio=0.12)),
            m.FadeIn(lab_low, shift=0.15 * m.UP),
        )
        self.say(
            "[A fork doesn't start empty.] [It copies Otto's whole conversation], [then adds its own task at the end.]",
            m.Indicate(lab_low[0], color=AZURE, scale_factor=1.08),
            m.LaggedStart(*[m.FadeIn(b, target_position=a) for a, b in zip(top, low)], lag_ratio=0.1),
            m.FadeIn(task_seg, shift=0.2 * m.LEFT),
        )
        self.say(
            "[The point isn't isolation.] [It's to reuse the prompt cache], [from episode seven.]",
            m.FadeIn(goal, shift=0.1 * m.UP),
            m.AnimationGroup(m.Create(hit_box), m.FadeIn(hit[0], shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(hit[1], shift=0.1 * m.UP),
        )
        self.say(
            "[That only works if five things match, byte for byte]: [the model], [the thinking settings], [the system prompt], [the tools], [and the message prefix.]",
            m.Circumscribe(m.VGroup(top, low), color=MINT, buff=0.1),
            *[m.AnimationGroup(m.FadeIn(e, scale=0.6), m.Circumscribe(m.VGroup(a, b), color=MINT, buff=0.05)) for e, a, b in zip(eqs, top, low)],
        )
        self.say(
            "[Change any one of them], [and the fork pays for the whole prefix again.]",
            m.AnimationGroup(m.FadeOut(low[0]), m.FadeIn(other), m.FadeOut(eqs[0]), m.FadeIn(neq)),
            m.AnimationGroup(hit_box.animate.set_stroke(ROSE), m.FadeOut(hit, shift=0.1 * m.DOWN), m.FadeIn(miss, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the OS analogy

    def analogy(self):
        self.next_section("7 · An operating system", notes="Li, AI Agents in Depth, ch. 10, Table 10-2 (multi-agent collaboration without shared context): fork → spawn_subagent, kill → cancel_subagent, ps → list_agents, exit code and wait() → the structured summary the sub-agent returns ('sub-agents return structured summaries rather than full trajectories').")
        pairs = [("fork", "spawn_subagent"), ("kill", "cancel_subagent"), ("ps", "list_agents"), ("exit code", None)]
        h_os = text("operating system", 26, MUTED, weight=m.SEMIBOLD).move_to([-3.4, 2.15, 0])
        h_ag = text("subagents", 26, AZURE, weight=m.SEMIBOLD).move_to([2.6, 2.15, 0])
        rows = m.VGroup()
        for i, (os_name, ag) in enumerate(pairs):
            yy = 1.25 - 0.95 * i
            a = code(os_name, 28, INK, weight=m.BOLD).move_to([-3.4, yy, 0])
            b = code(ag, 28, AZURE) if ag else text("its structured report", 28, MINT, weight=m.SEMIBOLD)
            b.move_to([2.6, yy, 0])
            sit(b, a[0].get_bottom()[1]) if ag else None
            link = arrow([-1.55, yy, 0], [0.35, yy, 0], MUTED, buff=0.0)
            band = card(11.6, 0.78, EDGE, CARD, radius=0.14, stroke=1.5).move_to([-0.4, yy, 0]).set_z_index(-1)
            rows.add(m.VGroup(band, a, link, b))
        src = cite("Li, AI Agents in Depth · ch. 10").move_to([0, -2.85, 0])
        nothist = text("not the whole history", 24, MUTED).next_to(rows[3][3], m.DOWN, buff=0.3)
        nothist.move_to([rows[3][3].get_x(), -2.2, 0])

        self.say(
            "[The book compares all this to an operating system.] [Spawning a subagent is fork], [cancelling one is kill], [and listing them is the process list.]",
            m.AnimationGroup(self.new_header("07 · analogy", "Subagents are processes"), m.FadeIn(h_os), m.FadeIn(h_ag), m.FadeIn(src)),
            m.FadeIn(rows[0], shift=0.15 * m.UP),
            m.FadeIn(rows[1], shift=0.15 * m.UP),
            m.FadeIn(rows[2], shift=0.15 * m.UP),
        )
        self.say(
            "[And the report is the exit code]: [a small, structured result], [not the whole history.]",
            m.FadeIn(rows[3], shift=0.15 * m.UP),
            m.Circumscribe(rows[3][3], color=MINT, buff=0.08),
            m.FadeIn(nothist, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Spawn a child with a self-contained description, isolate its work in a fresh context, take back only its report.")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Spawn", AZURE), ("Isolate", SUN), ("Report", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("task(description)", "a fresh messages list", "only the final text")])
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
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Keep the digging in the child, and only the answer in Otto's context.", 28, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [spawn], [isolate], [report.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Keep the digging in the child], [and only the answer in Otto's context.]", m.Write(tagline), m.Circumscribe(words[2], color=MINT, buff=0.1))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)


def tokens_of_sum(messages):
    return sum(tokens_of(msg) for msg in messages)
