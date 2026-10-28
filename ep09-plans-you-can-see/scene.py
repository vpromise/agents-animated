"""Agents, Animated · Episode 9 · Plans You Can See (Module 2 · Context Engineering)

Why long tasks need a visible plan and a status bar. Running example: Otto adds a discount code to
tinyshop in five todos. The runs are not drawn by hand: the scene runs a plain loop (the drift run) and
the loop with Learn Claude Code's nag (the todo run) against scripted stand-ins for the model and
in-memory tools, and every number on screen comes from those runs or from the cited sources (asserted).

Sections: hook (long tasks drift) · TodoWrite (a tool that does no work; three states) · the todo run
(five todos ticking) · the nag (three rounds without an update → an appended reminder; Claude Code's real
nudge is different) · the agent status bar (Li, ch. 2: phone status bar, retrieval-like ICL, the Xfinity
example computed in code, Experiment 2-8, 15 vs 21 iterations, two warnings) · placement (append at the
end; never edit the system prompt) · recap.

Sources (facts dated October 2026):
- Learn Claude Code, s05 (learn.shareai.run/en/s05/): "after steps 1-3, the Agent starts improvising
  because steps 4-10 have been pushed out of its attention"; tool results dilute the system prompt;
  todo_write "does no actual work"; items {content, status: pending | in_progress | completed}, printed
  as "[ ]", "[▸]", "[✓]"; returns "Updated N tasks"; the nag: if rounds_since_todo >= 3, append
  {"role": "user", "content": "<reminder>Update your todos.</reminder>"} — a teaching mechanism.
  Dive into CC Source (the site's reading of Claude Code's source, not verified by us): "The CC source
  has no fixed '3 rounds' logic"; TodoWriteTool.ts "appends a verification nudge when 3+ todos are all
  completed without a verification item".
- Li, AI Agents in Depth, ch. 2 (Agent Status Bar): the phone status bar ("time, battery level, signal
  strength"); "in-context learning is more retrieval-like than reasoning-like"; the <agent_status>
  example ("'phone_call' has been invoked 3 times (Xfinity: 3 times)", "Maximum calls to Xfinity
  reached (3/3)"); Experiment 2-8 (Qwen3-0.6B: without the bar it "frequently violates the
  constraint", with it it "consistently adheres"); "Agents with TODO support complete tasks in an
  average of 15 iterations, while those without it require 21"; "Maintain the status bar with code";
  "never ask it to perform a batch count in one shot"; "models trust the status bar almost
  unconditionally"; put it at the end of the context as a user message, never edit the system message
  (it "would invalidate the cache for the entire prefix"); Claude Code's <system-reminder> is the
  append style (research notes §6).

Illustrative (for teaching): tinyshop's files, the five todos, Otto's choices in both runs, the discount
code SAVE10, the two new tests, the phone agent's rule wording, its user's request and its web searches,
the outcome wording of Experiment 2-8, Otto's own status bar fields.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

SOURCE = Path(__file__).read_text()

# ---------------------------------------------------------------------------- tinyshop, simulated

TASK = "Add a discount code to tinyshop."
CART = "def total(items):\n    s = 0\n    for item in items:\n        s += item\n    return s\n"
CHECKOUT = "from cart import total\n\n\ndef checkout(items):\n    return total(items)\n"
TESTS = "def test_total():\n    assert total([3, 4]) == 7\n"
OTHER_TESTS = 12  # tinyshop's other tests, all passing (as in Episodes 1-2)
DISCOUNT = '\n\ndef apply_discount(amount, code):\n    if code == "SAVE10":\n        return round(amount * 0.9, 2)\n    return amount\n'
CHECKOUT_NEW = "from cart import total, apply_discount\n\n\ndef checkout(items, code=None):\n    return apply_discount(total(items), code)\n"
CHECKOUT_RENAMED = CHECKOUT_NEW.replace("items", "basket")
NEW_TESTS = '\n\ndef test_save10():\n    assert checkout([12, 8], "SAVE10") == 18\n\n\ndef test_no_code():\n    assert checkout([12, 8]) == 20\n'


def fresh_files():
    return {"cart.py": CART, "checkout.py": CHECKOUT, "tests/test_cart.py": TESTS}


FILES = fresh_files()


def pytest_q(files):
    """tinyshop's suite: the test functions in tests/test_cart.py, run for real, plus twelve that pass."""
    space = {}
    exec(files["cart.py"], space)
    exec("\n".join(line for line in files["checkout.py"].splitlines() if not line.startswith("from cart")), space)
    exec(files["tests/test_cart.py"], space)
    ok = failed = 0
    for name in [n for n in space if n.startswith("test_")]:
        try:
            space[name]()
            ok += 1
        except AssertionError:
            failed += 1
    passed = OTHER_TESTS + ok
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


def run_bash(command):
    if command == "pytest -q":
        out = pytest_q(FILES)
    elif command == "python checkout.py":
        out = ""  # the module only defines functions
    else:
        raise ValueError(command)
    return out or "(no output)"


def read_file(path):
    return FILES[path]


def edit_file(path, old_text, new_text):
    assert old_text in FILES[path], (path, old_text)
    FILES[path] = FILES[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}

# ---------------------------------------------------------------------------- TodoWrite (learn-claude-code s05)

MARKS = {"pending": "[ ]", "in_progress": "[▸]", "completed": "[✓]"}
SCREEN = []  # what run_todo_write prints (the scene's stand-in for the terminal)


def print(*parts):  # noqa: A001 — captures run_todo_write's printout instead of writing to stdout
    SCREEN.append(" ".join(parts))


CURRENT_TODOS = []


def run_todo_write(todos):
    CURRENT_TODOS[:] = todos  # just stores it
    for t in todos:
        print(MARKS[t["status"]], t["content"])
    return f"Updated {len(todos)} tasks"


TOOL_HANDLERS["todo_write"] = run_todo_write

TODO_SRC = '''CURRENT_TODOS = []

def run_todo_write(todos):
    CURRENT_TODOS[:] = todos  # just stores it
    for t in todos:
        print(MARKS[t["status"]], t["content"])
    return f"Updated {len(todos)} tasks"

TOOL_HANDLERS["todo_write"] = run_todo_write'''
assert all(line.strip() in SOURCE for line in TODO_SRC.splitlines())  # the code on screen is the code that runs

P, I, C = "pending", "in_progress", "completed"
TODO_TEXT = ["read cart and checkout", "add apply_discount()", "use it in checkout", "write discount tests", "run pytest -q"]


def todos(*states):
    return [{"content": c, "status": s} for c, s in zip(TODO_TEXT, states)]


# ---------------------------------------------------------------------------- the model (scripted) and the loops

REMINDER = "<reminder>Update your todos.</reminder>"
FINAL = "Done. SAVE10 works; 15 tests pass."
DRIFT_FINAL = "Done. SAVE10 now works."

EDIT_CART = {"path": "cart.py", "old_text": "    return s\n", "new_text": "    return s\n" + DISCOUNT}
EDIT_CHECKOUT = {"path": "checkout.py", "old_text": CHECKOUT, "new_text": CHECKOUT_NEW}

# Without a plan (Otto as in Episode 4: bash, read_file, edit_file): three steps, an unplanned rename, done.
DRIFT_PLAN = [
    ("read_file", {"path": "cart.py"}),
    ("edit_file", EDIT_CART),
    ("edit_file", EDIT_CHECKOUT),
    ("edit_file", {"path": "checkout.py", "old_text": CHECKOUT_NEW, "new_text": CHECKOUT_RENAMED}),
    DRIFT_FINAL,
]
# With todo_write and the nag (illustrative choices of the model).
PLAN = [
    ("todo_write", {"todos": todos(I, P, P, P, P)}),
    ("read_file", {"path": "cart.py"}),
    ("read_file", {"path": "checkout.py"}),
    ("todo_write", {"todos": todos(C, I, P, P, P)}),
    ("edit_file", EDIT_CART),
    ("todo_write", {"todos": todos(C, C, I, P, P)}),
    ("edit_file", EDIT_CHECKOUT),
    ("read_file", {"path": "checkout.py"}),
    ("bash", {"command": "python checkout.py"}),
    ("todo_write", {"todos": todos(C, C, C, I, P)}),  # the answer to the reminder
    ("edit_file", {"path": "tests/test_cart.py", "old_text": TESTS, "new_text": TESTS + NEW_TESTS}),
    ("todo_write", {"todos": todos(C, C, C, C, I)}),
    ("bash", {"command": "pytest -q"}),
    ("todo_write", {"todos": todos(C, C, C, C, C)}),
    FINAL,
]


class ScriptedModel:
    """Stands in for client.messages: plays a plan, records how many messages each call was sent."""

    def __init__(self, plan):
        self.plan, self.sent = plan, []

    def create(self, messages, tools):
        self.sent.append(len(messages))
        k = len(self.sent) - 1
        step = self.plan[k]
        if isinstance(step, str):
            return NS(stop_reason="end_turn", content=[NS(type="text", text=step)])
        name, args = step
        assert name in [t["name"] for t in tools]
        block = NS(type="tool_use", id=f"toolu_{k + 1:02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


def tool_results(response):
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output = TOOL_HANDLERS[block.name](**block.input)
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
    return results


def tool_names(response):
    return [b.name for b in response.content if b.type == "tool_use"]


def plain_loop(messages):
    """The Episode 2 loop: no plan, no nag."""
    while True:
        response = client.messages.create(messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        messages.append({"role": "user", "content": tool_results(response)})


TRACE = []  # rounds_since_todo after each tool round


def agent_loop(messages):
    rounds_since_todo = 0
    while True:
        if rounds_since_todo >= 3 and messages:
            messages.append({"role": "user",
                "content": REMINDER})
            rounds_since_todo = 0
        response = client.messages.create(messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = tool_results(response)
        messages.append({"role": "user", "content": results})
        wrote = "todo_write" in tool_names(response)
        rounds_since_todo = 0 if wrote else rounds_since_todo + 1
        TRACE.append(rounds_since_todo)


NAG_SRC = '''rounds_since_todo = 0
while True:
    if rounds_since_todo >= 3 and messages:
        messages.append({"role": "user",
            "content": REMINDER})
        rounds_since_todo = 0
    response = client.messages.create(...)
    ...  # append the reply, run its tools (Episode 2)
    wrote = "todo_write" in tool_names(response)
    rounds_since_todo = 0 if wrote else rounds_since_todo + 1'''
_ran = inspect.getsource(agent_loop)
assert len(NAG_SRC.splitlines()) == 10
assert all(line.strip() in _ran for line in NAG_SRC.splitlines() if "..." not in line)

# the drift run
TOOLS = [{"name": n} for n in ("bash", "read_file", "edit_file")]
client = NS(messages=ScriptedModel(DRIFT_PLAN))
DRIFT = [{"role": "user", "content": TASK}]
plain_loop(DRIFT)
assert len(DRIFT) == 10  # the task, four calls and their results, the answer: all fit the column
assert FILES["tests/test_cart.py"] == TESTS  # "the tests were never written"
assert not any(b.name == "bash" for msg in DRIFT[1:-1:2] for b in msg["content"])  # "... and never run"
assert pytest_q(FILES) == "13 passed"  # nothing broke, nothing was tested

# the todo run
FILES.clear()
FILES.update(fresh_files())
TOOLS = [{"name": n} for n in ("bash", "read_file", "edit_file", "todo_write")]
client = NS(messages=ScriptedModel(PLAN))
MESSAGES = [{"role": "user", "content": TASK}]
ANSWER = agent_loop(MESSAGES)
SENT = client.messages.sent

CALLS = [msg["content"][0] for msg in MESSAGES if msg["role"] == "assistant"]
assert TRACE == [0, 1, 2, 0, 1, 0, 1, 2, 3, 0, 1, 0, 1, 0]  # "three rounds without an update"
R_AT = [i for i, msg in enumerate(MESSAGES) if msg["content"] == REMINDER]
assert len(R_AT) == 1 and R_AT[0] == 19  # appended once, after call nine's results
assert MESSAGES[20]["content"][0].name == "todo_write"  # Otto answers with a fresh list
assert len(MESSAGES) == 31 and ANSWER.stop_reason == "end_turn"
UPDATED = MESSAGES[2]["content"][0]["content"]
assert UPDATED == "Updated 5 tasks" and len(TODO_TEXT) == 5  # "five todos"
PYTEST_OUT = MESSAGES[27]["content"][0]["content"]
assert MESSAGES[26]["content"][0].input == {"command": "pytest -q"} and PYTEST_OUT == "15 passed"  # "fifteen passed"
assert all(t["status"] == C for t in CURRENT_TODOS) and SCREEN[-5:] == [f"[✓] {c}" for c in TODO_TEXT]
assert SCREEN[:5] == [f"{MARKS[s]} {c}" for s, c in zip((I, P, P, P, P), TODO_TEXT)]
TODO_STATES = [b.input["todos"] for b in CALLS if b.type == "tool_use" and b.name == "todo_write"]
assert [[t["status"] for t in s].count(C) for s in TODO_STATES] == [0, 1, 2, 3, 4, 5]


def otto_status(messages):
    """Otto's status bar, computed from the trajectory (never counted by the model)."""
    blocks = [b for msg in messages if msg["role"] == "assistant" for b in msg["content"] if b.type == "tool_use"]
    plan = [b for b in blocks if b.name == "todo_write"][-1].input["todos"]
    done = sum(t["status"] == C for t in plan)
    doing = [t["content"] for t in plan if t["status"] == I]
    edits = [i for i, b in enumerate(blocks) if b.name == "edit_file"]
    tests = [i for i, b in enumerate(blocks) if b.name == "bash" and b.input["command"] == "pytest -q"]
    fresh = bool(tests) and (not edits or tests[-1] > edits[-1])
    return done, len(plan), doing[0] if doing else None, fresh


SNAP = MESSAGES[:22]  # right after Otto answered the reminder
OTTO_DONE, OTTO_ALL, OTTO_DOING, OTTO_FRESH = otto_status(SNAP)
assert (OTTO_DONE, OTTO_ALL, OTTO_DOING, OTTO_FRESH) == (3, 5, "write discount tests", False)  # "three of five", "no test run since the last edit"
OTTO_BAR = (f"<agent_status>\n- todos: {OTTO_DONE}/{OTTO_ALL} completed\n- in progress: {OTTO_DOING}\n"
            f"- tests run since last edit: {'yes' if OTTO_FRESH else 'no'}\n</agent_status>")

# ---------------------------------------------------------------------------- the book's status bar example (Li, ch. 2)

LIMIT = 3
B_RULE = "At most 3 calls per company."
B_ASK = "Lower my Xfinity bill."
B_FOLLOW = "Can you call them again to follow up?"
_B_STEPS = [("phone_call", {"to": "Xfinity"}), ("web_search", {"query": "offers"}), ("phone_call", {"to": "Xfinity"}),
            ("web_search", {"query": "offers"}), ("phone_call", {"to": "Xfinity"})]
B_TRAJ = (
    [{"role": "system", "content": B_RULE}, {"role": "user", "content": B_ASK}]
    + [{"role": "assistant", "content": [NS(type="tool_use", name=n, input=a)]} for n, a in _B_STEPS]
    + [{"role": "user", "content": B_FOLLOW}]
)


def phone_calls(messages):
    return [b.input["to"] for msg in messages if msg["role"] == "assistant"
            for b in msg["content"] if b.type == "tool_use" and b.name == "phone_call"]


def render(calls, n, limit):
    state = "reached" if n >= limit else "not reached"
    return (f"<agent_status>\n- Tool call summary: 'phone_call'\n  has been invoked {len(calls)} times\n"
            f"  (Xfinity: {n} times)\n- Constraint check: Maximum calls\n  to Xfinity {state} ({n}/{limit})\n</agent_status>")


def agent_status(messages):
    calls = phone_calls(messages)
    n = calls.count("Xfinity")
    return render(calls, n, LIMIT)


STATUS_SRC = '''def agent_status(messages):
    calls = phone_calls(messages)
    n = calls.count("Xfinity")
    return render(calls, n, LIMIT)'''
assert all(line.strip() in inspect.getsource(agent_status) for line in STATUS_SRC.splitlines())
B_STATUS = agent_status(B_TRAJ)
_flat = " ".join(B_STATUS.split())
assert "Tool call summary: 'phone_call' has been invoked 3 times (Xfinity: 3 times)" in _flat  # the book's line
assert "Constraint check: Maximum calls to Xfinity reached (3/3)" in _flat  # "three of three … limit reached"
B_ROWS_X = [i for i, msg in enumerate(B_TRAJ) if msg["role"] == "assistant" and msg["content"][0].name == "phone_call"]
assert len(B_ROWS_X) == LIMIT == 3

ITER_WITH, ITER_WITHOUT = 15, 21  # Li, ch. 2: average iterations per task with / without TODO support
assert ITER_WITH < ITER_WITHOUT

# ---------------------------------------------------------------------------- look

GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep09")
GEAR_MINT = recolored("gear", MINT, MINT_FILL, prefix="ep09")
ALERT_ROSE = recolored("alert", ROSE, ROSE_FILL, prefix="ep09")
MODEL_ICON = recolored("llm", CORAL, CORAL_FILL, prefix="ep09")

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
JSON_T2C = {r"\"[^\"\n]*\"(?=\s*:)": AZURE, r":\s*\"[^\"\n]*\"": MINT}
MARK_COLOR = {P: MUTED, I: SUN, C: MINT}

ROW_H, ROW_GAP = 0.46, 0.07
COL_C, COL_W, COL_H = (3.35, -0.7, 0), 6.9, 5.8


def cite(s):
    return text(s, 20, MUTED)


def baseline(words):
    chars = [c for c in words.text if not c.isspace()]
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, y):
    """Put a Text's baseline at y."""
    words.shift((y - baseline(words)) * m.UP)
    return words


def cells(states, size=0.22):
    """A todo list in miniature: one square per item (completed MINT, in progress SUN, pending empty)."""
    group = m.VGroup()
    for s in states:
        color, fill = {C: (MINT, MINT), I: (SUN, SUN), P: (FAINT, CARD)}[s]
        group.add(m.RoundedRectangle(corner_radius=0.05, width=size, height=size, stroke_color=color,
                                     stroke_width=2, fill_color=fill, fill_opacity=1))
    return group.arrange(m.RIGHT, buff=0.08)


def row(role, body, width, accent=None):
    """One message on one line: a stripe in the block's colour, the role, the content."""
    accent = accent or ROLES[role][0]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    if body.width > width - 2.13:
        raise ValueError(f"row body too wide ({body.width:.2f} > {width - 2.13:.2f})")
    body.move_to(box).align_to(box, m.LEFT).shift(1.98 * m.RIGHT)
    base = box.get_y() - 0.09
    sit(label, base)
    lead = body if isinstance(body, m.Text) else body[0]
    body.shift((base - baseline(lead)) * m.UP)
    if not isinstance(body, m.Text):  # a code label followed by marks: centre the marks on the letters
        body[1].set_y(base + 0.09)
    return m.VGroup(box, stripe, label, body)


def stack(role, words, width, accent=None):
    """A message whose content needs several lines (kit.message, with ready-made words)."""
    color = ROLES[role][0]
    label = code(role, 22, color, weight=m.BOLD)
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    if words.width > width - 0.5:
        raise ValueError(f"stacked body too wide ({words.width:.2f} > {width - 0.5:.2f})")
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=accent or color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def short(out, room=26):
    """A tool result on one row: a file shows its first line, cut to fit."""
    lines = out.strip().splitlines()
    if len(lines) == 1 and len(out) <= room:
        return out
    first = lines[0]
    return (first if len(first) <= room else first[:room].rstrip(" ,")) + " …"


def call_body(block):
    a = block.input
    if block.name == "todo_write":
        return m.VGroup(code("todo_write", 20, SUN), cells([t["status"] for t in a["todos"]])).arrange(m.RIGHT, buff=0.22)
    if block.name == "edit_file":
        src = f'edit_file("{a["path"].split("/")[-1]}", …)'  # tests/test_cart.py shown by its file name
    elif block.name == "read_file":
        src = f'read_file("{a["path"]}")'
    elif block.name == "bash":
        src = f'bash("{a["command"]}")'
    elif block.name == "phone_call":
        src = f'phone_call("{a["to"]}")'
    else:
        src = f'{block.name}("{a["query"]}")'
    return code(src, 20, SUN)


def reminder_msg(width):
    return stack("user", code(REMINDER, 20, AZURE), width, AZURE)


def msg_row(msg, width):
    role, content = msg["role"], msg["content"]
    if isinstance(content, str):
        if content == REMINDER:
            return reminder_msg(width)
        return row(role, text(content, 20, INK), width)
    block = content[0]
    if isinstance(block, dict):  # a tool_result block, inside a user message
        return row("user", code(short(block["content"]), 20, MINT), width, MINT)
    if block.type == "text":
        return row("assistant", text(block.text, 20, INK), width)
    return row("assistant", call_body(block), width, SUN)


def todo_words(states, size=20):
    src = "todo_write\n" + "\n".join(f"{MARKS[s]} {c}" for s, c in zip(states, TODO_TEXT))
    return code(src, size, INK, t2c={r"\[▸\]": SUN, r"\[✓\]": MINT, r"\[ \]": MUTED, r"todo_write": SUN})


def side_chip(pic, label, color, fill, width=2.5, height=0.8, size=24):
    box = card(width, height, color, fill, radius=0.18)
    body = m.VGroup(icon(pic, 0.48), text(label, size, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2).move_to(box)
    return m.VGroup(box, body)


class Slot:
    """One place on screen whose content is swapped (a counter, a caption)."""

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


class TodoPanel:
    """CURRENT_TODOS as the harness prints it: one line per item, "[ ]", "[▸]" or "[✓]" and its content."""

    def __init__(self, states, center, width, gap=0.56, size=22):
        self.size, self.gap = size, gap
        n = len(TODO_TEXT)
        self.box = card(width, 0.66 + n * gap + 0.16, EDGE, CARD, radius=0.18, stroke=2).move_to(center)
        self.title = code("CURRENT_TODOS", 20, MUTED)
        self.title.align_to(self.box, m.UL).shift(0.32 * m.RIGHT + 0.2 * m.DOWN)
        self.x0 = self.box.get_left()[0] + 0.36
        self.ys = [self.box.get_top()[1] - 0.66 - gap * (i + 0.5) for i in range(n)]
        self.states = list(states)
        self.marks, self.words = [], []
        for i, s in enumerate(states):
            mark, words = self.line(i, s)
            self.marks.append(mark)
            self.words.append(words)
        self.hl = None
        self.items = m.VGroup(*[m.VGroup(a, b) for a, b in zip(self.marks, self.words)]).set_z_index(2)
        self.frame = m.VGroup(self.box, self.title)

    def line(self, i, s):
        t = code(f"{MARKS[s]} {TODO_TEXT[i]}", self.size, INK)
        t.move_to([0, self.ys[i], 0]).align_to([self.x0, 0, 0], m.LEFT)
        sit(t, self.ys[i] - 0.09)
        k = 2 if s == P else 3  # "[ ]" draws two glyphs
        mark = m.VGroup(*t[:k]).set_color(MARK_COLOR[s])
        words = m.VGroup(*t[k:]).set_color(MUTED if s == C else INK)
        return mark.set_z_index(2), words.set_z_index(2)

    def bar(self, i):
        w = self.box.width - 0.24
        rect = m.Rectangle(width=w, height=0.46, stroke_width=0, fill_color=SUN, fill_opacity=0.13)
        rect.move_to([self.box.get_x(), self.ys[i], 0])
        tick = m.Rectangle(width=0.05, height=0.46, stroke_width=0, fill_color=SUN, fill_opacity=1)
        tick.move_to(rect.get_left() + 0.025 * m.RIGHT)
        return m.VGroup(rect, tick).set_z_index(1)

    def highlight(self):
        """The bar under the item in progress (none when nothing is)."""
        doing = [i for i, s in enumerate(self.states) if s == I]
        new = self.bar(doing[0]) if doing else None
        old, self.hl = self.hl, new
        if old is None and new is None:
            return m.Wait(0.05)
        if old is None:
            return m.FadeIn(new)
        if new is None:
            return m.FadeOut(old)
        self.hl = old
        return m.Transform(old, new)

    def to(self, i, s):
        """Item i moves to state s."""
        new_mark, _ = self.line(i, s)
        self.states[i] = s
        return m.AnimationGroup(
            m.Transform(self.marks[i], new_mark),
            self.words[i].animate.set_color(MUTED if s == C else INK),
        )


# ---------------------------------------------------------------------------- the episode


class Ep09PlansYouCanSee(Episode):
    number = 9

    def lesson(self):
        self.hook()
        self.todo_tool()
        self.ticking()
        self.nag()
        self.nag_code()
        self.status_bar()
        self.status_example()
        self.status_numbers()
        self.placement()
        self.recap()

    # helpers

    def speak(self, script, *animations):
        """say(), with every animation of an object not yet on screen marked as an introducer: a row
        pushed and then scrolled (or faded) in the same line must not show before its cue."""
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

    def pushes(self, col, rows, run_time=0.45):
        return m.Succession(*[col.push(r, run_time=run_time) for r in rows])

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook · long tasks drift", notes="Otto's job: add a discount code to tinyshop, five steps. Without a plan, every tool result lands in the context and the task gets buried; after three steps Otto improvises (renames variables) and calls the job done; the tests are never written or run (asserted from the scripted drift run). Learn Claude Code s05: 'after steps 1-3, the Agent starts improvising because steps 4-10 have been pushed out of its attention.' The run is illustrative. Facts as of October 2026.")
        bot = otto(0.85).move_to([-5.95, 1.6, 0])
        job = title_text("Five steps", 30).next_to(bot, m.RIGHT, buff=0.45).shift(0.05 * m.DOWN)
        steps_txt = ["read the code", "add a function", "use it at checkout", "write tests", "run them"]
        steps = m.VGroup()
        for k, s in enumerate(steps_txt):
            box = card(5.3, 0.56, EDGE, CARD, radius=0.14, stroke=1.5)
            num = title_text(str(k + 1), 24, SUN)
            words = text(s, 24, INK)
            num.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            words.move_to(box).align_to(box, m.LEFT).shift(0.8 * m.RIGHT)
            steps.add(m.VGroup(box, num, words))
        steps.arrange(m.DOWN, buff=0.12).move_to([-3.95, -0.75, 0])
        for st in steps:
            on_baseline(m.VGroup(st[1], st[2]))
        ticks = m.VGroup(*[title_text("✓", 28, MINT).move_to(st[0].get_right() + 0.4 * m.LEFT) for st in steps[:3]])
        crosses = m.VGroup(*[title_text("✗", 28, ROSE).move_to(st[0].get_right() + 0.4 * m.LEFT) for st in steps[3:]])
        extra = card(5.3, 0.56, ROSE, ROSE_FILL, radius=0.14, stroke=1.5)
        extra_words = text("rename variables", 24, INK)
        extra_tag = code("unplanned", 20, ROSE)
        extra.next_to(steps, m.DOWN, buff=0.3)
        extra_words.move_to(extra).align_to(extra, m.LEFT).shift(0.8 * m.RIGHT)
        extra_tag.move_to(extra).align_to(extra, m.RIGHT).shift(0.3 * m.LEFT)
        extra_words.shift((extra_tag[0].get_bottom()[1] - extra_words[0].get_bottom()[1]) * m.UP)
        plus = title_text("+", 26, ROSE).move_to(extra).align_to(extra, m.LEFT).shift(0.3 * m.RIGHT)
        extra_g = m.VGroup(extra, plus, extra_words, extra_tag)

        col = ContextColumn(center=COL_C, width=COL_W, height=COL_H, title="context", gap=ROW_GAP)
        rows = [msg_row(msg, col.width - 0.36) for msg in DRIFT]
        task_row = rows[0]
        first = col.push(task_row)

        self.speak(
            "[Today, Otto has a longer job]: [add a discount code to tinyshop.]",
            m.AnimationGroup(self.new_header("01 · drift", "Long tasks drift"), m.FadeIn(bot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(col.frame), first),
        )
        self.speak(
            "[It takes five steps]: [read the code], [add a function], [use it at checkout], [write tests], [and run them.]",
            m.FadeIn(job, shift=0.15 * m.UP),
            *[m.FadeIn(st, shift=0.15 * m.RIGHT) for st in steps],
        )
        a1 = self.pushes(col, rows[1:3])
        a2 = self.pushes(col, rows[3:7])
        self.speak(
            "[Otto gets to work.] [Every tool result lands in the context], [and the task gets buried under them.]",
            m.AnimationGroup(a1, m.FadeIn(ticks[0], scale=0.6)),
            m.AnimationGroup(a2, m.Succession(m.Wait(0.4), m.FadeIn(ticks[1], scale=0.6), m.Wait(0.3), m.FadeIn(ticks[2], scale=0.6))),
            m.AnimationGroup(task_row.animate.set_opacity(0.35), m.Circumscribe(task_row, color=AZURE, buff=0.05)),
        )
        a3 = self.pushes(col, rows[7:9])
        a4 = col.push(rows[9], run_time=0.5)
        self.speak(
            "[Three steps in, Otto starts improvising]: [it renames a few variables], [and calls the job done.]",
            m.FadeIn(extra_g, shift=0.15 * m.UP),
            a3,
            m.AnimationGroup(a4, m.Circumscribe(rows[9], color=CORAL, buff=0.05)),
        )
        self.speak(
            "[The tests were never written], [and never run.]",
            m.AnimationGroup(m.FadeIn(crosses[0], scale=0.6), steps[3].animate.set_opacity(0.45)),
            m.AnimationGroup(m.FadeIn(crosses[1], scale=0.6), steps[4].animate.set_opacity(0.45)),
        )
        quote = m.VGroup(*[text(s, 30, INK) for s in (
            "“after steps 1-3, the Agent",
            "starts improvising because",
            "steps 4-10 have been pushed",
            "out of its attention.”",
        )]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
        quote.move_to([-3.75, 0.35, 0])
        src = cite("learn-claude-code · s05").next_to(quote, m.DOWN, buff=0.4).align_to(quote, m.LEFT)
        self.speak(
            "[Learn Claude Code describes exactly this]: [after the first few steps, the agent starts improvising], [because the rest of the plan has left its attention.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bot, job, steps, ticks, crosses, extra_g), shift=0.2 * m.DOWN), m.FadeIn(src)),
            m.FadeIn(quote[:2], shift=0.15 * m.UP),
            m.FadeIn(quote[2:], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ TodoWrite

    def todo_tool(self):
        self.next_section("1 · TodoWrite", notes="Learn Claude Code s05: a todo_write tool whose input is a list of items {content, status}; status is pending, in_progress or completed, printed as '[ ]', '[▸]', '[✓]'. 'todo_write does no actual work': it stores the list (CURRENT_TODOS), prints it and returns 'Updated N tasks' (the code on screen is the code the scene runs). What helps is the act of writing: each call puts the whole plan at the end of the context again. Facts as of October 2026.")
        src = (
            '{"todos": [\n'
            f'  {{"content": "{TODO_TEXT[0]}",\n'
            '   "status": "in_progress"},\n'
            f'  {{"content": "{TODO_TEXT[1]}",\n'
            '   "status": "pending"},\n'
            '  …\n'
            ']}'
        )
        body = code(src, 20, INK, t2c=JSON_T2C)
        box = card(body.width + 0.7, body.height + 0.95, SUN, CARD, radius=0.16, stroke=2)
        cap = code("todo_write · input", 20, SUN, weight=m.BOLD)
        input_card = m.VGroup(box, body, cap)
        box.move_to([-3.45, 0.1, 0])
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.2 * m.DOWN)
        body.next_to(cap, m.DOWN, buff=0.3).align_to(cap, m.LEFT)
        contents = m.VGroup(*[glyphs(body, f'"{TODO_TEXT[k]}"') for k in (0, 1)])
        statuses = m.VGroup(glyphs(body, '"in_progress"'), glyphs(body, '"pending"'))

        state_rows = m.VGroup()
        for s in (P, I, C):
            mk = code(MARKS[s], 30, MARK_COLOR[s], weight=m.BOLD)
            name = code(s, 26, INK)
            state_rows.add(m.VGroup(mk, name).arrange(m.RIGHT, buff=0.35))
        state_rows.arrange(m.DOWN, buff=0.45, aligned_edge=m.LEFT).move_to([3.7, -0.1, 0])
        for r in state_rows:
            sit(r[1], baseline(r[1]))
        state_title = text("status", 24, MUTED).next_to(state_rows, m.UP, buff=0.5).align_to(state_rows, m.LEFT)

        self.speak(
            "[Its fix is a tool called TodoWrite.]",
            m.AnimationGroup(self.new_header("02 · todo_write", "A tool that does no work"), m.FadeIn(input_card, shift=0.2 * m.UP)),
        )
        self.speak(
            "[Its input is a list of items], [each with some content], [and a status.]",
            m.Circumscribe(body, color=SUN, buff=0.08),
            m.LaggedStart(*[m.Circumscribe(c, color=MINT, buff=0.05) for c in contents], lag_ratio=0.3),
            m.LaggedStart(*[m.Circumscribe(s, color=AZURE, buff=0.05) for s in statuses], lag_ratio=0.3),
        )
        self.speak(
            "[A status is one of three]: [pending], [in progress], [or completed.]",
            m.FadeIn(state_title, shift=0.1 * m.UP),
            *[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in state_rows],
        )
        # the code: it stores the list and prints it, nothing else
        block = code_block(TODO_SRC, width=8.0, title="run_todo_write · the handler")
        block.move_to([-2.75, -0.15, 0])
        lines = TODO_SRC.split("\n")
        body2 = block[1]
        line_glyphs, k = [], 0
        for line in lines:
            n = sum(not ch.isspace() for ch in line)
            line_glyphs.append(body2[k:k + n])
            k += n
        quote = text("“todo_write does no actual work”", 26, INK, weight=m.SEMIBOLD)
        quote.move_to([4.25, 1.55, 0])
        q_src = cite("learn-claude-code · s05").next_to(quote, m.DOWN, buff=0.18)
        nots = m.VGroup(*[pill(s, ROSE, ROSE_FILL, 22) for s in ("✗  read a file", "✗  run a command")]).arrange(m.RIGHT, buff=0.25)
        nots.next_to(q_src, m.DOWN, buff=0.45)
        term = terminal(SCREEN[:5], width=5.0, size=20, title="printed")
        term.next_to(nots, m.DOWN, buff=0.4)
        ret = code(f'returns "{UPDATED}"', 20, MINT).next_to(term, m.DOWN, buff=0.22).align_to(term, m.LEFT)

        def light(a, b=None, color=SUN):
            b = a if b is None else b
            grp = m.VGroup(*[line_glyphs[i] for i in range(a, b + 1)])
            return m.Circumscribe(grp, color=color, buff=0.07)

        self.speak(
            "[Now the odd part]: [TodoWrite does no actual work.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(input_card, state_rows, state_title), shift=0.2 * m.DOWN), m.FadeIn(block, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(q_src)),
        )
        self.speak(
            "[It can't read a file or run a command.] [It just stores the list], [prints it], [and says how many items it holds.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in nots], lag_ratio=0.3),
            light(3),
            m.AnimationGroup(light(4, 5), m.FadeIn(term, shift=0.1 * m.UP)),
            m.AnimationGroup(light(6), m.FadeIn(ret, shift=0.1 * m.UP)),
        )
        # why bother: the whole plan lands at the end of the context
        col = ContextColumn(center=COL_C, width=COL_W, height=COL_H, title="context", gap=ROW_GAP)
        history = [msg_row(msg, col.width - 0.36) for msg in DRIFT[:-1]]
        for r in history:
            col.push(r)
            r.set_opacity(0.4)
        plan_card = stack("assistant", todo_words((I, P, P, P, P)), col.width - 0.36, SUN)
        land = col.push(plan_card, run_time=0.9)
        big = m.VGroup(title_text("The whole plan,", 40), title_text("at the end.", 40, SUN)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        big.move_to([-3.9, 0.9, 0])
        bot = otto(0.9).next_to(big, m.DOWN, buff=0.7).align_to(big, m.LEFT)
        point = arrow(big[1].get_right() + 0.3 * m.RIGHT, [plan_card.get_left()[0] - 0.05, plan_card.get_y(), 0], SUN)

        self.speak(
            "[So why bother?] [Because writing the list puts the whole plan into the context again], [at the very end], [where it's newest.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(block, quote, q_src, nots, term, ret), shift=0.2 * m.DOWN),
                             m.FadeIn(col.frame), m.FadeIn(m.VGroup(*history))),
            land,
            m.AnimationGroup(m.FadeIn(big, shift=0.15 * m.UP), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(point), m.Circumscribe(plan_card, color=SUN, buff=0.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the todo run

    def build_stage(self):
        st = NS()
        st.panel = TodoPanel([t["status"] for t in TODO_STATES[0]], center=(-3.73, 0.43, 0), width=6.15)
        st.col = ContextColumn(center=COL_C, width=COL_W, height=COL_H, title="context", gap=ROW_GAP)
        st.rows = [msg_row(msg, st.col.width - 0.36) for msg in MESSAGES]
        st.label = code("rounds_since_todo =", 22, MUTED)
        st.label.move_to([0, -1.9, 0]).align_to(st.panel.box, m.LEFT).shift(0.3 * m.RIGHT)
        st.pips = m.VGroup(*[m.RoundedRectangle(corner_radius=0.05, width=0.3, height=0.3, stroke_color=FAINT, stroke_width=2,
                                                fill_color=CARD, fill_opacity=1) for _ in range(3)]).arrange(m.RIGHT, buff=0.12)
        st.pips.move_to([0, st.label.get_y(), 0]).align_to(st.panel.box, m.RIGHT).shift(0.3 * m.LEFT)
        st.count = Slot(self)
        st.bot = otto(0.55).move_to([-6.25, -2.95, 0])
        st.say = Slot(self)
        return st

    def num(self, st, n):
        words = code(str(n), 24, ROSE if n >= 3 else SUN, weight=m.BOLD)
        return words.next_to(st.label, m.RIGHT, buff=0.2).align_to(st.label, m.DOWN)

    def tick(self, st, n):
        """The counter shows n: its number swaps, n pips fill (all ROSE at three)."""
        anims = [st.count.to(self.num(st, n))]
        for k, pip in enumerate(st.pips):
            color = ROSE if n >= 3 else SUN
            if k < n:
                anims.append(pip.animate.set_fill(color, 1).set_stroke(color))
            else:
                anims.append(pip.animate.set_fill(CARD, 1).set_stroke(FAINT))
        return m.AnimationGroup(*anims)

    def caption(self, st, words, color=MUTED, fill=None):
        if fill:
            mob = pill(words, color, fill, 22)
        else:
            mob = text(words, 24, color)
        return mob.next_to(st.bot, m.RIGHT, buff=0.35)

    def ticking(self):
        self.next_section("2 · Five todos, ticking", notes=f"The todo run (scripted model, real loop with the nag, in-memory tinyshop). Call 1 writes the plan: five todos, the first in progress; the tool returns '{UPDATED}'. Each update rewrites the whole list, so the newest plan is near the end of the context. Counter values per round: {TRACE} (computed). The model's choices are illustrative. Facts as of October 2026.")
        st = self.st = self.build_stage()
        p, col, rows = st.panel, st.col, st.rows
        a0 = col.push(rows[0])
        a1 = self.pushes(col, rows[1:3])
        self.speak(
            "[Let's run it.] [Otto's first call writes the plan]: [five todos], [the first one in progress.]",
            m.AnimationGroup(self.new_header("03 · states", "Five todos, ticking"), m.FadeIn(col.frame), a0,
                             m.FadeIn(p.frame), m.FadeIn(st.bot, shift=0.1 * m.UP)),
            m.AnimationGroup(a1, m.FadeIn(st.label), m.FadeIn(st.pips), st.count.to(self.num(st, 0))),
            m.LaggedStart(*[m.FadeIn(item, shift=0.1 * m.RIGHT) for item in p.items], lag_ratio=0.15),
            m.AnimationGroup(p.highlight(), m.Circumscribe(p.items[0], color=SUN, buff=0.06)),
        )
        a2 = self.pushes(col, rows[3:7])
        a3 = self.pushes(col, rows[7:9])
        self.speak(
            "[First it reads the two files], [then marks the first todo completed], [and starts the second.]",
            m.AnimationGroup(a2, m.Succession(m.Wait(0.3), self.tick(st, 1), m.Wait(0.5), self.tick(st, 2))),
            m.AnimationGroup(a3, self.tick(st, 0), p.to(0, C)),
            m.AnimationGroup(p.to(1, I), self.hl_move(p)),
        )
        a4 = self.pushes(col, rows[9:11])
        a5 = self.pushes(col, rows[11:13])
        self.speak(
            "[Next, it adds the discount function to the cart], [and updates the list again.]",
            m.AnimationGroup(a4, self.tick(st, 1)),
            m.AnimationGroup(a5, self.tick(st, 0), p.to(1, C), m.Succession(m.Wait(0.3), m.AnimationGroup(p.to(2, I), self.hl_move(p)))),
        )
        old_plan, new_plan = rows[7], rows[11]
        self.speak(
            "[Every update rewrites the whole list], [so the latest plan is always near the end of the context.]",
            m.Circumscribe(p.box, color=SUN, buff=0.06),
            m.AnimationGroup(old_plan.animate.set_opacity(0.4), m.Circumscribe(new_plan, color=SUN, buff=0.05)),
        )
        self.wait(0.2)

    def hl_move(self, p):
        return p.highlight()

    # ------------------------------------------------------------------ the nag

    def nag(self):
        st = self.st
        p, col, rows = st.panel, st.col, st.rows
        self.next_section("3 · The nag", notes=f"Learn Claude Code s05: if rounds_since_todo >= 3, the harness appends a user message '{REMINDER}' and resets the counter (a teaching rule). Here: edit checkout, read it back, run a quick check = three rounds without an update; the reminder is appended before call ten; Otto answers with a fresh list (3 completed, 4 in progress); then tests are written and run: {PYTEST_OUT} (computed: 12 other tests + test_total + two new discount tests). Facts as of October 2026.")
        a6 = self.pushes(col, rows[13:15])
        a7 = self.pushes(col, rows[15:17])
        a8 = self.pushes(col, rows[17:19])
        self.speak(
            "[But models forget.] [Otto wires the discount into checkout], [reads the file back], [and runs a quick check.]",
            self.new_header("04 · the nag", "A reminder when the plan goes stale"),
            m.AnimationGroup(a6, self.tick(st, 1)),
            m.AnimationGroup(a7, self.tick(st, 2)),
            m.AnimationGroup(a8, self.tick(st, 3)),
        )
        self.speak(
            "[That's three rounds without an update.] [The plan is going stale.]",
            m.Circumscribe(m.VGroup(st.label, st.pips), color=ROSE, buff=0.1),
            m.AnimationGroup(p.box.animate.set_stroke(ROSE), p.title.animate.set_color(ROSE),
                             st.say.to(self.caption(st, "plan going stale", ROSE, ROSE_FILL))),
        )
        a9 = col.push(rows[19], run_time=0.6)
        harness = side_chip(GEAR_ICON, "harness", EDGE, CARD, width=2.3, height=0.7, size=22)
        harness.next_to(st.bot, m.RIGHT, buff=0.35)
        self.speak(
            "[So the harness steps in], [and appends a reminder]: [update your todos.]",
            st.say.to(harness),
            a9,
            m.AnimationGroup(self.tick(st, 0), m.Circumscribe(rows[19], color=AZURE, buff=0.05)),
        )
        a10 = self.pushes(col, rows[20:22])
        self.speak(
            "[Otto answers with a fresh list]: [step three completed], [step four in progress.]",
            m.AnimationGroup(a10, st.say.to(None), p.box.animate.set_stroke(EDGE), p.title.animate.set_color(MUTED)),
            p.to(2, C),
            m.AnimationGroup(p.to(3, I), self.hl_move(p)),
        )
        a11 = self.pushes(col, rows[22:26])
        a12 = self.pushes(col, rows[26:28])
        a13 = self.pushes(col, rows[28:31], run_time=0.45)
        passed = code(f"$ pytest -q  →  {PYTEST_OUT}", 22, INK, t2c={r"^\$": MINT, r"\d+ passed": MINT})
        passed.next_to(st.bot, m.RIGHT, buff=0.35)
        self.speak(
            "[It writes the tests and updates the list], [then runs them]: [fifteen passed.] [Last, it ticks off the final todo, and reports back.]",
            m.AnimationGroup(a11, m.Succession(m.Wait(0.5), self.tick(st, 1), m.Wait(0.4),
                                               m.AnimationGroup(self.tick(st, 0), p.to(3, C), p.to(4, I), self.hl_move(p)))),
            m.AnimationGroup(a12, self.tick(st, 1)),
            m.AnimationGroup(st.say.to(passed), m.Circumscribe(rows[27], color=MINT, buff=0.05)),
            m.AnimationGroup(a13, m.Succession(m.Wait(0.4), m.AnimationGroup(self.tick(st, 0), p.to(4, C), self.hl_move(p)))),
        )
        self.wait(0.3)
        self.clear()

    def nag_code(self):
        self.next_section("4 · The nag, in code", notes="The nag as code (excerpt of the loop the scene runs; the elided lines are Episode 2's loop body). The counter values under the rounds are the run's TRACE. Three rounds is Learn Claude Code's teaching rule. Claude Code itself (per Learn Claude Code's reading of its source, s05 'Dive into CC Source', not verified by us) has no fixed 3-round logic; TodoWriteTool 'appends a verification nudge when 3+ todos are all completed without a verification item'. Facts as of October 2026.")
        block = code_block(NAG_SRC, width=10.6, title="agent_loop(messages) · with the nag")
        block.move_to([0, 2.45 - block.height / 2, 0])
        lines = NAG_SRC.split("\n")
        body = block[1]
        lg, k = [], 0
        for line in lines:
            n = sum(not ch.isspace() for ch in line)
            lg.append(body[k:k + n])
            k += n

        def light(a, b=None, color=SUN):
            b = a if b is None else b
            return m.Circumscribe(m.VGroup(*[lg[i] for i in range(a, b + 1)]), color=color, buff=0.07)

        n = len(TRACE)
        wrote = [CALLS[i].name == "todo_write" for i in range(n)]
        boxes = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.06, width=0.4, height=0.4, stroke_color=SUN if w else EDGE, stroke_width=2,
                               fill_color=SUN_FILL if w else CARD, fill_opacity=1)
            for w in wrote
        ]).arrange(m.RIGHT, buff=0.13)
        # the reminder goes in after round nine
        gap_extra = 0.5
        for b in boxes[9:]:
            b.shift(gap_extra * m.RIGHT)
        boxes.move_to([-0.75, -2.5, 0])
        vals = m.VGroup(*[code(str(v), 22, ROSE if v >= 3 else MUTED).move_to(b.get_center() + 0.62 * m.DOWN) for v, b in zip(TRACE, boxes)])
        lead = code("rounds", 20, MUTED).next_to(boxes, m.LEFT, buff=0.35)
        lead2 = code("counter", 20, MUTED).next_to(vals, m.LEFT, buff=0.35).align_to(lead, m.RIGHT)
        sit(lead2, baseline(vals[0]))
        mark_x = (boxes[8].get_right()[0] + boxes[9].get_left()[0]) / 2
        rem = m.RoundedRectangle(corner_radius=0.06, width=0.2, height=0.4, stroke_width=0, fill_color=AZURE, fill_opacity=1)
        rem.move_to([mark_x, boxes.get_y(), 0])
        legend = m.VGroup(
            m.VGroup(boxes[0].copy().scale(0.6), code("todo_write", 20, MUTED)).arrange(m.RIGHT, buff=0.15),
            m.VGroup(rem.copy().scale(0.6), code("reminder", 20, MUTED)).arrange(m.RIGHT, buff=0.15),
        ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        legend.next_to(boxes, m.RIGHT, buff=0.45).align_to(boxes, m.UP)

        self.speak(
            "[Here's the nag, in code]: [a counter of rounds since the last update], [and at three], [one appended message.]",
            m.FadeIn(block, shift=0.2 * m.UP),
            m.AnimationGroup(light(8, 9), m.FadeIn(lead), m.FadeIn(lead2),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(b, shift=0.1 * m.UP), m.FadeIn(v)) for b, v in zip(boxes, vals)], lag_ratio=0.12)),
            m.AnimationGroup(light(2, color=ROSE), m.Circumscribe(vals[8], color=ROSE, buff=0.08)),
            m.AnimationGroup(light(3, 5, color=AZURE), m.FadeIn(rem, scale=0.5), m.FadeIn(legend)),
        )
        timeline = m.VGroup(boxes, vals, lead, lead2, rem, legend)
        teach = card(6.3, 2.75, SUN, CARD, radius=0.18, stroke=2)
        teach_t = text("Learn Claude Code · teaching rule", 24, SUN, weight=m.BOLD)
        teach_b = m.VGroup(text("3 rounds without todo_write", 26, INK),
                           m.VGroup(text("→ append", 26, INK), code("<reminder>", 22, AZURE)).arrange(m.RIGHT, buff=0.2))
        cc = card(6.3, 2.75, CORAL, CARD, radius=0.18, stroke=2)
        cc_t = text("Claude Code · its real nudge", 24, CORAL, weight=m.BOLD)
        cc_b = m.VGroup(text("3+ todos, all completed,", 26, INK), text("none a verification step", 26, INK),
                        text("→ a nudge to verify", 26, MINT))
        m.VGroup(teach, cc).arrange(m.RIGHT, buff=0.4).move_to([0, 0.05, 0])
        for box_, t_, b_ in ((teach, teach_t, teach_b), (cc, cc_t, cc_b)):
            b_.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
            m.VGroup(t_, b_).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to(box_).align_to(box_, m.LEFT).shift(0.4 * m.RIGHT)
            m.VGroup(t_, b_).align_to(box_, m.UP).shift(0.35 * m.DOWN)
        sit(teach_b[1][1], baseline(teach_b[1][0]))
        src = cite("learn-claude-code · s05, reading Claude Code's source").next_to(cc, m.DOWN, buff=0.25).align_to(cc, m.RIGHT)
        self.speak(
            "[Three rounds is Learn Claude Code's teaching rule.] [Claude Code's real nudge is different.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(block, timeline), shift=0.2 * m.DOWN), m.FadeIn(m.VGroup(teach, teach_t, teach_b), shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(m.VGroup(cc, cc_t), shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.speak(
            "[When three or more todos are all completed], [and none of them verifies the work], [it nudges the model to check.]",
            m.FadeIn(cc_b[0], shift=0.1 * m.UP),
            m.FadeIn(cc_b[1], shift=0.1 * m.UP),
            m.FadeIn(cc_b[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the status bar

    def status_bar(self):
        self.next_section("5 · The agent status bar", notes="Li, AI Agents in Depth, ch. 2 (Agent Status Bar): 'On a phone, the top of the screen displays the time, battery level, signal strength…'; the bar is 'a state summary injected by the Agent framework at the end of the context'. Why it works: 'in-context learning is more retrieval-like than reasoning-like' — 'The model is good at finding information that already exists in the context'; the book adds this does not negate chain-of-thought reasoning. Facts as of October 2026.")
        phone = card(2.5, 4.6, EDGE, "#100E0C", radius=0.32, stroke=2.5).move_to([-4.55, -0.55, 0])
        bar_y = phone.get_top()[1] - 0.38
        clock = text("9:41", 22, INK, weight=m.SEMIBOLD).move_to([phone.get_left()[0] + 0.62, bar_y, 0])
        sig = m.VGroup(*[m.Rectangle(width=0.07, height=0.08 + 0.05 * i, stroke_width=0, fill_color=INK, fill_opacity=1) for i in range(4)])
        sig.arrange(m.RIGHT, buff=0.04, aligned_edge=m.DOWN)
        batt = m.VGroup(
            m.RoundedRectangle(corner_radius=0.04, width=0.38, height=0.2, stroke_color=INK, stroke_width=2, fill_opacity=0),
            m.Rectangle(width=0.26, height=0.11, stroke_width=0, fill_color=MINT, fill_opacity=1),
        )
        batt[1].move_to(batt[0]).align_to(batt[0], m.LEFT).shift(0.05 * m.RIGHT)
        right = m.VGroup(sig, batt).arrange(m.RIGHT, buff=0.14)
        right.move_to([phone.get_right()[0] - 0.55, bar_y, 0])
        app = m.VGroup(*[m.RoundedRectangle(corner_radius=0.08, width=w, height=0.32, stroke_width=0, fill_color=CARD_HI, fill_opacity=1)
                         for w in (1.8, 1.4, 1.7, 1.1, 1.6)]).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        app.move_to(phone).shift(0.25 * m.DOWN)
        phone_g = m.VGroup(phone, clock, right, app)

        title = title_text("Agent status bar", 44, SUN).move_to([2.3, 1.75, 0])
        sub = text("the state, computed, at a glance", 26, MUTED).next_to(title, m.DOWN, buff=0.2)
        quote = m.VGroup(text("“in-context learning is more", 30, INK, weight=m.SEMIBOLD),
                         text("retrieval-like than reasoning-like”", 30, INK, weight=m.SEMIBOLD)).arrange(m.DOWN, buff=0.14)
        quote.move_to([2.3, 0.15, 0])
        q_src = cite("Li, AI Agents in Depth · ch. 2").next_to(quote, m.DOWN, buff=0.22)
        good = m.VGroup(title_text("✓", 28, MINT), text("find a fact that is in the context", 26, INK)).arrange(m.RIGHT, buff=0.3)
        bad = m.VGroup(title_text("✗", 28, ROSE), text("work it out from a long history", 26, INK)).arrange(m.RIGHT, buff=0.3)
        pair = m.VGroup(good, bad).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3).move_to([2.3, -2.25, 0])

        self.speak(
            "[A to-do list tracks the plan.] [The book AI Agents in Depth adds a second idea]: [the agent status bar.]",
            self.new_header("05 · status bar", "Hand the model conclusions"),
            m.FadeIn(phone_g, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(title, shift=0.15 * m.UP), m.FadeIn(sub)),
        )
        self.speak(
            "[Think of the top of your phone screen]: [the time], [the battery], [the signal], [all at a glance.]",
            m.Circumscribe(m.VGroup(clock, right), color=MUTED, buff=0.1),
            m.Circumscribe(clock, color=SUN, buff=0.07),
            m.Circumscribe(batt, color=SUN, buff=0.07),
            m.Circumscribe(sig, color=SUN, buff=0.07),
            m.Indicate(sub, color=INK, scale_factor=1.05),
        )
        self.speak(
            "[Why does it help?] [In the book's words], [in-context learning is more retrieval-like than reasoning-like.]",
            m.Wait(0.1),
            m.FadeIn(q_src),
            m.FadeIn(quote, shift=0.15 * m.UP),
        )
        self.speak(
            "[A model is good at finding a fact in its context], [and much less reliable at working one out from a long history.]",
            m.FadeIn(good, shift=0.1 * m.RIGHT),
            m.FadeIn(bad, shift=0.1 * m.RIGHT),
        )
        self.wait(0.3)
        self.clear()

    def status_example(self):
        self.next_section("6 · The book's example", notes="Li, ch. 2, Experiment 2-8: a phone agent (the book's author builds one that phones companies); the system prompt allows each merchant at most three calls; the agent has called Xfinity 3 times, interspersed with web searches; the user asks 'Can you call them again to follow up?'. The <agent_status> lines are the book's (wrapped), computed here by code from the trajectory. Result: a small model (Qwen3-0.6B) without the bar 'frequently violates the constraint'; with it, it 'consistently adheres'. The rule wording, the user's first request, the search queries and the outcome cards' wording are illustrative. Facts as of October 2026.")
        col = ContextColumn(center=(-3.4, -0.7, 0), width=6.8, height=5.8, title="context · a phone agent", gap=ROW_GAP)
        w = col.width - 0.36
        rows = [msg_row(msg, w) for msg in B_TRAJ[:-1]]
        follow = stack("user", text(B_FOLLOW, 20, INK), w)
        for r in rows + [follow]:
            col.push(r)
        status_row = row("user", code("<agent_status> …", 20, AZURE), w, AZURE)
        status_in = col.push(status_row, run_time=0.7)

        model = side_chip(MODEL_ICON, "model", CORAL, CORAL_FILL, width=2.3, height=0.8).move_to([2.2, 1.6, 0])
        think = text("how many calls so far?", 26, MUTED).next_to(model, m.DOWN, buff=0.3).align_to(model, m.LEFT)
        q = title_text("?", 40, CORAL).next_to(model, m.RIGHT, buff=0.3)

        block = code_block(STATUS_SRC, width=6.0, title="harness · computes the bar")
        block.move_to([3.65, 2.2 - block.height / 2, 0])
        swords = code(B_STATUS, 20, INK, t2c={r"</?agent_status>": MUTED, r"\b3 times\b": SUN, r"\(3/3\)": ROSE})
        status_card = stack("user", swords, 6.2, AZURE).move_to([3.65, 0, 0])
        status_card.align_to([0, block.get_bottom()[1] - 0.35, 0], m.UP)
        link = m.DashedLine(status_card.get_left() + 0.1 * m.LEFT, status_row.get_right() + 0.08 * m.RIGHT,
                            dash_length=0.08).set_stroke(AZURE, 2)
        tallies = m.VGroup(*[title_text(str(k + 1), 26, SUN).move_to([rows[i][0].get_right()[0] - 0.4, rows[i].get_y(), 0])
                             for k, i in enumerate(B_ROWS_X)])
        x_rows = [rows[i] for i in B_ROWS_X]

        self.speak(
            "[The book's example comes from a different agent], [one that phones companies for its user], [at most three calls each.]",
            m.FadeIn(col.frame),
            m.FadeIn(rows[1], shift=0.15 * m.UP),
            m.FadeIn(rows[0], shift=0.15 * m.UP),
        )
        self.speak(
            "[It has called Xfinity three times, between web searches.] [Then the user asks]: [can you call them again?]",
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in rows[2:]], lag_ratio=0.3),
            m.FadeIn(follow, shift=0.15 * m.UP),
            m.Circumscribe(follow, color=AZURE, buff=0.05),
        )
        self.speak(
            "[Without help, the model must count the calls itself], [scattered through the history.]",
            m.AnimationGroup(m.FadeIn(model, shift=0.1 * m.UP), m.FadeIn(q, scale=0.6), m.FadeIn(think)),
            m.LaggedStart(*[m.Circumscribe(r, color=CORAL, buff=0.04) for r in x_rows], lag_ratio=0.35),
        )
        self.speak(
            "[So the harness counts them in code], [and appends the result at the end]: [three calls to Xfinity], [limit reached.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(model, q, think), shift=0.15 * m.UP), m.FadeIn(block, shift=0.15 * m.UP),
                             m.LaggedStart(*[m.FadeIn(t, scale=0.5) for t in tallies], lag_ratio=0.4)),
            m.AnimationGroup(status_in, m.FadeIn(status_card, shift=0.15 * m.UP), m.Create(link)),
            m.Circumscribe(glyphs(swords, "3 times"), color=SUN, buff=0.06),
            m.Circumscribe(glyphs(swords, "(3/3)"), color=ROSE, buff=0.06),
        )
        no_bar = card(6.2, 1.4, ROSE, CARD, radius=0.16, stroke=2)
        no_t = text("without the bar", 22, MUTED)
        no_b = m.VGroup(title_text("✗", 26, ROSE), text("calls Xfinity a fourth time", 24, INK)).arrange(m.RIGHT, buff=0.25)
        with_bar = card(6.2, 1.4, MINT, CARD, radius=0.16, stroke=2)
        w_t = text("with the bar", 22, MUTED)
        w_b = m.VGroup(title_text("✓", 26, MINT), text("declines: the limit is reached", 24, INK)).arrange(m.RIGHT, buff=0.25)
        m.VGroup(no_bar, with_bar).arrange(m.DOWN, buff=0.35).move_to([3.65, 0.25, 0])
        for box_, t_, b_ in ((no_bar, no_t, no_b), (with_bar, w_t, w_b)):
            m.VGroup(t_, b_).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16).move_to(box_).align_to(box_, m.LEFT).shift(0.35 * m.RIGHT)
        who = code("Qwen3-0.6B", 22, INK, weight=m.BOLD)
        exp = cite("Li, AI Agents in Depth · ch. 2, Exp. 2-8")
        foot = m.VGroup(who, exp).arrange(m.DOWN, buff=0.12).next_to(with_bar, m.DOWN, buff=0.4)
        self.speak(
            "[In the book's test, a very small model often broke the limit without the bar], [and kept to it with one.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(block, status_card, link), shift=0.15 * m.DOWN),
                             m.FadeIn(m.VGroup(no_bar, no_t, no_b), shift=0.15 * m.UP), m.FadeIn(foot)),
            m.FadeIn(m.VGroup(with_bar, w_t, w_b), shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    def status_numbers(self):
        self.next_section("7 · Numbers and warnings", notes=f"Li, ch. 2: 'Agents with TODO support complete tasks in an average of {ITER_WITH} iterations, while those without it require {ITER_WITHOUT} iterations and often miss subtasks' (the book's own experiment; task set not described). Warnings: 'Maintain the status bar with code whenever possible'; 'never ask it to perform a batch count in one shot'; 'models trust the status bar almost unconditionally'. Facts as of October 2026.")
        scale = 5.4 / ITER_WITHOUT
        head = text("iterations per task, on average", 24, MUTED).move_to([-3.75, 1.9, 0])
        labels = [text("with a to-do list", 26, INK), text("without", 26, INK)]
        bars, vals = [], []
        for k, (n, color) in enumerate(((ITER_WITH, MINT), (ITER_WITHOUT, FAINT))):
            y = 0.55 - 1.45 * k
            lab = labels[k].move_to([0, y + 0.5, 0]).align_to([-6.5, 0, 0], m.LEFT)
            bar = m.RoundedRectangle(corner_radius=0.06, width=n * scale, height=0.5, stroke_width=0, fill_color=color, fill_opacity=1)
            bar.move_to([0, y - 0.05, 0]).align_to([-6.5, 0, 0], m.LEFT)
            val = title_text(str(n), 40, INK).next_to(bar, m.RIGHT, buff=0.25)
            bars.append(bar)
            vals.append(val)
        head.align_to([-6.5, 0, 0], m.LEFT)
        miss = text("and often miss subtasks", 22, MUTED).next_to(bars[1], m.DOWN, buff=0.25).align_to(bars[1], m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 2").next_to(miss, m.DOWN, buff=0.4).align_to(miss, m.LEFT)

        t1 = tile(GEAR_MINT, "Maintain it in code", "never ask the model to count", MINT, width=6.6, height=1.7)
        t2 = tile(ALERT_ROSE, "Get it right", "trusted almost unconditionally", ROSE, width=6.6, height=1.7)
        tiles = m.VGroup(t1, t2).arrange(m.DOWN, buff=0.4).move_to([3.45, -0.15, 0])
        warn = text("two warnings", 24, MUTED).next_to(tiles, m.UP, buff=0.3).align_to(tiles, m.LEFT)

        self.speak(
            "[The book reports a to-do list helps too]: [fifteen iterations per task on average], [instead of twenty-one.]",
            m.AnimationGroup(self.new_header("05 · status bar", "And a to-do list helps"), m.FadeIn(head), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(labels[0]), m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(vals[0], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(labels[1]), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(vals[1], shift=0.1 * m.LEFT), m.FadeIn(miss)),
        )
        self.speak(
            "[Two warnings.] [Keep the bar in code, never ask the model to count], [and get it right], [because models trust it almost unconditionally.]",
            m.FadeIn(warn),
            m.FadeIn(t1, shift=0.15 * m.LEFT),
            m.FadeIn(t2, shift=0.15 * m.LEFT),
            m.Circumscribe(t2[1][1][1], color=ROSE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ placement

    def placement(self):
        self.next_section("8 · Reminders go at the end", notes="Both references: put reminders and status at the END of the context, as a new message; never edit the system prompt — Li, ch. 2: modifying the system message 'would invalidate the cache for the entire prefix' (KV cache: Episode 7). Claude Code's <system-reminder> uses the append style (Li, ch. 2, per our research notes). Otto's bar is computed from the todo run right after the reminder was answered: " + OTTO_BAR.replace("\n", " ") + " Facts as of October 2026.")
        col = ContextColumn(center=(-3.35, -0.7, 0), width=6.9, height=5.8, title="context", gap=ROW_GAP)
        w = col.width - 0.36
        sys_msg = stack("system", text("You are Otto, a coding agent.", 20, INK), w)
        rows = [sys_msg] + [msg_row(msg, w) for msg in MESSAGES[:4]]
        for r in rows:
            col.push(r)
        prefix = m.VGroup(*rows)
        rem = stack("user", code("<reminder>…</reminder>", 20, AZURE), 4.2, AZURE)
        rem.move_to([3.9, sys_msg.get_y(), 0])
        status_words = code(OTTO_BAR, 20, INK, t2c={r"</?agent_status>": MUTED, r"\d/\d": SUN, r"\bno\b": ROSE})
        otto_bar = stack("user", status_words, w, AZURE)

        # where the appended message lands
        last = rows[-1]
        end_y = last.get_bottom()[1] - ROW_GAP
        rem_end = reminder_msg(w).move_to([col.panel.get_x(), 0, 0]).align_to([0, end_y, 0], m.UP)
        otto_bar.move_to([col.panel.get_x(), 0, 0]).align_to([0, end_y, 0], m.UP)
        assert otto_bar.get_bottom()[1] > col.panel.get_bottom()[1] + 0.1  # it fits without scrolling

        x_bar = col.panel.get_right()[0] + 0.3
        bracket = m.VGroup(
            m.Line([x_bar, prefix.get_top()[1], 0], [x_bar, prefix.get_bottom()[1], 0]),
            m.Line([x_bar - 0.15, prefix.get_top()[1], 0], [x_bar, prefix.get_top()[1], 0]),
            m.Line([x_bar - 0.15, prefix.get_bottom()[1], 0], [x_bar, prefix.get_bottom()[1], 0]),
        ).set_stroke(MINT, 3)
        cached = text("cached prefix", 24, MINT).next_to(bracket, m.RIGHT, buff=0.25)
        into = m.Arrow(rem.get_left() + 0.1 * m.LEFT, sys_msg.get_right() + 0.08 * m.RIGHT, buff=0, color=ROSE, stroke_width=5,
                       max_tip_length_to_length_ratio=0.25)
        cross = title_text("✗", 40, ROSE).move_to(into.get_center() + 0.42 * m.UP)
        miss = m.VGroup(text("prefix changed:", 24, ROSE), text("the whole cache is lost", 24, ROSE)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
        miss.move_to([0, -0.2, 0]).align_to([x_bar + 0.25, 0, 0], m.LEFT)
        ep7 = pill("Episode 7 · KV cache", SUN, SUN_FILL, 20, mono=True).next_to(miss, m.DOWN, buff=0.35).align_to(miss, m.LEFT)

        def read_last(target):
            label = text("read last", 24, AZURE)
            label.move_to([0, target.get_y(), 0]).align_to([x_bar + 0.55, 0, 0], m.LEFT)
            tip = m.Arrow(label.get_left() + 0.12 * m.LEFT, [target.get_right()[0] + 0.1, target.get_y(), 0], buff=0, color=AZURE,
                          stroke_width=4, max_tip_length_to_length_ratio=0.35)
            return m.VGroup(tip, label)

        newest = read_last(rem_end)
        newest_bar = read_last(otto_bar)
        cc = m.VGroup(text("Claude Code does this too:", 24, INK), code("<system-reminder>…</system-reminder>", 20, AZURE)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.15)
        cc.move_to([0, 1.75, 0]).align_to([x_bar + 0.25, 0, 0], m.LEFT)
        cc_src = cite("Li, AI Agents in Depth · ch. 2").next_to(cc, m.DOWN, buff=0.15).align_to(cc, m.LEFT)
        cached.shift((min(cached.get_y(), cc_src.get_bottom()[1] - 0.5) - cached.get_y()) * m.UP)

        self.speak(
            "[Last question: where should reminders and status bars go?]",
            m.AnimationGroup(self.new_header("06 · placement", "Reminders go at the end"), m.FadeIn(col.frame), m.FadeIn(prefix),
                             m.FadeIn(rem, shift=0.2 * m.LEFT)),
        )
        self.speak(
            "[Not into the system prompt.] [Editing the start of the context] [throws away the whole cached prefix], [as episode seven explains.]",
            m.AnimationGroup(m.GrowArrow(into), m.FadeIn(cross, scale=0.6)),
            sys_msg[0].animate.set_stroke(ROSE),
            m.AnimationGroup(m.LaggedStart(*[r[0].animate.set_stroke(ROSE) for r in rows[1:]], lag_ratio=0.15), m.FadeIn(miss, shift=0.1 * m.UP)),
            m.FadeIn(ep7, shift=0.1 * m.UP),
        )
        self.speak(
            "[Append them at the end instead], [as a new message.] [The prefix stays cached], [and the reminder sits where the model reads last.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(into, cross, miss, ep7)), *[r[0].animate.set_stroke(EDGE) for r in rows]),
            m.ReplacementTransform(rem, rem_end),
            m.AnimationGroup(m.Create(bracket), m.FadeIn(cached, shift=0.1 * m.LEFT)),
            m.FadeIn(newest, shift=0.1 * m.LEFT),
        )
        self.speak(
            "[Claude Code works this way too], [with its system reminder tags.]",
            m.FadeIn(cc[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(cc[1], shift=0.1 * m.UP), m.FadeIn(cc_src)),
        )
        bar_lines = status_words.text.split("\n")
        self.speak(
            "[For Otto, the bar could read]: [three of five todos done], [and no test run since the last edit.]",
            m.AnimationGroup(m.FadeOut(rem_end), m.FadeIn(otto_bar), m.FadeOut(newest), m.FadeIn(newest_bar)),
            m.Circumscribe(glyphs(status_words, bar_lines[1].strip()), color=SUN, buff=0.06),
            m.Circumscribe(glyphs(status_words, bar_lines[3].strip()), color=ROSE, buff=0.06),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Plan (todo_write), remind (the nag, appended at the end), compute (a status bar maintained by code).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Plan", SUN), ("Remind", AZURE), ("Compute", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("todo_write", "<reminder>", "<agent_status>")])
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
        tagline = text("Keep the plan in view. Hand the model conclusions, at the end.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.speak(
            "[To sum up]: [plan], [remind], [compute.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.speak("[Keep the plan in view], [and hand the model conclusions, at the end of its context.]",
                   m.Write(tagline), m.Circumscribe(row_, color=SUN, buff=0.2))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
