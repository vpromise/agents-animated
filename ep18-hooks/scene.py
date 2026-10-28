"""Agents, Animated · Episode 18 · Hooks (Module 4 · Tools & Safety)

How to add behaviour around the loop without changing it. Running example: running the tests after
every edit Otto makes. The run is not drawn by hand: the scene runs Episode 2's loop with Learn Claude
Code's four hook points (s04) against a scripted stand-in for the model and an in-memory tinyshop, with
four real hook functions registered (a formatter and the tests after every edit, a guard on tests/, and
a stop hook that wants the tests green). Every number on screen comes from that run (asserted below).

Sections: hook (tests after every edit: a prompt can forget, a loop full of ifs is a tangle) · the four
events clipped onto the loop map, HOOKS + trigger_hooks · observe: PostToolUse formatter + tests after
Otto's first (wrong) edit · block: a PreToolUse hook refuses an edit to tests/, the reason becomes the
tool_result; the loop lines · permissions: Episode 17's check as a PreToolUse hook; a hook's allow cannot
beat deny/ask rules · stop: the stop hook appends a user message and the loop goes on, until the tests
pass · the stop-hook death spiral and its fixes · Claude Code's events (27 in Learn Claude Code's reading,
33 in the docs) · recap.

Sources (facts dated October 2026):
- Learn Claude Code, s04 "Hooks — Hang on the Loop, Don't Write into It" (learn.shareai.run/en/s04/):
  HOOKS = {UserPromptSubmit, PreToolUse, PostToolUse, Stop}; register_hook; trigger_hooks returns the
  first non-None result ('return value ≠ None → hook says "stop"'); UserPromptSubmit after input and
  before the model; PreToolUse after a tool_use block and before its handler, a non-None result becomes
  the tool_result and the call is skipped; PostToolUse after the handler, before the result is appended;
  Stop before the loop exits, a non-None result is appended as a user message and the loop continues;
  the permission check of s03 becomes a PreToolUse hook; "The loop stays boring on purpose: it calls
  trigger_hooks(event), and the registry decides what extra logic runs." Dive into CC Source (the site's
  reading of Claude Code's source, not verified by us): 27 hook events in 7 groups; "Hook 'allow' cannot
  bypass deny/ask rules", called the most important security design in CC's permission system.
- Claude Code docs, hooks (code.claude.com/docs/en/hooks, read October 2026): 33 hook events listed;
  PreToolUse "allow" skips the permission prompt, and "Deny and ask rules are still evaluated regardless
  of what the hook returns"; "The stop_hook_active field is true when Claude Code is already continuing as
  a result of a stop hook. Check this value … to avoid blocking on a condition that will never resolve."
- Li, AI Agents in Depth, ch. 5 (Failure and Error Recovery, "Termination"): the death spiral — a
  context-overflow stop fires a stop hook that commits code and asks the LLM for a commit message, which
  overflows again and fires the hook again; fixes: disable model-invoking side effects on the error path,
  and a recursion-depth counter.
- Li, ch. 5 "constraints over guidance" (Episode 8 moved a rule from the prompt into a linter check).
- ruff 0.17.1, measured: `ruff format` turns `range(1,len(items))` into `range(1, len(items))` and prints
  "1 file reformatted"; on a formatted file it prints "1 file left unchanged".

Illustrative (for teaching): tinyshop and its bug (harness_reference.py), Otto's choices in the run (a
wrong first fix, the attempt to edit the test, giving up, re-reading, the right fix), the hook function
names and messages, the tangled loop, the rm -rf call and its deny rule, the round counter of the spiral.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- tinyshop, in memory

TASK = "The checkout test fails. Fix it."
CART = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s\n"
TEST = "def test_total():\n    assert total([3, 4]) == 7\n"
OTHER_TESTS = 12  # tinyshop's other tests, all passing (illustrative)
FILES = {"cart.py": CART, "tests/test_cart.py": TEST}
SHELL = []  # (command, output) of every command the hooks ran


def pytest_q():
    """tinyshop's suite: test_total (total([3, 4]) == the number in the test file) plus twelve that pass."""
    space = {}
    exec(FILES["cart.py"], space)
    want = int(re.search(r"== (\d+)", FILES["tests/test_cart.py"]).group(1))
    ok = space["total"]([3, 4]) == want
    passed, failed = OTHER_TESTS + ok, int(not ok)
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


def ruff_format(path):
    """ruff format, for this file (measured with ruff 0.17.1): a space after each comma, and its message."""
    before = FILES[path]
    FILES[path] = re.sub(r",(?=\S)", ", ", before)
    return "1 file reformatted" if FILES[path] != before else "1 file left unchanged"


def run_bash(command):
    if command == "pytest -q":
        out = pytest_q()
    elif command.startswith("ruff format "):
        out = ruff_format(command.split()[-1])
    else:
        raise ValueError(command)
    SHELL.append((command, out))
    return out


def read_file(path):
    return FILES[path]


def edit_file(path, old_text, new_text):
    FILES[path] = FILES[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOLS = [{"name": "read_file"}, {"name": "edit_file"}]
TOOL_HANDLERS = {"read_file": read_file, "edit_file": edit_file}


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


# ---------------------------------------------------------------------------- hooks (learn-claude-code s04)

HOOKS = {"UserPromptSubmit": [],
         "PreToolUse": [],
         "PostToolUse": [],
         "Stop": []}


def register_hook(event, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def format_file(block, output):
    if block.name == "edit_file":
        run_bash("ruff format " + block.input["path"])


def run_tests(block, output):
    if block.name == "edit_file":
        return run_bash("pytest -q")


def protect_tests(block):
    path = block.input.get("path", "")
    if path.startswith("tests/"):
        return "tests/ is read-only"


def tests_must_pass(messages):
    out = run_bash("pytest -q")
    if "failed" in out:
        return "Tests fail: " + out


FIRED = []  # a logging hook on every event (it only observes: it returns None)


def logger(event):
    def log(*args):
        FIRED.append(event)
    return log


for _event in HOOKS:
    register_hook(_event, logger(_event))
register_hook("PreToolUse", protect_tests)
register_hook("PostToolUse", format_file)
register_hook("PostToolUse", run_tests)
register_hook("Stop", tests_must_pass)

# ---------------------------------------------------------------------------- the model (scripted) and the loop

PLAN = [  # Otto's choices in this run (illustrative)
    ("edit_file", {"path": "cart.py", "old_text": "range(len(items) - 1)", "new_text": "range(1,len(items))"}),
    ("edit_file", {"path": "tests/test_cart.py", "old_text": "== 7", "new_text": "== 4"}),
    "The test looks wrong. Stopping.",
    ("read_file", {"path": "cart.py"}),
    ("edit_file", {"path": "cart.py", "old_text": "range(1, len(items))", "new_text": "range(len(items))"}),
    "Fixed. All 13 tests pass.",
]


class ScriptedModel:
    def __init__(self):
        self.sent = []

    def create(self, messages, tools):
        self.sent.append(len(messages))
        step = PLAN[len(self.sent) - 1]
        if isinstance(step, str):
            return NS(stop_reason="end_turn", content=[NS(type="text", text=step)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{len(self.sent):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=ScriptedModel())


def agent_loop(messages):
    while True:
        response = client.messages.create(messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            force = trigger_hooks("Stop", messages)
            if force is None:
                return response
            messages.append({"role": "user",
                             "content": force})
            continue
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = trigger_hooks("PreToolUse", block)
                if output is None:  # nobody objected
                    output = run_tool(block)
                    note = trigger_hooks("PostToolUse", block, output)
                    if note: output += "\n" + note
                results.append({"type": "tool_result",
                    "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


def submit(prompt, history):
    trigger_hooks("UserPromptSubmit", prompt)
    history.append({"role": "user", "content": prompt})
    return agent_loop(history)


MESSAGES = []
ANSWER = submit(TASK, MESSAGES)
SENT = client.messages.sent


def result_of(i):
    content = MESSAGES[i]["content"]
    return content if isinstance(content, str) else content[0]["content"]


R_FIRST, R_BLOCK, FORCE, R_READ, R_FIX = result_of(2), result_of(4), result_of(6), result_of(8), result_of(10)

# ---------------------------------------------------------------------------- the code on screen (excerpts of the code that ran)

S_HOOKS = '''HOOKS = {"UserPromptSubmit": [],
         "PreToolUse": [],
         "PostToolUse": [],
         "Stop": []}'''
S_TRIGGER = '''def trigger_hooks(event, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None'''
S_POST = '''def run_tests(block, output):
    if block.name == "edit_file":
        return run_bash("pytest -q")

register_hook("PostToolUse", format_file)
register_hook("PostToolUse", run_tests)'''
S_PROTECT = '''def protect_tests(block):
    path = block.input.get("path", "")
    if path.startswith("tests/"):
        return "tests/ is read-only"

register_hook("PreToolUse", protect_tests)'''
S_TOOLS = '''for block in response.content:
    if block.type == "tool_use":
        output = trigger_hooks("PreToolUse", block)
        if output is None:  # nobody objected
            output = run_tool(block)
            note = trigger_hooks("PostToolUse", block, output)
            if note: output += "\\n" + note
        results.append({"type": "tool_result",
            "tool_use_id": block.id, "content": output})'''
S_STOPHOOK = '''def tests_must_pass(messages):
    out = run_bash("pytest -q")
    if "failed" in out:
        return "Tests fail: " + out

register_hook("Stop", tests_must_pass)'''
S_STOP = '''if response.stop_reason != "tool_use":
    force = trigger_hooks("Stop", messages)
    if force is None:
        return response
    messages.append({"role": "user",
                     "content": force})
    continue'''

# everything above this block of excerpts is the code that ran (HOOKS and register_hook calls are module-level)
_ran = Path(__file__).read_text().split("# " + "-" * 76 + " the code on screen")[0]
for _src in (S_HOOKS, S_TRIGGER, S_POST, S_PROTECT, S_TOOLS, S_STOPHOOK, S_STOP):
    assert len(_src.split("\n")) <= 10
    for _line in _src.split("\n"):
        assert _line.strip() in _ran, _line

# ---------------------------------------------------------------------------- the run, checked

_wrong = {}
exec(CART.replace("range(len(items) - 1)", "range(1, len(items))"), _wrong)
WRONG_TOTAL = _wrong["total"]([3, 4])
_buggy = {}
exec(CART, _buggy)
assert _buggy["total"]([3, 4]) == 3 and WRONG_TOTAL == 4  # the wrong fix skips the first item: 4, not 7
assert SENT == [1, 3, 5, 7, 9, 11] and len(MESSAGES) == 12  # six calls, twelve messages
assert R_FIRST == "edited cart.py\n1 failed, 12 passed"  # "one failed, twelve passed"
assert SHELL[:2] == [("ruff format cart.py", "1 file reformatted"), ("pytest -q", "1 failed, 12 passed")]
assert R_BLOCK == "tests/ is read-only" and FILES["tests/test_cart.py"] == TEST  # blocked: the test file is untouched
assert MESSAGES[5]["content"][0].text == PLAN[2]
assert MESSAGES[6] == {"role": "user", "content": "Tests fail: 1 failed, 12 passed"}  # the stop hook's message
assert "range(1, len(items))" in R_READ  # Otto reads the file the formatter changed
assert R_FIX == "edited cart.py\n13 passed" and ("ruff format cart.py", "1 file left unchanged") in SHELL  # "thirteen passed"
assert ANSWER.stop_reason == "end_turn" and ANSWER.content[0].text == PLAN[5]
assert {e: FIRED.count(e) for e in HOOKS} == {"UserPromptSubmit": 1, "PreToolUse": 4, "PostToolUse": 3, "Stop": 2}
assert "for i in range(len(items)):" in FILES["cart.py"]

# Claude Code's hook events: Learn Claude Code's reading of the source (s04), and the docs (October 2026)
CC_GROUPS = [
    ("tools", ["PreToolUse", "PostToolUse", "PostToolUseFailure"]),
    ("session", ["SessionStart", "SessionEnd", "Stop", "StopFailure", "Setup"]),
    ("user", ["UserPromptSubmit", "Notification", "PermissionRequest", "PermissionDenied"]),
    ("subagents", ["SubagentStart", "SubagentStop"]),
    ("compaction", ["PreCompact", "PostCompact"]),
    ("team", ["TeammateIdle", "TaskCreated", "TaskCompleted"]),
    ("other", ["Elicitation", "ElicitationResult", "ConfigChange", "WorktreeCreate"]),
    ("", ["WorktreeRemove", "InstructionsLoaded", "CwdChanged", "FileChanged"]),
]
CC_EVENTS = [e for _, names in CC_GROUPS for e in names]
DOCS_EXTRA = ["UserPromptExpansion", "PostToolBatch", "MessageDisplay", "DirectoryAdded", "PreModelSwitch", "PostModelSwitch"]
assert len(CC_EVENTS) == len(set(CC_EVENTS)) == 27 and set(HOOKS) <= set(CC_EVENTS)  # "twenty-seven"
assert len(set(CC_EVENTS) | set(DOCS_EXTRA)) == 33  # "thirty-three"

# ---------------------------------------------------------------------------- look

HOOK = "ep18-hook"
CODE_T2C = {
    r"#[^\n]*": FAINT,
    r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
    r"\b(def|while|for|in|if|return|True|False|None|not|and|or|is|continue)\b": ORCHID,
    r"\b\d+(\.\d+)?\b": SUN,
    r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CW = 0.154  # Menlo 20: one character's advance (measured)
EVENTS = ["UserPromptSubmit", "PreToolUse", "PostToolUse", "Stop"]
LEFT, RIGHT, TOP, BOTTOM = -6.6, 6.6, 2.6, -3.5


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def set_base(words, src, y):
    return words.shift((y - baseline(words, src.replace(" ", ""))) * m.UP)


class CodeCard:
    """kit.code_block with its lines as glyph groups, and a line cursor (a tinted bar)."""

    def __init__(self, src, title=None, width=None, accent=EDGE):
        self.group = code_block(src, width=width, title=title, accent=accent)
        self.box, self.body = self.group[0], self.group[1]
        self.body.set_z_index(2)
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n

    def split(self):
        """Each line becomes its own Text, introduced on its own (see intro): one line can then be
        faded out and replaced without the rest of the card."""
        lines = []
        for g, src in zip(self.lines, self.src):
            if not src.strip():
                lines.append(m.VGroup())
                continue
            t = code(src.strip(), 20, INK, t2c=CODE_T2C).set_z_index(2)
            t.move_to([g.get_left()[0], 0, 0], aligned_edge=m.LEFT)
            lines.append(set_base(t, src.strip(), baseline(g, src.replace(" ", ""))))
        self.lines = lines
        self.frame = m.VGroup(self.box, *self.group[2:])
        self.body = m.VGroup(*[t for t in lines if len(t)])
        self.group = m.VGroup(self.frame, self.body)
        return self

    def intro(self, shift=0.2 * m.UP):
        return m.AnimationGroup(m.FadeIn(self.frame, shift=shift), *[m.FadeIn(t, shift=shift) for t in self.body])

    def base(self, i):
        return baseline(self.lines[i], self.src[i].replace(" ", ""))

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        return self.lines[i][start - skip:start - skip + sum(not c.isspace() for c in key)]

    def bar(self, a, b=None, color=MINT):
        b = a if b is None else b
        top, bot = self.base(a) + 0.27, self.base(b) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


class Slot:
    """One place on screen whose content is swapped."""

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


def clip_card(event, example):
    """A hook as it hangs on the loop: the hook icon, the event's name, what it is used for."""
    words = m.VGroup(code(event, 20, MINT, weight=m.BOLD), text(example, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
    body = m.VGroup(icon(HOOK, 0.55), words).arrange(m.RIGHT, buff=0.22)
    box = card(body.width + 0.5, body.height + 0.32, MINT, CARD, radius=0.14, stroke=2)
    return m.VGroup(box, body.move_to(box))


def hook_tag(event):
    words = code(event, 20, MINT, weight=m.BOLD)
    body = m.VGroup(icon(HOOK, 0.34), words).arrange(m.RIGHT, buff=0.14)
    box = card(body.width + 0.45, 0.5, MINT, MINT_FILL, radius=0.25, stroke=2)
    return m.VGroup(box, body.move_to(box))


def flag(label, color, fill, size=22):
    return pill(label, color, fill, size)


# ---------------------------------------------------------------------------- the loop as a map

P0, M0, T0, A0 = (np.array(p, dtype=float) for p in ([-5.55, 0.6, 0], [-1.8, 0.6, 0], [3.6, 0.6, 0], [-1.8, -2.2, 0]))
CHIP_W, CHIP_H = 2.0, 0.8
MINI_K, MINI_AT = 0.3, np.array([4.9, 2.85, 0])


class LoopMap:
    """prompt → model ⇄ tools, model → answer; four hook points with their clips."""

    def __init__(self):
        spec = [("prompt", AZURE, AZURE_FILL, P0), ("model", CORAL, CORAL_FILL, M0),
                ("tools", SUN, SUN_FILL, T0), ("answer", CORAL, CORAL_FILL, A0)]
        self.chips = m.VGroup(*[chip(n, c, f, width=CHIP_W, height=CHIP_H, size=26).move_to(p) for n, c, f, p in spec])
        self.a_in = arrow(P0 + [CHIP_W / 2, 0, 0], M0 - [CHIP_W / 2, 0, 0], AZURE, buff=0.12)
        use = m.ArcBetweenPoints(M0 + [CHIP_W / 2 + 0.08, 0.2, 0], T0 + [-CHIP_W / 2 - 0.08, 0.2, 0], angle=-1.4)
        res = m.ArcBetweenPoints(T0 + [-CHIP_W / 2 - 0.08, -0.2, 0], M0 + [CHIP_W / 2 + 0.08, -0.2, 0], angle=-1.4)
        self.points = [
            (self.a_in.get_start() + self.a_in.get_end()) / 2,
            use.point_from_proportion(0.82),
            res.point_from_proportion(0.18),
            (M0 + A0) / 2,
        ]
        self.a_use = use.set_stroke(SUN, 5)
        self.a_use.add_tip(tip_length=0.22, tip_width=0.22)
        self.a_res = res.set_stroke(MINT, 5)
        self.a_res.add_tip(tip_length=0.22, tip_width=0.22)
        self.a_out = arrow(M0 - [0, CHIP_H / 2, 0], A0 + [0, CHIP_H / 2, 0], CORAL, buff=0.12)
        self.arrows = m.VGroup(self.a_in, self.a_use, self.a_res, self.a_out)
        mid = (M0[0] + T0[0]) / 2
        self.l_use = code("tool_use", 20, SUN).move_to([mid, 1.0, 0])
        self.l_res = code("tool_result", 20, MINT).move_to([mid, 0.2, 0])
        self.dots = m.VGroup(*[m.Dot(p, radius=0.12, color=MINT) for p in self.points])
        examples = ["log it · add context", "check it · block it", "format · run the tests", "keep it going"]
        self.clips = [clip_card(e, x) for e, x in zip(EVENTS, examples)]
        ups, pre, post, stop = self.clips
        ups.move_to([self.points[0][0], 1.95, 0])
        pre.move_to([self.points[1][0] + 0.15, 2.0, 0])
        post.move_to([self.points[2][0] + 0.15, -1.05, 0])
        stop.move_to([0, self.points[3][1], 0]).align_to([M0[0] - 0.85, 0, 0], m.RIGHT)
        ends = [ups.get_bottom(), pre.get_bottom(), post.get_top(), stop.get_right()]
        self.hangers = m.VGroup(*[m.Line(p, e).set_stroke(MINT, 3) for p, e in zip(self.points, ends)])

    def clip_on(self, i):
        """A hook clips on at point i: its card drops in, its hanger draws, its dot snaps."""
        c, h, d = self.clips[i], self.hangers[i], self.dots[i]
        away = (c.get_center() - self.points[i])
        away = 0.45 * away / np.linalg.norm(away)
        return m.AnimationGroup(
            m.FadeIn(c, shift=-away),
            m.Create(h),
            m.GrowFromCenter(d),
        )

    def pulse_to(self, i, color):
        """A pulse along the arrow that carries point i."""
        a = [self.a_in, self.a_use, self.a_res, self.a_out][i]
        if i in (1, 2):
            path = (self.a_use if i == 1 else self.a_res).copy()
            path.remove(*[t for t in path.submobjects])  # no tip
            return m.ShowPassingFlash(path.set_stroke(color, 9), time_width=0.6, run_time=1.0)
        return pulse(a, color)


def mini_map(active=None):
    """The map in the top-right corner, without words: where the current hook hangs."""
    full = LoopMap()
    shapes = m.VGroup(*[c[0].copy() for c in full.chips], *[a.copy() for a in full.arrows])
    center = shapes.get_center()
    shapes.scale(MINI_K, about_point=center).shift(MINI_AT - center)
    for s in shapes[4:]:
        s.set_stroke(width=2.5)
    for s in shapes[:4]:
        s.set_stroke(width=1.5)
    dots = m.VGroup(*[
        m.Dot(MINI_AT + MINI_K * (p - center), radius=0.07, color=MINT if active in (None, i) else FAINT)
        for i, p in enumerate(full.points)
    ])
    group = m.VGroup(shapes, dots)
    if active is not None:
        group.add(m.Circle(radius=0.15).set_stroke(MINT, 2.5).move_to(dots[active]))
    return group


def corner(active):
    """The mini map plus the active event's tag under it."""
    mm = mini_map(active)
    tag = hook_tag(EVENTS[active]).next_to(mm, m.DOWN, buff=0.18).align_to([RIGHT, 0, 0], m.RIGHT)
    return m.VGroup(mm, tag)


def flash_dot(c, i, color=MINT):
    return m.Flash(c[0][1][i], color=color, line_length=0.14, flash_radius=0.2, run_time=0.8)


# ---------------------------------------------------------------------------- the messages column

def msg_card(ctx, i):
    """Message i of the run, as the context column shows it."""
    msg = MESSAGES[i]
    content = msg["content"]
    if isinstance(content, str):
        return ctx.message(msg["role"], content, size=19)
    block = content[0]
    if isinstance(block, dict):  # a tool_result
        out = block["content"]
        body = "def total(items): …" if out.startswith("def total") else out
        card_ = ctx.message("tool_result", body, size=20, mono=True)
        if "failed" in body:
            paint(card_[2][1], body, {r"\d+ failed, \d+ passed": ROSE})
        elif "passed" in body:
            paint(card_[2][1], body, {r"\d+ passed": MINT})
        elif "read-only" in body:
            paint(card_[2][1], body, {r".+": ROSE})
        return card_
    if block.type == "tool_use":
        return ctx.message("tool_use", f'{block.name}("{block.input["path"]}")', size=20, mono=True)
    return ctx.message("assistant", block.text, size=19)


# ---------------------------------------------------------------------------- the episode


class Ep18Hooks(Episode):
    number = 18

    def lesson(self):
        self.hook()
        self.events()
        self.observe()
        self.block()
        self.permissions()
        self.stop()
        self.spiral()
        self.claude_code()
        self.recap()

    def detach(self, parent, parts):
        """Glyphs faded out of a Text must leave it, or a later fade of the Text brings them back."""
        parent.remove(*parts)
        self.remove(*parts)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook · tests after every edit", notes="Running example: run the tests after every edit Otto makes. A rule in the prompt is a request the model may forget (Episode 8, constraints over guidance: Li, AI Agents in Depth, ch. 5); writing every wish into the loop turns it into a tangle (illustrative). Learn Claude Code s04's subtitle: 'Hang on the Loop, Don't Write into It'. Facts as of October 2026.")
        bot = otto(1.2).move_to([-5.0, 1.15, 0])
        files = ["cart.py", "checkout.py", "cart.py"]
        rows = m.VGroup()
        for i, f in enumerate(files):
            e = chip(f"edit {f}", SUN, SUN_FILL, width=3.3, height=0.66, size=24, mono=True)
            t = chip("pytest -q", MINT, MINT_FILL, width=2.5, height=0.66, size=24, mono=True)
            m.VGroup(e, t).arrange(m.RIGHT, buff=0.9).move_to([1.9, 1.75 - 1.05 * i, 0])
            rows.add(m.VGroup(e, arrow(e.get_right(), t.get_left(), MUTED, buff=0.1), t))
        rule = message("system", "Always run pytest -q\nafter every edit.", width=4.6, size=22).move_to([-4.4, -1.35, 0])
        skipped = chip("skipped", ROSE, ROSE_FILL, width=2.5, height=0.66, size=24).move_to(rows[2][2])
        example = text("for example", 20, MUTED).next_to(skipped, m.DOWN, buff=0.18)
        ep8 = pill("Episode 8 · a check in code beats a rule in the prompt", SUN, SUN_FILL, 22).move_to([1.9, -2.55, 0])

        # the loop, then the loop with every wish written into it
        clean = ["while True:", "    response = client.messages.create(...)", '    if response.stop_reason != "tool_use":',
                 "        return response", "    for block in response.content:", "        output = run_tool(block)",
                 "        results.append(...)"]
        extra = {"tests": "        output += run_tests(block)          # tests",
                 "log": "        log(block)                          # logging",
                 "safety": "        if is_dangerous(block): continue    # safety"}
        lines = {}
        for s in clean + list(extra.values()):
            words = code(s.strip(), 20, INK, t2c=CODE_T2C)
            lines[s] = words
        states = [
            clean,
            clean[:6] + [extra["tests"]] + clean[6:],
            clean[:5] + [extra["log"]] + clean[5:6] + [extra["tests"]] + clean[6:],
            clean[:5] + [extra["safety"], extra["log"]] + clean[5:6] + [extra["tests"]] + clean[6:],
        ]
        lh, left, top = 0.43, -4.3, 2.05
        width = max(len(s) for s in states[-1]) * CW + 0.9

        def place(state):
            spots = {}
            for k, s in enumerate(state):
                w = lines[s]
                x = left + (len(s) - len(s.lstrip())) * CW
                spots[s] = (x, top - k * lh)
            return spots

        def box_for(state):
            h = len(state) * lh + 0.45
            return card(width, h, EDGE, CARD, radius=0.16, stroke=2).move_to([left - 0.45 + width / 2, top + 0.33 - h / 2, 0])

        for k, key in ((1, "tests"), (2, "log"), (3, "safety")):
            s = extra[key]
            x, y = place(states[k])[s]
            lines[s].move_to([x, 0, 0], aligned_edge=m.LEFT)
            set_base(lines[s], s.strip(), y)
            lines[s].set_z_index(2)
        clean_spots = place(clean)
        for s in clean:
            lines[s].move_to([clean_spots[s][0], 0, 0], aligned_edge=m.LEFT)
            set_base(lines[s], s.strip(), clean_spots[s][1])
            lines[s].set_z_index(2)
        box = box_for(clean)
        loop = m.VGroup(box, *[lines[s] for s in clean])

        def grow(k, key):
            spots = place(states[k])
            moves = [lines[s].animate.shift((spots[s][1] - (top - states[k - 1].index(s) * lh)) * m.UP)
                     for s in states[k - 1] if states[k].index(s) != states[k - 1].index(s)]
            new = extra[key]
            return m.AnimationGroup(m.Transform(box, box_for(states[k])), *moves, m.FadeIn(lines[new], shift=0.1 * m.RIGHT))

        motto = title_text("Hang on the loop. Don't write into it.", 40, MINT).move_to([0, -1.95, 0])
        src = cite("learn-claude-code · s04").next_to(motto, m.DOWN, buff=0.22)

        self.say(
            "[Say you want Otto to run the tests] [after every edit it makes.]",
            m.FadeIn(bot, shift=0.3 * m.UP),
            m.LaggedStart(*[m.AnimationGroup(*[m.FadeIn(x, shift=0.2 * m.UP) for x in r]) for r in rows], lag_ratio=0.35),
        )
        self.say(
            "[You could put it in the prompt.] [But a model can forget], [and episode eight showed that a check in code beats a rule in the prompt.]",
            m.FadeIn(rule, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeOut(rows[2][2], scale=0.8), m.FadeIn(skipped, scale=0.8), m.FadeIn(example)),
            m.FadeIn(ep8, shift=0.15 * m.UP),
        )
        self.say(
            "[Or you could write it into the loop itself]: [a line for the tests], [one for logging], [one for safety], [until the loop is a tangle.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bot, rows[0], rows[1], rows[2][:2], skipped, example, rule, ep8), shift=0.2 * m.DOWN), m.FadeIn(loop, shift=0.2 * m.UP)),
            grow(1, "tests"),
            grow(2, "log"),
            grow(3, "safety"),
            m.Circumscribe(m.VGroup(*[lines[s] for s in extra.values()]), color=ROSE, buff=0.08),
        )
        self.remove(bot, rows, skipped, example, rule, ep8)
        back = [lines[s].animate.shift(((top - clean.index(s) * lh) - (top - states[3].index(s) * lh)) * m.UP)
                for s in clean if states[3].index(s) != clean.index(s)]
        self.say(
            "[Hooks are the third way.] [The loop stays exactly as it was], [and your code hangs on it, from the outside.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(*[lines[s] for s in extra.values()]), shift=0.15 * m.RIGHT), m.Transform(box, box_for(clean)), *back),
            m.Circumscribe(box, color=MINT, buff=0.08),
            m.AnimationGroup(m.FadeIn(motto, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the four events

    def events(self):
        self.next_section("1 · Four events", notes="Learn Claude Code s04: HOOKS has four events. UserPromptSubmit: after input, before the model. PreToolUse: after a tool_use block, before its handler. PostToolUse: after the handler, before the result goes back. Stop: when the model answers without a tool call, before the loop ends. register_hook appends a function to an event's list; trigger_hooks runs them in order and returns the first non-None result ('return value ≠ None → hook says \"stop\"'); a hook that returns None only observes. 'The loop stays boring on purpose.' Facts as of October 2026.")
        mp = LoopMap()
        mini = mini_map()
        hooks_card = CodeCard(S_HOOKS, title="hooks.py")
        hooks_card.group.move_to([0, 0, 0]).align_to([LEFT, 2.0, 0], m.UL)
        trig = CodeCard(S_TRIGGER)
        trig.group.next_to(hooks_card.group, m.RIGHT, buff=0.45).align_to(hooks_card.group, m.UP)
        observe = flag("returns None · it only watches", AZURE, AZURE_FILL)
        act = flag("returns a value · it decides", ROSE, ROSE_FILL)
        mid_x = (hooks_card.box.get_left()[0] + trig.box.get_right()[0]) / 2
        m.VGroup(observe, act).arrange(m.RIGHT, buff=0.4).move_to([mid_x, -1.45, 0])
        quote = title_text("“The loop stays boring on purpose.”", 38, INK).move_to([mid_x, -2.5, 0])
        src = cite("learn-claude-code · s04").next_to(quote, m.DOWN, buff=0.2)

        self.say(
            "[Here's the loop as a map.] [A prompt comes in], [the model calls tools], [results come back], [and at the end, the model answers.]",
            m.AnimationGroup(self.new_header("01 · events", "Four places to hang code"), m.FadeIn(mp.chips[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(mp.chips[0], shift=0.2 * m.RIGHT), m.GrowArrow(mp.a_in)),
            m.AnimationGroup(m.Create(mp.a_use), m.FadeIn(mp.chips[2], shift=0.2 * m.LEFT), m.FadeIn(mp.l_use)),
            m.AnimationGroup(m.Create(mp.a_res), m.FadeIn(mp.l_res)),
            m.AnimationGroup(m.GrowArrow(mp.a_out), m.FadeIn(mp.chips[3], shift=0.2 * m.DOWN)),
        )
        self.say(
            "[User prompt submit fires when a prompt arrives], [before the model reads it.]",
            mp.clip_on(0),
            mp.pulse_to(0, AZURE),
        )
        self.say(
            "[Pre tool use fires when the model asks for a tool], [before the handler runs it.]",
            mp.clip_on(1),
            mp.pulse_to(1, SUN),
        )
        self.say(
            "[Post tool use fires after the handler], [before the result goes back.]",
            mp.clip_on(2),
            mp.pulse_to(2, MINT),
        )
        self.say(
            "[And the stop hook fires when the model wants to finish], [before the loop ends.]",
            mp.clip_on(3),
            mp.pulse_to(3, CORAL),
        )
        full = m.VGroup(mp.chips, mp.arrows, mp.dots, mp.l_use, mp.l_res, *mp.clips, mp.hangers)
        self.say(
            "[In code, each event is just a list of functions], [and at each of the four points], [the loop calls trigger hooks.]",
            m.AnimationGroup(
                m.FadeOut(full, target_position=MINI_AT, scale=MINI_K),
                m.FadeIn(mini, target_position=full.get_center(), scale=1 / MINI_K),
                m.FadeIn(hooks_card.group, shift=0.2 * m.UP),
            ),
            m.LaggedStart(*[m.Flash(d, color=MINT, line_length=0.14, flash_radius=0.2) for d in mini[1]], lag_ratio=0.25),
            m.FadeIn(trig.group, shift=0.2 * m.UP),
        )
        b1 = trig.bar(5, color=AZURE)
        b2 = trig.bar(3, 4, color=ROSE)
        self.say(
            "[A hook that returns nothing only watches.] [The first one that returns something] [decides what happens next.]",
            m.AnimationGroup(m.FadeIn(b1), m.FadeIn(observe, shift=0.15 * m.UP)),
            m.FadeIn(b2),
            m.FadeIn(act, shift=0.15 * m.UP),
        )
        self.say(
            "[As Learn Claude Code puts it], [the loop stays boring on purpose.]",
            m.FadeIn(src),
            m.FadeIn(quote, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ observe: tests after every edit

    def observe(self):
        self.next_section("2 · Observe: tests after every edit", notes=f"Two PostToolUse hooks, in order: format_file (ruff format on the edited file; returns None, so it only observes) and run_tests (pytest -q after every edit_file; its return value is added to the tool result, our harness's way of letting a PostToolUse hook talk back; Claude Code's equivalent is additionalContext or exit code 2). Otto's first edit (illustrative) changes range(len(items) - 1) to range(1,len(items)); ruff 0.17.1 (measured) adds the space and prints '1 file reformatted'; the tests then say '1 failed, 12 passed' (computed: total([3, 4]) = {WRONG_TOTAL}, not 7). The tool result Otto gets: {R_FIRST!r}. Facts as of October 2026.")
        top = corner(2)
        post = CodeCard(S_POST)
        post.group.align_to([LEFT, TOP, 0], m.UL)
        use = message("tool_use", "edit_file  cart.py\nold: range(len(items) - 1)\nnew: range(1,len(items))", width=5.8, size=22, mono=True)
        use.align_to([RIGHT, 1.3, 0], m.UR)
        cart = CodeCard(CART.rstrip(), title="tinyshop/cart.py")
        cart.group.align_to([LEFT, -0.45, 0], m.UL)
        cart.split()
        old = cart.lines[2]
        base, x0 = cart.base(2), old.get_left()[0]
        new1 = set_base(code("for i in range(1,len(items)):", 20, INK, t2c=CODE_T2C).move_to([x0, 0, 0], aligned_edge=m.LEFT), "foriinrange(1,len(items)):", base).set_z_index(2)
        new2 = set_base(code("for i in range(1, len(items)):", 20, INK, t2c=CODE_T2C).move_to([x0, 0, 0], aligned_edge=m.LEFT), "foriinrange(1,len(items)):", base).set_z_index(2)
        bar = cart.bar(2, color=SUN)
        bad = cart.bar(2, color=ROSE)
        wrong = code(f"total([3, 4]) = {WRONG_TOTAL}, not 7", 20, ROSE).next_to(cart.box, m.DOWN, buff=0.16).align_to(cart.box, m.LEFT)
        fmt, tests = SHELL[0], SHELL[1]
        term = terminal([f"$ {fmt[0]}", fmt[1], f"$ {tests[0]}", f"! {tests[1]}"], width=6.3, title="hooks ran")
        term.align_to([RIGHT, -0.5, 0], m.UR)
        res = message("tool_result", R_FIRST, width=5.8, size=22, mono=True).align_to(use, m.UR)
        paint(res[2][1], R_FIRST, {r"\d+ failed, \d+ passed": ROSE})

        self.say(
            "[Back to our wish.] [We hang two hooks after every tool call]: [a formatter], [then the tests.]",
            m.AnimationGroup(self.new_header("02 · observe", "Tests after every edit"), m.FadeIn(top, shift=0.15 * m.DOWN)),
            m.FadeIn(post.group, shift=0.2 * m.UP),
            m.Circumscribe(post.lines[4], color=MINT, buff=0.06),
            m.Circumscribe(m.VGroup(*post.lines[0:3], post.lines[5]), color=MINT, buff=0.06),
        )
        gone = m.FadeOut(old, shift=0.1 * m.UP)
        gone.introducer = True  # old arrives with the card earlier in this same line
        self.say(
            "[Otto's first try at the bug] [changes the range in the cart's loop.]",
            m.AnimationGroup(cart.intro(), m.FadeIn(use, shift=0.2 * m.LEFT)),
            m.AnimationGroup(gone, m.FadeIn(new1, target_position=use), m.FadeIn(bar)),
        )
        self.say(
            "[The formatter fires first], [and adds the missing space after the comma.]",
            m.AnimationGroup(flash_dot(top, 2), m.FadeIn(term[:4]), m.FadeIn(term[4][0], shift=0.1 * m.UP), m.FadeIn(term[4][1], shift=0.1 * m.UP)),
            m.ReplacementTransform(new1, new2),
        )
        self.say(
            "[Then the tests run]: [one failed, twelve passed.] [The first try was wrong.]",
            m.FadeIn(term[4][2], shift=0.1 * m.UP),
            m.FadeIn(term[4][3], shift=0.1 * m.UP),
            m.AnimationGroup(m.Transform(bar, bad), m.FadeIn(wrong, shift=0.1 * m.UP)),
        )
        self.say(
            "[That verdict rides back in the tool result], [so Otto knows at once, not three steps later.]",
            m.AnimationGroup(m.FadeOut(use, shift=0.2 * m.RIGHT), m.FadeIn(res, target_position=term)),
            m.Circumscribe(glyphs(res[2][1], "1 failed, 12 passed"), color=ROSE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ block

    def block(self):
        self.next_section("3 · Block: a hook can say no", notes=f"A PreToolUse hook (protect_tests) returns a reason when an edit targets tests/: trigger_hooks returns it, the handler never runs, and the reason becomes the call's tool_result ({R_BLOCK!r}); the test file is unchanged (asserted). Otto's attempt to make the test expect 4 is illustrative (a known shortcut: change the test, not the code). The loop excerpt is the code the scene runs (learn-claude-code s04's shape; the PostToolUse note is this series' addition). Facts as of October 2026.")
        top = corner(1)
        guard = CodeCard(S_PROTECT)
        guard.group.align_to([LEFT, TOP, 0], m.UL)
        use = message("tool_use", "edit_file  tests/test_cart.py\nold: == 7\nnew: == 4", width=5.4, size=22, mono=True)
        use.align_to([RIGHT, 1.3, 0], m.UR)
        example = text("for example", 20, MUTED).next_to(use, m.LEFT, buff=0.2).align_to(use, m.DOWN)
        test = CodeCard(TEST.rstrip(), title="tinyshop/tests/test_cart.py")
        test.group.align_to([RIGHT, -1.65, 0], m.UR)
        x = use.get_center()[0]
        link = arrow([x, use.get_bottom()[1], 0], [x, test.box.get_top()[1], 0], SUN, buff=0.12)
        gate = icon(HOOK, 0.5).next_to(link, m.LEFT, buff=0.15)
        gate_name = code("protect_tests", 20, MINT).next_to(gate, m.LEFT, buff=0.15)
        mid = link.get_center()
        cross = m.VGroup(
            m.Line(mid + [-0.22, -0.22, 0], mid + [0.22, 0.22, 0]),
            m.Line(mid + [-0.22, 0.22, 0], mid + [0.22, -0.22, 0]),
        ).set_stroke(ROSE, 7)
        blocked = flag("blocked", ROSE, ROSE_FILL).next_to(cross, m.RIGHT, buff=0.25)
        seven = test.part(1, "7")
        res = message("tool_result", R_BLOCK, width=5.4, size=22, mono=True).align_to(use, m.UR)
        paint(res[2][1], R_BLOCK, {r".+": ROSE})

        loop = CodeCard(S_TOOLS, title="agent_loop · the tool calls")
        loop.group.align_to([LEFT, 2.15, 0], m.UL)
        b_pre = loop.bar(2, color=MINT)
        b_run = loop.bar(3, 4, color=MINT)
        b_post = loop.bar(5, 6, color=MINT)
        n1 = flag("a reason → it's the result", ROSE, ROSE_FILL)
        n2 = flag("None → run the tool", MINT, MINT_FILL)
        m.VGroup(n1, n2).arrange(m.RIGHT, buff=0.4).move_to([loop.box.get_center()[0], -2.4, 0])

        self.say(
            "[Next, Otto tries a shortcut]: [it edits the test, to expect four instead of seven.]",
            m.AnimationGroup(self.new_header("03 · block", "A hook can say no"), m.FadeIn(top, shift=0.15 * m.DOWN), m.FadeIn(test.group, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(use, shift=0.2 * m.LEFT), m.GrowArrow(link), m.FadeIn(example)),
        )
        self.say(
            "[A pre tool use hook guards the tests folder.] [It returns a reason], [so the handler never runs], [and the test stays as it was.]",
            m.AnimationGroup(m.FadeIn(guard.group, shift=0.2 * m.UP), m.FadeIn(gate, shift=0.2 * m.DOWN), m.FadeIn(gate_name), flash_dot(top, 1)),
            m.Circumscribe(guard.lines[3], color=ROSE, buff=0.06),
            m.AnimationGroup(m.Create(cross), m.FadeIn(blocked, shift=0.1 * m.LEFT), link.animate.set_opacity(0.35)),
            m.Circumscribe(seven, color=MINT, buff=0.08),
        )
        self.say(
            "[The reason goes back as the tool result], [so Otto reads why.]",
            m.AnimationGroup(m.FadeOut(use, shift=0.2 * m.RIGHT), m.FadeOut(example), m.FadeIn(res, target_position=cross)),
            m.Circumscribe(res[2][1], color=ROSE, buff=0.06),
        )
        self.say(
            "[In the loop, it's a few lines]: [ask the pre tool use hooks first], [run the tool only if nobody objects], [then let the post tool use hooks add a note.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(guard.group, test.group, link, gate, gate_name, cross, blocked, res), shift=0.2 * m.DOWN), m.FadeIn(loop.group, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(b_pre), m.FadeIn(n1, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Transform(b_pre, b_run), m.FadeIn(n2, shift=0.15 * m.UP)),
            m.AnimationGroup(m.Transform(b_pre, b_post), flash_dot(top, 1, MUTED)),
        )
        self.wait(0.3)
        self.clear(top)
        self.top = top

    # ------------------------------------------------------------------ permissions

    def permissions(self):
        self.next_section("4 · Permissions: allow can't beat deny", notes="Learn Claude Code s04: the permission check of s03 (Episode 17's gate: deny, ask, allow) moves into a PreToolUse hook. Claude Code (s04's reading of its source, toolHooks.ts, not verified by us): 'Hook \"allow\" cannot bypass deny/ask rules', called the most important security design in CC's permission system. Claude Code docs (hooks, October 2026): a PreToolUse 'allow' skips the permission prompt, and 'Deny and ask rules are still evaluated regardless of what the hook returns'. The rm -rf call and its deny rule are illustrative.")
        top = self.top
        label = code('HOOKS["PreToolUse"]', 22, INK).align_to([LEFT, 2.3, 0], m.UL)
        glyphs(label, "PreToolUse").set_color(MINT)

        def hook_card(name, desc, extra=None):
            words = m.VGroup(code(name, 22, MINT, weight=m.BOLD), text(desc, 22, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            body = m.VGroup(icon(HOOK, 0.6), words).arrange(m.RIGHT, buff=0.3)
            if extra is not None:
                body.add(extra)
                extra.next_to(words, m.DOWN, buff=0.2).align_to(words, m.LEFT)
            box = card(5.9, body.height + 0.5, MINT, CARD, radius=0.16, stroke=2)
            body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            return m.VGroup(box, body)

        gates = m.VGroup(*[chip(w, c, f, width=1.3, height=0.52, size=22) for w, c, f in
                           (("deny", ROSE, ROSE_FILL), ("ask", SUN, SUN_FILL), ("allow", MINT, MINT_FILL))]).arrange(m.RIGHT, buff=0.18)
        h1 = hook_card("protect_tests", "blocks edits to tests/")
        h2 = hook_card("check_permission", "the gate from Episode 17", gates)
        stack = m.VGroup(h1, h2).arrange(m.DOWN, buff=0.25).next_to(label, m.DOWN, buff=0.3).align_to([LEFT, 0, 0], m.LEFT)
        ep17 = pill("Episode 17 · Permissions & Sandboxes", MINT, MINT_FILL, 20).next_to(stack, m.DOWN, buff=0.3).align_to(stack, m.LEFT)

        cx = 3.75
        call = message("tool_use", "bash  rm -rf ~/tinyshop", width=5.0, size=22, mono=True)
        call.move_to([cx, 0, 0]).align_to([0, 1.3, 0], m.UP)
        allow = flag("hook says: allow", MINT, MINT_FILL).move_to([cx, -0.5, 0])
        rule_words = m.VGroup(code("settings · deny", 20, ROSE, weight=m.BOLD), code("Bash(rm -rf:*)", 22, INK)).arrange(m.DOWN, buff=0.12)
        rule_box = card(rule_words.width + 0.7, rule_words.height + 0.45, ROSE, ROSE_FILL, radius=0.16, stroke=2.5)
        rule = m.VGroup(rule_box, rule_words.move_to(rule_box)).move_to([cx, -1.85, 0])
        a1 = arrow(call.get_bottom(), allow.get_top(), MUTED, buff=0.06)
        a2 = arrow(allow.get_bottom(), rule.get_top(), MINT, buff=0.06)
        denied = title_text("denied", 34, ROSE).next_to(rule, m.DOWN, buff=0.22)
        quote = text("“the most important security design”", 24, INK).next_to(ep17, m.DOWN, buff=0.45).align_to([LEFT, 0, 0], m.LEFT)
        src_a = cite("learn-claude-code · s04 · reading CC's source").next_to(quote, m.DOWN, buff=0.14).align_to(quote, m.LEFT)
        src_b = cite("Claude Code docs · hooks, 2026").next_to(denied, m.DOWN, buff=0.14)

        self.say(
            "[The permission check from episode seventeen belongs here too]: [it becomes one more pre tool use hook.]",
            m.AnimationGroup(self.new_header("04 · permissions", "Allow can't beat deny"), m.FadeIn(label), m.FadeIn(h1, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(h2, shift=0.25 * m.DOWN), m.FadeIn(ep17, shift=0.1 * m.UP), flash_dot(top, 1)),
        )
        self.say(
            "[Claude Code keeps one rule above every hook.] [Say a hook answers allow.] [A deny rule still wins.]",
            m.FadeIn(call, shift=0.2 * m.DOWN),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(allow, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(a2), m.FadeIn(rule, shift=0.15 * m.DOWN), m.FadeIn(denied, scale=0.8)),
        )
        self.say(
            "[A hook can add checks, but it can't switch yours off.] [Learn Claude Code calls this the most important security design in Claude Code's permission system.]",
            m.AnimationGroup(m.Circumscribe(rule, color=ROSE, buff=0.08), m.FadeIn(src_b)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src_a)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ stop

    def stop(self):
        self.next_section("5 · Stop: not done until the tests pass", notes=f"The Stop event (learn-claude-code s04): when the model answers without a tool call, the stop hooks run; a non-None result is appended as a user message and the loop continues. Here tests_must_pass runs pytest -q; Otto, blocked, gives up ({PLAN[2]!r}); the hook returns {FORCE!r}; Otto re-reads cart.py (the formatter had changed it), fixes the range, the PostToolUse hooks report '13 passed', Otto answers, the stop hook returns None and the loop returns. Six calls, twelve messages (computed). Otto's choices are illustrative. Facts as of October 2026.")
        top = corner(3)
        hook = CodeCard(S_STOPHOOK)
        hook.group.align_to([LEFT, TOP, 0], m.UL)
        loop = CodeCard(S_STOP)
        loop.group.align_to([LEFT, -0.05, 0], m.UL)
        ctx = ContextColumn(center=(3.9, -1.07, 0), width=5.4, height=4.75, title="messages", gap=0.1)
        first = [msg_card(ctx, i) for i in range(5)]
        for c in first:
            ctx.push(c)
        shown = list(ctx.items)
        cards = {i: msg_card(ctx, i) for i in range(5, 12)}
        end = pill('stop_reason: "end_turn"', CORAL, CORAL_FILL, 22, mono=True).move_to([-3.3, -0.6, 0])
        done = flag(f"done · {len(SENT)} calls · {len(MESSAGES)} messages", MINT, MINT_FILL, 20)
        done.next_to(loop.box, m.DOWN, buff=0.14).align_to(loop.box, m.LEFT)

        self.say(
            "[Blocked, Otto gives up]: [it says the test looks wrong, and stops.]",
            m.AnimationGroup(self.new_header("05 · stop", "Not done until the tests pass"), m.FadeIn(top, shift=0.15 * m.DOWN), m.FadeIn(ctx.frame), *[m.FadeIn(c) for c in shown]),
            m.AnimationGroup(ctx.push(cards[5]), m.FadeIn(end, shift=0.1 * m.UP)),
        )
        self.say(
            "[But a stop hook runs the tests first.] [They still fail], [so it returns a message.]",
            m.AnimationGroup(m.FadeIn(hook.group, shift=0.2 * m.UP), flash_dot(top, 3)),
            m.FadeIn(hook.bar(1, 2, color=ROSE)),
            m.Circumscribe(hook.lines[3], color=AZURE, buff=0.06),
        )
        b = loop.bar(4, 5, color=AZURE)
        self.say(
            "[The loop appends it as a user message], [and goes around again.]",
            m.AnimationGroup(m.FadeOut(end), m.FadeIn(loop.group, shift=0.2 * m.UP), m.FadeIn(b), ctx.push(cards[6])),
            m.AnimationGroup(m.Transform(b, loop.bar(6, color=AZURE)), flash_dot(top, 3, AZURE)),
        )
        self.say(
            "[This time, Otto reads the cart], [fixes the range], [and the hooks report: thirteen passed.]",
            m.Succession(ctx.push(cards[7], run_time=0.8), ctx.push(cards[8], run_time=0.8)),
            ctx.push(cards[9]),
            m.AnimationGroup(ctx.push(cards[10]), flash_dot(top, 2)),
        )
        self.say(
            "[Otto answers], [the stop hook has nothing to add], [and the loop ends.]",
            ctx.push(cards[11]),
            m.AnimationGroup(m.Transform(b, loop.bar(2, 3, color=MINT)), flash_dot(top, 3)),
            m.FadeIn(done, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the death spiral

    def spiral(self):
        self.next_section("6 · The death spiral", notes="Li, AI Agents in Depth, ch. 5 (Failure and Error Recovery, 'Termination'): a context-overflow stop fires a stop hook meant to commit code; the hook asks the LLM for a commit message; that call overflows again and fires the hook again — the death spiral. Fixes: disable model-invoking side effects on the error path, and a recursion-depth counter. Our own tests_must_pass hook can loop the same way if the tests can never pass. Claude Code docs (hooks, October 2026): 'The stop_hook_active field is true when Claude Code is already continuing as a result of a stop hook. Check this value … to avoid blocking on a condition that will never resolve.' The round counter is illustrative.")
        C, R = np.array([-3.2, -0.45, 0]), 2.0

        def node(title, sub, color, fill):
            words = m.VGroup(text(title, 24, INK, weight=m.SEMIBOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08)
            box = card(max(words.width + 0.5, 2.3), words.height + 0.36, color, fill, radius=0.16, stroke=2.5)
            return m.VGroup(box, words.move_to(box))

        at = lambda a: C + R * np.array([np.cos(a), np.sin(a), 0])  # noqa: E731
        nodes = m.VGroup(
            node("context overflow", "the agent stops", ROSE, ROSE_FILL).move_to(at(np.pi / 2)),
            node("stop hook", "commit the code", MINT, MINT_FILL).move_to(at(-np.pi / 6)),
            node("model call", "write a commit message", CORAL, CORAL_FILL).move_to(at(7 * np.pi / 6)),
        )
        ends = [
            (nodes[0].get_right() + [0.12, 0, 0], nodes[1].get_top() + [0.3, 0.12, 0]),
            (nodes[1].get_bottom() + [-0.3, -0.12, 0], nodes[2].get_bottom() + [0.3, -0.12, 0]),
            (nodes[2].get_top() + [-0.3, 0.12, 0], nodes[0].get_left() + [-0.12, 0, 0]),
        ]
        arcs = m.VGroup()
        for a, b in ends:
            arc = m.ArcBetweenPoints(a, b, angle=-1.3).set_stroke(MUTED, 4)
            arc.add_tip(tip_length=0.2, tip_width=0.2)
            arcs.add(arc)
        name = title_text("death spiral", 30, ROSE).move_to(C)
        src = cite("Li, AI Agents in Depth · ch. 5").align_to([LEFT, -3.3, 0], m.DL)

        meter_label = code("context", 20, MUTED).move_to([1.4, 2.05, 0], aligned_edge=m.LEFT)
        meter = Meter(width=3.6, fraction=0.62, color=SUN).next_to(meter_label, m.RIGHT, buff=0.3)
        rounds = Slot(self)
        round_at = lambda n: title_text(f"round {n}", 40, ROSE).move_to([3.4, 1.0, 0])  # noqa: E731
        no_exit = flag("no way out", ROSE, ROSE_FILL).move_to([3.4, 0.15, 0])

        def fix(title, sub, color=MINT, mono=True):
            words = m.VGroup(
                code(title, 22, color, weight=m.BOLD) if not mono else text(title, 24, INK, weight=m.SEMIBOLD),
                code(sub, 20, MINT) if mono else text(sub, 21, ROSE),
            ).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            box = card(5.4, words.height + 0.42, color, CARD, radius=0.16, stroke=2)
            return m.VGroup(box, words.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT))

        f1 = fix("no model calls on an error path", "if error: return None")
        f2 = fix("a recursion counter", "if depth > 1: return None")
        ours = fix("tests_must_pass", "tests never pass, so Otto never stops", color=ROSE, mono=False)
        ours[1][0].set_color(MINT)
        flag_card = json_card({"stop_hook_active": True}, size=20, accent=MINT, title="a stop hook's input", width=5.4)
        src_b = cite("Claude Code docs · hooks, 2026")
        m.VGroup(f1, f2, ours, flag_card, src_b).arrange(m.DOWN, buff=0.2).move_to([3.85, 0, 0]).align_to([0, 2.45, 0], m.UP)
        x_mark = m.VGroup(
            m.Line([-0.2, -0.2, 0], [0.2, 0.2, 0]), m.Line([-0.2, 0.2, 0], [0.2, -0.2, 0])
        ).set_stroke(ROSE, 7).move_to(arcs[1].point_from_proportion(0.5))

        def around(n_from, n_to, run_time=1.8):
            steps, each = [], run_time / (n_to - n_from + 1)
            for n in range(n_from, n_to + 1):
                lap = m.Succession(*[pulse_arc(a, run_time=each / 3) for a in arcs])
                steps.append(m.AnimationGroup(rounds.to(round_at(n), shift=0.1), lap, run_time=each))
            return m.Succession(*steps)

        def pulse_arc(a, run_time=0.8):
            path = a.copy()
            path.remove(*[t for t in path.submobjects])
            return m.ShowPassingFlash(path.set_stroke(ROSE, 8), time_width=0.6, run_time=run_time)

        self.say(
            "[Hooks can go wrong, too]: [here's a death spiral], [from the book AI Agents in Depth.]",
            self.new_header("06 · death spiral", "A hook that feeds itself"),
            m.FadeIn(name, scale=0.85),
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(meter_label), m.FadeIn(meter)),
        )
        self.say(
            "[An agent stops because its context overflowed.] [That fires a stop hook], [meant to commit the code.]",
            m.AnimationGroup(m.FadeIn(nodes[0], shift=0.15 * m.DOWN), meter.fill_to(1.0, run_time=1.6)),
            m.AnimationGroup(m.Create(arcs[0]), m.FadeIn(nodes[1], shift=0.15 * m.LEFT), rounds.to(round_at(1))),
            m.Circumscribe(nodes[1], color=MINT, buff=0.08),
        )
        self.say(
            "[The hook asks the model for a commit message.] [But the context is still full], [so that call overflows too], [and the hook fires again.]",
            m.AnimationGroup(m.Create(arcs[1]), m.FadeIn(nodes[2], shift=0.15 * m.RIGHT)),
            m.Circumscribe(meter, color=ROSE, buff=0.1),
            m.AnimationGroup(m.Create(arcs[2]), m.Circumscribe(nodes[0], color=ROSE, buff=0.08)),
            m.AnimationGroup(pulse_arc(arcs[0]), rounds.to(round_at(2))),
        )
        self.say(
            "[Round and round it goes], [each failure setting off the next.]",
            around(3, 6, run_time=2.6),
            m.FadeIn(no_exit, scale=0.85),
        )
        self.say(
            "[The fix has two parts]: [no model calls on an error path], [and a recursion counter, to break any loop that's left.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(rounds.current, no_exit, meter, meter_label), shift=0.15 * m.UP), arcs[1].animate.set_color(FAINT), m.Create(x_mark)),
            m.FadeIn(f1, shift=0.15 * m.UP),
            m.FadeIn(f2, shift=0.15 * m.UP),
        )
        rounds.current = None
        self.say(
            "[Our own stop hook has the same risk]: [if the tests can never pass], [Otto can never stop.]",
            m.AnimationGroup(m.FadeIn(ours[0], shift=0.15 * m.UP), m.FadeIn(ours[1][0], shift=0.15 * m.UP)),
            m.FadeIn(ours[1][1], shift=0.1 * m.UP),
            m.Circumscribe(ours[1][1], color=ROSE, buff=0.06),
        )
        self.say(
            "[So Claude Code tells a stop hook when it already forced one more round], [and the hook can let the agent finish.]",
            m.AnimationGroup(m.FadeIn(flag_card, shift=0.15 * m.UP), m.FadeIn(src_b)),
            m.Circumscribe(glyphs(flag_card[1], '"stop_hook_active": true'), color=MINT, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ Claude Code

    def claude_code(self):
        self.next_section("7 · Claude Code's events", notes=f"Learn Claude Code s04, 'Dive into CC Source' (the site's reading of coreTypes.ts, not verified by us): 'Not just 4, but 27' hook events, in seven groups (tool, session, user interaction, subagents, compaction, team, other). Claude Code's docs (code.claude.com/docs/en/hooks, October 2026) list 33, adding {', '.join(DOCS_EXTRA)}. The four events of this episode are highlighted. Counts computed and asserted.")
        rows = m.VGroup()
        y = 2.3
        core = set(EVENTS)
        chips_ = {}
        for label, names in CC_GROUPS:
            cs = m.VGroup()
            for n in names:
                w = code(n, 20, MINT if n in core else INK, weight=m.BOLD if n in core else m.NORMAL)
                box = card(w.width + 0.3, 0.46, MINT if n in core else EDGE, MINT_FILL if n in core else CARD, radius=0.12, stroke=2)
                c = m.VGroup(box, w.move_to(box))
                chips_[n] = c
                cs.add(c)
            cs.arrange(m.RIGHT, buff=0.15).move_to([0, y, 0]).align_to([LEFT + 1.95, 0, 0], m.LEFT)
            lab = text(label, 24, MUTED).move_to([LEFT, y, 0], aligned_edge=m.LEFT)
            rows.add(m.VGroup(lab, cs))
            y -= 0.6
        core_chips = m.VGroup(*[chips_[e] for e in EVENTS])
        others = m.VGroup(*[chips_[e] for e in CC_EVENTS if e not in core])
        labels = m.VGroup(*[r[0] for r in rows])
        count = m.VGroup(title_text(str(len(CC_EVENTS)), 64, MINT), text("hook events", 22, MUTED)).arrange(m.DOWN, buff=0.08)
        count.move_to([4.9, -0.1, 0])
        src_a = cite("learn-claude-code · s04 · reading CC's source").align_to([RIGHT, -2.85, 0], m.DR)
        docs = flag(f"Claude Code docs, October 2026: {len(set(CC_EVENTS) | set(DOCS_EXTRA))} events", SUN, SUN_FILL, 20)
        docs.align_to([RIGHT, -3.45, 0], m.DR)

        self.say(
            "[Our harness has four events.] [Claude Code has many more]: [twenty-seven, in Learn Claude Code's reading of its source], [and thirty-three in its documentation today.]",
            m.AnimationGroup(self.new_header("07 · claude code", "Four events, or dozens"), m.LaggedStart(*[m.FadeIn(c, scale=0.85) for c in core_chips], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeIn(labels), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in others], lag_ratio=0.05)),
            m.AnimationGroup(m.FadeIn(count, scale=0.85), m.FadeIn(src_a)),
            m.FadeIn(docs, shift=0.1 * m.UP),
        )
        self.say(
            "[Sessions, subagents, compaction, files]: [each one is a named point where your code can watch, or step in.]",
            m.LaggedStart(*[m.Circumscribe(rows[i][1], color=MUTED, buff=0.06) for i in (1, 3, 4, 7)], lag_ratio=0.3),
            m.Circumscribe(core_chips, color=MINT, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Hang code on the loop at four events; a hook that returns None observes; a hook that returns a reason blocks a call (PreToolUse) or keeps the loop going (Stop). 'The loop stays boring on purpose' (learn-claude-code s04).")
        halo = glow(MINT, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Hang", MINT), ("Observe", AZURE), ("Block", ROSE))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("four events", "return None", "return a reason")])
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
        row = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = text("Hang your code on the loop, and keep the loop boring.", 30, MUTED).next_to(row, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [hang], [observe], [block.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Hang your code on the loop, and keep the loop itself boring.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
