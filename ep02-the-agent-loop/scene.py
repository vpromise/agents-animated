"""Agents, Animated · Episode 2 · The Agent Loop (Module 1 · Foundations)

The one loop inside every agent. Running example: Otto's first full run on tinyshop — run the tests,
read cart.py, edit one line, run the tests again, answer. As in learn-claude-code s01, Otto has ONE tool
here, bash (cat to read, sed to edit); Episode 3 adds read_file and edit_file without touching the loop. The run is not drawn by hand: the scene runs
the reference loop (the same lines it shows, checked with inspect) against a scripted stand-in for the
model and in-memory tools, and every number on screen comes from that run (asserted below).

Sections: hook · the stateless model · one round trip (tool_use → tool_result, matching ids) · the loop
in code, stepped call by call beside the growing messages array · the append-only trajectory and the
re-sent history (1, 3, 5, 7, 9 = 25 messages) · two envelopes (Anthropic-style user message of
tool_result blocks vs OpenAI-style role "tool" messages) · exits and the iteration cap (Claude Code's
src/query.ts, 1,729 lines around a 30-line while True, per Learn Claude Code's reading) · what later
episodes add around the loop · recap.

Sources (facts dated October 2026):
- Learn Claude Code (learn.shareai.run/en/s01/): the loop; "Model raises hand: 'I need a tool'";
  "The loop itself never changes."; Dive into CC Source: src/query.ts is 1729 lines, its core is the
  30-line while True; production exit paths (prompt too long, abort, hook stop, max turns, token budget…).
- Li, AI Agents in Depth: ch. 1 "The essence of the ReAct loop is appending to the trajectory, over and
  over"; ch. 2 the stateless API ("the Agent framework must include the relevant history in every
  request"), OpenAI-style role "tool" messages with tool_call_id, and "Agents can become stuck
  repeating the same tool calls forever" (max_iterations).

Illustrative (for teaching): tinyshop, its bug and test counts (from harness_reference.py), the model's
choices in the run, the stateless "Which tests?" reply, the two-file parallel read, the stuck loop.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- Otto's run, computed

TASK = "The checkout test fails. Fix it."
CART = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s\n"
OLD = "range(len(items) - 1)"
BUG_LINE = "for i in range(len(items) - 1):"
OTHER_TESTS = 12  # tinyshop's other tests, all passing (illustrative)
FINAL = "Fixed. All 13 tests pass."
FILES = {"cart.py": CART}


def pytest_q(files):
    """tinyshop's test suite: test_total (total([3, 4]) == 7) plus twelve that pass."""
    space = {}
    exec(files["cart.py"], space)
    ok = space["total"]([3, 4]) == 7
    passed, failed = OTHER_TESTS + ok, int(not ok)
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


TEST, READ, SED = "pytest -q", "cat cart.py", "sed -i 's/ - 1//' cart.py"
assert CART.count(" - 1") == 1  # so sed's first-match-per-line edit touches exactly the bug


def run_bash(command):
    """Otto's one tool in this episode (as in learn-claude-code s01): a shell, simulated for tinyshop.
    Like s01's run_bash, a command that prints nothing returns "(no output)"."""
    if command == TEST:
        out = pytest_q(FILES)
    elif command == READ:
        out = FILES["cart.py"]
    elif command == SED:
        FILES["cart.py"] = FILES["cart.py"].replace(" - 1", "", 1)
        out = ""
    else:
        raise ValueError(command)
    return out[:50_000] if out else "(no output)"


TOOLS = [{"name": "bash"}]  # one tool; Episode 3 adds read_file and edit_file


# The model's decisions in this run (illustrative): what a model would plausibly do, call by call.
PLAN = [
    ("bash", {"command": TEST}),
    ("bash", {"command": READ}),
    ("bash", {"command": SED}),
    ("bash", {"command": TEST}),
    None,  # answers in words
]


class ScriptedModel:
    """Stands in for client.messages: records how many messages each call was sent."""

    def __init__(self):
        self.sent = []

    def create(self, messages, tools):
        self.sent.append(len(messages))
        step = PLAN[len(self.sent) - 1]
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=FINAL)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{len(self.sent):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=ScriptedModel())


def tool_results(response):
    """harness_reference.agent_loop's inner for-loop (s01 style: one tool, run_bash), named so the loop fits in ten lines."""
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output = run_bash(block.input["command"])
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
    return results


def agent_loop(messages):
    while True:
        response = client.messages.create(
            messages=messages, tools=TOOLS)
        messages.append({"role": "assistant",
            "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = tool_results(response)
        messages.append({"role": "user",
            "content": results})


LOOP_SRC = '''while True:
    response = client.messages.create(
        messages=messages, tools=TOOLS)
    messages.append({"role": "assistant",
        "content": response.content})
    if response.stop_reason != "tool_use":
        return response
    results = tool_results(response)
    messages.append({"role": "user",
        "content": results})'''
CODE_TITLE = "harness_reference.py · agent_loop(messages)"

MESSAGES = [{"role": "user", "content": TASK}]
ANSWER = agent_loop(MESSAGES)
SENT = client.messages.sent

# the code on screen is the code that ran
_ran = inspect.getsource(agent_loop)
assert all(line.strip() in _ran for line in LOOP_SRC.split("\n")) and len(LOOP_SRC.split("\n")) == 10
# the run
_buggy = {}
exec(CART, _buggy)
assert _buggy["total"]([3, 4]) == 3  # the bug: the last item is skipped
assert ANSWER.stop_reason == "end_turn" and ANSWER.content[0].text == FINAL
assert SENT == [1, 3, 5, 7, 9] and sum(SENT) == 25 and len(MESSAGES) == 10  # "one, three, five, seven, nine"; "twenty-five … ten"
RESULTS = [msg["content"][0]["content"] for msg in MESSAGES[2::2]]
assert RESULTS == ["1 failed, 12 passed", CART, "(no output)", "13 passed"]  # "one failed, twelve passed"; "thirteen passed"
CALLS = [msg["content"][0] for msg in MESSAGES[1:-1:2]]
assert [c.id for c in CALLS] == [msg["content"][0]["tool_use_id"] for msg in MESSAGES[2::2]]  # ids match
assert sum(a != b for a, b in zip(CART.splitlines(), FILES["cart.py"].splitlines())) == 1  # "changed one line"
assert BUG_LINE in CART and OLD in BUG_LINE
TOOL_USES = sum(1 for c in CALLS if c.type == "tool_use")
assert len(SENT) == 5 and TOOL_USES == 4  # five calls, four tool runs

# ---------------------------------------------------------------------------- look

MODEL_ICON = recolored("llm", CORAL, CORAL_FILL, prefix="ep02")
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep02")
HAND_ICON = "ep02-hand"

CODE_T2C = {
    r"#[^\n]*": FAINT,
    r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
    r"\b(def|while|for|in|if|return|True|False|None|not|and|or)\b": ORCHID,
    r"\b\d+(\.\d+)?\b": SUN,
    r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
}
JSON_T2C = {
    r"\"[^\"\n]*\"(?=\s*:)": AZURE,
    r":\s*\"[^\"\n]*\"": MINT,
    r"\b-?\d+(\.\d+)?\b": SUN,
}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

ROW_W, ROW_H = 5.44, 0.46
COL_CENTER, COL_W, COL_H = (3.85, -0.7, 0), 5.8, 5.8
CODE_W, CODE_TOP = 7.4, 2.58


def cite(s):
    return text(s, 20, MUTED)


def kind_of(msg):
    content = msg["content"]
    if isinstance(content, str):
        return "text"
    block = content[0]
    if isinstance(block, dict):
        return "tool_result"
    return block.type


def accent_of(role, kind):
    return SUN if kind == "tool_use" else MINT if kind == "tool_result" else ROLES[role][0]


def summary(msg):
    """(role, body, kind) of one message, as a row of the messages column shows it."""
    kind = kind_of(msg)
    if kind == "text" and isinstance(msg["content"], str):
        return msg["role"], msg["content"], "text"
    block = msg["content"][0]
    if kind == "tool_result":
        out = block["content"]
        return "user", ("def total(items): …" if out.startswith("def total") else out), kind
    if kind == "tool_use":
        arg = block.input["command"]
        if len(arg) > 14:
            arg = " ".join(arg.split()[:2]) + " …"
        return "assistant", f'{block.name}("{arg}")', kind
    return "assistant", block.text, "text"


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def row(role, body, kind, width=ROW_W):
    """One message of the array on one line: a stripe in the block's colour, the role, the content."""
    accent = accent_of(role, kind)
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    words = text(body, 20, INK) if kind == "text" else code(body, 20, accent)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(1.98 * m.RIGHT)
    base = box.get_y() - 0.09
    label.shift((base - label[0].get_bottom()[1]) * m.UP)
    words.shift((base - words[0].get_bottom()[1]) * m.UP)
    return m.VGroup(box, stripe, label, words)


def msg_row(msg):
    return row(*summary(msg))


def packet(colors, at):
    """A copy of the array in miniature: one bar per message, in the messages' colours."""
    bars = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.02, width=0.56, height=0.075, stroke_width=0, fill_color=c, fill_opacity=1)
        for c in colors
    ]).arrange(m.DOWN, buff=0.035)
    return bars.move_to(at)


def colors_of(messages):
    return [accent_of(msg["role"], kind_of(msg)) for msg in messages]


def send(colors, source, target):
    """The array flying from the column into the model, and vanishing there (one cue)."""
    p = packet(colors, target)
    out = m.FadeOut(p, scale=0.3, run_time=0.35)
    out.introducer = True  # p is introduced by the FadeIn just before, in the same line
    return m.Succession(m.FadeIn(p, target_position=source, run_time=0.8), out)


def wire(src, accent, size=20):
    """A message as it travels over the wire: compact JSON in Menlo, in a card of the block's colour."""
    body = code(src, size, INK, t2c=JSON_T2C)
    box = card(body.width + 0.7, body.height + 0.55, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box)
    return m.VGroup(box, body)


def role_msg(label, color, body, width, size=20):
    """kit.message() for any role label (OpenAI's "tool" role has no entry in ROLES)."""
    tag = code(label, 22, color, weight=m.BOLD)
    words = code(body, size, INK, t2c={r"toolu_\d+|call_[a-z]+": SUN})
    inner = m.VGroup(tag, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def side_chip(pic, label, color, fill, width=2.5, height=0.85, size=24):
    box = card(width, height, color, fill, radius=0.18)
    body = m.VGroup(icon(pic, 0.52), text(label, size, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2).move_to(box)
    return m.VGroup(box, body)


class LoopCode:
    """The loop's code card with a line cursor (a tinted bar) and a loop-back arc in its gutter."""

    def __init__(self, center=None):
        block = code_block(LOOP_SRC, width=CODE_W, title=CODE_TITLE)
        self.box, self.body, self.cap = block[0], block[1], block[2]
        self.body.shift(0.25 * m.RIGHT)
        self.cap.shift(0.25 * m.RIGHT)
        block.move_to(center if center is not None else [-6.8 + CODE_W / 2, CODE_TOP - self.box.height / 2, 0])
        self.body.set_z_index(2)
        self.cap.set_z_index(2)
        self.src = LOOP_SRC.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n
        self.bases = [baseline(g, line.replace(" ", "")) for g, line in zip(self.lines, self.src)]
        self.cap_base = baseline(self.cap, CODE_TITLE.replace(" ", ""))
        gx = self.box.get_left()[0] + 0.36
        self.arc = m.ArcBetweenPoints([gx, self.bases[9] + 0.08, 0], [gx, self.bases[0] + 0.08, 0], angle=-0.6)
        self.arc.set_stroke(FAINT, 3)
        self.flash_path = self.arc.copy()
        self.arc.add_tip(tip_length=0.14, tip_width=0.14)
        self.arc.set_z_index(1)
        self.group = m.VGroup(block, self.arc)

    def part(self, i, key):
        """The glyphs of `key` within code line i (spaces draw no glyphs)."""
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=CORAL):
        b = a if b is None else b
        top_base = self.cap_base if a == "cap" else self.bases[a]
        bot_base = self.cap_base if b == "cap" else self.bases[b]
        top, bot = top_base + 0.25, bot_base - 0.1
        left, right = self.box.get_left()[0] + 0.5, self.box.get_right()[0] - 0.1
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)

    def loop_back(self, bar):
        """Flash the gutter arc and move the cursor back to `while True:`."""
        flash = m.ShowPassingFlash(self.flash_path.copy().set_stroke(CORAL, 6), time_width=0.7, run_time=0.7)
        return m.AnimationGroup(flash, m.Transform(bar, self.bar(0), run_time=0.7))


class Slot:
    """One place on screen whose content is swapped (status line, tool output, a counter)."""

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


# ---------------------------------------------------------------------------- the episode


class Ep02TheAgentLoop(Episode):
    number = 2

    def lesson(self):
        self.hook()
        self.stateless()
        self.round_trip()
        self.loop()
        self.trajectory()
        self.envelopes()
        self.exits()
        self.around()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Recap of Episode 1: Otto fixed tinyshop's failing test by itself (pytest → read cart.py → edit → pytest). Who chose the steps, and what made it stop? One loop of about ten lines. Bug and test counts from harness_reference.py (illustrative).")
        bot = otto(1.3).move_to([-5.1, 0.9, 0])
        person = icon("user", 0.95).move_to([5.6, 0.55, 0])
        ask = bubble(text(TASK, 28, INK), AZURE, AZURE_FILL, tail="right")
        ask.move_to([2.1, 1.55, 0])
        ask.shift((person.get_left()[0] - 0.15 - ask[0].get_vertices()[2][0]) * m.RIGHT)

        steps = ["pytest -q", "read cart.py", "edit cart.py", "pytest -q"]
        chips = m.VGroup(*[chip(s, SUN, SUN_FILL, width=2.6, height=0.72, size=24, mono=True) for s in steps])
        chips.arrange(m.RIGHT, buff=0.7).move_to([0, -1.35, 0])
        arrows = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.1) for a, b in zip(chips, chips[1:])])
        first = code(RESULTS[0], 20, ROSE).next_to(chips[0], m.DOWN, buff=0.22)
        last = code(RESULTS[3], 20, MINT).next_to(chips[3], m.DOWN, buff=0.22)
        asks = m.VGroup(*[title_text("?", 40, SUN).move_to(a.get_center() + 0.65 * m.UP) for a in arrows])
        stop = text("why stop here?", 22, ROSE).next_to(last, m.DOWN, buff=0.2)

        loop = LoopCode()
        loop.group.scale(0.9).move_to([2.35, -0.05, 0])
        lines = pill(f"{len(LOOP_SRC.splitlines())} lines", SUN, SUN_FILL, 22)
        lines.next_to(loop.box, m.UP, buff=0.22).align_to(loop.box, m.RIGHT)
        inside = text("inside every agent", 26, MUTED).next_to(bot, m.DOWN, buff=0.45)

        self.say(
            "[In episode one, Otto fixed a failing test, all on its own.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP), lag_ratio=0.25),
        )
        self.say(
            "[It ran the tests], [read the cart file], [changed one line], [and ran the tests again.]",
            m.AnimationGroup(m.FadeIn(chips[0], shift=0.2 * m.UP), m.FadeIn(first, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(chips[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[1]), m.FadeIn(chips[2], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(arrows[2]), m.FadeIn(chips[3], shift=0.2 * m.UP), m.FadeIn(last, shift=0.1 * m.UP)),
        )
        self.say(
            "[Nobody scripted those steps.] [So who chose each one], [and what told Otto to stop?]",
            m.Circumscribe(chips, color=MUTED, buff=0.15),
            m.LaggedStart(*[m.FadeIn(q, scale=0.6) for q in asks], lag_ratio=0.25),
            m.FadeIn(stop, shift=0.1 * m.UP),
        )
        self.say(
            "[The answer is a loop], [about ten lines of code], [and it sits inside every agent.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(person, ask, chips, arrows, first, last, asks, stop), shift=0.2 * m.DOWN),
                m.FadeIn(loop.group, shift=0.2 * m.UP),
            ),
            m.FadeIn(lines, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.Circumscribe(bot, color=CORAL, buff=0.15), m.FadeIn(inside, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ stateless

    def stateless(self):
        self.next_section("1 · The stateless model", notes="Model APIs are stateless: each call sees only what it is sent, so the harness keeps the conversation (the messages array) and sends all of it on every call. Li, AI Agents in Depth, ch. 2: 'the Agent framework must include the relevant history in every request'. The 'Which tests?' reply is illustrative.")
        box = card(2.8, 2.5, CORAL, CORAL_FILL)
        brain = m.VGroup(icon(MODEL_ICON, 1.1), text("model", 30, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.22)
        model = m.VGroup(box, brain.move_to(box)).move_to([-3.6, 0.45, 0])
        forgets = pill("no memory between calls", ROSE, ROSE_FILL, 22).next_to(model, m.DOWN, buff=0.45)

        req = message("user", "Did the tests pass?", width=4.8, size=22).move_to([3.0, 1.45, 0])
        reply = message("assistant", "Which tests? I have no context.", width=4.8, size=22).move_to([3.0, -0.65, 0])
        a_in = arrow(req.get_left(), model.get_right() + 0.5 * m.UP, AZURE, buff=0.15)
        a_out = arrow(model.get_right() + 0.5 * m.DOWN, reply.get_left(), CORAL, buff=0.15)
        example = text("for example", 20, MUTED).next_to(reply, m.DOWN, buff=0.18).align_to(reply, m.RIGHT)

        frame = card(6.4, 4.15, EDGE, CARD, radius=0.22, stroke=2).move_to([3.3, 0.35, 0])
        head = m.VGroup(icon(GEAR_ICON, 0.48), text("harness", 26, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        head.move_to(frame.get_corner(m.UL) + [0.3, -0.42, 0], aligned_edge=m.LEFT)
        ctx = ContextColumn(center=(3.3, -0.35, 0), width=5.8, height=1.55, title="messages", gap=0.08)
        first = msg_row(MESSAGES[0])
        ctx.push(first)
        every = text("the whole list, sent on every call", 24, MUTED).next_to(frame, m.DOWN, buff=0.3)
        src = cite("Li, AI Agents in Depth · ch. 2").to_corner(m.DR, buff=0.4)

        self.say(
            "[First, the model itself.] [It remembers nothing between calls.]",
            m.AnimationGroup(self.new_header("01 · stateless", "The model remembers nothing"), m.FadeIn(model, shift=0.2 * m.UP)),
            m.FadeIn(forgets, shift=0.15 * m.UP),
        )
        self.say(
            "[Send it just: did the tests pass?] [It has no idea which tests you mean.]",
            m.AnimationGroup(m.FadeIn(req, shift=0.2 * m.LEFT), m.GrowArrow(a_in)),
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(reply, shift=0.2 * m.RIGHT), m.FadeIn(example)),
        )
        self.say(
            "[So the harness, the code around the model], [keeps the conversation in a list called messages], [and sends all of it, on every call.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(req, reply, a_in, a_out, example), shift=0.2 * m.DOWN), m.FadeIn(frame), m.FadeIn(head, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(ctx.frame), m.FadeIn(first, shift=0.2 * m.UP)),
            m.AnimationGroup(send(colors_of(MESSAGES[:1]), first.get_center(), model.get_center()), m.FadeIn(every, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ one round trip

    def round_trip(self):
        call, res = CALLS[0], MESSAGES[2]["content"][0]
        self.next_section("2 · One round trip", notes=f"The reply's content holds a tool_use block (id {call.id}); stop_reason 'tool_use' — 'Model raises hand: I need a tool' (learn-claude-code · s01). The harness, not the model, runs the command; the output returns as a tool_result block inside a USER message, tool_use_id = the call's id.")
        model = side_chip(MODEL_ICON, "model", CORAL, CORAL_FILL).move_to([-3.5, 1.95, 0])
        harness = side_chip(GEAR_ICON, "harness", EDGE, CARD).move_to([3.5, 1.95, 0])
        go = arrow(model.get_right() + 0.15 * m.UP, harness.get_left() + 0.15 * m.UP, SUN, buff=0.15)
        back = arrow(harness.get_left() + 0.17 * m.DOWN, model.get_right() + 0.17 * m.DOWN, MINT, buff=0.15)

        assistant_src = (
            '{"role": "assistant",\n "content": [{\n   "type": "tool_use",\n'
            f'   "id": "{call.id}",\n   "name": "{call.name}",\n'
            f'   "input": {{"command": "{call.input["command"]}"}}}}]}}'
        )
        user_src = (
            '{"role": "user",\n "content": [{\n   "type": "tool_result",\n'
            f'   "tool_use_id": "{res["tool_use_id"]}",\n   "content": "{res["content"]}"}}]}}'
        )
        reply = wire(assistant_src, SUN)
        reply.move_to([0, 0, 0]).align_to([-6.75, 0, 0], m.LEFT).align_to([0, 1.2, 0], m.UP)
        r_body = reply[1]
        glyphs(r_body, '"assistant"').set_color(CORAL)
        glyphs(r_body, f'"{call.id}"').set_color(SUN)
        result = wire(user_src, MINT)
        result.align_to([6.75, 0, 0], m.RIGHT).align_to([0, -0.65, 0], m.UP)
        u_body = result[1]
        glyphs(u_body, '"user"').set_color(AZURE)
        glyphs(u_body, f'"{res["tool_use_id"]}"').set_color(SUN)

        term = terminal([f"$ {call.input['command']}", f"! {res['content']}"], width=result.width, title="tinyshop")
        term.align_to(result, m.RIGHT).align_to([0, 1.2, 0], m.UP)
        stop = pill('stop_reason: "tool_use"', SUN, SUN_FILL, 22, mono=True)
        hand = icon(HAND_ICON, 0.8)
        raised = m.VGroup(hand, stop).arrange(m.RIGHT, buff=0.3).next_to(reply, m.DOWN, buff=0.35)
        raised.align_to(reply, m.LEFT).shift(0.1 * m.RIGHT)
        src = cite("learn-claude-code · s01").next_to(raised, m.DOWN, buff=0.3).align_to(raised, m.LEFT)

        self.say(
            "[Here's Otto's first call.] [The reply isn't an answer]: [it's a tool use block.]",
            m.AnimationGroup(self.new_header("02 · round trip", "The model raises its hand"), m.FadeIn(model, shift=0.2 * m.UP)),
            m.FadeIn(reply, target_position=model),
            m.Circumscribe(glyphs(r_body, '"tool_use"'), color=SUN, buff=0.06),
        )
        self.say(
            "[It names a tool], [gives the input], [and carries an ID.]",
            m.Circumscribe(glyphs(r_body, f'"{call.name}"'), color=SUN, buff=0.06),
            m.Circumscribe(glyphs(r_body, f'"{call.input["command"]}"'), color=SUN, buff=0.06),
            m.Circumscribe(glyphs(r_body, f'"{call.id}"'), color=SUN, buff=0.06),
        )
        self.say(
            "[Its stop reason is tool use.] [In Learn Claude Code's words], [the model raises its hand: I need a tool.]",
            m.FadeIn(stop, shift=0.15 * m.UP),
            m.FadeIn(src, shift=0.1 * m.UP),
            m.FadeIn(hand, shift=0.35 * m.UP, scale=0.8),
        )
        self.say(
            "[The model can't run anything itself.] [The harness does]: [it runs pytest], [one failed, twelve passed.]",
            m.Circumscribe(model, color=ROSE, buff=0.08),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(harness, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(term[:4]), m.FadeIn(term[4][0], shift=0.1 * m.UP)),
            m.FadeIn(term[4][1], shift=0.1 * m.UP),
        )
        self.say(
            "[The output goes back as a tool result], [inside a user message], [with the same ID], [so the model knows which call it answers.]",
            m.FadeIn(result, target_position=term),
            m.Circumscribe(glyphs(u_body, '"user"'), color=AZURE, buff=0.06),
            m.AnimationGroup(
                m.Circumscribe(glyphs(r_body, f'"{call.id}"'), color=SUN, buff=0.06),
                m.Circumscribe(glyphs(u_body, f'"{res["tool_use_id"]}"'), color=SUN, buff=0.06),
            ),
            m.AnimationGroup(m.GrowArrow(back), pulse(back, MINT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the loop in code

    def build_stage(self):
        st = NS()
        st.code = LoopCode()
        st.bar = st.code.bar("cap")
        st.ctx = ContextColumn(center=COL_CENTER, width=COL_W, height=COL_H, title="messages", gap=0.08)
        lead = code("len(messages) =", 20, MUTED)
        lead.align_to(st.ctx.panel, m.RIGHT).shift(0.45 * m.LEFT)
        lead.shift((baseline(st.ctx.title, "messages") - baseline(lead, "len(messages)=")) * m.UP)
        st.lead = lead
        st.count_at = lambda n: code(str(n), 20, SUN, weight=m.BOLD).next_to(lead, m.RIGHT, buff=0.15).align_to(lead, m.DOWN)
        st.count = Slot(self)

        y = -3.6 + 1.63 / 2
        st.hcard = card(5.6, 1.63, EDGE, CARD, radius=0.16, stroke=2).move_to([-4.0, y, 0])
        st.hlabel = code("harness", 20, MUTED).move_to(st.hcard.get_corner(m.UL) + [0.3, -0.33, 0], aligned_edge=m.LEFT)
        st.mcard = card(1.65, 1.63, CORAL, CORAL_FILL, radius=0.16, stroke=2).move_to([-0.225, y, 0])
        st.micon = icon(MODEL_ICON, 0.62).move_to(st.mcard).shift(0.22 * m.UP)
        st.mlabel = text("model", 20, INK, weight=m.SEMIBOLD).next_to(st.micon, m.DOWN, buff=0.12)
        st.model = m.VGroup(st.mcard, st.micon, st.mlabel)
        st.desk = m.VGroup(st.hcard, st.hlabel, st.model)
        st.status = Slot(self)
        st.out = Slot(self)
        st.out_at = np.array([st.hcard.get_left()[0] + 0.3, y - 0.2, 0])
        return st

    def status_at(self, st, k):
        n = SENT[k - 1]
        words = code(f"call {k} · {n} message{'s' * (n != 1)}", 20, MUTED, t2c={r"\d+": SUN})
        words.align_to(st.hcard, m.RIGHT).shift(0.3 * m.LEFT)
        return words.shift((st.hlabel[0].get_bottom()[1] - words[0].get_bottom()[1]) * m.UP)

    def output(self, st, lines):
        """Tool output in the harness card: (string, colour) pairs, one per line."""
        rows = m.VGroup(*[code(s, 20, c, t2c={r"^\$": MINT}) for s, c in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        return rows.move_to(st.out_at, aligned_edge=m.LEFT)

    def push(self, st, i, source):
        """Message i joins the array (flying from `source`), and the count goes up."""
        r = msg_row(MESSAGES[i])
        st.ctx.push(r)  # positions the row (all ten fit: nothing scrolls, so its animation is not needed)
        arrive = m.FadeIn(r, shift=0.25 * m.UP) if source is None else m.FadeIn(r, target_position=source)
        return m.AnimationGroup(arrive, st.count.to(st.count_at(i + 1))), r

    def call(self, st, k):
        """Call k: the cursor on the create call, the status line, the array flying to the model."""
        n = SENT[k - 1]
        rows = m.VGroup(*st.ctx.items)
        return m.AnimationGroup(
            m.Transform(st.bar, st.code.bar(1, 2)),
            st.status.to(self.status_at(st, k)),
            send(colors_of(MESSAGES[:n]), rows.get_center(), st.micon.get_center()),
        )

    def loop(self):
        self.next_section("3 · The loop in code", notes=f"harness_reference.agent_loop, stepped (the scene runs this exact code against a scripted model). {len(SENT)} calls send {SENT} messages; {TOOL_USES} tool runs; the array ends with {len(MESSAGES)} messages; the last stop_reason is end_turn. tool_results() = the reference's inner for-loop, named so the loop fits in ten lines. Otto's only tool here is bash (s01); Episode 3 adds read_file/edit_file. The model's choices are illustrative.")
        st = self.st = self.build_stage()
        code_, bar = st.code, st.bar

        self.say(
            "[Now the code.] [Here's the whole loop, from our reference harness], [beside the list it builds.]",
            self.new_header("03 · the loop", "Ten lines, five calls"),
            m.FadeIn(code_.group, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(st.ctx.frame), m.FadeIn(st.lead), m.FadeIn(st.desk, shift=0.15 * m.UP)),
        )
        a0, r0 = self.push(st, 0, None)
        self.say(
            "[It starts with one message]: [Otto's task.]",
            m.FadeIn(bar),
            a0,
        )
        # call one, stepped line by line
        self.say(
            "[Call one] sends the whole list to the model, [just one message so far.]",
            self.call(st, 1),
            m.Circumscribe(r0, color=AZURE, buff=0.05),
        )
        a1, r1 = self.push(st, 1, st.model)
        self.say(
            "[The reply is appended], as an assistant message: [run pytest.]",
            m.AnimationGroup(m.Transform(bar, code_.bar(3, 4)), a1),
            m.Circumscribe(r1[3], color=SUN, buff=0.06),
        )
        hand = icon(HAND_ICON, 0.55)
        stop = pill('stop_reason: "tool_use"', SUN, SUN_FILL, 22, mono=True)
        raised = m.VGroup(hand, stop).arrange(m.RIGHT, buff=0.2).move_to(st.out_at, aligned_edge=m.LEFT)
        self.say(
            "[Its stop reason is tool use], [so the loop doesn't return yet.]",
            m.AnimationGroup(m.Transform(bar, code_.bar(5)), st.out.to(raised)),
            m.Circumscribe(code_.lines[6], color=ROSE, buff=0.06),
        )
        run1 = self.output(st, [("$ pytest -q", INK), (RESULTS[0], ROSE)])
        self.say(
            "[The harness runs the tool]: [one failed, twelve passed.]",
            m.AnimationGroup(m.Transform(bar, code_.bar(7)), st.out.to(None), m.FadeIn(run1[0], shift=0.1 * m.UP)),
            m.FadeIn(run1[1], shift=0.1 * m.UP),
        )
        st.out.current = run1
        a2, r2 = self.push(st, 2, run1)
        self.say(
            "[The result goes back] as one user message, [and the loop turns again.]",
            m.AnimationGroup(m.Transform(bar, code_.bar(8, 9)), a2),
            code_.loop_back(bar),
        )
        # calls two to four, one sweep each
        a3, r3 = self.push(st, 3, st.model)
        cart = self.output(st, [(f"$ {READ}", INK), (BUG_LINE, INK)])
        a4, r4 = self.push(st, 4, cart)
        self.say(
            "[Call two sends three messages.] [The model asks for the cart file], [and the harness reads it back.]",
            m.AnimationGroup(self.call(st, 2), st.out.to(None)),
            m.AnimationGroup(m.Transform(bar, code_.bar(3, 4)), a3),
            m.Succession(
                m.Transform(bar, code_.bar(5), run_time=0.35),
                m.AnimationGroup(m.Transform(bar, code_.bar(7)), st.out.to(cart), run_time=0.6),
                m.AnimationGroup(m.Transform(bar, code_.bar(8, 9)), a4, run_time=0.8),
            ),
        )
        minus = glyphs(cart[1], "- 1")
        self.say(
            "[There's the bug]: [the loop stops one item early.]",
            m.Circumscribe(cart[1], color=ROSE, buff=0.06),
            m.AnimationGroup(minus.animate.set_color(ROSE), m.Circumscribe(minus, color=ROSE, buff=0.05)),
        )
        a5, r5 = self.push(st, 5, st.model)
        diff = self.output(st, [(f"$ {SED}", INK), (RESULTS[2], MUTED)])
        a6, r6 = self.push(st, 6, diff)
        self.say(
            "[Call three edits that one line], [with a single shell command.]",
            m.Succession(
                code_.loop_back(bar),
                m.AnimationGroup(self.call(st, 3), st.out.to(None)),
                m.AnimationGroup(m.Transform(bar, code_.bar(3, 4)), a5, run_time=0.8),
            ),
            m.Succession(
                m.Transform(bar, code_.bar(5), run_time=0.35),
                m.AnimationGroup(m.Transform(bar, code_.bar(7)), st.out.to(diff), run_time=0.6),
                m.AnimationGroup(m.Transform(bar, code_.bar(8, 9)), a6, run_time=0.8),
            ),
        )
        a7, r7 = self.push(st, 7, st.model)
        run2 = self.output(st, [("$ pytest -q", INK), (RESULTS[3], MINT)])
        a8, r8 = self.push(st, 8, run2)
        self.say(
            "[Call four runs the tests again]: [thirteen passed.]",
            m.Succession(
                code_.loop_back(bar),
                m.AnimationGroup(self.call(st, 4), st.out.to(None)),
                m.AnimationGroup(m.Transform(bar, code_.bar(3, 4)), a7, run_time=0.8),
            ),
            m.Succession(
                m.Transform(bar, code_.bar(5), run_time=0.35),
                m.AnimationGroup(m.Transform(bar, code_.bar(7)), st.out.to(run2), run_time=0.6),
                m.AnimationGroup(m.Transform(bar, code_.bar(8, 9)), a8, run_time=0.8),
            ),
        )
        a9, r9 = self.push(st, 9, st.model)
        self.say(
            "[Call five sends all nine messages], [and this time, the model answers in words.]",
            m.Succession(code_.loop_back(bar), m.AnimationGroup(self.call(st, 5), st.out.to(None))),
            m.AnimationGroup(m.Transform(bar, code_.bar(3, 4)), a9),
        )
        end = pill(f'stop_reason: "{ANSWER.stop_reason}"', CORAL, CORAL_FILL, 22, mono=True).move_to(st.out_at, aligned_edge=m.LEFT)
        done = code(f"done · {len(SENT)} calls · {len(MESSAGES)} messages", 20, MINT, t2c={r"\d+": SUN}).move_to(st.out_at, aligned_edge=m.LEFT)
        self.say(
            "[The stop reason is end turn], [not tool use], [so the loop returns], [and the job is done.]",
            m.AnimationGroup(m.Transform(bar, code_.bar(5)), st.out.to(end)),
            m.Circumscribe(code_.part(5, '"tool_use"'), color=CORAL, buff=0.06),
            m.Transform(bar, code_.bar(6, color=MINT)),
            m.AnimationGroup(st.out.to(done), m.Circumscribe(r9, color=CORAL, buff=0.05)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ trajectory

    def trajectory(self):
        st = self.st
        self.next_section("4 · Append-only", notes=f"The array only grows: nothing edited, nothing removed — the trajectory. Li, AI Agents in Depth, ch. 1: 'The essence of the ReAct loop is appending to the trajectory, over and over.' Calls sent {SENT} messages = {sum(SENT)} for a {len(MESSAGES)}-message list (computed). Prompt caching: Episode 7.")
        left_x = -3.15
        only = pill("append-only", MINT, MINT_FILL, 24).move_to([left_x, 1.95, 0])
        never = text("nothing edited · nothing removed", 24, MUTED).next_to(only, m.DOWN, buff=0.3)
        name = title_text("the trajectory", 48, CORAL).move_to([left_x, 0.25, 0])
        quote = text("“The essence of the ReAct loop is appending\nto the trajectory, over and over.”", 24, INK)
        quote.next_to(name, m.DOWN, buff=0.45)
        src = cite("Li, AI Agents in Depth · ch. 1").next_to(quote, m.DOWN, buff=0.22).align_to(quote, m.RIGHT)

        heading = text("sent to the model, call by call", 24, MUTED).move_to([left_x, 2.2, 0])
        colors = colors_of(MESSAGES)
        rows = m.VGroup()
        for k, n in enumerate(SENT, start=1):
            y = 1.55 - 0.62 * (k - 1)
            label = code(f"call {k}", 20, MUTED).move_to([-6.5, y, 0], aligned_edge=m.LEFT)
            blocks = m.VGroup(*[
                m.RoundedRectangle(corner_radius=0.05, width=0.3, height=0.38, stroke_width=0, fill_color=c, fill_opacity=1)
                for c in colors[:n]
            ]).arrange(m.RIGHT, buff=0.07).move_to([-5.2, y, 0], aligned_edge=m.LEFT)
            count = code(f"{n} message{'s' * (n != 1)}", 20, INK).move_to([-1.75, y, 0], aligned_edge=m.LEFT)
            rows.add(m.VGroup(label, blocks, count))
        total = title_text(f"{sum(SENT)} messages sent", 44, SUN).move_to([left_x, -1.75, 0])
        kept = text(f"for a list of {len(MESSAGES)}", 26, MUTED).next_to(total, m.DOWN, buff=0.18)
        nxt = pill("Episode 7 · KV Cache & Prompt Caching", SUN, SUN_FILL, 22).move_to([left_x, -3.15, 0])

        self.say(
            "[Look at the list now.] [Each step only appended to it.] [Nothing was edited, and nothing removed.]",
            m.AnimationGroup(self.new_header("04 · trajectory", "Append, never edit"), m.FadeOut(m.VGroup(st.code.group, st.bar, st.desk, st.status.current, st.out.current), shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Circumscribe(st.ctx.panel, color=MINT, buff=0.06), m.FadeIn(only, shift=0.15 * m.UP)),
            m.FadeIn(never, shift=0.1 * m.UP),
        )
        self.say(
            "[This list is the agent's trajectory.] [The book AI Agents in Depth calls appending to it, over and over, the essence of the loop.]",
            m.FadeIn(name, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[And because the model is stateless], [each call re-sent the list so far]: [one message], [then three], [five], [seven], [and nine.]",
            m.FadeOut(m.VGroup(only, never, name, quote, src), shift=0.2 * m.UP),
            m.FadeIn(heading, shift=0.1 * m.UP),
            *[m.AnimationGroup(m.FadeIn(r[0]), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.RIGHT) for b in r[1]], lag_ratio=0.12), m.FadeIn(r[2])) for r in rows],
        )
        self.say(
            "[That's twenty-five messages sent, for a list of ten.] [Episode seven shows how caching keeps that affordable.]",
            m.AnimationGroup(m.FadeIn(total, shift=0.15 * m.UP), m.FadeIn(kept, shift=0.1 * m.UP)),
            m.FadeIn(nxt, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ envelopes

    def envelopes(self):
        self.next_section("5 · Two envelopes", notes="Several tool_use blocks in one reply: the harness runs them all and returns ALL results in ONE user message of tool_result blocks, each tool_use_id = its call's id (Anthropic-style, harness_reference). OpenAI-style APIs: the assistant message has content null and tool_calls; each result is its own message with role 'tool' and tool_call_id (Li, AI Agents in Depth, ch. 2; ids call_abc/call_def from the book's example). The two-file read is illustrative.")
        width = 6.3
        panels = m.VGroup(*[card(width, 5.3, EDGE, "#18150F", radius=0.2, stroke=2) for _ in range(2)])
        panels.arrange(m.RIGHT, buff=0.45).move_to([0, -0.18, 0])
        titles = m.VGroup(
            title_text("Anthropic-style", 28, INK),
            title_text("OpenAI-style", 28, INK),
        )
        for t, p in zip(titles, panels):
            t.move_to(p.get_corner(m.UL) + [0.32, -0.42, 0], aligned_edge=m.LEFT)
        inner = width - 0.5
        a_call = role_msg("assistant", CORAL, "tool_use · toolu_02 · bash\ntool_use · toolu_03 · bash", inner)
        a_res = role_msg("user", AZURE, "tool_result · toolu_02\ntool_result · toolu_03", inner)
        left = m.VGroup(a_call, a_res).arrange(m.DOWN, buff=0.18).next_to(titles[0], m.DOWN, buff=0.3)
        left.align_to(panels[0], m.LEFT).shift(0.25 * m.RIGHT)
        o_call = role_msg("assistant", CORAL, "content: null\ntool_calls: call_abc, call_def", inner)
        o_res1 = role_msg("tool", MINT, "tool_call_id: call_abc", inner)
        o_res2 = role_msg("tool", MINT, "tool_call_id: call_def", inner)
        right = m.VGroup(o_call, o_res1, o_res2).arrange(m.DOWN, buff=0.15).next_to(titles[1], m.DOWN, buff=0.3)
        right.align_to(panels[1], m.LEFT).shift(0.25 * m.RIGHT)
        note_l = text("all results, one user message", 22, MUTED).next_to(left, m.DOWN, buff=0.3).align_to(left, m.LEFT)
        note_r = text("one message per result", 22, MUTED).next_to(right, m.DOWN, buff=0.24).align_to(right, m.LEFT)
        same = pill("same loop", CORAL, CORAL_FILL, 24).move_to([0, -3.32, 0])
        ids_call = m.VGroup(glyphs(a_call[2][1], "toolu_02"), glyphs(a_call[2][1], "toolu_03"))
        ids_res = m.VGroup(glyphs(a_res[2][1], "toolu_02"), glyphs(a_res[2][1], "toolu_03"))

        self.say(
            "[Sometimes the model asks for two tools in one reply], [say, to read two files.]",
            m.AnimationGroup(self.new_header("05 · envelopes", "Same loop, different envelope"), m.FadeIn(panels[0]), m.FadeIn(titles[0])),
            m.FadeIn(a_call, shift=0.15 * m.UP),
        )
        self.say(
            "[The harness runs both], [and sends every result back together, in one user message], [each with its call's ID.]",
            m.Circumscribe(a_call, color=SUN, buff=0.05),
            m.AnimationGroup(m.FadeIn(a_res, shift=0.15 * m.UP), m.FadeIn(note_l)),
            m.AnimationGroup(*[m.Circumscribe(g, color=SUN, buff=0.05) for g in (*ids_call, *ids_res)]),
        )
        self.say(
            "[OpenAI-style APIs use a different envelope]: [each result is its own message], [with the role tool.]",
            m.AnimationGroup(m.FadeIn(panels[1]), m.FadeIn(titles[1]), m.FadeIn(o_call, shift=0.15 * m.UP)),
            m.LaggedStart(m.FadeIn(o_res1, shift=0.15 * m.UP), m.FadeIn(o_res2, shift=0.15 * m.UP), m.FadeIn(note_r), lag_ratio=0.35),
            m.AnimationGroup(m.Circumscribe(o_res1[2][0], color=MINT, buff=0.05), m.Circumscribe(o_res2[2][0], color=MINT, buff=0.05)),
        )
        self.say(
            "[Different envelope], [same loop.]",
            m.AnimationGroup(m.Circumscribe(a_res, color=AZURE, buff=0.05), m.Circumscribe(m.VGroup(o_res1, o_res2), color=MINT, buff=0.05)),
            m.FadeIn(same, scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ exits

    def exits(self):
        self.next_section("6 · Exits and the iteration cap", notes="The loop's only exit is a reply without a tool call. With no cap an agent can repeat the same call forever (Li, AI Agents in Depth, ch. 2: 'Production code needs a max_iterations cap'). Real harnesses add max turns, token budgets and more exits; Claude Code's src/query.ts is 1729 lines whose core is the 30-line while True, with exit paths for prompt too long, abort, hook stop, max turns, token budget (learn-claude-code · s01, the site's reading of CC's source; not verified by us). The stuck loop is illustrative.")
        lc = LoopCode()
        bar = lc.bar(5, 6, color=MINT)
        # line 1 as its own text, so it can be replaced
        first = code(lc.src[0], 20, INK, t2c=CODE_T2C)
        first.move_to(lc.lines[0], aligned_edge=m.LEFT)
        first.shift((lc.bases[0] - baseline(first, lc.src[0].replace(" ", ""))) * m.UP)
        lc.body.remove(*lc.lines[0])
        capped = code("for turn in range(MAX_TURNS):", 20, INK, t2c=CODE_T2C)
        capped.move_to(lc.lines[0], aligned_edge=m.LEFT)
        capped.shift((lc.bases[0] - baseline(capped, "forturninrange(MAX_TURNS):")) * m.UP)

        ctx = ContextColumn(center=COL_CENTER, width=COL_W, height=COL_H, title="messages", gap=0.08, meter=True)
        tokens = code("tokens", 20, MUTED).next_to(ctx.meter, m.LEFT, buff=0.2)
        start = [ctx.push(msg_row(MESSAGES[i])) for i in range(3)]
        call_row = lambda: msg_row(MESSAGES[1])  # noqa: E731 — the same call, again
        fail_row = lambda: msg_row(MESSAGES[2])  # noqa: E731 — the same failure, again

        def pair():
            return m.AnimationGroup(ctx.push(call_row()), ctx.push(fail_row()), lag_ratio=0.5)

        strip_y = -3.6 + 1.63 / 2
        turn = Slot(self)
        turn_at = lambda n: title_text(f"turn {n}", 44, ROSE).move_to([-5.35, strip_y, 0])  # noqa: E731
        same = text("same call, same failure", 24, MUTED).move_to([-2.0, strip_y, 0])
        src_b = cite("Li, AI Agents in Depth · ch. 2").move_to([-2.0, strip_y - 0.55, 0])

        exit_names = ["max turns", "token budget", "prompt too long", "abort", "hook stop"]
        exits = m.VGroup(*[chip(e, MINT, MINT_FILL, width=text(e, 22, weight=m.SEMIBOLD).width + 0.5, height=0.56, size=22) for e in exit_names])
        exits[:3].arrange(m.RIGHT, buff=0.22)
        exits[3:].arrange(m.RIGHT, buff=0.22)
        m.VGroup(exits[:3], exits[3:]).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to([-3.1, strip_y, 0])

        where = code("Claude Code · src/query.ts", 22, MUTED).move_to([3.75, 1.55, 0])
        big = title_text("1,729 lines", 72, INK).next_to(where, m.DOWN, buff=0.35)
        core = m.VGroup(text("around a 30-line", 30, MUTED), code("while True", 30, ORCHID)).arrange(m.RIGHT, buff=0.2)
        core.next_to(big, m.DOWN, buff=0.45)
        src_a = cite("learn-claude-code · s01 · reading CC's source").next_to(core, m.DOWN, buff=0.55)

        self.say(
            "[Now, the exits.] [Our loop has just one way out]: [a reply that doesn't ask for a tool.]",
            m.AnimationGroup(self.new_header("06 · exits", "A loop needs a way out"), m.FadeIn(lc.group), m.FadeIn(first), m.FadeIn(ctx.frame), m.FadeIn(tokens), *start),
            m.FadeIn(bar),
            m.Circumscribe(lc.part(5, '!= "tool_use"'), color=MINT, buff=0.06),
        )
        self.say(
            "[What if that reply never comes?] [Say the test keeps failing], [and the model keeps running the same command.]",
            m.AnimationGroup(lc.loop_back(bar), turn.to(turn_at(2)), pair(), ctx.meter.fill_to(0.42)),
            m.AnimationGroup(turn.to(turn_at(3)), pair(), m.FadeIn(same), ctx.meter.fill_to(0.56)),
            m.AnimationGroup(turn.to(turn_at(4)), pair(), ctx.meter.fill_to(0.7)),
        )
        self.say(
            "[With no cap], [an agent can get stuck], [repeating the same calls forever.]",
            m.AnimationGroup(turn.to(turn_at(5)), ctx.push(call_row()), ctx.meter.fill_to(0.8)),
            m.AnimationGroup(ctx.push(fail_row()), m.FadeIn(src_b), ctx.meter.fill_to(0.88)),
            m.AnimationGroup(m.Succession(*[turn.to(turn_at(n), shift=0.1) for n in range(6, 10)], run_time=1.6), ctx.meter.fill_to(0.97, run_time=1.6)),
        )
        self.say(
            "[So real harnesses add more exits]: [a maximum number of turns], [a token budget], [and more.]",
            m.AnimationGroup(
                turn.to(None), m.FadeOut(m.VGroup(same, src_b)),
                m.FadeOut(first, shift=0.2 * m.UP), m.FadeIn(capped, shift=0.2 * m.UP),
                m.Transform(bar, lc.bar(0, color=SUN)),
            ),
            m.FadeIn(exits[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(exits[1], shift=0.15 * m.UP), m.Circumscribe(ctx.meter, color=SUN, buff=0.08)),
            m.LaggedStart(*[m.FadeIn(e, shift=0.15 * m.UP) for e in exits[2:]], lag_ratio=0.3),
        )
        self.say(
            "[In Learn Claude Code's reading of its source], [Claude Code's query loop runs to one thousand seven hundred and twenty-nine lines.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ctx.frame, tokens, *ctx.items), shift=0.2 * m.RIGHT), m.FadeIn(where, shift=0.1 * m.UP), m.FadeIn(src_a)),
            m.FadeIn(big, scale=0.85),
        )
        self.say(
            "[Yet at its core is a thirty-line while true]: [the same loop you just saw.]",
            m.FadeIn(core, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(lc.box, color=CORAL, buff=0.08), lc.loop_back(bar)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ around the loop

    def around(self):
        self.next_section("7 · Around the loop", notes="'The loop itself never changes' (learn-claude-code · s01): later episodes add things AROUND it — tools (Ep 3), the context window (Ep 6), memory (Ep 12), permissions (Ep 17), hooks (Ep 18), agent teams (Ep 24).")
        ring = loop_ring(["call", "run", "append"], radius=1.5, colors=[CORAL, SUN, MINT], size=26, center=(0, -0.15, 0))
        quote = title_text("“The loop itself never changes.”", 34, INK).move_to([0, -2.75, 0])
        src = cite("learn-claude-code · s01").next_to(quote, m.DOWN, buff=0.2)
        later = [("tools", 3), ("context", 6), ("memory", 12), ("permissions", 17), ("hooks", 18), ("teams", 24)]
        chips = m.VGroup()
        for name, ep in later:
            accent = MODULES[EPISODES[ep][1]][1]
            box = card(3.25, 0.72, EDGE, CARD, radius=0.16, stroke=2)
            body = m.VGroup(text(name, 24, INK, weight=m.SEMIBOLD), code(f"ep {ep}", 20, accent, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
            chips.add(m.VGroup(box, body.move_to(box)))
        for i, c in enumerate(chips):
            side = -1 if i < 3 else 1
            c.move_to([side * 4.95, 1.15 - 1.25 * (i % 3), 0])
        links = m.VGroup(*[
            m.DashedLine(c.get_right() if i < 3 else c.get_left(), ring.get_center() + (np.array([-1, 0, 0]) if i < 3 else np.array([1, 0, 0])) * 2.35 + (c.get_y() - ring.get_y()) * 0.45 * m.UP, dash_length=0.08).set_stroke(FAINT, 2)
            for i, c in enumerate(chips)
        ])

        self.say(
            "[As Learn Claude Code puts it], [the loop itself never changes.]",
            m.AnimationGroup(self.new_header("07 · around the loop", "The loop never changes"), m.FadeIn(ring[0], lag_ratio=0.2), m.Create(ring[1])),
            m.AnimationGroup(m.FadeIn(quote, shift=0.15 * m.UP), m.FadeIn(src), m.Succession(*[m.Indicate(n, color=n[0].get_stroke_color(), scale_factor=1.1) for n in ring[0]])),
        )
        self.say(
            "[Every later episode adds something around it]: [tools], [context and memory], [permissions and hooks], [and whole teams of agents.]",
            m.Succession(*[m.Indicate(n, color=n[0].get_stroke_color(), scale_factor=1.1, run_time=0.5) for n in ring[0]]),
            m.AnimationGroup(m.FadeIn(chips[0], shift=0.2 * m.RIGHT), m.Create(links[0])),
            m.AnimationGroup(*[m.FadeIn(chips[i], shift=0.2 * m.RIGHT) for i in (1, 2)], *[m.Create(links[i]) for i in (1, 2)]),
            m.AnimationGroup(*[m.FadeIn(chips[i], shift=0.2 * m.LEFT) for i in (3, 4)], *[m.Create(links[i]) for i in (3, 4)]),
            m.AnimationGroup(m.FadeIn(chips[5], shift=0.2 * m.LEFT), m.Create(links[5])),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Call, run, append — and repeat until the model stops asking for tools.")
        halo = glow(CORAL, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Call", CORAL), ("Run", SUN), ("Append", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("client.messages.create", "tool_results(response)", "messages.append")])
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
        tagline = text("Repeat until the model stops asking for tools.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [call], [run], [append.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Then repeat, until the model stops asking for tools.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
