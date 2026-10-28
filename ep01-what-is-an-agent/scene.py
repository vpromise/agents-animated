"""Agents, Animated · Episode 1 · What Is an Agent?  (Module 1 · Foundations)

The course opener. Running example: Otto, the series' agent, is asked "The checkout test fails. Fix
it." about tinyshop, a tiny repo whose cart.total() skips the last item (tests/test_cart.py::test_total
expects total([3, 4]) == 7 and gets 3; pytest -q says "1 failed, 12 passed").

- Hook: meet Otto and tinyshop; one failing test; the request.
- A chatbot: a language model alone answers at once from memory with a plausible fix in the wrong
  file (checkout.py); it never saw the code, the tests or the error, and cannot check: one model
  call, zero tool calls.
- An agent: the same model with tools (bash, read_file, edit_file) and a loop runs the tests, reads
  cart.py, edits the bug, re-runs (13 passed) and only then answers: five model calls, four tool
  calls; the context column fills (and scrolls) with every call and result.
- Chatbot vs fixed script vs agent: who chooses the next step, who can act.
- Li's formula (AI Agents in Depth): Agent = LLM + Context + Tools; brain, eyes, hands; policy,
  observation, action — "not one-to-one" (a tool such as read_file is also how the agent looks).
- Observation and action spaces: what is outside the first does not exist for the model; an action
  outside the second can only be recommended in words (the chatbot's situation).
- Agent = Model + Harness: the harness builds context, exposes tools, keeps the loop (shown in
  harness_reference.agent_loop), and applies permissions, verification and correction (Eps 17/18/20).
- The reins: "harness" first meant a horse's reins and tack; LangChain's 52.8% -> 66.5% on
  Terminal-Bench 2.0 from harness changes only (as reported in Li, ch. 1; no primary URL given).
- Recap (brain, eyes, hands) and a map of the course's six modules.

Colours (series-wide): user AZURE, model/assistant CORAL, tool call SUN, tool result / observation
MINT, good MINT, bad ROSE, fixed code and the harness frame neutral (INK/MUTED). Headers CORAL
(Module 1).

Illustrative: the chatbot's reply, the exact terminal text, tinyshop's code (from
harness_reference.py's docstring), and the things outside Otto's spaces. Asserts below tie every
number the narration quotes to the data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep01"

# ---------------------------------------------------------------------------- the running example

TASK = "The checkout test fails. Fix it."
BUGGY_SRC = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s"
FIXED_SRC = "def total(items):\n    s = 0\n    for item in items:\n        s += item\n    return s"
BUG_LINES = ("for i in range(len(items) - 1):", "s += items[i]")
FIX_LINES = ("for item in items:", "s += item")


def _run(src, items):
    space = {}
    exec(src, space)
    return space["total"](items)


TEST_ITEMS, EXPECTED = [3, 4], 7
GOT = _run(BUGGY_SRC, TEST_ITEMS)
assert GOT == 3 and _run(FIXED_SRC, TEST_ITEMS) == EXPECTED  # the bug skips the last item; the fix sums all
N_TESTS, N_FAILED = 13, 1
BEFORE = f"{N_FAILED} failed, {N_TESTS - N_FAILED} passed"
AFTER = f"{N_TESTS} passed"
assert BEFORE == "1 failed, 12 passed" and AFTER == "13 passed"  # "one failed, twelve passed"; "thirteen passed"

# Otto's run: every message the harness appends after the request (kind, tool, body)
RUN = [
    ("tool_use", "bash", 'bash("pytest -q")'),
    ("tool_result", None, f"FAILED tests/test_cart.py::test_total\n{BEFORE}"),
    ("tool_use", "read_file", 'read_file("cart.py")'),
    ("tool_result", None, f"{BUG_LINES[0]}\n    {BUG_LINES[1]}"),
    ("tool_use", "edit_file", 'edit_file("cart.py", ...)'),
    ("tool_result", None, "edited cart.py"),
    ("tool_use", "bash", 'bash("pytest -q")'),
    ("tool_result", None, AFTER),
    ("assistant", None, "Fixed: total() skipped the last item."),
]
TOOL_CALLS = sum(kind == "tool_use" for kind, _, _ in RUN)
MODEL_CALLS = sum(kind in ("tool_use", "assistant") for kind, _, _ in RUN)  # each tool_use or answer is one model reply
assert (MODEL_CALLS, TOOL_CALLS) == (5, 4)  # "five model calls, four tool calls"
CHATBOT_CALLS = (1, 0)  # the chatbot: one model call, no tool calls

# ---------------------------------------------------------------------------- reported numbers

TB_BEFORE, TB_AFTER = 52.8, 66.5  # LangChain on Terminal-Bench 2.0, as reported in Li, AI Agents in Depth, ch. 1
TB_GAIN = round(TB_AFTER - TB_BEFORE, 1)
assert TB_GAIN == 13.7  # "+13.7 points" on screen

# ---------------------------------------------------------------------------- the course map

MODULE_LINES = {
    1: "the loop, tools, and when to use one",
    2: "what goes into the window, and why",
    3: "memory across sessions, search, files",
    4: "good tools, MCP, permissions, hooks",
    5: "subagents, task graphs, agent teams",
    6: "computer use, evals, training, production",
}
RANGES = {k: (min(n for n, (_, mod) in EPISODES.items() if mod == k), max(n for n, (_, mod) in EPISODES.items() if mod == k)) for k in MODULES}
assert RANGES == {1: (1, 5), 2: (6, 11), 3: (12, 14), 4: (15, 20), 5: (21, 26), 6: (27, 33)}

# ---------------------------------------------------------------------------- helpers

LLM = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
SCRIPT = recolored("code", MUTED, CARD_HI, prefix=PREFIX)


def cite(s):
    return text(s, 20, MUTED)


def top_cite(s):
    """A source in small muted text, top right, level with the header."""
    return cite(s).to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)


def mono_chip(word, color, fill, size=22, height=0.6, pad=0.45):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def word_chip(word, color, fill, size=22, height=0.6, pad=0.5):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def put_head(bot, point):
    """Move an otto() so that its head (not the antenna) is centred on `point`."""
    return bot.shift(np.array(point) - bot[0].get_center())


class Column(ContextColumn):
    """The kit's ContextColumn, but one say() line may push several messages (a call, then its
    result): positions are tracked as targets and scroll moves are absolute, so a second push built
    before the first has played still lands in the right place. When it scrolls, the survivors move
    up to the top of the panel (no gap above them)."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.slots = []  # [message, its top y once every built push has played]

    def push(self, msg, run_time=0.9):
        room = self.top - self.bottom
        total = sum(item.height + self.gap for item, _ in self.slots) + msg.height
        leaving = []
        while self.slots and total > room + 1e-6:  # the oldest messages scroll out
            item, _ = self.slots.pop(0)
            leaving.append(item)
            total -= item.height + self.gap
        anims = []
        if leaving:  # the survivors move up to the top of the panel
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
        anims.append(m.FadeIn(msg, shift=0.25 * m.UP))
        self.slots.append([msg, y])
        self.items = [item for item, _ in self.slots]
        return m.AnimationGroup(*anims, run_time=run_time)


# ---------------------------------------------------------------------------- the episode


class Ep01WhatIsAnAgent(Episode):
    number = 1

    def lesson(self):
        self.meet_otto()
        self.chatbot()
        self.agent()
        self.difference()
        self.formula()
        self.spaces()
        self.harness()
        self.reins()
        self.recap()
        self.course()

    # ------------------------------------------------------------------ hook

    def meet_otto(self):
        self.next_section(
            "Meet Otto",
            notes="Otto is the series' agent; tinyshop is its repo (cart.py, checkout.py, tests/). One test fails "
            "(tests/test_cart.py::test_total). The user's words are loose on purpose: they say 'checkout'.",
        )
        bot = put_head(otto(1.35), [-3.9, 1.0, 0])
        halo = glow(CORAL, radius=2.3).move_to(bot[0])
        name = title_text("Otto", 50).next_to(bot, m.DOWN, buff=0.4)
        sub = text("the agent we build in this course", 24, MUTED).next_to(name, m.DOWN, buff=0.16)

        tree = file_tree(
            ["tinyshop/", "tinyshop/cart.py", "tinyshop/checkout.py", "tinyshop/tests/",
             "tinyshop/tests/test_cart.py", "tinyshop/tests/test_checkout.py"],
            size=24,
        )
        repo = card(4.9, tree.height + 0.9).move_to([3.3, 0.95, 0])
        tree.move_to(repo).align_to(repo, m.LEFT).shift(0.45 * m.RIGHT)
        failing = code("← 1 failing", 20, ROSE, weight=m.BOLD).next_to(tree[3], m.RIGHT, buff=0.3)

        user = icon("user", 0.85)
        ask_words = text(TASK, 28, INK)
        ask = bubble(ask_words, AZURE, AZURE_FILL, tail="left")
        row = m.VGroup(user, ask).arrange(m.RIGHT, buff=0.35).move_to([0, -2.45, 0])
        ask.shift((user.get_y() - ask[1].get_y()) * m.UP)

        self.say(
            "[This is Otto]: [the coding agent we'll build, piece by piece, in this course.]",
            m.AnimationGroup(self.new_header("MEET OTTO", "One agent, one bug"), m.FadeIn(halo, scale=0.6), m.FadeIn(bot, shift=0.3 * m.UP, scale=0.8)),
            m.AnimationGroup(m.FadeIn(name, shift=0.15 * m.UP), m.FadeIn(sub, shift=0.15 * m.UP)),
        )
        self.say(
            "[Otto works in a small repository called tinyshop]: [a cart, a checkout, and their tests.]",
            m.AnimationGroup(m.FadeIn(repo), m.FadeIn(tree[0], shift=0.1 * m.RIGHT)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in tree[1:]], lag_ratio=0.25),
        )
        self.say(
            "[One test is failing], [so we ask Otto]: [the checkout test fails, fix it.]",
            m.AnimationGroup(m.FadeIn(failing, scale=1.2), tree[3].animate.set_color(ROSE)),
            m.AnimationGroup(m.FadeIn(user, shift=0.2 * m.UP), m.FadeIn(ask[:2], shift=0.2 * m.UP)),
            m.AddTextLetterByLetter(ask_words, time_per_char=0.03),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the chatbot

    def chatbot(self):
        self.next_section(
            "1 · A chatbot",
            notes="A language model alone answers in one call, from what it learned in training. Its reply (tax "
            "rounding in checkout.py) is invented for the film: plausible, confident, unchecked and wrong. "
            "Li, AI Agents in Depth, ch. 1: without tools a model still returns 'a neatly formatted, confidently "
            "worded answer' from parametric memory. Fact date: October 2026.",
        )
        model = icon(LLM, 1.45).move_to([-5.2, 0.85, 0])
        m_label = text("language model", 26, INK, weight=m.BOLD).next_to(model, m.DOWN, buff=0.28)
        m_sub = text("no tools · no loop", 22, MUTED).next_to(m_label, m.DOWN, buff=0.1)
        tally = m.VGroup(
            code(f"{CHATBOT_CALLS[0]} model call", 20, CORAL), code(f"{CHATBOT_CALLS[1]} tool calls", 20, MUTED)
        ).arrange(m.DOWN, buff=0.14).next_to(m_sub, m.DOWN, buff=0.5)

        q = message("user", TASK, width=8.0, size=24)
        reply = "It's probably the tax rounding in checkout.py.\nTry:  total = round(subtotal * 1.08, 2)"
        a = message("assistant", reply, width=8.0, size=24)
        chat_title = code("chat", 20, MUTED)
        inner = m.VGroup(chat_title, q, a).arrange(m.DOWN, buff=0.2)
        chat_title.align_to(q, m.LEFT)
        chat = card(8.7, inner.height + 0.6).move_to([1.95, 0.55, 0])
        inner.move_to(chat)

        never = m.VGroup(*[pill(f"✗  {w}", ROSE, ROSE_FILL, 22) for w in ("no code", "no tests", "no error output")])
        never.arrange(m.RIGHT, buff=0.35).move_to([chat.get_x(), -1.95, 0])
        verdict = text("a plausible guess it cannot check", 26, ROSE, weight=m.SEMIBOLD).move_to([chat.get_x(), -2.8, 0])
        source = top_cite("Li, AI Agents in Depth · ch. 1")

        self.say(
            "[But first, picture a plain chatbot]: [a language model on its own, with no tools and no loop.]",
            m.AnimationGroup(self.new_header("1 · A CHATBOT", "One answer, from memory"), m.FadeIn(model, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(m_label, shift=0.1 * m.UP), m.FadeIn(m_sub, shift=0.1 * m.UP)),
        )
        self.say(
            "[It reads the request], [and answers at once, with a fix for the tax rounding in the checkout code.]",
            m.AnimationGroup(m.FadeIn(chat), m.FadeIn(chat_title), m.FadeIn(q, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(a[0]), m.FadeIn(a[1]), m.FadeIn(a[2][0]), m.AddTextLetterByLetter(a[2][1], time_per_char=0.022)),
        )
        self.say(
            "[It sounds sure of itself.] But it never saw tinyshop: [not the code], [not the tests], [not the error.]",
            m.AnimationGroup(m.Circumscribe(a, color=CORAL, buff=0.06), m.FadeIn(source)),
            *[m.FadeIn(p, shift=0.15 * m.UP) for p in never],
        )
        self.say(
            "[It can't run anything to check its guess.] [One reply, and its turn is over.]",
            m.FadeIn(verdict, shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.RIGHT) for t in tally], lag_ratio=0.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the agent

    def agent(self):
        self.next_section(
            "2 · An agent",
            notes=f"The same model with three tools and a loop (harness_reference.py). Otto's run: pytest -q -> "
            f"'{BEFORE}' (tests/test_cart.py::test_total: total([3, 4]) gave {GOT}, expected {EXPECTED}) -> read_file "
            f"cart.py -> edit_file (range(len(items) - 1) -> for item in items) -> pytest -q -> '{AFTER}' -> answer. "
            f"{MODEL_CALLS} model calls, {TOOL_CALLS} tool calls (computed from RUN). Terminal text simplified.",
        )
        head = np.array([-5.75, 1.42, 0])
        model = icon(LLM, 1.05).move_to(head)
        bot = put_head(otto(0.75), head)
        loop = m.Arc(radius=0.98, start_angle=m.PI / 2 - 0.5, angle=-(m.TAU - 1.0), arc_center=head)
        loop.set_stroke(MUTED, 4).add_tip(tip_length=0.2, tip_width=0.2)
        loop[-1].set_fill(MUTED, 1).set_stroke(MUTED, width=0)
        loop_label = code("loop", 20, MUTED).move_to(head + 0.98 * m.UP)

        tools = m.VGroup(*[mono_chip(t, SUN, SUN_FILL) for t in ("bash", "read_file", "edit_file")])
        tools.arrange(m.RIGHT, buff=0.18).move_to([0, head[1], 0]).align_to([-4.45, 0, 0], m.LEFT)
        tools_label = code("tools", 20, MUTED).next_to(tools, m.UP, buff=0.14).align_to(tools, m.LEFT)
        bash, read, edit = tools

        calls_label = code("model calls", 20, MUTED).move_to([-4.45, 0.72, 0], aligned_edge=m.LEFT)
        tools_tally = code("tool calls", 20, MUTED).move_to([-4.45, 0.27, 0], aligned_edge=m.LEFT)
        tally_x = calls_label.get_right()[0] + 0.3
        dots = m.VGroup(*[m.Dot(radius=0.085, color=CORAL) for _ in range(MODEL_CALLS)]).arrange(m.RIGHT, buff=0.16)
        dots.move_to([tally_x, calls_label.get_y(), 0], aligned_edge=m.LEFT)
        squares = m.VGroup(*[m.Square(0.17, stroke_width=0, fill_color=SUN, fill_opacity=1) for _ in range(TOOL_CALLS)]).arrange(m.RIGHT, buff=0.16)
        squares.move_to([tally_x, tools_tally.get_y(), 0], aligned_edge=m.LEFT)
        n_calls = title_text(str(MODEL_CALLS), 30, CORAL).next_to(dots, m.RIGHT, buff=0.25)
        n_tools = title_text(str(TOOL_CALLS), 30, SUN).next_to(squares, m.RIGHT, buff=0.25).set_x(n_calls.get_x())

        work = np.array([-3.05, -1.78, 0])
        term1 = terminal(["$ pytest -q", "! FAILED tests/test_cart.py::test_total", f"assert {GOT} == {EXPECTED}", BEFORE], width=7.1, size=22).move_to(work)
        term2 = terminal(["$ pytest -q", AFTER], width=7.1, size=22)
        term2.move_to(work).align_to(term1, m.UP)
        term2[4][1].set_color(MINT)
        block = code_block(BUGGY_SRC, size=22, width=7.1, title="cart.py").move_to(work)
        fixed = code_block(FIXED_SRC, size=22, width=7.1, title="cart.py").move_to(work)
        fixed[1].align_to(block[1], m.UL)
        old = m.VGroup(*[glyphs(block[1], s) for s in BUG_LINES])
        new = m.VGroup(*[glyphs(fixed[1], s) for s in FIX_LINES])
        gutter = block[0].get_left()[0] + 0.17
        bug_box = m.Line([gutter, old.get_top()[1] + 0.04, 0], [gutter, old.get_bottom()[1] - 0.04, 0]).set_stroke(ROSE, 6)
        fix_box = m.Line([gutter, new.get_top()[1] + 0.04, 0], [gutter, new.get_bottom()[1] - 0.04, 0]).set_stroke(MINT, 6)

        col = Column(center=(3.75, -0.55, 0), width=5.6, height=5.55)
        msgs = [col.message("user", TASK)]
        for kind, _, body in RUN:
            msgs.append(col.message(kind, body, mono=kind != "assistant"))
        user_msg, use1, res1, use2, res2, use3, res3, use4, res4, answer = msgs
        spin = lambda: m.Rotate(loop, angle=-m.TAU, about_point=head, run_time=1.2)  # noqa: E731

        self.say(
            "[Now take the same model], [give it tools], [and run it in a loop]: [that's Otto.]",
            m.AnimationGroup(self.new_header("2 · AN AGENT", "The same model, with tools and a loop"), m.FadeIn(model, shift=0.2 * m.UP)),
            m.LaggedStart(m.FadeIn(tools_label), *[m.FadeIn(t, shift=0.15 * m.RIGHT) for t in tools], lag_ratio=0.25),
            m.AnimationGroup(m.Create(loop), m.FadeIn(loop_label)),
            m.AnimationGroup(m.FadeOut(model, scale=0.6), m.FadeIn(bot, scale=0.6)),
        )
        self.say(
            "[Same request]: [the checkout test fails, fix it.]",
            m.FadeIn(col.frame),
            col.push(user_msg),
        )
        self.say(
            "[Instead of guessing], [Otto runs the tests.]",
            m.AnimationGroup(m.FadeIn(calls_label), m.FadeIn(dots[0], scale=0.4), m.Indicate(bash, color=SUN, scale_factor=1.12)),
            m.AnimationGroup(
                col.push(use1), m.FadeIn(tools_tally), m.FadeIn(squares[0], scale=0.4),
                m.FadeIn(term1[:4]), m.AddTextLetterByLetter(term1[4][0], time_per_char=0.04),
            ),
        )
        self.say(
            "[One failed, twelve passed], [and the failure is in the cart, not the checkout.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in term1[4][1:]], lag_ratio=0.3), col.push(res1)),
            m.Circumscribe(glyphs(term1[4][1], "test_cart.py"), color=ROSE, buff=0.06),
        )
        self.say(
            "[Next turn of the loop]: [Otto reads the cart file], [and spots the bug: the sum stops one item early.]",
            m.AnimationGroup(spin(), m.FadeIn(dots[1], scale=0.4)),
            m.AnimationGroup(
                m.Indicate(read, color=SUN, scale_factor=1.12), col.push(use2), m.FadeIn(squares[1], scale=0.4),
                m.FadeOut(term1), m.FadeIn(block, shift=0.15 * m.UP),
            ),
            m.AnimationGroup(col.push(res2), m.Create(bug_box), m.Circumscribe(old, color=ROSE, buff=0.08)),
        )
        self.say(
            "[It edits that line], [so every item counts.]",
            m.AnimationGroup(
                spin(), m.FadeIn(dots[2], scale=0.4), m.Indicate(edit, color=SUN, scale_factor=1.12),
                col.push(use3), m.FadeIn(squares[2], scale=0.4),
            ),
            m.AnimationGroup(
                m.FadeOut(old, shift=0.15 * m.UP), m.FadeIn(new, shift=0.15 * m.UP),
                m.FadeOut(bug_box), m.FadeIn(fix_box), col.push(res3),
            ),
        )
        block[1].remove(*old)  # gone for good: a later FadeOut of the block must not bring them back
        self.say(
            "[Then it runs the tests again]: [thirteen passed.]",
            m.AnimationGroup(
                spin(), m.FadeIn(dots[3], scale=0.4), m.Indicate(bash, color=SUN, scale_factor=1.12),
                col.push(use4), m.FadeIn(squares[3], scale=0.4),
                m.FadeOut(block), m.FadeOut(new), m.FadeOut(fix_box), m.FadeIn(term2[:4]), m.AddTextLetterByLetter(term2[4][0], time_per_char=0.04),
            ),
            m.AnimationGroup(m.FadeIn(term2[4][1], shift=0.1 * m.UP), col.push(res4)),
        )
        self.say(
            "[Only then does it answer]: [fixed, the total skipped the last item.]",
            m.AnimationGroup(spin(), m.FadeIn(dots[4], scale=0.4)),
            col.push(answer),
        )
        self.say(
            "[Five model calls, four tool calls.] [The loop ends when the model answers, instead of calling a tool.]",
            m.AnimationGroup(m.FadeIn(n_calls, scale=1.3), m.FadeIn(n_tools, scale=1.3)),
            m.Circumscribe(answer, color=CORAL, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ chatbot, script, agent

    def difference(self):
        self.next_section(
            "3 · The difference",
            notes="A chatbot chooses words but cannot act; a fixed script acts but its steps were chosen in advance "
            "by a programmer; an agent's model chooses each next step at run time and its harness carries it out "
            "through tools. Workflows sit in between: Episode 5.",
        )
        specs = (
            ("Chatbot", "a model on its own", icon(LLM, 0.95), (True, False), CORAL),
            ("Fixed script", "steps written in advance", icon(SCRIPT, 0.95), (False, True), MUTED),
            ("Agent", "model + tools + loop", otto(0.68), (True, True), CORAL),
        )
        cols, marks = m.VGroup(), []
        top_y = 1.75
        for x, (title, sub, pic, (decides, acts), color) in zip((-4.45, 0.0, 4.45), specs):
            box = card(4.15, 3.95, EDGE, CARD).move_to([x, top_y - 3.95 / 2, 0])
            pic.move_to([x, top_y - 0.68, 0])
            name = title_text(title, 32).move_to([x, top_y - 1.5, 0])
            what = text(sub, 22, MUTED)
            what.shift((top_y - 1.98 - what.get_top()[1]) * m.UP).set_x(x)
            rule = m.Line([x - 1.78, top_y - 2.48, 0], [x + 1.78, top_y - 2.48, 0]).set_stroke(EDGE, 2)
            words, row_marks = m.VGroup(), []
            for k, (label, ok) in enumerate((("chooses the next step", decides), ("acts on the repo", acts))):
                y = top_y - 2.95 - 0.58 * k
                words.add(text(label, 22, INK).move_to([x - 1.75, y, 0], aligned_edge=m.LEFT))
                row_marks.append(icon("check" if ok else "ep01-cross", 0.42).move_to([x + 1.66, y, 0]))
            cols.add(m.VGroup(box, pic, name, what, rule, words))
            marks.append(row_marks)
        on_baseline(m.VGroup(*[c[2] for c in cols]))
        (bot_chat, bot_act), (scr_chat, scr_act), (ag_chat, ag_act) = marks
        note = cite("workflows sit in between · Episode 5").move_to([0, -2.85, 0])

        self.say(
            "[So what makes Otto an agent?] [Compare a chatbot], [a fixed script], [and an agent.]",
            self.new_header("3 · THE DIFFERENCE", "Who chooses, and who acts"),
            m.FadeIn(cols[0], shift=0.2 * m.UP),
            m.FadeIn(cols[1], shift=0.2 * m.UP),
            m.FadeIn(cols[2], shift=0.2 * m.UP),
        )
        self.say(
            "[A chatbot chooses what to say], [but it can't act.] [A script acts], [but a programmer chose every step in advance.]",
            m.FadeIn(bot_chat, scale=1.4),
            m.FadeIn(bot_act, scale=1.4),
            m.FadeIn(scr_act, scale=1.4),
            m.FadeIn(scr_chat, scale=1.4),
        )
        self.say(
            "[An agent does both]: [its model chooses each next step as it goes], [and its tools carry it out.] [Workflows sit in between; that's episode five.]",
            m.Circumscribe(cols[2], color=CORAL, buff=0.08),
            m.FadeIn(ag_chat, scale=1.4),
            m.FadeIn(ag_act, scale=1.4),
            m.FadeIn(note, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the formula

    def formula(self):
        self.next_section(
            "4 · The formula",
            notes="Li, AI Agents in Depth (introduction): 'Agent = LLM + Context + Tools ... More intuitively, it is "
            "Brain + Eyes + Hands and Feet'; LLM = Policy, Context = Observation Space, Tools = Action Space. "
            "Ch. 1: the components 'do not map one-to-one': context is only part of the observation space, and "
            "tools define which observations and actions the agent can access (read_file is a tool that looks). "
            "Fact date: October 2026.",
        )
        parts = (("LLM", CORAL, CORAL_FILL, "ep01-brain", "brain", "policy"),
                 ("Context", MINT, MINT_FILL, "ep01-eye", "eyes", "observation"),
                 ("Tools", SUN, SUN_FILL, "ep01-hand", "hands", "action"))
        agent_word = title_text("Agent", 60)
        eq = title_text("=", 60, MUTED)
        boxes = []
        for label, color, fill, *_ in parts:
            b = card(2.55, 1.25, color, fill)
            boxes.append(m.VGroup(b, title_text(label, 40).move_to(b)))
        plus = [title_text("+", 54, MUTED) for _ in range(2)]
        row = m.VGroup(agent_word, eq, boxes[0], plus[0], boxes[1], plus[1], boxes[2]).arrange(m.RIGHT, buff=0.32).move_to([0.2, 1.25, 0])
        for p in plus:
            p.set_y(boxes[0].get_y())
        eq.set_y(boxes[0].get_y())

        pics, words, rl = [], [], []
        for b, (_, color, _, pic, word, rl_word) in zip(boxes, parts):
            p = icon(pic, 1.05).move_to([b.get_x(), -0.3, 0])
            w = text(word, 28, color, weight=m.BOLD).move_to([b.get_x(), -1.15, 0])
            r = code(rl_word, 26, color, weight=m.BOLD).move_to([b.get_x(), -2.25, 0])
            pics.append(p), words.append(w), rl.append(r)
        on_baseline(m.VGroup(*words))
        on_baseline(m.VGroup(*rl))
        side1 = text("intuitively", 22, MUTED).move_to([agent_word.get_x(), -0.45, 0])
        side2 = text("in RL terms", 22, MUTED).move_to([agent_word.get_x(), -2.25, 0])
        source = top_cite("Li, AI Agents in Depth · intro, ch. 1")
        start = rl[2].get_bottom() + 0.12 * m.DOWN
        end = rl[1].get_bottom() + 0.12 * m.DOWN
        also = m.CurvedArrow(start, end, angle=-0.9, color=SUN, stroke_width=3.5, tip_length=0.18)
        also_label = m.VGroup(code("read_file", 20, SUN), text("is a tool, yet it looks", 20, MUTED)).arrange(m.RIGHT, buff=0.15)
        also_label.move_to([(start[0] + end[0]) / 2, -3.35, 0])
        not_one = pill("not one-to-one", ROSE, ROSE_FILL, 20).next_to(side2, m.DOWN, buff=0.3)

        self.say(
            "[The book AI Agents in Depth sums it up in one line]: [an agent is a language model], [plus context], [plus tools.]",
            m.AnimationGroup(self.new_header("4 · THE FORMULA", "LLM, context, tools"), m.FadeIn(agent_word, shift=0.2 * m.UP), m.FadeIn(eq), m.FadeIn(source)),
            m.FadeIn(boxes[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(plus[0]), m.FadeIn(boxes[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(plus[1]), m.FadeIn(boxes[2], shift=0.2 * m.UP)),
        )
        self.say(
            "More intuitively: [a brain that decides], [eyes, which are its context], [and hands, which are its tools.]",
            *[m.AnimationGroup(m.FadeIn(p, shift=0.2 * m.UP, scale=0.8), m.FadeIn(w, shift=0.1 * m.UP), *([m.FadeIn(side1)] if i == 0 else [])) for i, (p, w) in enumerate(zip(pics, words))],
        )
        self.say(
            "[In reinforcement learning terms]: [the model is a policy], [the context is its observation], [and the tools are its actions.]",
            m.FadeIn(side2, shift=0.1 * m.UP),
            *[m.FadeIn(r, shift=0.1 * m.UP) for r in rl],
        )
        self.say(
            "[The book adds that the match isn't one-to-one]: [reading a file is a tool call, yet it works like eyes.]",
            m.FadeIn(not_one, scale=1.2),
            m.AnimationGroup(m.Create(also), m.FadeIn(also_label, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ observation and action spaces

    def spaces(self):
        self.next_section(
            "5 · The spaces",
            notes="Li, AI Agents in Depth, ch. 1, 'Observation and Action Spaces': 'Information outside the observation "
            "space effectively does not exist for the model. An operation outside the action space remains something "
            "the model can only recommend in words.' 'Many problems that appear to require a smarter model are really "
            "interface problems.' The outside examples (the checkout page in a browser, deploying) are illustrative. "
            "Episode 28 widens both spaces (screens, voice, robots). Fact date: October 2026.",
        )
        def region(cx, color, fill, title, sub):
            box = card(6.15, 3.0, color, CARD, stroke=2.5).move_to([cx, 0.85, 0])
            head = m.VGroup(code(title, 22, color, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.RIGHT, buff=0.25)
            head.move_to(box.get_corner(m.UL) + np.array([0.35, -0.42, 0]), aligned_edge=m.LEFT)
            return m.VGroup(box, head)

        obs = region(-3.35, MINT, MINT_FILL, "observation space", "what it can see")
        act = region(3.35, SUN, SUN_FILL, "action space", "what it can do")
        seen = m.VGroup(
            mono_chip("your request", AZURE, AZURE_FILL), mono_chip("pytest output", MINT, MINT_FILL),
            mono_chip("cart.py", MINT, MINT_FILL), mono_chip("test_cart.py", MINT, MINT_FILL),
        ).arrange_in_grid(rows=2, buff=(0.3, 0.3)).move_to(obs[0]).shift(0.25 * m.DOWN)
        done = m.VGroup(*[mono_chip(t, SUN, SUN_FILL) for t in ("bash", "read_file", "edit_file")])
        done.arrange(m.RIGHT, buff=0.3).move_to(act[0]).shift(0.25 * m.DOWN)

        def outside(label, cx):
            c = word_chip(label, EDGE, CARD, size=22)
            c[1].set_color(MUTED)
            tag = code("outside", 20, MUTED)
            return m.VGroup(tag, c).arrange(m.RIGHT, buff=0.25).move_to([cx, -1.35, 0])

        unseen = outside("the checkout page in a browser", -3.35)
        gone = text("does not exist for the model", 22, ROSE, weight=m.SEMIBOLD).move_to([-3.35, -2.05, 0])
        undone = outside("deploy to production", 3.35)
        advice_words = text("You should deploy this fix.", 24, INK)
        advice = bubble(advice_words, CORAL, CORAL_FILL, tail="left").move_to([3.35, -1.75, 0])
        quote = text("Many problems that seem to need a smarter model are really interface problems.", 26, INK,
                     t2c={"interface problems": SUN}).move_to([0, -3.1, 0])
        source = top_cite("Li, AI Agents in Depth · ch. 1")

        self.say(
            "[The observation space is everything Otto can see]: [the request, the test output, the files it reads.]",
            m.AnimationGroup(self.new_header("5 · THE SPACES", "What it sees, what it can do"), m.FadeIn(obs), m.FadeIn(source)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in seen], lag_ratio=0.3),
        )
        self.say(
            "[Anything outside it, like how the checkout page looks in a browser], [effectively does not exist for the model.]",
            m.FadeIn(unseen, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(gone, shift=0.1 * m.UP), unseen.animate.set_opacity(0.35)),
        )
        self.say(
            "[The action space is everything it can do]: [run commands, read files, and edit them.]",
            m.FadeIn(act),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in done], lag_ratio=0.3),
        )
        self.say(
            "[An action outside it, like deploying the fix], [can only be recommended in words], [just like our chatbot's answer.]",
            m.FadeIn(undone, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeOut(undone, shift=0.2 * m.DOWN), m.FadeIn(advice, shift=0.2 * m.DOWN)),
            m.Circumscribe(advice, color=CORAL, buff=0.08),
        )
        self.say(
            "[Many problems that seem to need a smarter model] [are really interface problems.]",
            m.FadeIn(quote, shift=0.15 * m.UP),
            m.Circumscribe(glyphs(quote, "interface problems"), color=SUN, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ model + harness

    def harness(self):
        self.next_section(
            "6 · The harness",
            notes="Li, AI Agents in Depth, ch. 1: Agent = Model + Harness; the harness, 'the runtime and governance "
            "layer inside the Agent boundary', builds context, exposes tool interfaces, maintains loops and state, "
            "and applies permissions, verification and correction. The code is an excerpt of harness_reference.py "
            "(agent_loop), simplified. Permissions: Ep 17; hooks (verify after every edit): Ep 18; error recovery: Ep 20. "
            "Fact date: October 2026.",
        )
        frame = card(5.5, 5.15, INK, CARD_HI, radius=0.35, stroke=3).move_to([-4.02, -0.5, 0]).set_z_index(-1)
        frame_label = code("harness", 24, INK, weight=m.BOLD).move_to(frame.get_corner(m.UL) + np.array([0.38, -0.38, 0]), aligned_edge=m.LEFT)
        core = card(2.3, 1.75, CORAL, CORAL_FILL)
        core_body = m.VGroup(icon(LLM, 0.85), text("model", 24, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.12).move_to(core)
        model = m.VGroup(core, core_body).move_to(frame).shift(0.05 * m.DOWN).set_z_index(1)

        job = lambda w, color, fill: chip(w, color, fill, width=1.62, height=0.52, size=20).set_z_index(1)  # noqa: E731
        top_jobs = m.VGroup(job("context", MINT, MINT_FILL), job("tools", SUN, SUN_FILL), job("loop", CORAL, CORAL_FILL))
        top_jobs.arrange(m.RIGHT, buff=0.12).move_to([frame.get_x(), 0.95, 0])
        low_jobs = m.VGroup(*[job(w, MUTED, CARD) for w in ("permissions", "verification", "correction")])
        low_jobs.arrange(m.RIGHT, buff=0.12).move_to([frame.get_x(), -1.9, 0])
        eps = m.VGroup(*[code(f"Ep {n}", 20, FAINT).next_to(j, m.DOWN, buff=0.14) for j, n in zip(low_jobs, (17, 18, 20))])

        src = (
            "def agent_loop(messages):\n"
            "    while True:\n"
            "        response = client.messages.create(\n"
            "            model=MODEL, system=SYSTEM,\n"
            "            messages=messages, tools=TOOLS)\n"
            "        messages.append(...)  # the reply\n"
            '        if response.stop_reason != "tool_use":\n'
            "            return response  # done\n"
            "        messages.append(...)  # tool results"
        )
        loop_code = code_block(src, size=20, title="harness_reference.py")
        loop_code.move_to([0, -0.5, 0]).align_to([6.75, 0, 0], m.RIGHT)
        assert loop_code.get_left()[0] > frame.get_right()[0] + 0.15, "the code must sit beside the harness frame"
        body = loop_code[1]
        spots = [(glyphs(body, k), c) for k, c in (("messages=messages", MINT), ("tools=TOOLS", SUN), ("while True:", CORAL))]

        def light(k):
            part, color = spots[k]
            return m.AnimationGroup(
                m.FadeIn(top_jobs[k], shift=0.15 * m.DOWN), part.animate.set_color(color),
                m.Circumscribe(part, color=color, buff=0.05),
            )
        source = top_cite("Li, AI Agents in Depth · ch. 1")

        self.say(
            "[Another way to see it]: [an agent is a model], [plus a harness.]",
            m.AnimationGroup(self.new_header("6 · THE HARNESS", "Agent = model + harness"), m.FadeIn(source)),
            m.FadeIn(model, scale=0.9),
            m.AnimationGroup(m.FadeIn(frame, scale=0.95), m.FadeIn(frame_label), m.FadeIn(loop_code, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[The harness is the ordinary code around the model.] [It builds the context], [exposes the tools], [and keeps the loop turning.]",
            m.Circumscribe(loop_code, color=INK, buff=0.06),
            light(0),
            light(1),
            light(2),
        )
        self.say(
            "[It also applies permissions], [checks the work], [and corrects course]: [each gets its own episode later on.]",
            m.FadeIn(low_jobs[0], shift=0.15 * m.UP),
            m.FadeIn(low_jobs[1], shift=0.15 * m.UP),
            m.FadeIn(low_jobs[2], shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(e) for e in eps], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the reins

    def reins(self):
        self.next_section(
            "7 · The reins",
            notes="Li, AI Agents in Depth, ch. 1: 'The word Harness originally referred to the reins and tack fitted to a "
            "horse' — not to limit its ability to run but to direct that power; the model is 'the powerful yet "
            "unpredictable horse'; 'the stronger the model, the more the surrounding Harness matters.' LangChain's "
            f"coding agent on Terminal-Bench 2.0: {TB_BEFORE}% -> {TB_AFTER}% (outside the top 30 -> top 5) by changing "
            "only the harness (self-checks, loop detection) — as reported in Li, ch. 1 (no primary URL given there). "
            "Fact date: October 2026.",
        )
        horse = icon("ep01-horse", 3.3).move_to([-0.7, 0.15, 0])
        bit = horse[7].get_center()
        harness_tag = pill("harness", INK, CARD_HI, 24, mono=True).move_to([-5.2, 1.45, 0])
        handle = harness_tag.get_right()
        reins = m.CubicBezier(bit, bit + np.array([-1.3, 1.9, 0]), handle + np.array([1.7, 0.6, 0]), handle).set_stroke(INK, 3.5)
        horse_label = m.VGroup(text("the model", 26, CORAL, weight=m.BOLD), text("powerful, but unpredictable", 22, MUTED)).arrange(m.DOWN, buff=0.1)
        horse_label.next_to(horse, m.DOWN, buff=0.3)
        task = word_chip("the task: tests pass", MINT, MINT_FILL, size=24, height=0.7).move_to([4.6, bit[1], 0])
        aim = arrow(horse.get_right() + np.array([0.15, bit[1] - horse.get_y(), 0]), task.get_left(), MINT, buff=0.1)
        pictured = m.VGroup(horse, harness_tag, reins, horse_label, task, aim)

        length = 7.4
        rows = m.VGroup()
        for label, value, color in (("old harness", TB_BEFORE, MUTED), ("new harness", TB_AFTER, MINT)):
            name = text(label, 26, INK, weight=m.SEMIBOLD)
            bar = m.RoundedRectangle(corner_radius=0.12, width=length * value / 100, height=0.62, stroke_width=0, fill_color=color, fill_opacity=1)
            val = title_text(f"{value}%", 34, color)
            rows.add(m.VGroup(name, bar, val))
        for i, r in enumerate(rows):
            y = 0.85 - 1.15 * i
            r[0].move_to([-3.4, y, 0], aligned_edge=m.RIGHT)
            r[1].move_to([-3.0, y, 0], aligned_edge=m.LEFT)
            r[2].next_to(r[1], m.RIGHT, buff=0.25)
        bench = m.VGroup(code("Terminal-Bench 2.0", 22, MUTED), pill("same model", CORAL, CORAL_FILL, 20)).arrange(m.RIGHT, buff=0.35)
        bench.move_to([0, 1.95, 0])
        gain = text(f"+{TB_GAIN} points", 28, MINT, weight=m.BOLD).next_to(rows[1][2], m.RIGHT, buff=0.35)
        tb_cite = cite("LangChain, as reported in Li, AI Agents in Depth · ch. 1").move_to([0, -1.3, 0])
        quote = text("“The stronger the model, the more the surrounding harness matters.”", 28, INK).move_to([0, -2.5, 0])
        q_cite = cite("Li, AI Agents in Depth · ch. 1").next_to(quote, m.DOWN, buff=0.2)

        self.say(
            "[Why the name harness?] [It first meant the reins and tack fitted to a horse.]",
            self.new_header("7 · THE REINS", "Power, pointed at the task"),
            m.AnimationGroup(m.FadeIn(horse, shift=0.2 * m.LEFT), m.Create(reins), m.FadeIn(harness_tag, shift=0.15 * m.RIGHT)),
        )
        self.say(
            "[The model is the horse]: [powerful, but unpredictable.] [The reins don't stop it running]; [they point that power at the task.]",
            m.FadeIn(horse_label[0], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(horse_label[1], shift=0.1 * m.UP), m.Wiggle(horse, scale_value=1.04)),
            m.ShowPassingFlash(reins.copy().set_stroke(SUN, 6), time_width=0.6, run_time=1.2),
            m.AnimationGroup(m.GrowArrow(aim), m.FadeIn(task, shift=0.15 * m.LEFT)),
        )
        self.say(
            "[As the book reports], [one coding agent on Terminal-Bench went from fifty-two point eight percent] [to sixty-six point five], [by changing only the harness.]",
            m.AnimationGroup(m.FadeOut(pictured), m.FadeIn(bench, shift=0.1 * m.DOWN), m.FadeIn(tb_cite)),
            m.AnimationGroup(m.FadeIn(rows[0][0]), m.GrowFromEdge(rows[0][1], m.LEFT), m.FadeIn(rows[0][2])),
            m.AnimationGroup(m.FadeIn(rows[1][0]), m.GrowFromEdge(rows[1][1], m.LEFT), m.FadeIn(rows[1][2])),
            m.AnimationGroup(m.FadeIn(gain, shift=0.15 * m.LEFT), m.Circumscribe(bench[1], color=CORAL, buff=0.06)),
        )
        self.say(
            "[In the book's words]: [the stronger the model, the more the surrounding harness matters.]",
            m.FadeIn(q_cite),
            m.Write(quote),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Brain, eyes, hands: the model, its context, its tools — run in a loop until the job is done.")
        halo = glow(CORAL, radius=4.0)
        spec = (("Brain", "ep01-brain", CORAL, "LLM"), ("Eyes", "ep01-eye", MINT, "context"), ("Hands", "ep01-hand", SUN, "tools"))
        cols = m.VGroup()
        for word, pic, color, sub in spec:
            cols.add(m.VGroup(icon(pic, 1.15), title_text(word, 60, color), code(sub, 22, MUTED)).arrange(m.DOWN, buff=0.25))
        cols.arrange(m.RIGHT, buff=1.4).move_to(0.55 * m.UP)
        on_baseline(m.VGroup(*[c[1] for c in cols]))
        for c in cols:
            c[2].shift((cols[0][1].get_bottom()[1] - 0.3 - c[2].get_top()[1]) * m.UP)
        tagline = text("A chatbot answers once. An agent acts, and checks, until the job is done.", 30, MUTED).next_to(cols, m.DOWN, buff=0.65)

        self.say(
            "[To sum up]: [a brain], [eyes], [and hands.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            *[m.FadeIn(c, shift=0.3 * m.UP) for c in cols],
        )
        self.say("[A chatbot answers once. An agent acts, and checks, until the job is done.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, cols, tagline)], run_time=0.9)

    # ------------------------------------------------------------------ the course

    def course(self):
        self.next_section(
            "The course",
            notes="The six modules and their episode ranges (from kit.EPISODES): 1–5, 6–11, 12–14, 15–20, 21–26, 27–33.",
        )
        rows = m.VGroup()
        names = {1: "Foundations", 2: "Context Engineering", 3: "Memory & Knowledge", 4: "Tools & Safety", 5: "Scaling Up", 6: "Beyond the Loop"}
        for k, (_, color, fill) in MODULES.items():
            badge = chip(str(k), color, fill, width=0.62, height=0.56, size=24)
            title = title_text(names[k], 26, color)
            desc = text(MODULE_LINES[k], 24, INK)
            lo, hi = RANGES[k]
            span = code(f"Ep {lo}–{hi}", 20, MUTED)
            y = 1.95 - 0.95 * (k - 1)
            badge.move_to([-6.15, y, 0])
            title.move_to([-5.55, y, 0], aligned_edge=m.LEFT)
            desc.move_to([-1.05, y, 0], aligned_edge=m.LEFT)
            span.move_to([6.65, y, 0], aligned_edge=m.RIGHT)
            rows.add(m.VGroup(badge, title, desc, span))
        here = otto(0.32)
        put_head(here, [rows[0][2].get_right()[0] + 0.55, rows[0].get_y(), 0])

        self.say(
            "[Here's the road ahead, in six modules.] [Foundations: the loop, tools, and when to use an agent.]",
            self.new_header("THE COURSE", "Six modules, one agent"),
            m.AnimationGroup(m.FadeIn(rows[0], shift=0.15 * m.RIGHT), m.FadeIn(here, scale=0.6)),
        )
        self.say(
            "[Context engineering: what goes into the window.] [Memory and knowledge: what Otto remembers.] [Tools and safety: what it may do, and what stops it.]",
            *[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows[1:4]],
        )
        self.say(
            "[Scaling up: subagents and teams.] [And beyond the loop: evaluation, training, and production.]",
            *[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows[4:6]],
        )
        self.wait(0.8)
        self.clear()
