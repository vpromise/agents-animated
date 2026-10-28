"""Agents, Animated · Episode 4 · Think, Act, Observe

Why interleaving reasoning and action works, and when to stop. Running example, all film long:
Otto's first guess about tinyshop's failing test ("the bug is probably in checkout.py"), disproved
by one pytest run, then the real bug found, fixed and verified.

- ReAct (Yao et al., 2022, arXiv 2210.03629, ICLR 2023): interleave reasoning traces ("Thought") with
  actions ("Act") and observations ("Obs"); reasoning helps "induce, track, and update action plans",
  actions "interface with external sources"; it overcomes "hallucination and error propagation"
  seen in chain-of-thought reasoning. Trace labels follow the paper's "Thought 1 / Act 1 / Obs 1".
- The loop (Episode 2) and B's wording: "The essence of the ReAct loop is appending to the
  trajectory, over and over" (Li, AI Agents in Depth, ch. 1, "The ReAct loop").
- Stop conditions (B, ch. 1): a final output tool, a response without any tool calls, round limits;
  "Agents can become stuck repeating the same tool calls forever" (B, ch. 2); "errors compound" (B,
  ch. 1). The 95%-per-step example (0.95^20 = 36%) is illustrative arithmetic, computed below.
- B's second example (ch. 1): quarterly revenue in four currencies, three parallel convert_currency calls,
  then code_interpreter, then the answer ($9,602,895.73): "3 iterations and 4 tool calls".
- Context ablation (B, ch. 1, Experiment 1-1): without tool definitions the model "still returns a
  neatly formatted, confidently worded answer" from parametric memory; without tool results it runs
  "blind", retrying until the iteration budget is gone; "'produced an answer' is not 'completed the
  task'".
- The running bug (harness_reference.py): cart.total() loops range(len(items) - 1); test_total
  asserts total([3, 4]) == 7 and gets 3; pytest -q says "1 failed, 12 passed". The two versions of
  cart.py below are real code, executed here; every number on screen comes from them or from TRACE.
  Illustrative: the pytest timings (0.41 s, 0.38 s), the content of the tool-less answer, the fixed
  plan's steps.

Colours: thoughts / the model CORAL, actions (tool calls) SUN, observations (tool results) MINT,
the user AZURE; wrong ROSE, verified MINT; headers CORAL (Module 1).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

FACTS = "Facts checked October 2026."
YAO = "Yao et al., 2022"
BOOK = "Li, AI Agents in Depth · ch. 1"

# ---------------------------------------------------------------------------- the numbers

BUGGY_SRC = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s"
FIXED_SRC = BUGGY_SRC.replace("len(items) - 1", "len(items)")


def _total(src):
    scope = {}
    exec(src, scope)
    return scope["total"]


CART = [3, 4]
GOT, WANT = _total(BUGGY_SRC)(CART), _total(FIXED_SRC)(CART)
assert (GOT, WANT) == (3, 7) and GOT == CART[0] and WANT == sum(CART)  # it drops the last item
FAILED, PASSED = 1, 12
TESTS = FAILED + PASSED
assert TESTS == 13

# Otto's ReAct trace (kind, body, mono). Rounds = model calls; each Act is one tool call.
TRACE = [
    ("Thought", "The checkout test fails, so the\nbug is probably in checkout.py.", False),
    ("Act", 'bash("pytest -q")', True),
    ("Obs", f"test_total fails: total([3, 4])\nreturned {GOT}, not {WANT}.", False),
    ("Thought", f"{GOT} is just the first item, so\ncart.total() skips the last one.", False),
    ("Act", 'read_file("cart.py")', True),
    ("Obs", "range(len(items) - 1)", True),
    ("Thought", "Off by one. Count every item.", False),
    ("Act", 'edit_file("cart.py", ...)', True),
    ("Obs", "edited cart.py", True),
    ("Thought", "An edit is a guess too. Test it.", False),
    ("Act", 'bash("pytest -q")', True),
    ("Obs", f"{TESTS} passed", True),
    ("Answer", f"Fixed: cart.total() skipped the\nlast item. All {TESTS} tests pass.", False),
]
ACTS = [body for kind, body, _ in TRACE if kind == "Act"]
TOOL_CALLS = len(ACTS)
ROUNDS = TOOL_CALLS + 1  # every tool call took a round; the answer took one more, with no call
FILES_READ = sum(a.startswith("read_file") for a in ACTS)
TEST_RUNS = sum("pytest" in a for a in ACTS)
assert (TOOL_CALLS, ROUNDS, FILES_READ, TEST_RUNS) == (4, 5, 1, 2)
assert TRACE[-1][0] == "Answer" and TRACE[-2] == ("Obs", "13 passed", True)

# The book's second example (Li, AI Agents in Depth, ch. 1, "The ReAct loop"): quarterly revenue in four
# currencies. Round 1: three parallel convert_currency calls; round 2: one code_interpreter call; round 3:
# the answer, total $9,602,895.73 (average $2,400,723.93). "3 iterations and 4 tool calls".
FX_ROUNDS = [["EUR → USD", "GBP → USD", "JPY → USD"], ["code_interpreter"], []]  # tool calls per round
FX_CALLS = sum(len(r) for r in FX_ROUNDS)
FX_TOTAL, FX_AVERAGE = 9_602_895.73, 2_400_723.93
assert (len(FX_ROUNDS), FX_CALLS) == (3, 4) and abs(FX_TOTAL / 4 - FX_AVERAGE) < 0.01

STEP_OK, STEPS = 0.95, 20  # illustrative: each step right 95% of the time


def all_right(n):
    return STEP_OK**n


assert round(100 * all_right(1)) == 95 and round(100 * all_right(10)) == 60 and round(100 * all_right(STEPS)) == 36
assert 0.3 < all_right(STEPS) < 0.4  # "about a third"

# ---------------------------------------------------------------------------- helpers

KIND_COLOR = {"Thought": CORAL, "Act": SUN, "Obs": MINT, "Answer": CORAL}


def cite(s):
    return text(s, 20, MUTED)


def trace_row(kind, n, body, width, mono=False, size=21, label_w=1.62):
    """One line of a ReAct trace, as in the paper: a coloured label ("Thought 1", "Act 1", "Obs 1")
    and its content, in a card with a stripe of the role's colour."""
    color = KIND_COLOR[kind]
    label = code(f"{kind} {n}" if n else kind, 20, color, weight=m.BOLD)
    words = code(body, size - 2, INK) if mono else text(body, size, INK)
    assert words.width <= width - label_w - 0.57, (body, words.width)
    box = card(width, max(label.height, words.height) + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift((0.32 + label_w) * m.RIGHT)
    label.align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    label.shift((words[0].get_bottom()[1] - label[0].get_bottom()[1]) * m.UP)
    return m.VGroup(box, stripe, m.VGroup(label, words))


def wrong_mark(row):
    """A ROSE cross under a trace row's label: this thought was disproved."""
    label = row[2][0]
    return text("✗", 30, ROSE, weight=m.BOLD).next_to(label, m.DOWN, buff=0.1).align_to(label, m.LEFT)


def disprove(row, mark):
    """The animation that marks a trace row as disproved: dimmed, a ROSE stripe and a cross."""
    return m.AnimationGroup(row[2].animate.set_opacity(0.45), row[1].animate.set_fill(ROSE), m.FadeIn(mark, scale=1.4))


def stat(value, label, color=INK):
    """A big number over a small label."""
    number = title_text(str(value), 44, color)
    words = text(label, 21, MUTED)
    return m.VGroup(number, words).arrange(m.DOWN, buff=0.1)


def verdict(label, ok, size=21):
    mark = "✓" if ok else "✗"
    return pill(f"{label} {mark}", MINT if ok else ROSE, MINT_FILL if ok else ROSE_FILL, size)


def ring_flash(center, radius, color=INK):
    """A light running once around a loop_ring, clockwise from the top (the ring's order)."""
    path = m.Arc(radius=radius, start_angle=m.PI / 2, angle=-2 * m.PI, arc_center=np.array(center, dtype=float))
    return m.ShowPassingFlash(path.set_stroke(color, 7), time_width=0.35, run_time=1.6)


def blink(bot):
    return bot[1].animate(rate_func=m.there_and_back, run_time=0.5).stretch(0.12, 1)


# ---------------------------------------------------------------------------- the episode


class Ep04ThinkActObserve(Episode):
    number = 4

    def lesson(self):
        self.task()
        self.react()
        self.trace()
        self.stopping()
        self.no_tools()
        self.fixed_plan()
        self.recap()

    # ------------------------------------------------------------------ the task

    def task(self):
        self.next_section(
            "The task",
            notes="Running example (COURSE.md): 'The checkout test fails. Fix it.' Otto's first thought blames "
            "checkout.py; it has read 0 files and run 0 tests. " + FACTS,
        )
        bot = otto(1.6).move_to([-4.7, -0.1, 0])
        ask = message("user", "The checkout test fails. Fix it.", width=6.3, size=25).move_to([1.75, 1.75, 0])
        guess = bubble(text("The bug is probably in checkout.py.", 28), CORAL, CORAL_FILL, tail="left")
        guess.move_to([1.75, 0.05, 0])
        tag = pill("a guess", SUN, SUN_FILL, 22).next_to(guess[1], m.DOWN, buff=0.25).align_to(guess[1], m.RIGHT)
        files = stat(0, "files read").move_to([0.4, -2.2, 0])
        tests = stat(0, "tests run").move_to([3.1, -2.2, 0])
        teaser = m.VGroup(
            chip("think", CORAL, CORAL_FILL, width=1.3, height=0.6, size=24), text("→", 26, MUTED),
            chip("act", SUN, SUN_FILL, width=1.0, height=0.6, size=24), text("→", 26, MUTED),
            chip("observe", MINT, MINT_FILL, width=1.55, height=0.6, size=24),
        ).arrange(m.RIGHT, buff=0.2).move_to([0, -3.25, 0])

        self.say(
            "[Otto has a task]: [the checkout test fails. Fix it.]",
            m.AnimationGroup(self.new_header("THE TASK", "A first guess"), m.FadeIn(bot, shift=0.3 * m.UP)),
            m.FadeIn(ask, shift=0.25 * m.UP),
        )
        self.say(
            "[Before touching anything, it forms a hunch]: [the bug is probably in the checkout code.]",
            blink(bot),
            m.FadeIn(guess, shift=0.2 * m.UP, scale=0.95),
        )
        self.say(
            "[A reasonable guess.] [But so far, Otto has read no files], [and run no tests.]",
            m.FadeIn(tag, shift=0.15 * m.UP),
            m.FadeIn(files, shift=0.2 * m.UP),
            m.FadeIn(tests, shift=0.2 * m.UP),
        )
        self.say(
            "[So how does an agent find out?] [It thinks], [it acts], [and it looks at what happened.]",
            m.Circumscribe(guess[1], color=SUN, buff=0.08),
            m.FadeIn(teaser[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(teaser[1]), m.FadeIn(teaser[2], shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(teaser[3]), m.FadeIn(teaser[4], shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ ReAct

    def react(self):
        self.next_section(
            "ReAct",
            notes="ReAct: Yao et al., 2022 (arXiv 2210.03629, ICLR 2023): reasoning traces and actions interleaved; "
            "reasoning helps 'induce, track, and update action plans'; actions 'interface with external sources'; it "
            "overcomes 'hallucination and error propagation' of chain-of-thought. Thought = assistant text, action = "
            "tool_use block, observation = tool_result (Episode 2). Quote: Li, AI Agents in Depth, ch. 1. " + FACTS,
        )
        word = title_text("ReAct", 110, INK, t2c={"Re": CORAL, "Act": SUN}).move_to([0, 1.45, 0])
        sub = text("reasoning + acting", 30, MUTED, t2c={"reasoning": CORAL, "acting": SUN}).next_to(word, m.DOWN, buff=0.2)
        source = cite(YAO).next_to(sub, m.DOWN, buff=0.18)

        def one_sided(title, color, fill, chips, flaw, x):
            box = card(5.6, 2.35, color, CARD, stroke=2).move_to([x, -1.85, 0])
            head = text(title, 26, INK, weight=m.BOLD).move_to(box.get_top() + 0.42 * m.DOWN)
            row = m.VGroup()
            for i, w in enumerate(chips):
                if i:
                    row.add(text("→", 24, MUTED))
                row.add(chip(w, color, fill, width=1.25, height=0.56, size=21))
            row.arrange(m.RIGHT, buff=0.14).move_to(box).shift(0.02 * m.DOWN)
            bad = text(flaw, 22, ROSE, weight=m.SEMIBOLD).move_to(box.get_bottom() + 0.38 * m.UP)
            return m.VGroup(box, head, row, bad)

        think_only = one_sided("Reasoning only", CORAL, CORAL_FILL, ("think", "think", "think"), "can't check its facts", -3.2)
        act_only = one_sided("Acting only", SUN, SUN_FILL, ("act", "act", "act"), "has no plan", 3.2)

        center = np.array([-3.75, -0.2, 0])
        radius = 1.55
        ring = loop_ring(["think", "act", "observe"], radius=radius, colors=[CORAL, SUN, MINT], size=24, center=center)
        nodes, arcs = ring
        small = title_text("ReAct", 44, INK, t2c={"Re": CORAL, "Act": SUN}).move_to(center + 0.12 * m.UP)
        small_src = cite(YAO).move_to(center + 2.2 * m.DOWN)

        col = ContextColumn(center=(3.4, 0.0, 0), width=5.9, height=4.75, title="messages: the trajectory")
        msgs = [
            col.message("assistant", "I'll run the tests first.", size=21),
            col.message("tool_use", 'bash("pytest -q")', size=21, mono=True),
            col.message("tool_result", f"{FAILED} failed, {PASSED} passed", size=21, mono=True),
            col.message("assistant", "So the bug is in cart.py.", size=21),
        ]
        back = m.CurvedArrow(
            col.panel.get_left() + [-0.15, 1.55, 0], nodes[0].get_right() + [0.15, 0.08, 0], angle=-0.5
        ).set_stroke(MUTED, 4).set_color(MUTED)
        legend = m.VGroup()
        for step, color, fill, gloss in (
            ("think", CORAL, CORAL_FILL, "changes only the plan"),
            ("act", SUN, SUN_FILL, "touches the world"),
            ("observe", MINT, MINT_FILL, "brings back what happened"),
        ):
            legend.add(m.VGroup(chip(step, color, fill, width=1.6, height=0.66, size=24), text(gloss, 25, INK)).arrange(m.RIGHT, buff=0.35))
        legend.arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT).move_to([2.6, -0.2, 0])
        quote = text("“The essence of the ReAct loop is appending to the trajectory, over and over.”", 23, INK, slant=m.ITALIC)
        quote.move_to([0, -2.98, 0])
        quote_src = cite(BOOK).next_to(quote, m.DOWN, buff=0.12)

        self.say(
            "[In twenty twenty-two, Yao and colleagues named a pattern for this]: [ReAct, short for reasoning and acting.]",
            m.AnimationGroup(self.new_header("REACT", "Reasoning and acting, interleaved"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(word, shift=0.3 * m.UP, scale=0.9), m.FadeIn(sub, shift=0.2 * m.UP)),
        )
        self.say(
            "[Reasoning alone can't check its facts.] [Acting alone has no plan.]",
            m.FadeIn(think_only, shift=0.2 * m.UP),
            m.FadeIn(act_only, shift=0.2 * m.UP),
        )
        self.say(
            "[ReAct interleaves them]: [think, act, observe], [then think again, with the new fact in view.]",
            m.AnimationGroup(
                m.FadeOut(think_only), m.FadeOut(act_only), m.FadeOut(sub), m.FadeOut(source),
                m.ReplacementTransform(word, small), m.FadeIn(small_src),
            ),
            m.AnimationGroup(
                m.LaggedStart(*[m.FadeIn(n, scale=0.8) for n in nodes], lag_ratio=0.35),
                m.Create(arcs[0]), m.Create(arcs[1]),
            ),
            m.Succession(m.Create(arcs[2]), ring_flash(center, radius, CORAL)),
        )
        self.say(
            "[A thought changes nothing in the world, only the plan.] [An action touches the world.] "
            "[An observation brings back what happened.]",
            m.AnimationGroup(m.FadeIn(legend[0], shift=0.15 * m.RIGHT), m.Indicate(nodes[0], color=CORAL, scale_factor=1.12)),
            m.AnimationGroup(m.FadeIn(legend[1], shift=0.15 * m.RIGHT), m.Indicate(nodes[1], color=SUN, scale_factor=1.12)),
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.15 * m.RIGHT), m.Indicate(nodes[2], color=MINT, scale_factor=1.12)),
        )
        self.say(
            "[In the loop from episode two], [a thought is the model's text], [an action is a tool use block], "
            "[and an observation is a tool result.]",
            m.AnimationGroup(m.FadeOut(legend, shift=0.2 * m.UP), m.FadeIn(col.frame)),
            m.AnimationGroup(col.push(msgs[0]), m.Indicate(nodes[0], color=CORAL, scale_factor=1.12)),
            m.AnimationGroup(col.push(msgs[1]), m.Indicate(nodes[1], color=SUN, scale_factor=1.12)),
            m.AnimationGroup(col.push(msgs[2]), m.Indicate(nodes[2], color=MINT, scale_factor=1.12)),
        )
        self.say(
            "[Every step is appended to the trajectory], [and the whole trajectory goes back to the model, every round.]",
            m.AnimationGroup(col.push(msgs[3]), m.FadeIn(quote), m.FadeIn(quote_src)),
            m.AnimationGroup(m.Create(back), ring_flash(center, radius, CORAL)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the trace

    def trace(self):
        self.next_section(
            "The trace",
            notes=f"Otto's run as a ReAct trace (labels as in Yao et al., 2022). cart.py is executed here: the buggy "
            f"total([3, 4]) = {GOT}, the fixed one = {WANT}. pytest: {FAILED} failed, {PASSED} passed, then {TESTS} passed. "
            f"{TOOL_CALLS} tool calls, {ROUNDS} rounds. Timings (0.41 s, 0.38 s) illustrative.",
        )
        col = ContextColumn(center=(-3.45, -0.45, 0), width=6.5, height=5.6, title="Otto's trajectory", gap=0.1)
        counter = {"Thought": 0, "Act": 0, "Obs": 0}
        rows = []
        for kind, body, mono in TRACE:
            n = None
            if kind in counter:
                counter[kind] += 1
                n = counter[kind]
            rows.append(trace_row(kind, n, body, col.width - 0.36, mono))

        wx, ww = 3.45, 6.4
        run1 = terminal(
            ["$ pytest -q", "! FAILED tests/test_cart.py::test_total", f"! assert {GOT} == {WANT}",
             f"+ where {GOT} = total([3, 4])", f"{FAILED} failed, {PASSED} passed in 0.41s"],
            width=ww, size=19,
        )
        run1.move_to([wx, 0, 0]).align_to([0, 2.62, 0], m.UP)
        box1, out1 = m.VGroup(*run1[:4]), run1[4]
        editor = code_block(BUGGY_SRC, size=19, width=ww, title="cart.py")
        editor.move_to([wx, 0, 0]).align_to(run1, m.DOWN).shift(0.28 * m.DOWN + editor.height * m.DOWN)
        src = editor[1]
        minus_one = glyphs(src, "- 1")
        closing = glyphs(src, "1):")[1:]
        fixed_src = code_block(FIXED_SRC, size=19, width=ww, title="cart.py").move_to(editor)[1]
        slide = glyphs(fixed_src, "items)):")[-2].get_x() - closing[0].get_x()
        where = out1[3]

        run2 = terminal(["$ pytest -q", "." * TESTS + "  [100%]", f"{TESTS} passed in 0.38s"], width=ww, size=19)
        run2.move_to([wx, 0, 0]).align_to([0, 2.62, 0], m.UP)
        run2[4][2].set_color(MINT)
        box2, out2 = m.VGroup(*run2[:4]), run2[4]
        done = m.VGroup(
            pill("no tool call", CORAL, CORAL_FILL, 22),
            text("→", 26, MUTED),
            pill("the loop ends", MUTED, CARD, 22),
        ).arrange(m.RIGHT, buff=0.2).move_to([wx, (run2.get_bottom()[1] + editor.get_top()[1]) / 2, 0])

        kinds = [kind for kind, _, _ in TRACE]
        counts = {k: kinds.count(k) for k in ("Thought", "Act", "Obs", "Answer")}
        assert counts == {"Thought": 4, "Act": 4, "Obs": 4, "Answer": 1} and counts["Act"] == TOOL_CALLS
        tally_head = title_text(f"{ROUNDS} rounds", 34, INK)
        tally_stats = m.VGroup(
            stat(counts["Thought"], "thoughts", CORAL), stat(counts["Act"], "actions", SUN),
            stat(counts["Obs"], "observations", MINT), stat(counts["Answer"], "final answer", CORAL),
        ).arrange(m.RIGHT, buff=0.38, aligned_edge=m.UP)
        tally = m.VGroup(tally_head, tally_stats).arrange(m.DOWN, buff=0.35).move_to(editor)
        self.say(
            "[Here is Otto's run, line by line.] [Thought one: the bug is probably in the checkout code.]",
            m.AnimationGroup(self.new_header("THE TRACE", "Otto, line by line"), m.FadeIn(col.frame)),
            col.push(rows[0]),
        )
        self.say(
            "[Instead of editing, it acts first]: [it runs the tests.]",
            col.push(rows[1]),
            m.AnimationGroup(m.FadeIn(box1, shift=0.2 * m.UP), m.AddTextLetterByLetter(out1[0], time_per_char=0.05)),
        )
        self.say(
            "[The observation]: [one test fails. The total of three and four came back as three], [not seven.]",
            col.push(rows[2]),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in out1[1:]], lag_ratio=0.3),
            m.Circumscribe(out1[2], color=ROSE, buff=0.08),
        )
        cross = wrong_mark(rows[0])
        self.say(
            "[That one observation changes the plan.] [The checkout code was never the problem.]",
            m.Indicate(rows[2], color=MINT, scale_factor=1.04),
            disprove(rows[0], cross),
        )
        self.remove(cross)
        rows[0].add(cross)
        self.say(
            "[Thought two: three is just the first item], [so the total must be skipping the last one.]",
            col.push(rows[3]),
            m.Circumscribe(where, color=CORAL, buff=0.08),
        )
        self.say(
            "[Action]: [read the cart file.]",
            col.push(rows[4]),
            m.FadeIn(editor, shift=0.2 * m.UP),
        )
        self.say(
            "[The observation]: [the loop stops one item early.]",
            col.push(rows[5]),
            m.AnimationGroup(minus_one.animate.set_color(ROSE), m.Circumscribe(minus_one, color=ROSE, buff=0.06)),
        )
        self.say(
            "[Thought three]: [it's off by one. Count every item.]",
            col.push(rows[6]),
            m.Indicate(minus_one, color=ROSE, scale_factor=1.3),
        )
        self.say(
            "[Action]: [edit the file.]",
            col.push(rows[7]),
            m.AnimationGroup(
                m.FadeOut(minus_one, shift=0.25 * m.UP),
                closing.animate.shift(slide * m.RIGHT),
            ),
        )
        src.remove(*minus_one)
        self.say(
            "[The observation: edited.] [Thought four: an edit is a guess too, so test it.]",
            col.push(rows[8]),
            col.push(rows[9]),
        )
        self.say(
            "[Action]: [run the tests again.]",
            col.push(rows[10]),
            m.AnimationGroup(
                m.FadeOut(box1), m.FadeOut(out1),
                m.FadeIn(box2, shift=0.2 * m.UP), m.AddTextLetterByLetter(out2[0], time_per_char=0.05),
            ),
        )
        self.say(
            "[Observation]: [thirteen passed.]",
            col.push(rows[11]),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in out2[1:]], lag_ratio=0.4),
        )
        self.say(
            "[Now Otto answers in words, with no tool call], [and the loop stops.]",
            col.push(rows[12]),
            m.FadeIn(done, shift=0.2 * m.UP),
        )
        self.say(
            f"[{say_number(ROUNDS).capitalize()} rounds in all]: [four thoughts, four actions, four observations], "
            "[and one answer that the tests agree with.]",
            m.AnimationGroup(m.FadeOut(editor, shift=0.2 * m.DOWN), m.FadeIn(tally_head, shift=0.15 * m.UP)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in tally_stats[:3]], lag_ratio=0.3),
            m.FadeIn(tally_stats[3], shift=0.15 * m.UP),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ stopping

    def stopping(self):
        self.next_section(
            "Stopping",
            notes="Stop conditions (Li, AI Agents in Depth, ch. 1): a response without tool calls, a final output tool, "
            "round limits; without a cap 'Agents can become stuck repeating the same tool calls forever' (ch. 2). "
            f"Otto: {ROUNDS} rounds, {TOOL_CALLS} tool calls. The book's currency example (ch. 1): {len(FX_ROUNDS)} rounds, "
            f"{FX_CALLS} tool calls, total ${FX_TOTAL:,.2f}. Errors compound: illustrative 95% per step, "
            f"0.95^20 = {all_right(STEPS):.3f}. MAX_TURNS and the 'finish' tool are written in the harness's style. " + FACTS,
        )
        stops = m.VGroup(
            code_block('if response.stop_reason != "tool_use":\n    return response', 19, 6.7,
                       "1 · a reply with no tool call", CORAL),
            code_block('if block.type == "tool_use":\n    if block.name == "finish":\n        return block.input', 19, 6.7,
                       "2 · a final-output tool", SUN),
            code_block('for turn in range(MAX_TURNS):\n    ...\nreturn "stopped: too many rounds"', 19, 6.7,
                       "3 · a cap on rounds", ROSE),
        ).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to([-3.3, -0.45, 0])
        for block in stops:
            block[2].set_color(INK)

        rx = 3.65
        last = trace_row("Answer", None, TRACE[-1][1], 5.9).move_to([rx, 1.85, 0])
        reason = code('stop_reason: "end_turn"', 20, MUTED, t2c={r'"end_turn"': MINT})
        tally = text(f"{ROUNDS} rounds · {TOOL_CALLS} tool calls", 21, MUTED)
        facts = m.VGroup(reason, tally).arrange(m.DOWN, buff=0.15).next_to(last, m.DOWN, buff=0.25)
        stuck = cite("Li, AI Agents in Depth · ch. 1–2").next_to(stops, m.DOWN, buff=0.12).align_to(stops, m.LEFT)

        # the book's currency example: tool calls per round
        fx_cols, fx_heads = [], m.VGroup()
        for i, calls in enumerate(FX_ROUNDS):
            col = m.VGroup(*[chip(c, SUN, SUN_FILL, width=max(1.75, code(c, 19).width + 0.4), height=0.48, size=23, mono=True) for c in calls])
            if not calls:
                col.add(chip(f"${FX_TOTAL / 1e6:.1f}M", CORAL, CORAL_FILL, width=1.3, height=0.48, size=23, mono=True))
            fx_cols.append(col.arrange(m.DOWN, buff=0.1))
        row = m.VGroup(*fx_cols).arrange(m.RIGHT, buff=0.25, aligned_edge=m.UP).move_to([rx - 0.05, 1.55, 0]).align_to([0, 2.05, 0], m.UP)
        for i, col in enumerate(fx_cols):
            fx_heads.add(code(f"round {i + 1}", 20, MUTED).next_to(col, m.UP, buff=0.15))
        fx_tally = text(f"{len(FX_ROUNDS)} rounds · {FX_CALLS} tool calls", 22, INK, weight=m.SEMIBOLD).next_to(row, m.DOWN, buff=0.3)
        fx_tally.align_to(row, m.LEFT)
        fx_src = cite("Li, AI Agents in Depth · ch. 1").next_to(fx_tally, m.DOWN, buff=0.1).align_to(fx_tally, m.LEFT)
        fx = m.VGroup(fx_heads, *fx_cols, fx_tally, fx_src)

        # errors compound: chance that n steps in a row all go right
        base_y, top_h, bw, gap = -2.95, 2.3, 0.2, 0.085
        x0 = rx - (STEPS * bw + (STEPS - 1) * gap) / 2
        bars = m.VGroup()
        for n in range(1, STEPS + 1):
            f = all_right(n)
            color = m.interpolate_color(m.ManimColor(MINT), m.ManimColor(ROSE), (1 - f) / (1 - all_right(STEPS)))
            b = m.Rectangle(width=bw, height=top_h * f, stroke_width=0).set_fill(color, opacity=0.95)
            bars.add(b.move_to([x0 + (n - 1) * (bw + gap) + bw / 2, base_y, 0], aligned_edge=m.DOWN))
        axis = m.Line([x0 - 0.15, base_y, 0], [x0 + STEPS * (bw + gap), base_y, 0]).set_stroke(FAINT, 2.5)
        ticks = m.VGroup(*[text(str(n), 20, MUTED).next_to(bars[n - 1], m.DOWN, buff=0.12) for n in (1, 10, STEPS)])
        steps_label = text("steps", 20, MUTED).next_to(ticks[1], m.RIGHT, buff=0.5).align_to(ticks[1], m.DOWN)
        marks = m.VGroup(*[
            text(f"{round(100 * all_right(n))}%", 22, INK, weight=m.SEMIBOLD).next_to(bars[n - 1], m.UP, buff=0.2)
            for n in (1, 10, STEPS)
        ])
        marks[2].set_color(ROSE)
        chart_title = text("chance every step goes right", 22, MUTED).move_to([rx, base_y + top_h + 0.62, 0])
        each = text("(95% each, for example)", 20, MUTED).next_to(chart_title, m.DOWN, buff=0.08)

        self.say(
            "[So when should an agent stop?] [Otto stopped the first way]: [in round five, it replied with no tool call.]",
            m.AnimationGroup(self.new_header("STOPPING", "When is the loop done?"), m.FadeIn(stops[0], shift=0.2 * m.UP)),
            m.FadeIn(last, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(facts, shift=0.15 * m.UP), m.Circumscribe(stops[0][1], color=CORAL, buff=0.06)),
        )
        self.say(
            "[Second, a final-output tool]: [the model calls something like finish], [and the harness returns its input as the result.]",
            m.FadeIn(stops[1], shift=0.2 * m.UP),
            m.Indicate(glyphs(stops[1][1], '"finish"'), color=SUN, scale_factor=1.2),
            m.Indicate(glyphs(stops[1][1], "return block.input"), color=SUN, scale_factor=1.08),
        )
        self.say(
            "[Third, a safety net]: [a cap on rounds], [because an agent can get stuck repeating the same call forever.]",
            m.FadeIn(stops[2], shift=0.2 * m.UP),
            m.Indicate(glyphs(stops[2][1], "MAX_TURNS"), color=ROSE, scale_factor=1.2),
            m.FadeIn(stuck),
        )
        self.say(
            "[A cap counts rounds, not tool calls.] [In the book's currency example, round one ran three conversions at once], "
            "[and round two, one calculation.]",
            m.AnimationGroup(m.FadeOut(last, shift=0.2 * m.UP), m.FadeOut(facts, shift=0.2 * m.UP), m.FadeIn(fx_heads)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in fx_cols[0]], lag_ratio=0.15),
            m.FadeIn(fx_cols[1], shift=0.15 * m.DOWN),
        )
        self.say(
            "[Round three was the answer]: [four tool calls, three rounds.]",
            m.FadeIn(fx_cols[2], shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeIn(fx_tally, shift=0.15 * m.UP), m.FadeIn(fx_src)),
        )
        self.say(
            "[Caps matter because errors compound.] [For example, say each step is right ninety-five times in a hundred.]",
            m.AnimationGroup(m.FadeOut(fx), m.FadeIn(chart_title), m.Create(axis)),
            m.AnimationGroup(m.FadeIn(each), m.GrowFromEdge(bars[0], m.DOWN), m.FadeIn(marks[0]), m.FadeIn(ticks[0])),
        )
        self.say(
            "[Then twenty steps in a row all go right] [only about a third of the time.]",
            m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars[1:]], lag_ratio=0.12),
            m.AnimationGroup(m.FadeIn(ticks[1:]), m.FadeIn(steps_label), m.FadeIn(marks[1:])),
        )
        caught = m.VGroup(
            chip("Thought 1", ROSE, ROSE_FILL, width=1.9, height=0.6, size=22, mono=True),
            text("→", 26, MUTED),
            chip("Obs 1", MINT, MINT_FILL, width=1.3, height=0.6, size=22, mono=True),
            text("→", 26, MUTED),
            chip("Thought 2", CORAL, CORAL_FILL, width=1.9, height=0.6, size=22, mono=True),
        ).arrange(m.RIGHT, buff=0.15).move_to([rx, 1.75, 0])
        caught_note = text("a wrong guess, caught after one step", 22, MINT, weight=m.SEMIBOLD).next_to(caught, m.DOWN, buff=0.25)
        self.say(
            "[That's why ReAct checks as it goes]: [Otto's wrong guess was caught by the first observation], "
            "[before anything was built on it.]",
            m.FadeIn(caught[0], shift=0.15 * m.RIGHT),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in caught[1:]], lag_ratio=0.3),
            m.FadeIn(caught_note, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ without tools

    def no_tools(self):
        self.next_section(
            "No tools",
            notes="Context ablation (Li, AI Agents in Depth, ch. 1, Experiment 1-1): without tool definitions the model "
            "'still returns a neatly formatted, confidently worded answer'; without tool results it runs 'blind', retrying "
            "until the iteration budget is gone; \"'produced an answer' is not 'completed the task'\". The tool-less answer's "
            f"text is illustrative. Otto's counts come from TRACE: {FILES_READ} file read, {TEST_RUNS} test runs. " + FACTS,
        )
        xs, w, h, cy = (-4.55, 0.0, 4.55), 4.15, 4.1, 0.35

        def panel(x, title, color):
            box = card(w, h, color, CARD, stroke=2).move_to([x, cy, 0])
            head = code(title, 21, color, weight=m.BOLD).move_to(box.get_top() + 0.36 * m.DOWN)
            return m.VGroup(box, head)

        pa, pb, pc = panel(xs[0], "no tools", ROSE), panel(xs[1], "no tool results", ROSE), panel(xs[2], "ReAct", MINT)

        # A: the confident answer from memory
        answer = text(
            "Root cause: checkout.py\nFix: apply the discount\nbefore the total.\nStatus: resolved ✓",
            21, INK, t2c={"Root cause:": CORAL, "Fix:": CORAL, "Status:": CORAL},
        )
        a_box = card(w - 0.5, answer.height + 0.5, CORAL, CORAL_FILL, radius=0.16, stroke=2)
        a_box.move_to([xs[0], cy + 0.5, 0])
        answer.move_to(a_box).align_to(a_box, m.LEFT).shift(0.25 * m.RIGHT)
        a_stats = m.VGroup(stat(0, "files read", ROSE), stat(0, "tests run", ROSE)).arrange(m.RIGHT, buff=0.6)
        a_stats.move_to([xs[0], cy - 1.35, 0])
        wrong_file = glyphs(answer, "checkout.py")
        a_src = cite(BOOK).move_to([0, -3.38, 0])

        # B: acting blind
        blind = m.VGroup()
        for i in range(2):
            blind.add(trace_row("Act", None, 'bash("pytest -q")', w - 0.3, True, size=21, label_w=0.7))
            blind.add(trace_row("Obs", None, "(hidden)", w - 0.3, False, size=21, label_w=0.7))
        blind.arrange(m.DOWN, buff=0.07).move_to([xs[1], 0, 0]).align_to([0, pb[1].get_bottom()[1] - 0.22, 0], m.UP)
        for r in blind[1::2]:
            r[2][1].set_color(FAINT)
        gauge = Meter(w - 1.6, 0.0, SUN, 0.22).move_to([xs[1] + 0.45, cy - 1.6, 0])
        gauge_label = text("rounds", 21, MUTED).next_to(gauge, m.LEFT, buff=0.2)

        # C: Otto's real run
        c_stats = m.VGroup(stat(FILES_READ, "file read", MINT), stat(TEST_RUNS, "test runs", MINT)).arrange(m.RIGHT, buff=0.6)
        c_stats.move_to([xs[2], cy + 0.8, 0])
        c_pass = code(f"{TESTS} passed", 24, MINT, weight=m.BOLD)
        c_box = card(c_pass.width + 0.7, c_pass.height + 0.45, MINT, MINT_FILL, radius=0.14, stroke=2)
        c_box.move_to([xs[2], cy - 0.55, 0])
        c_pass.move_to(c_box)
        c_bot = otto(0.5).move_to([xs[2], cy - 1.4, 0])

        verdicts = []
        for x, (ans, task) in zip(xs, ((True, False), (False, False), (True, True))):
            pair = m.VGroup(verdict("answer", ans), verdict("task", task)).arrange(m.RIGHT, buff=0.2)
            verdicts.append(pair.move_to([x, cy - h / 2 - 0.45, 0]))
        quote = text("“produced an answer” is not “completed the task”", 26, INK, weight=m.SEMIBOLD, slant=m.ITALIC)
        quote.move_to([0, -2.92, 0])

        self.say(
            "[Now take Otto's tools away], [and ask the same question.]",
            m.AnimationGroup(self.new_header("NO TOOLS", "An answer is not a task"), m.FadeIn(pa, shift=0.2 * m.UP)),
            m.FadeIn(a_box, shift=0.15 * m.UP),
        )
        self.say(
            "[It still answers, neatly formatted and confidently worded.]",
            m.AnimationGroup(m.AddTextLetterByLetter(answer, time_per_char=0.03), m.FadeIn(a_src)),
        )
        self.say(
            "[But it read no files and ran no tests]: [it's the first guess, dressed up as a fix.]",
            m.FadeIn(a_stats, shift=0.15 * m.UP),
            m.AnimationGroup(wrong_file.animate.set_color(ROSE), m.Circumscribe(wrong_file, color=ROSE, buff=0.06)),
        )
        self.say(
            "[Notice that nothing here raises an error.] [The failure looks like a flawless answer.]",
            m.Circumscribe(pa[0], color=MUTED, buff=0.06),
            m.Indicate(m.VGroup(a_box, answer), color=CORAL, scale_factor=1.05),
        )
        self.say(
            "[Give it tools but hide the results], [and it acts blind], [retrying until it runs out of rounds.]",
            m.AnimationGroup(m.FadeIn(pb, shift=0.2 * m.UP), m.FadeIn(gauge), m.FadeIn(gauge_label)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in blind], lag_ratio=0.3),
            gauge.fill_to(1.0, run_time=1.6),
        )
        self.say(
            "[Otto's real run read one file, ran the test suite twice], [and saw it pass.]",
            m.AnimationGroup(m.FadeIn(pc, shift=0.2 * m.UP), m.FadeIn(c_stats, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(c_box, scale=0.9), m.FadeIn(c_pass, scale=0.9), m.FadeIn(c_bot, shift=0.15 * m.UP)),
        )
        self.say(
            "[Two of these runs produced an answer.] [Only one completed the task.]",
            m.LaggedStart(*[m.FadeIn(v[0], shift=0.15 * m.UP) for v in verdicts], lag_ratio=0.25),
            m.LaggedStart(*[m.FadeIn(v[1], shift=0.15 * m.UP) for v in verdicts], lag_ratio=0.25),
        )
        self.say(
            "[In the words of the book AI Agents in Depth]: [producing an answer is not completing the task.]",
            m.Indicate(a_src, color=INK, scale_factor=1.08),
            m.FadeIn(quote, shift=0.2 * m.DOWN),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ a fixed plan

    def fixed_plan(self):
        self.next_section(
            "A fixed plan",
            notes="Contrast: a plan fixed in advance (plan-then-execute without re-planning, like a workflow; Episode 5) "
            "cannot react to an observation. The plan's steps are illustrative. ReAct decides the next step after each "
            "observation (Yao et al., 2022). Plans that update: Episode 9.",
        )
        pw, ph, cy = 6.2, 4.6, -0.35
        left = card(pw, ph, ROSE, CARD, stroke=2).move_to([-3.4, cy, 0])
        right = card(pw, ph, MINT, CARD, stroke=2).move_to([3.4, cy, 0])
        l_head = code("a fixed plan, written up front", 21, ROSE, weight=m.BOLD).move_to(left.get_top() + 0.38 * m.DOWN)
        r_head = code("ReAct: one step at a time", 21, MINT, weight=m.BOLD).move_to(right.get_top() + 0.38 * m.DOWN)

        steps = m.VGroup()
        for i, s in enumerate(("read checkout.py", "fix the total in checkout.py", "reply: fixed"), 1):
            num = chip(str(i), SUN, SUN_FILL, width=0.55, height=0.55, size=22)
            steps.add(m.VGroup(num, code(s, 20, INK)).arrange(m.RIGHT, buff=0.25))
        steps.arrange(m.DOWN, buff=0.78, aligned_edge=m.LEFT).align_to(left, m.LEFT).shift(0.45 * m.RIGHT)
        notes = [
            text("observation: looks fine", 21, MINT),
            text("edits the wrong file anyway", 21, ROSE, weight=m.SEMIBOLD),
            text("claims success", 21, ROSE, weight=m.SEMIBOLD),
        ]
        for s, n in zip(steps, notes):
            n.next_to(s[1], m.DOWN, buff=0.12).align_to(s[1], m.LEFT)
        plan = m.VGroup(steps, *notes)
        plan.shift((l_head.get_bottom()[1] - 0.45 - plan.get_top()[1]) * m.UP)

        rw = pw - 1.15
        mini = m.VGroup(
            trace_row("Thought", 1, "The bug is probably\nin checkout.py.", rw, size=21),
            trace_row("Act", 1, 'bash("pytest -q")', rw, mono=True),
            trace_row("Obs", 1, f"test_total: {GOT} != {WANT}", rw, mono=True),
            trace_row("Thought", 2, "Then cart.total()\nis wrong.", rw, size=21),
        ).arrange(m.DOWN, buff=0.12).move_to(right).shift(0.25 * m.DOWN + 0.3 * m.LEFT)
        mini[0].add(wrong_mark(mini[0]))
        mini[0][2].set_opacity(0.45)
        mini[0][1].set_fill(ROSE)
        turn = m.CurvedArrow(mini[2].get_right() + [0.08, 0.0, 0], mini[3].get_right() + [0.08, 0.0, 0], angle=-2.2)
        turn.set_stroke(MINT, 3.5).set_color(MINT)
        nine = pill("plans that can change · episode 9", SUN, SUN_FILL, 22).move_to([0, -3.3, 0])

        self.say(
            "[Compare a plan fixed in advance], [then carried out step by step.]",
            m.AnimationGroup(self.new_header("A FIXED PLAN", "Plans that can't listen"), m.FadeIn(left), m.FadeIn(l_head)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.RIGHT) for s in steps], lag_ratio=0.35),
        )
        self.say(
            "[Step one finds nothing wrong in checkout], [but step two edits it anyway]: [the plan never hears the observation.]",
            m.FadeIn(notes[0], shift=0.1 * m.DOWN),
            m.FadeIn(notes[1], shift=0.1 * m.DOWN),
            m.FadeIn(notes[2], shift=0.1 * m.DOWN),
        )
        self.say(
            "[ReAct only commits to the next step], [after reading the last observation.]",
            m.AnimationGroup(m.FadeIn(right), m.FadeIn(r_head), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in mini[:3]], lag_ratio=0.3)),
            m.AnimationGroup(m.Create(turn), m.FadeIn(mini[3], shift=0.1 * m.UP)),
        )
        self.say(
            "[Plans still help, as long as they can change]: [more on that in episode nine.]",
            m.Circumscribe(mini[3], color=MINT, buff=0.06),
            m.FadeIn(nine, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Think, act, observe; stop when the world (the tests) confirms the task, not when the answer sounds done.")
        halo = glow(CORAL, radius=4.0)
        words = on_baseline(
            m.VGroup(*[title_text(w, 76, c) for w, c in zip(("Think", "Act", "Observe"), (CORAL, SUN, MINT))]).arrange(m.RIGHT, buff=1.3)
        )
        mid = words[0].get_y()
        links = m.VGroup(*[arrow([a.get_right()[0], mid, 0], [b.get_left()[0], mid, 0], MUTED, buff=0.25) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, links).move_to(0.75 * m.UP)
        again = m.CurvedArrow(words[2].get_bottom() + [0, -0.2, 0], words[0].get_bottom() + [0, -0.2, 0], angle=-0.5)
        again.set_stroke(MUTED, 4).set_color(MUTED)
        tagline = on_baseline(m.VGroup(text("Stop when the tests say it's done,", 30, INK), text("not when the answer sounds done.", 30, MUTED)).arrange(m.RIGHT, buff=0.18))
        tagline.move_to(1.75 * m.DOWN)

        self.say(
            "[To sum up]: [think], [act], [observe], [and again.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
            m.Create(again),
        )
        self.say("[Stop when the tests say it's done], [not when the answer sounds done.]", m.FadeIn(tagline[0], shift=0.15 * m.UP), m.FadeIn(tagline[1], shift=0.15 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, again, tagline)], run_time=0.9)
