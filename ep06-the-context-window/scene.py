"""Agents, Animated · Episode 6 · The Context Window (Module 2 · Context Engineering)

Everything an agent knows in a turn is what its harness puts in the context. Running example: Otto's
context after ten turns on tinyshop (run the tests, read cart.py, search, read checkout.py and the test
file, edit, test, add a test, test, answer), taken apart piece by piece.

Sections: hook (Otto knows the bug, but the model remembers nothing: it is all in the context) ·
rebuilt for every call (client.messages.create: system + tools = static prefix, messages = trajectory)
· piece by piece (a ContextColumn with a token meter filling with every kind of message; 20 messages on
top of a 3,000-token prefix = 9,700 tokens) · tokens and the budget (1–3 tokens per English word; a
200,000-token window, about 5% used; tool results about 60%) · re-sent history (Li's 1,000 + 2,000 +
3,000 = 6,000 for 3,000 new; Otto's ten calls sum to 71,560) · overflow vs rot · the pet-store log
(lookup vs counting) · attention (sink, lost in the middle, the request buried; first or last) · what
the rest of Module 2 does with this picture (Eps 7, 8, 11) · recap: fill, fit, find.

Sources (facts dated October 2026):
- Li, AI Agents in Depth, ch. 2: "Because model APIs are stateless, the Agent framework must reconstruct
  sufficient context for every call"; system prompt + tool definitions "form a static prefix"; "static
  prefix + trajectory"; "one English word typically corresponds to 1–3 tokens"; Experiment 2-2: the first
  token "often absorbs an abnormally high amount of attention weight", sometimes over 70% of the total
  (attention sink); position bias: put the most critical information at the beginning or end; context rot
  vs overflow ("overflow means 'cannot fit any more,' while rot means 'it fits but cannot be found'"); the
  pet-store inspection log (100 cages, 90 black, 10 white; "which cat is in cage 37?" vs "how many black
  cats?"; "Current statistics: 90 black cats, 10 white cats"); tool results "routinely run to tens of
  thousands of characters".
- Li, AI Agents in Depth, ch. 7 (cost analysis): "Round 1 sends 1000 tokens, Round 2 sends 2000…, totaling
  6000 instead of 3000".
- Liu et al., "Lost in the Middle: How Language Models Use Long Contexts", TACL 2024.
- harness_reference.py: SYSTEM, TOOLS (bash, read_file, edit_file), client.messages.create(model, system,
  messages, tools, max_tokens).

Illustrative (said "for example" or labelled on screen): every token count of Otto's pieces and turns,
the 200,000-token window size, the token split of "Run pytest in tinyshop.", the 40,000-token result that
overflows, the cage arrangement (seeded), and the shape of the attention curve. Asserts below tie every
number the narration quotes to the data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep06"

# ---------------------------------------------------------------------------- Otto's context, computed

TASK = "The checkout test fails. Fix it."
WINDOW = 200_000  # a context window size, for example
SYSTEM_TOKENS, TOOLS_TOKENS, TASK_TOKENS = 1_800, 1_200, 10  # illustrative

# Otto's ten turns (illustrative): (tokens the assistant wrote, (tool, argument) or None, what came back, its tokens)
TURNS = [
    (60, ("bash", "pytest -q"), "1 failed, 12 passed", 850),
    (50, ("read_file", "cart.py"), "def total(items): …", 600),
    (70, ("bash", "grep -rn total ."), "9 matches", 700),
    (50, ("read_file", "checkout.py"), "class Checkout: …", 2_400),
    (60, ("read_file", "tests/test_cart.py"), "def test_total(): …", 1_300),
    (120, ("edit_file", "cart.py"), "edited cart.py", 10),
    (40, ("bash", "pytest -q"), "13 passed", 30),
    (180, ("edit_file", "tests/test_cart.py"), "edited tests/test_cart.py", 10),
    (40, ("bash", "pytest -q"), "14 passed", 30),
    (90, None, "Fixed: total() skipped the last item.", 0),
]
LATER = ["search", "read", "read", "edit", "test", "edit", "test", "answer"]  # turns 3–10, as chips


def build_pieces():
    """The context in order: (kind, role label, what a row shows, tokens)."""
    out = [
        ("system", "system", "You are Otto …", SYSTEM_TOKENS),
        ("tools", "tools", "bash, read_file, edit_file", TOOLS_TOKENS),
        ("user", "user", TASK, TASK_TOKENS),
    ]
    for said, call, back, back_tokens in TURNS:
        if call is None:
            out.append(("answer", "assistant", back, said))
            continue
        name, arg = call
        out.append(("tool_use", "assistant", f'{name}("{arg}")', said))
        out.append(("tool_result", "user", back, back_tokens))
    return out


PIECES = build_pieces()
TOTAL = sum(p[3] for p in PIECES)
PREFIX_TOKENS = SYSTEM_TOKENS + TOOLS_TOKENS
MESSAGES = len(PIECES) - 2  # everything after the static prefix
RESULT_TOKENS = sum(p[3] for p in PIECES if p[0] == "tool_result")
ASSISTANT_TOKENS = sum(p[3] for p in PIECES if p[0] in ("tool_use", "answer"))
SHARES = [  # (label, share, colour) for the composition chart
    ("tool results", RESULT_TOKENS / TOTAL, MINT),
    ("static prefix", PREFIX_TOKENS / TOTAL, ORCHID),
    ("Otto's turns", ASSISTANT_TOKENS / TOTAL, CORAL),
    ("user request", TASK_TOKENS / TOTAL, AZURE),
]
RUNNING = [PREFIX_TOKENS + sum(p[3] for p in PIECES[2:k]) for k in range(3, 8)]  # after each of the first rows

# the model is stateless: every call re-sends the context so far
SENT = []
_context = PREFIX_TOKENS + TASK_TOKENS
for _said, _call, _back, _back_tokens in TURNS:
    SENT.append(_context)
    _context += _said + _back_tokens

# Li's arithmetic (ch. 7): 1,000 new tokens per round, three rounds
ROUND_NEW = 1_000
ROUNDS = [ROUND_NEW * k for k in (1, 2, 3)]

# the pet-store inspection log (Li, ch. 2): 100 cages, 10 white cats placed by a seed
CAGES, ASKED = 100, 37
WHITE = set(random.Random(6).sample(range(1, CAGES + 1), 10))
CATS = ["white" if c in WHITE else "black" for c in range(1, CAGES + 1)]

# a token split, for example
WORDS = "Run pytest in tinyshop"
TOKEN_SPLIT = [["Run"], ["py", "test"], ["in"], ["tiny", "shop"]]

REQUEST_AT = PREFIX_TOKENS / TOTAL  # where the user's request sits now (fraction of the context)

assert len(TURNS) == 10 and MESSAGES == 20  # "ten turns", "twenty messages on top of the prefix"
assert PREFIX_TOKENS == 3_000  # "three thousand tokens before any work begins"
assert TURNS[0][3] == 850 and TURNS[1][3] == 600  # "eight hundred and fifty", "six hundred more"
assert len(LATER) == 8 == len(TURNS) - 2  # "eight more turns"
assert TOTAL == 9_700 == _context  # "nine thousand seven hundred tokens"
assert round(100 * TOTAL / WINDOW) == 5  # "about five percent"
assert round(SHARES[0][1], 1) == 0.6  # tool results: "about sixty percent"
assert round(SHARES[1][1], 1) == 0.3  # static prefix: "about thirty percent"
assert SHARES[2][1] < 0.1  # Otto's own messages: "less than a tenth"
assert ROUNDS == [1_000, 2_000, 3_000] and sum(ROUNDS) == 6_000 and ROUND_NEW * len(ROUNDS) == 3_000
assert len(SENT) == 10 and sum(SENT) == 71_560 and round(sum(SENT), -3) == 72_000  # "about seventy-two thousand"
assert SENT[-1] < TOTAL < 10_000  # "a context under ten thousand"
assert CATS.count("black") == 90 and CATS.count("white") == 10 and CATS[ASKED - 1] == "black"
assert " ".join("".join(w) for w in TOKEN_SPLIT) == WORDS and sum(map(len, TOKEN_SPLIT)) == 6  # "4 words, 6 tokens"
assert all(1 <= len(w) <= 3 for w in TOKEN_SPLIT)  # "one to three tokens" a word
assert 0.25 < REQUEST_AT < 0.4  # the request now sits a third of the way in: the middle

# ---------------------------------------------------------------------------- look

LLM = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
ACCENT = {"system": ORCHID, "tools": ORCHID, "user": AZURE, "tool_use": SUN, "tool_result": MINT, "answer": CORAL}
LABEL = {"system": ORCHID, "tools": ORCHID, "user": AZURE, "assistant": CORAL}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL_FILL = "#18150F"

COL_CENTER, COL_W, COL_H = (3.5, -0.55, 0), 6.6, 5.0
ROW_H, ROW_W = 0.46, COL_W - 0.36
BODY_X = 0.28 + code("assistant", 22, weight=m.BOLD).width + 0.25
BAR_H, BAR_GAP = 0.15, 0.05
LEFT_X = -6.75  # left edge of the content area


def cite(s):
    return text(s, 20, MUTED)


def sit(words, base):
    """Put a Text's first glyph on a baseline (all row texts start with a flat-bottomed glyph)."""
    return words.shift((base - words[0].get_bottom()[1]) * m.UP)


def row(piece, width=ROW_W):
    """One piece of the context on one line: a stripe in its colour, the role, the content, its tokens."""
    kind, label, body, tokens = piece
    accent = ACCENT[kind]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    lab = code(label, 22, LABEL[label], weight=m.BOLD)
    words = text(body, 20, INK) if kind in ("system", "tools", "user", "answer") else code(body, 20, accent)
    count = code(f"{tokens:,}", 20, MUTED)
    lab.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(BODY_X * m.RIGHT)
    count.move_to(box).align_to(box, m.RIGHT).shift(0.2 * m.LEFT)
    base = box.get_y() - 0.09
    for t in (lab, words, count):
        sit(t, base)
    assert words.get_right()[0] < count.get_left()[0] - 0.15, body
    return m.VGroup(box, stripe, lab, words, count)


def slim(piece, width, height):
    """A piece as a plain bar in its colour (the zoomed-out view of the context)."""
    return m.RoundedRectangle(corner_radius=min(0.05, height / 2), width=width, height=height, stroke_width=0,
                              fill_color=ACCENT[piece[0]], fill_opacity=1)


def map_heights(total_h, gap, min_h=0.035):
    """Heights proportional to tokens, filling total_h, with a minimum so tiny pieces stay visible."""
    avail = total_h - gap * (len(PIECES) - 1)
    small = set()
    while True:
        scale = (avail - min_h * len(small)) / sum(p[3] for i, p in enumerate(PIECES) if i not in small)
        now = {i for i, p in enumerate(PIECES) if p[3] * scale < min_h}
        if now == small:
            break
        small = now
    return [min_h if i in small else p[3] * scale for i, p in enumerate(PIECES)]


def token_map(x, top, width, total_h, gap=0.025):
    """Otto's context drawn to scale: one bar per piece, height ∝ tokens, in order."""
    blocks = m.VGroup()
    y = top
    for piece, h in zip(PIECES, map_heights(total_h, gap)):
        blocks.add(slim(piece, width, h).move_to([x, y - h / 2, 0]))
        y -= h + gap
    return blocks


def bracket(top, bottom, x, color, label):
    """A thin bracket left of a column, with its label on the left."""
    spine = m.Line([x, top, 0], [x, bottom, 0]).set_stroke(color, 4)
    ticks = m.VGroup(*[m.Line([x, y, 0], [x + 0.13, y, 0]).set_stroke(color, 4) for y in (top, bottom)])
    words = text(label, 22, color, weight=m.SEMIBOLD).next_to(spine, m.LEFT, buff=0.15)
    return m.VGroup(spine, ticks, words)


def model_card(at, width=2.2, height=2.0):
    box = card(width, height, CORAL, CORAL_FILL)
    body = m.VGroup(icon(LLM, 0.85), text("model", 24, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.15).move_to(box)
    return m.VGroup(box, body).move_to(at)


def send(group, target):
    """A miniature copy of `group` flying into `target` and vanishing there (one cue)."""
    p = group.copy().scale(0.3).move_to(target.get_center()).set_z_index(5)
    out = m.FadeOut(p, scale=0.3, run_time=0.35)
    out.introducer = True  # p is introduced by the FadeIn just before, in the same line
    return m.Succession(m.FadeIn(p, target_position=group.get_center(), run_time=0.8), out)


def cat(color, r=0.1):
    """A cat in a cage log: a round head with two ears."""
    fill, edge = ("#0D0B09", MUTED) if color == "black" else (INK, INK)
    head = m.Circle(radius=r).set_fill(fill, 1).set_stroke(edge, 1.4)
    ears = m.VGroup(*[
        m.Polygon([s * 0.35 * r, 0.75 * r, 0], [s * 0.95 * r, 0.55 * r, 0], [s * 0.85 * r, 1.35 * r, 0])
        .set_fill(fill, 1).set_stroke(edge, 1.4)
        for s in (-1, 1)
    ])
    return m.VGroup(ears, head)


def attention_weight(p):
    """An illustrative U-shape: strong at the start and the end, weak in the middle."""
    return 0.14 + 0.86 * abs(2 * p - 1) ** 1.8


class Slot:
    """One place on screen whose content is swapped (a counter, the detail card)."""

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


class Ep06TheContextWindow(Episode):
    number = 6

    def lesson(self):
        self.hook()
        self.rebuilt()
        self.piece_by_piece()
        self.budget()
        self.resent()
        self.overflow_rot()
        self.pet_store()
        self.attention()
        self.ahead()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Ten turns into a tinyshop session Otto 'knows' the bug, the file and the failing test, yet the model behind it remembers nothing between calls (Episode 2): all of it lives in the context the harness sends. The facts are those of the series' running bug (harness_reference.py).")
        bot = otto(1.5).move_to([-5.0, 0.35, 0])
        facts = [("bug", "total() skips the last item"), ("file", "cart.py"), ("test", "test_total")]
        val_w = max(code(v, 20).width for _, v in facts)
        chips = m.VGroup()
        for key, value in facts:
            box = card(val_w + 1.55, 0.7, MINT, MINT_FILL, radius=0.16, stroke=2)
            k = text(key, 22, MUTED, weight=m.SEMIBOLD).move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            v = code(value, 20, INK).move_to(box).align_to(box, m.LEFT).shift(1.15 * m.RIGHT)
            base = box.get_y() - 0.09
            sit(k, base)
            sit(v, base)
            chips.add(m.VGroup(box, k, v))
        chips.arrange(m.DOWN, buff=0.25).move_to([-0.3, 0.3, 0])
        model = model_card([5.4, 0.3, 0])
        forgets = pill("remembers nothing", ROSE, ROSE_FILL, 22).next_to(model, m.DOWN, buff=0.35)
        asks = title_text("?", 64, SUN).move_to([-3.75, 1.75, 0])
        panel = card(chips.width + 0.6, chips.height + 0.7, EDGE, PANEL_FILL, radius=0.2, stroke=2).move_to(chips).set_z_index(-1)
        title = code("context window", 20, MUTED).next_to(panel, m.UP, buff=0.12).align_to(panel, m.LEFT)
        link = arrow(panel.get_right(), model.get_left(), SUN, buff=0.15)
        every = text("every call", 22, MUTED).next_to(link, m.UP, buff=0.12)
        turns = pill("after ten turns", SUN, SUN_FILL, 20).next_to(panel, m.UP, buff=0.12).align_to(panel, m.RIGHT)

        self.say(
            "[Ten turns into a session on tinyshop], [Otto knows the bug], [the file it lives in], [and the test that catches it.]",
            m.FadeIn(bot, shift=0.3 * m.UP),
            m.FadeIn(chips[0], shift=0.2 * m.RIGHT),
            m.FadeIn(chips[1], shift=0.2 * m.RIGHT),
            m.FadeIn(chips[2], shift=0.2 * m.RIGHT),
        )
        self.say(
            "[But the model behind Otto remembers nothing between calls.] [So where does that knowledge live?]",
            m.AnimationGroup(m.FadeIn(model, shift=0.2 * m.LEFT), m.FadeIn(forgets, shift=0.1 * m.UP)),
            m.FadeIn(asks, scale=0.6),
        )
        self.say(
            "[In the context window]: [the text the harness hands the model on every call.]",
            m.AnimationGroup(m.FadeOut(asks), m.FadeIn(panel), m.FadeIn(title)),
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(every), send(m.VGroup(panel, chips), model)),
        )
        self.say(
            "[Today we open Otto's context], [and take it apart, piece by piece.]",
            m.AnimationGroup(m.Circumscribe(panel, color=SUN, buff=0.08), m.FadeIn(turns, shift=0.1 * m.DOWN)),
            m.LaggedStart(*[m.Circumscribe(c, color=MINT, buff=0.04) for c in chips], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ rebuilt for every call

    def rebuilt(self):
        self.next_section("1 · Rebuilt for every call", notes="Model APIs are stateless, so the harness rebuilds the context for every call (Li, AI Agents in Depth, ch. 2: 'Because model APIs are stateless, the Agent framework must reconstruct sufficient context for every call'). In harness_reference's client.messages.create, system and tools are the same on every call (the static prefix); messages is the trajectory. Li: 'Agent context = static prefix + trajectory'.")
        src = (
            "response = client.messages.create(\n"
            "    model=MODEL,\n"
            "    system=SYSTEM,\n"
            "    messages=messages,\n"
            "    tools=TOOLS,\n"
            "    max_tokens=8000)"
        )
        block = code_block(src, size=20, title="harness_reference.py · agent_loop")
        block.move_to([0, 0, 0]).align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, 2.35, 0], m.UP)
        block[1].set_z_index(2)
        block[2].set_z_index(2)
        body = block[1]

        def line_bar(key, color, opacity=0.16):
            g = glyphs(body, key)
            left, right = block[0].get_left()[0] + 0.12, block[0].get_right()[0] - 0.12
            rect = m.Rectangle(width=right - left, height=0.36, stroke_width=0, fill_color=color, fill_opacity=opacity)
            rect.move_to([(left + right) / 2, g.get_center()[1], 0])
            mark = m.Rectangle(width=0.05, height=0.36, stroke_width=0, fill_color=color, fill_opacity=1)
            mark.move_to([left + 0.025, g.get_center()[1], 0])
            return m.VGroup(rect, mark).set_z_index(1)

        bar_sys, bar_tools = line_bar("system=SYSTEM", ORCHID), line_bar("tools=TOOLS", ORCHID)
        bar_msgs = line_bar("messages=messages", INK, 0.1)

        sx, top = 0.35, 2.2
        pre_box = card(1.6, 1.0, ORCHID, ORCHID_FILL, radius=0.12, stroke=2)
        pre_words = m.VGroup(text("system", 20, INK), text("+ tools", 20, INK)).arrange(m.DOWN, buff=0.06).move_to(pre_box)
        prefix = m.VGroup(pre_box, pre_words).move_to([sx, top - 0.5, 0])
        kinds = ["user", "tool_use", "tool_result", "tool_use", "tool_result"]
        traj = m.VGroup()
        for i, kind in enumerate(kinds):
            b = m.RoundedRectangle(corner_radius=0.06, width=1.6, height=0.2, stroke_width=0, fill_color=ACCENT[kind], fill_opacity=1)
            traj.add(b.move_to([sx, top - 1.0 - 0.14 - 0.1 - i * 0.28, 0]))
        lab_prefix = text("static prefix", 24, ORCHID, weight=m.SEMIBOLD).next_to(prefix, m.RIGHT, buff=0.25)
        lab_traj = text("trajectory", 24, INK, weight=m.SEMIBOLD).next_to(traj, m.RIGHT, buff=0.25)
        model = model_card([5.55, 1.2, 0], width=2.0, height=1.9)
        calls = Slot(self)
        call_at = lambda n: code(f"call {n}", 22, MUTED).next_to(model, m.DOWN, buff=0.3)  # noqa: E731

        formula = title_text("context = static prefix + trajectory", 44, INK,
                             t2c={"context": SUN, "static prefix": ORCHID, "=": MUTED, "+": MUTED})
        formula.move_to([0, -2.1, 0])
        src_b = cite("Li, AI Agents in Depth · ch. 2").next_to(formula, m.DOWN, buff=0.3)

        self.say(
            "[Before every call, the harness rebuilds the context from scratch], [and sends all of it to the model.]",
            m.AnimationGroup(
                self.new_header("01 · rebuilt", "Rebuilt for every call"), m.FadeIn(block, shift=0.15 * m.UP),
                m.FadeIn(model, shift=0.15 * m.LEFT), m.FadeIn(prefix, shift=0.15 * m.UP), m.FadeIn(traj[0], shift=0.15 * m.UP),
                calls.to(call_at(1)),
            ),
            send(m.VGroup(prefix, traj[0]), model),
        )
        self.say(
            "[Look at the call.] [The system prompt and the tools are the same every time]: [that's the static prefix.]",
            m.Circumscribe(block[0], color=MUTED, buff=0.06),
            m.AnimationGroup(m.FadeIn(bar_sys), m.FadeIn(bar_tools)),
            m.AnimationGroup(m.FadeIn(lab_prefix, shift=0.15 * m.LEFT), m.Circumscribe(prefix, color=ORCHID, buff=0.06)),
        )
        self.say(
            "[The messages list is the trajectory], [and it grows with every turn]: [call two], [call three.]",
            m.AnimationGroup(m.FadeIn(bar_msgs), m.FadeIn(lab_traj, shift=0.15 * m.LEFT)),
            m.Circumscribe(traj[0], color=AZURE, buff=0.05),
            m.Succession(m.AnimationGroup(calls.to(call_at(2)), m.FadeIn(traj[1:3], shift=0.15 * m.UP)), send(m.VGroup(prefix, traj[:3]), model)),
            m.Succession(m.AnimationGroup(calls.to(call_at(3)), m.FadeIn(traj[3:5], shift=0.15 * m.UP)), send(m.VGroup(prefix, traj), model)),
        )
        self.say(
            "[The book AI Agents in Depth sums it up in one line]: [agent context equals static prefix plus trajectory.]",
            m.FadeIn(src_b),
            m.FadeIn(formula, shift=0.2 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ piece by piece

    def piece_by_piece(self):
        self.next_section("2 · Piece by piece", notes=f"Otto's context after ten turns, pushed row by row into a ContextColumn with a token meter (kit). Static prefix = system prompt {SYSTEM_TOKENS:,} + tool definitions {TOOLS_TOKENS:,} = {PREFIX_TOKENS:,} tokens; then the request, ten assistant turns and nine tool results: {MESSAGES} messages, {TOTAL:,} tokens in all (computed). All token counts are illustrative; the system prompt's first sentence and the bash schema are harness_reference.py's. Episode 8 assembles a real system prompt.")
        col = ContextColumn(center=COL_CENTER, width=COL_W, height=COL_H, title="context window", gap=0.08, meter=True)
        rows = [row(p) for p in PIECES[:7]]
        counter = Slot(self)
        count_at = lambda n: code(f"{n:,} tokens", 22, INK).align_to(col.panel, m.RIGHT).set_y(col.panel.get_bottom()[1] - 0.32)  # noqa: E731
        detail = Slot(self)
        top_left = lambda mob, top=2.3: mob.align_to([LEFT_X, 0, 0], m.LEFT).align_to([0, top, 0], m.UP)  # noqa: E731

        sys_box = card(4.95, 2.0, ORCHID, CARD, radius=0.16, stroke=2)
        sys_cap = code("system prompt", 20, ORCHID, weight=m.BOLD)
        sys_lines = m.VGroup(text("You are Otto, a coding agent\nworking in the tinyshop repo.", 22, INK),
                             text("+ how to work, the rules, the repo …", 20, MUTED)).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        sys_card = m.VGroup(sys_box, m.VGroup(sys_cap, sys_lines).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).move_to(sys_box))
        top_left(sys_card)

        schema = ('{"name": "bash",\n "description":\n   "Run a shell command…",\n'
                  ' "input_schema": {\n   "properties": {\n     "command": …}}}')
        body = code(schema, 20, INK, t2c={r"\"[^\"\n]*\"(?=\s*:)": AZURE, r"\"[^\"\n]*\"": MINT})
        cap = code("tool definition · 1 of 3", 20, ORCHID, weight=m.BOLD)
        box = card(max(body.width, cap.width) + 0.7, body.height + cap.height + 0.85, ORCHID, CARD, radius=0.16, stroke=2)
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.2 * m.DOWN)
        body.next_to(cap, m.DOWN, buff=0.25).align_to(cap, m.LEFT)
        tool_def = top_left(m.VGroup(box, cap, body))

        ask = top_left(bubble(text(TASK, 26, INK), AZURE, AZURE_FILL, tail="left"), 0.7)
        call = top_left(tool_call("bash", {"command": "pytest -q"}, size=20, id="toolu_01"))
        term = top_left(terminal(["$ pytest -q", "! FAILED test_total", "assert 3 == 7", "… traceback …", "1 failed, 12 passed"], width=4.95, size=20, title="tinyshop"))
        cart = code_block("def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s", size=20, title="read_file · cart.py")
        top_left(cart, 0.9)
        later = m.VGroup(*[
            chip(w, CORAL if w == "answer" else SUN, CORAL_FILL if w == "answer" else SUN_FILL,
                 width=text(w, 24, weight=m.SEMIBOLD).width + 0.45, height=0.62, size=24)
            for w in LATER
        ])
        later[:4].arrange(m.RIGHT, buff=0.15)
        later[4:].arrange(m.RIGHT, buff=0.15)
        m.VGroup(later[:4], later[4:]).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        later_cap = code("turns 3 to 10", 20, MUTED)
        later_box = top_left(m.VGroup(later_cap, later).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT), 1.0)
        big = title_text(f"{TOTAL:,} tokens", 60, SUN)
        big_sub = text(f"{MESSAGES} messages + the static prefix", 26, MUTED)
        big_box = m.VGroup(big, big_sub).arrange(m.DOWN, buff=0.25).move_to([-3.9, 0.75, 0])

        # the zoomed-out view: every piece as a slim bar
        top = col.top
        bars = [slim(p, ROW_W, BAR_H).move_to([col.panel.get_x(), top - BAR_H / 2 - i * (BAR_H + BAR_GAP), 0]) for i, p in enumerate(PIECES)]
        bx = col.panel.get_left()[0] - 0.2
        row_top = lambda i: top - i * (ROW_H + 0.08)  # noqa: E731
        pre_row = bracket(row_top(0), row_top(1) - ROW_H, bx, ORCHID, "static prefix")
        pre_bar = bracket(bars[0].get_top()[1], bars[1].get_bottom()[1], bx, ORCHID, "static prefix")
        traj_bar = bracket(bars[2].get_top()[1], bars[-1].get_bottom()[1], bx, INK, "trajectory")
        self.ctx = (col, bars, counter, pre_row, traj_bar)  # pre_row becomes pre_bar on screen

        self.say(
            "[Here is Otto's context after those ten turns], [built up one piece at a time.]",
            m.AnimationGroup(self.new_header("02 · piece by piece", "Otto's context, after ten turns"), m.FadeIn(col.frame)),
            counter.to(count_at(0)),
        )
        self.say(
            "[First, the system prompt]: [who Otto is, and how it should work.]",
            detail.to(sys_card),
            m.AnimationGroup(col.push(rows[0]), counter.to(count_at(SYSTEM_TOKENS)), col.meter.fill_to(SYSTEM_TOKENS / WINDOW)),
        )
        self.say(
            "[Then the tool definitions]: [a name, a description and an input schema for each tool.]",
            detail.to(tool_def),
            m.AnimationGroup(col.push(rows[1]), counter.to(count_at(PREFIX_TOKENS)), col.meter.fill_to(PREFIX_TOKENS / WINDOW)),
        )
        self.say(
            "[Together they form the static prefix]: [three thousand tokens before any work begins.]",
            m.FadeIn(pre_row, shift=0.1 * m.RIGHT),
            m.Circumscribe(counter.current, color=ORCHID, buff=0.06),
        )
        self.say(
            "[Then the trajectory starts with the user's request]: [the checkout test fails, fix it.]",
            m.AnimationGroup(detail.to(ask), col.push(rows[2]), counter.to(count_at(RUNNING[0]))),
            m.Circumscribe(rows[2], color=AZURE, buff=0.04),
        )
        self.say(
            "[Each assistant turn holds a little text and a tool call]: [here, run the tests.]",
            detail.to(call),
            m.AnimationGroup(col.push(rows[3]), counter.to(count_at(RUNNING[1])), col.meter.fill_to(RUNNING[1] / WINDOW)),
        )
        self.say(
            "[The harness runs it], [and the output comes back as a tool result]: [eight hundred and fifty tokens of it.]",
            detail.to(term),
            m.AnimationGroup(col.push(rows[4]), counter.to(count_at(RUNNING[2])), col.meter.fill_to(RUNNING[2] / WINDOW)),
            m.Circumscribe(rows[4][4], color=MINT, buff=0.06),
        )
        self.say(
            "[Turn two reads the cart file]: [six hundred more.]",
            m.AnimationGroup(detail.to(cart), col.push(rows[5]), counter.to(count_at(RUNNING[3]))),
            m.AnimationGroup(col.push(rows[6]), counter.to(count_at(RUNNING[4])), col.meter.fill_to(RUNNING[4] / WINDOW)),
        )
        self.say(
            "[Eight more turns follow]: [searching, reading, editing and testing], [until Otto answers.]",
            m.AnimationGroup(
                *[m.FadeTransform(r, b) for r, b in zip(rows, bars)], m.Transform(pre_row, pre_bar),
                detail.to(later_box),
            ),
            m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.UP) for b in bars[7:-1]], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(bars[-1], shift=0.1 * m.UP), counter.to(count_at(TOTAL)), col.meter.fill_to(TOTAL / WINDOW)),
        )
        self.say(
            "[That's twenty messages on top of the prefix], [nine thousand seven hundred tokens]: [everything Otto knows right now.]",
            m.AnimationGroup(m.FadeIn(traj_bar, shift=0.1 * m.RIGHT), detail.to(None)),
            m.AnimationGroup(m.FadeIn(big, scale=0.85), m.FadeIn(big_sub, shift=0.1 * m.UP)),
            m.Circumscribe(col.panel, color=SUN, buff=0.06),
        )
        self.big = big_box
        self.wait(0.3)

    # ------------------------------------------------------------------ tokens and the budget

    def budget(self):
        col, bars, counter, pre_row, traj_bar = self.ctx
        pcts = ", ".join(f"{label} {100 * share:.1f}%" for label, share, _ in SHARES)
        self.next_section("3 · Tokens and the budget", notes=f"Tokens: 'one English word typically corresponds to 1–3 tokens' (Li, AI Agents in Depth, ch. 2); the split of '{WORDS}' is illustrative. Window: {WINDOW:,} tokens, for example (Claude models offer 200K, some larger; Li ch. 4 also speaks of 'a 200K context window'). Otto uses {TOTAL:,} = {100 * TOTAL / WINDOW:.2f}%. Composition (computed): {pcts}. Li ch. 2: tool results 'routinely run to tens of thousands of characters'.")
        cx = -3.55
        chips = m.VGroup()
        word_groups = m.VGroup()
        for w, word in enumerate(TOKEN_SPLIT):
            g = m.VGroup()
            for t, tok in enumerate(word):
                color, fill = (SUN, SUN_FILL) if (w + t) % 2 == 0 else (MUTED, CARD_HI)
                g.add(chip(tok, color, fill, width=code(tok, 20).width + 0.4, height=0.6, size=24, mono=True))
            g.arrange(m.RIGHT, buff=0.05)
            word_groups.add(g)
        word_groups.arrange(m.RIGHT, buff=0.32).move_to([cx, 1.45, 0])
        chips = m.VGroup(*[c for g in word_groups for c in g])
        sentence = text(WORDS, 30, INK).move_to([cx, 2.2, 0])
        caption = text(f"{len(WORDS.split())} words, {len(chips)} tokens · for example", 22, MUTED).next_to(word_groups, m.DOWN, buff=0.22)
        src = cite("Li, AI Agents in Depth · ch. 2").next_to(caption, m.DOWN, buff=0.1)

        meter = Meter(6.3, 0.0, SUN, 0.22).move_to([cx, -0.55, 0])
        meter_cap = text(f"context window · {WINDOW:,} tokens, for example", 22, MUTED).next_to(meter, m.UP, buff=0.18).align_to(meter, m.LEFT)
        used = code(f"{TOTAL:,} used · about {round(100 * TOTAL / WINDOW)}%", 22, SUN).next_to(meter, m.DOWN, buff=0.18).align_to(meter, m.LEFT)

        heights = map_heights(col.top - col.bottom, 0.025)
        prop, y = [], col.top
        for piece, h in zip(PIECES, heights):
            prop.append(slim(piece, ROW_W, h).move_to([col.panel.get_x(), y - h / 2, 0]))
            y -= h + 0.025

        comp = m.VGroup()
        full = 4.2
        for k, (label, share, color) in enumerate(SHARES):
            yk = -1.75 - 0.52 * k
            lab = text(label, 22, INK).move_to([-4.45, yk, 0], aligned_edge=m.RIGHT)
            bar = m.RoundedRectangle(corner_radius=0.06, width=max(full * share, 0.04), height=0.3, stroke_width=0, fill_color=color, fill_opacity=1)
            bar.move_to([-4.25, yk, 0], aligned_edge=m.LEFT)
            pct = f"{100 * share:.0f}%" if share >= 0.01 else f"{100 * share:.1f}%"
            num = title_text(pct, 26, color).next_to(bar, m.RIGHT, buff=0.15)
            comp.add(m.VGroup(lab, bar, num))
        for c in comp:
            sit(c[2], c[0][0].get_bottom()[1])

        self.say(
            "[All of this is counted in tokens], [small chunks of text.] [An English word is about one to three tokens.]",
            m.AnimationGroup(self.new_header("03 · tokens", "A budget counted in tokens"), m.FadeOut(self.big), m.FadeOut(pre_row), m.FadeOut(traj_bar), m.FadeIn(sentence, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(caption), m.FadeIn(src)),
        )
        self.say(
            "[The window is a fixed budget], [say, two hundred thousand tokens.] [Otto has used about five percent of it.]",
            m.AnimationGroup(m.FadeIn(meter), m.FadeIn(meter_cap, shift=0.1 * m.UP)),
            m.Circumscribe(meter, color=SUN, buff=0.08),
            m.AnimationGroup(meter.fill_to(TOTAL / WINDOW), m.FadeIn(used, shift=0.1 * m.UP), m.Circumscribe(counter.current, color=SUN, buff=0.06)),
        )
        self.say(
            "[Size each piece by its tokens], [and the tool results stand out]: [about sixty percent of Otto's context.]",
            m.AnimationGroup(*[m.Transform(b, p) for b, p in zip(bars, prop)]),
            m.AnimationGroup(m.FadeIn(comp[0][0]), m.GrowFromEdge(comp[0][1], m.LEFT)),
            m.FadeIn(comp[0][2], scale=0.8),
        )
        self.say(
            "[The static prefix is about thirty percent], [and Otto's own messages, less than a tenth.]",
            m.AnimationGroup(m.FadeIn(comp[1][0]), m.GrowFromEdge(comp[1][1], m.LEFT), m.FadeIn(comp[1][2])),
            m.AnimationGroup(*[m.AnimationGroup(m.FadeIn(c[0]), m.GrowFromEdge(c[1], m.LEFT), m.FadeIn(c[2])) for c in comp[2:]]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ re-sent history

    def resent(self):
        self.next_section("4 · Sent again, every call", notes=f"Li, AI Agents in Depth, ch. 7 (cost analysis): 'Round 1 sends 1000 tokens, Round 2 sends 2000…, totaling 6000 instead of 3000'. Otto's ten calls send the context so far each time: {', '.join(f'{s:,}' for s in SENT)} = {sum(SENT):,} tokens for a {TOTAL:,}-token context (computed from the illustrative counts). Episode 7: KV cache and prompt caching (cache reads cost about a tenth, per Li ch. 2).")
        lx, rx = -3.6, 3.65
        l_title = text(f"Li's example: {ROUND_NEW:,} new tokens a round", 24, MUTED).move_to([lx, 2.15, 0])
        rounds = m.VGroup()
        for k, n in enumerate(ROUNDS, start=1):
            yk = 1.35 - 0.75 * (k - 1)
            label = code(f"round {k}", 20, MUTED).move_to([LEFT_X + 0.05, yk, 0], aligned_edge=m.LEFT)
            blocks = m.VGroup()
            for j in range(k):
                new = j == k - 1
                b = card(1.15, 0.46, SUN, SUN if new else SUN_FILL, radius=0.08, stroke=2 if new else 1.5)
                t = code(f"{ROUND_NEW:,}", 20, BG if new else MUTED).move_to(b)
                blocks.add(m.VGroup(b, t))
            blocks.arrange(m.RIGHT, buff=0.08).move_to([-5.35, yk, 0], aligned_edge=m.LEFT)
            total = code(f"{n:,} sent", 20, INK).move_to([-1.6, yk, 0], aligned_edge=m.LEFT)
            rounds.add(m.VGroup(label, blocks, total))
        l_total = title_text(f"{sum(ROUNDS):,} sent", 44, SUN).move_to([lx, -1.0, 0])
        l_sub = text(f"for {ROUND_NEW * len(ROUNDS):,} tokens of new content", 24, MUTED).next_to(l_total, m.DOWN, buff=0.18)
        l_src = cite("Li, AI Agents in Depth · ch. 7").next_to(l_sub, m.DOWN, buff=0.22)

        r_title = text("Otto's ten calls", 24, MUTED).move_to([rx, 2.15, 0])
        scale = 3.9 / max(SENT)
        calls = m.VGroup()
        for k, n in enumerate(SENT, start=1):
            yk = 1.6 - 0.36 * (k - 1)
            label = code(f"call {k}", 20, MUTED).move_to([0.45, yk, 0], aligned_edge=m.LEFT)
            pre = m.Rectangle(width=PREFIX_TOKENS * scale, height=0.22, stroke_width=0, fill_color=ORCHID, fill_opacity=1)
            rest = m.Rectangle(width=(n - PREFIX_TOKENS) * scale, height=0.22, stroke_width=0, fill_color=MUTED, fill_opacity=0.8)
            bar = m.VGroup(pre, rest).arrange(m.RIGHT, buff=0).move_to([1.75, yk, 0], aligned_edge=m.LEFT)
            num = code(f"{n:,}", 20, INK).next_to(bar, m.RIGHT, buff=0.15)
            calls.add(m.VGroup(label, bar, num))
        r_total = title_text(f"{sum(SENT):,} sent", 44, SUN).move_to([rx, -2.55, 0])
        r_sub = text(f"for a context of {TOTAL:,} tokens", 24, MUTED).next_to(r_total, m.DOWN, buff=0.18)
        legend = m.VGroup(
            m.Square(0.2, stroke_width=0, fill_color=ORCHID, fill_opacity=1), text("static prefix", 20, MUTED),
            m.Square(0.2, stroke_width=0, fill_color=MUTED, fill_opacity=0.8), text("trajectory", 20, MUTED),
        ).arrange(m.RIGHT, buff=0.15)
        legend[2].shift(0.2 * m.RIGHT)
        legend[3].shift(0.2 * m.RIGHT)
        legend.next_to(r_title, m.DOWN, buff=0.12)
        calls.shift(0.25 * m.DOWN)
        nxt = pill("Episode 7 · KV Cache & Prompt Caching", SUN, SUN_FILL, 22).move_to([lx, -3.25, 0])

        self.say(
            "[And every call sends the whole context again.] [AI Agents in Depth does the arithmetic.]",
            self.new_header("04 · sent again", "Every call re-sends it all"),
            m.FadeIn(l_title, shift=0.1 * m.UP),
        )
        self.say(
            "[Round one sends a thousand tokens], [round two, two thousand], [round three, three thousand.]",
            *[m.AnimationGroup(m.FadeIn(r[0]), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.RIGHT) for b in r[1]], lag_ratio=0.2), m.FadeIn(r[2])) for r in rounds],
        )
        self.say(
            "[That's six thousand tokens sent], [for only three thousand tokens of new content.]",
            m.FadeIn(l_total, scale=0.85),
            m.AnimationGroup(m.FadeIn(l_sub, shift=0.1 * m.UP), m.FadeIn(l_src), *[m.Circumscribe(r[1][-1], color=SUN, buff=0.05) for r in rounds]),
        )
        self.say(
            "[Otto's ten calls add up the same way]: [about seventy-two thousand tokens sent], [for a context under ten thousand.]",
            m.AnimationGroup(m.FadeIn(r_title), m.FadeIn(legend), m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c[0]), m.GrowFromEdge(c[1], m.LEFT), m.FadeIn(c[2])) for c in calls], lag_ratio=0.18)),
            m.FadeIn(r_total, scale=0.85),
            m.FadeIn(r_sub, shift=0.1 * m.UP),
        )
        self.say(
            "[That's why episode seven is about caching]: [it makes the repeated prefix much cheaper.]",
            m.FadeIn(nxt, shift=0.1 * m.UP),
            m.LaggedStart(*[m.Indicate(c[1][0], color=ORCHID, scale_factor=1.15) for c in calls], lag_ratio=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ overflow vs rot

    def overflow_rot(self):
        self.next_section("5 · Overflow and rot", notes="Li, AI Agents in Depth, ch. 2: 'Context rot is different from context overflow (running out of window space): overflow means \"cannot fit any more,\" while rot means \"it fits but cannot be found.\"' Overflow ends in an API error (prompt too long). The 40,000-token result and the fill levels are illustrative; the 'needle' is turn one's test failure.")
        panels = m.VGroup(*[card(6.3, 5.15, EDGE, PANEL_FILL, radius=0.2, stroke=2) for _ in range(2)])
        panels.arrange(m.RIGHT, buff=0.5).move_to([0, -0.5, 0])
        titles = m.VGroup(title_text("Overflow", 32, ROSE), title_text("Rot", 32, SUN))
        subs = m.VGroup(text("“cannot fit any more”", 24, INK), text("“it fits but cannot be found”", 24, INK))
        for t, s, p in zip(titles, subs, panels):
            t.move_to(p.get_corner(m.UL) + [0.35, -0.45, 0], aligned_edge=m.LEFT)
            s.next_to(t, m.DOWN, buff=0.18).align_to(t, m.LEFT)

        def mini(cx):
            frame = card(2.1, 2.75, EDGE, CARD, radius=0.14, stroke=2).move_to([cx, -1.6, 0])
            blocks = token_map(cx, frame.get_top()[1] - 0.12, 1.75, 2.75 - 0.24, gap=0.02)
            return frame, blocks

        lf, lb = mini(panels[0].get_left()[0] + 1.45)
        rf, rb = mini(panels[1].get_left()[0] + 1.45)
        lm = Meter(2.1, 0.8, SUN, 0.2).next_to(lf, m.UP, buff=0.25)
        rm = Meter(2.1, 0.55, SUN, 0.2).next_to(rf, m.UP, buff=0.25)
        incoming_box = card(2.7, 1.05, MINT, CARD, radius=0.14, stroke=2)
        incoming = m.VGroup(incoming_box, m.VGroup(code("tool_result", 20, MINT, weight=m.BOLD), code("40,000 tokens", 20, INK)).arrange(m.DOWN, buff=0.1).move_to(incoming_box))
        incoming.move_to([panels[0].get_right()[0] - 1.65, -1.0, 0])
        err = pill("prompt too long", ROSE, ROSE_FILL, 22, mono=True).next_to(incoming, m.DOWN, buff=0.4)

        needle = rb[4]  # turn one's test failure
        tag_x = panels[1].get_right()[0] - 1.6
        tag = m.VGroup(code("turn 1:", 20, MINT), code("test failure", 20, MINT)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        tag.move_to([tag_x, needle.get_y() + 0.2, 0])
        link = m.DashedLine(tag.get_left() + 0.1 * m.LEFT, needle.get_right() + 0.08 * m.RIGHT, dash_length=0.08).set_stroke(MINT, 2)
        lost = title_text("?", 48, SUN).next_to(tag, m.DOWN, buff=0.25)
        src = cite("Li, AI Agents in Depth · ch. 2").move_to([0, -3.4, 0])

        self.say(
            "[As a context grows], [two different things can go wrong.]",
            m.AnimationGroup(self.new_header("05 · two failures", "Overflow and rot"), m.FadeIn(panels)),
            m.AnimationGroup(m.FadeIn(titles[0], shift=0.1 * m.UP), m.FadeIn(titles[1], shift=0.1 * m.UP)),
        )
        self.say(
            "[The first is overflow]: [the window is full], [and the next tool result simply doesn't fit.]",
            m.AnimationGroup(m.FadeIn(subs[0]), m.FadeIn(lf), m.FadeIn(lb), m.FadeIn(lm)),
            lm.fill_to(1.0),
            m.Succession(m.FadeIn(incoming, shift=0.6 * m.LEFT), m.AnimationGroup(m.FadeIn(err, shift=0.1 * m.UP), m.Circumscribe(incoming, color=ROSE, buff=0.06))),
        )
        self.say(
            "[The second is rot, and it's quieter]: [everything fits], [but the model can no longer find what it needs.]",
            m.FadeIn(subs[1]),
            m.AnimationGroup(m.FadeIn(rf), m.FadeIn(rb), m.FadeIn(rm)),
            m.Succession(
                m.AnimationGroup(m.FadeIn(tag, shift=0.1 * m.LEFT), m.Create(link)),
                m.AnimationGroup(needle.animate.set_fill(opacity=0.2), m.FadeIn(lost, scale=0.6)),
            ),
        )
        self.say(
            "[As the book puts it]: [overflow means it cannot fit any more], [rot means it fits, but cannot be found.]",
            m.FadeIn(src),
            m.Circumscribe(subs[0], color=ROSE, buff=0.08),
            m.Circumscribe(subs[1], color=SUN, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the pet store

    def pet_store(self):
        self.next_section("6 · The pet-store log", notes=f"Li, AI Agents in Depth, ch. 2 ('retrieval, not reasoning'): a pet-store inspection log of 100 cages, 90 black and 10 white cats. 'Which cat is in cage 37?' is one lookup, which attention handles well; 'how many black cats?' needs a scan and a running count, and the model may miscount. Writing 'Current statistics: 90 black cats, 10 white cats' into the context makes the answer retrievable. Cage placement seeded (random.Random(6)); cage {ASKED} holds a {CATS[ASKED - 1]} cat (asserted).")
        cell, step = 0.36, 0.41
        gx0, gy0 = -6.1, 1.15  # centre of cage 1
        cages = m.VGroup()
        for c in range(CAGES):
            r_, k_ = divmod(c, 10)
            box = card(cell, cell, EDGE, CARD, radius=0.06, stroke=1.2)
            g = m.VGroup(box, cat(CATS[c], 0.085).move_to(box).shift(0.02 * m.DOWN))
            cages.add(g.move_to([gx0 + k_ * step, gy0 - r_ * step, 0]))
        grid_w = cages.width
        log_cap = code(f"inspection log · {CAGES} cages", 20, MUTED).next_to(cages, m.DOWN, buff=0.25).align_to(cages, m.LEFT)
        legend = m.VGroup(cat("black", 0.1), text(f"{CATS.count('black')} black", 22, INK),
                          cat("white", 0.1), text(f"{CATS.count('white')} white", 22, INK)).arrange(m.RIGHT, buff=0.15)
        legend[2].shift(0.25 * m.RIGHT)
        legend[3].shift(0.25 * m.RIGHT)
        legend.next_to(log_cap, m.DOWN, buff=0.18).align_to(cages, m.LEFT)
        summary_words = code("Current statistics:\n90 black cats, 10 white cats", 20, INK)
        summary_box = card(max(grid_w, summary_words.width + 0.5), summary_words.height + 0.4, MINT, MINT_FILL, radius=0.12, stroke=2)
        summary = m.VGroup(summary_box, summary_words.move_to(summary_box)).next_to(cages, m.UP, buff=0.2).align_to(cages, m.LEFT)
        assert f"{CATS.count('black')} black cats, {CATS.count('white')} white cats" in summary_words.text

        target = cages[ASKED - 1]
        ring = m.RoundedRectangle(corner_radius=0.08, width=cell + 0.12, height=cell + 0.12, stroke_color=SUN, stroke_width=4).move_to(target)
        qx = 0.35
        q1 = message("user", f"Which cat is in cage {ASKED}?", width=5.9, size=24).move_to([0, 0, 0]).align_to([qx, 0, 0], m.LEFT).align_to([0, 2.3, 0], m.UP)
        a1 = m.VGroup(pill("one lookup · easy", MINT, MINT_FILL, 22), text(f"a {CATS[ASKED - 1]} cat", 24, INK)).arrange(m.RIGHT, buff=0.3)
        a1.next_to(q1, m.DOWN, buff=0.22).align_to(q1, m.LEFT)
        link = m.DashedLine(ring.get_right(), [qx - 0.1, q1.get_y(), 0], dash_length=0.08).set_stroke(SUN, 2)
        q2 = message("user", "How many black cats are there?", width=5.9, size=24).next_to(a1, m.DOWN, buff=0.45).align_to(q1, m.LEFT)
        a2 = m.VGroup(pill(f"count all {CAGES} · hard", ROSE, ROSE_FILL, 22), text("may miscount", 24, INK)).arrange(m.RIGHT, buff=0.3)
        a2.next_to(q2, m.DOWN, buff=0.22).align_to(q1, m.LEFT)
        a3 = m.VGroup(pill("a lookup again", MINT, MINT_FILL, 22), text("read the summary line", 24, INK)).arrange(m.RIGHT, buff=0.3)
        a3.next_to(a2, m.DOWN, buff=0.45).align_to(q1, m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 2").to_corner(m.DR, buff=0.4)

        def scan_at(r_):
            return m.Rectangle(width=grid_w + 0.12, height=step, stroke_color=SUN, stroke_width=2, fill_color=SUN, fill_opacity=0.16).move_to([cages.get_x(), gy0 - r_ * step, 0])

        scan = scan_at(0)
        sweep = m.Transform(scan, scan_at(9), rate_func=m.linear, run_time=2.4)
        sweep.introducer = True  # scan is introduced by the FadeIn just before, in the same line
        gone = m.FadeOut(scan, run_time=0.3)
        gone.introducer = True

        self.say(
            "[The book shows rot with a pet-store log]: [a hundred cages], [ninety black cats and ten white.]",
            m.AnimationGroup(self.new_header("06 · rot, up close", "The pet-store log"), m.FadeIn(src)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in cages], lag_ratio=0.012), m.FadeIn(log_cap)),
            m.FadeIn(legend, shift=0.1 * m.UP),
        )
        self.say(
            "[Ask which cat is in cage thirty-seven], [and the model finds it]: [one lookup.]",
            m.FadeIn(q1, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Create(ring), m.Create(link)),
            m.FadeIn(a1, shift=0.1 * m.UP),
        )
        self.say(
            "[Ask how many black cats there are], [and it must count all hundred entries], [so it may miscount.]",
            m.AnimationGroup(m.FadeIn(q2, shift=0.15 * m.LEFT), m.FadeOut(ring), m.FadeOut(link)),
            m.Succession(m.FadeIn(scan, run_time=0.3), sweep, gone),
            m.FadeIn(a2, shift=0.1 * m.UP),
        )
        self.say(
            "[Write one summary line into the context], [and the count becomes a lookup again.]",
            m.FadeIn(summary, shift=0.2 * m.DOWN),
            m.AnimationGroup(m.FadeIn(a3, shift=0.1 * m.UP), m.Circumscribe(summary, color=MINT, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ attention

    def attention(self):
        self.next_section("7 · Where things sit", notes=f"Attention is not spread evenly. Li, AI Agents in Depth, ch. 2 (Experiment 2-2): the first token 'often absorbs an abnormally high amount of attention weight', sometimes over 70% of the total (attention sink); position bias: put the most critical information at the beginning or end. Liu et al., 'Lost in the Middle', TACL 2024: models use information in the middle of long contexts worst. The curve's shape is illustrative. Otto's request sat last at call one; now it starts {100 * REQUEST_AT:.0f}% of the way in (computed). Reminders at the end: Episode 9.")
        px, panel_h = 2.75, 5.3
        panel = card(2.1, panel_h, EDGE, PANEL_FILL, radius=0.16, stroke=2).move_to([px, -0.45, 0])
        p_top, p_bot = panel.get_top()[1] - 0.15, panel.get_bottom()[1] + 0.15
        blocks = token_map(px, p_top, 1.75, p_top - p_bot)
        title = code("Otto's context", 20, MUTED).next_to(panel, m.UP, buff=0.12)
        x0, reach = panel.get_left()[0] - 0.18, 1.9
        ps = np.linspace(0, 1, 61)
        curve = m.VMobject().set_points_smoothly([[x0 - reach * attention_weight(p), p_top - p * (p_top - p_bot), 0] for p in ps])
        curve.set_stroke(SUN, 4)
        axis = m.DashedLine([x0, p_top, 0], [x0, p_bot, 0], dash_length=0.08).set_stroke(FAINT, 2)
        att_cap = code("← attention", 20, MUTED).move_to([x0 - 1.0, title.get_y(), 0])
        shape_note = text("shape illustrative", 20, FAINT).next_to(axis, m.DOWN, buff=0.15).align_to(axis, m.RIGHT)
        n = 36
        slice_h = (p_top - p_bot) / n
        shade = m.VGroup(*[
            m.Rectangle(width=1.8, height=slice_h + 0.004, stroke_width=0, fill_color=BG,
                        fill_opacity=0.72 * (1 - attention_weight((i + 0.5) / n)))
            .move_to([px, p_top - (i + 0.5) * slice_h, 0])
            for i in range(n)
        ]).set_z_index(3)
        spike = m.Line([x0, p_top, 0], [x0 - reach * 1.35, p_top, 0]).set_stroke(SUN, 7)
        spike_dot = m.Dot([x0 - reach * 1.35, p_top, 0], radius=0.08, color=SUN)
        mid = m.Line([x0 - 0.06, p_top - 0.25 * (p_top - p_bot), 0], [x0 - 0.06, p_top - 0.8 * (p_top - p_bot), 0]).set_stroke(ROSE, 6)

        def item(name, color, fill, desc, source, y):
            p = pill(name, color, fill, 22)
            d = text(desc, 24, INK)
            g = m.VGroup(p, d).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
            if source:
                g.add(cite(source).next_to(d, m.DOWN, buff=0.1).align_to(d, m.LEFT))
            return g.move_to([0, y, 0], aligned_edge=m.UP).align_to([LEFT_X, 0, 0], m.LEFT)

        sink = item("attention sink", SUN, SUN_FILL, "the first token can draw\nover 70% of the attention", "Li, AI Agents in Depth · ch. 2", 2.3)
        middle = item("lost in the middle", ROSE, ROSE_FILL, "the middle of a long context\nis used worst", "Liu et al., 2024 · TACL", 0.25)
        place = item("first or last", MINT, MINT_FILL, "rules at the top,\nreminders at the end", None, -1.8)
        ep9 = pill("Episode 9 · Plans You Can See", SUN, SUN_FILL, 20).next_to(place, m.DOWN, buff=0.25).align_to(place, m.LEFT)

        req_i = 2
        req_y = blocks[req_i].get_y()
        rx = panel.get_right()[0] + 0.12

        def marker(y):
            tip = m.Triangle(fill_color=AZURE, fill_opacity=1, stroke_width=0).scale(0.11).rotate(m.PI / 2).move_to([rx + 0.1, y, 0])
            words = text("the request", 22, AZURE, weight=m.SEMIBOLD).next_to(tip, m.RIGHT, buff=0.12)
            return m.VGroup(tip, words)

        start_tag = code("call 1: last", 20, MUTED)
        req_then = marker(p_bot + 0.15)
        start_tag.next_to(req_then[1], m.DOWN, buff=0.1).align_to(req_then[1], m.LEFT)
        req_then.add(start_tag)
        req_now = marker(req_y)
        now_tag = code("call 10: middle", 20, MUTED).next_to(req_now[1], m.DOWN, buff=0.1).align_to(req_now[1], m.LEFT)
        req_now.add(now_tag)
        moved = m.Transform(req_then, req_now, path_arc=0.0, run_time=1.4)
        moved.introducer = True  # req_then is introduced earlier in the same line
        pre_lab = text("system + tools", 22, ORCHID, weight=m.SEMIBOLD).next_to(m.VGroup(blocks[0], blocks[1]), m.RIGHT, buff=0.32)
        reminder = pill("+ reminder", AZURE, AZURE_FILL, 20).move_to([rx + 0.55, p_bot + 0.2, 0], aligned_edge=m.LEFT)
        reminder.add(arrow(reminder.get_left(), [rx - 0.1, p_bot + 0.2, 0], AZURE, buff=0.06))
        self.att = (panel, blocks, title)

        self.say(
            "[Where a fact sits matters too.] [The model reads every token], [but its attention isn't spread evenly.]",
            m.AnimationGroup(self.new_header("07 · position", "First and last, not the middle"), m.FadeIn(panel), m.FadeIn(blocks), m.FadeIn(title)),
            m.AnimationGroup(m.FadeIn(axis), m.FadeIn(att_cap), m.Create(curve)),
            m.AnimationGroup(m.FadeIn(shade), m.FadeIn(shape_note)),
        )
        self.say(
            "[In the book's experiments], [the very first token can draw over seventy percent of the attention]: [an attention sink.]",
            m.FadeIn(sink[2]),
            m.AnimationGroup(m.FadeIn(sink[1], shift=0.1 * m.UP), m.Create(spike), m.FadeIn(spike_dot, scale=0.5)),
            m.FadeIn(sink[0], shift=0.1 * m.UP),
        )
        self.say(
            "[And facts in the middle of a long context are used worst]: [researchers call it lost in the middle.]",
            m.AnimationGroup(m.FadeIn(middle[1], shift=0.1 * m.UP), m.Create(mid)),
            m.AnimationGroup(m.FadeIn(middle[0], shift=0.1 * m.UP), m.FadeIn(middle[2])),
        )
        self.say(
            "[When the session began, Otto's request was the last thing in its context.] [Ten turns later, it sits in the middle.]",
            m.FadeIn(req_then, shift=0.15 * m.LEFT),
            moved,
        )
        self.say(
            "[So put critical things first or last]: [rules at the top, in the system prompt], [and reminders at the end], [as episode nine will show.]",
            m.FadeIn(place[:2], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(pre_lab, shift=0.1 * m.LEFT), m.Circumscribe(m.VGroup(blocks[0], blocks[1]), color=ORCHID, buff=0.05)),
            m.FadeIn(reminder, shift=0.15 * m.UP),
            m.FadeIn(ep9, shift=0.1 * m.UP),
        )
        self.att_extra = m.VGroup(curve, axis, att_cap, shape_note, shade, spike, spike_dot, mid, sink, middle, place, ep9, req_then, pre_lab, reminder)
        self.wait(0.3)

    # ------------------------------------------------------------------ the rest of the module

    def ahead(self):
        self.next_section("8 · The rest of Module 2", notes="The rest of Module 2 works on this picture: Episode 7 caches the static prefix (KV cache, prompt caching), Episode 8 assembles the system prompt from parts, Episode 11 compacts the trajectory when the window fills. Episodes 9 (plans and reminders) and 10 (skills) also decide what goes in.")
        panel, blocks, title = self.att
        pre = m.VGroup(blocks[0], blocks[1])
        traj = m.VGroup(*blocks[2:])
        ends = panel.get_left()[0] - 0.12

        def plan(ep, words, y):
            box = card(5.1, 0.72, EDGE, CARD, radius=0.16, stroke=2)
            body = m.VGroup(code(f"ep {ep}", 20, SUN, weight=m.BOLD), text(words, 24, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.3)
            body.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            g = m.VGroup(box, body).move_to([-1.85, y, 0])
            return g, arrow(g.get_right(), [ends, y, 0], MUTED, buff=0.1)

        c7, a7 = plan(7, "cache the prefix", pre.get_y() + 0.42)
        c8, a8 = plan(8, "assemble the prefix", pre.get_y() - 0.42)
        c11, a11 = plan(11, "compact the trajectory", traj.get_y())

        self.say(
            "[The rest of this module works on this picture]: [caching the prefix], [assembling it from parts], [and compacting the trajectory as it fills.]",
            m.AnimationGroup(self.new_header("08 · next in module 2", "What comes next"), m.FadeOut(self.att_extra)),
            m.AnimationGroup(m.FadeIn(c7, shift=0.15 * m.RIGHT), m.GrowArrow(a7)),
            m.AnimationGroup(m.FadeIn(c8, shift=0.15 * m.RIGHT), m.GrowArrow(a8)),
            m.AnimationGroup(m.FadeIn(c11, shift=0.15 * m.RIGHT), m.GrowArrow(a11), m.Circumscribe(traj, color=MINT, buff=0.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Fill (the static prefix and the trajectory, rebuilt for every call), fit (a budget of tokens: overflow), find (rot; first or last, not the middle).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Fill", ORCHID), ("Fit", SUN), ("Find", MINT))])
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("prefix + trajectory", "a budget of tokens", "first or last, not the middle")])
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
        tagline = text("An agent knows only what its harness puts in the context.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [fill], [fit], [find.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[An agent knows only what its harness puts in the context.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
