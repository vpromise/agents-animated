"""Agents, Animated · Episode 7 · KV Cache & Prompt Caching (Module 2 · Context Engineering)

Why agents keep their context append-only. Running example: one "helpful" line, "Current time: …",
added to Otto's system prompt, which turns every call into a cache miss and nearly doubles the bill.

Sections: hook (Episode 2's run: 25 messages sent for a list of 10, 16 of them repeats) · prefill and
decode (keys and values computed for the whole prompt in one pass, then one token at a time; time to
first token) · the KV cache (computed once and kept; 500,500 vs 1,000 key/value computations for 1,000
tokens) · prompt caching (call two re-sends call one's prefix; the provider reuses its keys and values;
cache reads about a tenth of the input price) · the prefix rule (the first differing position decides:
edits in the middle, at the end, near the start) · the current-time line (B7's 1,000 / 2,000 / 3,000
example: billed 3,300 with a stable prefix vs 6,000 with a clock: x1.8; one team's reported incident;
the fix: the time goes into the newest message) · B's three rules · cache breakers and 2^N prefixes ·
Claude Code's static / boundary / dynamic split · Table 7-4 (savings do not add) · recap.

Sources (facts dated October 2026):
- Li, AI Agents in Depth, ch. 2 (bojieli.github.io/ai-agent-book/book-en/chapter2/): KV cache and prompt
  cache ("if the token sequence first differs at some position, the KV states for that token and
  everything after it must be recomputed"); the three rules (never change the system prompt or tools once
  finalized, "even adding a single space"; "Always append dynamic information to the end"; use the
  standard API format, never hand-concatenated text); cache reads "about one-tenth the price with
  Anthropic, DeepSeek, and GPT-5" (no cache-WRITE prices are claimed here); the reported incident (a
  customer-service agent, 100,000 conversations a day, a timestamp in the system prompt: "TTFT ... from
  0.5 seconds to 3-5 seconds, and the monthly inference bill nearly doubled"); 2^N cache keys ("2x2x2 = 8");
  Experiment 2-3's harmful patterns (dynamic system prompt; tools reordered by frequency: "a fixed order
  has almost no effect on tool-selection accuracy"; sliding windows: agents "often fell into loops";
  text-flattened history: "one of the most harmful patterns"); Claude Code's prompt "split by a cache
  boundary marker". Q/K/V as search request / label / content.
- Li, ch. 7 (cost analysis): "Round 1 sends 1000 tokens, Round 2 sends 2000 ... totaling 6000 instead
  of 3000"; Table 7-4 (eight-turn refund workflow, gpt-4o-mini): costs $0.003776 / 0.002707 / 0.003115 /
  0.002643 = savings 28.3% / 17.5% / 30.0%, "not additive".
- learn-claude-code s10 (learn.shareai.run/en/s10/, the site's reading of Claude Code's source, not
  verified by us): SYSTEM_PROMPT_DYNAMIC_BOUNDARY between static and dynamic parts, static sections in
  one global cache block; CLAUDE.md and currentDate prepended as a <system-reminder> user message.

Illustrative (for teaching, said "for example" or shown as schematic): the tokenisation of Otto's task,
the two decode tokens, the timeline proportions, the strips' block counts, the 1,000-token blocks and the
"billed as full-price tokens" unit (cache reads x0.1, writes counted at the normal price for simplicity),
the clock times, the three yes/no switches. Asserts below tie every quoted number to the data.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the numbers

# Episode 2's run: call k sends the 2k - 1 messages so far; the list ends with ten.
SENT = [2 * k - 1 for k in range(1, 6)]
RESENT = [0] + SENT[:-1]  # what each call had already sent before
EP2_COLORS = [AZURE] + [SUN, MINT] * 4 + [CORAL]  # task, (tool_use, tool_result) x 4, the answer
assert SENT == [1, 3, 5, 7, 9] and sum(SENT) == 25 and len(EP2_COLORS) == 10  # "twenty-five ... ten"
assert sum(RESENT) == 16  # "sixteen of those twenty-five were repeats"

# Prefill and decode (tokenisation illustrative)
PROMPT_TOKENS = ["The", "checkout", "test", "fails.", "Fix", "it."]
REPLY_TOKENS = ["Run", "pytest"]
TOKENS = PROMPT_TOKENS + REPLY_TOKENS
P = len(PROMPT_TOKENS)  # the first reply token's index
N_TOK = 1000
NO_CACHE = sum(range(1, N_TOK + 1))  # step n recomputes the keys and values of n tokens
WITH_CACHE = N_TOK  # each token's keys and values once
assert NO_CACHE == N_TOK * (N_TOK + 1) // 2 == 500_500 and round(NO_CACHE, -5) == 500_000  # "half a million"


def first_difference(a, b):
    """The first position where two token sequences differ (the cache keeps everything before it)."""
    return next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b)))


# Prompt caching: Otto's prompt as blocks of tokens (schematic)
SEG_1 = [("system", 4), ("tools", 6), ("user", 3)]
SEG_2 = SEG_1 + [("assistant", 4), ("tool_result", 4)]
CALL_1 = [f"{r}{i}" for i, r in enumerate(r for r, n in SEG_1 for _ in range(n))]
CALL_2 = [f"{r}{i}" for i, r in enumerate(r for r, n in SEG_2 for _ in range(n))]
N1, N2 = len(CALL_1), len(CALL_2)
REUSED = first_difference(CALL_1, CALL_2)
assert (N1, N2, REUSED) == (13, 21, 13)  # call two reuses all of call one, computes 8 new


def edited(seq, i):
    out = list(seq)
    out[i] += "*"
    return out


EDITS = [13, 20, 1]  # middle, very end, near the start
RULE = [(first_difference(CALL_2, edited(CALL_2, i)), N2 - first_difference(CALL_2, edited(CALL_2, i))) for i in EDITS]
assert RULE == [(13, 8), (20, 1), (1, 20)]  # "lose almost nothing" / "the whole prompt goes cold"

# The current-time line (B7's example: each call adds 1,000 tokens; cache reads at a tenth)
BLOCK, READ = 1000, 0.1
TIMES = ["09:41:07", "09:41:08", "09:41:09"]


def prompts(clock):
    """Three calls as lists of 1,000-token blocks: the system block (with or without the time), then history."""
    return [[f"system@{TIMES[k]}" if clock else "system"] + [f"turn{j}" for j in range(1, k + 1)] for k in range(3)]


def billed(calls):
    """Each call, in full-price tokens: the prefix shared with the previous call at READ, the rest at 1."""
    out, previous = [], []
    for call in calls:
        hit = first_difference(previous, call)
        out.append(round(BLOCK * (READ * hit + (len(call) - hit))))
        previous = call
    return out


SENT_T = [BLOCK * len(c) for c in prompts(False)]
WARM, COLD = billed(prompts(False)), billed(prompts(True))
RATIO = sum(COLD) / sum(WARM)
assert SENT_T == [1000, 2000, 3000] and sum(SENT_T) == 6000 and 3 * BLOCK == 3000  # "six thousand sent, for three thousand new"
assert WARM == [1000, 1100, 1200] and sum(WARM) == 3300  # "three thousand three hundred"
assert COLD == [1000, 2000, 3000] and sum(COLD) == 6000  # "all six thousand at full price"
assert f"{RATIO:.1f}" == "1.8" and 1.75 < RATIO < 2  # "nearly double"

# Switches before the cache boundary
SWITCHES = 3
PREFIXES = [format(i, f"0{SWITCHES}b") for i in range(2**SWITCHES)]
assert len(PREFIXES) == 8  # "three yes-or-no options ... eight different prefixes"

# Table 7-4 (Li, ch. 7): cost of an eight-turn refund workflow, gpt-4o-mini
COST = {"baseline": 0.003776, "stable": 0.002707, "compress": 0.003115, "both": 0.002643}
SAVED = {k: round(100 * (1 - COST[k] / COST["baseline"]), 1) for k in ("stable", "compress", "both")}
ADDED = round(SAVED["stable"] + SAVED["compress"], 1)
assert SAVED == {"stable": 28.3, "compress": 17.5, "both": 30.0}  # "twenty-eight point three", "seventeen point five", "thirty"
assert ADDED == 45.8 and round(ADDED) == 46  # "not forty-six"

# ---------------------------------------------------------------------------- look

PREFIX = "ep07"
ROLE_C = {"system": ORCHID, "tools": MUTED, "user": AZURE, "assistant": CORAL, "tool_result": MINT}
STATES = {"cold": (EDGE, CARD), "computed": (SUN, SUN_FILL), "cached": (MINT, MINT_FILL), "recomputed": (ROSE, ROSE_FILL)}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def cite(s):
    return text(s, 20, MUTED)


def baseline(words):
    """The baseline of a Text: the median bottom of its flat-bottomed glyphs."""
    chars = [c for c in words.text if not c.isspace()]
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, y):
    """Put a Text's baseline at y."""
    return words.shift((y - baseline(words)) * m.UP)


def cell(state="cold", w=0.42, h=0.52):
    color, fill = STATES[state]
    return m.RoundedRectangle(corner_radius=0.08, width=w, height=h, stroke_color=color, stroke_width=2.5,
                              fill_color=fill, fill_opacity=1)


def paint(mobs, state, lag=0.06, run_time=0.9):
    """Cells (or cards) changing cache state, left to right."""
    color, fill = STATES[state]
    return m.LaggedStart(*[x.animate.set_fill(fill, opacity=1).set_stroke(color) for x in mobs], lag_ratio=lag, run_time=run_time)


def stroke_to(mobs, color, lag=0.1, run_time=0.8):
    return m.LaggedStart(*[x.animate.set_stroke(color) for x in mobs], lag_ratio=lag, run_time=run_time)


class Strip:
    """A prompt as a row of token blocks, labelled by segment (system, tools, user, …)."""

    def __init__(self, segments, y, x0=-5.35, pitch=0.52, w=0.42, h=0.52, state="cold", labels=True):
        self.roles = [r for r, n in segments for _ in range(n)]
        self.cells = m.VGroup(*[cell(state, w, h).move_to([x0 + i * pitch, y, 0]) for i in range(len(self.roles))])
        self.labels = m.VGroup()
        self.bounds = []
        i = 0
        for role, n in segments:
            self.bounds.append((i, i + n))
            if labels:
                lab = code(role, 22, ROLE_C[role], weight=m.BOLD)
                lab.set_x(self.cells[i:i + n].get_x())
                sit(lab, y + h / 2 + 0.16)
                self.labels.add(lab)
            i += n

    def span(self, a, b=None):
        return m.VGroup(*self.cells[a:b])

    def under(self, a, b, words, gap=0.22):
        """A caption centred under cells a..b."""
        seg = self.span(a, b)
        return words.move_to([seg.get_x(), seg.get_bottom()[1] - gap - words.height / 2, 0])

    def glow(self, a, b, color=MINT):
        seg = self.span(a, b)
        g = m.RoundedRectangle(corner_radius=0.16, width=seg.width + 0.26, height=seg.height + 0.22, stroke_width=0,
                               fill_color=color, fill_opacity=0.16)
        return g.move_to(seg).set_z_index(-1)


def row_label(s, strip):
    lab = code(s, 20, MUTED)
    first = strip.cells[0]
    lab.move_to(first).align_to([first.get_left()[0] - 0.3, 0, 0], m.RIGHT)
    return sit(lab, first.get_y() - 0.07)


def msg_card(label, color, body, width, size=20, mono=False, body_color=INK, stroke=EDGE):
    """A message: role label in its colour above the content, a coloured stripe on the left."""
    tag = code(label, 22, color, weight=m.BOLD)
    words = code(body, size, body_color) if mono else text(body, size, body_color)
    inner = m.VGroup(tag, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, stroke, CARD, radius=0.12, stroke=2)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def msg_row(role, body, width=6.2, mono=True, height=0.5):
    """One message on one line (Episode 2's style): stripe, role, content."""
    color = ROLE_C[role]
    box = card(width, height, EDGE, CARD, radius=0.1, stroke=2)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=height - 0.16, stroke_width=0,
                                fill_color=color if role != "tool_result" else MINT, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    tag = code("user" if role == "tool_result" else role, 22, AZURE if role == "tool_result" else color, weight=m.BOLD)
    words = code(body, 20, MINT if role == "tool_result" else INK) if mono else text(body, 20, INK)
    tag.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(1.95 * m.RIGHT)
    base = box.get_y() - 0.09
    sit(tag, base)
    sit(words, base)
    return m.VGroup(box, stripe, tag, words)


def otto_system(width=5.6, size=22, t=TIMES[0]):
    """Otto's system prompt, with room for the clock line (returned separately, to animate)."""
    tag = code("system", 22, ORCHID, weight=m.BOLD)
    body = text("You are Otto, a coding agent\nworking in the tinyshop repo.", size, INK)
    clock = code(f"Current time: {t}", size, ROSE)
    inner = m.VGroup(tag, body, clock).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
    box = card(width, inner.height + 0.42, EDGE, CARD, radius=0.14, stroke=2)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.18, stroke_width=0,
                                fill_color=ORCHID, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.34 * m.RIGHT)
    return m.VGroup(box, stripe, tag, body), clock


def clock_at(clock, t):
    return code(f"Current time: {t}", 22, ROSE).move_to(clock)


def bracket(x0, x1, y, color, up=True):
    """A flat bracket with end ticks (for spans under a row)."""
    d = 0.14 if up else -0.14
    return m.VGroup(
        m.Line([x0, y, 0], [x1, y, 0]), m.Line([x0, y, 0], [x0, y + d, 0]), m.Line([x1, y, 0], [x1, y + d, 0])
    ).set_stroke(color, 2.5)


def kv_cell(color, fill, w=0.8, h=0.34):
    return m.RoundedRectangle(corner_radius=0.07, width=w, height=h, stroke_color=color, stroke_width=2.5,
                              fill_color=fill, fill_opacity=1)


def stat(label, value, sub, color):
    """A label in its colour, a big number, a small line under it."""
    a = text(label, 24, color, weight=m.BOLD)
    b = title_text(value, 50, INK)
    c = code(sub, 20, MUTED)
    return m.VGroup(a, b, c).arrange(m.DOWN, buff=0.16)


class Slot:
    """One place on screen whose content is swapped (from Episode 2)."""

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


# ---------------------------------------------------------------------------- the episode


class Ep07KVCachePromptCaching(Episode):
    number = 7

    def lesson(self):
        self.hook()
        self.prefill_decode()
        self.kv_cache()
        self.prompt_caching()
        self.prefix_rule()
        self.current_time()
        self.rules()
        self.breakers()
        self.claude_code()
        self.measured()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 2's run: calls sent {SENT} messages = {sum(SENT)} for a list of 10; {sum(RESENT)} of them had been sent before (computed). A cache makes the repeats cheap — until a timestamp in the system prompt breaks it. Prompt caching keeps that affordable (promised in Episode 2).")
        rows = m.VGroup()
        for k, n in enumerate(SENT, start=1):
            y = 1.75 - 0.6 * (k - 1)
            label = code(f"call {k}", 20, MUTED).move_to([-6.75, y, 0], aligned_edge=m.LEFT)
            blocks = m.VGroup(*[
                m.RoundedRectangle(corner_radius=0.05, width=0.3, height=0.38, stroke_width=0, fill_color=c, fill_opacity=1)
                for c in EP2_COLORS[:n]
            ]).arrange(m.RIGHT, buff=0.07).move_to([-5.45, y, 0], aligned_edge=m.LEFT)
            rows.add(m.VGroup(label, blocks))
        outlines = m.VGroup(*[
            m.SurroundingRectangle(r[1][:p], buff=0.07, corner_radius=0.08).set_stroke(SUN, 2.5)
            for r, p in zip(rows, RESENT) if p
        ])
        sent = text(f"{sum(SENT)} messages sent, for a list of {len(EP2_COLORS)}", 26, INK).move_to([-3.75, -1.45, 0])
        repeats = text(f"{sum(RESENT)} of them repeats", 26, SUN, weight=m.BOLD).next_to(sent, m.DOWN, buff=0.3)
        src = cite("Episode 2 · The Agent Loop").next_to(repeats, m.DOWN, buff=0.3)

        sysc, clock = otto_system()
        m.VGroup(sysc, clock).move_to([3.55, 0.95, 0])
        ticks = [clock_at(clock, t) for t in TIMES[1:]]
        cached = text("read from cache", 22, MINT, weight=m.BOLD).move_to([rows[3][1].get_right()[0] + 0.45, outlines.get_y(), 0], aligned_edge=m.LEFT)
        append = pill("append-only", MINT, MINT_FILL, 26).move_to([3.55, -1.25, 0])

        self.say(
            "[In episode two, Otto's run sent the model twenty-five messages], [for a list of only ten.]",
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.25),
            m.FadeIn(sent, shift=0.1 * m.UP),
        )
        self.say(
            "[Every call re-sends the whole history]: [sixteen of those twenty-five were repeats.]",
            m.LaggedStart(*[m.Create(o) for o in outlines], lag_ratio=0.25),
            m.AnimationGroup(m.FadeIn(repeats, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[A cache makes those repeats cheap], [until one innocent line in the system prompt breaks it.]",
            m.AnimationGroup(stroke_to(outlines, MINT), m.FadeIn(cached, shift=0.1 * m.LEFT)),
            m.Succession(
                m.AnimationGroup(m.FadeIn(sysc, shift=0.2 * m.UP), m.FadeIn(clock, shift=0.2 * m.UP), run_time=0.8),
                m.Transform(clock, ticks[0], run_time=0.4),
            ),
        )
        self.say(
            "[Today: how that cache works], [and why agents keep their context append-only.]",
            m.AnimationGroup(m.Transform(clock, ticks[1], run_time=0.4), m.Circumscribe(clock, color=ROSE, buff=0.08)),
            m.FadeIn(append, scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ prefill and decode

    def build_kv(self):
        st = NS()
        xs = np.linspace(-5.3, 6.0, len(TOKENS))
        st.xs = xs
        CHIP_Y, V_Y, K_Y, Q_Y = 1.85, 1.0, 0.42, -0.35
        st.chips = m.VGroup(*[
            chip(w, AZURE if i < P else CORAL, AZURE_FILL if i < P else CORAL_FILL, width=1.45, height=0.62, size=22).move_to([x, CHIP_Y, 0])
            for i, (w, x) in enumerate(zip(TOKENS, xs))
        ])
        st.v = m.VGroup(*[kv_cell(SUN, SUN_FILL, w=0.9).move_to([x, V_Y, 0]) for x in xs])
        st.k = m.VGroup(*[kv_cell(SUN, SUN_FILL, w=0.9).move_to([x, K_Y, 0]) for x in xs])
        st.q = {i: kv_cell(CORAL, CORAL_FILL, w=0.9).move_to([xs[i], Q_Y, 0]) for i in (P, P + 1)}
        st.v_tag = text("V", 24, SUN, weight=m.BOLD).move_to([-6.6, V_Y, 0])
        st.k_tag = text("K", 24, SUN, weight=m.BOLD).move_to([-6.6, K_Y, 0])
        st.q_tag = text("query", 22, CORAL, weight=m.BOLD).next_to(st.q[P], m.DOWN, buff=0.1)
        st.lines8 = m.VGroup(*[m.Line(st.q[P].get_top(), st.k[j].get_bottom()).set_stroke(CORAL, 2, opacity=0.75) for j in range(P)])
        st.lines9 = m.VGroup(*[m.Line(st.q[P + 1].get_top(), st.k[j].get_bottom()).set_stroke(CORAL, 2, opacity=0.75) for j in range(P + 1)])
        # phase brackets
        h = 0.65
        by = -1.12
        st.pre_br = bracket(xs[0] - h, xs[P - 1] + h, by, SUN)
        st.dec_br = bracket(xs[P] - h, xs[P + 1] + h, by, CORAL)
        pre = m.VGroup(text("prefill", 26, SUN, weight=m.BOLD), text("the whole prompt, in one pass", 22, MUTED)).arrange(m.RIGHT, buff=0.3)
        st.pre_lab = pre.move_to([st.pre_br.get_x(), by - 0.38, 0])
        dec = m.VGroup(text("decode", 26, CORAL, weight=m.BOLD), text("one token at a time", 20, MUTED)).arrange(m.DOWN, buff=0.1)
        st.dec_lab = dec.next_to(st.dec_br, m.DOWN, buff=0.18)
        # timeline
        ty = -2.65
        st.axis = m.Arrow([-6.3, ty, 0], [6.6, ty, 0], buff=0, stroke_width=2.5, color=FAINT, max_tip_length_to_length_ratio=0.02)
        st.pre_bar = m.Rectangle(width=8.6, height=0.3, stroke_width=0, fill_color=SUN, fill_opacity=0.85).move_to([-6.3 + 4.3, ty + 0.15 + 0.04, 0])
        st.dec_bars = m.VGroup(*[
            m.Rectangle(width=0.32, height=0.3, stroke_width=0, fill_color=CORAL, fill_opacity=0.9).move_to([2.55 + 0.45 * i, ty + 0.19, 0])
            for i in range(2)
        ])
        st.ttft = bracket(-6.3, 2.39, ty - 0.22, INK, up=True)
        st.ttft_lab = text("time to first token", 22, INK, weight=m.SEMIBOLD).next_to(st.ttft, m.DOWN, buff=0.12)
        st.time_lab = code("time", 20, MUTED).next_to(st.axis.get_end(), m.UP, buff=0.3).align_to(st.axis.get_end() + 0.3 * m.LEFT, m.RIGHT)
        st.example = text("schematic", 20, MUTED).move_to([4.9, ty - 0.55, 0])
        return st

    def prefill_decode(self):
        self.next_section("1 · Prefill and decode", notes="Prefill: the model reads the whole prompt in one pass, computing a key and a value for every token (Li, ch. 2: Q/K/V as search request / label / content). Decode: one token at a time; each new token's query meets every key and reads the values. For a long prompt the time to first token is mostly prefill. Tokenisation and timeline proportions are illustrative.")
        st = self.st = self.build_kv()
        prompt = st.chips[:P]
        self.say(
            "[When a request arrives], [the model works in two phases.]",
            m.AnimationGroup(self.new_header("01 · prefill & decode", "Read it all, then write"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in prompt], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(st.pre_br), m.FadeIn(st.pre_lab, shift=0.1 * m.UP), m.FadeIn(st.dec_br), m.FadeIn(st.dec_lab, shift=0.1 * m.UP)),
        )
        self.say(
            "[First, prefill]: [it reads the whole prompt in one pass], [computing a key and a value for every token.]",
            m.Circumscribe(st.pre_lab[0], color=SUN, buff=0.08),
            m.Circumscribe(prompt, color=SUN, buff=0.1),
            m.AnimationGroup(m.FadeIn(st.v_tag), m.FadeIn(st.k_tag),
                             *[m.FadeIn(c, shift=0.15 * m.DOWN) for c in (*st.v[:P], *st.k[:P])]),
        )
        self.say(
            "[A key is like a label to search by], [and a value is the content behind it.]",
            m.Circumscribe(m.VGroup(st.k_tag, st.k[:P]), color=SUN, buff=0.08),
            m.Circumscribe(m.VGroup(st.v_tag, st.v[:P]), color=SUN, buff=0.08),
        )
        self.say(
            "[Then, decode]: [the reply comes out one token at a time.]",
            m.Circumscribe(st.dec_lab[0], color=CORAL, buff=0.08),
            m.AnimationGroup(m.FadeIn(st.chips[P], shift=0.2 * m.LEFT), m.FadeIn(st.q[P], shift=0.1 * m.UP), m.FadeIn(st.q_tag),
                             m.FadeIn(st.k[P], shift=0.1 * m.DOWN), m.FadeIn(st.v[P], shift=0.1 * m.DOWN)),
        )
        self.say(
            "[Each new token sends a query to every key], [and reads the matching values.]",
            m.LaggedStart(*[m.Create(l) for l in reversed(st.lines8)], lag_ratio=0.1),
            m.Indicate(st.v[:P], color=CORAL, scale_factor=1.08),
        )
        self.say(
            "[For a long prompt], [the wait for the first word, the time to first token], [is mostly prefill.]",
            m.AnimationGroup(m.Create(st.axis), m.FadeIn(st.time_lab), m.GrowFromEdge(st.pre_bar, m.LEFT), m.FadeIn(st.example)),
            m.AnimationGroup(m.FadeIn(st.dec_bars[0], shift=0.1 * m.UP), m.FadeIn(st.ttft), m.FadeIn(st.ttft_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(st.pre_bar, color=SUN, buff=0.06), m.FadeIn(st.dec_bars[1], shift=0.1 * m.UP)),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ KV cache

    def kv_cache(self):
        st = self.st
        self.next_section("2 · The KV cache", notes=f"Without a cache, step n would recompute the keys and values of all n tokens: 1 + 2 + … + {N_TOK:,} = {NO_CACHE:,} for {N_TOK:,} tokens (∝ N²); with the KV cache each token's keys and values are computed once: {WITH_CACHE:,} (Li, ch. 2: 'without a cache, K/V are recomputed every step'). Attention itself still reads the whole cache each step.")
        old = m.VGroup(st.k[:P + 1], st.v[:P + 1])
        nocache = pill("no cache: every step recomputes them all", ROSE, ROSE_FILL, 22).move_to([0, -1.3, 0])
        box = m.SurroundingRectangle(m.VGroup(st.v[:P + 1], st.k[:P + 1]), buff=0.13, corner_radius=0.14).set_stroke(MINT, 2.5)
        box_big = m.SurroundingRectangle(m.VGroup(st.v, st.k), buff=0.13, corner_radius=0.14).set_stroke(MINT, 2.5)
        tag = pill("KV cache", MINT, MINT_FILL, 22).move_to([-4.9, -0.35, 0])
        once = text("computed once, then kept", 22, MINT).next_to(tag, m.RIGHT, buff=0.3)
        head = text(f"keys and values computed for {N_TOK:,} tokens", 24, MUTED).move_to([0, -1.25, 0])
        left = stat("no cache", f"{NO_CACHE:,}", f"1 + 2 + … + {N_TOK:,}", ROSE).move_to([-3.3, -2.55, 0])
        right = stat("KV cache", f"{WITH_CACHE:,}", "one per token", MINT).move_to([3.3, -2.55, 0])
        grows = m.VGroup(text("grows like N²", 22, ROSE), text("grows like N", 22, MINT))
        grows[0].next_to(left[1], m.RIGHT, buff=0.35)
        grows[1].next_to(right[1], m.RIGHT, buff=0.35)

        timeline = m.VGroup(st.axis, st.time_lab, st.pre_bar, st.dec_bars, st.ttft, st.ttft_lab, st.example)
        self.say(
            "[Without a cache], [every new token would recompute the keys and values of every token before it.]",
            m.AnimationGroup(self.new_header("02 · KV cache", "Compute once, keep it"),
                             m.FadeOut(m.VGroup(timeline, st.pre_br, st.pre_lab, st.dec_br, st.dec_lab, st.lines8), shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.Indicate(old, color=ROSE, scale_factor=1.06), m.FadeIn(nocache, shift=0.15 * m.UP)),
        )
        self.say(
            "[The KV cache keeps them instead]: [each token's keys and values are computed once, and kept.]",
            m.AnimationGroup(m.FadeOut(nocache), paint(list(st.k[:P + 1]) + list(st.v[:P + 1]), "cached", lag=0.03), m.Create(box), m.FadeIn(tag, shift=0.1 * m.UP)),
            m.FadeIn(once, shift=0.1 * m.LEFT),
        )
        self.say(
            "[The next token adds only its own], [and reads the rest from the cache.]",
            m.AnimationGroup(m.FadeIn(st.chips[P + 1], shift=0.2 * m.LEFT), m.FadeIn(st.q[P + 1], shift=0.1 * m.UP),
                             m.FadeIn(st.k[P + 1], shift=0.1 * m.DOWN), m.FadeIn(st.v[P + 1], shift=0.1 * m.DOWN), m.Transform(box, box_big)),
            m.LaggedStart(*[m.Create(l) for l in reversed(st.lines9)], lag_ratio=0.08),
        )
        self.say(
            "[Without it, the work grows with the square of the length]: [for a thousand tokens], [half a million computations instead of a thousand.]",
            m.AnimationGroup(m.FadeIn(head, shift=0.1 * m.UP), m.FadeIn(left[0]), m.FadeIn(right[0])),
            m.AnimationGroup(m.FadeIn(left[2]), m.FadeIn(right[2])),
            m.AnimationGroup(m.FadeIn(left[1], scale=0.85), m.FadeIn(right[1], scale=0.85), m.FadeIn(grows)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ prompt caching

    def prompt_caching(self):
        self.next_section("3 · Prompt caching", notes=f"Across requests: call two re-sends call one's {N1} blocks plus {N2 - N1} new ones; the provider keeps the prefix's keys and values and reuses them, so only the tail is computed (Li, ch. 2: 'Prompt Cache ... reduces redundant prefix computation across requests'). Cache reads cost 'about one-tenth the price with Anthropic, DeepSeek, and GPT-5' (Li, ch. 2; October 2026). No cache-write prices claimed. Anthropic's docs (checked October 2026): cache reads 0.1x base input (some newest models lower), hits need identical prefixes, and the cache order is tools → system → messages; the strip's system-then-tools order is schematic.")
        r1 = Strip(SEG_1, 1.2)
        r2 = Strip(SEG_2, -0.3)
        r2.labels = m.VGroup(*r2.labels[3:])  # the shared prefix is labelled once, above call one
        l1, l2 = row_label("call 1", r1), row_label("call 2", r2)
        s1 = r1.under(0, N1, text("computed", 22, SUN, weight=m.SEMIBOLD))
        s2a = r2.under(0, REUSED, text("from cache", 22, MINT, weight=m.SEMIBOLD))
        s2b = r2.under(REUSED, N2, text("computed", 22, SUN, weight=m.SEMIBOLD))
        glow = r2.glow(0, REUSED)
        x_cut = (r2.cells[REUSED - 1].get_right()[0] + r2.cells[REUSED].get_left()[0]) / 2
        cut = m.DashedLine([x_cut, 1.55, 0], [x_cut, -0.68, 0], dash_length=0.09).set_stroke(MUTED, 2)
        sooner = text("less to compute, so the first token comes sooner", 24, MUTED).move_to([0, -1.55, 0])
        prices = m.VGroup(
            pill("input token  ×1", SUN, SUN_FILL, 24),
            pill("cache read  ≈ ×0.1", MINT, MINT_FILL, 24),
        ).arrange(m.RIGHT, buff=0.5).move_to([0, -2.35, 0])
        src = cite("Li, AI Agents in Depth · ch. 2 · Anthropic, DeepSeek, GPT-5").next_to(prices, m.DOWN, buff=0.3)

        self.say(
            "[Now, across requests.] [Call one sends Otto's prompt], [and the model computes all of it.]",
            self.new_header("03 · prompt caching", "Reuse the prefix"),
            m.AnimationGroup(m.FadeIn(l1), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in r1.cells], lag_ratio=0.04), m.FadeIn(r1.labels)),
            m.AnimationGroup(paint(r1.cells, "computed", lag=0.05), m.FadeIn(s1, shift=0.1 * m.UP)),
        )
        self.say(
            "[Call two re-sends exactly the same tokens], [plus a few new ones at the end.]",
            m.AnimationGroup(m.FadeIn(l2), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in r2.cells[:REUSED]], lag_ratio=0.04)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.LEFT) for c in r2.cells[REUSED:]], lag_ratio=0.06), m.FadeIn(r2.labels), m.Create(cut)),
        )
        self.say(
            "[So the provider keeps the prefix's keys and values from call one], [and reuses them]: [only the new tail is computed.]",
            m.Circumscribe(r1.span(0, N1), color=MINT, buff=0.08),
            m.AnimationGroup(m.FadeIn(glow), paint(r2.cells[:REUSED], "cached", lag=0.05), m.FadeIn(s2a, shift=0.1 * m.UP)),
            m.AnimationGroup(paint(r2.cells[REUSED:], "computed", lag=0.08), m.FadeIn(s2b, shift=0.1 * m.UP)),
        )
        self.say(
            "[That's prompt caching.] [The first word arrives sooner, because less is computed.]",
            m.Circumscribe(m.VGroup(r2.span(0, REUSED), glow), color=MINT, buff=0.06),
            m.FadeIn(sooner, shift=0.1 * m.UP),
        )
        self.say(
            "[And it's cheaper]: [cache reads cost about one tenth of the normal input price], [with Anthropic, DeepSeek and GPT five.]",
            m.FadeIn(prices[0], shift=0.1 * m.UP),
            m.FadeIn(prices[1], shift=0.1 * m.UP),
            m.FadeIn(src),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the prefix rule

    def prefix_rule(self):
        self.next_section("4 · The prefix rule", notes=f"The cache matches an exact prefix. 'If the token sequence first differs at some position, the KV states for that token and everything after it must be recomputed' (Li, ch. 2). Edits at block {EDITS}: cached/recomputed = {RULE} of {N2} (computed with first_difference). Schematic blocks.")
        ys = [1.35, 0.0, -1.35]
        rows = [Strip(SEG_2, y, x0=-6.05, pitch=0.5, w=0.4, h=0.48, state="cached", labels=(i == 0)) for i, y in enumerate(ys)]
        arrows = []
        status = []
        for r, at, (kept, lost) in zip(rows, EDITS, RULE):
            c = r.cells[at]
            arrows.append(m.Arrow(c.get_bottom() + 0.5 * m.DOWN, c.get_bottom() + 0.04 * m.DOWN, buff=0, color=ROSE, stroke_width=5,
                                  max_tip_length_to_length_ratio=0.45))
            pair = m.VGroup(code(f"{kept} cached", 20, MINT), code(f"{lost} recomputed", 20, ROSE)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
            status.append(pair.move_to([4.6, r.cells[0].get_y(), 0], aligned_edge=m.LEFT))
        scan = m.Line(rows[0].cells[0].get_bottom() + 0.12 * m.DOWN, rows[0].cells[-1].get_bottom() + 0.12 * m.DOWN).set_stroke(MINT, 8)
        quote = text("“If the token sequence first differs at some position,\neverything after it must be recomputed.”", 22, INK)
        quote.move_to([-1.2, -2.8, 0])
        src = cite("Li, AI Agents in Depth · ch. 2").move_to([0, -3.3, 0]).align_to([6.8, 0, 0], m.RIGHT)

        self.say(
            "[The catch]: [the cache only matches an exact prefix, token by token, from the start.]",
            m.AnimationGroup(self.new_header("04 · the prefix rule", "One change, everything after"),
                             m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in rows[0].cells], lag_ratio=0.03), m.FadeIn(rows[0].labels)),
            m.ShowPassingFlash(scan, time_width=0.4, run_time=1.6),
        )
        self.say(
            "[If the sequence first differs at some position], [everything after it must be recomputed.]",
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.AnimationGroup(rows[0].cells[EDITS[0]].animate.set_fill(ROSE_FILL).set_stroke(ROSE))),
            m.AnimationGroup(paint(rows[0].cells[EDITS[0] + 1:], "recomputed", lag=0.08), m.FadeIn(status[0], shift=0.1 * m.LEFT),
                             m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[Change a token at the very end], [and you lose almost nothing.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in rows[1].cells], lag_ratio=0.03), m.GrowArrow(arrows[1])),
            m.AnimationGroup(rows[1].cells[EDITS[1]].animate.set_fill(ROSE_FILL).set_stroke(ROSE), m.FadeIn(status[1], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Change one near the start], [and the whole prompt goes cold.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in rows[2].cells], lag_ratio=0.03), m.GrowArrow(arrows[2])),
            m.AnimationGroup(paint(rows[2].cells[EDITS[2]:], "recomputed", lag=0.04), m.FadeIn(status[2], shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the current time

    def current_time(self):
        self.next_section("5 · The current time", notes=f"Running example. A 'Current time' line in Otto's system prompt changes every call, so the first differing token is inside the system prompt and nothing after it is reused. For example (Li, ch. 7's numbers): calls of {SENT_T} tokens = {sum(SENT_T):,} sent for {3 * BLOCK:,} new. Billed in full-price tokens with cache reads at x{READ}: stable prefix {WARM} = {sum(WARM):,}; clock in the prompt {COLD} = {sum(COLD):,}; ratio {RATIO:.2f} (computed; cache writes counted at the normal price for simplicity). Reported incident (Li, ch. 2, 'one team'): a customer-service agent, 100,000 conversations a day, TTFT 0.5 s → 3–5 s, monthly bill nearly doubled. Fix: dynamic information goes at the end (the newest message) or behind a tool.")
        sysc, clock = otto_system()
        m.VGroup(sysc, clock).move_to([-3.95, 0.95, 0])
        ticks = [clock_at(clock, t) for t in TIMES[1:]]
        miss = text("a cache miss on every call", 22, ROSE, weight=m.SEMIBOLD).next_to(sysc, m.DOWN, buff=0.3).align_to(sysc, m.LEFT)
        miss2 = miss.copy()
        calm = text("no clock: the prefix repeats", 22, MINT, weight=m.SEMIBOLD).move_to(miss, aligned_edge=m.LEFT)

        legend = text("for example · each block = 1,000 tokens", 20, MUTED).move_to([2.55, 2.45, 0])
        rows, labels, blocks, costs = [], [], [], []
        for k, call in enumerate(prompts(False)):
            y = 1.8 - 0.8 * k
            lab = code(f"call {k + 1}", 20, MUTED).move_to([0, y, 0]).align_to([0.2, 0, 0], m.RIGHT)
            bs = []
            for j in range(len(call)):
                b = card(1.3, 0.62, EDGE, CARD, radius=0.12, stroke=2.5).move_to([1.15 + 1.42 * j, y, 0])
                bs.append(b)
            labels.append(lab)
            blocks.append(bs)
        tags0 = [[code("1,000", 20, MUTED).move_to(b) for b in bs] for bs in blocks]
        tags_warm = [[code("×0.1" if j < k else "×1", 20, INK).move_to(b) for j, b in enumerate(bs)] for k, bs in enumerate(blocks)]
        tags_cold = [[code("×1", 20, INK).move_to(b) for b in bs] for bs in blocks]
        warm_cost = [code(f"= {c:,}", 22, INK).move_to([4.85, b[0].get_y(), 0], aligned_edge=m.LEFT) for c, b in zip(WARM, blocks)]
        cold_cost = [code(f"= {c:,}", 22, ROSE).move_to([4.85, b[0].get_y(), 0], aligned_edge=m.LEFT) for c, b in zip(COLD, blocks)]
        totals = text(f"{sum(SENT_T):,} sent · {3 * BLOCK:,} new", 24, INK).move_to([2.55, -0.45, 0])
        all_blocks = [b for bs in blocks for b in bs]
        warm_states = [("cached" if j < k else "computed") for k, bs in enumerate(blocks) for j, _ in enumerate(bs)]

        scale = 7.0 / max(sum(COLD), sum(WARM))
        bx = -3.3
        bar_w = m.Rectangle(width=scale * sum(WARM), height=0.42, stroke_width=0, fill_color=MINT, fill_opacity=1).move_to([bx, -1.45, 0], aligned_edge=m.LEFT)
        bar_c = m.Rectangle(width=scale * sum(COLD), height=0.42, stroke_width=0, fill_color=ROSE, fill_opacity=1).move_to([bx, -2.15, 0], aligned_edge=m.LEFT)
        lab_w = text("stable prefix", 22, INK).move_to(bar_w).align_to([bx - 0.25, 0, 0], m.RIGHT)
        lab_c = text("clock in the prompt", 22, INK).move_to(bar_c).align_to([bx - 0.25, 0, 0], m.RIGHT)
        val_w = code(f"{sum(WARM):,}", 22, INK).next_to(bar_w, m.RIGHT, buff=0.2)
        val_c = code(f"{sum(COLD):,}", 22, INK).next_to(bar_c, m.RIGHT, buff=0.2)
        unit = text("billed, in full-price tokens", 20, MUTED).move_to([bx, -2.75, 0], aligned_edge=m.LEFT)
        ratio = title_text(f"×{RATIO:.1f}", 52, ROSE).move_to([5.85, -1.65, 0])
        nearly = text("nearly double", 22, MUTED).next_to(ratio, m.DOWN, buff=0.15)

        def tile(label, value):
            box = card(6.2, 1.45, EDGE, CARD, radius=0.18, stroke=2)
            a = text(label, 22, MUTED).move_to(box.get_corner(m.UL) + [0.35, -0.35, 0], aligned_edge=m.LEFT)
            b = title_text(value, 40, ROSE).move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + 0.22 * m.DOWN)
            return m.VGroup(box, a, b)

        t1 = tile("time to first token", "0.5 s  →  3–5 s").move_to([3.45, 1.55, 0])
        t2 = tile("monthly bill", "nearly ×2").move_to([3.45, -0.1, 0])
        who = m.VGroup(text("reported by one team: a support agent,", 22, INK), text("100,000 conversations a day", 22, INK)).arrange(m.DOWN, buff=0.1).move_to([3.45, -1.45, 0])
        who_src = cite("Li, AI Agents in Depth · ch. 2 · as reported").next_to(who, m.DOWN, buff=0.2)

        # the fix: the clock moves into the newest message
        hist = [
            msg_row("user", "The checkout test fails. Fix it.", mono=False),
            msg_row("assistant", 'bash("pytest -q")'),
            msg_row("tool_result", "1 failed, 12 passed"),
        ]
        newest = msg_row("user", f"Current time: {TIMES[2]}")
        col = m.VGroup(*hist, newest).arrange(m.DOWN, buff=0.16).move_to([3.45, 0.75, 0])
        prefix_pill = pill("cached prefix", MINT, MINT_FILL, 22).next_to(sysc, m.DOWN, buff=0.35).align_to(sysc, m.LEFT)
        only_new = text("only this is new", 22, SUN, weight=m.SEMIBOLD).next_to(newest, m.DOWN, buff=0.2).align_to(newest, m.RIGHT)
        rule2 = text("dynamic information goes at the end", 24, INK).move_to([0, -2.3, 0])

        self.say(
            "[Back to Otto: say we add one helpful line to its system prompt]: [the current time, down to the second.]",
            m.AnimationGroup(self.new_header("05 · the current time", "One line, double the bill"), m.FadeIn(sysc, shift=0.2 * m.UP)),
            m.FadeIn(clock, shift=0.15 * m.UP),
        )
        self.say(
            "[Now every call has a different time], [so the first changed token sits inside the system prompt], [and nothing after it matches.]",
            m.Transform(clock, ticks[0], run_time=0.5),
            m.Circumscribe(clock, color=ROSE, buff=0.08),
            m.FadeIn(miss, shift=0.1 * m.UP),
        )
        self.say(
            "[For example, say each call adds a thousand tokens]: [one thousand, then two, then three]: [six thousand sent, for three thousand new.]",
            m.FadeIn(legend),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(labels[k]), *[m.FadeIn(b, shift=0.1 * m.RIGHT) for b in blocks[k]], *[m.FadeIn(t) for t in tags0[k]]) for k in range(3)], lag_ratio=0.4),
            m.FadeIn(totals, shift=0.1 * m.UP),
        )
        self.say(
            "[With a stable prefix], [the repeated tokens are read from cache at a tenth of the price]: [billed like three thousand three hundred.]",
            m.AnimationGroup(clock.animate.set_opacity(0.2), m.FadeOut(miss), m.FadeIn(calm),
                             m.LaggedStart(*[b.animate.set_fill(STATES[s][1]).set_stroke(STATES[s][0]) for b, s in zip(all_blocks, warm_states)], lag_ratio=0.1, run_time=1.0)),
            m.AnimationGroup(*[m.FadeOut(t) for ts in tags0 for t in ts], *[m.FadeIn(t) for ts in tags_warm for t in ts], *[m.FadeIn(c, shift=0.1 * m.LEFT) for c in warm_cost]),
            m.AnimationGroup(m.FadeIn(lab_w), m.GrowFromEdge(bar_w, m.LEFT), m.FadeIn(val_w), m.FadeIn(unit)),
        )
        self.say(
            "[With the clock in the prompt], [nothing is reused]: [all six thousand at full price], [nearly double.]",
            m.AnimationGroup(m.Transform(clock, ticks[1], run_time=0.5), m.FadeOut(calm), m.FadeIn(miss2)),
            m.AnimationGroup(paint(all_blocks, "recomputed", lag=0.08), *[m.FadeOut(t) for ts in tags_warm for t in ts], *[m.FadeIn(t) for ts in tags_cold for t in ts],
                             *[m.FadeOut(c) for c in warm_cost], *[m.FadeIn(c, shift=0.1 * m.LEFT) for c in cold_cost]),
            m.AnimationGroup(m.FadeIn(lab_c), m.GrowFromEdge(bar_c, m.LEFT), m.FadeIn(val_c)),
            m.AnimationGroup(m.FadeIn(ratio, scale=0.8), m.FadeIn(nearly)),
        )
        right = m.VGroup(legend, *labels, *all_blocks, *[t for ts in tags_cold for t in ts], *cold_cost, totals)
        bars = m.VGroup(bar_w, bar_c, lab_w, lab_c, val_w, val_c, unit, ratio, nearly)
        self.say(
            "[One team reported exactly this]: [their time to first token went from half a second to three to five seconds], [and the monthly bill nearly doubled.]",
            m.AnimationGroup(m.FadeOut(right, shift=0.2 * m.RIGHT), m.FadeOut(bars, shift=0.2 * m.DOWN), m.FadeIn(who, shift=0.1 * m.UP), m.FadeIn(who_src)),
            m.FadeIn(t1, shift=0.15 * m.UP),
            m.FadeIn(t2, shift=0.15 * m.UP),
        )
        self.say(
            "[The fix]: [move the time out of the system prompt, into the newest message], [and the prefix lights up again.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(t1, t2, who, who_src, miss2), shift=0.2 * m.DOWN), m.LaggedStart(*[m.FadeIn(h, shift=0.1 * m.UP) for h in hist], lag_ratio=0.2)),
            m.AnimationGroup(m.FadeOut(clock, target_position=newest), m.FadeIn(newest, target_position=clock)),
            m.AnimationGroup(stroke_to([sysc[0], *[h[0] for h in hist]], MINT), newest[0].animate.set_stroke(SUN),
                             m.FadeIn(prefix_pill, shift=0.1 * m.UP), m.FadeIn(only_new), m.FadeIn(rule2, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ three rules

    def rules(self):
        self.next_section("6 · Three rules", notes="Li, AI Agents in Depth, ch. 2 ('KV cache-friendly context design'): 1. once the system prompt and tool definitions are finalized, do not change them ('even adding a single space'); 2. 'Always append dynamic information to the end'; 3. use the standard API message format, never hand-concatenated 'USER: / ASSISTANT:' text.")
        specs = [
            ("1", "Freeze the prefix", "system prompt and tools:\nset once, never edited,\nnot even a single space", [("SYSTEM, TOOLS  # fixed", MUTED)]),
            ("2", "Append at the end", "the time, user status,\nnew facts: always in\nnew messages", [("messages.append(…)", MINT)]),
            ("3", "Standard format", "role-tagged messages,\nnot history pasted\ninto one string", [('{"role": "user", …}', MINT), ('"USER: … ASSISTANT: …"', ROSE)]),
        ]
        cards = m.VGroup()
        for (num, title, desc, snippets), x in zip(specs, (-4.45, 0, 4.45)):
            box = card(4.1, 4.0, EDGE, CARD, radius=0.2, stroke=2).move_to([x, -0.15, 0])
            n = title_text(num, 56, SUN).move_to(box.get_corner(m.UL) + [0.55, -0.6, 0])
            t = text(title, 28, INK, weight=m.BOLD).next_to(n, m.DOWN, buff=0.25).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            d = text(desc, 22, MUTED).next_to(t, m.DOWN, buff=0.2).align_to(t, m.LEFT)
            sn = m.VGroup(*[code(s, 20, c) for s, c in snippets]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
            sn.align_to(t, m.LEFT).align_to(box.get_bottom() + 0.35 * m.UP, m.DOWN)
            cards.add(m.VGroup(box, n, t, d, sn))
        src = cite("Li, AI Agents in Depth · ch. 2").move_to([0, -3.3, 0]).align_to([6.75, 0, 0], m.RIGHT)

        self.say(
            "[The book AI Agents in Depth turns this into three rules.] [One]: [once the system prompt and tools are set, never change them, not even by a single space.]",
            m.AnimationGroup(self.new_header("06 · three rules", "Keep the cache warm"), m.FadeIn(src)),
            m.FadeIn(cards[0][:3], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(cards[0][3:], shift=0.1 * m.UP)),
        )
        self.say(
            "[Two]: [always append dynamic information at the end.]",
            m.FadeIn(cards[1][:3], shift=0.15 * m.UP),
            m.FadeIn(cards[1][3:], shift=0.1 * m.UP),
        )
        self.say(
            "[Three]: [use the standard message format], [not history pasted together into one string.]",
            m.FadeIn(cards[2][:3], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(cards[2][3]), m.FadeIn(cards[2][4][0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cards[2][4][1], shift=0.1 * m.UP), m.Circumscribe(cards[2][4][0], color=MINT, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ cache breakers

    def breakers(self):
        self.next_section("7 · Cache breakers", notes=f"Li, ch. 2, Experiment 2-3's harmful patterns: a dynamic system prompt (timestamps, per-user details); tools reordered by frequency ('a fixed order has almost no effect on tool-selection accuracy but substantially improves performance'); sliding windows ('Agents using sliding windows often fell into loops, repeatedly executing the same tool calls'); text-flattened history ('one of the most harmful patterns'). N binary conditions before the cache boundary give 2^N cache keys: {SWITCHES} → {len(PREFIXES)} ('2×2×2 = 8'). The switch tree is illustrative.")
        specs = [
            ("Dynamic system prompt", "the time, the user's name, per-turn details"),
            ("Reordered tools", "sorted by use: no accuracy gained, cache lost"),
            ("Sliding window", "drops old turns: the history's start shifts"),
            ("Flattened history", "“one of the most harmful patterns”"),
        ]
        tiles = m.VGroup()
        for (t, d), y in zip(specs, (1.95, 0.72, -0.51, -1.74)):
            box = card(6.5, 1.06, EDGE, CARD, radius=0.16, stroke=2).move_to([-3.5, y, 0])
            stripe = m.RoundedRectangle(corner_radius=0.04, width=0.09, height=0.8, stroke_width=0, fill_color=ROSE, fill_opacity=1)
            stripe.move_to(box).align_to(box, m.LEFT).shift(0.13 * m.RIGHT)
            words = m.VGroup(text(t, 24, INK, weight=m.BOLD), text(d, 20, ROSE if d.startswith("“") else MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
            words.move_to(box).align_to(box, m.LEFT).shift(0.42 * m.RIGHT)
            tiles.add(m.VGroup(box, stripe, words))
        loops = text("often fell into loops", 22, ROSE, weight=m.SEMIBOLD).move_to(tiles[2][2][0]).align_to(tiles[2][0], m.RIGHT).shift(0.3 * m.LEFT)
        sit(loops, baseline(tiles[2][2][0]))
        harmful = tiles[3][2][1]
        tiles[3][2].remove(harmful)
        src = cite("Li, AI Agents in Depth · ch. 2 · Experiment 2-3").move_to([0, -2.75, 0]).align_to([-6.75, 0, 0], m.LEFT)

        # the switch tree: root, 2, 4, then 8 leaves
        leaf_y = [1.95 - 0.62 * i for i in range(8)]
        levels = [leaf_y]
        for _ in range(SWITCHES):
            prev = levels[-1]
            levels.append([(prev[2 * i] + prev[2 * i + 1]) / 2 for i in range(len(prev) // 2)])
        levels = levels[::-1]  # root first
        xs = [0.75, 2.3, 3.85, 5.65]
        dots = [m.VGroup(*[m.Dot([x, y, 0], radius=0.1, color=SUN) for y in ys]) for x, ys in zip(xs[:3], levels[:3])]
        leaves = m.VGroup()
        for code_, y in zip(PREFIXES, levels[3]):
            box = card(0.95, 0.44, SUN, SUN_FILL, radius=0.1, stroke=2).move_to([xs[3], y, 0])
            leaves.add(m.VGroup(box, code(code_, 20, INK).move_to(box)))
        edges = []
        for lvl in range(SWITCHES):
            group = m.VGroup()
            for i, y in enumerate(levels[lvl]):
                for c in (2 * i, 2 * i + 1):
                    end = [xs[lvl + 1] - (0.5 if lvl == SWITCHES - 1 else 0.1), levels[lvl + 1][c], 0]
                    group.add(m.Line([xs[lvl] + 0.1, y, 0], end).set_stroke(MUTED, 2.5))
            edges.append(group)
        caption = text(f"{SWITCHES} yes/no switches → 2³ = {len(PREFIXES)} prefixes", 24, INK).move_to([3.4, -3.05, 0])

        self.say(
            "[Watch for cache breakers]: [a system prompt that changes per user or per turn], [and tools reordered by how often they're used.]",
            m.AnimationGroup(self.new_header("07 · cache breakers", "What breaks the cache"), m.FadeIn(src)),
            m.FadeIn(tiles[0], shift=0.15 * m.RIGHT),
            m.FadeIn(tiles[1], shift=0.15 * m.RIGHT),
        )
        self.say(
            "[A sliding window that drops old messages shifts the history every turn], [and in the book's tests, those agents often fell into loops.]",
            m.FadeIn(tiles[2], shift=0.15 * m.RIGHT),
            m.FadeIn(loops, scale=0.85),
        )
        self.say(
            "[And history flattened into plain text], [which the book calls one of the most harmful patterns.]",
            m.FadeIn(tiles[3], shift=0.15 * m.RIGHT),
            m.FadeIn(harmful, scale=0.85),
        )
        self.say(
            "[Switches multiply, too]: [three yes-or-no options before the cache boundary], [mean eight different prefixes to cache.]",
            m.AnimationGroup(m.FadeIn(dots[0], scale=0.5), m.Create(edges[0]), m.FadeIn(dots[1], scale=0.5)),
            m.AnimationGroup(m.Create(edges[1]), m.FadeIn(dots[2], scale=0.5), m.Create(edges[2])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(l, shift=0.1 * m.LEFT) for l in leaves], lag_ratio=0.1), m.FadeIn(caption, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ Claude Code

    def claude_code(self):
        self.next_section("8 · Claude Code", notes="learn-claude-code · s10 (the site's reading of Claude Code's source, not verified by us): the system prompt is 'separated by SYSTEM_PROMPT_DYNAMIC_BOUNDARY between static and dynamic parts'; 'the static parts hit global cache'; CLAUDE.md content and currentDate are prepended as a <system-reminder> user message. Li, ch. 2 agrees: Claude Code's prompt is 'split by a cache boundary marker'. October 2026.")
        w, x = 7.3, -2.35
        static = msg_card("system · static", ORCHID, "the same for every user and every session", w)
        dynamic = msg_card("system · dynamic", ORCHID, "this session's details", w)
        reminder = msg_card("user · <system-reminder>", AZURE, "CLAUDE.md · currentDate", w, mono=True)
        task = msg_card("user", AZURE, "The checkout test fails. Fix it.", w)
        static.move_to([x, 1.85, 0])
        line_y = static.get_bottom()[1] - 0.3
        boundary = m.DashedLine([x - w / 2, line_y, 0], [x + w / 2, line_y, 0], dash_length=0.12).set_stroke(SUN, 3)
        b_lab = code("SYSTEM_PROMPT_DYNAMIC_BOUNDARY", 20, SUN).next_to(boundary, m.DOWN, buff=0.16).align_to(boundary, m.LEFT)
        dynamic.next_to(b_lab, m.DOWN, buff=0.25).set_x(x)
        reminder.next_to(dynamic, m.DOWN, buff=0.22).set_x(x)
        task.next_to(reminder, m.DOWN, buff=0.22).set_x(x)
        notes_x = x + w / 2 + 0.4
        n1 = text("cached globally", 24, MINT, weight=m.BOLD).move_to(static).align_to([notes_x, 0, 0], m.LEFT)
        n2 = text("per session", 24, SUN, weight=m.BOLD).move_to(dynamic).align_to([notes_x, 0, 0], m.LEFT)
        n3 = m.VGroup(text("instructions + date,", 22, AZURE, weight=m.BOLD), text("as a message", 22, AZURE, weight=m.BOLD)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        n3.move_to(reminder).align_to([notes_x, 0, 0], m.LEFT)
        src = cite("learn-claude-code · s10 · reading CC's source  ·  Li, AI Agents in Depth · ch. 2").move_to([0, -3.3, 0])

        self.say(
            "[Claude Code is built this way.] [A boundary marker splits its system prompt]: [a static part, cached globally], [and a dynamic part after it.]",
            m.AnimationGroup(self.new_header("08 · Claude Code", "Static first, dynamic after"), m.FadeIn(src)),
            m.AnimationGroup(m.Create(boundary), m.FadeIn(b_lab, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(static, shift=0.15 * m.DOWN), m.FadeIn(n1, shift=0.1 * m.LEFT), static[0].animate.set_stroke(MINT)),
            m.AnimationGroup(m.FadeIn(dynamic, shift=0.15 * m.UP), m.FadeIn(n2, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[The project's instructions file and today's date aren't in the system prompt at all]: [they arrive as a reminder message, after it.]",
            m.Circumscribe(m.VGroup(static, dynamic), color=ORCHID, buff=0.08),
            m.AnimationGroup(m.FadeIn(reminder, shift=0.15 * m.UP), m.FadeIn(n3, shift=0.1 * m.LEFT), m.FadeIn(task, shift=0.15 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ measured

    def measured(self):
        self.next_section("9 · Measured", notes=f"Li, AI Agents in Depth, ch. 7, Table 7-4: an eight-turn refund workflow with real gpt-4o-mini calls. Costs {COST} → savings {SAVED} (computed). 'The gains were not additive': {SAVED['stable']} + {SAVED['compress']} = {ADDED} ≠ {SAVED['both']}, because 'compressing history also shortened the prefix available for cache reuse'. Compression is Episode 11.")
        rows = [("stable prefix", SAVED["stable"]), ("history compression", SAVED["compress"]), ("both", SAVED["both"])]
        ys = [1.45, 0.45, -0.55, -1.55]
        x0, per = -2.4, 0.15
        axis = m.Line([x0, 2.0, 0], [x0, -2.0, 0]).set_stroke(FAINT, 2.5)
        bars, labels, values = [], [], []
        for (name, v), y in zip(rows, ys):
            bars.append(m.RoundedRectangle(corner_radius=0.06, width=per * v, height=0.5, stroke_width=0, fill_color=SUN, fill_opacity=1).move_to([x0 + per * v / 2, y, 0]))
            labels.append(text(name, 24, INK).move_to([0, y, 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT))
            values.append(code(f"{v:.1f}%", 24, INK).next_to(bars[-1], m.RIGHT, buff=0.2))
        ghost_box = m.RoundedRectangle(corner_radius=0.06, width=per * ADDED, height=0.5, stroke_width=0, fill_color=FAINT, fill_opacity=0.25).move_to([x0 + per * ADDED / 2, ys[3], 0])
        ghost_edge = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.06, width=per * ADDED, height=0.5).set_stroke(MUTED, 2), num_dashes=40).move_to(ghost_box)
        ghost_lab = text("if they added up", 24, MUTED).move_to([0, ys[3], 0]).align_to([x0 - 0.3, 0, 0], m.RIGHT)
        ghost_val = code(f"{ADDED:.1f}%", 24, ROSE).next_to(ghost_box, m.RIGHT, buff=0.2)
        caption = text("cost saved vs. no cache and no compression · eight-turn refund workflow · gpt-4o-mini", 20, MUTED).move_to([0, -2.45, 0])
        src = cite("Li, AI Agents in Depth · ch. 7 · Table 7-4").move_to([0, 2.45, 0]).align_to([6.75, 0, 0], m.RIGHT)
        lesson = pill("measure the whole workflow", SUN, SUN_FILL, 24).move_to([0, -3.15, 0])

        self.say(
            "[Does it pay?] [In the book's eight-turn refund experiment], [a stable prefix alone cut the cost by twenty-eight point three percent.]",
            m.AnimationGroup(self.new_header("09 · measured", "Savings don't add up"), m.Create(axis)),
            m.AnimationGroup(m.FadeIn(caption), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(labels[0]), m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(values[0])),
        )
        self.say(
            "[Compressing the history alone saved seventeen point five.] [Both together saved thirty], [not forty-six.]",
            m.AnimationGroup(m.FadeIn(labels[1]), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(values[1])),
            m.AnimationGroup(m.FadeIn(labels[2]), m.GrowFromEdge(bars[2], m.LEFT), m.FadeIn(values[2])),
            m.AnimationGroup(m.FadeIn(ghost_lab), m.FadeIn(ghost_box), m.Create(ghost_edge), m.FadeIn(ghost_val)),
        )
        self.say(
            "[Compression had shortened the prefix the cache could reuse]: [so measure the whole workflow], [and never just add up the savings.]",
            m.Circumscribe(m.VGroup(labels[2], bars[2], values[2]), color=SUN, buff=0.1),
            m.FadeIn(lesson, scale=0.85),
            m.Circumscribe(m.VGroup(ghost_box, ghost_val), color=ROSE, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Freeze the prefix (system prompt and tools), append everything new at the end, and the cache reuses the rest (reads at about a tenth of the price).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Freeze", ORCHID), ("Append", SUN), ("Reuse", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("system prompt + tools", "new info at the end", "cache reads ≈ ×0.1")])
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
        tagline = text("Keep the prefix identical; the cache does the rest.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [freeze], [append], [reuse.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Freeze the prefix, append everything new at the end, and the cache reuses all the rest.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
