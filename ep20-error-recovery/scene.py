"""Agents, Animated · Episode 20 · Error Recovery (Module 4 · Tools & Safety)

How agents survive failures. Running example: Otto, the series' coding agent, is halfway through adding
a product catalogue to tinyshop when two things go wrong: the model's API pushes back (a 429 rate limit,
then 529 overloads), and a very long edit (a 3,000-product test fixture) is cut off at the output cap.
Nothing here is drawn by hand: the scene runs Learn Claude Code s11's retry_delay (seeded jitter), a
with_retry against a stand-in API that stays overloaded, the max_tokens branch of the loop against a
scripted model that needs 90,000 output tokens, a stand-in payments server with and without an
idempotency key, and a fingerprint of a repeated call. Every number on screen comes from those runs and
is asserted below.

Arc: hook (529 → a bare loop crashes; "Errors aren't the end, they're the start of a retry") · 1 four
failure layers (Li ch. 5) · 2 classify first: would retrying help? · 3 backoff with jitter (the curve,
the cap, lockstep vs jitter, Retry-After) · 4 fallback model after three 529s · 5 safe to do twice?
(idempotency keys, Li ch. 4) · 6 a truncated edit: escalate 8K → 64K once, then continue (up to 3) ·
7 loop fingerprints · 8 a counter and a ceiling per path; the breaker statistic (link to Ep 11); tell
the user what was tried · 9 escalation by visibility and "the unit of error handling" · recap
(Classify · Recover · Report).

Colours (series roles): model CORAL, tool call SUN, tool result MINT, user AZURE; errors ROSE; "wait,
then retry" MINT; "change the input" SUN; headers MINT (Module 4).

Facts (checked October 2026):
- Learn Claude Code s11 (learn.shareai.run/en/s11/): "Errors aren't the end, they're the start of a
  retry"; the agent without recovery "just crashes" on "Error: 529 overloaded"; three paths: max_tokens →
  escalate 8000 → 64000 once and retry the same request WITHOUT appending the cut output, then append it
  with a continuation prompt up to 3 times ("Output token limit hit. Resume directly — no apology, no
  recap. Pick up mid-thought."); prompt_too_long → reactive compact once; 429/529 → backoff
  min(500·2^attempt, 32000) ms + random 0–25 %, Retry-After takes priority, MAX_RETRIES = 10, after 3
  consecutive 529s switch to FALLBACK_MODEL. RecoveryState{has_escalated, recovery_count,
  consecutive_529, has_attempted_reactive_compact, current_model}. Dive into CC Source (the site's
  reading of Claude Code's source, not verified by us): CC switches e.g. Opus → Sonnet after 3
  consecutive 529s and shows the user a notice.
- Li, AI Agents in Depth, ch. 5 "Failure and Error Recovery": four layers (API: 429, overload, timeouts,
  truncation; tool: hallucinated calls, malformed arguments; context: overflow, compaction failure, "a
  tool call missing its paired result message"; control flow: infinite loops and death spirals);
  "classify first, then count", "Would retrying help?"; fingerprints of "tool name + arguments"; each
  recovery path keeps its own counter; silent retry with backoff and jitter "while honoring the server's
  suggested wait duration" → degrade (raise the output cap, fall back to another model) → surface to the
  user "together with the recovery actions already attempted"; Claude Code's compaction breaker: "one
  session once failed over three thousand times in a row" (B's study of CC; told in Ep 11); "the unit of
  error handling is not the single request, but the entire recovery loop."
- Li, ch. 4 "Idempotency and Cancellation Semantics": a timed-out transfer "might have already
  transferred the money"; a client-generated idempotency key lets the server return the first result for
  duplicates instead of executing again.

Illustrative (for teaching): the catalogue job and its size (3,000 products × 30 tokens), the refund
(order 1042, $30, key rf-1042), the retry-after value (20 s), five agents in lockstep, the seeded jitter
draws, the user notice wording.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import hashlib
import inspect
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the runs, computed

# Learn Claude Code s11's teaching constants
DEFAULT_MAX_TOKENS = 8000
ESCALATED_MAX_TOKENS = 64000
MAX_RECOVERY_RETRIES = 3
MAX_RETRIES = 10
BASE_DELAY_MS = 500
MAX_CONSECUTIVE_529 = 3
CONTINUATION_PROMPT = "Output token limit hit. Resume directly — no apology, no recap. Pick up mid-thought."
FALLBACK_MODEL = "fallback"


def retry_delay(attempt, retry_after=None):
    if retry_after:
        return retry_after
    base = min(500 * 2**attempt, 32000) / 1000
    return base + random.uniform(0, base*0.25)


DELAY_SRC = '''def retry_delay(attempt, retry_after=None):
    if retry_after:
        return retry_after
    base = min(500 * 2**attempt, 32000) / 1000
    return base + random.uniform(0, base*0.25)'''
assert inspect.getsource(retry_delay).strip() == DELAY_SRC  # the code on screen is the code that ran

BASES = [min(BASE_DELAY_MS * 2 ** a, 32000) / 1000 for a in range(MAX_RETRIES)]
assert BASES == [0.5, 1, 2, 4, 8, 16, 32, 32, 32, 32]  # "half a second, one, two, four … thirty-two"
random.seed(20)
DELAYS = [retry_delay(a) for a in range(MAX_RETRIES)]  # one seeded run (illustrative draws)
assert all(b <= d <= 1.25 * b for b, d in zip(BASES, DELAYS))  # "up to a quarter more"
TOTAL_LOW, TOTAL_HIGH, TOTAL_MEAN = sum(BASES), 1.25 * sum(BASES), 1.125 * sum(BASES)
assert TOTAL_LOW == 159.5 and round(TOTAL_MEAN / 60) == 3  # "about three minutes in all" (179 s on average)
assert retry_delay(0, retry_after=20) == 20  # "that wins"
RETRY_AFTER = 20

# five agents rate-limited at the same instant: their first three retries
AGENTS = 5
LOCKSTEP = [[sum(BASES[: k + 1]) for k in range(3)] for _ in range(AGENTS)]
assert LOCKSTEP[0] == [0.5, 1.5, 3.5] and all(row == LOCKSTEP[0] for row in LOCKSTEP)
_rng = random.Random(7)
SPREAD = []
for _ in range(AGENTS):
    t, times = 0.0, []
    for k in range(3):
        t += BASES[k] + _rng.uniform(0, BASES[k] * 0.25)
        times.append(t)
    SPREAD.append(times)
assert len({round(times[2], 2) for times in SPREAD}) == AGENTS  # no two agents retry together


class RateLimited(Exception):
    pass


class Overloaded(Exception):
    pass


class RecoveryState:
    def __init__(self):
        self.has_escalated = False
        self.recovery_count = 0
        self.consecutive_529 = 0
        self.has_attempted_reactive_compact = False
        self.current_model = "primary"


def with_retry(call, state, waited):
    for attempt in range(MAX_RETRIES):
        try:
            return call(state.current_model)
        except (RateLimited, Overloaded) as error:
            waited.append(retry_delay(attempt))
            is_overloaded = isinstance(error, Overloaded)
            if is_overloaded:
                state.consecutive_529 += 1
                if state.consecutive_529 >= 3:
                    state.current_model = FALLBACK_MODEL
    raise RuntimeError("MaxRetriesExceeded")


FALLBACK_SRC = '''if is_overloaded:
    state.consecutive_529 += 1
    if state.consecutive_529 >= 3:
        state.current_model = FALLBACK_MODEL'''
assert all(line.strip() in inspect.getsource(with_retry) for line in FALLBACK_SRC.split("\n"))

API_LOG = []


def busy_api(model):
    """The primary model stays overloaded; the fallback answers."""
    API_LOG.append(model)
    if model == "primary":
        raise Overloaded()
    return "ok"


_state, _waited = RecoveryState(), []
assert with_retry(busy_api, _state, _waited) == "ok"
assert API_LOG == ["primary"] * 3 + ["fallback"] and _state.consecutive_529 == MAX_CONSECUTIVE_529  # "three in a row"

# the truncated edit: Otto writes a 3,000-product fixture in one edit_file call
PRODUCTS, TOKENS_EACH = 3000, 30
NEED = PRODUCTS * TOKENS_EACH  # 90,000 output tokens (illustrative)
FIXTURE = "tests/fixtures/catalog.json"
TASK = "Write catalog.json"


def assistant(response):
    return {"role": "assistant", "content": response.content}


def user(content):
    return {"role": "user", "content": content}


class ScriptedModel:
    """Stands in for client.messages: writes what is left of the fixture, up to max_tokens."""

    def __init__(self):
        self.calls = []

    def create(self, messages, max_tokens):
        if isinstance(messages[-1]["content"], dict):  # the tool ran: the job is done
            self.calls.append((max_tokens, 0, "end_turn"))
            return NS(stop_reason="end_turn", content=NS(tokens=0))
        written = sum(msg["content"].tokens for msg in messages if msg["role"] == "assistant")
        left = NEED - written
        n = min(left, max_tokens)
        stop = "max_tokens" if left > max_tokens else "tool_use"
        self.calls.append((max_tokens, n, stop))
        return NS(stop_reason=stop, content=NS(tokens=n))


client = NS(messages=ScriptedModel())


def tool_results(response):
    return {"type": "tool_result", "content": f"wrote {FIXTURE}"}


def agent_loop(messages):
    state = RecoveryState()
    max_tokens = 8000
    while True:
        response = client.messages.create(messages=messages, max_tokens=max_tokens)
        if response.stop_reason == "max_tokens":
            if not state.has_escalated:
                state.has_escalated = True
                max_tokens = 64000  # was 8000
                continue  # same request, more room
            if state.recovery_count < 3:
                state.recovery_count += 1
                messages.append(assistant(response))
                messages.append(user(CONTINUATION_PROMPT))
                continue
            return response  # out of continuations: stop, report
        messages.append(assistant(response))
        if response.stop_reason != "tool_use":
            return state
        messages.append(user(tool_results(response)))


TRUNC_SRC = '''if response.stop_reason == "max_tokens":
    if not state.has_escalated:
        state.has_escalated = True
        max_tokens = 64000  # was 8000
        continue  # same request, more room
    if state.recovery_count < 3:
        state.recovery_count += 1
        messages.append(assistant(response))
        messages.append(user(CONTINUATION_PROMPT))
        continue'''
_loop_src = inspect.getsource(agent_loop)
assert all(line.strip() in _loop_src for line in TRUNC_SRC.split("\n")) and len(TRUNC_SRC.split("\n")) == 10

MESSAGES = [user(TASK)]
FINAL_STATE = agent_loop(MESSAGES)
CALLS = client.messages.calls
assert CALLS == [(8000, 8000, "max_tokens"), (64000, 64000, "max_tokens"), (64000, 26000, "tool_use"), (64000, 0, "end_turn")]
DISCARDED = CALLS[0][1]  # the first 8,000 tokens are thrown away, never appended
PIECES = [msg["content"].tokens for msg in MESSAGES if msg["role"] == "assistant" and msg["content"].tokens]
assert PIECES == [64000, 26000] and sum(PIECES) == NEED == 90000  # "in a second piece"
assert FINAL_STATE.has_escalated and FINAL_STATE.recovery_count == 1 <= MAX_RECOVERY_RETRIES
assert MESSAGES[2]["content"] == CONTINUATION_PROMPT and [msg["role"] for msg in MESSAGES][:4] == ["user", "assistant", "user", "assistant"]


# safe to do twice? a stand-in payments server
class Payments:
    def __init__(self):
        self.ledger, self.seen = [], {}

    def refund(self, order, amount, idempotency_key=None):
        if idempotency_key in self.seen:
            return self.seen[idempotency_key]  # the first result, not a second refund
        self.ledger.append((order, amount))
        result = f"refund #{len(self.ledger)}"
        if idempotency_key:
            self.seen[idempotency_key] = result
        return result


ORDER, AMOUNT, KEY = 1042, 30, "rf-1042"
_blind = Payments()
_blind.refund(ORDER, AMOUNT)  # the reply times out, but the refund went through
_blind.refund(ORDER, AMOUNT)  # a blind retry
BLIND_TOTAL = sum(a for _, a in _blind.ledger)
assert len(_blind.ledger) == 2 and BLIND_TOTAL == 60  # "refunds it twice"
_keyed = Payments()
FIRST = _keyed.refund(ORDER, AMOUNT, idempotency_key=KEY)
AGAIN = _keyed.refund(ORDER, AMOUNT, idempotency_key=KEY)
assert FIRST == AGAIN == "refund #1" and len(_keyed.ledger) == 1  # "returns the first result"


# loop fingerprints: tool name + arguments
def fingerprint(name, args):
    return hashlib.sha256((name + json.dumps(args, sort_keys=True)).encode()).hexdigest()[:8]


REPEAT = [("bash", {"command": "pytest -q"})] * 3
PRINTS = [fingerprint(n, a) for n, a in REPEAT]
FP = PRINTS[0]
assert len(set(PRINTS)) == 1 and fingerprint("bash", {"command": "pytest -q tests/test_cart.py"}) != FP
TEST_OUT = "1 failed, 12 passed"  # tinyshop's running bug (harness_reference.py)

# the ceilings (s11 teaching code; Ep 11 for the compaction breaker)
CEILINGS = [
    ("retry 429 · 529", "retry_attempt", MAX_RETRIES),
    ("bigger output cap", "has_escalated", 1),
    ("continue the reply", "recovery_count", MAX_RECOVERY_RETRIES),
    ("compact a long prompt", "reactive_compact", 1),
    ("compaction · ep 11", "failures in a row", 3),
]
assert [c for _, _, c in CEILINGS[:4]] == [10, 1, 3, 1]  # "ten retries, one bigger cap, three continuations, one compaction"

# ---------------------------------------------------------------------------- look

LLM_ICON = recolored("llm", CORAL, CORAL_FILL, prefix="ep20")
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep20")
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ROW_H = 0.46


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def sit(words, base):
    """Put a Text on a baseline (by its flat-bottomed glyphs)."""
    return words.shift((base - baseline(words, words.text.replace(" ", "").replace("\n", ""))) * m.UP)


def err_chip(label):
    return pill(label, ROSE, ROSE_FILL, 22, mono=True)


def row(role, body, kind, width, at=1.98, role_size=22):
    """One message on one line: a stripe in the block's colour, the role, the content."""
    accent = SUN if kind == "tool_use" else MINT if kind == "tool_result" else ROLES[role][0]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, role_size, ROLES[role][0], weight=m.BOLD)
    words = text(body, 20, INK) if kind == "text" else code(body, 20, accent)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(at * m.RIGHT)
    base = box.get_y() - 0.09
    label.shift((base - label[0].get_bottom()[1]) * m.UP)
    words.shift((base - words[0].get_bottom()[1]) * m.UP)
    return m.VGroup(box, stripe, label, words)


def ep_chip(label, ep, width=None):
    """A link to another episode: a neutral card, a label, "ep N" in that episode's module colour."""
    accent = MODULES[EPISODES[ep][1]][1]
    body = m.VGroup(text(label, 22, INK, weight=m.SEMIBOLD), code(f"ep {ep}", 20, accent, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
    sit(body[1], baseline(body[0], label.replace(" ", "")))
    box = card(width or body.width + 0.6, 0.66, EDGE, CARD, radius=0.16, stroke=2)
    return m.VGroup(box, body.move_to(box))


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


class CodeCard:
    """A code_block with a line cursor (a tinted bar) that can step through its lines."""

    def __init__(self, src, title, width=None, size=20):
        self.block = code_block(src, size=size, width=width, title=title)
        self.box, self.body, self.cap = self.block[0], self.block[1], self.block[2]
        self.body.set_z_index(2)
        self.cap.set_z_index(2)
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n

    def base(self, i):
        return baseline(self.lines[i], self.src[i].replace(" ", ""))

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=MINT):
        b = a if b is None else b
        top, bot = self.base(a) + 0.25, self.base(b) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


def model_card(label, width=2.7, height=2.1, size=26, icon_h=0.9):
    box = card(width, height, CORAL, CORAL_FILL)
    body = m.VGroup(icon(LLM_ICON, icon_h), text(label, size, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.18)
    return m.VGroup(box, body.move_to(box))


# ---------------------------------------------------------------------------- the episode


class Ep20ErrorRecovery(Episode):
    number = 20

    def lesson(self):
        self.hook()
        self.layers()
        self.classify()
        self.backoff()
        self.fallback()
        self.idempotent()
        self.truncation()
        self.loops()
        self.ceilings()
        self.recovery_loop()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Otto is halfway through adding a product catalogue to tinyshop (illustrative) when the API answers HTTP 529 with Anthropic's error shape {type: error, error: {type: overloaded_error}}. Learn Claude Code s11, 'The Problem': without recovery the agent 'just crashes' on 'Error: 529 overloaded'. Quote: \"Errors aren't the end, they're the start of a retry\" (learn-claude-code · s11). Facts dated October 2026.")
        bot = otto(1.2).move_to([-4.9, 1.45, 0])
        job = text("adding a product catalogue", 24, MUTED).next_to(bot, m.DOWN, buff=0.35)
        meter = Meter(3.0, 0.6, MINT, 0.2).next_to(job, m.DOWN, buff=0.3)
        step = code("step 3 of 5", 20, MUTED).next_to(meter, m.DOWN, buff=0.2)
        api = model_card("model API").move_to([4.6, 1.45, 0])
        go = arrow(bot.get_right() + 0.3 * m.RIGHT, api.get_left(), MUTED, buff=0.15)
        err = json_card({"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}},
                        size=20, accent=ROSE, title="HTTP 529")
        err.move_to([3.5, -1.55, 0])
        term = terminal(["$ python otto.py", "! Error: 529 overloaded", "! run stopped"], width=5.0, title="tinyshop")
        term.move_to([-4.1, -2.15, 0])
        lines = term[4]

        q1 = title_text("“Errors aren’t the end,", 50, INK)
        q2 = title_text("they’re the start of a retry.”", 50, INK)
        quote = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.3).move_to([0, 0.9, 0])
        src = cite("learn-claude-code · s11").next_to(quote, m.DOWN, buff=0.35)
        asks = m.VGroup(*[pill(s, SUN, SUN_FILL, 26) for s in ("retry what?", "how soon?", "how many times?")])
        asks.arrange(m.RIGHT, buff=0.5).move_to([0, -2.1, 0])

        self.say(
            "[Otto is halfway through a job in tinyshop], [when the model's API sends back an error]: [five twenty-nine, overloaded.]",
            m.LaggedStart(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(m.VGroup(job, meter, step), shift=0.2 * m.UP), m.FadeIn(api, shift=0.2 * m.UP), lag_ratio=0.3),
            m.AnimationGroup(m.GrowArrow(go), pulse(go, CORAL)),
            m.FadeIn(err, target_position=api),
        )
        self.say(
            "[A bare loop has no plan for this.] [The call raises an error], [and the whole run stops.]",
            m.AnimationGroup(
                bot[0].animate.set_stroke(FAINT), bot[1].animate.set_fill(FAINT), bot[2].animate.set_stroke(FAINT), bot[3].animate.set_stroke(FAINT),
                meter.bar.animate.set_fill(ROSE), m.FadeIn(m.VGroup(*term[:4]), shift=0.15 * m.UP), m.FadeIn(lines[0], shift=0.1 * m.UP),
            ),
            m.FadeIn(lines[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(lines[2], shift=0.1 * m.UP), m.Circumscribe(term, color=ROSE, buff=0.06)),
        )
        self.say(
            "[Learn Claude Code puts the cure in one line]: [errors aren't the end], [they're the start of a retry.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bot, job, meter, step, api, go, err, term), shift=0.2 * m.DOWN), m.FadeIn(src)),
            m.FadeIn(q1, shift=0.2 * m.UP),
            m.FadeIn(q2, shift=0.2 * m.UP),
        )
        self.say(
            "[But retry what], [how soon], [and how many times?]",
            m.FadeIn(asks[0], shift=0.15 * m.UP),
            m.FadeIn(asks[1], shift=0.15 * m.UP),
            m.FadeIn(asks[2], shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · four layers

    def layers(self):
        self.next_section("1 · Four failure layers", notes="Li, AI Agents in Depth, ch. 5 'Failure and Error Recovery': failures fall into four layers by where they occur — API (rate limiting HTTP 429, overload, timeouts, output truncated at the token limit: 'infrastructure noise'), tool (hallucinated calls, malformed arguments, execution exceptions), context (window overflow, compaction failure, 'a tool call missing its paired result message'), control flow (infinite loops, death spirals — the stop-hook spiral is Episode 18).")
        xs = [-4.95, -1.65, 1.65, 4.95]
        names = ["API", "Tool", "Context", "Control flow"]
        stack = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.02, width=0.42, height=0.07, stroke_width=0, fill_color=c, fill_opacity=1)
            for c in (ORCHID, AZURE, CORAL, MINT)
        ]).arrange(m.DOWN, buff=0.045)
        loop = m.Arc(radius=0.17, start_angle=0.5, angle=5.0).set_stroke(MUTED, 4).add_tip(tip_length=0.12, tip_width=0.12)
        marks = [icon(LLM_ICON, 0.42), icon("ep03-wrench", 0.42), stack, loop]
        bins, titles = [], []
        for x, name, mark in zip(xs, names, marks):
            box = card(3.05, 3.55, EDGE, "#18150F", radius=0.2, stroke=2).move_to([x, -1.15, 0])
            title = title_text(name, 26, INK)
            head = m.VGroup(mark, title).arrange(m.RIGHT, buff=0.18)
            head.move_to(box.get_top() + 0.42 * m.DOWN)
            bins.append(m.VGroup(box, head))
            titles.append(title)
        groups = [
            ["429 rate limit", "529 overloaded", "timeout", "max_tokens"],
            ["unknown tool", "bad arguments"],
            ["prompt too long", "missing result"],
            ["same call again", "death spiral"],
        ]
        chips = []
        for b, labels in zip(bins, groups):
            row_ = []
            for i, label in enumerate(labels):
                c = err_chip(label).move_to(b[0].get_top() + (1.08 + 0.62 * i) * m.DOWN)
                row_.append(c)
            chips.append(row_)
        spiral_link = code("see ep 18", 20, MUTED).next_to(chips[3][1], m.DOWN, buff=0.22)
        source = pill("incoming errors", ROSE, ROSE_FILL, 22).move_to([0, 1.75, 0])
        src = cite("Li, AI Agents in Depth · ch. 5").move_to([4.95, 1.75, 0])
        at = source.get_center()

        def fly(c):
            return m.FadeIn(c, target_position=at, scale=0.8)

        self.say(
            "[First, sort the failure.] [The book AI Agents in Depth names four layers.]",
            m.AnimationGroup(self.new_header("01 · layers", "Four places things break"), m.FadeIn(source, shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.UP) for b in bins], lag_ratio=0.2), m.FadeIn(src)),
        )
        self.say(
            "[The API layer]: [rate limits, overload, timeouts], [and replies cut off at the token limit.]",
            m.Indicate(titles[0], color=CORAL, scale_factor=1.15),
            m.LaggedStart(*[fly(c) for c in chips[0][:3]], lag_ratio=0.35),
            fly(chips[0][3]),
        )
        self.say(
            "[The tool layer]: [calls to tools that don't exist], [and arguments that break the rules.]",
            m.Indicate(titles[1], color=SUN, scale_factor=1.15),
            fly(chips[1][0]),
            fly(chips[1][1]),
        )
        self.say(
            "[The context layer]: [a window that overflows], [or a tool call that never got its result.]",
            m.Indicate(titles[2], color=ORCHID, scale_factor=1.15),
            fly(chips[2][0]),
            fly(chips[2][1]),
        )
        self.say(
            "[And control flow]: [loops that make no progress], [and death spirals, like the hook in episode eighteen.]",
            m.Indicate(titles[3], color=INK, scale_factor=1.15),
            fly(chips[3][0]),
            m.AnimationGroup(fly(chips[3][1]), m.FadeIn(spiral_link, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.st1 = NS(bins=bins, chips=chips, source=source, src=src, link=spiral_link)

    # ------------------------------------------------------------------ 2 · classify

    def classify(self):
        self.next_section("2 · Classify first", notes="Li, AI Agents in Depth, ch. 5: 'Detection: classify first, then count. When a failure occurs, the first question is not \"Should we retry?\" but \"Would retrying help?\"' Retryable: rate limiting, overload, network jitter. Non-retryable (invalid arguments, insufficient permissions, nonexistent tool) 'will produce the same result no matter how many times they are retried as-is — the input or strategy must change'; 'a mapping from error types to recovery strategies, rather than a blanket retry on error'. Tool errors go back to the model as tool results that teach: Episode 15.")
        s = self.st1
        keep_yes = s.chips[0][:3]
        keep_no = s.chips[1]
        rest = [*s.bins, s.source, s.src, s.link, s.chips[0][3], *s.chips[2], *s.chips[3]]
        pile = m.VGroup(*[c.copy() for c in (*keep_yes, *keep_no)]).arrange(m.RIGHT, buff=0.25).move_to([0, 1.8, 0])
        question = title_text("Would retrying help?", 36, INK).move_to([0, 0.75, 0])
        src = cite("Li, AI Agents in Depth · ch. 5").next_to(question, m.DOWN, buff=0.12)
        lanes = []
        for x, title, color, fill in ((-3.35, "Yes · wait, then retry", MINT, MINT_FILL), (3.35, "No · change the input", SUN, SUN_FILL)):
            box = card(6.0, 2.95, color, "#18150F", radius=0.2, stroke=2.5).move_to([x, -1.75, 0])
            head = title_text(title, 26, color).move_to(box.get_corner(m.UL) + [0.35, -0.42, 0], aligned_edge=m.LEFT)
            lanes.append(m.VGroup(box, head))
        for c in (*keep_yes, *keep_no):
            c.set_z_index(3)
        slots = []
        for lane, chips in ((lanes[0], keep_yes), (lanes[1], keep_no)):
            for i, c in enumerate(chips):
                spot = lane[0].get_corner(m.UL) + [0.35 + c.width / 2, -1.12 - 0.62 * i, 0]
                slots.append((c, spot))
        teach = ep_chip("errors that teach", 15).move_to(lanes[1][0].get_corner(m.UL) + [0.35, -2.5, 0], aligned_edge=m.LEFT)

        self.say(
            "[Then ask one question before any retry]: [would retrying help?]",
            m.AnimationGroup(
                self.new_header("02 · classify", "Classify first, then count"),
                m.FadeOut(m.VGroup(*rest), shift=0.2 * m.DOWN),
                *[c.animate.move_to(p) for c, p in zip((*keep_yes, *keep_no), pile)],
            ),
            m.AnimationGroup(m.FadeIn(question, shift=0.15 * m.UP), m.FadeIn(src), m.LaggedStart(*[m.FadeIn(l, shift=0.2 * m.UP) for l in lanes], lag_ratio=0.3)),
        )
        yes = slots[:3]
        no = slots[3:]
        self.say(
            "[A rate limit, an overload, or a timeout usually passes], [so wait, and try again.]",
            m.LaggedStart(*[c.animate.move_to(p) for c, p in yes], lag_ratio=0.3),
            m.Circumscribe(lanes[0][1], color=MINT, buff=0.08),
        )
        self.say(
            "[A bad argument fails the same way every time], [so the input must change], [with errors that teach, as in episode fifteen.]",
            m.LaggedStart(*[c.animate.move_to(p) for c, p in no], lag_ratio=0.3),
            m.Circumscribe(lanes[1][1], color=SUN, buff=0.08),
            m.FadeIn(teach, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · backoff

    def backoff(self):
        self.next_section("3 · Backoff with jitter", notes=f"Learn Claude Code s11 retry_delay (shown and run): base = min(500·2^attempt, 32000) ms → {BASES} s for retries 1–10; jitter adds 0–25 % (random.uniform(0, base*0.25)); Retry-After takes priority; MAX_RETRIES = 10. Ten retries wait {TOTAL_LOW:.1f}–{TOTAL_HIGH:.1f} s, {TOTAL_MEAN:.0f} s on average ('about three minutes'). Dots: one seeded run. Li ch. 5: 'exponential backoff with random jitter to prevent fleets of clients from retrying in lockstep … while honoring the server's suggested wait duration'. Five agents in lockstep and retry-after: 20 are illustrative; HTTP 429 rate_limit_error is the Anthropic API's error type.")
        # left: the error, the code, five agents in lockstep
        bot = otto(0.5).move_to([-6.1, 2.05, 0])
        err = pill("429 · rate_limit_error", ROSE, ROSE_FILL, 22, mono=True)
        err.next_to(bot, m.RIGHT, buff=0.45)
        hit = arrow(bot.get_right(), err.get_left(), MUTED, buff=0.08)
        ra = code(f"retry-after: {RETRY_AFTER}", 20, MUTED, t2c={r"\d+": SUN}).next_to(err, m.RIGHT, buff=0.3)
        cc = CodeCard(DELAY_SRC, "learn-claude-code · s11")
        cc.block.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, 1.4, 0], m.UP)
        cursor = cc.bar(3)

        panel = card(7.2, 2.05, EDGE, "#18150F", radius=0.18, stroke=2).move_to([-3.0, -2.4, 0])
        p_title = code("5 agents, rate-limited at once", 20, MUTED).move_to(panel.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        t0x, t_scale = -5.5, 1.2  # x of t = 0 s, units per second
        rows_y = [-1.98 - 0.29 * i for i in range(AGENTS)]
        minis = m.VGroup(*[otto(0.17).move_to([-6.25, y, 0]) for y in rows_y])
        lanes = m.VGroup(*[m.Line([t0x, y, 0], [0.35, y, 0]).set_stroke(EDGE, 1.5) for y in rows_y])
        dots = m.VGroup(*[m.Dot([t0x + t_scale * t, y, 0], radius=0.065, color=ROSE) for times, y in zip(LOCKSTEP, rows_y) for t in times])
        spread = [np.array([t0x + t_scale * t, y, 0]) for times, y in zip(SPREAD, rows_y) for t in times]
        spikes = m.VGroup(*[
            m.RoundedRectangle(corner_radius=0.08, width=0.26, height=0.29 * AGENTS + 0.06, stroke_width=0, fill_color=ROSE, fill_opacity=0.22)
            .move_to([t0x + t_scale * t, np.mean(rows_y), 0])
            for t in LOCKSTEP[0]
        ])

        # right: the backoff chart
        ox, oy, step, bar_w = 2.35, -2.6, 0.415, 0.27
        sy = 4.55 / 40  # units per second
        x_axis = m.Line([ox, oy, 0], [6.55, oy, 0]).set_stroke(FAINT, 2.5)
        y_axis = m.Line([ox, oy, 0], [ox, oy + 4.7, 0]).set_stroke(FAINT, 2.5)
        bx = [ox + 0.36 + step * i for i in range(MAX_RETRIES)]
        bars = [m.Rectangle(width=bar_w, height=max(b * sy, 0.02), stroke_width=0, fill_color=MINT, fill_opacity=1)
                .move_to([x, oy + b * sy / 2, 0]) for x, b in zip(bx, BASES)]
        bands = [m.Rectangle(width=bar_w, height=0.25 * b * sy, stroke_width=0, fill_color=MINT, fill_opacity=0.32)
                 .move_to([x, oy + 1.125 * b * sy, 0]) for x, b in zip(bx, BASES)]
        jit = m.VGroup(*[m.Dot([x, oy + d * sy, 0], radius=0.05, color=INK) for x, d in zip(bx, DELAYS)])
        nums = m.VGroup(*[code(str(i + 1), 20, MUTED).move_to([x, oy - 0.27, 0]) for i, x in enumerate(bx)])
        vals = [code(f"{b:g}", 20, INK).move_to([x, oy + 1.25 * b * sy + 0.26, 0]) for x, b in zip(bx[:4], BASES[:4])]
        y_lab = m.VGroup(*[code(f"{v} s", 20, MUTED).next_to([ox, oy + v * sy, 0], m.LEFT, buff=0.12) for v in (8, 16, 32)])
        ticks = m.VGroup(*[m.Line([ox - 0.07, oy + v * sy, 0], [ox + 0.07, oy + v * sy, 0]).set_stroke(FAINT, 2.5) for v in (8, 16, 32)])
        cap_y = oy + 32 * sy
        cap = m.DashedLine([ox, cap_y, 0], [6.55, cap_y, 0], dash_length=0.1).set_stroke(SUN, 2.5)
        cap_lab = code("cap 32 s", 20, SUN).move_to([bx[2], cap_y + 0.27, 0])
        axis_lab = code("retry", 20, MUTED).next_to(nums, m.DOWN, buff=0.14)
        total = text("10 retries · about 3 min of waiting", 22, MUTED).move_to([4.45, 2.3, 0])
        chart = m.VGroup(x_axis, y_axis, nums, y_lab, ticks, axis_lab)

        self.say(
            "[Start with the rate limit.] [Otto's call comes back four twenty-nine]: [too many requests.]",
            m.AnimationGroup(self.new_header("03 · backoff", "Wait longer each time"), m.FadeIn(bot, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(hit), m.FadeIn(err, shift=0.15 * m.LEFT)),
            m.Circumscribe(err, color=ROSE, buff=0.06),
        )
        self.say(
            "[Retry right away, and you hit the same wall.] [So each wait doubles]: [half a second], [one], [two], [four.]",
            m.Wiggle(err),
            m.AnimationGroup(m.FadeIn(cc.block, shift=0.15 * m.UP), m.FadeIn(cursor), m.FadeIn(chart)),
            *[m.AnimationGroup(m.GrowFromEdge(bars[i], m.DOWN), m.FadeIn(vals[i], shift=0.1 * m.UP), run_time=0.45) for i in range(4)],
        )
        self.say(
            "[The wait stops growing at thirty-two seconds], [and after ten retries, about three minutes in all, the harness gives up.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(b, m.DOWN) for b in bars[4:]], lag_ratio=0.25), m.Create(cap), m.FadeIn(cap_lab)),
            m.AnimationGroup(m.FadeIn(total, shift=0.1 * m.DOWN), m.Circumscribe(m.VGroup(*bars), color=MUTED, buff=0.1)),
        )
        self.say(
            "[Each wait also gets a random extra, up to a quarter more]: [that's jitter.]",
            m.AnimationGroup(m.Transform(cursor, cc.bar(4)), m.LaggedStart(*[m.FadeIn(b) for b in bands], lag_ratio=0.1), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in jit], lag_ratio=0.1)),
            m.Circumscribe(cc.part(4, "random.uniform(0, base*0.25)"), color=MINT, buff=0.06),
        )
        self.say(
            "[Without jitter, agents that failed together retry in lockstep], [and hit the server at the same instant.] [With it, they spread out.]",
            m.AnimationGroup(m.FadeIn(panel), m.FadeIn(p_title), m.FadeIn(minis), m.FadeIn(lanes), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.04)),
            m.LaggedStart(*[m.FadeIn(s, scale=0.8) for s in spikes], lag_ratio=0.3),
            m.AnimationGroup(m.FadeOut(spikes), *[d.animate.move_to(p).set_color(MINT) for d, p in zip(dots, spread)]),
        )
        self.say(
            "[And if the server says how long to wait, in a retry after header], [that wins.]",
            m.FadeIn(ra, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.Transform(cursor, cc.bar(1, 2)), m.Circumscribe(m.VGroup(*cc.lines[1], *cc.lines[2]), color=SUN, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · fallback

    def fallback(self):
        self.next_section("4 · Fallback model", notes=f"Learn Claude Code s11: after {MAX_CONSECUTIVE_529} consecutive 529s, switch to FALLBACK_MODEL (the four lines shown run in this scene's with_retry against an API whose primary model stays overloaded: calls {API_LOG}). Dive into CC Source (the site's reading of Claude Code, unverified): Claude Code switches e.g. Opus → Sonnet and shows the user a notice. Li ch. 5: 'When the primary model is persistently overloaded, fall back to another model'. Model names are Claude Code's example, not a recommendation.")
        log = card(7.0, 2.55, EDGE, "#18150F", radius=0.18, stroke=2).move_to([-3.1, 1.05, 0])
        l_title = code("calls", 20, MUTED).move_to(log.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        rows = []
        for i, model in enumerate(API_LOG[:3]):
            r = code(f"attempt {i + 1} · {model} → 529 overloaded", 20, INK, t2c={r"529 overloaded": ROSE, r"\d(?= ·)": SUN})
            r.move_to(log.get_corner(m.UL) + [0.3, -0.75 - 0.45 * i, 0], aligned_edge=m.LEFT)
            rows.append(r)
        count_at = lambda n: title_text(str(n), 30, ROSE)  # noqa: E731
        lead = code("state.consecutive_529 =", 20, MUTED)
        counter = Slot(self)
        fc = CodeCard(FALLBACK_SRC, "learn-claude-code · s11")
        fc.block.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, -0.5, 0], m.UP)
        cursor = fc.bar(0, 1)

        def model_row(label, eg, y):
            box = card(3.9, 1.3, CORAL, CORAL_FILL)
            words = text(label, 24, INK, weight=m.BOLD)
            sub = code(eg, 20, MUTED)
            col_ = m.VGroup(words, sub).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            mark = icon(LLM_ICON, 0.62)
            m.VGroup(mark, col_).arrange(m.RIGHT, buff=0.3).move_to(box)
            m.VGroup(box, mark, col_).move_to([4.4, y, 0])
            return m.VGroup(box, mark, words), sub

        primary, sub_p = model_row("primary model", "e.g. Opus", 1.8)
        backup, sub_b = model_row("fallback model", "e.g. Sonnet", -0.85)
        names = m.VGroup(sub_p, sub_b)
        busy = pill("529 · busy", ROSE, ROSE_FILL, 22, mono=True).next_to(primary, m.DOWN, buff=0.2)
        down = arrow(busy.get_bottom(), backup.get_top(), MUTED, buff=0.08)
        notice = m.VGroup(code("to the user", 20, AZURE, weight=m.BOLD), text("Switched to the fallback", 22, INK),
                          text("model due to high demand.", 22, INK)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        n_box = card(notice.width + 0.7, notice.height + 0.4, AZURE, CARD, radius=0.16, stroke=2)
        note = m.VGroup(n_box, notice.move_to(n_box)).move_to([4.4, -2.75, 0])
        src = cite("learn-claude-code · s11, on Claude Code").next_to(backup, m.DOWN, buff=0.16).align_to(backup, m.RIGHT)
        lead.move_to(log.get_corner(m.DL) + [0.3, 0.38, 0], aligned_edge=m.LEFT)

        def count(n):
            c = count_at(n).next_to(lead, m.RIGHT, buff=0.2)
            return c.shift((baseline(lead, "state.consecutive_529=") - c.get_bottom()[1]) * m.UP)

        self.say(
            "[An overload is different]: [the model itself is busy.] [After three five twenty-nines in a row], [Learn Claude Code switches to a fallback model.]",
            m.AnimationGroup(self.new_header("04 · fallback", "When the model is overloaded"), m.FadeIn(primary, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(busy, shift=0.15 * m.DOWN), primary[0].animate.set_stroke(ROSE)),
            m.AnimationGroup(
                m.FadeIn(m.VGroup(log, l_title, lead)), m.FadeIn(fc.block), m.FadeIn(cursor),
                m.Succession(*[m.AnimationGroup(m.FadeIn(r, shift=0.1 * m.UP), counter.to(count(i + 1)), run_time=0.5) for i, r in enumerate(rows)]),
            ),
            m.AnimationGroup(m.Transform(cursor, fc.bar(2, 3)), m.GrowArrow(down), m.FadeIn(backup, target_position=primary)),
        )
        self.say(
            "[In Learn Claude Code's reading, Claude Code does the same], [for example, Opus to Sonnet], [and tells the user it switched.]",
            m.FadeIn(src),
            m.LaggedStart(m.FadeIn(names[0]), m.FadeIn(names[1]), lag_ratio=0.4),
            m.FadeIn(note, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · safe to do twice

    def idempotent(self):
        self.next_section("5 · Safe to do twice?", notes=f"Li, AI Agents in Depth, ch. 4 'Idempotency and Cancellation Semantics': a call that times out 'might have already transferred the money … if the Agent retries without checking, it could duplicate the transfer'; fix: 'a client-generated idempotency key, which the server uses for deduplication, returning the first result for duplicate requests instead of executing again' (or query before mutating). Run here against a stand-in payments server: blind retry → {len(_blind.ledger)} refunds, ${BLIND_TOTAL}; with key {KEY} → {FIRST!r} both times. The refund tool, order {ORDER} and ${AMOUNT} are illustrative.")
        bot = otto(0.55).move_to([-6.05, 1.75, 0])
        call_src = f"refund(order={ORDER}, amount={AMOUNT})"
        call = code(call_src, 20, INK, t2c={r"\d+": SUN, r"refund": AZURE})
        c_box = card(call.width + 0.6, 0.72, SUN, CARD, radius=0.16, stroke=2)
        call_card = m.VGroup(c_box, call.move_to(c_box)).next_to(bot, m.RIGHT, buff=0.35)
        keyed_src = f'refund(order={ORDER}, amount={AMOUNT},\n       idempotency_key="{KEY}")'
        keyed = code(keyed_src, 20, INK, t2c={r"\"[^\"]*\"": MINT, r"\d+(?=[,)])": SUN, r"^refund": AZURE, r"idempotency_key": ORCHID})
        k_box = card(keyed.width + 0.6, keyed.height + 0.4, SUN, CARD, radius=0.16, stroke=2)
        keyed_card = m.VGroup(k_box, keyed.move_to(k_box))
        keyed_card.move_to(call_card).align_to(call_card, m.LEFT)

        pay_box = card(3.4, 0.95, EDGE, CARD, radius=0.18, stroke=2)
        pay = m.VGroup(pay_box, m.VGroup(icon(GEAR_ICON, 0.45), text("payments API", 26, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2).move_to(pay_box))
        pay.move_to([4.25, 1.75, 0])
        a0, a1 = keyed_card.get_right()[0] + 0.15, pay.get_left()[0] - 0.15
        out = arrow([a0, 1.9, 0], [a1, 1.9, 0], SUN, buff=0)
        back = m.DashedLine([a1, 1.55, 0], [a0, 1.55, 0], dash_length=0.1).set_stroke(MUTED, 3)
        mid = (a0 + a1) / 2
        timeout = pill("timeout", ROSE, ROSE_FILL, 22, mono=True).move_to([mid, 0.9, 0])
        retry_lab = code("retry", 20, MUTED).move_to([mid, 2.25, 0])

        ledger = card(4.6, 2.3, EDGE, "#18150F", radius=0.18, stroke=2).move_to([4.25, -0.9, 0])
        l_title = code("ledger", 20, MUTED).move_to(ledger.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        entries = [code(f"refund #{i + 1} · order {o} · ${a}", 20, INK, t2c={r"\$\d+": ROSE if i else INK})
                   .move_to(ledger.get_corner(m.UL) + [0.3, -0.85 - 0.48 * i, 0], aligned_edge=m.LEFT)
                   for i, (o, a) in enumerate(_blind.ledger)]
        seen = code(f"key {KEY} → {FIRST}", 20, MINT).move_to(ledger.get_corner(m.UL) + [0.3, -1.33, 0], aligned_edge=m.LEFT)
        total_bad = title_text(f"${BLIND_TOTAL} refunded", 34, ROSE).move_to([4.25, -2.6, 0])
        sub_bad = text(f"for one ${AMOUNT} order", 22, MUTED).next_to(total_bad, m.DOWN, buff=0.12)
        total_ok = title_text(f"${sum(a for _, a in _keyed.ledger)} refunded", 34, MINT).move_to(total_bad)
        sub_ok = text("once, whatever the retries", 22, MUTED).next_to(total_ok, m.DOWN, buff=0.12)
        same = pill(f"same result: {AGAIN}", MINT, MINT_FILL, 22, mono=True).move_to(timeout)

        ask = title_text("Safe to do twice?", 40, SUN).move_to([-3.3, -0.35, 0])
        rule = text("same key → same result, done once", 24, MUTED).next_to(ask, m.DOWN, buff=0.3)
        src = cite("Li, AI Agents in Depth · ch. 4").move_to([-3.3, -3.2, 0])

        self.say(
            "[Before any retry, one more check]: [is the action safe to do twice?]",
            m.AnimationGroup(self.new_header("05 · idempotency", "Never pay twice"), m.FadeIn(bot, shift=0.15 * m.UP), m.FadeIn(call_card, shift=0.15 * m.UP), m.FadeIn(pay, shift=0.15 * m.UP)),
            m.FadeIn(ask, shift=0.15 * m.UP),
        )
        self.say(
            "[Say Otto's tool refunds an order, and the call times out.] [The money may already be sent], [and a blind retry refunds it twice.]",
            m.Succession(m.AnimationGroup(m.GrowArrow(out), pulse(out, SUN), run_time=0.7), m.AnimationGroup(m.Create(back), m.FadeIn(timeout, shift=0.1 * m.UP), run_time=0.7)),
            m.AnimationGroup(m.FadeIn(ledger), m.FadeIn(l_title), m.FadeIn(entries[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(retry_lab), pulse(out, SUN), m.FadeIn(entries[1], shift=0.1 * m.UP), m.FadeIn(total_bad, shift=0.1 * m.UP), m.FadeIn(sub_bad), m.FadeIn(src)),
        )
        self.say(
            "[So the call carries an idempotency key.] [The server sees the same key again], [and returns the first result, instead of paying twice.]",
            m.AnimationGroup(m.FadeOut(call_card), m.FadeIn(keyed_card), m.FadeOut(entries[1], shift=0.15 * m.RIGHT), m.FadeIn(rule, shift=0.1 * m.UP)),
            m.AnimationGroup(pulse(out, SUN), m.FadeIn(seen, shift=0.1 * m.UP), m.Circumscribe(glyphs(keyed, f'"{KEY}"'), color=MINT, buff=0.05)),
            m.AnimationGroup(m.FadeOut(timeout), m.FadeIn(same, shift=0.1 * m.LEFT),
                             m.FadeOut(m.VGroup(total_bad, sub_bad), shift=0.1 * m.UP), m.FadeIn(m.VGroup(total_ok, sub_ok), shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · truncation

    def truncation(self):
        self.next_section("6 · A truncated edit", notes=f"Learn Claude Code s11, path 1 (max_tokens): 'escalate max_tokens from 8K to 64K … and retry the same request — the truncated output is NOT appended'; if 64K is still not enough, append it with a continuation prompt ('{CONTINUATION_PROMPT}'), up to 3 times; 'Escalation gets one chance; continuation gets up to 3.' The ten lines shown run in this scene's agent_loop against a scripted model: calls (max_tokens, written, stop_reason) = {CALLS}; {DISCARDED:,} tokens thrown away; pieces kept {PIECES}. The fixture ({PRODUCTS:,} products × {TOKENS_EACH} tokens = {NEED:,}) is illustrative. Claude Code's continuation prompt adds 'Break remaining work into smaller pieces' (s11 Dive into CC Source).")
        # the output track
        left, right, ty = -5.15, 6.2, 2.0
        span = right - left
        tx = lambda t: left + span * t / NEED  # noqa: E731
        track = m.RoundedRectangle(corner_radius=0.15, width=span, height=0.3, stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
        track.move_to([(left + right) / 2, ty, 0])
        t_lab = code("output", 20, MUTED).next_to(track, m.LEFT, buff=0.25)
        need = code(f"{NEED:,} needed", 20, MUTED).move_to([0, ty + 0.52, 0]).align_to([right, 0, 0], m.RIGHT)

        def seg(a, b, opacity=1.0):
            r = m.RoundedRectangle(corner_radius=0.12, width=span * (b - a) / NEED, height=0.24, stroke_width=0, fill_color=SUN, fill_opacity=opacity)
            return r.move_to([(tx(a) + tx(b)) / 2, ty, 0]).set_z_index(1)

        def cap_at(t):
            line = m.Line([tx(t), ty - 0.24, 0], [tx(t), ty + 0.22, 0]).set_stroke(SUN, 4)
            lab = code(f"cap {t:,}", 20, SUN).move_to([tx(t), ty + 0.52, 0])
            return m.VGroup(line, lab).set_z_index(2)

        def cut_at(t):
            return m.Line([tx(t), ty - 0.2, 0], [tx(t), ty + 0.2, 0]).set_stroke(ROSE, 5).set_z_index(3)

        def stop_at(t, label, color, fill):
            p = pill(label, color, fill, 22, mono=True)
            return p.move_to([min(max(tx(t), left + p.width / 2), right - p.width / 2), ty - 0.5, 0])

        bar8, bar64, bar_rest = seg(0, DEFAULT_MAX_TOKENS), seg(0, ESCALATED_MAX_TOKENS), seg(ESCALATED_MAX_TOKENS, NEED, 0.55)
        cap = Slot(self)
        stop = Slot(self)
        cut8, cut64 = cut_at(DEFAULT_MAX_TOKENS), cut_at(ESCALATED_MAX_TOKENS)

        # the code
        def state_words(escalated, count):
            s = f"state · has_escalated={escalated} · recovery_count={count}"
            return code(s, 20, MUTED, t2c={r"(?<==)(True|False)": ORCHID, r"(?<==)\d": SUN})

        cc = CodeCard(TRUNC_SRC, state_words(False, 0).text)
        cc.block.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, 0.95, 0], m.UP)
        state0 = state_words(False, 0).move_to(cc.cap).set_z_index(2)
        cc.block.remove(cc.cap)
        title = Slot(self, state0)

        def state_at(escalated, count):
            return state_words(escalated, count).move_to(state0, aligned_edge=m.LEFT).set_z_index(2)
        cursor = cc.bar(0)
        bottom = Slot(self)
        parts = ["“Output token limit hit.", "Resume directly —", "no apology, no recap.", "Pick up mid-thought.”"]
        assert " ".join(parts)[1:-1] == CONTINUATION_PROMPT
        prompt_card = m.VGroup(*[text(p_, 20, INK) for p_ in parts], cite("learn-claude-code · s11")).arrange(m.DOWN, buff=0.07, aligned_edge=m.LEFT)
        prompt_card.move_to([4.4, -2.68, 0]).align_to([2.3, 0, 0], m.LEFT)
        rule = m.VGroup(text("“Escalation gets one chance;", 24, INK), text("continuation gets up to 3.”", 24, INK),
                        cite("learn-claude-code · s11")).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        rule.move_to([4.4, -2.65, 0]).align_to([2.3, 0, 0], m.LEFT)

        # messages and state
        col = ContextColumn(center=(4.35, -0.5, 0), width=4.5, height=2.5, title="messages", gap=0.08)
        rw = col.width - 0.3
        r_task = row("user", TASK, "text", rw, at=1.9)
        r_cut = row("assistant", "edit_file(…", "tool_use", rw, at=1.9)
        r_go = row("user", "Resume directly…", "text", rw, at=1.9)
        r_end = row("assistant", "…the rest)", "tool_use", rw, at=1.9)
        for r in (r_task, r_cut, r_go, r_end):
            col.push(r)

        self.say(
            "[The second failure is quieter.] [Say Otto writes a big test fixture in one edit]: [three thousand products, about ninety thousand tokens.]",
            m.AnimationGroup(self.new_header("06 · truncation", "Raise the cap, then continue"), m.FadeIn(m.VGroup(track, t_lab), shift=0.15 * m.UP), cap.to(cap_at(DEFAULT_MAX_TOKENS))),
            m.AnimationGroup(m.FadeIn(col.frame), m.FadeIn(r_task, shift=0.2 * m.UP), m.FadeIn(cc.block), m.FadeIn(state0)),
            m.FadeIn(need, shift=0.1 * m.DOWN),
        )
        self.say(
            "[At eight thousand tokens, the reply stops mid-word.] [Its stop reason is max tokens], [and half an edit can't run.]",
            m.AnimationGroup(m.GrowFromEdge(bar8, m.LEFT, run_time=1.2), m.Succession(m.Wait(1.1), m.FadeIn(cut8, scale=1.5, run_time=0.3))),
            m.AnimationGroup(stop.to(stop_at(DEFAULT_MAX_TOKENS, 'stop_reason: "max_tokens"', ROSE, ROSE_FILL)), m.FadeIn(cursor)),
            m.Circumscribe(m.VGroup(bar8, cut8), color=ROSE, buff=0.08),
        )
        self.say(
            "[Step one: raise the output cap, once.] [The same request goes again, with room for sixty-four thousand], [and the cut reply is never added to messages.]",
            m.AnimationGroup(m.Transform(cursor, cc.bar(1, 3)), title.to(state_at(True, 0)), cap.to(cap_at(ESCALATED_MAX_TOKENS))),
            m.AnimationGroup(m.Transform(cursor, cc.bar(4)), m.FadeOut(m.VGroup(bar8, cut8), shift=0.2 * m.DOWN), stop.to(None)),
            m.Circumscribe(col.panel, color=MUTED, buff=0.05),
        )
        self.say(
            "[At sixty-four thousand, it's cut again.] [Step two: keep the partial reply], [and add a short user message]: [output token limit hit, resume directly.]",
            m.AnimationGroup(
                m.Transform(cursor, cc.bar(0)), m.GrowFromEdge(bar64, m.LEFT, run_time=1.6),
                m.Succession(m.Wait(1.5), m.AnimationGroup(m.FadeIn(cut64, scale=1.5), stop.to(stop_at(ESCALATED_MAX_TOKENS, 'stop_reason: "max_tokens"', ROSE, ROSE_FILL)), run_time=0.4)),
            ),
            m.AnimationGroup(m.Transform(cursor, cc.bar(5, 7)), title.to(state_at(True, 1)), m.FadeIn(r_cut, target_position=bar64)),
            m.AnimationGroup(m.Transform(cursor, cc.bar(8)), m.FadeIn(r_go, shift=0.2 * m.UP)),
            bottom.to(prompt_card),
        )
        done = stop_at(NEED, f"done · {NEED // 1000}K in {len(PIECES)} pieces", MINT, MINT_FILL)
        self.say(
            "[Otto picks up mid-thought], [and finishes the file in a second piece.]",
            m.AnimationGroup(m.Transform(cursor, cc.bar(9)), m.GrowFromEdge(bar_rest, m.LEFT, run_time=1.2)),
            m.AnimationGroup(m.FadeIn(r_end, target_position=bar_rest), stop.to(done)),
        )
        self.say(
            "[Escalation gets one chance], [continuation gets up to three.]",
            m.AnimationGroup(m.Transform(cursor, cc.bar(1)), bottom.to(rule)),
            m.AnimationGroup(m.Transform(cursor, cc.bar(5)), m.Circumscribe(cc.part(5, "< 3"), color=SUN, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · loops

    def loops(self):
        self.next_section("7 · Loop fingerprints", notes=f"Li, AI Agents in Depth, ch. 5: 'repeated-call fingerprints: hash the \"tool name + arguments\" pair; the same fingerprint recurring is a clear signal of a no-progress loop.' Here sha256(name + sorted-JSON args)[:8] = {FP} for bash {{\"command\": \"pytest -q\"}}, computed. The repeated pytest run (tinyshop's '{TEST_OUT}') is illustrative. Episode 2's iteration cap is the last resort.")
        rw = 5.3
        pairs = []
        for i in range(3):
            c = row("assistant", 'bash("pytest -q")', "tool_use", rw)
            r = row("user", TEST_OUT, "tool_result", rw)
            m.VGroup(c, r).arrange(m.DOWN, buff=0.08).move_to([-6.6 + rw / 2, 1.95 - 1.12 * i, 0], aligned_edge=m.UP)
            pairs.append((c, r))
        fps = [code(f"#{p}", 20, MUTED).next_to(c, m.RIGHT, buff=0.25) for (c, _), p in zip(pairs, PRINTS)]
        f_src = f'fingerprint = hash(name + args)\nname = "bash"\nargs = {{"command": "pytest -q"}}'
        formula = code_block(f_src, 20, title="per call")
        formula.move_to([0, 0, 0]).align_to([6.55, 0, 0], m.RIGHT).align_to([0, 2.45, 0], m.UP)
        result = code(f"→ #{FP}", 22, SUN, weight=m.BOLD).next_to(formula, m.DOWN, buff=0.25).align_to(formula, m.LEFT).shift(0.35 * m.RIGHT)
        times = title_text("×3", 64, ROSE)
        same = m.VGroup(text("same fingerprint,", 24, INK), text("same result", 24, INK)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        verdict = m.VGroup(times, same).arrange(m.RIGHT, buff=0.35).move_to([3.75, -0.95, 0])
        stuck = pill("no progress", ROSE, ROSE_FILL, 24).next_to(verdict, m.DOWN, buff=0.35)
        cap = ep_chip("turn cap: the last resort", 2).move_to([-3.95, -2.35, 0])
        src = cite("Li, AI Agents in Depth · ch. 5").next_to(stuck, m.DOWN, buff=0.35)

        self.say(
            "[Some failures never raise an error.] [Otto runs the same command, with the same arguments], [again and again.]",
            m.AnimationGroup(self.new_header("07 · loops", "Same call, same result"), m.FadeIn(m.VGroup(*pairs[0]), shift=0.15 * m.UP)),
            m.FadeIn(m.VGroup(*pairs[1]), shift=0.15 * m.UP),
            m.FadeIn(m.VGroup(*pairs[2]), shift=0.15 * m.UP),
        )
        self.say(
            "[So the harness fingerprints each call]: [the tool name plus its arguments.] [A repeat means no progress], [long before episode two's turn cap.]",
            m.LaggedStart(*[m.FadeIn(f, shift=0.15 * m.LEFT) for f in fps], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(formula, shift=0.15 * m.UP), m.FadeIn(result, shift=0.1 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(*[f.animate.set_color(ROSE) for f in fps], m.FadeIn(times, scale=0.8), m.FadeIn(same, shift=0.1 * m.UP), m.FadeIn(stuck, shift=0.1 * m.UP)),
            m.FadeIn(cap, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · ceilings

    def ceilings(self):
        self.next_section("8 · A ceiling per path", notes="Learn Claude Code s11 teaching constants: MAX_RETRIES = 10, one escalation (has_escalated), MAX_RECOVERY_RETRIES = 3 continuations (recovery_count), reactive compaction once ('compacting again won't make it any smaller'). Li, AI Agents in Depth, ch. 5: each recovery path keeps its own counter; 'every recovery path needs a ceiling'; Claude Code's compaction breaker (3 consecutive failures) comes from real session statistics — 'one session once failed over three thousand times in a row' (B's study of Claude Code; told in Episode 11). 'Only after all automatic means are exhausted is the error presented — together with the recovery actions already attempted.' The notice wording is illustrative.")
        table = card(7.6, 4.25, EDGE, "#18150F", radius=0.2, stroke=2).move_to([-2.8, 0.45, 0])
        tl = table.get_left()[0]
        heads = m.VGroup(code("path", 20, FAINT), code("counter", 20, FAINT), code("ceiling", 20, FAINT))
        col_x = [tl + 0.35, tl + 3.85, table.get_right()[0] - 0.35]
        top_y = table.get_top()[1] - 0.38
        heads[0].move_to([col_x[0], top_y, 0], aligned_edge=m.LEFT)
        heads[1].move_to([col_x[1], top_y, 0], aligned_edge=m.LEFT)
        heads[2].move_to([col_x[2], top_y, 0], aligned_edge=m.RIGHT)
        for h in heads[1:]:
            sit(h, baseline(heads[0], "path"))
        rows = []
        for i, (path, counter, ceiling) in enumerate(CEILINGS):
            y = top_y - 0.72 - 0.68 * i
            a = text(path, 22, INK).move_to([col_x[0], y, 0], aligned_edge=m.LEFT)
            b = code(counter, 20, MUTED).move_to([col_x[1], y, 0], aligned_edge=m.LEFT)
            c = title_text(str(ceiling), 30, SUN).move_to([col_x[2], y, 0], aligned_edge=m.RIGHT)
            base = baseline(a, path.replace(" ", ""))
            sit(b, base)
            c.shift((base - c.get_bottom()[1]) * m.UP)
            line = m.Line([tl + 0.25, y + 0.36, 0], [table.get_right()[0] - 0.25, y + 0.36, 0]).set_stroke(EDGE, 1.5)
            rows.append(m.VGroup(line, a, b, c))
        big = title_text("3,000+", 80, ROSE).move_to([3.95, 1.45, 0])
        what = m.VGroup(text("failures in a row,", 26, INK), text("one Claude Code session", 26, INK)).arrange(m.DOWN, buff=0.1)
        what.next_to(big, m.DOWN, buff=0.25)
        src = cite("Li, AI Agents in Depth · ch. 5").next_to(what, m.DOWN, buff=0.25)
        stop = pill("every path spent → stop", ROSE, ROSE_FILL, 22).move_to([3.95, -1.25, 0])
        msg = m.VGroup(
            code("to the user", 20, AZURE, weight=m.BOLD),
            text("Stopped: the model API stayed overloaded.", 22, INK),
            text("Tried: 10 retries with backoff, then a fallback model.", 22, INK),
        ).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        m_box = card(msg.width + 0.7, msg.height + 0.45, AZURE, CARD, radius=0.18, stroke=2)
        note = m.VGroup(m_box, msg.move_to(m_box)).move_to([0, -2.7, 0])

        self.say(
            "[Every path keeps its own counter, and its own ceiling]: [ten retries], [one bigger cap], [three continuations], [and one compaction when the prompt is too long.]",
            m.AnimationGroup(self.new_header("08 · ceilings", "Every path has a ceiling"), m.FadeIn(table), m.FadeIn(heads)),
            *[m.FadeIn(r, shift=0.12 * m.UP) for r in rows[:4]],
        )
        self.say(
            "[Episode eleven's breaker stops compaction after three failures in a row]: [in the book's study of Claude Code, one session failed over three thousand times.]",
            m.FadeIn(rows[4], shift=0.12 * m.UP),
            m.AnimationGroup(m.FadeIn(big, scale=0.85), m.FadeIn(what, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[When every path is spent, stop], [and tell the user what failed, and what was tried.]",
            m.AnimationGroup(m.FadeIn(stop, shift=0.1 * m.UP), m.Circumscribe(table, color=ROSE, buff=0.06)),
            m.FadeIn(note, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · the recovery loop

    def recovery_loop(self):
        self.next_section("9 · The recovery loop", notes="Li, AI Agents in Depth, ch. 5, escalation by visibility: silent retry (backoff + jitter; background calls dropped) → degrade and continue (raise the output cap, fall back to another model) → surface to the user with the recovery actions attempted. 'The core principle of this section is: the unit of error handling is not the single request, but the entire recovery loop.'")
        steps = [
            ("1 · retry quietly", "backoff + jitter", MINT, MINT_FILL),
            ("2 · change the request", "bigger cap · fallback model", SUN, SUN_FILL),
            ("3 · tell the user", "what failed · what was tried", AZURE, AZURE_FILL),
        ]
        cards = []
        for i, (title, sub, color, fill) in enumerate(steps):
            box = card(4.2, 1.45, color, fill, radius=0.2, stroke=2.5)
            body = m.VGroup(title_text(title, 24, INK), text(sub, 22, color)).arrange(m.DOWN, buff=0.2)
            cards.append(m.VGroup(box, body.move_to(box)).move_to([-4.45 + 4.45 * i, -0.75 + 0.7 * i, 0]))
        rise = arrow([-4.45, 0.45, 0], [1.9, 1.95, 0], MUTED, buff=0)
        rise_lab = code("more visible", 20, MUTED).move_to([-1.85, 1.55, 0]).rotate(0)
        q = m.VGroup(text("“The unit of error handling is not the single request,", 28, INK),
                     text("but the entire recovery loop.”", 28, INK)).arrange(m.DOWN, buff=0.12)
        q.move_to([0, -2.55, 0])
        src = cite("Li, AI Agents in Depth · ch. 5").next_to(q, m.DOWN, buff=0.2).align_to(q, m.RIGHT)

        self.say(
            "[Put together, recovery escalates]: [retry quietly], [then change the request], [and only then, tell the user.]",
            m.AnimationGroup(self.new_header("09 · the loop", "Recover quietly, then escalate"), m.GrowArrow(rise), m.FadeIn(rise_lab)),
            m.FadeIn(cards[0], shift=0.2 * m.UP),
            m.FadeIn(cards[1], shift=0.2 * m.UP),
            m.FadeIn(cards[2], shift=0.2 * m.UP),
        )
        self.say(
            "[As the book puts it]: [the unit of error handling is not the single request, but the entire recovery loop.]",
            m.FadeIn(src),
            m.FadeIn(q, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Classify (would retrying help?), recover (backoff, a bigger cap, a fallback model, a continuation), report (what failed, what was tried) — each path with its own counter and ceiling.")
        halo = glow(MINT, radius=4.0)
        words = m.VGroup(*[title_text(w, 68, c) for w, c in (("Classify", SUN), ("Recover", MINT), ("Report", AZURE))])
        subs = m.VGroup(*[text(s, 22, MUTED) for s in ("would retrying help?", "backoff · bigger cap · fallback", "what failed, what was tried")])
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
        tagline = text("Every failure gets its own path, its own counter, and its own ceiling.", 28, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [classify], [recover], [report.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Every failure gets its own path, its own counter, and its own ceiling.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
