"""Agents, Animated · Episode 28 · Computer Use, Voice & Robots  (Module 6 · Beyond the Loop)

How agents act beyond text, and why timing changes everything. This episode leaves tinyshop's terminal
and uses three examples of its own: a browser task (Otto adds a mug and a tea to tinyshop's cart in a
web page and checks the total: $3 + $4 = $7, the same sum as the course's running test), a phone call
(a customer rings tinyshop: "can I still change my delivery address?"), and a robot asked to put a red
cup on a tray (Li's pick(red_cup) / place(red_cup, tray) example).

Arc: beyond text (Episode 1's spaces were text only; Episode 27 made Otto general) · two axes, modality
and timing, on a log timescale · the screenshot -> model -> action -> wait loop · what a few seconds
between frames misses · grounding (numbered marks, an element index, raw coordinates + Li's scaling
example) · voice as a cascade (VAD -> ASR -> LLM -> TTS: waits add up, tone is lost) · streaming
(speculative model on the provisional transcript, first sentence to speech at once; Episode 7's time to
first token) · end-to-end, full-duplex, fast talker + slow thinker · the robot layer cake from minutes
to milliseconds · bounded skills instead of joint angles, a fresh look after every skill · the shared
skeleton · recap (ground, stream, layer).

Sources (facts dated October 2026):
- Li, AI Agents in Depth, ch. 6 (bojieli.github.io/ai-agent-book/book-en/chapter6/): observation and
  action spaces "can be expanded along two dimensions" (modality, timing); "Turn-taking is an interaction
  convention of the model and its interface, not a property of the environment"; timescales: async and
  event-driven "seconds — days", voice "10 ms — 1 s", computer use "sub-second — seconds", robotics
  "milliseconds"; the computer-use loop (screenshot, think, act, "waits for the interface to respond, takes
  another screenshot"); "An Agent that opens its eyes only once every 3–5 seconds and has no ears cannot
  see or hear what happens between two frames"; visual grounding: numbered marks turn localisation into
  "a multiple-choice problem" ("multiple-choice questions are easier to answer correctly than
  fill-in-the-blank questions"), structured element indexing, pure coordinate prediction; scaling: "if the
  actual screen resolution is 2560×1440 (16:9), the most suitable target among Claude's three supported
  options is FWXGA (1366×768)"; "after the model outputs the click coordinates (683, 384), they are
  inversely mapped" (to 1280, 720: computed and asserted below). Voice: cascaded pipeline, "Latency
  accumulates and paralinguistic information is lost at interfaces"; Omni models "still assume
  turn-taking"; full-duplex "listens and speaks continuously"; streaming ASR, "the provisional transcript
  is sent to the LLM as soon as it exists", "the first speakable sentence goes to TTS without waiting for
  the full reply"; fast talker + slow advisor. NO millisecond budget is given, so the voice timeline shows
  structure only (illustrative units, said so on screen). Robotics: goal (minutes), planning (seconds to
  minutes), skills such as pick(red_cup) "about 1–3 s", VLA "about 1–10 Hz", low-level control "about
  50–1000 Hz"; "the model should not emit arbitrary joint angles directly" — it chooses bounded skills run
  by a calibrated, speed-limited controller; "only once the cup is confirmed to be held may the planner
  submit place(red_cup, tray)". Shared skeleton: "keep perceiving → judge current state and timing →
  choose a reply or an action → let the output enter the environment → observe the feedback → continue,
  correct, retry, stop, or replan".
- Set-of-Mark prompting: Yang et al., 2023. Anthropic's computer-use docs (platform.claude.com, read October
  2026): the current toolset (computer_toolset_20260801) has the model emit a tool_use block named "left_click"
  with toolset_name "computer" and input {"coordinate": [x, y]}; "Coordinates are in screenshot pixels"; "If you
  scale screenshots down before returning them, scale Claude's coordinates back up"; for web applications
  "use 1280x800 or 1366x768". (The 2024–2025 versions used one "computer" tool with an "action" field.) Moshi (Kyutai, 2024) as a
  full-duplex example (named in Li, ch. 6).

Illustrative (said "for example" or marked on screen): the tinyshop web page, its prices and layout (the
page is drawn at the screenshot's 1366×768 aspect, so a point on it IS a pixel: the mug's button is
computed at (249, 384), the tea's at the centre (683, 384)); the four-second frame interval and the
pop-up; the DOM lines; the voice timeline's units; the joint angles; the band edges "sub-second" = 0.5 s,
"seconds" = 10 s, "days" = 2 days (the robotics band is the 50–1000 Hz control loop: 1–20 ms).
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep28"
SOURCE = "Li, AI Agents in Depth · ch. 6"

# ---------------------------------------------------------------------------- grounding (Li, ch. 6)

SCREEN = (2560, 1440)  # the real screen
SHOWN = (1366, 768)  # what the model is shown (FWXGA)
MODEL_PT = (683, 384)  # the model's click on the image
SCALE = (SCREEN[0] / SHOWN[0], SCREEN[1] / SHOWN[1])
REAL_PT = (round(MODEL_PT[0] * SCALE[0]), round(MODEL_PT[1] * SCALE[1]))
assert REAL_PT == (1280, 720)  # Li's example, recomputed
assert MODEL_PT == (SHOWN[0] // 2, SHOWN[1] // 2) and REAL_PT == (SCREEN[0] // 2, SCREEN[1] // 2)  # centre → centre
assert MODEL_PT[0] * SCREEN[0] / SHOWN[0] == 1280 and MODEL_PT[1] * SCREEN[1] / SHOWN[1] == 720  # exact, no rounding
PIXELS = SHOWN[0] * SHOWN[1]
assert PIXELS == 1_049_088  # the "fill-in-the-blank": any one of 1,049,088 pixels

# ---------------------------------------------------------------------------- the browser task

BW = 6.3
BH = BW * SHOWN[1] / SHOWN[0]  # drawn at the screenshot's aspect: a point on the page is a pixel
PRODUCTS = (("Mug", 3), ("Tea", 4), ("Bowl", 5))
TILE_X = (-2.0, 0.0, 2.0)  # product tiles, relative to the page centre; buttons sit on the centre line
SEARCH_REL, CHECKOUT_REL = (0.0, 1.07), (2.0, -0.95)
REST = np.array([0.55, -0.3, 0.0])  # where the pointer rests after a click (off the button's label)


def to_px(xr, yr):
    """A point relative to the page centre → its pixel in the 1366×768 screenshot."""
    return (round((xr + BW / 2) / BW * SHOWN[0]), round((BH / 2 - yr) / BH * SHOWN[1]))


MUG_PX, TEA_PX = to_px(TILE_X[0], 0.0), to_px(TILE_X[1], 0.0)
assert TEA_PX == MODEL_PT  # the tea's button is the centre of the screenshot: Li's (683, 384)
assert MUG_PX == (249, 384)
TOTAL = PRODUCTS[0][1] + PRODUCTS[1][1]
assert TOTAL == 7  # "seven dollars": a mug and a tea

MARKS = ("search", "mug", "tea", "bowl", "checkout")  # numbered marks 1–5
TEA_MARK = MARKS.index("tea") + 1
DOM = (
    '[1] <input placeholder="Search">',
    '[2] <button aria-label="Add mug">',
    '[3] <button aria-label="Add tea">',
    '[4] <button aria-label="Add bowl">',
    "[5] <button>Checkout</button>",
)
TEA_INDEX = 3
assert TEA_MARK == TEA_INDEX == 3 and "tea" in DOM[TEA_INDEX - 1] and len(DOM) == len(MARKS)

FRAME_EVERY = 4  # seconds between screenshots, for example (Li: "once every 3–5 seconds")
FRAMES = tuple(range(0, 13, FRAME_EVERY))
POPUP = (5, 7)  # a pop-up shown from second 5 to second 7, for example
assert FRAMES == (0, 4, 8, 12) and not any(POPUP[0] <= f <= POPUP[1] for f in FRAMES)  # never in a frame

# ---------------------------------------------------------------------------- timescales (Li, ch. 6)

CONTROL_HZ, VLA_HZ, SKILL_S = (50, 1000), (1, 10), (1, 3)
DAY = 86400
# (row, modality, lo seconds, hi seconds, label)
BANDS = (
    ("Async & events", "text, files, messages", 1.0, 2 * DAY, "seconds – days"),
    ("Computer use", "a screen: pixels, clicks", 0.5, 10.0, "sub-second – seconds"),
    ("Voice", "sound, both ways", 0.01, 1.0, "10 ms – 1 s"),
    ("Robotics", "a body in the world", 1 / CONTROL_HZ[1], 1 / CONTROL_HZ[0], "milliseconds"),
)
assert (BANDS[3][2], BANDS[3][3]) == (0.001, 0.02)  # the 50–1000 Hz control loop: 1 to 20 ms per step
T_LO, T_HI = -3.0, 5.4  # log10 seconds: 1 ms … about 2.9 days
AX_L, AX_R = -2.45, 6.3
TICKS = ((0.001, "1 ms"), (0.01, "10 ms"), (0.1, "100 ms"), (1, "1 s"), (10, "10 s"), (3600, "1 h"), (DAY, "1 day"))
assert all(T_LO <= np.log10(lo) < np.log10(hi) <= T_HI for _, _, lo, hi, _ in BANDS)


def tx(seconds):
    return AX_L + (np.log10(seconds) - T_LO) / (T_HI - T_LO) * (AX_R - AX_L)


# ---------------------------------------------------------------------------- the phone call (illustrative units)

LANES = ("caller", "VAD", "ASR", "LLM", "TTS", "reply")
CASCADE = {"caller": (0.0, 3.0), "VAD": (3.0, 3.8), "ASR": (3.8, 5.0), "LLM": (5.0, 7.0), "TTS": (7.0, 7.9), "reply": (7.9, 10.4)}
STREAM = {"caller": (0.0, 3.0), "VAD": (3.0, 3.5), "ASR": (0.3, 3.3), "LLM": (2.4, 4.6), "TTS": (3.6, 6.1), "reply": (4.0, 6.5)}
for _a, _b in zip(LANES, LANES[1:]):
    assert CASCADE[_b][0] == CASCADE[_a][1]  # a cascade: each stage starts when the one before ends
assert STREAM["LLM"][0] < STREAM["caller"][1]  # speculative: the model starts before the caller has finished
assert STREAM["reply"][0] > STREAM["VAD"][1]  # but speaks only once the turn is over


def silence(plan):
    return plan["reply"][0] - plan["caller"][1]


assert silence(STREAM) < silence(CASCADE)
assert np.isclose(CASCADE["reply"][1] - CASCADE["reply"][0], STREAM["reply"][1] - STREAM["reply"][0])  # same reply
G_X0, G_U, LANE_Y0, LANE_DY = -5.0, 1.08, -0.45, 0.4


def gx(t):
    return G_X0 + G_U * t


# ---------------------------------------------------------------------------- the robot (Li, ch. 6)

LAYERS = (  # name, example, rate, colour, mono example?
    ("Goal", "put the red cup on the tray", "minutes", AZURE, False),
    ("Plan", "pick, check, place", "seconds–minutes", CORAL, True),
    ("Skill", "pick(red_cup)", f"~{SKILL_S[0]}–{SKILL_S[1]} s", SUN, True),
    ("VLA", "vision-language-action model", f"~{VLA_HZ[0]}–{VLA_HZ[1]} Hz", LIME, False),
    ("Control", "motor commands", f"~{CONTROL_HZ[0]}–{CONTROL_HZ[1]} Hz", MUTED, False),
)
assert [r[2] for r in LAYERS[2:]] == ["~1–3 s", "~1–10 Hz", "~50–1000 Hz"]
JOINTS = (0.42, -1.10, 0.87)  # arbitrary joint angles (radians), for example

# ---------------------------------------------------------------------------- helpers


def cite(s):
    return text(s, 20, MUTED)


def top_cite(s):
    return cite(s).to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)


def mono_chip(word, color, fill, size=22, height=0.6, pad=0.45):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def word_chip(word, color, fill, size=22, height=0.6, pad=0.5):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def put_head(bot, point):
    return bot.shift(np.array(point) - bot[0].get_center())


JSON_T2C = {
    r"\"[^\"\n]*\"(?=\s*:)": AZURE,
    r":\s*\"[^\"\n]*\"": MINT,
    r"\b-?\d+(\.\d+)?\b": SUN,
}


def jcard(src, accent=SUN, title="tool_use", size=20, width=None):
    """A JSON card like kit.json_card, from a pre-formatted string (kit's json.dumps would put a
    coordinate pair on four lines)."""
    body = code(src, size, INK, t2c=JSON_T2C)
    cap = code(title, 20, accent, weight=m.BOLD)
    top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, cap.width + 0.7)
    box = card(w, body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
    return m.VGroup(box, body, cap)


def row_msg(role, body, width=6.6, size=20):
    """A one-row message: the role label and the content side by side (a compact kit.message)."""
    color, _ = ROLES[role]
    label = code(role, size, color, weight=m.BOLD)
    words = code(body, size, INK)
    box = card(width, 0.56, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=0.4, stroke_width=0, fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    label.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(2.3 * m.RIGHT)
    return m.VGroup(box, stripe, label, words)


def wave(color, width=0.9, height=0.55, n=7, seed=3):
    """A little audio waveform: rounded bars of varying height."""
    rng = np.random.default_rng(seed)
    heights = 0.3 + 0.7 * rng.random(n)
    step = width / n
    bars = m.VGroup(*[
        m.RoundedRectangle(corner_radius=step * 0.25, width=step * 0.55, height=height * h, stroke_width=0,
                           fill_color=color, fill_opacity=1).move_to([i * step, 0, 0])
        for i, h in enumerate(heights)
    ])
    return bars.move_to(m.ORIGIN)


def cursor():
    """A mouse pointer whose tip is the group's point [0]; use point_at()."""
    pts = [(0, 0), (0, -0.44), (0.11, -0.33), (0.2, -0.52), (0.28, -0.48), (0.19, -0.3), (0.33, -0.3)]
    shape = m.Polygon(*[np.array([x, y, 0.0]) for x, y in pts], stroke_color=BG, stroke_width=2.5, fill_color=INK, fill_opacity=1)
    shape.set_z_index(6)
    return shape


def point_at(ptr, target):
    return ptr.shift(np.array(target) - ptr.get_corner(m.UL))


def glide(ptr, target, run_time=0.6):
    """Move the pointer so that its tip lands on `target` (absolute: ManimGX applies .animate methods
    when the animation begins, so a relative shift computed earlier would add up wrongly)."""
    return ptr.animate(run_time=run_time).move_to(np.array(target) + ptr.get_center() - ptr.get_corner(m.UL))


def crosshair(center, color=SUN, w=1.7, h=0.5):
    """A target frame around a button (it never crosses the button's label)."""
    c = np.array(center)
    frame = m.RoundedRectangle(corner_radius=0.14, width=w, height=h).set_stroke(color, 3).set_fill(opacity=0).move_to(c)
    ticks = m.VGroup(*[m.Line(c + d * (w / 2), c + d * (w / 2 + 0.16)).set_stroke(color, 3) for d in (m.LEFT, m.RIGHT)])
    return m.VGroup(frame, ticks).set_z_index(5)


def shutter(box, color=MINT):
    """A camera-flash sweep around a window: the harness takes a screenshot."""
    return m.ShowPassingFlash(box.copy().set_fill(opacity=0).set_stroke(color, 7), time_width=0.6, run_time=0.8)


def arc_flash(arc, color, run_time=0.6):
    return m.ShowPassingFlash(arc.copy().set_stroke(color, 7), time_width=0.7, run_time=run_time)


class Slot:
    """One place on screen whose content is swapped (a count, a reply). (From Episode 2.)"""

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


class Browser:
    """tinyshop's shop page in a browser window, drawn at the screenshot's aspect (1366×768).
    labels=False draws a text-free miniature (for thumbnails), safe to scale down."""

    def __init__(self, center=(0, 0, 0), labels=True):
        c = np.array(center, dtype=float)
        self.c = c
        P = lambda xr, yr: c + np.array([xr, yr, 0.0])  # noqa: E731
        self.P = P
        self.box = card(BW, BH, EDGE, "#171410", radius=0.16, stroke=2).move_to(c)
        top = c[1] + BH / 2
        bar_y = top - 0.42
        self.bar = m.Line([c[0] - BW / 2, bar_y, 0], [c[0] + BW / 2, bar_y, 0]).set_stroke(EDGE, 2)
        self.dots = m.VGroup(*[m.Dot(radius=0.06, color=col) for col in (ROSE, SUN, MINT)]).arrange(m.RIGHT, buff=0.12)
        self.dots.move_to([c[0] - BW / 2 + 0.4, top - 0.21, 0])
        chrome = [self.box, self.bar, self.dots]
        if labels:
            chrome.append(code("tinyshop.example/shop", 20, FAINT).move_to([c[0], top - 0.21, 0]))
        # header row: brand, search box, cart
        self.search = m.RoundedRectangle(corner_radius=0.1, width=1.8, height=0.4, stroke_color=EDGE, stroke_width=1.5,
                                         fill_color=CARD, fill_opacity=1).move_to(P(*SEARCH_REL))
        self.cart_box = m.RoundedRectangle(corner_radius=0.2, width=1.15, height=0.4, stroke_color=MUTED, stroke_width=1.5,
                                           fill_color=CARD_HI, fill_opacity=1).move_to(P(2.35, SEARCH_REL[1]))
        page = [self.search, self.cart_box]
        if labels:
            brand = title_text("tinyshop", 24).move_to(P(-2.75, SEARCH_REL[1]), aligned_edge=m.LEFT)
            hint = text("Search", 20, FAINT).move_to(self.search).align_to(self.search, m.LEFT).shift(0.18 * m.RIGHT)
            self.cart_word = code("cart", 20, MUTED).move_to(self.cart_box).shift(0.17 * m.LEFT)
            self.count = code("0", 20, INK, weight=m.BOLD).next_to(self.cart_word, m.RIGHT, buff=0.15)
            page += [brand, hint, self.cart_word]
        else:
            page += [m.RoundedRectangle(corner_radius=0.06, width=1.1, height=0.18, stroke_width=0, fill_color=FAINT,
                                        fill_opacity=1).move_to(P(-2.2, SEARCH_REL[1]))]
            self.count = None
        # product tiles with their buttons
        self.tiles, self.buttons = m.VGroup(), []
        for (name, price), xr in zip(PRODUCTS, TILE_X):
            tile = m.RoundedRectangle(corner_radius=0.12, width=1.85, height=1.12, stroke_width=0, fill_color=CARD_HI,
                                      fill_opacity=1).move_to(P(xr, 0.16))
            btn = m.RoundedRectangle(corner_radius=0.1, width=1.5, height=0.4, stroke_color=MUTED, stroke_width=1.5,
                                     fill_color="#332D25", fill_opacity=1).move_to(P(xr, 0.0))
            parts = [tile, btn]
            if labels:
                row = m.VGroup(text(name, 22, INK, weight=m.BOLD), code(f"${price}", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
                on_baseline(row)
                row.move_to(P(xr, 0.44))
                parts += [row, text("Add", 20, INK, weight=m.SEMIBOLD).move_to(btn)]
            else:
                parts.append(m.RoundedRectangle(corner_radius=0.05, width=0.9, height=0.14, stroke_width=0, fill_color=FAINT,
                                                fill_opacity=1).move_to(P(xr, 0.44)))
            self.tiles.add(m.VGroup(*parts))
            self.buttons.append(btn)
        # total and checkout
        self.checkout = m.RoundedRectangle(corner_radius=0.1, width=1.6, height=0.42, stroke_color=MUTED, stroke_width=1.5,
                                           fill_color="#332D25", fill_opacity=1).move_to(P(*CHECKOUT_REL))
        foot = [self.checkout]
        if labels:
            self.total_word = text("total", 22, MUTED).move_to(P(-2.75, CHECKOUT_REL[1]), aligned_edge=m.LEFT)
            self.amount = title_text("$0", 26).next_to(self.total_word, m.RIGHT, buff=0.2)
            self.amount.shift((self.total_word[0].get_bottom()[1] - self.amount[1].get_bottom()[1]) * m.UP)
            foot += [self.total_word, text("Checkout", 20, INK, weight=m.SEMIBOLD).move_to(self.checkout)]
        else:
            self.amount = None
            foot.append(m.RoundedRectangle(corner_radius=0.05, width=1.0, height=0.16, stroke_width=0, fill_color=FAINT,
                                           fill_opacity=1).move_to(P(-2.2, CHECKOUT_REL[1])))
        self.group = m.VGroup(*chrome, *page, self.tiles, *foot)  # count and amount stay apart: Slots swap them

    def targets(self):
        """Centres of the five clickable elements, in mark order."""
        return [self.search.get_center(), *[b.get_center() for b in self.buttons], self.checkout.get_center()]


class RobotArm:
    """A two-link arm on a base, with a gripper that always points down. Poses are wrist positions
    (inverse kinematics, elbow up); move() returns an animation that interpolates the wrist in a
    straight line and may carry the cup."""

    L1, L2, STEM, FINGER = 1.5, 1.3, 0.2, 0.26

    def __init__(self, shoulder, scale=1.0):
        self.s = np.array(shoulder, dtype=float)
        self.k = scale

    def ik(self, wrist):
        d = (np.array(wrist, dtype=float) - self.s) / self.k
        r = np.hypot(d[0], d[1])
        cos_b = (self.L1 ** 2 + r ** 2 - self.L2 ** 2) / (2 * self.L1 * r)
        assert -1 <= cos_b <= 1, "wrist out of reach"
        base = np.arctan2(d[1], d[0])
        best = None
        for sign in (1, -1):
            a1 = base + sign * np.arccos(cos_b)
            elbow = self.s + self.k * self.L1 * np.array([np.cos(a1), np.sin(a1), 0])
            if best is None or elbow[1] > best[1][1]:
                best = (a1, elbow)
        return best[1]

    def grip_point(self, wrist):
        return np.array(wrist, dtype=float) + self.k * (self.STEM + 0.18) * m.DOWN

    def build(self, wrist, opening=1.0):
        k, w = self.k, np.array(wrist, dtype=float)
        e = self.ik(w)
        link = lambda a, b: m.Line(a, b).set_stroke("#8A8174", 15 * k)  # noqa: E731
        joint = lambda p: m.Circle(radius=0.13 * k).set_fill(CARD, 1).set_stroke(LIME, 3.5 * k).move_to(p)  # noqa: E731
        stem_end = w + k * self.STEM * m.DOWN
        half = k * (0.12 + 0.19 * opening)
        bar = m.Line(stem_end + half * m.LEFT, stem_end + half * m.RIGHT).set_stroke(LIME, 5 * k)
        fingers = m.VGroup(*[m.Line(stem_end + side * half * m.RIGHT, stem_end + side * half * m.RIGHT + k * self.FINGER * m.DOWN)
                             .set_stroke(LIME, 5 * k) for side in (-1, 1)])
        stem = m.Line(w, stem_end).set_stroke(LIME, 5 * k)
        return m.VGroup(link(self.s, e), link(e, w), stem, bar, fingers, joint(self.s), joint(e), joint(w))

    def move(self, arm, start, end, opening=(1.0, 1.0), carry=None, run_time=1.0):
        start, end = np.array(start, dtype=float), np.array(end, dtype=float)

        def update(mob, t):
            mob.become(self.build(start + (end - start) * t, opening[0] + (opening[1] - opening[0]) * t))

        anims = [m.UpdateFromAlphaFunc(arm, update)]
        if carry is not None:
            offset = np.array([0.0, -0.04 * self.k, 0.0])  # the gripped cup's centre, relative to the grip point

            def follow(mob, t):
                mob.move_to(self.grip_point(start + (end - start) * t) + offset)

            anims.append(m.UpdateFromAlphaFunc(carry, follow))
        return m.AnimationGroup(*anims, run_time=run_time)


def red_cup(k=1.0):
    pts = [(-0.22, 0.26), (0.22, 0.26), (0.17, -0.26), (-0.17, -0.26)]
    body = m.Polygon(*[np.array([x * k, y * k, 0.0]) for x, y in pts], stroke_color=ROSE, stroke_width=3 * k,
                     fill_color=ROSE_FILL, fill_opacity=1)
    rim = m.Line(np.array([-0.22 * k, 0.17 * k, 0]), np.array([0.22 * k, 0.17 * k, 0])).set_stroke(ROSE, 2 * k)
    return m.VGroup(body, rim)


class RobotScene:
    """The robot's world: a table, a base, the arm, a red cup and a tray (positions in one place)."""

    def __init__(self, origin=(-4.6, -2.25, 0), scale=1.0, labels=True, table=(-1.9, 2.1)):
        o, k = np.array(origin, dtype=float), scale
        self.k = k
        self.table = m.Line(o + k * table[0] * m.RIGHT, o + k * table[1] * m.RIGHT).set_stroke(EDGE, 4 * k)
        self.base = m.RoundedRectangle(corner_radius=0.08 * k, width=0.8 * k, height=0.45 * k, stroke_color=MUTED,
                                       stroke_width=2 * k, fill_color=CARD_HI, fill_opacity=1).move_to(o + k * 0.225 * m.UP)
        self.arm = RobotArm(o + k * 0.6 * m.UP, scale=k)
        self.cup_x, self.tray_x = o[0] + k * 1.55, o[0] - k * 1.4
        self.cup = red_cup(k).move_to([self.cup_x, o[1] + k * 0.26, 0])
        self.tray = m.RoundedRectangle(corner_radius=0.04 * k, width=0.95 * k, height=0.1 * k, stroke_color=MUTED, stroke_width=2 * k,
                                       fill_color=CARD_HI, fill_opacity=1).move_to([self.tray_x, o[1] + k * 0.05, 0])
        grip = lambda x, cup_center_y: np.array([x, cup_center_y + 0.04 * k + k * (RobotArm.STEM + 0.18), 0])  # noqa: E731
        self.at_cup = grip(self.cup_x, o[1] + k * 0.26)
        self.above_cup = self.at_cup + k * 0.62 * m.UP
        self.at_tray = grip(self.tray_x, o[1] + k * (0.1 + 0.26))
        self.above_tray = self.at_tray + k * 0.52 * m.UP
        self.home = o + k * np.array([0.55, 1.95, 0])
        self.arm_mob = self.arm.build(self.home)
        parts = [self.table, self.tray, self.base, self.cup, self.arm_mob]
        if labels:
            self.cup_label = code("red_cup", 20, MUTED).move_to([self.cup_x, o[1] - 0.32, 0])
            self.tray_label = code("tray", 20, MUTED).move_to([self.tray_x, o[1] - 0.32, 0])
            parts += [self.cup_label, self.tray_label]
        self.group = m.VGroup(*parts)


# ---------------------------------------------------------------------------- the episode


class Ep28ComputerUseVoiceRobots(Episode):
    number = 28

    def lesson(self):
        self.hook()
        self.axes()
        self.computer_use()
        self.between_frames()
        self.grounding()
        self.phone_call()
        self.voice_designs()
        self.robots()
        self.skills()
        self.skeleton()
        self.recap()

    # ------------------------------------------------------------------ beyond text

    def hook(self):
        self.next_section(
            "Beyond text",
            notes="Episode 1: what is outside the observation space does not exist for the model (Li, ch. 1). Episode 27: "
            "a shell and a file system made Otto general, but its world is still text. This episode widens both spaces "
            "with three examples of its own: a browser task, a phone call, a robot picking up a cup.",
        )
        bot = put_head(otto(1.25), [-4.75, 0.95, 0])
        halo = glow(CORAL, radius=2.0).move_to(bot[0])
        sub = text("a general agent · Ep 27", 22, MUTED).next_to(bot, m.DOWN, buff=0.35)
        region = card(6.9, 2.7, MINT, CARD).move_to([2.55, 0.85, 0])
        head = m.VGroup(code("observation space", 22, MINT, weight=m.BOLD), text("text only", 20, MUTED)).arrange(m.RIGHT, buff=0.25)
        head.move_to(region.get_corner(m.UL) + np.array([0.35, -0.42, 0]), aligned_edge=m.LEFT)
        chips = m.VGroup(mono_chip("the request", AZURE, AZURE_FILL), mono_chip("cart.py", MINT, MINT_FILL),
                         mono_chip("pytest output", MINT, MINT_FILL)).arrange(m.RIGHT, buff=0.3)
        chips.move_to(region).shift(0.3 * m.DOWN)

        def outside(label):
            c = word_chip(label, EDGE, CARD, size=22)
            c[1].set_color(MUTED)
            return c

        tag = code("outside", 20, MUTED)
        far = [outside(w) for w in ("a web page", "a voice", "a cup on a table")]
        row = m.VGroup(tag, *far).arrange(m.RIGHT, buff=0.3).move_to([0, -1.45, 0])
        none = text("none of these exist for Otto", 24, ROSE, weight=m.SEMIBOLD).move_to([0, -2.35, 0])
        old = m.VGroup(halo, bot, sub, region, head, chips, row, none)

        # the three examples
        cards = m.VGroup()
        pics = []
        mini = Browser(labels=False).group.scale(2.6 / BW)
        phone = m.VGroup(icon("ep28-phone", 1.35), wave(AZURE, 0.9, 0.7, seed=5)).arrange(m.RIGHT, buff=0.25)
        robot = RobotScene(origin=(0, 0, 0), scale=0.62, labels=False, table=(-2.3, 2.2)).group
        pics = (mini, phone, robot)
        specs = (("A browser task", "sees pixels · clicks and types"), ("A phone call", "hears sound · speaks"),
                 ("A robot", "sees a camera · moves a body"))
        for x, pic, (title, what) in zip((-4.45, 0.0, 4.45), pics, specs):
            box = card(4.0, 3.75, EDGE, CARD).move_to([x, -0.45, 0])
            pic.move_to([x, 0.55, 0])
            name = title_text(title, 28).move_to([x, -0.95, 0])
            line = text(what, 21, MUTED).move_to([x, -1.6, 0])
            cards.add(m.VGroup(box, pic, name, line))
        on_baseline(m.VGroup(*[c[2] for c in cards]))

        self.say(
            "[Last episode, a shell and a file system made Otto a general agent.] [But its world is still text]: [requests, files, and command output.]",
            m.AnimationGroup(self.new_header("beyond text", "Eyes, ears and hands"), m.FadeIn(halo, scale=0.6),
                             m.FadeIn(bot, shift=0.3 * m.UP, scale=0.8), m.FadeIn(sub, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(region), m.FadeIn(head)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.3),
        )
        self.say(
            "[A web page], [a voice], [a cup on a table]: [none of them exist for Otto.]",
            m.AnimationGroup(m.FadeIn(tag), m.FadeIn(far[0], shift=0.15 * m.UP)),
            m.FadeIn(far[1], shift=0.15 * m.UP),
            m.FadeIn(far[2], shift=0.15 * m.UP),
            m.FadeIn(none, shift=0.1 * m.UP),
        )
        self.say(
            "[This episode widens both spaces, with three examples of its own]: [a browser task], [a phone call], [and a robot picking up a cup.]",
            m.FadeOut(old, shift=0.2 * m.UP),
            m.FadeIn(cards[0], shift=0.25 * m.UP),
            m.FadeIn(cards[1], shift=0.25 * m.UP),
            m.FadeIn(cards[2], shift=0.25 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ two axes

    def axes(self):
        self.next_section(
            "01 · Two axes",
            notes="Li, ch. 6: observation and action spaces expand along two dimensions, modality and timing. Timescales from "
            "Li's table: async and event-driven seconds–days, voice 10 ms–1 s, computer use sub-second–seconds, robotics "
            "milliseconds. Drawn on a log scale; band edges read from Li's words ('sub-second' = 0.5 s, 'seconds' = 10 s, "
            "'days' = 2 days; robotics = the 50–1000 Hz control loop, 1–20 ms). Quote verbatim from Li, ch. 6. Async work "
            "is Episode 23. Fact date: October 2026.",
        )
        source = top_cite(SOURCE)
        rows_y = (1.55, 0.55, -0.45, -1.45)
        mod_label = code("modality", 20, MUTED).move_to([-6.55, 2.25, 0], aligned_edge=m.LEFT)
        rule = m.Line([-2.75, 2.05, 0], [-2.75, -1.85, 0]).set_stroke(EDGE, 2)
        axis = m.Arrow([AX_L - 0.2, -2.05, 0], [AX_R + 0.28, -2.05, 0], buff=0, color=MUTED, stroke_width=3,
                       max_tip_length_to_length_ratio=0.03)
        time_label = code("timescale (log)", 20, MUTED).move_to([AX_R, -1.73, 0], aligned_edge=m.RIGHT)
        labels = []
        for (title, sub, *_), y in zip(BANDS, rows_y):
            g = m.VGroup(text(title, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            labels.append(g.move_to([-6.55, y, 0], aligned_edge=m.LEFT))
        ticks = m.VGroup()
        for sec, lab in TICKS:
            x = tx(sec)
            ticks.add(m.VGroup(m.Line([x, -1.97, 0], [x, -2.13, 0]).set_stroke(MUTED, 2), code(lab, 20, MUTED).move_to([x, -2.42, 0])))
        bands, band_labels = [], []
        for (_, _, lo, hi, lab), y in zip(BANDS, rows_y):
            b = m.RoundedRectangle(corner_radius=0.12, width=tx(hi) - tx(lo), height=0.5, stroke_color=LIME, stroke_width=2.5,
                                   fill_color=LIME_FILL, fill_opacity=1).move_to([(tx(lo) + tx(hi)) / 2, y, 0])
            words = code(lab, 20, LIME)
            if words.width + 0.4 < b.width:
                words.move_to(b)
            else:
                words.next_to(b, m.RIGHT, buff=0.2)
            bands.append(b)
            band_labels.append(words)
        quote = text("“Turn-taking is an interaction convention of the model and its interface, not a property of the environment.”",
                     22, INK, slant=m.ITALIC)
        if quote.width > 12.6:
            quote.scale(12.6 / quote.width)
        quote.move_to([0, -3.1, 0])

        self.say(
            "[The book AI Agents in Depth stretches both spaces along two axes]: [modality], [and timing.]",
            m.AnimationGroup(self.new_header("01 · two axes", "Modality and timing"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(mod_label, shift=0.1 * m.DOWN), m.Create(rule)),
            m.AnimationGroup(m.GrowArrow(axis), m.FadeIn(time_label)),
        )
        self.say(
            "Modality is what the agent sees and does: [text], [a screen], [sound], [or a body.]",
            *[m.FadeIn(g, shift=0.15 * m.RIGHT) for g in labels],
        )
        self.say(
            "[Timing is how fast the world moves.] [Asynchronous agents work over seconds to days]; [computer use, from under a second to a few seconds.]",
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in ticks], lag_ratio=0.12),
            m.AnimationGroup(m.GrowFromEdge(bands[0], m.LEFT), m.FadeIn(band_labels[0])),
            m.AnimationGroup(m.GrowFromEdge(bands[1], m.LEFT), m.FadeIn(band_labels[1])),
        )
        self.say(
            "[A voice call lives between ten milliseconds and a second], [and a robot's control loop runs in milliseconds.]",
            m.AnimationGroup(m.GrowFromEdge(bands[2], m.LEFT), m.FadeIn(band_labels[2])),
            m.AnimationGroup(m.GrowFromEdge(bands[3], m.LEFT), m.FadeIn(band_labels[3])),
        )
        self.say(
            "[In a chat, the world waits for the model's turn.] [But turn-taking is a convention of the model and its interface], [not a property of the world.]",
            m.Circumscribe(m.VGroup(labels[0], bands[0]), color=LIME, buff=0.1),
            m.FadeIn(quote, shift=0.15 * m.UP),
            m.Circumscribe(glyphs(quote, "not a property of the environment"), color=LIME, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ computer use

    def computer_use(self):
        self.next_section(
            "02 · Computer use",
            notes=f"Li, ch. 6: the agent takes a screenshot, the model decides, the harness acts, waits for the interface to "
            f"respond and takes another screenshot. Anthropic's computer-use docs (October 2026): a tool_use block named "
            f"'left_click' in the 'computer' toolset, input {{'coordinate': [x, y]}} in screenshot pixels. The page is illustrative and drawn at 1366×768's aspect: "
            f"the mug's 'Add' button is computed at {MUG_PX}, the tea's at {TEA_PX}; ${PRODUCTS[0][1]} + ${PRODUCTS[1][1]} = "
            f"${TOTAL}. The same loop as Episode 4 (think, act, observe). Fact date: October 2026.",
        )
        br = Browser(center=(-3.35, 0.5, 0))
        ptr = point_at(cursor(), br.P(-0.95, -0.75))
        task = message("user", "Add a mug and a tea to the cart, then check the total.", width=BW, size=22)
        task.move_to([-3.35, -2.35, 0])
        ring_c = np.array([3.55, 0.95, 0])
        ring = loop_ring(["screenshot", "model", "action", "wait"], radius=1.25, colors=[MINT, CORAL, SUN, MUTED], size=22, center=ring_c)
        nodes, arcs = ring
        bot = put_head(otto(0.5), ring_c + 0.08 * m.DOWN)
        thumb = Browser(labels=False).group.scale(0.95 / BW).move_to([6.0, ring_c[1], 0])
        mug_pt, tea_pt = br.buttons[0].get_center(), br.buttons[1].get_center()

        def click_card(px):
            return jcard('{"name": "left_click",\n "toolset_name": "computer",\n "input": {"coordinate": [%d, %d]}}' % px).move_to([3.55, -1.95, 0])

        reply = Slot(self)
        count, amount = Slot(self, br.count), Slot(self, br.amount)

        def new_count(n):
            return code(str(n), 20, INK, weight=m.BOLD).move_to(br.count)

        def new_amount(v, color=INK):
            a = title_text(f"${v}", 26, color).move_to(br.amount, aligned_edge=m.LEFT)
            return a

        aim1 = crosshair(mug_pt)
        aim2 = crosshair(tea_pt)

        self.say(
            "[First, the browser task.] [Otto gets a browser]: [add a mug and a tea to the cart, then check the total.]",
            self.new_header("02 · computer use", "Screenshot, think, click"),
            m.AnimationGroup(m.FadeIn(br.group, shift=0.2 * m.UP), m.FadeIn(br.count, shift=0.2 * m.UP),
                             m.FadeIn(br.amount, shift=0.2 * m.UP), m.FadeIn(ptr)),
            m.AnimationGroup(m.FadeIn(task[0]), m.FadeIn(task[1]), m.FadeIn(task[2][0]),
                             m.AddTextLetterByLetter(task[2][1], time_per_char=0.025)),
        )
        self.say(
            "[Each turn, the harness takes a screenshot] [and sends it to the model, as an image.]",
            m.AnimationGroup(m.FadeIn(ring, scale=0.9), m.FadeIn(bot, scale=0.7), shutter(br.box)),
            m.AnimationGroup(m.FadeIn(thumb, target_position=br.box.get_center(), scale=0.6), arc_flash(arcs[0], MINT, 0.9)),
        )
        self.say(
            "[The model answers with an action]: [a left click, at this point on the image.]",
            m.AnimationGroup(m.FadeOut(thumb, target_position=nodes[1].get_center(), scale=0.4),
                             m.Indicate(nodes[1], color=CORAL, scale_factor=1.12), arc_flash(arcs[1], CORAL, 0.9)),
            m.AnimationGroup(reply.to(click_card(MUG_PX)), m.FadeIn(aim1, scale=1.5)),
        )
        self.say(
            "[The harness moves the mouse and clicks], [waits for the page to respond], [and takes the next screenshot.]",
            m.AnimationGroup(
                m.Indicate(nodes[2], color=SUN, scale_factor=1.12),
                m.Succession(glide(ptr, mug_pt),
                             m.AnimationGroup(m.Indicate(br.buttons[0], color=SUN, scale_factor=1.08),
                                              m.Flash(mug_pt, color=SUN, line_length=0.14, flash_radius=0.45),
                                              m.FadeOut(aim1), count.to(new_count(1)), amount.to(new_amount(PRODUCTS[0][1]))),
                             glide(ptr, mug_pt + REST, 0.3)),
            ),
            m.AnimationGroup(arc_flash(arcs[2], SUN, 0.8), m.Indicate(nodes[3], color=MUTED, scale_factor=1.12)),
            m.AnimationGroup(arc_flash(arcs[3], MUTED, 0.8), shutter(br.box), m.Indicate(nodes[0], color=MINT, scale_factor=1.12)),
        )
        self.say(
            "[It's the loop from episode four: observe, think, act.] [On the next turn, the model clicks the tea], [and the total reads seven dollars.]",
            m.Succession(*[arc_flash(a, c, 0.45) for a, c in zip(arcs, (MINT, CORAL, SUN, MUTED))]),
            m.AnimationGroup(
                reply.to(click_card(TEA_PX)),
                m.Succession(m.FadeIn(aim2, scale=1.5, run_time=0.5), glide(ptr, tea_pt),
                             m.AnimationGroup(m.Indicate(br.buttons[1], color=SUN, scale_factor=1.08),
                                              m.Flash(tea_pt, color=SUN, line_length=0.14, flash_radius=0.45),
                                              m.FadeOut(aim2), count.to(new_count(2))),
                             glide(ptr, tea_pt + REST, 0.3)),
            ),
            m.AnimationGroup(amount.to(new_amount(TOTAL, MINT)),
                             m.Circumscribe(m.VGroup(br.total_word, amount.current), color=MINT, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ between frames

    def between_frames(self):
        self.next_section(
            "02b · Between frames",
            notes=f"Li, ch. 6 (verbatim): 'An Agent that opens its eyes only once every 3–5 seconds and has no ears cannot see "
            f"or hear what happens between two frames.' For example: a screenshot every {FRAME_EVERY} s (frames at {FRAMES} s) "
            f"and a pop-up shown from second {POPUP[0]} to {POPUP[1]}, which no frame catches (asserted).",
        )
        x0, x1, t_end = -3.65, 5.75, FRAMES[-1]
        X = lambda t: x0 + (x1 - x0) * t / t_end  # noqa: E731
        screen_y, frame_y = 1.3, -0.45
        lane = m.RoundedRectangle(corner_radius=0.1, width=x1 - x0, height=0.3, stroke_color=EDGE, stroke_width=2,
                                  fill_color=CARD_HI, fill_opacity=1).move_to([(x0 + x1) / 2, screen_y, 0])
        lane_label = code("the screen", 20, MUTED).move_to([-6.5, screen_y, 0], aligned_edge=m.LEFT)
        seen_label = m.VGroup(code("what the", 20, MUTED), code("model sees", 20, MUTED)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        seen_label.move_to([-6.5, frame_y, 0], aligned_edge=m.LEFT)
        pop = m.Rectangle(width=X(POPUP[1]) - X(POPUP[0]), height=0.3, stroke_width=0, fill_color=ROSE, fill_opacity=1)
        pop.move_to([(X(POPUP[0]) + X(POPUP[1])) / 2, screen_y, 0])
        pop_label = text("a pop-up, there and gone", 20, ROSE, weight=m.SEMIBOLD).next_to(pop, m.UP, buff=0.4)
        frames, stamps, links = [], [], []
        for f in FRAMES:
            fr = Browser(labels=False).group.scale(1.6 / BW).move_to([X(f), frame_y, 0])
            frames.append(fr)
            stamps.append(code(f"{f} s", 20, MUTED).next_to(fr, m.DOWN, buff=0.15))
            links.append(m.DashedLine([X(f), fr.get_top()[1] + 0.05, 0], [X(f), screen_y - 0.2, 0], dash_length=0.08).set_stroke(MINT, 2))
        missed = text("never seen", 22, ROSE, weight=m.BOLD).move_to([(X(POPUP[0]) + X(POPUP[1])) / 2, screen_y - 0.5, 0])
        head = m.Line([x0, screen_y - 0.32, 0], [x0, screen_y + 0.32, 0]).set_stroke(INK, 4).set_z_index(3)
        stamp_note = cite("for example, one screenshot every four seconds").move_to([(x0 + x1) / 2, -1.7, 0])
        quote = m.VGroup(
            text("“An Agent that opens its eyes only once every 3–5 seconds and has no ears", 24, INK, slant=m.ITALIC),
            text("cannot see or hear what happens between two frames.”", 24, INK, slant=m.ITALIC),
        ).arrange(m.DOWN, buff=0.12).move_to([0, -2.55, 0])
        q_cite = cite(SOURCE).next_to(quote, m.DOWN, buff=0.15)

        def sample(i):
            return m.AnimationGroup(m.FadeIn(frames[i], shift=0.15 * m.UP), m.FadeIn(stamps[i]), m.Create(links[i]), run_time=0.45)

        self.say(
            "[But this loop is slow.] [It looks only every few seconds], [and a pop-up that comes and goes in between] [is never seen.]",
            m.AnimationGroup(self.new_header("02 · computer use", "What happens between frames"), m.FadeIn(lane), m.FadeIn(lane_label),
                             m.FadeIn(seen_label), m.FadeIn(head), m.FadeIn(stamp_note)),
            m.Succession(sample(0), head.animate(run_time=0.9, rate_func=m.linear).set_x(X(FRAMES[1])), sample(1)),
            m.Succession(head.animate(run_time=0.25, rate_func=m.linear).set_x(X(POPUP[0])),
                         m.AnimationGroup(head.animate(rate_func=m.linear).set_x(X(POPUP[1])), m.GrowFromEdge(pop, m.LEFT),
                                          m.FadeIn(pop_label), run_time=0.6),
                         head.animate(run_time=0.25, rate_func=m.linear).set_x(X(FRAMES[2])), sample(2)),
            m.Succession(head.animate(run_time=0.8, rate_func=m.linear).set_x(X(FRAMES[3])), sample(3),
                         m.FadeIn(missed, scale=1.2, run_time=0.5)),
        )
        self.say(
            "[As the book puts it], [an agent that opens its eyes only once every three to five seconds cannot see what happens between two frames.]",
            m.AnimationGroup(m.FadeOut(stamp_note), m.FadeIn(q_cite)),
            m.FadeIn(quote, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ grounding

    def grounding(self):
        self.next_section(
            "03 · Grounding",
            notes=f"Li, ch. 6 'visual grounding': numbered marks (Set-of-Mark prompting, Yang et al., 2023) turn localisation into "
            f"a multiple-choice problem ('easier ... than fill-in-the-blank'): here 5 marks instead of any of {PIXELS:,} pixels "
            f"(1366 × 768, computed); structured element indexing (the DOM lines are illustrative); pure coordinate prediction. "
            f"Scaling (Li's example): a {SCREEN[0]}×{SCREEN[1]} screen shown as {SHOWN[0]}×{SHOWN[1]}; the model's {MODEL_PT} "
            f"maps back to {REAL_PT} (× {SCALE[0]:.3f}, × {SCALE[1]:.3f}). Anthropic's docs (Oct 2026): coordinates are in "
            f"screenshot pixels; scale them back up if you shrank the screenshot. The click/mark/index replies are illustrative "
            f"in the same tool_use shape. Both screens drawn at the same scale. Fact date: October 2026.",
        )
        br = Browser(center=(-3.35, 0.35, 0))
        tgt = br.targets()
        marks = m.VGroup()
        for i, p in enumerate(tgt):
            if i == 0:
                at = p + np.array([-1.15, 0, 0])
            elif i == 4:
                at = p + np.array([-1.05, 0, 0])
            else:
                at = p + np.array([-0.8, 0.38, 0])
            dot = m.Circle(radius=0.19).set_fill(SUN, 1).set_stroke(BG, 2).move_to(at)
            marks.add(m.VGroup(dot, code(str(i + 1), 20, BG, weight=m.BOLD).move_to(at)).set_z_index(4))
        tab_names = ("1 · marks", "2 · index", "3 · coordinates")
        tabs = m.VGroup(*[code(t, 20, MUTED) for t in tab_names]).arrange(m.RIGHT, buff=0.6).move_to([3.45, 2.05, 0])
        under = [m.Line(t.get_corner(m.DL) + 0.12 * m.DOWN, t.get_corner(m.DR) + 0.12 * m.DOWN).set_stroke(SUN, 4) for t in tabs]
        panel = card(6.2, 3.35, EDGE, CARD).move_to([3.45, 0.05, 0])
        says = lambda: text("the model answers", 20, MUTED)  # noqa: E731

        def answer(src, size=21):
            return code(src, size, INK, t2c={**JSON_T2C})

        a1 = m.VGroup(says(), answer('{"name": "click",\n "input": {"mark": 3}}')).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to(panel)
        dom = m.VGroup(*[code(line, 19, INK, t2c={r"^\[\d\]": SUN, r"\"[^\"]*\"": MINT}) for line in DOM])
        dom.arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
        dom_reply = answer('{"name": "click",\n "input": {"index": 3}}')
        a2 = m.VGroup(dom, dom_reply).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).move_to(panel)
        hl = m.Rectangle(width=panel.width - 0.3, height=dom[2].height + 0.16, stroke_width=0, fill_color=SUN, fill_opacity=0.16)
        hl.move_to([panel.get_x(), dom[2].get_y(), 0]).set_z_index(1)
        dom.set_z_index(2)
        a3 = m.VGroup(says(), answer('{"name": "left_click",\n "input": {"coordinate": [%d, %d]}}' % MODEL_PT, 20),
                      text("a point on the image, in pixels", 20, MUTED)).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).move_to(panel)
        aim = crosshair(br.buttons[1].get_center())
        method = Slot(self)
        quote = text("“Multiple-choice questions are easier to answer correctly than fill-in-the-blank.”", 24, INK, slant=m.ITALIC)
        quote.move_to([0, -2.35, 0])
        q_note = m.VGroup(cite(f"5 marks to choose from, instead of any of {PIXELS:,} pixels ·"), cite("Li, ch. 6 · Yang et al., 2023"))
        q_note.arrange(m.RIGHT, buff=0.15).next_to(quote, m.DOWN, buff=0.2)
        source = top_cite(SOURCE)
        under_slot = Slot(self)

        self.say(
            "[Next, grounding.] [The model sees pixels], [but it must say exactly where to click.]",
            m.AnimationGroup(self.new_header("03 · grounding", "Where exactly is the button?"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(br.group, shift=0.2 * m.UP), m.FadeIn(br.count, shift=0.2 * m.UP), m.FadeIn(br.amount, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(panel), m.FadeIn(tabs)),
        )
        self.say(
            "[One way: the harness draws numbered marks on the screenshot], [and the model just picks a number.]",
            m.AnimationGroup(under_slot.to(under[0], shift=0), m.LaggedStart(*[m.FadeIn(mk, scale=1.4) for mk in marks], lag_ratio=0.2)),
            m.AnimationGroup(method.to(a1), m.Circumscribe(marks[TEA_MARK - 1], color=SUN, buff=0.06)),
        )
        self.say(
            "[As the book says], [multiple choice is easier than fill-in-the-blank.]",
            m.FadeIn(q_note),
            m.FadeIn(quote, shift=0.15 * m.UP),
        )
        self.say(
            "[Another way skips the pixels]: [the harness lists the page's elements], [and the model picks one by index.]",
            m.AnimationGroup(m.FadeOut(marks), m.FadeOut(quote), m.FadeOut(q_note), under_slot.to(under[1], shift=0), method.to(None)),
            m.LaggedStart(*[m.FadeIn(d, shift=0.1 * m.RIGHT) for d in dom], lag_ratio=0.2),
            m.AnimationGroup(m.FadeIn(hl), m.FadeIn(dom_reply, shift=0.1 * m.UP)),
        )
        self.say(
            "[The third way is raw coordinates]: [the model names a point], [the way a person points.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(dom, dom_reply, hl)), under_slot.to(under[2], shift=0)),
            method.to(a3),
            m.FadeIn(aim, scale=1.5),
        )
        # scaling: both screens at the same scale
        s = 6.0 / SCREEN[0]
        small = Browser(labels=False).group.scale(SHOWN[0] * s / BW).move_to([-4.45, 0.25, 0])
        big = Browser(labels=False).group.scale(SCREEN[0] * s / BW).move_to([3.0, 0.25, 0])
        for g, w in ((small, SHOWN[0] * s), (big, SCREEN[0] * s)):
            assert abs(g.width - w) < 0.02
        lab_s = m.VGroup(text("what the model sees", 22, INK, weight=m.BOLD), code(f"{SHOWN[0]} × {SHOWN[1]}", 22, SUN)).arrange(m.RIGHT, buff=0.25)
        lab_s.next_to(small, m.UP, buff=0.3)
        lab_b = m.VGroup(text("the real screen", 22, INK, weight=m.BOLD), code(f"{SCREEN[0]} × {SCREEN[1]}", 22, SUN)).arrange(m.RIGHT, buff=0.25)
        lab_b.next_to(big, m.UP, buff=0.3)
        ps, pb = small.get_center(), big.get_center()
        dot_s = m.Dot(ps, radius=0.08, color=SUN).set_z_index(5)
        dot_b = m.Dot(pb, radius=0.1, color=SUN).set_z_index(5)
        pt_s = code(f"({MODEL_PT[0]}, {MODEL_PT[1]})", 22, SUN).next_to(small, m.DOWN, buff=0.25)
        pt_b = code(f"({REAL_PT[0]}, {REAL_PT[1]})", 22, SUN).next_to(big, m.DOWN, buff=0.25)
        bridge = m.CurvedArrow(ps + np.array([0.15, 0.1, 0]), pb + np.array([-0.15, 0.12, 0]), angle=-0.7, color=SUN,
                               stroke_width=4, tip_length=0.2).set_z_index(5)
        fx = code(f"x: {MODEL_PT[0]} × {SCREEN[0]} / {SHOWN[0]} = {REAL_PT[0]}", 22, INK, t2c={r"\d+$": SUN})
        fy = code(f"y: {MODEL_PT[1]} × {SCREEN[1]} / {SHOWN[1]} = {REAL_PT[1]}", 22, INK, t2c={r"\d+$": SUN})
        formulas = m.VGroup(fx, fy).arrange(m.DOWN, buff=0.15, aligned_edge=m.LEFT).move_to([-4.45, -2.55, 0])
        harness_note = m.VGroup(text("the harness does the arithmetic", 22, MUTED),
                                cite("Li, ch. 6 · Anthropic computer-use docs")).arrange(m.DOWN, buff=0.12).move_to([3.0, -2.75, 0])
        old = m.VGroup(br.group, br.count, br.amount, aim, panel, tabs)

        self.say(
            "[But the model saw a shrunken copy of the screen], [so the harness scales every point back up.]",
            m.AnimationGroup(m.FadeOut(old), method.to(None), under_slot.to(None, shift=0),
                             m.FadeIn(small, target_position=br.box.get_center()), m.FadeIn(lab_s, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(big, scale=0.55), m.FadeIn(lab_b, shift=0.1 * m.DOWN)),
        )
        self.say(
            "[The centre of the small image] [lands on the centre of the real screen]: [the harness does that arithmetic, not the model.]",
            m.AnimationGroup(m.FadeIn(dot_s, scale=2), m.FadeIn(pt_s, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(bridge), m.FadeIn(dot_b, scale=2), m.FadeIn(pt_b, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(f, shift=0.1 * m.UP) for f in formulas], lag_ratio=0.4), m.FadeIn(harness_note)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ voice: the cascade

    def phone_call(self):
        self.next_section(
            "04 · Voice",
            notes="Li, ch. 6: the cascaded pipeline VAD → ASR → LLM → TTS: 'every boundary can add waiting time'; 'Latency "
            "accumulates and paralinguistic information is lost at interfaces'. Li gives NO millisecond budget: the timeline "
            "shows structure only, in illustrative units (marked on screen). The call and its words are illustrative. "
            "Fact date: October 2026.",
        )
        y = 1.45
        caller = icon("user", 0.85).move_to([-5.95, y, 0])
        specs = (("VAD", "turn over?"), ("ASR", "speech → text"), ("LLM", "writes a reply"), ("TTS", "text → speech"))
        nodes = []
        for x, (lab, sub) in zip((-3.95, -1.35, 1.25, 3.85), specs):
            color, fill = (CORAL, CORAL_FILL) if lab == "LLM" else (MUTED, CARD)
            box = card(1.9, 1.25, color, fill)
            body = m.VGroup(code(lab, 24, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1).move_to(box)
            nodes.append(m.VGroup(box, body).move_to([x, y, 0]))
        out = wave(CORAL, 0.8, 0.6, seed=11).move_to([5.95, y, 0])
        chain = [caller, *nodes, out]
        arrows = [arrow(a.get_right(), b.get_left(), MUTED, buff=0.1) for a, b in zip(chain, chain[1:])]
        ask_words = text("Hi, can I still change my delivery address?", 26, INK)
        ask = bubble(ask_words, AZURE, AZURE_FILL, tail="left")
        ask.next_to(caller, m.RIGHT, buff=0.35).shift(0.35 * m.UP)
        source = top_cite(SOURCE)

        # the timeline
        lane_y = {name: LANE_Y0 - i * LANE_DY for i, name in enumerate(LANES)}
        lane_labels = m.VGroup(*[code(n, 20, MUTED).move_to([-6.55, lane_y[n], 0], aligned_edge=m.LEFT) for n in LANES])
        colors = {"caller": AZURE, "VAD": MUTED, "ASR": MUTED, "LLM": CORAL, "TTS": MUTED, "reply": CORAL}

        def bar(name, span, opacity=1.0):
            a, b = span
            return m.RoundedRectangle(corner_radius=0.07, width=G_U * (b - a), height=0.24, stroke_width=0,
                                      fill_color=colors[name], fill_opacity=opacity).move_to([(gx(a) + gx(b)) / 2, lane_y[name], 0])

        bars = {n: bar(n, CASCADE[n]) for n in LANES}
        sil_y = lane_y["reply"] - 0.42

        def gap(span):
            a, b = gx(span[0]), gx(span[1])
            line = m.VGroup(m.Line([a, sil_y, 0], [b, sil_y, 0]).set_stroke(ROSE, 4),
                            m.Line([a, sil_y - 0.1, 0], [a, sil_y + 0.1, 0]).set_stroke(ROSE, 4),
                            m.Line([b, sil_y - 0.1, 0], [b, sil_y + 0.1, 0]).set_stroke(ROSE, 4))
            label = text("silence", 22, ROSE, weight=m.SEMIBOLD).move_to([(a + b) / 2, sil_y - 0.3, 0])
            return m.VGroup(line, label)

        hush = gap((CASCADE["caller"][1], CASCADE["reply"][0]))
        note = cite("illustrative timing, not measured").move_to([6.5, lane_y["caller"], 0], aligned_edge=m.RIGHT)
        tone = pill("worried tone", AZURE, AZURE_FILL, 20).move_to([-5.45, 0.3, 0])
        transcript = code("“can I still change my delivery address?”", 20, INK).move_to([1.3, 0.3, 0])
        lost = text("tone lost", 22, ROSE, weight=m.BOLD).move_to([-3.45, 0.3, 0])

        self.say(
            "[Now, the phone call.] [A customer rings tinyshop]: [hi, can I still change my delivery address?]",
            m.AnimationGroup(self.new_header("04 · voice", "A phone call, stage by stage"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(caller, shift=0.2 * m.UP), m.FadeIn(ask[:2], shift=0.2 * m.UP)),
            m.AddTextLetterByLetter(ask_words, time_per_char=0.03),
        )
        self.say(
            "[The classic design is a cascade]: [voice activity detection], [speech recognition], [a language model], [and text to speech.]",
            m.AnimationGroup(m.FadeOut(ask, shift=0.2 * m.UP), m.GrowArrow(arrows[0])),
            m.AnimationGroup(m.FadeIn(nodes[0], shift=0.15 * m.UP), m.GrowArrow(arrows[1])),
            m.AnimationGroup(m.FadeIn(nodes[1], shift=0.15 * m.UP), m.GrowArrow(arrows[2])),
            m.AnimationGroup(m.FadeIn(nodes[2], shift=0.15 * m.UP), m.GrowArrow(arrows[3])),
            m.AnimationGroup(m.FadeIn(nodes[3], shift=0.15 * m.UP), m.GrowArrow(arrows[4]), m.FadeIn(out, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[Each stage waits for the one before it], [so the delays add up], [and the caller hears silence.]",
            m.AnimationGroup(m.FadeIn(lane_labels), m.GrowFromEdge(bars["caller"], m.LEFT), m.Succession(m.Wait(0.4), m.GrowFromEdge(bars["VAD"], m.LEFT))),
            m.Succession(*[m.GrowFromEdge(bars[n], m.LEFT, run_time=0.5) for n in ("ASR", "LLM", "TTS")]),
            m.AnimationGroup(m.GrowFromEdge(bars["reply"], m.LEFT), m.FadeIn(hush, shift=0.1 * m.UP), m.FadeIn(note)),
        )
        self.say(
            "[Something is lost at each boundary, too]: [the caller sounded worried], [but the transcript is only words.]",
            m.LaggedStart(*[pulse(a, ROSE) for a in arrows], lag_ratio=0.25),
            m.FadeIn(tone, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(transcript, shift=0.1 * m.RIGHT),
                             m.Succession(tone.animate(run_time=0.6).move_to([-3.6, 0.3, 0]), m.FadeOut(tone, scale=0.6, run_time=0.4),
                                          m.FadeIn(lost, scale=1.2, run_time=0.4))),
        )
        # streaming: the same picture, the bars slide
        self.next_section(
            "05 · Streaming",
            notes="Li, ch. 6: streaming ASR; 'the provisional transcript is sent to the LLM as soon as it exists' (speculative; "
            "discarded if the final text differs); 'the first speakable sentence goes to TTS without waiting for the full reply'. "
            "Same illustrative units; the reply is as long as before, the silence shorter (asserted). Episode 7: time to first token.",
        )
        new = {n: bar(n, STREAM[n]) for n in LANES}
        spec_label = text("speculative", 20, CORAL, weight=m.SEMIBOLD).move_to([gx(STREAM["LLM"][0]) - 0.15, lane_y["LLM"], 0], aligned_edge=m.RIGHT)
        hush2 = gap((STREAM["caller"][1], STREAM["reply"][0]))
        ttft = cite("like time to first token · Ep 7").next_to(hush2[1], m.RIGHT, buff=0.35)
        self.say(
            "[Streaming hides much of that wait.] [Recognition writes words while the caller is still talking], [and the model starts on that provisional transcript.]",
            m.AnimationGroup(self.new_header("05 · streaming", "Don't wait for the whole turn"), m.FadeOut(lost)),
            m.AnimationGroup(m.Transform(bars["ASR"], new["ASR"]), m.Transform(bars["VAD"], new["VAD"])),
            m.AnimationGroup(m.Transform(bars["LLM"], new["LLM"]), m.FadeIn(spec_label, shift=0.1 * m.RIGHT)),
        )
        self.say(
            "[The first sentence goes to speech right away.] [As with time to first token in episode seven], [what the caller feels is the wait for the first word.]",
            m.AnimationGroup(m.Transform(bars["TTS"], new["TTS"]), m.Transform(bars["reply"], new["reply"])),
            m.AnimationGroup(m.Transform(hush, hush2), m.FadeIn(ttft, shift=0.1 * m.LEFT)),
            m.Circumscribe(hush, color=ROSE, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ voice: three designs

    def voice_designs(self):
        self.next_section(
            "06 · Voice designs",
            notes="Li, ch. 6: end-to-end 'Omni' models listen, reply and speak with one model, yet 'still assume turn-taking'; "
            "full-duplex models 'listen and speak continuously' and turn-taking becomes a continuous model decision (e.g. "
            "Moshi, Kyutai 2024); fast/slow: a fast talker sustains the conversation while a slow reasoner advises in the "
            "background ('through a status bar or dedicated interface'). Fact date: October 2026.",
        )
        source = top_cite(SOURCE)
        xs = (-4.45, 0.0, 4.45)
        cy = -0.3
        boxes = [card(4.15, 4.5, EDGE, CARD).move_to([x, cy, 0]) for x in xs]
        titles = [title_text(t, 30).move_to([x, cy + 0.0, 0]) for t, x in zip(("End-to-end", "Full-duplex", "Fast + slow"), xs)]
        on_baseline(m.VGroup(*titles))

        def mark_row(ok, words, x, y):
            mk = icon("check" if ok else "ep01-cross", 0.36)
            return m.VGroup(mk, text(words, 22, INK)).arrange(m.RIGHT, buff=0.18).move_to([x, y, 0])

        rows = [
            (mark_row(True, "tone survives", xs[0], cy - 0.8), mark_row(False, "still takes turns", xs[0], cy - 1.4)),
            (mark_row(True, "listens while it talks", xs[1], cy - 0.8), mark_row(True, "decides when to speak", xs[1], cy - 1.4)),
            (mark_row(True, "keeps the call flowing", xs[2], cy - 0.8), mark_row(True, "reasons, then advises", xs[2], cy - 1.4)),
        ]
        for r1, r2 in rows:
            left = min(r1.get_left()[0], r2.get_left()[0])
            r1.align_to([left, 0, 0], m.LEFT)
            r2.align_to([left, 0, 0], m.LEFT)
        dy = cy + 1.15
        # 1 · one model
        one = word_chip("one model", CORAL, CORAL_FILL, size=22)
        d1 = m.VGroup(wave(AZURE, 0.55, 0.5, seed=2), m.Arrow(m.LEFT * 0.17, m.RIGHT * 0.17, buff=0, color=MUTED, stroke_width=4,
                      max_tip_length_to_length_ratio=0.45), one,
                      m.Arrow(m.LEFT * 0.17, m.RIGHT * 0.17, buff=0, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.45),
                      wave(CORAL, 0.55, 0.5, seed=7)).arrange(m.RIGHT, buff=0.12).move_to([xs[0], dy, 0])
        # 2 · two lanes at once
        lane_w = 2.9

        def lane(label, color, spans, y):
            name = code(label, 20, color).move_to([xs[1] - lane_w / 2 - 0.05, y, 0], aligned_edge=m.RIGHT)
            segs = m.VGroup(*[m.RoundedRectangle(corner_radius=0.06, width=lane_w * (b - a), height=0.2, stroke_width=0, fill_color=color,
                                                 fill_opacity=1).move_to([xs[1] - lane_w / 2 + lane_w * (a + b) / 2 + 0.25, y, 0]) for a, b in spans])
            return m.VGroup(name, segs)

        d2 = m.VGroup(lane("in", AZURE, ((0.0, 0.45), (0.55, 0.8), (0.88, 1.0)), dy + 0.22),
                      lane("out", CORAL, ((0.3, 0.42), (0.62, 0.95)), dy - 0.22))
        # 3 · fast talker + slow thinker
        fast = word_chip("fast talker", CORAL, CORAL_FILL, size=20, pad=0.36)
        slow = word_chip("slow thinker", CORAL, CARD, size=20, pad=0.36)
        pair = m.VGroup(fast, slow).arrange(m.RIGHT, buff=0.5).move_to([xs[2], dy + 0.15, 0])
        advice = m.Arrow(slow.get_left(), fast.get_right(), buff=0.05, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.4)
        advice_label = code("advice", 20, MUTED).move_to([advice.get_x(), dy - 0.42, 0])
        moshi = cite("e.g. Moshi · Kyutai, 2024").move_to([xs[1], cy - 1.95, 0])

        self.say(
            "[End-to-end models hear and speak with one network], [so tone survives], [but they still take turns.]",
            m.AnimationGroup(self.new_header("06 · voice designs", "Three ways to talk"), m.FadeIn(source),
                             m.FadeIn(boxes[0]), m.FadeIn(d1, shift=0.1 * m.RIGHT), m.FadeIn(titles[0])),
            m.FadeIn(rows[0][0], shift=0.1 * m.UP),
            m.FadeIn(rows[0][1], shift=0.1 * m.UP),
        )
        self.say(
            "[Full-duplex models listen while they talk], [and decide, moment to moment, when to speak.]",
            m.AnimationGroup(m.FadeIn(boxes[1]), m.FadeIn(titles[1]), m.FadeIn(d2[0][0]), m.FadeIn(d2[1][0]),
                             m.LaggedStart(*[m.GrowFromEdge(s, m.LEFT) for s in [*d2[0][1], *d2[1][1]]], lag_ratio=0.15),
                             m.FadeIn(rows[1][0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(rows[1][1], shift=0.1 * m.UP), m.FadeIn(moshi)),
        )
        self.say(
            "[A third design pairs a fast talker with a slow thinker]: [one keeps the call flowing], [the other reasons in the background and passes it advice.]",
            m.AnimationGroup(m.FadeIn(boxes[2]), m.FadeIn(titles[2]), m.FadeIn(fast, shift=0.1 * m.UP), m.FadeIn(slow, shift=0.1 * m.UP)),
            m.FadeIn(rows[2][0], shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowArrow(advice), m.FadeIn(advice_label), m.FadeIn(rows[2][1], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ robots: the layer cake

    def robots(self):
        self.next_section(
            "07 · Robots",
            notes="Li, ch. 6 (robot control layers): task goal ~minutes; planning seconds to minutes; basic skills such as "
            "pick(red_cup) about 1–3 s; VLA inference about 1–10 Hz ('A large model can usually run inference only 1–10 times "
            "per second'); low-level control about 50–1000 Hz. Without separation 'the high-level model's inference latency "
            "drags down low-level control'. The arm drawing is schematic. Fact date: October 2026.",
        )
        world = RobotScene()
        self.world = world
        goal = message("user", "Put the red cup on the tray.", width=4.0, size=22).move_to([-4.45, 1.85, 0])
        self.goal = goal
        left, right = -2.15, 5.95
        rows_y = (1.95, 1.0, 0.05, -0.9, -1.85)
        layer_boxes, layer_text = [], []
        for (name, example, rate, color, mono), y in zip(LAYERS, rows_y):
            box = card(right - left, 0.8, color, CARD, radius=0.14, stroke=2.2).move_to([(left + right) / 2, y, 0])
            nm = title_text(name, 24, color).move_to([left + 0.3, y, 0], aligned_edge=m.LEFT)
            ex = (code(example, 20, INK) if mono else text(example, 22, INK)).move_to([left + 1.85, y, 0], aligned_edge=m.LEFT)
            rt = code(rate, 20, color if color != MUTED else INK, weight=m.BOLD).move_to([right - 0.3, y, 0], aligned_edge=m.RIGHT)
            layer_boxes.append(box)
            layer_text.append((nm, ex, rt))
        clock = m.Arrow([6.3, 2.25, 0], [6.3, -2.2, 0], buff=0, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.04)
        top_lab = code("minutes", 20, MUTED).move_to([6.5, 2.55, 0], aligned_edge=m.RIGHT)
        bot_lab = code("milliseconds", 20, MUTED).move_to([6.5, -2.5, 0], aligned_edge=m.RIGHT)

        source = top_cite(SOURCE)

        def beat(box, n, color, run_time):
            return m.Succession(*[m.ShowPassingFlash(box.copy().set_fill(opacity=0).set_stroke(color, 6), time_width=0.5,
                                                     run_time=run_time / n) for _ in range(n)])

        self.say(
            "[Last, a robot], [asked to put the red cup on the tray.]",
            m.AnimationGroup(self.new_header("07 · robots", "From minutes to milliseconds"), m.FadeIn(source),
                             m.FadeIn(world.group, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(goal[0]), m.FadeIn(goal[1]), m.FadeIn(goal[2][0]), m.AddTextLetterByLetter(goal[2][1], time_per_char=0.03)),
        )
        self.say(
            "[Robots are built in layers], [from minutes at the top], [to milliseconds at the bottom.]",
            m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.DOWN) for b in layer_boxes], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(top_lab, shift=0.1 * m.DOWN), m.GrowArrow(clock)),
            m.FadeIn(bot_lab, shift=0.1 * m.UP),
        )
        self.say(
            "[The goal takes minutes.] [A planner breaks it into skills], [like pick red cup, which takes one to three seconds.]",
            m.FadeIn(m.VGroup(*layer_text[0]), shift=0.1 * m.RIGHT),
            m.FadeIn(m.VGroup(*layer_text[1]), shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(m.VGroup(*layer_text[2]), shift=0.1 * m.RIGHT), beat(layer_boxes[2], 1, SUN, 1.2)),
        )
        self.say(
            "[Below that, a vision-language-action model turns the skill into motion], [one to ten times a second.]",
            m.FadeIn(m.VGroup(*layer_text[3]), shift=0.1 * m.RIGHT),
            beat(layer_boxes[3], 4, LIME, 2.0),
        )
        self.say(
            "[And a controller drives the motors], [fifty to a thousand times a second.]",
            m.FadeIn(m.VGroup(*layer_text[4]), shift=0.1 * m.RIGHT),
            beat(layer_boxes[4], 12, INK, 2.2),
        )
        self.wait(0.3)
        self.clear(world.group, goal[0], goal[1], goal[2][0], goal[2][1])

    # ------------------------------------------------------------------ robots: bounded skills

    def skills(self):
        self.next_section(
            "08 · Bounded skills",
            notes="Li, ch. 6: 'the model should not emit arbitrary joint angles directly'; it chooses bounded skills, executed by a "
            "calibrated, speed-limited controller. XLeRobot case: the planner submits pick(red_cup); 'only once the cup is "
            "confirmed to be held may the planner submit place(red_cup, tray)'; every success confirmed by a fresh observation. "
            "The joint angles and the result texts are illustrative. Fact date: October 2026.",
        )
        w = self.world
        arm, cup = w.arm, w.cup
        x0, width = -0.1, 6.6
        cx = x0 + width / 2
        bad = card(width, 1.15, ROSE, CARD).move_to([cx, 1.75, 0])
        joints = "set_joints([" + ", ".join(f"{a:.2f}" for a in JOINTS) + "])"
        bad_body = m.VGroup(icon("ep01-cross", 0.46),
                            m.VGroup(code(joints, 20, INK), text("arbitrary joint angles", 20, ROSE, weight=m.SEMIBOLD))
                            .arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)).arrange(m.RIGHT, buff=0.3)
        bad_body.move_to(bad).align_to(bad, m.LEFT).shift(0.35 * m.RIGHT)
        skill_label = text("bounded skills", 22, MUTED)
        skill_chips = [mono_chip(s, SUN, SUN_FILL, size=22, height=0.52) for s in ("pick", "place", "open")]
        skill_row = m.VGroup(skill_label, *skill_chips).arrange(m.RIGHT, buff=0.25).move_to([cx, 0.72, 0]).align_to([x0, 0, 0], m.LEFT)
        ctrl = cite("each run by a calibrated, speed-limited controller").move_to([x0, 0.2, 0], aligned_edge=m.LEFT)
        msgs = [row_msg("tool_use", 'pick("red_cup")'), row_msg("tool_result", "camera: red_cup in gripper"),
                row_msg("tool_use", 'place("red_cup", "tray")'), row_msg("tool_result", "camera: red_cup on tray")]
        y = -0.42
        for msg in msgs:
            msg.move_to([cx, y, 0])
            y -= msg.height + 0.12
        source = top_cite(SOURCE)
        mob = w.arm_mob
        pick = m.Succession(
            arm.move(mob, w.home, w.above_cup, run_time=0.5),
            arm.move(mob, w.above_cup, w.at_cup, run_time=0.4),
            arm.move(mob, w.at_cup, w.at_cup, opening=(1.0, 0.45), run_time=0.3),
            arm.move(mob, w.at_cup, w.above_cup, opening=(0.45, 0.45), carry=cup, run_time=0.45),
        )
        place = m.Succession(
            arm.move(mob, w.above_cup, w.above_tray, opening=(0.45, 0.45), carry=cup, run_time=0.8),
            arm.move(mob, w.above_tray, w.at_tray, opening=(0.45, 0.45), carry=cup, run_time=0.4),
            arm.move(mob, w.at_tray, w.at_tray, opening=(0.45, 1.0), run_time=0.25),
            arm.move(mob, w.at_tray, w.above_tray, run_time=0.35),
        )

        self.say(
            "[So the language model never sends raw joint angles.] [It only chooses bounded skills], [each one called like a tool.]",
            m.AnimationGroup(self.new_header("08 · bounded skills", "Skills, not joint angles"), m.FadeIn(source),
                             m.FadeIn(bad), m.FadeIn(bad_body, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(skill_label), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in skill_chips], lag_ratio=0.2),
                             m.FadeIn(ctrl)),
            m.AnimationGroup(m.FadeIn(msgs[0], shift=0.15 * m.UP), m.Indicate(skill_chips[0], color=SUN, scale_factor=1.12)),
        )
        self.say(
            "[After each skill, it looks again]: [the cup is in the gripper.] [Only then does it call place.]",
            pick,
            m.AnimationGroup(m.FadeIn(msgs[1], shift=0.15 * m.UP), m.Circumscribe(cup, color=MINT, buff=0.1)),
            m.Succession(m.FadeIn(msgs[2], shift=0.15 * m.UP, run_time=0.5), place, m.FadeIn(msgs[3], shift=0.15 * m.UP, run_time=0.5)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the shared skeleton

    def skeleton(self):
        self.next_section(
            "09 · One skeleton",
            notes="Li, ch. 6 summary: 'keep perceiving → judge current state and timing → choose a reply or an action → let the "
            "output enter the environment → observe the feedback → continue, correct, retry, stop, or replan.' Shortened here "
            "to perceive, judge state & timing, act, observe, correct. The example rates repeat section 01 (screenshot every "
            "few seconds, voice 10 ms–1 s, VLA 1–10 Hz).",
        )
        source = top_cite(SOURCE)
        c = np.array([1.75, -0.35, 0])
        ring = loop_ring(["perceive", "judge state & timing", "act", "observe", "correct"], radius=2.0,
                         colors=[MINT, CORAL, SUN, MINT, LIME], size=22, center=c)
        nodes, arcs = ring
        bot = put_head(otto(0.6), c + 0.05 * m.DOWN)
        examples = []
        for (word, color, fill, rate), y in zip((("screenshot", MINT, MINT_FILL, "every few seconds"),
                                                 ("caller's voice", AZURE, AZURE_FILL, "10 ms – 1 s"),
                                                 ("camera frame", LIME, LIME_FILL, f"{VLA_HZ[0]}–{VLA_HZ[1]} Hz")), (1.25, 0.0, -1.25)):
            examples.append(m.VGroup(mono_chip(word, color, fill), code(rate, 20, MUTED)).arrange(m.DOWN, buff=0.14).move_to([-4.65, y, 0]))
        caption = title_text("Same loop. Different clocks.", 30, LIME).move_to([c[0], -3.05, 0])

        self.say(
            "[All three share one skeleton]: [perceive], [judge the state and the timing], [act], [observe], [and correct.]",
            m.AnimationGroup(self.new_header("09 · one skeleton", "Same loop, different clocks"), m.FadeIn(source), m.FadeIn(bot, scale=0.7)),
            m.FadeIn(nodes[0], scale=0.8),
            m.AnimationGroup(m.Create(arcs[0]), m.FadeIn(nodes[1], scale=0.8)),
            m.AnimationGroup(m.Create(arcs[1]), m.FadeIn(nodes[2], scale=0.8)),
            m.AnimationGroup(m.Create(arcs[2]), m.FadeIn(nodes[3], scale=0.8)),
            m.AnimationGroup(m.Create(arcs[3]), m.FadeIn(nodes[4], scale=0.8), m.Create(arcs[4])),
        )
        self.say(
            "[A screenshot], [a caller's voice], [a camera frame]: [the loop stays the same. Only the clock changes.]",
            m.FadeIn(examples[0], shift=0.15 * m.RIGHT),
            m.FadeIn(examples[1], shift=0.15 * m.RIGHT),
            m.FadeIn(examples[2], shift=0.15 * m.RIGHT),
            m.AnimationGroup(m.Succession(*[arc_flash(a, col, 0.4) for a, col in zip(arcs, (MINT, CORAL, SUN, MINT, LIME))]),
                             m.FadeIn(caption, shift=0.15 * m.UP)),
        )
        self.wait(0.5)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Ground every click on a real target; stream every word; layer every motion under a "
                          "bounded skill. Turn-taking is a convention: the harness keeps time.")
        halo = glow(LIME, radius=4.0)
        mark = m.VGroup(m.Circle(radius=0.34).set_fill(SUN, 1).set_stroke(BG, 2), code("3", 28, BG, weight=m.BOLD))
        ptr = point_at(cursor().scale(1.3), mark[0].get_center() + np.array([0.18, -0.18, 0]))
        pic_ground = m.VGroup(mark, ptr)
        pic_stream = m.VGroup(*[m.RoundedRectangle(corner_radius=0.07, width=w, height=0.22, stroke_width=0, fill_color=col,
                                                   fill_opacity=1).move_to([x, y, 0])
                                for w, x, y, col in ((1.3, -0.3, 0.3, AZURE), (1.2, 0.1, 0.0, MUTED), (1.3, 0.45, -0.3, CORAL))])
        pic_layer = m.VGroup(*[m.RoundedRectangle(corner_radius=0.06, width=1.6, height=0.2, stroke_color=col, stroke_width=2,
                                                  fill_color=CARD, fill_opacity=1).move_to([0, 0.36 - 0.24 * i, 0])
                               for i, col in enumerate((AZURE, CORAL, SUN, LIME))])
        spec = (("Ground", SUN, pic_ground, "clicks on real targets"), ("Stream", AZURE, pic_stream, "words as they come"),
                ("Layer", LIME, pic_layer, "motion inside skills"))
        cols = m.VGroup()
        for word, color, pic, sub in spec:
            if pic.height > 0.9:
                pic.scale(0.9 / pic.height)
            cols.add(m.VGroup(pic, title_text(word, 60, color), code(sub, 22, MUTED)).arrange(m.DOWN, buff=0.3))
        cols.arrange(m.RIGHT, buff=1.1).move_to(0.6 * m.UP)
        on_baseline(m.VGroup(*[c[1] for c in cols]))
        for c in cols:
            c[0].set_y(cols[0][0].get_y())
            c[2].shift((cols[0][1].get_bottom()[1] - 0.3 - c[2].get_top()[1]) * m.UP)
        tag_a = text("The world won't wait for the model's turn,", 30, MUTED)
        tag_b = text("so the harness keeps time.", 30, LIME, weight=m.SEMIBOLD)
        tagline = m.VGroup(tag_a, tag_b).arrange(m.RIGHT, buff=0.18).next_to(cols, m.DOWN, buff=0.75)
        on_baseline(tagline)

        self.say(
            "[To sum up]: [ground], [stream], [and layer.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            *[m.FadeIn(c, shift=0.3 * m.UP) for c in cols],
        )
        self.say(
            "[The world won't wait for the model's turn], [so the harness keeps time.]",
            m.FadeIn(tag_a, shift=0.1 * m.UP),
            m.FadeIn(tag_b, shift=0.1 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, cols, tagline)], run_time=0.9)
