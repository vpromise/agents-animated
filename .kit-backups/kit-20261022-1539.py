"""The series kit for "Agents, Animated": design system, agent components, voice, and the Episode
base class with the series intro and next-episode outro. Episodes import it:

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from kit import *  # noqa: F403

    class Ep02TheAgentLoop(Episode):
        number = 2
        def lesson(self): ...

Run from an episode folder: `VOICE=draft manimgx inspect scene.py` (silent draft) or, with the
narrator serving from the series folder, `manimgx render scene.py`. A tour of every component:
`kit_demo.py` in the series folder.
"""

import base64
import json
import os
import re
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

import manimgx as m
from manimgx.audio import Word

SERIES = Path(__file__).resolve().parent
ASSETS = SERIES / "assets"
TITLE_WIDTH = 12.5  # the widest an intro/outro card title may be (the frame is about 14.2 wide)

# ---------------------------------------------------------------------------- engine fixes
# (carried over from the "AI, Explained" kit: each one fixes a ManimGX timing fault, see the
# tutorial-video skill's reference/manimgx.md, gotchas 19–21)


# TransformFromCopy animates the target itself, so say() shows the finished target from the start
# of its spoken line. FadeIn from the source waits for its cue and looks nearly the same.
def _from_copy(mobject, target_mobject, **kwargs):
    return m.FadeIn(target_mobject, target_position=mobject, **kwargs)


# FadeTransform and FadeTransformPieces show their target early the same way; a cross-fade where
# each side glides toward the other waits for its cue and looks the same.
def _fade_transform(mobject, target_mobject, stretch=True, dim_to_match=1, **kwargs):
    return m.AnimationGroup(
        m.FadeOut(mobject, target_position=target_mobject),
        m.FadeIn(target_mobject, target_position=mobject),
        **kwargs,
    )


m.TransformFromCopy = _from_copy
m.FadeTransform = _fade_transform
m.FadeTransformPieces = _fade_transform


# A line that introduces a group and also moves, rotates or highlights a PART of it shows that
# part alone from the start of the line: play() puts the object of every non-introducing animation
# on screen at once, and say() plays the whole line as one play. Marking those animations as
# introducers keeps play() from adding their parts early. Episode.say applies this.
def _leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from _leaves(sub)
    else:
        yield animation


def _cue_safe(scene, animations):
    on_screen = {id(x) for x in scene.get_mobject_family_members()}
    roots, family = set(), set()
    for animation in animations:
        for leaf in _leaves(animation):
            if leaf.is_introducer() and getattr(leaf, "mobject", None) is not None:
                roots.add(id(leaf.mobject))
                family |= {id(x) for x in leaf.mobject.get_family()}
    for animation in animations:
        for leaf in _leaves(animation):
            mob = getattr(leaf, "mobject", None)
            if mob is None or leaf.is_introducer() or id(mob) in roots or id(mob) in on_screen:
                continue
            if id(mob) in family:  # a part of something introduced in this line
                leaf.introducer = True
    return animations


# Compiled under ManimGX's package path: ManimGX labels each play with its first caller outside
# its package, so the label still points at the episode's own scene.py line.
_SAY = """
def say(self, script, *animations, voice=None):
    return _scene_say(self, script, *_cue_safe(self, animations), voice=voice)
"""
_say_namespace = {"_scene_say": m.Scene.say, "_cue_safe": _cue_safe}
exec(compile(_SAY, str(Path(m.__file__).parent / "_series_kit_say.py"), "exec"), _say_namespace)

# ---------------------------------------------------------------------------- design system
# Warm graphite stage, coral brand. Colours carry meaning across the whole series:
#   system ORCHID · user AZURE · assistant (the model) CORAL · tool call SUN · tool result MINT
#   good MINT · bad ROSE · plain structure EDGE/MUTED.

BG = "#14120F"
CARD = "#1E1B17"
CARD_HI = "#262219"
EDGE = "#3B352D"
INK = "#F2ECE2"
MUTED = "#A9A094"
FAINT = "#5E574C"
CORAL, CORAL_FILL = "#FF7A59", "#3A1E16"
SUN, SUN_FILL = "#FFC857", "#3A2E14"
MINT, MINT_FILL = "#3DDC97", "#12301F"
AZURE, AZURE_FILL = "#5AB8FF", "#132838"
ORCHID, ORCHID_FILL = "#C792EA", "#2C1F38"
ROSE, ROSE_FILL = "#FF5C7A", "#3A1520"
LIME, LIME_FILL = "#B8E35A", "#26300F"
GOOD, GOOD_FILL = MINT, MINT_FILL
BAD, BAD_FILL = ROSE, ROSE_FILL

ROLES = {
    "system": (ORCHID, ORCHID_FILL),
    "user": (AZURE, AZURE_FILL),
    "assistant": (CORAL, CORAL_FILL),
    "tool_use": (SUN, SUN_FILL),
    "tool_result": (MINT, MINT_FILL),
}

HEAD = "Futura"  # titles and headers (bold)
SANS = "Avenir Next"  # everything you read
MONO = "Menlo"  # code, JSON, terminals, labels: no ligatures, so "<|", "->", "!=" stay as typed

m.config.background_color = BG

AGENT_NAME = "Otto"  # the series' running agent (drawn by otto())

# the course: module → (name, accent, fill); episode → (title, module). See COURSE.md.
MODULES = {
    1: ("FOUNDATIONS", CORAL, CORAL_FILL),
    2: ("CONTEXT ENGINEERING", SUN, SUN_FILL),
    3: ("MEMORY & KNOWLEDGE", ORCHID, ORCHID_FILL),
    4: ("TOOLS & SAFETY", MINT, MINT_FILL),
    5: ("SCALING UP", AZURE, AZURE_FILL),
    6: ("BEYOND THE LOOP", LIME, LIME_FILL),
}
EPISODES = {
    1: ("What Is an Agent?", 1), 2: ("The Agent Loop", 1), 3: ("Tools: Giving the Model Hands", 1),
    4: ("Think, Act, Observe", 1), 5: ("Workflows or Agents?", 1),
    6: ("The Context Window", 2), 7: ("KV Cache & Prompt Caching", 2), 8: ("The System Prompt, Assembled", 2),
    9: ("Plans You Can See", 2), 10: ("Skills: Knowledge on Demand", 2), 11: ("Compaction", 2),
    12: ("Memory Across Sessions", 3), 13: ("Agentic Search & RAG", 3), 14: ("The File System as Memory", 3),
    15: ("Designing Good Tools", 4), 16: ("MCP: One Protocol for Tools", 4), 17: ("Permissions & Sandboxes", 4),
    18: ("Hooks", 4), 19: ("Prompt Injection", 4), 20: ("Error Recovery", 4),
    21: ("Subagents", 5), 22: ("Task Graphs", 5), 23: ("Background Work & Schedules", 5),
    24: ("Agent Teams", 5), 25: ("Autonomy & Worktrees", 5), 26: ("Multi-Agent Patterns", 5),
    27: ("Coding Agents Are General Agents", 6), 28: ("Computer Use, Voice & Events", 6),
    29: ("Evaluating Agents", 6), 30: ("Training Agents with RL", 6),
    31: ("Agents That Learn on the Job", 6), 32: ("The Complete Agent", 6),
}
LEVELS = MODULES  # the "AI, Explained" name, for code written against that kit

_SMALL = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "twenty thirty forty fifty sixty seventy eighty ninety".split()


def spoken_title(title):
    """A title as the narrator should read it: no symbols."""
    return title.rstrip("?.").replace(" & ", " and ").replace(":", ",")


def say_number(n):
    """A number under a hundred as words, for the narrator."""
    return _SMALL[n] if n < 20 else _TENS[n // 10 - 2] + ("" if n % 10 == 0 else "-" + _SMALL[n % 10])


# ---------------------------------------------------------------------------- the voice

NARRATOR = os.environ.get("NARRATOR_URL", "http://127.0.0.1:8765/speak")
VOICE_REF = ""  # a clean 10–20 s WAV of a voice to copy; "" = the built-in voice


def narrator(text, voice_ref=VOICE_REF, exaggeration=0.5, cfg_weight=0.3, temperature=0.8):
    """Chatterbox, run locally by the tutorial-video skill's tts_server.py, timed by Whisper.
    Start it from the series folder: ~/.claude/skills/tutorial-video/scripts/narrator.sh"""
    settings = dict(voice_ref=voice_ref, exaggeration=exaggeration, cfg_weight=cfg_weight, temperature=temperature)
    request = urllib.request.Request(
        NARRATOR, json.dumps({"text": text, **settings}).encode(), {"Content-Type": "application/json"}
    )
    try:
        with urllib.request.urlopen(request, timeout=1800) as reply:
            said = json.load(reply)
    except urllib.error.URLError as error:
        raise RuntimeError(
            f"no narrator at {NARRATOR}: start ~/.claude/skills/tutorial-video/scripts/narrator.sh"
            " in the series folder (or render with VOICE=draft)"
        ) from error
    return m.Speech(base64.b64decode(said["audio"]), text=text, words=[Word(*w) for w in said["words"]])


def draft_voice(text, seconds_per_word=0.3):
    """A silent stand-in at about the real voice's pace: for layout and timing passes."""
    words = re.findall(r"\S+", text)
    rate, lead = 24000, 0.15
    timed = [Word(w, lead + i * seconds_per_word, lead + (i + 1) * seconds_per_word) for i, w in enumerate(words)]
    silence = np.zeros(int((2 * lead + seconds_per_word * len(words)) * rate), dtype=np.float32)
    return m.Speech(silence, text=text, words=timed, rate=rate)


# ---------------------------------------------------------------------------- basic components


def text(s, size=32, color=INK, weight=m.NORMAL, **kw):
    """Body text (Avenir Next)."""
    return m.Text(s, font=SANS, font_size=size, color=color, weight=weight, **kw)


def title_text(s, size=40, color=INK, **kw):
    """Display text (Futura Bold): titles, big numbers, recap words."""
    return m.Text(s, font=HEAD, font_size=size, color=color, weight=m.BOLD, **kw)


def code(s, size=22, color=INK, **kw):
    """Monospace text (Menlo). `t2c={"regex": colour}` colours matching parts."""
    return m.Text(s, font=MONO, font_size=size, color=color, **kw)


def tex(s, size=40, color=INK, **kw):
    """A formula (LaTeX math)."""
    return m.MathTex(s, font_size=size, color=color, **kw)


def glyphs(words, key):
    """The glyphs that draw the first `key` in a Text (spaces and newlines draw none)."""
    start = words.text.index(key)
    skip = sum(c.isspace() for c in words.text[:start])
    width = sum(not c.isspace() for c in key)
    return words[start - skip : start - skip + width]


def icon(name, height):
    return m.SVGMobject(str(ASSETS / f"{name}.svg"), height=height)


def card(width, height, color=EDGE, fill=CARD, radius=0.22, stroke=2.5):
    return m.RoundedRectangle(
        corner_radius=radius, width=width, height=height,
        stroke_color=color, stroke_width=stroke, fill_color=fill, fill_opacity=1,
    )


def pill(label, color, fill, size=22, mono=False):
    words = code(label, size - 2, color, weight=m.BOLD) if mono else text(label, size, color, weight=m.BOLD)
    height = words.height + 0.3
    box = card(words.width + 0.55, height, color, fill, radius=height / 2, stroke=2)
    return m.VGroup(box, words.move_to(box))


def chip(word, color=AZURE, fill=AZURE_FILL, width=1.5, height=0.72, size=28, mono=False):
    """A word in a small rounded box, on the same baseline as its neighbours'."""
    box = card(width, height, color, fill, radius=0.16, stroke=2.5)
    label = (code(word, size - 4, INK) if mono else text(word, size, INK, weight=m.SEMIBOLD)).move_to(box)
    sitting = [g for g, c in zip(label, word.replace(" ", "")) if c not in "gjpqy,;_()[]{}"] or list(label)
    baseline = box.get_center()[1] - 0.449 * size / 48 / 2
    label.shift((baseline - min(g.get_bottom()[1] for g in sitting)) * m.UP)
    return m.VGroup(box, label)


def header(tag, title, color, fill):
    """The section header: an accent bar, a mono tag, a Futura title (top left)."""
    tag_words = code(tag.upper(), 20, color, weight=m.BOLD)
    title_words = title_text(title, 34)
    bar = m.RoundedRectangle(corner_radius=0.04, width=0.1, height=0.62, stroke_width=0, fill_color=color, fill_opacity=1)
    row = m.VGroup(bar, tag_words, title_words).arrange(m.RIGHT, buff=0.3)
    tag_words.align_to(title_words, m.DOWN)
    return row.to_corner(m.UL, buff=0.5)


def bubble(words, color, fill, tail="left"):
    box = card(words.width + 0.7, words.height + 0.55, color, fill, radius=0.3)
    words.move_to(box)
    side = -1 if tail == "left" else 1
    base = box.get_bottom() + np.array([side * (box.width / 2 - 0.6), 0.15, 0])
    nub = m.Polygon(
        base + [-0.2, 0, 0], base + [0.2, 0, 0], base + [side * 0.5, -0.55, 0],
        stroke_color=color, stroke_width=2.5, fill_color=fill, fill_opacity=1,
    )
    return m.VGroup(nub, box, words)


def node(name, label, sub, color, fill, width=2.1, height=2.1, label_size=26):
    box = card(width, height, color, fill)
    body = m.VGroup(icon(name, 0.8), text(label, label_size, INK, weight=m.BOLD), text(sub, 20, MUTED))
    body.arrange(m.DOWN, buff=0.13).move_to(box)
    return m.VGroup(box, body)


def tile(name, title, desc, color, width=6.3, height=2.1):
    box = card(width, height, color, CARD, stroke=2)
    words = m.VGroup(text(title, 30, INK, weight=m.BOLD), text(desc, 22, MUTED))
    words.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
    body = m.VGroup(icon(name, 1.05), words).arrange(m.RIGHT, buff=0.45)
    body.move_to(box).align_to(box, m.LEFT).shift(0.45 * m.RIGHT)
    return m.VGroup(box, body)


def arrow(start, end, color=MUTED, buff=0.1):
    return m.Arrow(start, end, buff=buff, color=color, stroke_width=5, max_tip_length_to_length_ratio=0.3)


def pulse(a, color, width=10):
    return m.ShowPassingFlash(m.Line(a.get_start(), a.get_end()).set_stroke(color, width=width), time_width=0.7)


def on_baseline(words):
    for w in words[1:]:
        w.shift((words[0][0].get_bottom()[1] - w[0].get_bottom()[1]) * m.UP)
    return words


def glow(color, radius=3.0, layers=28):
    return m.VGroup(
        *[
            m.Circle(radius=radius * (1 - i / layers), stroke_width=0).set_fill(color, opacity=0.009)
            for i in range(layers)
        ]
    )


def backdrop():
    image = m.ImageMobject(str(ASSETS / "backdrop.png"))
    image.height = 8
    return image


def axes(x_range, y_range, x_length, y_length, color=FAINT):
    """Quiet axes for the dark stage: thin, no numbers (label them with text())."""
    return m.Axes(
        x_range=x_range, y_range=y_range, x_length=x_length, y_length=y_length,
        axis_config={"color": color, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18},
    )


def recolored(name, color, fill, prefix):
    """An icon in another accent, written as assets/<prefix>-<name>-<hex>.svg (pass your episode's
    prefix, e.g. "ep07", so parallel episodes never write the same file). Returns the new name."""

    def swap(match):
        hexcode = match.group(0)
        if hexcode.upper() in (BG.upper(), CARD.upper()):
            return hexcode
        r, g, b = (int(hexcode[i : i + 2], 16) / 255 for i in (1, 3, 5))
        return color if 0.2126 * r + 0.7152 * g + 0.0722 * b > 0.3 else fill

    variant = f"{prefix}-{name}-{color.lstrip('#').lower()}"
    out = re.sub(r"#[0-9A-Fa-f]{6}", swap, (ASSETS / f"{name}.svg").read_text())
    path = ASSETS / f"{variant}.svg"
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return variant


# ---------------------------------------------------------------------------- agent components

_PY_KEYWORDS = r"\b(def|while|for|in|if|elif|else|return|import|from|as|with|try|except|raise|class|break|continue|not|and|or|is|None|True|False|lambda|yield|await|async)\b"


def code_block(src, size=20, width=None, title=None, accent=EDGE):
    """Python (or pseudo-code) in a dark card, lightly highlighted: keywords ORCHID, strings MINT,
    comments FAINT, numbers SUN, called names AZURE. `src` is a str with newlines; keep lines short
    (about 48 characters at size 20 fill a 7-unit card)."""
    t2c = {
        r"#[^\n]*": FAINT,
        r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
        _PY_KEYWORDS: ORCHID,
        r"\b\d+(\.\d+)?\b": SUN,
        r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
    }
    body = code(src, size, INK, t2c=t2c, line_spacing=1.15)
    w = max(width or 0, body.width + 0.7)
    top = 0.0
    parts = []
    if title:
        cap = code(title, 16, MUTED)
        top = cap.height + 0.25
        parts.append(cap)
    box = card(w, body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    if title:
        body.shift(-top / 2 * m.UP)
        parts[0].align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
    return m.VGroup(box, body, *parts)


def terminal(lines, width=6.5, size=20, title="terminal"):
    """A terminal window. Lines starting with "$ " are commands (the prompt in MINT); lines starting
    with "! " are errors (ROSE); others are output (MUTED)."""
    rows = []
    for line in lines:
        if line.startswith("$ "):
            rows.append(code(line, size, INK, t2c={r"^\$": MINT}))
        elif line.startswith("! "):
            rows.append(code(line[2:], size, ROSE))
        else:
            rows.append(code(line or " ", size, MUTED))
    body = m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    bar_h = 0.42
    w = max(width, body.width + 0.7)
    box = card(w, body.height + 0.55 + bar_h, EDGE, "#100E0C", radius=0.16, stroke=2)
    bar = m.Line(box.get_corner(m.UL) + [0.0, -bar_h, 0], box.get_corner(m.UR) + [0.0, -bar_h, 0]).set_stroke(EDGE, 2)
    dots = m.VGroup(*[m.Dot(radius=0.06, color=c) for c in (ROSE, SUN, MINT)]).arrange(m.RIGHT, buff=0.12)
    dots.move_to(box.get_corner(m.UL) + [0.4, -bar_h / 2, 0])
    cap = code(title, 15, FAINT).move_to(box.get_top() + [0, -bar_h / 2, 0])
    body.next_to(bar, m.DOWN, buff=0.28).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, bar, dots, cap, body)


def json_card(obj, size=19, accent=SUN, fill=CARD, width=None, title=None):
    """A JSON object as a card: keys AZURE, strings MINT, numbers SUN, true/false/null ORCHID.
    `obj` is any JSON-able value (dicts keep their order)."""
    src = json.dumps(obj, indent=2, ensure_ascii=False)
    t2c = {
        r"\"[^\"\n]*\"(?=\s*:)": AZURE,
        r":\s*\"[^\"\n]*\"": MINT,
        r"\b-?\d+(\.\d+)?\b": SUN,
        r"\b(true|false|null)\b": ORCHID,
    }
    body = code(src, size, INK, t2c=t2c, line_spacing=1.1)
    top = 0.0
    cap = None
    if title:
        cap = code(title, 16, accent, weight=m.BOLD)
        top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, (cap.width + 0.7) if cap else 0)
    box = card(w, body.height + 0.55 + top, accent, fill, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def tool_call(name, args, size=19, width=None):
    """The model's request to run a tool, as the JSON block it emits (SUN)."""
    return json_card({"type": "tool_use", "name": name, "input": args}, size, SUN, CARD, width, "tool_use")


def tool_result(content, size=19, width=None, error=False):
    """What the harness sends back (MINT; ROSE when the tool failed)."""
    obj = {"type": "tool_result", "content": content}
    if error:
        obj["is_error"] = True
    return json_card(obj, size, ROSE if error else MINT, CARD, width, "tool_result")


def message(role, body, width=5.2, size=20, mono=False):
    """One message in a conversation: a role label in its colour and the content.
    role: system | user | assistant | tool_use | tool_result. `body` may contain newlines."""
    color, fill = ROLES[role]
    label = code(role, 15, color, weight=m.BOLD)
    words = (code(body, size - 2, INK) if mono else text(body, size, INK))
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    if words.width > width - 0.6:  # too long for the column: shrink rather than spill
        words.scale((width - 0.6) / words.width)
        inner.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    box = card(width, inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0,
                                fill_color=color, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


class Meter(m.VGroup):
    """A horizontal gauge (e.g. tokens used of the context window). `fill_to(f)` animates it."""

    def __init__(self, width=4.0, fraction=0.0, color=SUN, height=0.24):
        super().__init__()
        self.track_width, self.bar_height, self.color = width, height, color
        self.track = m.RoundedRectangle(corner_radius=height / 2, width=width, height=height,
                                        stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
        self.bar = self._bar(fraction)
        self.add(self.track, self.bar)

    def _bar(self, fraction):
        f = float(np.clip(fraction, 0.004, 1.0))
        color = ROSE if fraction > 0.9 else self.color
        bar = m.RoundedRectangle(corner_radius=self.bar_height / 2, width=self.track_width * f,
                                 height=self.bar_height, stroke_width=0, fill_color=color, fill_opacity=1)
        return bar.align_to(self.track, m.LEFT).set_y(self.track.get_y())

    def fill_to(self, fraction, **kwargs):
        """The animation that moves the gauge to `fraction` (0–1); it turns ROSE above 90%."""
        return m.Transform(self.bar, self._bar(fraction).align_to(self.track, m.LEFT), **kwargs)


class ContextColumn:
    """A tall panel showing the context window as a stack of messages. `push(msg)` returns the
    animation that adds a message below the last one; when the stack would overflow, older
    messages scroll up and the ones leaving the panel fade out (as a real chat log scrolls)."""

    def __init__(self, center=(3.6, -0.4, 0), width=5.6, height=5.8, title="context window", gap=0.12, meter=False):
        self.width, self.gap = width, gap
        self.panel = card(width, height, EDGE, "#18150F", radius=0.2, stroke=2)
        self.panel.move_to(center)
        self.title = code(title, 17, MUTED).next_to(self.panel, m.UP, buff=0.12).align_to(self.panel, m.LEFT)
        self.frame = m.VGroup(self.panel, self.title)
        self.meter = None
        if meter:
            self.meter = Meter(width * 0.42, 0.0, SUN, 0.16)
            self.meter.next_to(self.panel, m.UP, buff=0.18).align_to(self.panel, m.RIGHT)
            self.frame.add(self.meter)
        self.items = []
        self.top = self.panel.get_top()[1] - 0.18
        self.bottom = self.panel.get_bottom()[1] + 0.18

    def message(self, role, body, size=19, mono=False):
        """A message sized for this column."""
        return message(role, body, self.width - 0.36, size, mono)

    def push(self, msg, run_time=0.9):
        y = (self.items[-1].get_bottom()[1] - self.gap) if self.items else self.top
        msg.move_to([self.panel.get_x(), 0, 0]).align_to([0, y, 0], m.UP)
        overflow = self.bottom - msg.get_bottom()[1]
        anims = []
        if overflow > 0:
            msg.shift(overflow * m.UP)
            for item in list(self.items):
                if item.get_bottom()[1] + overflow > self.top:
                    anims.append(m.FadeOut(item, shift=overflow * m.UP))
                    self.items.remove(item)
                else:
                    anims.append(item.animate.shift(overflow * m.UP))
        anims.append(m.FadeIn(msg, shift=0.25 * m.UP))
        self.items.append(msg)
        return m.AnimationGroup(*anims, run_time=run_time)


def otto(size=1.0, color=CORAL, fill=CORAL_FILL):
    """Otto, the series' agent: a rounded screen-head with two eyes and a loop antenna.
    Parts: [0] head, [1] eyes (VGroup of two), [2] antenna stem, [3] antenna loop."""
    head = m.RoundedRectangle(corner_radius=0.3 * size, width=1.3 * size, height=1.0 * size,
                              stroke_color=color, stroke_width=4 * size, fill_color=fill, fill_opacity=1)
    eye = lambda: m.RoundedRectangle(corner_radius=0.06 * size, width=0.15 * size, height=0.3 * size,
                                     stroke_width=0, fill_color=INK, fill_opacity=1)
    eyes = m.VGroup(eye(), eye()).arrange(m.RIGHT, buff=0.32 * size).move_to(head).shift(0.03 * size * m.UP)
    stem = m.Line(head.get_top(), head.get_top() + 0.22 * size * m.UP).set_stroke(color, 4 * size)
    loop = m.Circle(radius=0.11 * size).set_stroke(color, 4 * size).next_to(stem, m.UP, buff=0)
    return m.VGroup(head, eyes, stem, loop)


def loop_ring(labels, radius=1.6, colors=None, size=24, center=(0, 0, 0)):
    """A cycle of steps around a circle with arrows between them (e.g. think → act → observe).
    Returns VGroup(nodes, arcs); nodes[i] is a chip, arcs[i] runs from node i to node i+1."""
    n = len(labels)
    colors = colors or [CORAL, SUN, MINT, AZURE, ORCHID][:n]
    fills = {CORAL: CORAL_FILL, SUN: SUN_FILL, MINT: MINT_FILL, AZURE: AZURE_FILL, ORCHID: ORCHID_FILL,
             ROSE: ROSE_FILL, LIME: LIME_FILL}
    center = np.array(center, dtype=float)
    angles = [np.pi / 2 - 2 * np.pi * i / n for i in range(n)]
    nodes = m.VGroup()
    for label, color, a in zip(labels, colors, angles):
        words = text(label, size, INK, weight=m.SEMIBOLD)
        c = chip(label, color, fills.get(color, CARD), width=max(1.4, words.width + 0.5), height=0.66, size=size)
        nodes.add(c.move_to(center + radius * np.array([np.cos(a), np.sin(a), 0])))
    arcs = m.VGroup()
    for i in range(n):
        a0, a1 = angles[i], angles[(i + 1) % n]
        if a1 > a0:
            a1 -= 2 * np.pi
        span = a1 - a0
        pad = 0.36 * 2 * np.pi / max(n, 3) / 2
        arc = m.Arc(radius=radius, start_angle=a0 - pad, angle=span + 2 * pad, arc_center=center)
        arc.set_stroke(MUTED, 4).add_tip(tip_length=0.2)
        arcs.add(arc)
    return m.VGroup(nodes, arcs)


def file_tree(entries, size=19, highlight=()):
    """A directory listing. `entries` are paths like "src/", "src/app.py" (folders end in "/").
    Names in `highlight` are drawn in SUN."""
    rows = []
    for path in entries:
        depth = path.rstrip("/").count("/")
        name = path.rstrip("/").split("/")[-1] + ("/" if path.endswith("/") else "")
        color = SUN if path in highlight or name in highlight else (AZURE if path.endswith("/") else INK)
        rows.append(code("   " * depth + ("▸ " if path.endswith("/") else "· ") + name, size, color))
    return m.VGroup(*rows).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)


# ---------------------------------------------------------------------------- the episode


class Episode(m.Scene):
    """An episode of the series: its intro card, its lesson (`lesson()`), the next-episode card.
    (Not `film`: a Scene keeps its Film there.)"""

    number = 0
    voice = draft_voice if os.environ.get("VOICE") == "draft" else narrator
    say = _say_namespace["say"]  # Scene.say, made cue-safe for parts (see _cue_safe)

    def construct(self) -> None:
        self.head = None
        self.add(backdrop().set_z_index(-100))
        self.series_intro()
        self.lesson()
        self.series_outro()

    def lesson(self) -> None:
        raise NotImplementedError

    @property
    def module(self):
        return MODULES[EPISODES[self.number][1]]

    level = module  # the "AI, Explained" name

    # helpers

    def new_header(self, tag, title, color=None, fill=None):
        """The animation that swaps the header: pass it to say() with a short [span]."""
        _, accent, tint = self.module
        new = header(tag, title, color or accent, fill or tint)
        parts = [m.FadeIn(new, shift=0.25 * m.UP)]
        if self.head:
            parts.insert(0, m.FadeOut(self.head, shift=0.25 * m.UP))
        self.head = new
        return m.AnimationGroup(*parts, run_time=0.9)

    def clear(self, *keep, run_time=0.8):
        gone = [mob for mob in self.mobjects[1:] if mob not in keep and mob is not self.head]
        if gone:
            self.play(*[m.FadeOut(mob) for mob in gone], run_time=run_time)

    def drop_header(self):
        """The animation that removes the header (for full-frame endings)."""
        old, self.head = self.head, None
        return m.FadeOut(old, shift=0.25 * m.UP) if old else m.Wait(0.01)

    # the series cards

    def _fit(self, heading):
        if heading.width > TITLE_WIDTH:
            heading.scale(TITLE_WIDTH / heading.width)
        return heading

    def series_intro(self):
        title, module = EPISODES[self.number]
        name, accent, tint = MODULES[module]
        self.next_section(f"Episode {self.number} · {title}", notes="Series card.")
        halo = glow(accent, radius=4.2).move_to(0.1 * m.DOWN)
        prompt = code(f"$ agents --episode {self.number:02d}", 22, MUTED, t2c={r"^\$": MINT})
        brand = title_text("Agents, Animated", 40, INK, gradient=(CORAL, SUN))
        badge = pill(f"MODULE {module} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 72 if len(title) < 22 else 58))
        top = m.VGroup(prompt, brand).arrange(m.DOWN, buff=0.22)
        group = m.VGroup(top, badge, heading).arrange(m.DOWN, buff=0.45).move_to(0.25 * m.UP)
        half = min(heading.width, 6) / 2
        rule = m.Line([-half, 0, 0], [half, 0, 0]).set_stroke(accent, width=5).next_to(heading, m.DOWN, buff=0.35)
        mascot = otto(0.55).next_to(rule, m.DOWN, buff=0.45)

        self.wait(0.3)
        self.say(
            f"[Welcome to Agents, Animated.] [Episode {say_number(self.number)}]: [{spoken_title(title)}.]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.AddTextLetterByLetter(prompt), m.FadeIn(brand, shift=0.2 * m.DOWN), run_time=1.2),
            m.AnimationGroup(m.FadeIn(badge, shift=0.2 * m.UP), m.FadeIn(mascot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Write(heading), m.GrowFromCenter(rule)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group, rule, mascot)], run_time=0.8)

    def series_outro(self):
        upcoming = self.number + 1
        self.next_section("Next episode", notes="Next-episode card.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        if upcoming not in EPISODES:
            self.say("Thanks for watching!")
            return
        title, module = EPISODES[upcoming]
        name, accent, tint = MODULES[module]
        halo = glow(accent, radius=4.0)
        kicker = code("next episode", 24, MUTED)
        number = pill(f"EP {upcoming:02d} · {name}", accent, tint, 20, mono=True)
        heading = self._fit(title_text(title, 60 if len(title) < 26 else 48))
        group = m.VGroup(kicker, number, heading).arrange(m.DOWN, buff=0.4).move_to(0.2 * m.UP)
        self.say(
            f"[Next time, in episode {say_number(upcoming)}]: [{spoken_title(title)}.] Thanks for watching!",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(kicker), m.FadeIn(number, shift=0.2 * m.UP)),
            m.Write(heading),
        )
        self.wait(1.0)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
