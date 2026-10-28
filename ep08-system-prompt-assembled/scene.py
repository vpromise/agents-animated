"""Agents, Animated · Episode 8 · The System Prompt, Assembled (Module 2 · Context Engineering)

The system prompt is built by code from parts. Running example: Otto's system prompt, built for tinyshop.

- Hook: the system prompt is the first thing the model reads on every call; Otto's has been one hand-typed
  line (harness_reference.SYSTEM, 58 bytes); Claude Code's interactive core is ~20–30 KB of text
  (learn-claude-code s10, the site's reading of Claude Code's source).
- One string: a prompt that only grows; s10's three problems ("Switching projects requires rewriting the
  entire prompt", "One change can break others", "Every request carries everything").
- Assembly: s10's PROMPT_SECTIONS — identity ("You are a coding agent. Act, don't explain."), tools (from
  the tool table), workspace (repo, OS, git status), memory only if MEMORY.md exists ("based on real
  state, not keywords"); the function below is the one that runs, and every byte count is computed.
- Order: static before dynamic (Episode 7's prefix cache): a git-status change with the workspace first
  vs last; reused bytes = the common prefix of the two prompts, computed. Claude Code's
  SYSTEM_PROMPT_DYNAMIC_BOUNDARY and the date + CLAUDE.md sent as a <system-reminder> user message
  (learn-claude-code s10, reading of CC source).
- The new-team-member test (Li, AI Agents in Depth, ch. 2) and Otto's identity rewritten as job, steps,
  limits (Markdown for hierarchy).
- Structure over tone: Li's own tau-bench experiment (ch. 2): tone changes had a "relatively limited"
  effect; removing the hierarchy of the same rules dropped success "by over 30%" (refunds without identity
  checks). An example from customer service, not tinyshop — said so in the narration.
- Project files: AGENTS.md / CLAUDE.md "acting as project-level system prompts", "the most economical
  stable prefix", the AI-ready newcomer test (Li, ch. 5).
- Constraints over guidance (Li, ch. 5): a rule moved from AGENTS.md into a linter check (ruff, rule
  T201). The check's output was measured with ruff 0.17.1 on the cart.py shown (2026-10-22):
  "cart.py:4:9: T201 `print` found" and, after the fix, "All checks passed!".
- Recap: assemble, order, enforce.

Facts dated October 2026. Illustrative (for teaching): the grown one-string prompt, tinyshop's context
values (macOS, git status, no MEMORY.md), the "blog" repo, Otto's job/steps/limits text, the refund SOP
wording, AGENTS.md's lines, Otto's debug print.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep08"

# ---------------------------------------------------------------------------- the running example

HARNESS = (SERIES / "harness_reference.py").read_text()
OLD_SYSTEM = "You are Otto, a coding agent working in the tinyshop repo."
assert f'SYSTEM = "{OLD_SYSTEM}"' in HARNESS  # Otto's prompt so far, as in the canonical harness
OLD_BYTES = len(OLD_SYSTEM.encode())
assert OLD_BYTES == 58
TOOL_NAMES = re.findall(r'"name": "(\w+)"', HARNESS.split("TOOLS = [")[1].split("\n]\n")[0])  # the tool table
assert TOOL_NAMES == ["bash", "read_file", "edit_file"]  # "bash, read file, edit file"
CC_KB = (20, 30)  # Claude Code's interactive core, "~20-30KB text" (learn-claude-code s10, reading of CC source)
TASK = "The checkout test fails. Fix it."

PROMPT_SECTIONS = {"identity": "You are a coding agent. Act, don't explain."}  # learn-claude-code s10


def tools_section(context):
    return "Tools: " + ", ".join(context["tools"])


def workspace_section(context):
    return f"Repo: {context['repo']} ({context['os']})\ngit status: {context['git']}"


def memory_section(context):
    return "Relevant memories:\n" + context["memories"]


def assemble_system_prompt(ctx):
    parts = [PROMPT_SECTIONS["identity"],
             tools_section(ctx),
             workspace_section(ctx)]
    if ctx.get("memories"):
        parts.append(memory_section(ctx))
    return "\n\n".join(parts)


ASSEMBLE_SRC = inspect.getsource(assemble_system_prompt).rstrip()
assert len(ASSEMBLE_SRC.splitlines()) == 7  # the code on screen is the code that runs


def workspace_first(context):
    """The same sections in the wrong order (for the cache comparison)."""
    identity, tools, workspace = assemble_system_prompt(context).split("\n\n")
    return "\n\n".join([workspace, identity, tools])


TINYSHOP = ["AGENTS.md", "cart.py", "checkout.py", "tests/test_cart.py"]  # no .memory/MEMORY.md yet
CONTEXT = {"tools": TOOL_NAMES, "repo": "tinyshop", "os": "macOS", "git": "clean",
           "memories": open(".memory/MEMORY.md").read() if ".memory/MEMORY.md" in TINYSHOP else ""}
PROMPT = assemble_system_prompt(CONTEXT)
SECTIONS = PROMPT.split("\n\n")
assert len(SECTIONS) == 3  # memory skipped: tinyshop has no MEMORY.md
SIZES = [len(s.encode()) for s in SECTIONS]
TOTAL = len(PROMPT.encode())
assert SIZES == [43, 33, 40] and TOTAL == 120 and TOTAL == sum(SIZES) + 2 * len("\n\n")

# another repo: only the workspace section changes
OTHER = {**CONTEXT, "repo": "blog"}
OTHER_SECTIONS = assemble_system_prompt(OTHER).split("\n\n")
assert [a != b for a, b in zip(SECTIONS, OTHER_SECTIONS)] == [False, False, True]
OTHER_SIZE, OTHER_TOTAL = len(OTHER_SECTIONS[2].encode()), len(assemble_system_prompt(OTHER).encode())
assert (OTHER_SIZE, OTHER_TOTAL) == (36, 116)
JOINS = 2 * len("\n\n")  # the two blank-line joins between three sections
assert TOTAL == sum(SIZES) + JOINS and OTHER_TOTAL == SIZES[0] + SIZES[1] + OTHER_SIZE + JOINS  # "+ 4 B of joins"

# Otto edits cart.py: the git status changes; what can a prefix cache reuse?
EDITED = {**CONTEXT, "git": "modified cart.py"}
GOOD_BEFORE, GOOD_AFTER = PROMPT, assemble_system_prompt(EDITED)
BAD_BEFORE, BAD_AFTER = workspace_first(CONTEXT), workspace_first(EDITED)
GOOD_REUSED = len(os.path.commonprefix([GOOD_BEFORE, GOOD_AFTER]))
BAD_REUSED = len(os.path.commonprefix([BAD_BEFORE, BAD_AFTER]))
NEW_TOTAL = len(GOOD_AFTER.encode())
assert len(BAD_AFTER.encode()) == NEW_TOTAL == 131
assert (GOOD_REUSED, BAD_REUSED) == (115, 35)  # "reused 115 of 131 B" vs "reused 35 of 131 B"
assert GOOD_AFTER[GOOD_REUSED - len("git status: "):GOOD_REUSED] == "git status: "  # the change is in the git line

# Li's tau-bench ablation (AI Agents in Depth, ch. 2): the author's own experiment, numbers as stated
ABLATION_DROP = 30  # success dropped "by over 30%" when the hierarchy of the same rules was removed

# constraints over guidance: a rule moved from AGENTS.md into a linter check
AGENTS_MD = ["# tinyshop", "- Run tests: pytest -q", "- Prices are integer cents", "- Don't leave print() in the code"]
RULE = AGENTS_MD[-1]
CART_FIXED = ["def total(items):", "    s = 0", "    for item in items:", "        s += item", "    return s"]
DEBUG = '        print("total", s)'
CART_DEBUG = CART_FIXED[:3] + [DEBUG] + CART_FIXED[3:]


def lint(lines):
    """ruff's T201 rule, as far as this file needs it: report each print( call."""
    hits = [f"cart.py:{i + 1}:{line.index('print(') + 1}: T201 `print` found"
            for i, line in enumerate(lines) if re.search(r"\bprint\(", line)]
    return hits or ["All checks passed!"]


LINT_BEFORE, LINT_AFTER = lint(CART_DEBUG), lint(CART_FIXED)
assert LINT_BEFORE == ["cart.py:4:9: T201 `print` found"]  # measured: ruff 0.17.1 on this file
assert LINT_AFTER == ["All checks passed!"]

# ---------------------------------------------------------------------------- look

USER_ICON = recolored("user", MUTED, CARD_HI, prefix=PREFIX)
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix=PREFIX)
LOCK_ICON = recolored("lock", MINT, MINT_FILL, prefix=PREFIX)
LLM_ICON = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
MD_T2C = {r"^#.*$": ORCHID, r"^\s*(\d+\.|-)": SUN}


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def split_lines(body, src):
    """The glyphs of each line of a multi-line Text (spaces draw no glyph)."""
    lines, k = [], 0
    for line in src.split("\n"):
        n = sum(not c.isspace() for c in line)
        lines.append(body[k:k + n])
        k += n
    return lines


class CodeCard:
    """kit.code_block with its lines addressable and a tinted line cursor (as in Episode 2)."""

    def __init__(self, src, title=None, width=None, size=20, t2c=None, accent=EDGE):
        if t2c is None:
            block = code_block(src, size=size, width=width, title=title, accent=accent)
            self.box, self.body = block[0], block[1]
            self.cap = block[2] if title else None
        else:  # markdown and other non-Python text
            body = code(src, size, INK, t2c=t2c)
            top = 0.0
            self.cap = None
            if title:
                self.cap = code(title, 20, MUTED)
                top = self.cap.height + 0.25
            self.box = card(max(width or 0, body.width + 0.7), body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
            body.move_to(self.box).align_to(self.box, m.LEFT).shift(0.35 * m.RIGHT)
            if title:
                body.shift(-top / 2 * m.UP)
                self.cap.align_to(self.box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
            self.body = body
            block = m.VGroup(self.box, body, *([self.cap] if title else []))
        self.group = block
        self.src = src.split("\n")
        self.lines = split_lines(self.body, src)

    def frame(self):
        return m.VGroup(self.box, *([self.cap] if self.cap else []))

    def base(self, i):
        return baseline(self.lines[i], self.src[i].replace(" ", ""))

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=SUN, opacity=0.16):
        b = a if b is None else b
        top, bot = self.base(a) + 0.25, self.base(b) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=opacity)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)

    def raise_text(self):
        self.body.set_z_index(2)
        if self.cap:
            self.cap.set_z_index(2)
        return self


def section_block(name, rows, width, nbytes=None, tag=None, tag_color=MUTED, color=ORCHID, dashed=False):
    """One section of the system prompt: an ORCHID stripe, the section's name, its text (Menlo), and its
    size in bytes (or a tag) on the right. Parts: [0] box, [1] stripe, [2] name, [3] rows, [4] size/tag."""
    label = code(name, 20, color, weight=m.BOLD)
    lines = m.VGroup(*[code(r, 20, INK if not dashed else FAINT) for r in rows]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    inner = m.VGroup(label, lines).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
    box = card(width, inner.height + 0.38, EDGE, CARD, radius=0.12, stroke=1.5)
    if dashed:
        box = m.DashedVMobject(box.set_fill(opacity=0), num_dashes=60).set_stroke(FAINT, 1.5)
    stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=box.height - 0.18, stroke_width=0,
                                fill_color=color if not dashed else FAINT, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    right = code(f"{nbytes} B" if nbytes is not None else (tag or ""), 20, tag_color)
    right.align_to(box, m.RIGHT).shift(0.25 * m.LEFT)
    right.shift((baseline(label, name.replace(" ", "")) - baseline(right, right.text.replace(" ", ""))) * m.UP)
    return m.VGroup(box, stripe, label, lines, right)


def wrap(section, at=". "):
    """A section's text as display rows (the identity sentence breaks after its first full stop)."""
    head, _, tail = section.partition(at)
    return [head + at.strip(), tail] if tail else [section]


def stack(blocks, top, x, gap=0.12):
    y = top
    for b in blocks:
        b.move_to([x, y - b.height / 2, 0])
        y -= b.height + gap
    return m.VGroup(*blocks)


def side_chip(pic, label, color, fill, width=2.4, height=0.8, size=22):
    box = card(width, height, color, fill, radius=0.18)
    body = m.VGroup(icon(pic, 0.48), text(label, size, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2).move_to(box)
    return m.VGroup(box, body)


def mono_chip(word, color, fill, size=22, height=0.6, pad=0.45):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def word_chip(word, color, fill, size=22, height=0.62, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def packet(n, color, at):
    """A message in miniature: one bar per line, flying to the model."""
    bars = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.02, width=0.6, height=0.075, stroke_width=0, fill_color=color, fill_opacity=1)
        for _ in range(n)
    ]).arrange(m.DOWN, buff=0.035)
    return bars.move_to(at)


def cache_bar(top, bottom, split, x):
    """A vertical gauge beside a prompt: MINT where the prefix cache is reused, ROSE where it is recomputed."""
    parts = m.VGroup()
    if top > split + 1e-6:
        parts.add(m.RoundedRectangle(corner_radius=0.05, width=0.12, height=top - split, stroke_width=0,
                                     fill_color=MINT, fill_opacity=1).move_to([x, (top + split) / 2, 0]))
    if split > bottom + 1e-6:
        parts.add(m.RoundedRectangle(corner_radius=0.05, width=0.12, height=split - bottom, stroke_width=0,
                                     fill_color=ROSE, fill_opacity=1).move_to([x, (split + bottom) / 2, 0]))
    return parts


def into(mob, target):
    """Fly into `target` and vanish: for a mobject introduced earlier in the same line."""
    out = m.FadeOut(mob, target_position=target.get_center(), scale=0.4)
    out.introducer = True
    return out


def detach(root, *parts):
    """Take faded-out parts out of their parent groups: a later fade of the whole would bring them
    back (gotcha 22)."""
    ids = {id(p) for p in parts}
    for mob in root.get_family():
        gone = [s for s in mob.submobjects if id(s) in ids]
        if gone:
            mob.remove(*gone)


def verdict(where, what, color, fill, width=5.6):
    box = card(width, 1.3, color, fill, radius=0.2)
    words = m.VGroup(code(where, 20, MUTED), title_text(what, 34, color)).arrange(m.DOWN, buff=0.16).move_to(box)
    return m.VGroup(box, words)


# ---------------------------------------------------------------------------- the episode


class Ep08SystemPromptAssembled(Episode):
    number = 8

    def lesson(self):
        self.hook()
        self.one_string()
        self.assembly()
        self.order()
        self.new_hire()
        self.structure()
        self.project_files()
        self.constraints()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"The system prompt is the first thing the model reads on every call. Otto's so far: harness_reference.SYSTEM, one hand-typed line ({OLD_BYTES} bytes). Claude Code's interactive core: ~{CC_KB[0]}–{CC_KB[1]} KB of text (learn-claude-code s10, the site's reading of Claude Code's source; unverified). Bars to scale: 30 KB = the full track; Otto's sliver is drawn at a minimum visible width.")
        bot = otto(1.25).move_to([-4.7, 1.35, 0])
        col = ContextColumn(center=(3.55, -0.45, 0), width=6.0, height=5.5, title="context window")
        sys_msg = col.message("system", "You are Otto, a coding agent\nworking in the tinyshop repo.")
        msgs = [sys_msg, col.message("user", TASK), col.message("tool_use", 'bash("pytest -q")', mono=True),
                col.message("tool_result", "1 failed, 12 passed", mono=True)]
        pushes = [col.push(msg) for msg in msgs]
        sp = pill("system prompt", ORCHID, ORCHID_FILL, 22)
        sp.move_to([-1.75, sys_msg.get_y(), 0])
        sp_arrow = arrow(sp.get_right(), sys_msg.get_left(), ORCHID, buff=0.12)

        old = CodeCard('SYSTEM = (\n    "You are Otto, a coding agent "\n    "working in the tinyshop repo.")', title="harness_reference.py")
        old.group.move_to([-3.6, -1.35, 0])
        typed = code(f"{OLD_BYTES} bytes, typed by hand", 20, MUTED).next_to(old.box, m.DOWN, buff=0.2).align_to(old.box, m.LEFT)

        # sizes to scale (30 KB = the whole track)
        track_w, left_x = 5.9, -6.55
        def size_row(label, y):
            lab = text(label, 22, INK, weight=m.SEMIBOLD).move_to([0, y + 0.42, 0]).align_to([left_x, 0, 0], m.LEFT)
            track = m.RoundedRectangle(corner_radius=0.08, width=track_w, height=0.32, stroke_color=EDGE, stroke_width=1.5,
                                       fill_color=CARD, fill_opacity=1).move_to([left_x + track_w / 2, y, 0])
            return lab, track
        lab1, track1 = size_row("Otto, so far", -0.35)
        bar1 = m.Rectangle(width=max(track_w * OLD_BYTES / (CC_KB[1] * 1000), 0.05), height=0.32, stroke_width=0,
                           fill_color=ORCHID, fill_opacity=1).align_to(track1, m.LEFT).set_y(track1.get_y())
        val1 = code(f"{OLD_BYTES} B", 20, ORCHID).next_to(lab1, m.RIGHT, buff=0.3).align_to(lab1, m.DOWN).align_to(track1, m.RIGHT)
        lab2, track2 = size_row("Claude Code, interactive core", -1.75)
        solid = m.Rectangle(width=track_w * CC_KB[0] / CC_KB[1], height=0.32, stroke_width=0, fill_color=ORCHID, fill_opacity=1)
        solid.align_to(track2, m.LEFT).set_y(track2.get_y())
        light = m.Rectangle(width=track_w * (CC_KB[1] - CC_KB[0]) / CC_KB[1], height=0.32, stroke_width=0, fill_color=ORCHID, fill_opacity=0.35)
        light.next_to(solid, m.RIGHT, buff=0)
        val2 = code(f"≈ {CC_KB[0]}–{CC_KB[1]} KB", 20, ORCHID).next_to(lab2, m.RIGHT, buff=0.3).align_to(lab2, m.DOWN)
        val2.align_to(track2, m.RIGHT)
        src = cite("learn-claude-code · s10, reading Claude Code's source").next_to(track2, m.DOWN, buff=0.35).align_to(track2, m.LEFT)
        row1 = m.VGroup(lab1, track1, bar1, val1)
        row2 = m.VGroup(lab2, track2, solid, light, val2)

        gear = m.VGroup(icon(GEAR_ICON, 0.7), code("assemble_system_prompt()", 20, AZURE)).arrange(m.RIGHT, buff=0.25)
        gear.move_to([-3.4, -0.35, 0])
        names = ["identity", "tools", "workspace", "memory"]
        parts = m.VGroup(*[mono_chip(n, ORCHID, ORCHID_FILL, 22) for n in names]).arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.25))
        parts.next_to(gear, m.DOWN, buff=0.45)

        self.say(
            "[Every time Otto calls the model], [it sends the system prompt along, ahead of the conversation.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(col.frame), m.LaggedStart(*pushes, lag_ratio=0.3)),
            m.AnimationGroup(m.Circumscribe(sys_msg, color=ORCHID, buff=0.06), m.FadeIn(sp, shift=0.2 * m.RIGHT), m.GrowArrow(sp_arrow)),
        )
        self.say(
            "[So far, Otto's system prompt has been one line], [typed by hand.]",
            m.FadeIn(old.frame(), shift=0.2 * m.UP),
            m.AnimationGroup(m.AddTextLetterByLetter(old.body, time_per_char=0.03), m.FadeIn(typed)),
        )
        self.say(
            "[Real agents send far more.] [Claude Code's core prompt is twenty to thirty kilobytes of text], [by Learn Claude Code's reading of its source.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(old.group, typed), shift=0.2 * m.DOWN), m.FadeIn(row1, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(m.VGroup(lab2, track2, val2), shift=0.15 * m.UP), m.GrowFromEdge(m.VGroup(solid, light), m.LEFT)),
            m.FadeIn(src),
        )
        self.say(
            "[Nobody types that by hand.] [Code assembles it from named parts], [right before each call.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(row1, row2, src), shift=0.2 * m.DOWN), m.FadeIn(gear, shift=0.15 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, target_position=gear[0]) for p in parts], lag_ratio=0.2),
            m.AnimationGroup(*[into(p, sys_msg) for p in parts], m.Circumscribe(sys_msg, color=ORCHID, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ one string

    def one_string(self):
        self.next_section("1 · One string", notes="Without assembly a prompt grows by appending (illustrative text). learn-claude-code s10 lists the problems: 'Switching projects requires rewriting the entire prompt', 'One change can break others' (a new tool description conflicting with an earlier instruction), 'Every request carries everything'.")
        src = ('SYSTEM = (\n'
               '    "You are Otto, working in tinyshop. "\n'
               '    "Use bash for everything. "\n'
               '    "Run the tests with pytest -q. "\n'
               '    "The user prefers tabs. "\n'
               '    "edit_file replaces text in a file. "\n'
               '    "Always answer politely. "\n'
               ')')
        grown = CodeCard(src, title="harness.py · before").raise_text()
        grown.group.align_to([-6.75, 0, 0], m.LEFT).align_to([0, 2.45, 0], m.UP)
        problems = [
            ("1", "Switching projects requires\nrewriting the entire prompt"),
            ("2", "One change can break others"),
            ("3", "Every request carries everything"),
        ]
        cards = m.VGroup()
        for num, words in problems:
            w = text(words, 24, INK)
            n = title_text(num, 28, ROSE)
            box = card(6.1, max(w.height + 0.5, 0.95), ROSE, CARD, radius=0.18, stroke=2)
            n.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            w.next_to(n, m.RIGHT, buff=0.3)
            cards.add(m.VGroup(box, n, w))
        cards.arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).align_to([6.75, 0, 0], m.RIGHT).align_to([0, 2.45, 0], m.UP)
        src_cite = cite("learn-claude-code · s10").next_to(cards, m.DOWN, buff=0.25).align_to(cards, m.RIGHT)

        model = side_chip(LLM_ICON, "model", CORAL, CORAL_FILL).move_to([grown.box.get_x(), -2.95, 0])
        down = arrow(grown.box.get_bottom(), model.get_top(), MUTED, buff=0.1)
        every = text("sent with every call", 22, MUTED).next_to(model, m.RIGHT, buff=0.35)
        pk = packet(len(grown.src), ORCHID, model.get_center())
        p_out = m.FadeOut(pk, scale=0.3, run_time=0.35)
        p_out.introducer = True  # pk is introduced by the FadeIn just before, in the same line
        flight = m.Succession(m.FadeIn(pk, target_position=grown.box, run_time=0.8), p_out)

        clash_a, clash_b = grown.bar(5, color=ROSE, opacity=0.2), grown.bar(2, color=ROSE, opacity=0.2)

        self.say(
            "[Without assembly, a prompt tends to grow like this]: [one string], [and every new need is appended to the end.]",
            self.new_header("01 · one string", "The prompt that only grows"),
            m.AnimationGroup(m.FadeIn(grown.frame()), m.FadeIn(grown.lines[0]), m.FadeIn(grown.lines[1], shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(line, shift=0.15 * m.UP) for line in grown.lines[2:]], lag_ratio=0.35),
        )
        self.say(
            "[Learn Claude Code names the problems.] [Switch projects, and you rewrite the whole thing.]",
            m.FadeIn(src_cite),
            m.AnimationGroup(m.FadeIn(cards[0], shift=0.2 * m.LEFT), m.Circumscribe(grown.part(1, "tinyshop"), color=ROSE, buff=0.06)),
        )
        self.say(
            "[One change can break others]: [the new line about edit file] [contradicts an older rule: use bash for everything.]",
            m.FadeIn(cards[1], shift=0.2 * m.LEFT),
            m.FadeIn(clash_a),
            m.FadeIn(clash_b),
        )
        self.say(
            "[And every request carries everything], [whether this turn needs it or not.]",
            m.FadeIn(cards[2], shift=0.2 * m.LEFT),
            m.AnimationGroup(m.FadeIn(model, shift=0.15 * m.UP), m.GrowArrow(down), m.FadeIn(every), flight),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ assembly

    def assembly(self):
        self.next_section("2 · Assembly", notes=f"learn-claude-code s10: PROMPT_SECTIONS; identity '{PROMPT_SECTIONS['identity']}'; tools from the enabled tools; workspace; memory only if .memory/MEMORY.md exists and has content ('based on real state, not keywords'). The function on screen is the one that ran; byte counts {SIZES}, total {TOTAL} (two blank-line joins). Workspace adds OS and git status (Claude Code appends git status as system context, s10). Another repo ('blog', illustrative) changes only the workspace section ({OTHER_SIZE} B, total {OTHER_TOTAL}).")
        fn = CodeCard(ASSEMBLE_SRC).raise_text()
        fn.group.align_to([-6.75, 0, 0], m.LEFT).align_to([0, 2.5, 0], m.UP)
        fn_title = cite("learn-claude-code · s10, simplified")

        keys = [("tools", ", ".join(CONTEXT["tools"])), ("repo", CONTEXT["repo"]), ("os", CONTEXT["os"]),
                ("git", CONTEXT["git"]), ("memories", "(no MEMORY.md)")]
        state_src = "\n".join(f"{k:<10}{v}" for k, v in keys)
        state = CodeCard(state_src, title="context · read from tinyshop", t2c={r"^\w+": AZURE, r"\(no MEMORY\.md\)": FAINT})
        state.group.next_to(fn.box, m.DOWN, buff=0.3).align_to(fn.box, m.LEFT)
        x, w = 3.675, 6.15
        head = code("system prompt", 20, MUTED).move_to([x, 2.4, 0]).align_to([x - w / 2, 0, 0], m.LEFT)
        blocks = [
            section_block("identity", wrap(SECTIONS[0]), w, SIZES[0]),
            section_block("tools", [SECTIONS[1]], w, SIZES[1]),
            section_block("workspace", SECTIONS[2].split("\n"), w, SIZES[2]),
            section_block("memory", ["skipped: no MEMORY.md"], w, tag="0 B", dashed=True),
        ]
        stk = stack(blocks, 1.82, x)
        quote = cite("“based on real state, not keywords”").next_to(blocks[3], m.DOWN, buff=0.2).align_to(blocks[3], m.LEFT)
        fn_title.to_corner(m.DR, buff=0.4).shift(0.2 * m.DOWN)
        fn_title.next_to(state.box, m.DOWN, buff=0.15).align_to(state.box, m.LEFT)
        total = code(f"{TOTAL} bytes", 20, ORCHID, weight=m.BOLD).move_to(head).align_to(blocks[0], m.RIGHT)
        total.shift((baseline(head, "systemprompt") - baseline(total, f"{TOTAL}bytes")) * m.UP)
        joins = code(f"+ {JOINS} B of joins (\\n\\n)", 20, MUTED).next_to(total, m.DOWN, buff=0.1).align_to(total, m.RIGHT)
        brace = m.Line(blocks[0].get_corner(m.UL) + [-0.18, 0, 0], blocks[2].get_corner(m.DL) + [-0.18, 0, 0]).set_stroke(ORCHID, 4)

        other_val = code("blog", 20, INK).move_to(state.part(1, "tinyshop"), aligned_edge=m.LEFT)
        other_val.shift((state.base(1) - baseline(other_val, "blog")) * m.UP)
        ws_rows = blocks[2][3]
        other_row = code(OTHER_SECTIONS[2].split("\n")[0], 20, INK).move_to(ws_rows[0], aligned_edge=m.LEFT)
        other_size = code(f"{OTHER_SIZE} B", 20, SUN).move_to(blocks[2][4], aligned_edge=m.RIGHT)
        other_total = code(f"{OTHER_TOTAL} bytes", 20, ORCHID, weight=m.BOLD).move_to(total, aligned_edge=m.RIGHT)
        same = m.VGroup(*[code("unchanged", 20, MINT).move_to(b[4], aligned_edge=m.RIGHT) for b in blocks[:2]])

        cur = fn.bar(0)
        self.say(
            "[Learn Claude Code's fix, in session ten]: [a function builds the prompt from named sections.]",
            m.AnimationGroup(self.new_header("02 · assembly", "Built from sections"), m.FadeIn(fn.group, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(cur), m.FadeIn(state.group, shift=0.15 * m.UP), m.FadeIn(fn_title), m.FadeIn(head)),
        )
        self.say(
            "[First, identity]: [You are a coding agent. Act, don't explain.]",
            m.AnimationGroup(m.Transform(cur, fn.bar(1)), m.FadeIn(m.VGroup(*blocks[0][:3], blocks[0][4]), target_position=fn.lines[1])),
            m.LaggedStart(*[m.AddTextLetterByLetter(r, time_per_char=0.035) for r in blocks[0][3]], lag_ratio=1.0),
        )
        self.say(
            "[Then tools], [straight from the tool table]: [bash, read file, edit file.]",
            m.Transform(cur, fn.bar(2)),
            m.Circumscribe(state.lines[0], color=SUN, buff=0.06),
            m.FadeIn(blocks[1], target_position=state.lines[0]),
        )
        self.say(
            "[Then the workspace]: [the repo, the operating system, and the git status.]",
            m.Transform(cur, fn.bar(3)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(*state.lines[1:4]), color=SUN, buff=0.06), m.FadeIn(blocks[2], target_position=m.VGroup(*state.lines[1:4]))),
        )
        self.say(
            "[Memory joins only if a memory file exists.] [tinyshop has none yet, so it's skipped]: [real state decides, not keywords.]",
            m.Transform(cur, fn.bar(4, 5)),
            m.AnimationGroup(m.Circumscribe(state.lines[4], color=ROSE, buff=0.06), m.FadeIn(blocks[3], shift=0.1 * m.UP)),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.say(
            "[The last line joins the parts], [and Otto's prompt is ready]: [built from state, not typed.]",
            m.Transform(cur, fn.bar(6)),
            m.AnimationGroup(m.FadeOut(m.VGroup(blocks[3], quote)), m.Create(brace), m.FadeIn(total, shift=0.1 * m.DOWN), m.FadeIn(joins)),
            m.Circumscribe(m.VGroup(*blocks[:3]), color=ORCHID, buff=0.1),
        )
        self.say(
            "[Point the same code at another repo], [and only the workspace section changes.]",
            m.AnimationGroup(m.FadeOut(cur), m.FadeOut(state.part(1, "tinyshop"), shift=0.15 * m.UP), m.FadeIn(other_val, shift=0.15 * m.UP)),
            m.AnimationGroup(
                m.FadeOut(ws_rows[0], shift=0.15 * m.UP), m.FadeIn(other_row, shift=0.15 * m.UP),
                m.FadeOut(blocks[2][4]), m.FadeIn(other_size), m.FadeOut(total), m.FadeIn(other_total),
                m.FadeOut(m.VGroup(blocks[0][4], blocks[1][4])), m.FadeIn(same),
                m.Circumscribe(blocks[2], color=SUN, buff=0.06),
            ),
        )
        detach(state.group, *state.part(1, "tinyshop"))
        detach(stk, ws_rows[0], blocks[0][4], blocks[1][4], blocks[2][4])
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ order

    def order(self):
        self.next_section("3 · Order", notes=f"Static before dynamic (Episode 7: a prefix cache is reused up to the first changed token; bytes shown as a stand-in for tokens). Otto edits cart.py, git status 'clean' -> 'modified cart.py'. Stable first: reused {GOOD_REUSED} of {NEW_TOTAL} B; workspace first: {BAD_REUSED} of {NEW_TOTAL} B (common prefixes, computed). Claude Code: SYSTEM_PROMPT_DYNAMIC_BOUNDARY splits static from dynamic sections; CLAUDE.md and the date are sent as a <system-reminder> user message (learn-claude-code s10, reading of CC source).")
        w = 5.75
        tops = 1.85

        def build(order, x, title):
            names = {"identity": wrap(SECTIONS[0]), "tools": [SECTIONS[1]], "workspace": SECTIONS[2].split("\n")}
            tags = {"identity": "static", "tools": "static", "workspace": "changes"}
            bs = [section_block(n, names[n], w, tag=tags[n], tag_color=SUN if n == "workspace" else MUTED) for n in order]
            st = stack(bs, tops, x)
            cap = text(title, 26, INK, weight=m.BOLD).move_to([x, tops + 0.4, 0])
            return bs, st, cap

        good, good_st, good_cap = build(["identity", "tools", "workspace"], 3.625, "stable first")
        bad, bad_st, bad_cap = build(["workspace", "identity", "tools"], -3.425, "workspace first")
        for b in good:
            b[4].set_opacity(0)  # tags appear later
        git_new = EDITED["git"]

        def git_swap(bs):
            ws = next(b for b in bs if b[2].text == "workspace")
            row = ws[3][1]
            new = code(f"git status: {git_new}", 20, SUN).move_to(row, aligned_edge=m.LEFT)
            return ws, row, new

        g_ws, g_row, g_new = git_swap(good)
        b_ws, b_row, b_new = git_swap(bad)
        top_g, bot_g = good_st.get_top()[1], good_st.get_bottom()[1]
        top_b, bot_b = bad_st.get_top()[1], bad_st.get_bottom()[1]
        gx, bx = good_st.get_left()[0] - 0.25, bad_st.get_left()[0] - 0.25
        g_full = cache_bar(top_g, bot_g, bot_g, gx)
        g_split = cache_bar(top_g, bot_g, g_row.get_top()[1] + 0.07, gx)
        b_split = cache_bar(top_b, bot_b, b_row.get_top()[1] + 0.07, bx)
        g_lab = code("cached", 20, MINT).next_to(good_st, m.DOWN, buff=0.25).align_to(good_st, m.LEFT)
        g_lab2 = code(f"reused {GOOD_REUSED} of {NEW_TOTAL} B", 20, MINT, t2c={r"\d+": SUN}).move_to(g_lab, aligned_edge=m.LEFT)
        b_lab = code(f"reused {BAD_REUSED} of {NEW_TOTAL} B", 20, ROSE, t2c={r"\d+": SUN}).next_to(bad_st, m.DOWN, buff=0.25).align_to(bad_st, m.LEFT)
        cold = m.Rectangle(width=w, height=b_row.get_top()[1] + 0.07 - bot_b, stroke_width=0, fill_color=ROSE, fill_opacity=0.12)
        cold.move_to([bad_st.get_x(), (b_row.get_top()[1] + 0.07 + bot_b) / 2, 0])
        ep7 = cite("Episode 7 · KV cache & prompt caching").next_to(good_cap, m.UP, buff=0.12)
        rule = pill("static first  ·  dynamic last", MINT, MINT_FILL, 22).next_to(g_lab, m.DOWN, buff=0.3).align_to(good_st, m.LEFT)

        # Claude Code's split, shown on the stable-first stack after it moves left
        shift_left = (-3.0 - good_st.get_x()) * m.RIGHT
        line_y = (good[1].get_bottom()[1] + good[2].get_top()[1]) / 2 + shift_left[1]
        boundary = m.DashedLine([good_st.get_left()[0] + shift_left[0] - 0.1, line_y, 0], [6.7, line_y, 0], dash_length=0.12).set_stroke(ORCHID, 3)
        b_label = code("SYSTEM_PROMPT_DYNAMIC_BOUNDARY", 20, ORCHID).next_to([4.5, line_y, 0], m.UP, buff=0.14)
        reminder = message("user", "<system-reminder>\ntoday's date · CLAUDE.md", width=4.9, size=20, mono=True)
        reminder.move_to([4.05, -1.75, 0])
        cc_cite = cite("learn-claude-code · s10, reading Claude Code's source").to_corner(m.DR, buff=0.4)
        # the stable-first stack starts centred and moves right when the other order joins it
        pre = -good_st.get_x() * m.RIGHT
        for mob in (good_cap, good_st, g_new, g_full, g_split, g_lab, g_lab2, ep7):
            mob.shift(pre)

        self.say(
            "[Order matters too.] [Remember episode seven]: [a cache reuses a prompt only up to its first change.]",
            m.AnimationGroup(self.new_header("03 · order", "Stable first, changing last"), m.FadeIn(good_cap), m.FadeIn(good_st, shift=0.15 * m.UP)),
            m.FadeIn(ep7, shift=0.1 * m.UP),
            m.AnimationGroup(m.GrowFromEdge(g_full, m.UP), m.FadeIn(g_lab)),
        )
        self.say(
            "[Identity and tools stay the same all session], [but the git status changes] [as soon as Otto edits the cart file.]",
            m.AnimationGroup(*[b[4].animate.set_opacity(1) for b in good[:2]]),
            m.AnimationGroup(good[2][4].animate.set_opacity(1), m.Circumscribe(g_row, color=SUN, buff=0.06)),
            m.AnimationGroup(
                m.FadeOut(g_row, shift=0.15 * m.UP), m.FadeIn(g_new, shift=0.15 * m.UP),
                m.FadeOut(g_full), m.FadeIn(g_split), m.FadeOut(g_lab), m.FadeIn(g_lab2),
            ),
        )
        detach(good_st, g_row)
        self.say(
            "[Only the tail is computed again.] [Now flip the order, workspace first], [and make the same edit]: [everything after the change goes cold.]",
            m.Circumscribe(g_split[1], color=ROSE, buff=0.08),
            m.AnimationGroup(
                m.VGroup(good_cap, good_st, g_new, g_split, g_lab2, ep7).animate.shift(-pre),
                m.Succession(m.Wait(0.4), m.AnimationGroup(m.FadeIn(bad_cap), m.FadeIn(bad_st, shift=0.15 * m.UP))),
            ),
            m.AnimationGroup(m.FadeOut(b_row, shift=0.15 * m.UP), m.FadeIn(b_new, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(cold), m.GrowFromEdge(b_split, m.UP), m.FadeIn(b_lab)),
        )
        detach(bad_st, b_row)
        self.say(
            "[So: stable parts first], [changing parts last.]",
            m.Circumscribe(m.VGroup(good_st, g_split), color=MINT, buff=0.12),
            m.FadeIn(rule, shift=0.1 * m.UP),
        )
        left_side = m.VGroup(bad_cap, bad_st, b_new, b_split, b_lab, cold)
        right_side = m.VGroup(good_cap, good_st, g_new, g_split, g_lab2, ep7)
        self.say(
            "[Claude Code splits its prompt at a boundary marker], [and sends the date and its project notes as a separate reminder message.]",
            m.AnimationGroup(
                m.FadeOut(left_side, shift=0.2 * m.LEFT), m.FadeOut(rule),
                right_side.animate.shift(shift_left),
                m.Succession(m.Wait(0.6), m.AnimationGroup(m.Create(boundary), m.FadeIn(b_label))),
            ),
            m.AnimationGroup(m.FadeIn(reminder, shift=0.2 * m.UP), m.FadeIn(cc_cite)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the new-team-member test

    def new_hire(self):
        self.next_section("4 · The test", notes="Li, AI Agents in Depth, ch. 2: 'If such a new team member, after reading your system prompt, still does not know what to do, neither will the Agent.' Otto's identity rewritten as job, steps, limits, with Markdown headings for hierarchy (illustrative wording).")
        person = icon(USER_ICON, 1.5).move_to([-5.9, 0.35, 0])
        role = text("new team member", 20, MUTED).next_to(person, m.DOWN, buff=0.2)
        bad = m.VGroup(code("system prompt", 20, ORCHID, weight=m.BOLD), text("You are a helpful assistant.", 26, INK)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18)
        bad_box = card(bad.width + 0.7, bad.height + 0.6, ORCHID, CARD, radius=0.16, stroke=2)
        bad_card = m.VGroup(bad_box, bad.move_to(bad_box)).move_to([0, 0.35, 0]).align_to([-4.45, 0, 0], m.LEFT)
        marks = m.VGroup(*[title_text("?", s, SUN).move_to(person.get_center() + d) for s, d in
                           ((44, [-0.7, 1.15, 0]), (36, [0.25, 1.35, 0]), (30, [0.95, 1.0, 0]))])
        fails = pill("fails the test", ROSE, ROSE_FILL, 22).next_to(bad_card, m.DOWN, buff=0.3)

        quote = text("“If such a new team member,\nafter reading your system prompt,\nstill does not know what to do,\nneither will the Agent.”", 26, INK)
        q_box = card(quote.width + 0.8, quote.height + 0.8, EDGE, CARD, radius=0.2, stroke=2)
        q_card = m.VGroup(q_box, quote.move_to(q_box)).align_to([6.75, 0, 0], m.RIGHT).set_y(0.55)
        q_cite = cite("Li, AI Agents in Depth · ch. 2").next_to(q_card, m.DOWN, buff=0.2).align_to(q_card, m.RIGHT)
        bot = otto(0.6).next_to(q_cite, m.DOWN, buff=0.45).align_to(q_card, m.RIGHT).shift(0.3 * m.LEFT)
        bot_q = title_text("?", 34, SUN).next_to(bot, m.LEFT, buff=0.25)

        good_src = "## Job\nFix bugs in tinyshop.\n## Steps\n1. Run pytest -q\n2. Read before you edit\n3. Re-run tests, then answer\n## Never\nEdit a test to make it pass."
        good = CodeCard(good_src, title="system prompt · identity", t2c=MD_T2C, accent=ORCHID)
        good.group.move_to([0, 0.3, 0]).align_to([-4.45, 0, 0], m.LEFT)
        ok = icon("check", 0.55).next_to(person, m.UP, buff=0.15) if (ASSETS / "check.svg").exists() else title_text("✓", 40, MINT).next_to(person, m.UP, buff=0.15)

        self.say(
            "[So what should the sections say?] [The book AI Agents in Depth offers a simple test.]",
            self.new_header("04 · the test", "Would a new hire know what to do?"),
            m.AnimationGroup(m.FadeIn(q_card, shift=0.15 * m.UP), m.FadeIn(q_cite)),
        )
        self.say(
            "[Picture a capable new team member], [who gets nothing but your system prompt.]",
            m.AnimationGroup(m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(role)),
            m.FadeIn(bad_card, shift=0.2 * m.LEFT),
        )
        self.say(
            "[If, after reading it, they still don't know what to do], [neither will the agent.]",
            m.LaggedStart(*[m.FadeIn(q, scale=0.5) for q in marks], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(bot, shift=0.15 * m.UP), m.FadeIn(bot_q, scale=0.5)),
        )
        self.say(
            "[You are a helpful assistant fails that test.]",
            m.AnimationGroup(m.FadeIn(fails, shift=0.1 * m.UP), bad_box.animate.set_stroke(ROSE)),
        )
        self.say(
            "[Otto's identity section names the job], [the steps], [and the limits.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(bad_card, fails), shift=0.2 * m.DOWN), m.FadeOut(marks), m.FadeOut(bot_q),
                m.FadeIn(good.frame()), m.FadeIn(m.VGroup(*good.lines[:2]), shift=0.1 * m.UP),
            ),
            m.FadeIn(m.VGroup(*good.lines[2:6]), shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(m.VGroup(*good.lines[6:]), shift=0.1 * m.UP), m.FadeIn(ok, scale=0.6)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ structure over tone

    def structure(self):
        self.next_section("5 · Structure", notes=f"Li, AI Agents in Depth, ch. 2 — the author's own tau-bench experiment (customer service, not tinyshop): three tones ('Trump style', casual) had a 'relatively limited' effect; keeping all the rules but removing their hierarchy dropped task success 'by over {ABLATION_DROP}%' — e.g. refunds issued without identity verification. The SOP wording is illustrative.")
        sop_src = "## Refund\n1. Verify the customer's identity\n2. Look up the order\n3. Issue the refund"
        flat_src = "- Issue the refund\n- Look up the order\n- Verify the customer's identity"
        sop = CodeCard(sop_src, title="system prompt · refunds", t2c=MD_T2C, accent=ORCHID)
        sop.group.align_to([-6.75, 0, 0], m.LEFT).set_y(0.55)
        flat = code(flat_src, 20, INK, t2c=MD_T2C).move_to(sop.body, aligned_edge=m.UL)
        flat_lines = split_lines(flat, flat_src)
        bench = cite("Li's own test on tau-bench · customer service").next_to(sop.box, m.DOWN, buff=0.22).align_to(sop.box, m.LEFT)
        src_cite = cite("Li, AI Agents in Depth · ch. 2").next_to(bench, m.DOWN, buff=0.1).align_to(bench, m.LEFT)

        def row(label, result, color, y):
            box = card(6.0, 1.2, EDGE, CARD, radius=0.18, stroke=2).move_to([3.75, y, 0])
            lab = text(label, 22, INK).move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            res = text(result, 24, color, weight=m.BOLD).move_to(box).align_to(box, m.RIGHT).shift(0.35 * m.LEFT)
            return m.VGroup(box, lab), res

        t_row, t_res = row("same rules,\nnew tone", "limited effect", MUTED, 1.75)
        f_row, f_res = row("same rules,\nno hierarchy", f"success down\nover {ABLATION_DROP}%", ROSE, 0.3)
        outcome = m.VGroup(code("refund issued", 22, ROSE, weight=m.BOLD), text("identity never checked", 24, INK)).arrange(m.DOWN, buff=0.14)
        o_box = card(6.0, outcome.height + 0.55, ROSE, ROSE_FILL, radius=0.18, stroke=2)
        o_card = m.VGroup(o_box, outcome.move_to(o_box)).move_to([3.75, -1.35, 0])

        self.say(
            "[Structure matters more than tone.] [The book's author tested this on a customer service benchmark, with refund rules like these.]",
            self.new_header("05 · structure", "Hierarchy beats tone"),
            m.AnimationGroup(m.FadeIn(sop.group, shift=0.15 * m.UP), m.FadeIn(bench), m.FadeIn(src_cite)),
        )
        self.say(
            "[Rewriting the rules in a different tone] [had a limited effect.]",
            m.FadeIn(t_row, shift=0.15 * m.LEFT),
            m.FadeIn(t_res, shift=0.1 * m.LEFT),
        )
        self.say(
            "[But flattening the same rules into an unordered list] [cut success by over thirty percent.]",
            m.AnimationGroup(
                m.FadeOut(sop.body, shift=0.15 * m.DOWN), m.FadeIn(flat, shift=0.15 * m.DOWN),
                m.FadeIn(f_row, shift=0.15 * m.LEFT),
            ),
            m.AnimationGroup(m.FadeIn(f_res, shift=0.1 * m.LEFT), f_row[0].animate.set_stroke(ROSE)),
        )
        detach(sop.group, sop.body)
        self.say(
            "[The agent sometimes issued the refund], [without checking who was asking.]",
            m.FadeIn(o_card, shift=0.15 * m.UP),
            m.Circumscribe(flat_lines[2], color=ROSE, buff=0.06),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ project files

    def project_files(self):
        self.next_section("6 · Project files", notes="Li, AI Agents in Depth, ch. 5: project instruction files (CLAUDE.md, AGENTS.md, .cursorrules) are 'automatically injected into the context at the start of every session, acting as project-level system prompts' and are 'the most economical stable prefix'; AI-ready test: 'can a remote newcomer, relying only on the repository and the documentation, start working independently?'. In Claude Code, CLAUDE.md arrives in a <system-reminder> user message (learn-claude-code s10). tinyshop's AGENTS.md lines are illustrative.")
        tree = file_tree(["tinyshop/", "tinyshop/AGENTS.md", "tinyshop/cart.py", "tinyshop/checkout.py", "tinyshop/tests/", "tinyshop/tests/test_cart.py"],
                         size=20, highlight=("AGENTS.md",))
        tree.align_to([-6.6, 0, 0], m.LEFT).align_to([0, 2.45, 0], m.UP)
        md = CodeCard("\n".join(AGENTS_MD), title="tinyshop/AGENTS.md", t2c=MD_T2C, accent=SUN)
        md.group.align_to([-6.75, 0, 0], m.LEFT).next_to(tree, m.DOWN, buff=0.4)
        md.group.align_to([-6.75, 0, 0], m.LEFT)
        others = m.VGroup(mono_chip("CLAUDE.md", ORCHID, ORCHID_FILL, 22), mono_chip("AGENTS.md", SUN, SUN_FILL, 22))
        others.arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT).next_to(tree, m.RIGHT, buff=0.8).align_to(tree, m.UP).shift(0.15 * m.DOWN)
        tags = m.VGroup(text("Claude Code", 20, MUTED), text("many other agents", 20, MUTED))
        for t, c in zip(tags, others):
            t.next_to(c, m.RIGHT, buff=0.2)

        col = ContextColumn(center=(3.85, -0.55, 0), width=5.4, height=5.3, title="context window")
        sys_msg = col.message("system", "identity · tools · workspace", mono=True)
        rem = col.message("user", "<system-reminder>\n# tinyshop\n- Run tests: pytest -q  …", mono=True)
        ask = col.message("user", TASK)
        call = col.message("tool_use", 'bash("pytest -q")', mono=True)
        p_sys, p_rem, p_ask, p_call = col.push(sys_msg), col.push(rem), col.push(ask), col.push(call)
        bracket = m.Line(sys_msg.get_corner(m.UL) + [-0.22, 0, 0], rem.get_corner(m.DL) + [-0.22, 0, 0]).set_stroke(MINT, 5)
        stable = code("stable", 20, MINT).next_to(bracket, m.LEFT, buff=0.15).set_y(bracket.get_y())
        li_cite = cite("Li, AI Agents in Depth · ch. 5").to_corner(m.DR, buff=0.4).shift(0.25 * m.DOWN)
        newcomer = text("Could a remote newcomer start work\nfrom the repo and its docs alone?", 22, INK)
        newcomer.next_to(md.box, m.DOWN, buff=0.3).align_to(md.box, m.LEFT)

        self.say(
            "[Some rules belong to the project, not to Otto.] [So tinyshop keeps them in a file]: [AGENTS.md.]",
            m.AnimationGroup(self.new_header("06 · project files", "A prompt for the project"), m.FadeIn(tree, shift=0.1 * m.RIGHT)),
            m.Circumscribe(tree[1], color=SUN, buff=0.06),
            m.FadeIn(md.group, target_position=tree[1]),
        )
        self.say(
            "[At the start of every session, the harness reads it into the context], [where it works as a project-level system prompt.]",
            m.AnimationGroup(m.FadeIn(col.frame), p_sys, m.FadeIn(rem, target_position=md.box)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(sys_msg, rem), color=ORCHID, buff=0.08), m.FadeIn(li_cite)),
        )
        self.say(
            "[Claude Code reads the same kind of file under its own name]; [many other agents read AGENTS.md.]",
            m.AnimationGroup(m.FadeIn(others[0], shift=0.1 * m.LEFT), m.FadeIn(tags[0])),
            m.AnimationGroup(m.FadeIn(others[1], shift=0.1 * m.LEFT), m.FadeIn(tags[1])),
        )
        self.say(
            "[The conversation grows below it], [but the file rarely changes], [so it's the cheapest stable prefix you can write.]",
            m.AnimationGroup(p_ask, p_call),
            m.Circumscribe(rem, color=MINT, buff=0.06),
            m.AnimationGroup(m.Create(bracket), m.FadeIn(stable)),
        )
        self.say(
            "[The book's test for the repo itself]: [could a remote newcomer start work from the repo and its docs alone?]",
            m.Circumscribe(md.box, color=SUN, buff=0.06),
            m.FadeIn(newcomer, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ constraints over guidance

    def constraints(self):
        self.next_section("7 · Constraints", notes=f"Li, AI Agents in Depth, ch. 5, 'Constraints over guidance': 'Rules that can be enforced with code should be encoded there, not merely suggested in documentation'; a linter rule or CI check means 'cannot be done', a prompt's 'please follow...' is merely 'advised against'. The check is ruff's T201 (print found); output measured with ruff 0.17.1: '{LINT_BEFORE[0]}', then '{LINT_AFTER[0]}'. Episode 18 (hooks) runs checks after every edit.")
        md = CodeCard("\n".join(AGENTS_MD), title="tinyshop/AGENTS.md", t2c=MD_T2C, accent=SUN)
        md.group.align_to([-6.75, 0, 0], m.LEFT).align_to([0, 2.45, 0], m.UP)
        rule_bar = md.bar(3, color=SUN)
        rule = md.lines[3]
        short_box = card(md.box.width, md.box.height - (md.base(2) - md.base(3)), SUN, CARD, radius=0.16, stroke=2).align_to(md.box, m.UL)

        cart = CodeCard("\n".join(CART_DEBUG), title="tinyshop/cart.py").raise_text()
        cart.group.align_to([-6.75, 0, 0], m.LEFT).next_to(md.box, m.DOWN, buff=0.35)
        cart.group.align_to([-6.75, 0, 0], m.LEFT)
        dy = cart.base(3) - cart.base(4)
        tail = m.VGroup(cart.lines[4], cart.lines[5])
        cart_short = card(cart.box.width, cart.box.height - dy, EDGE, CARD, radius=0.16, stroke=2).align_to(cart.box, m.UL)
        cart_tall = cart.box
        debug_line = cart.lines[3]
        debug_bar = cart.bar(3, color=ROSE, opacity=0.2)
        tail.shift(dy * m.UP)  # before the slip: the file without the print line
        before_head = m.VGroup(cart.cap, *cart.lines[:3])

        check_words = m.VGroup(code("lint check", 20, MINT, weight=m.BOLD), text("no print() in the code", 24, INK), code("ruff · rule T201", 20, MUTED))
        check_words.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        lock = icon(LOCK_ICON, 0.75)
        check_body = m.VGroup(lock, check_words).arrange(m.RIGHT, buff=0.35)
        c_box = card(6.1, check_body.height + 0.5, MINT, CARD, radius=0.18, stroke=2)
        check = m.VGroup(c_box, check_body.move_to(c_box).align_to(c_box, m.LEFT).shift(0.35 * m.RIGHT))
        check.align_to([6.75, 0, 0], m.RIGHT).align_to([0, 2.45, 0], m.UP)

        term = terminal(["$ ruff check .", "! " + LINT_BEFORE[0], "$ ruff check .", LINT_AFTER[0]], width=6.1, title="tinyshop")
        term[4][3].set_color(MINT)
        term.align_to(check, m.RIGHT).next_to(check, m.DOWN, buff=0.3)
        term.align_to([6.75, 0, 0], m.RIGHT)
        bot = otto(0.7).move_to([1.35, -2.55, 0])
        result = message("tool_result", LINT_BEFORE[0], width=4.6, size=20, mono=True)
        result.next_to(bot, m.RIGHT, buff=0.4)
        result.align_to([6.75, 0, 0], m.RIGHT)
        bot.next_to(result, m.LEFT, buff=0.75)
        result[1].set_fill(ROSE)
        back = arrow(result.get_left(), bot.get_right(), ROSE, buff=0.1)

        v_left = verdict("a rule in the prompt", "advised against", SUN, SUN_FILL, 6.1)
        v_left.align_to([-6.75, 0, 0], m.LEFT).set_y(-2.3)
        v_right = verdict("a rule in a check", "cannot be done", MINT, MINT_FILL, 6.1)
        v_right.align_to([6.75, 0, 0], m.RIGHT).set_y(-2.3)
        li_cite = cite("Li, AI Agents in Depth · ch. 5").next_to(v_left, m.DOWN, buff=0.22).align_to(v_left, m.LEFT)
        ep18 = pill("Episode 18 · Hooks", MINT, MINT_FILL, 20).next_to(v_right, m.DOWN, buff=0.2).align_to(v_right, m.RIGHT)

        self.say(
            "[One last principle]: [constraints over guidance.]",
            self.new_header("07 · constraints", "Constraints over guidance"),
            m.AnimationGroup(m.FadeIn(md.group, shift=0.15 * m.UP)),
        )
        self.say(
            "[AGENTS.md says]: [don't leave print calls in the code.]",
            m.Circumscribe(md.cap, color=SUN, buff=0.06),
            m.FadeIn(rule_bar),
        )
        self.say(
            "[That's guidance. Deep in a long task, Otto can miss it], [and a debug print slips into the cart file.]",
            m.AnimationGroup(m.FadeIn(cart_short), m.FadeIn(before_head), m.FadeIn(tail)),
            m.AnimationGroup(
                m.Transform(cart_short, cart_tall.copy()), tail.animate.shift(dy * m.DOWN),
                m.Succession(m.Wait(0.3), m.AnimationGroup(m.FadeIn(debug_line, shift=0.1 * m.RIGHT), m.FadeIn(debug_bar))),
            ),
        )
        self.say(
            "[So move the rule out of the prompt], [and into a linter check.]",
            m.AnimationGroup(
                m.FadeOut(rule, target_position=check.get_center(), scale=0.8), m.FadeOut(rule_bar),
                m.Transform(md.box, short_box),
            ),
            m.FadeIn(check, scale=0.9),
        )
        detach(md.group, *rule)
        self.say(
            "[Now the same slip] [fails the check], [and the error goes back to Otto as a tool result.]",
            m.Circumscribe(debug_line, color=ROSE, buff=0.06),
            m.AnimationGroup(m.FadeIn(term[:4]), m.FadeIn(term[4][0]), m.FadeIn(term[4][1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(bot, shift=0.15 * m.UP), m.FadeIn(result, target_position=term), m.GrowArrow(back)),
        )
        self.say(
            "[Otto deletes the line], [and the check passes.]",
            m.AnimationGroup(
                m.FadeOut(debug_line, shift=0.1 * m.LEFT), m.FadeOut(debug_bar),
                tail.animate.shift(dy * m.UP), m.Transform(cart_short, card(cart.box.width, cart.box.height - dy, EDGE, CARD, radius=0.16, stroke=2).align_to(cart.box, m.UL)),
            ),
            m.AnimationGroup(m.FadeIn(term[4][2], shift=0.1 * m.UP), m.FadeIn(term[4][3], shift=0.1 * m.UP)),
        )
        self.say(
            "[In the book's words], [a rule in the prompt is merely advised against.] [A rule in a check] [cannot be done.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(cart_short, before_head, tail, bot, result, back), shift=0.2 * m.DOWN),
                m.FadeIn(li_cite),
            ),
            m.FadeIn(v_left, shift=0.15 * m.UP),
            m.Circumscribe(check, color=MINT, buff=0.08),
            m.FadeIn(v_right, shift=0.15 * m.UP),
        )
        self.say(
            "[Episode eighteen runs checks like this after every edit, with hooks.]",
            m.FadeIn(ep18, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Assemble (sections from state), order (static first, dynamic last), enforce (rules as checks).")
        halo = glow(SUN, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Assemble", ORCHID), ("Order", SUN), ("Enforce", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("sections from state", "static first", "rules as checks")])
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
        if row_.width > 12.8:
            row_.scale(12.8 / row_.width)
        tagline = text("Build it from parts. Keep it stable. Enforce what matters.", 30, MUTED)
        tagline.next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [assemble], [order], [enforce.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Build the prompt from parts, keep the stable parts first, and turn the rules that matter into checks.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
