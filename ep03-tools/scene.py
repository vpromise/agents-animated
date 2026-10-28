"""Agents, Animated · Episode 3 · Tools: Giving the Model Hands  (Module 1 · Foundations)

The viewer learns how a model "calls" code it cannot run. Running example: Otto, the series' coding
agent in the tinyshop repo, gets read_file and edit_file without any change to its loop, reads
cart.py, fixes cart.total() (it skips the last item) and sees "13 passed".

Arc: the puzzle (a model only reads and writes text) → the schema (name, description, input_schema;
TOOLS[1] parsed from harness_reference.py) → the tool_use block (name, input, id; stop_reason
"tool_use") → dispatch (TOOL_HANDLERS, the handler, a tool_result with the same id, appended to
the messages) → the one changed line (run_bash(...) → TOOL_HANDLERS[block.name](**block.input));
adding a tool = one schema + one handler; Otto edits cart.py → parallel calls (several tool_use
blocks in one reply; Claude Code's batches of concurrency-safe calls) → large outputs (never cut
silently: head + tail + what was omitted + where it was saved) → coming up (Ep15, Ep16) → recap.

Colours (series roles): model/assistant CORAL, tool call SUN, tool result MINT, user AZURE, system
ORCHID; bad ROSE; the harness is neutral (EDGE/MUTED); headers CORAL (Module 1).

Facts (checked October 2026):
- learn-claude-code s02 (learn.shareai.run/en/s02): "The only change is in that one line of tool
  execution: run_bash() is replaced with TOOL_HANDLERS[block.name]() dispatch lookup"; "Adding a
  tool = one entry in TOOLS array + one line in TOOL_HANDLERS dict. The loop stays the same."
  Its "Dive into CC source" part (the site's reading of Claude Code's source, unverified): tools
  declare isConcurrencySafe(input); partitionToolCalls() groups consecutive safe calls, e.g.
  [read A, read B, glob] [bash rm x] [read C], batches strictly in order; results over a tool's
  maxResultSizeChars are persisted to disk and the model gets a preview plus the path.
- Li, AI Agents in Depth, ch. 4 (execution tools): head retention "the first 50 lines", tail
  retention "the last 50 lines", "... [8523 lines omitted, full output saved to
  /tmp/execution_output.txt] ..."; perception tools: "Silent truncation is dangerous".
- Li, ch. 5 (implementation tips): concurrency-safe should "default [to] no, fail-safe".
- Anthropic Messages API: a reply may hold several tool_use blocks; every tool_result goes back in
  the next user message, matched by tool_use_id.
Illustrative (invented for teaching, said "for example" or shown as an example): the 8,623-line
debug log, its 64-character average line, the log lines and the 2.31 s timing, the durations in the
batch chart, the ids toolu_01…04, the three parallel calls, the bash commands of episode two.
Every number on screen is computed below and asserted.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the shared narrator serving
"""

import ast
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the canonical code
# Parsed rather than imported: harness_reference.py imports the anthropic SDK.

REF = (SERIES / "harness_reference.py").read_text()
_TREE = ast.parse(REF)


def _assigned(name):
    for node in _TREE.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise KeyError(name)


TOOLS = ast.literal_eval(_assigned("TOOLS"))
_handlers = _assigned("TOOL_HANDLERS")
TOOL_HANDLERS = {ast.literal_eval(k): v.id for k, v in zip(_handlers.keys, _handlers.values)}
SCHEMA = {tool["name"]: tool for tool in TOOLS}
assert list(SCHEMA) == ["bash", "read_file", "edit_file"]
assert TOOL_HANDLERS == {"bash": "run_bash", "read_file": "read_file", "edit_file": "edit_file"}
assert set(SCHEMA["read_file"]) == {"name", "description", "input_schema"}  # the three fields explained
assert SCHEMA["read_file"]["input_schema"]["required"] == ["path"]  # "one path, a string, and it's required"
OLD_CALL = 'run_bash(block.input["command"])'  # episode two's loop: every call went to the shell
NEW_CALL = "TOOL_HANDLERS[block.name](**block.input)"  # the one changed line (learn-claude-code s02)
assert f"return {NEW_CALL}" in REF and "def run_bash(command)" in REF
CUT = int(re.search(r"def run_bash.*?\[:([\d_]+)\]", REF, re.S).group(1).replace("_", ""))
assert CUT == 50_000  # "keeps the first fifty thousand characters"

# ---------------------------------------------------------------------------- the running bug

CART_SRC = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s"
OLD_TEXT, NEW_TEXT = "range(len(items) - 1)", "range(len(items))"
assert CART_SRC.count(OLD_TEXT) == 1  # edit_file replaces the first match: it must be unique
FIXED_SRC = CART_SRC.replace(OLD_TEXT, NEW_TEXT, 1)


def _total(src):
    scope = {}
    exec(src, scope)
    return scope["total"]


PRICES = [3, 4]
assert _total(CART_SRC)(PRICES) == 3 and _total(FIXED_SRC)(PRICES) == 7  # "stops one item short"
FAILED, PASSED = 1, 12  # the series' running test result before the fix
TESTS = FAILED + PASSED
assert TESTS == 13  # "thirteen passed"
SED = "sed -i 's/ - 1//' cart.py"  # episode two's shell edit: removes the first " - 1"
assert re.sub(r" - 1", "", CART_SRC, count=1) == FIXED_SRC

# ---------------------------------------------------------------------------- parallel calls

PARALLEL = [('read_file("cart.py")', "toolu_02", "text of cart.py"),
            ('read_file("checkout.py")', "toolu_03", "text of checkout.py"),
            ('bash("ls tests")', "toolu_04", "names of the tests")]
# Claude Code's batching, as learn-claude-code s02 reads its source: (call, concurrency-safe?, duration)
CALLS = [("read A", True, 1.0), ("read B", True, 1.35), ("glob", True, 0.7), ("bash rm x", False, 0.6), ("read C", True, 0.85)]


def partition(calls):
    """Consecutive concurrency-safe calls share a batch; an unsafe call runs alone."""
    batches = []
    for call in calls:
        if call[1] and batches and batches[-1][0][1]:
            batches[-1].append(call)
        else:
            batches.append([call])
    return batches


BATCHES = partition(CALLS)
assert [[c[0] for c in b] for b in BATCHES] == [["read A", "read B", "glob"], ["bash rm x"], ["read C"]]

# ---------------------------------------------------------------------------- a large output (illustrative)

LOG_LINES = 8623
HEAD = TAIL = 50  # Li, ch. 4: the first 50 and the last 50 lines
OMITTED = LOG_LINES - HEAD - TAIL
assert OMITTED == 8523  # Li's example marker: "... [8523 lines omitted, …] ..."
AVG_CHARS = 64  # average line length, newline included (illustrative)
LOG_CHARS = LOG_LINES * AVG_CHARS
KEPT = CUT / LOG_CHARS
assert LOG_CHARS == 551_872 and round(100 * KEPT) == 9  # "the first 50,000 characters, about 9%"
SAVED = "/tmp/execution_output.txt"
MARKER = f"... [{OMITTED} lines omitted, full output saved to {SAVED}] ..."
VERDICT = f"{TESTS} passed in 2.31s"


def spell(n):
    """A whole number as the narrator should say it (below a million)."""
    if n < 100:
        return say_number(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{say_number(h)} hundred" + (f" and {say_number(r)}" if r else "")
    t, r = divmod(n, 1000)
    return f"{spell(t)} thousand" + ((" and " if r < 100 else " ") + spell(r) if r else "")


assert spell(LOG_LINES) == "eight thousand six hundred and twenty-three" and spell(CUT) == "fifty thousand"

# ---------------------------------------------------------------------------- helpers
# ManimGX's t2c takes literal substrings, not regexes, so kit's regex colouring of code and JSON
# does nothing; paint() colours the glyphs of regex matches directly.

PY_RULES = [
    (r"\b(def|while|for|in|if|else|return|import|from|not|and|or|is|None|True|False)\b", ORCHID),
    (r"\b\d[\d_]*\b", SUN),
    (r"\b[a-z_][a-z0-9_]*(?=\()", AZURE),
    (r"\"[^\"\n]*\"|'[^'\n]*'", MINT),
    (r"#[^\n]*", FAINT),
]
JSON_RULES = [
    (r"\b-?\d+(\.\d+)?\b", SUN),
    (r"\b(true|false|null)\b", ORCHID),
    (r"\"[^\"\n]*\"", MINT),
    (r"\"[^\"\n]*\"(?=\s*:)", AZURE),
]


def paint(words, src, rules):
    """Colour the glyphs of `words` (a Text of `src`) that match each (regex, colour); later rules win."""
    index, k = [], 0
    for ch in src:
        index.append(None if ch.isspace() else k)
        k += not ch.isspace()
    assert k == len(words), (k, len(words))
    for pattern, color in rules:
        for match in re.finditer(pattern, src):
            for i in range(match.start(), match.end()):
                if index[i] is not None:
                    words[index[i]].set_color(color)
    return words


def py(src, size=20, color=INK):
    return paint(code(src, size, color), src, PY_RULES)


def py_block(src, size=20, width=None, title=None, accent=EDGE):
    """kit.code_block with working highlighting: VGroup(box, body[, title])."""
    body = py(src, size)
    w = max(width or 0, body.width + 0.7)
    top = 0.0
    cap = None
    if title:
        cap = code(title, 20, MUTED)
        top = cap.height + 0.25
    box = card(w, body.height + 0.6 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
        group.add(cap)
    return group


def compact(obj, inline_from=2, depth=0):
    """JSON with one key per line down to `inline_from` levels; deeper values on one line."""
    if isinstance(obj, dict) and depth < inline_from:
        pad = "  " * (depth + 1)
        rows = [f"{pad}{json.dumps(k)}: {compact(v, inline_from, depth + 1)}" for k, v in obj.items()]
        return "{\n" + ",\n".join(rows) + "\n" + "  " * depth + "}"
    return json.dumps(obj)


def json_block(src, accent, title, size=19, width=None, fill=CARD):
    """kit.json_card for a pre-formatted JSON string: VGroup(box, body, title)."""
    body = paint(code(src, size, INK), src, JSON_RULES)
    cap = code(title, 20, accent, weight=m.BOLD)
    top = cap.height + 0.22
    w = max(width or 0, body.width + 0.7, cap.width + 0.7)
    box = card(w, body.height + 0.55 + top, accent, fill, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
    return m.VGroup(box, body, cap)


def cite(s):
    return text(s, 20, MUTED)


def band(target, box, color=SUN_FILL, pad=0.08, height=None):
    """A highlight bar behind some lines of a card (drawn under the text, above the card)."""
    h = height or target.height + 2 * pad
    bar = m.RoundedRectangle(corner_radius=0.08, width=box.width - 0.24, height=h, stroke_width=0,
                             fill_color=color, fill_opacity=1)
    return bar.move_to([box.get_x(), target.get_y(), 0]).set_z_index(1)


def tool_tag(name, size=24, width=3.3, height=0.95):
    """A tool as Otto holds it: wrench + name (SUN)."""
    box = card(width, height, SUN, SUN_FILL, radius=0.2, stroke=2.5)
    body = m.VGroup(icon("ep03-wrench", 0.55), code(name, size, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.28)
    return m.VGroup(box, body.move_to(box))


def schema_chip(name, width=None):
    """A schema in the TOOLS list, folded to its name."""
    body = m.VGroup(icon("ep03-wrench", 0.36), code(name, 20, INK)).arrange(m.RIGHT, buff=0.22)
    box = card(width or body.width + 0.55, 0.62, SUN, CARD, radius=0.14, stroke=2)
    return m.VGroup(box, body.move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT))


def row_card(left, right, color, fill=CARD, width=None, height=0.62, gap=0.35):
    """A one-line card: `left` from the left edge, `right` against the right edge."""
    w = width or left.width + right.width + gap + 0.6
    box = card(w, height, color, fill, radius=0.14, stroke=2)
    left.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    right.move_to(box).align_to(box, m.RIGHT).shift(0.3 * m.LEFT)
    return m.VGroup(box, left, right)


def model_node():
    box = card(2.5, 2.3, CORAL, CORAL_FILL)
    body = m.VGroup(icon(recolored("llm", CORAL, CORAL_FILL, "ep03"), 1.0), text("model", 28, INK, weight=m.BOLD))
    return m.VGroup(box, body.arrange(m.DOWN, buff=0.22).move_to(box))


def harness_node(width=3.0, height=2.5):
    box = card(width, height, MUTED, CARD, stroke=2.5)
    body = m.VGroup(icon(recolored("gear", MUTED, CARD_HI, "ep03"), 0.95), text("harness", 28, INK, weight=m.BOLD))
    return m.VGroup(box, body.arrange(m.DOWN, buff=0.22).move_to(box))


# ---------------------------------------------------------------------------- the episode


class Ep03Tools(Episode):
    number = 3

    def lesson(self):
        self.hook()
        self.schema()
        call = self.tool_use()
        self.dispatch(call)
        self.one_line()
        self.parallel()
        self.large_outputs()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="A language model only reads and writes text: it cannot open cart.py or run pytest. Yet in Ep02 "
            "Otto read, edited and tested cart.py, all through one tool, bash (learn-claude-code s01). Today: "
            "read_file and edit_file, with the loop unchanged. The bash commands are illustrative.",
        )
        ask = message("user", "The checkout test fails. Fix it.", width=4.2, size=22)
        reply = message("assistant", "Try opening cart.py and\nchecking total().", width=4.2, size=22)
        brain = model_node()
        m.VGroup(ask, brain, reply).arrange(m.RIGHT, buff=0.75).move_to([0, 0.8, 0])
        a_in = arrow(ask.get_right(), brain.get_left(), AZURE)
        a_out = arrow(brain.get_right(), reply.get_left(), CORAL)

        def cant(content):
            box = card(4.1, 1.0, ROSE, ROSE_FILL, radius=0.2, stroke=2)
            content.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            cross = text("✗", 32, ROSE, weight=m.BOLD).move_to(box).align_to(box, m.RIGHT).shift(0.35 * m.LEFT)
            return m.VGroup(box, content, cross)

        no_file = cant(m.VGroup(icon("doc", 0.62), code("open cart.py", 22, INK)).arrange(m.RIGHT, buff=0.28))
        no_test = cant(m.VGroup(code(">_", 26, MINT, weight=m.BOLD), code("run pytest -q", 22, INK)).arrange(m.RIGHT, buff=0.3))
        cants = m.VGroup(no_file, no_test).arrange(m.RIGHT, buff=0.7).move_to([0, -1.75, 0])

        bot = otto(1.3).move_to([-5.15, 0.25, 0])
        shell = m.VGroup()
        for cmd in ("cat cart.py", SED, "pytest -q"):
            shell.add(row_card(code("bash", 20, SUN, weight=m.BOLD), code(cmd, 22, INK), SUN, width=5.6, height=0.72))
        for c in shell:  # the command sits right after the tool name, not against the edge
            c[2].next_to(c[1], m.RIGHT, buff=0.4)
        shell.arrange(m.DOWN, buff=0.3).move_to([-0.65, 0.25, 0])
        brace = m.Brace(shell, m.RIGHT, color=SUN)
        one = text("one tool:\nthe shell", 26, SUN, weight=m.SEMIBOLD).next_to(brace, m.RIGHT, buff=0.3)

        q1 = title_text("How does a model call code", 60)
        q2 = title_text("it cannot run?", 60, SUN)
        question = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.35).move_to([0, 0.2, 0])

        hero = otto(1.5).move_to([0, 0.75, 0])
        y_arm = hero[0].get_y()
        hand_l = tool_tag("read_file").move_to([-3.75, y_arm, 0])
        hand_r = tool_tag("edit_file").move_to([3.75, y_arm, 0])
        arms = m.VGroup(
            m.Line(hero[0].get_left(), hand_l.get_right()), m.Line(hero[0].get_right(), hand_r.get_left())
        ).set_stroke(SUN, 4)
        loop_chip = chip("agent_loop()", EDGE, CARD, width=3.1, height=0.7, size=24, mono=True)
        same = pill("unchanged", MINT, MINT_FILL, 22)
        loop_row = m.VGroup(loop_chip, same).arrange(m.RIGHT, buff=0.35).move_to([0, -1.75, 0])

        self.say(
            "[A language model does one thing]: [it reads text], [and it writes text.]",
            m.AnimationGroup(self.new_header("03 · tools", "Text in, text out"), m.FadeIn(brain, scale=0.9)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.3 * m.RIGHT), m.GrowArrow(a_in)),
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(reply, shift=0.3 * m.RIGHT)),
        )
        self.say(
            "[It cannot open a file.] [It cannot run a test.]",
            m.FadeIn(no_file, shift=0.25 * m.UP),
            m.FadeIn(no_test, shift=0.25 * m.UP),
        )
        self.say(
            "[Yet in episode two, Otto read the cart file], [edited it], [and ran the tests], [all with one tool: the shell.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(ask, brain, reply, a_in, a_out, cants), shift=0.2 * m.DOWN),
                m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(shell[0], shift=0.2 * m.RIGHT),
            ),
            m.FadeIn(shell[1], shift=0.2 * m.RIGHT),
            m.FadeIn(shell[2], shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.GrowFromCenter(brace), m.FadeIn(one, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[So how does a model call code] [that it cannot run?]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bot, shell, brace, one)), m.FadeIn(q1, shift=0.3 * m.UP)),
            m.FadeIn(q2, shift=0.3 * m.UP),
        )
        self.wait(0.5)
        self.say(
            "[Today, Otto gets two new tools: one reads files, one edits them], [and its loop doesn't change at all.]",
            m.AnimationGroup(
                m.FadeOut(question, shift=0.3 * m.UP),
                m.FadeIn(hero, scale=0.85),
                m.LaggedStart(m.AnimationGroup(m.Create(arms[0]), m.FadeIn(hand_l, shift=0.3 * m.LEFT)),
                              m.AnimationGroup(m.Create(arms[1]), m.FadeIn(hand_r, shift=0.3 * m.RIGHT)), lag_ratio=0.4),
            ),
            m.AnimationGroup(m.FadeIn(loop_chip, shift=0.2 * m.UP), m.FadeIn(same, scale=1.2)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ the schema

    def schema(self):
        self.next_section(
            "1 · Schema",
            notes="A tool is described to the model by a schema: name, description, input_schema (JSON Schema). "
            "Shown verbatim: harness_reference.TOOLS[1] (read_file). The schemas are sent with every request "
            "(client.messages.create(..., tools=TOOLS)), so the model sees its tools every turn.",
        )
        src = compact(SCHEMA["read_file"])
        sch = json_block(src, SUN, "TOOLS[1]")
        sch.move_to([0, 0.05, 0]).align_to([-6.75, 0, 0], m.LEFT)
        box, body = sch[0], sch[1]
        desc = json.dumps(SCHEMA["read_file"]["description"])
        f_name = glyphs(body, '"name": "read_file",')
        f_desc = glyphs(body, f'"description": {desc},')
        f_input = glyphs(body, src[src.index('"input_schema"') : src.rindex("}")])
        index_of = {id(g): i for i, g in enumerate(body)}

        def focus(field):  # light one field, dim the rest (one Transform of the whole text)
            lit = {index_of[id(g)] for g in field}
            target = body.copy()
            for i, g in enumerate(target):
                g.set_opacity(1 if i in lit else 0.32)
            return m.Transform(body, target)

        def gutter(field):  # a bar in the card's left margin, beside the field
            return m.RoundedRectangle(corner_radius=0.03, width=0.08, height=field.height + 0.1, stroke_width=0,
                                      fill_color=SUN, fill_opacity=1).move_to([box.get_left()[0] + 0.18, field.get_y(), 0])

        gut = gutter(f_name)

        notes = (("name", "the word the model writes\nto ask for this tool"),
                 ("description", "what the tool does, in\nwords the model reads"),
                 ("input_schema", "its arguments: names,\ntypes, which are required"))
        x0 = box.get_right()[0] + 0.65
        annots = m.VGroup()
        for (key, words), y in zip(notes, (1.85, 0.45, -1.25)):
            a = m.VGroup(code(key, 22, AZURE, weight=m.BOLD), text(words, 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            annots.add(a.move_to([0, y, 0]).align_to([x0, 0, 0], m.LEFT))
        links = m.VGroup(*[
            m.Line([box.get_right()[0] + 0.08, f.get_y(), 0], [x0 - 0.12, a[0].get_y(), 0]).set_stroke(SUN, 2)
            for f, a in zip((f_name, f_desc, f_input), annots)
        ])
        handler_src = "def read_file(path):\n    return open(path).read()"
        assert handler_src in REF  # the handler, as harness_reference.py writes it
        handler = py_block(handler_src, 20).move_to([0, -2.78, 0]).align_to([-6.75, 0, 0], m.LEFT)
        unsent = pill("never sent to the model", ROSE, ROSE_FILL, 22).next_to(handler, m.RIGHT, buff=0.45)
        props = glyphs(body, json.dumps(SCHEMA["read_file"]["input_schema"]["properties"]))
        required = glyphs(body, json.dumps(SCHEMA["read_file"]["input_schema"]["required"]))

        # the request: schemas travel next to the messages
        req_src = "response = client.messages.create(\n    model=MODEL,\n    system=SYSTEM,\n    messages=messages,\n    tools=TOOLS,\n)"
        req = py_block(req_src, 20, title="every turn").move_to([0.95, 0.15, 0])
        line_msgs = glyphs(req[1], "messages=messages,")
        line_tools = glyphs(req[1], "tools=TOOLS,")

        def panel(title_, items, center):
            inner = m.VGroup(code(title_, 20, MUTED), *items).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
            frame = card(3.85, inner.height + 0.5, EDGE, "#18150F", radius=0.18, stroke=2).move_to(center)
            inner.move_to(frame)
            return m.VGroup(frame, inner)

        msgs = panel("messages", [message("system", "You are Otto, a coding agent.", width=3.45, size=20),
                                  message("user", "The checkout test fails. Fix it.", width=3.45, size=20)], [-4.85, 1.3, 0])
        chip_bash, chip_read = schema_chip("bash", 3.45), schema_chip("read_file", 3.45)
        tools = panel("TOOLS", [chip_bash, chip_read], [-4.85, -1.55, 0])
        chip_read.set_z_index(1)  # it arrives (from the schema card) before its panel: keep it on top
        a_msgs = arrow([msgs.get_right()[0], msgs.get_y(), 0], [req.get_left()[0], line_msgs.get_y(), 0], AZURE, buff=0.08)
        a_tools = arrow([tools.get_right()[0], tools.get_y(), 0], [req.get_left()[0], line_tools.get_y(), 0], SUN, buff=0.08)
        bot = otto(1.1).move_to([5.75, 0.35, 0])
        a_bot = arrow(req.get_right(), [bot.get_left()[0], req.get_y(), 0], SUN, buff=0.12)
        wonder = text("which tool,\nif any?", 22, MUTED).next_to(bot, m.DOWN, buff=0.35)

        self.say(
            "[First, the model must learn that a tool exists.] [The harness describes it in a schema], [a small piece of JSON.]",
            self.new_header("1 · schema", "A tool is a description"),
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(sch[2])),
            m.AddTextLetterByLetter(body, time_per_char=0.012),
        )
        self.say(
            "[The name is what the model writes] [when it wants this tool.]",
            m.AnimationGroup(m.FadeIn(gut), focus(f_name)),
            m.AnimationGroup(m.Create(links[0]), m.FadeIn(annots[0], shift=0.2 * m.LEFT)),
        )
        self.say(
            "[The description says what the tool does], [in plain words the model reads like documentation.]",
            m.AnimationGroup(m.Transform(gut, gutter(f_desc)), focus(f_desc)),
            m.AnimationGroup(m.Create(links[1]), m.FadeIn(annots[1], shift=0.2 * m.LEFT)),
        )
        self.say(
            "[The input schema lists the arguments]: [here, one path, a string, and it's required.]",
            m.AnimationGroup(m.Transform(gut, gutter(f_input)), focus(f_input)),
            m.AnimationGroup(
                m.Create(links[2]), m.FadeIn(annots[2], shift=0.2 * m.LEFT),
                m.Succession(m.Circumscribe(props, color=SUN, buff=0.05), m.Circumscribe(required, color=SUN, buff=0.05)),
            ),
        )
        self.say(
            "[Notice what's missing]: [the model never sees the handler's code], [only this description.]",
            m.FadeIn(handler, shift=0.2 * m.UP),
            m.FadeIn(unsent, shift=0.2 * m.LEFT),
            m.Circumscribe(box, color=SUN, buff=0.06),
        )
        self.say(
            "[The schemas go out with every request], [right next to the messages.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(gut, links, annots, handler, unsent)),
                m.FadeTransform(sch, chip_read),
                m.FadeIn(tools[0]), m.FadeIn(tools[1][:2]),
                m.FadeIn(req, shift=0.2 * m.UP), m.GrowArrow(a_tools),
            ),
            m.AnimationGroup(m.FadeIn(msgs, shift=0.2 * m.RIGHT), m.GrowArrow(a_msgs)),
        )
        self.say(
            "[So on every turn, the model reads its tools] [before it decides what to do.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.LEFT), m.GrowArrow(a_bot), m.Circumscribe(tools, color=SUN, buff=0.06)),
            m.AnimationGroup(pulse(a_bot, SUN), m.FadeIn(wonder, shift=0.15 * m.UP)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ the tool_use block

    def tool_use(self):
        self.next_section(
            "2 · tool_use",
            notes="The model never runs code: it ends its reply with a tool_use content block (type, id, name, "
            "input) and stop_reason \"tool_use\", then waits. The harness, ordinary code on your machine, "
            "receives the block. The id toolu_01 is an example (real ids are longer).",
        )
        bot = otto(1.2).move_to([-5.55, -0.1, 0])
        thought = bubble(text("Let me look at cart.py.", 22, INK), CORAL, CORAL_FILL, tail="left")
        thought.next_to(bot, m.UP, buff=0.45).align_to([-6.6, 0, 0], m.LEFT)
        call_obj = {"type": "tool_use", "id": "toolu_01", "name": "read_file", "input": {"path": "cart.py"}}
        call = json_block(compact(call_obj, inline_from=1), SUN, "tool_use").move_to([-0.95, -0.1, 0])
        body = call[1]
        g_name = glyphs(body, '"name": "read_file"')
        g_input = glyphs(body, '"input": {"path": "cart.py"}')
        g_id = glyphs(body, '"id": "toolu_01"')
        no_run = pill("✗ runs no code", ROSE, ROSE_FILL, 22).next_to(bot, m.DOWN, buff=0.45)
        waiting = text("waiting…", 24, MUTED).move_to(no_run)
        stop = pill('stop_reason: "tool_use"', SUN, SUN_FILL, 22, mono=True).next_to(call, m.DOWN, buff=0.4)
        harness = harness_node().move_to([4.85, -0.1, 0])
        sub = text("ordinary code,\non your machine", 22, MUTED).next_to(harness, m.DOWN, buff=0.35)
        a_go = arrow(call.get_right(), harness.get_left(), SUN, buff=0.12)

        self.say(
            "[Now Otto wants] [to see the cart file.]",
            m.AnimationGroup(self.new_header("2 · tool_use", "The model writes a request"), m.FadeIn(bot, shift=0.2 * m.UP)),
            m.FadeIn(thought, shift=0.2 * m.UP),
        )
        self.say(
            "[It doesn't run anything.] [It writes a tool use block]: [a name, an input, and an ID.]",
            m.FadeIn(no_run, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(call[0]), m.FadeIn(call[2]), m.AddTextLetterByLetter(body, time_per_char=0.015)),
            m.Succession(*[m.Circumscribe(g, color=SUN, buff=0.05, run_time=0.8) for g in (g_name, g_input, g_id)]),
        )
        self.say(
            "[The reply ends with the stop reason tool use], [and the model waits.]",
            m.FadeIn(stop, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeOut(no_run), m.FadeIn(waiting), bot[1].animate.stretch(0.3, 1)),
        )
        self.say(
            "[The block goes to the harness]: [ordinary code, running on your machine.]",
            m.Succession(m.AnimationGroup(m.GrowArrow(a_go), m.FadeIn(harness, shift=0.2 * m.LEFT)), pulse(a_go, SUN)),
            m.AnimationGroup(m.FadeIn(sub, shift=0.15 * m.UP), m.Circumscribe(harness, color=MUTED, buff=0.08)),
        )
        self.wait(0.3)
        self.clear(call)
        return call

    # ------------------------------------------------------------------ dispatch

    def dispatch(self, call):
        self.next_section(
            "3 · Dispatch",
            notes="The harness looks the block's name up in TOOL_HANDLERS (the dispatch table) and calls the "
            "handler with the input as keyword arguments: read_file(path=\"cart.py\"). The output comes back "
            "as a tool_result with tool_use_id equal to the call's id, appended to messages as the next user "
            "turn; on its next turn the model reads it. cart.py is the series' running bug (harness_reference.py).",
        )
        cbody = call[1]
        c_name = glyphs(cbody, '"name": "read_file"')
        c_id = glyphs(cbody, '"toolu_01"')
        call_target = call.copy().move_to([-4.15, 1.15, 0])

        table_src = 'TOOL_HANDLERS = {\n    "bash": run_bash,\n    "read_file": read_file,\n}'
        table = py_block(table_src, 20, title="dispatch table").move_to([2.9, 1.2, 0])
        t_body = table[1]
        t_open, t_bash = glyphs(t_body, "TOOL_HANDLERS = {"), glyphs(t_body, '"bash": run_bash,')
        t_read, t_close = glyphs(t_body, '"read_file": read_file,'), glyphs(t_body, "}")
        t_body.set_z_index(2)
        row_hl = band(t_read, table[0], SUN_FILL, height=0.42)
        hcall_words = py('read_file(path="cart.py")', 22)
        hbox = card(hcall_words.width + 0.6, 0.72, EDGE, CARD_HI, radius=0.14, stroke=2)
        hcall = m.VGroup(hbox, hcall_words.move_to(hbox)).move_to([2.9, -1.05, 0])
        output = py_block(CART_SRC, 20, title="cart.py").move_to([-3.65, -1.95, 0]).set_z_index(2)

        # layout after the call shrinks
        c_name_t = glyphs(call_target[1], '"name": "read_file"')
        a_look = arrow([call_target.get_right()[0], c_name_t.get_y(), 0], [table.get_left()[0], t_read.get_y(), 0], SUN, buff=0.1)
        a_run = arrow(table.get_bottom(), hcall.get_top(), MUTED, buff=0.08)
        a_out = arrow(hcall.get_left(), [output.get_right()[0], hcall.get_y(), 0], MINT, buff=0.08)

        # the tool_result: JSON head, the output as its content, closing brace
        head_src = '{\n  "type": "tool_result",\n  "tool_use_id": "toolu_01",\n  "content":'
        r_head = paint(code(head_src, 19, INK), head_src, JSON_RULES)
        r_cap = code("tool_result", 20, MINT, weight=m.BOLD)
        r_close = code("}", 19, INK)
        slot = output.copy().scale(0.82)
        inner = m.VGroup(r_cap, r_head, slot, r_close).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
        slot.shift(0.45 * m.RIGHT)
        r_box = card(inner.width + 0.8, inner.height + 0.5, MINT, CARD, radius=0.16, stroke=2).move_to(inner)
        result = m.VGroup(r_box, r_cap, r_head, slot, r_close)
        r_id = glyphs(r_head, '"toolu_01"')
        call_low = call_target.copy().shift(0.95 * m.DOWN)  # both cards sit lower once the table is gone
        c_id_t = glyphs(call_low[1], '"toolu_01"')
        result.move_to([2.85, 0, 0]).shift((c_id_t.get_y() - r_id.get_y()) * m.UP)
        link = m.DashedLine([call_low.get_right()[0] + 0.06, r_id.get_y(), 0], [r_box.get_left()[0] - 0.06, r_id.get_y(), 0],
                            dash_length=0.12).set_stroke(MUTED, 3)
        same = pill("same ID", MUTED, CARD_HI, 20).next_to(link, m.UP, buff=0.18)

        # the context, one turn later
        ctx = ContextColumn(center=(3.7, -0.4, 0), width=5.7, height=5.8)
        pushes = [ctx.push(ctx.message("user", "The checkout test fails. Fix it.")),
                  ctx.push(ctx.message("tool_use", 'read_file(path="cart.py")', mono=True))]
        tr_msg = ctx.message("tool_result", CART_SRC, mono=True)
        pushes.append(ctx.push(tr_msg))
        found = ctx.message("assistant", "Found it: total() stops one item short.")
        push_found = ctx.push(found)
        bug = glyphs(tr_msg[2][1], "- 1")
        bot = otto(1.5).move_to([-3.6, 0.35, 0])
        a_read = arrow([ctx.panel.get_left()[0], bot.get_y(), 0], bot.get_right(), MINT, buff=0.2)
        next_turn = text("next request", 22, MUTED).next_to(a_read, m.UP, buff=0.15)

        self.say(
            "[The harness keeps a dispatch table]: [a dictionary from tool names to Python functions.]",
            m.AnimationGroup(
                self.new_header("3 · dispatch", "Look up the name, run the handler"),
                m.Transform(call, call_target),
                m.FadeIn(table[0]), m.FadeIn(table[2]), m.FadeIn(t_open), m.FadeIn(t_close),
            ),
            m.LaggedStart(m.FadeIn(t_bash, shift=0.15 * m.RIGHT), m.FadeIn(t_read, shift=0.15 * m.RIGHT), lag_ratio=0.5),
        )
        self.say(
            "[It finds that name in the table], [and calls the handler with the block's input as arguments.]",
            m.AnimationGroup(m.GrowArrow(a_look), m.FadeIn(row_hl), m.Circumscribe(c_name, color=SUN, buff=0.05)),
            m.AnimationGroup(m.GrowArrow(a_run), m.FadeIn(hcall, shift=0.2 * m.DOWN)),
        )
        self.say(
            "[The handler opens the file], [and its text becomes the output.]",
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(output[0]), m.FadeIn(output[2])),
            m.AddTextLetterByLetter(output[1], time_per_char=0.02),
        )
        self.say(
            "[The harness wraps it in a tool result], [with the same ID as the call.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(table, row_hl, hcall, a_look, a_run, a_out, t_open, t_bash, t_read, t_close)),
                m.Transform(output, slot), m.Transform(call, call_low),
                m.FadeIn(r_box), m.FadeIn(r_cap), m.FadeIn(r_head), m.FadeIn(r_close),
            ),
            m.AnimationGroup(m.Circumscribe(c_id, color=SUN, buff=0.05), m.Circumscribe(r_id, color=MINT, buff=0.05)),
        )
        self.say(
            "[That ID is how the model matches each result to its request.]",
            m.AnimationGroup(m.Create(link), m.FadeIn(same, shift=0.1 * m.UP)),
        )
        self.wait(0.4)
        self.say(
            "[The result is appended to the messages], [and on its next turn, Otto spots the bug]: [total stops one item short.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(call, output, r_box, r_cap, r_head, r_close, link, same)),
                m.FadeIn(ctx.frame), m.LaggedStart(*pushes, lag_ratio=0.45),
            ),
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.RIGHT), m.GrowArrow(a_read), m.FadeIn(next_turn), push_found),
            m.Circumscribe(bug, color=ROSE, buff=0.06),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ one line changes

    def one_line(self):
        self.next_section(
            "4 · One line changes",
            notes="learn-claude-code s01 → s02: the loop's tool call changes from run_bash(...) to "
            "TOOL_HANDLERS[block.name](**block.input) (run_tool in harness_reference.py); adding a tool = one "
            "schema in TOOLS + one handler in TOOL_HANDLERS; the loop stays the same. Otto then fixes cart.py "
            "with edit_file (old_text must match exactly once): total([3, 4]) goes from 3 to 7.",
        )
        loop_old = ("while True:\n    response = client.messages.create(...)\n    ...\n    results = []\n"
                    "    for block in response.content:\n        if block.type == \"tool_use\":\n"
                    f"            output = {OLD_CALL}\n            results.append({{..., \"content\": output}})\n"
                    "    messages.append({\"role\": \"user\", \"content\": results})")
        loop_new = loop_old.replace(OLD_CALL, NEW_CALL)
        assert loop_new.count(NEW_CALL) == 1
        width = max(py_block(loop_old, 20).width, py_block(loop_new, 20).width)
        old = py_block(loop_old, 20, width).move_to([-0.95, 0.75, 0]).align_to([0, 2.55, 0], m.UP)
        new = py_block(loop_new, 20, width).move_to(old)
        old_part, new_part = glyphs(old[1], OLD_CALL), glyphs(new[1], NEW_CALL)
        old_line = glyphs(old[1], f"output = {OLD_CALL}")

        def mark(color):  # a bar in the card's left margin, beside the changing line
            return m.RoundedRectangle(corner_radius=0.03, width=0.08, height=old_line.height + 0.1, stroke_width=0,
                                      fill_color=color, fill_opacity=1).move_to([old[0].get_left()[0] + 0.18, old_line.get_y(), 0])

        line_hl = mark(ROSE)
        unchanged = pill("unchanged", MINT, MINT_FILL, 22).move_to([old[0].get_right()[0] + 1.3, old.get_y(), 0])

        chips_ = [schema_chip(n) for n in SCHEMA]
        t_cap = code("TOOLS", 20, MUTED)
        t_inner = m.VGroup(t_cap, m.VGroup(*chips_).arrange(m.RIGHT, buff=0.2)).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        t_frame = card(t_inner.width + 0.6, t_inner.height + 0.45, EDGE, "#18150F", radius=0.18, stroke=2).move_to(t_inner)
        h_rows = [py(f'"{k}": {v},', 20) for k, v in TOOL_HANDLERS.items()]
        h_cap = code("TOOL_HANDLERS", 20, MUTED)
        h_inner = m.VGroup(h_cap, *h_rows).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        h_frame = card(h_inner.width + 0.6, h_inner.height + 0.45, EDGE, "#18150F", radius=0.18, stroke=2).move_to(h_inner)
        tools_p, hand_p = m.VGroup(t_frame, t_inner), m.VGroup(h_frame, h_inner)
        m.VGroup(tools_p, hand_p).arrange(m.RIGHT, buff=0.4, aligned_edge=m.UP).move_to([-0.4, 0, 0]).align_to(
            [0, old.get_bottom()[1] - 0.3, 0], m.UP)

        # the fix, with the new tool
        edit_obj = {"type": "tool_use", "name": "edit_file",
                    "input": {"path": "cart.py", "old_text": OLD_TEXT, "new_text": NEW_TEXT}}
        assert list(edit_obj["input"]) == SCHEMA["edit_file"]["input_schema"]["required"]
        edit = json_block(compact(edit_obj), SUN, "tool_use").move_to([-3.5, 0.15, 0])
        before = py_block(CART_SRC, 20, title="cart.py").move_to([3.7, 1.0, 0])
        after = py_block(FIXED_SRC, 20, before[0].width, title="cart.py").move_to(before)
        changed = glyphs(before[1], "for i in range(len(items) - 1):")
        minus = glyphs(before[1], "- 1")
        gone = {id(g) for g in minus}
        keep = m.VGroup(*[g for g in changed if id(g) not in gone])
        fixed_line = glyphs(after[1], "for i in range(len(items)):")
        assert len(keep) == len(fixed_line)
        assert 'f"edited {path}"' in REF
        res = json_block(compact({"type": "tool_result", "content": "edited cart.py"}), MINT, "tool_result")
        res.next_to(before, m.DOWN, buff=0.4)
        a_edit = arrow(edit.get_right(), [before.get_left()[0], edit.get_right()[1], 0], SUN, buff=0.12)
        check = code(f"total({PRICES}) == {_total(FIXED_SRC)(PRICES)}", 22, MINT, weight=m.BOLD)
        check.next_to(edit, m.DOWN, buff=0.4)

        self.say(
            "[Here is the loop from episode two], [when the shell was Otto's only tool.]",
            m.AnimationGroup(self.new_header("4 · the loop", "One line changes"), m.FadeIn(old, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(t_frame), m.FadeIn(t_cap), m.FadeIn(chips_[0], shift=0.15 * m.UP)),
        )
        self.say(
            "[Every tool call went straight to run bash.]",
            m.AnimationGroup(m.FadeIn(line_hl), old_part.animate.set_color(ROSE)),
        )
        self.say(
            "[To support many tools, exactly one line changes]: [look up the handler by name, and call it.]",
            m.AnimationGroup(
                old_part.animate.shift(0.3 * m.DOWN).set_opacity(0),
                m.FadeIn(new_part, shift=0.3 * m.DOWN),
                m.Transform(line_hl, mark(MINT)),
            ),
            m.AnimationGroup(
                m.FadeIn(h_frame), m.FadeIn(h_cap), m.FadeIn(h_rows[0]), m.FadeIn(h_rows[1]),
                m.FadeIn(chips_[1], shift=0.15 * m.UP), m.Circumscribe(new_part, color=MINT, buff=0.05),
            ),
        )
        self.remove(*old.get_family(), *old_part.get_family(), *new_part.get_family())  # same picture, one block
        self.add(new)
        self.wait(0.5)
        self.say(
            "[Now adding edit file takes two pieces]: [one schema in the tools list], [and one handler in the table.]",
            m.AnimationGroup(m.Circumscribe(tools_p, color=SUN, buff=0.06), m.Circumscribe(hand_p, color=SUN, buff=0.06)),
            m.FadeIn(chips_[2], shift=0.2 * m.UP),
            m.FadeIn(h_rows[2], shift=0.2 * m.RIGHT),
        )
        self.say(
            "[The loop itself stays exactly the same.]",
            m.AnimationGroup(m.FadeIn(unchanged, scale=1.15), m.Circumscribe(new[0], color=MINT, buff=0.06)),
        )
        self.wait(0.8)
        self.clear()
        # parts of `before`, which only fades in on this line's first cue: mark them as introducers so they
        # are not put on screen at the start of the line (manimgx.md gotcha 21)
        drop_minus = m.FadeOut(minus, shift=0.25 * m.UP)
        morph = m.Transform(keep, fixed_line)
        drop_minus.introducer = morph.introducer = True
        self.say(
            "[So Otto calls edit file on the cart], [the harness swaps the text], [and the last item counts again.]",
            m.AnimationGroup(m.FadeIn(edit, shift=0.2 * m.RIGHT), m.GrowArrow(a_edit), m.FadeIn(before)),
            m.AnimationGroup(drop_minus, morph),
            m.AnimationGroup(m.FadeIn(res, shift=0.2 * m.UP), m.FadeIn(check, shift=0.15 * m.UP)),
        )
        self.remove(*before.get_family(), *keep.get_family(), *minus.get_family())
        self.add(after)
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ parallel calls

    def parallel(self):
        self.next_section(
            "5 · Parallel calls",
            notes="One assistant reply may hold several tool_use blocks; the harness runs them and returns every "
            "tool_result in one user message, each matched by tool_use_id (Anthropic Messages API). Claude Code "
            "(learn-claude-code s02, the site's reading of CC source): isConcurrencySafe(input), partitionToolCalls() "
            "groups consecutive safe calls: [read A, read B, glob] together, [bash rm x] alone, [read C]; batches run "
            "in order. Li ch. 5: concurrency-safe should default to no. Durations and ids illustrative.",
        )
        bot = otto(1.1).move_to([-6.05, 0.0, 0])
        calls = m.VGroup(*[row_card(code(c, 20, INK), code(i, 20, MUTED), SUN, width=5.9) for c, i, _ in PARALLEL])
        calls.arrange(m.DOWN, buff=0.22).move_to([-1.95, 0.0, 0])
        results = m.VGroup(*[row_card(code(i, 20, MINT, weight=m.BOLD), text(d, 20, MUTED), MINT, width=4.4) for _, i, d in PARALLEL])
        results.arrange(m.DOWN, buff=0.22).move_to([4.45, 0.0, 0])

        def frame_(group, color, label):
            box = m.RoundedRectangle(corner_radius=0.2, width=group.width + 0.4, height=group.height + 0.4,
                                     stroke_color=color, stroke_width=2.5, fill_opacity=0).move_to(group)
            cap = code(label, 20, color, weight=m.BOLD).next_to(box, m.UP, buff=0.12).align_to(box, m.LEFT)
            return box, cap

        reply_box, reply_cap = frame_(calls, CORAL, "assistant · one reply")
        user_box, user_cap = frame_(results, AZURE, "user · one message")
        runs = m.VGroup(*[arrow([reply_box.get_right()[0], c.get_y(), 0], [user_box.get_left()[0], c.get_y(), 0], MUTED, buff=0.1) for c in calls])
        hint = text("the harness runs each call", 22, MUTED).move_to([1.65, -1.9, 0])

        # Claude Code's batches, drawn as a timeline
        rows_y = [1.55 - 0.66 * i for i in range(len(CALLS))]
        x0, scale, gap = -3.3, 3.2, 0.08
        labels = m.VGroup(*[
            chip(name, SUN if safe else ROSE, SUN_FILL if safe else ROSE_FILL, width=2.35, height=0.52, size=24, mono=True).move_to([-5.3, y, 0])
            for (name, safe, _), y in zip(CALLS, rows_y)
        ])
        bars, shades, tags = [], m.VGroup(), m.VGroup()
        t = 0.0
        row = 0
        for k, batch in enumerate(BATCHES):
            span = max(c[2] for c in batch)
            left = x0 + t * scale
            shade = m.RoundedRectangle(corner_radius=0.12, width=span * scale + 0.16, height=rows_y[0] - rows_y[-1] + 0.75,
                                       stroke_color=EDGE, stroke_width=2, fill_color=CARD_HI, fill_opacity=1)
            shade.move_to([left + span * scale / 2, (rows_y[0] + rows_y[-1]) / 2, 0]).set_z_index(-2)
            shades.add(shade)
            word = ("batch 1 · together", "batch 2 · alone", "batch 3")[k]
            tags.add(text(word, 22, INK if k != 1 else ROSE, weight=m.SEMIBOLD).next_to(shade, m.UP, buff=0.15))
            group = []
            for name, safe, dur in batch:
                bar = m.RoundedRectangle(corner_radius=0.08, width=dur * scale, height=0.34, stroke_width=0,
                                         fill_color=SUN if safe else ROSE, fill_opacity=0.9)
                group.append(bar.move_to([left + dur * scale / 2, rows_y[row], 0]))
                row += 1
            bars.append(group)
            t += span + gap
        assert row == len(CALLS) and x0 + (t - gap) * scale < 6.6
        axis = m.Arrow([x0 - 0.1, -1.75, 0], [6.5, -1.75, 0], buff=0, color=MUTED, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        axis_cap = text("time", 20, MUTED).next_to(axis, m.DOWN, buff=0.12).align_to(axis, m.RIGHT)
        reply_tag = text("one reply,\nfive calls", 20, MUTED).next_to(labels, m.DOWN, buff=0.3)
        default = pill("not sure it's safe? then it isn't", ROSE, ROSE_FILL, 22).move_to([0.9, -2.55, 0])
        src_a = cite("learn-claude-code · s02 (reading of Claude Code's source)").move_to([-3.0, -3.35, 0])
        src_b = cite("Li, AI Agents in Depth · ch. 5").move_to([4.0, -3.35, 0])

        self.say(
            "[One reply can hold several calls]: [read the cart], [read the checkout], [and list the tests.]",
            m.AnimationGroup(self.new_header("5 · parallel calls", "Several calls in one reply"),
                             m.FadeIn(bot, shift=0.2 * m.UP), m.Create(reply_box), m.FadeIn(reply_cap)),
            *[m.FadeIn(c, shift=0.2 * m.RIGHT) for c in calls],
        )
        self.say(
            "[The harness runs each one], [and sends every result back in one message], [each tagged with its call's ID.]",
            m.AnimationGroup(m.LaggedStart(*[m.GrowArrow(a) for a in runs], lag_ratio=0.25), m.FadeIn(hint)),
            m.AnimationGroup(m.Create(user_box), m.FadeIn(user_cap),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in results], lag_ratio=0.25)),
            m.LaggedStart(*[m.AnimationGroup(m.Circumscribe(c[2], color=SUN, buff=0.05), m.Circumscribe(r[1], color=MINT, buff=0.05))
                            for c, r in zip(calls, results)], lag_ratio=0.3),
        )
        self.wait(0.8)
        self.clear()
        self.say(
            "[Claude Code goes further.] [Calls that are safe to overlap run at the same time.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in labels], lag_ratio=0.15),
                             m.FadeIn(reply_tag), m.GrowArrow(axis), m.FadeIn(axis_cap), m.FadeIn(src_a)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(s), m.FadeIn(g, shift=0.1 * m.DOWN)) for s, g in zip(shades, tags)], lag_ratio=0.3),
        )
        self.say(
            "[Here, two reads and a file search run side by side], [the delete runs alone], [and the last read goes after it.]",
            *[m.AnimationGroup(*[m.GrowFromEdge(b, m.LEFT) for b in group]) for group in bars],
        )
        self.say(
            "[When in doubt, a tool counts as unsafe], [so nothing reads a file while another call is changing it.]",
            m.AnimationGroup(m.FadeIn(default, shift=0.15 * m.UP), m.FadeIn(src_b)),
            m.Circumscribe(shades[1], color=ROSE, buff=0.06),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ large outputs

    def large_outputs(self):
        self.next_section(
            "6 · Large outputs",
            notes=f"Never truncate silently. harness_reference.run_bash keeps the first {CUT:,} characters and says "
            f"nothing: of a {LOG_LINES:,}-line log at {AVG_CHARS} characters per line ({LOG_CHARS:,} characters, "
            f"illustrative) that is {100 * KEPT:.1f}%, and the verdict on the last line is lost. Li, ch. 4: keep the "
            f"first {HEAD} and last {TAIL} lines and say '{MARKER}'. Claude Code (learn-claude-code s02): results "
            "over maxResultSizeChars are saved to disk; the model gets a preview and the path.",
        )
        bl, br, by, bh = -6.2, 6.2, -2.8, 0.42

        def bx(f):
            return bl + f * (br - bl)

        bar = m.RoundedRectangle(corner_radius=0.08, width=br - bl, height=bh, stroke_color=EDGE, stroke_width=2,
                                 fill_color=CARD, fill_opacity=1).move_to([0, by, 0])
        ticks = m.VGroup(*[m.Line([x, by - bh / 2 + 0.09, 0], [x, by + bh / 2 - 0.09, 0]).set_stroke(FAINT, 1.5)
                           for x in np.linspace(bl + 0.15, br - 0.15, 72)])
        log = m.VGroup(bar, ticks)
        l_first = code("line 1", 20, MUTED).next_to(bar, m.DOWN, buff=0.14).align_to(bar, m.LEFT)
        l_last = code(f"line {LOG_LINES:,}: {VERDICT}", 20, MINT).next_to(bar, m.DOWN, buff=0.14).align_to(bar, m.RIGHT)
        lost = text("cut off: the model never sees it", 20, ROSE, weight=m.SEMIBOLD).next_to(bar, m.UP, buff=0.2).align_to(bar, m.RIGHT)

        kept = m.RoundedRectangle(corner_radius=0.08, width=bx(KEPT) - bl, height=bh, stroke_width=0,
                                  fill_color=MUTED, fill_opacity=0.55).move_to([(bl + bx(KEPT)) / 2, by, 0])
        cut = m.DashedLine([bx(KEPT), by - 0.3, 0], [bx(KEPT), by + 0.3, 0], dash_length=0.08).set_stroke(ROSE, 3)
        kept_cap = text(f"kept: the first {CUT:,} characters, about {round(100 * KEPT)}%", 20, INK).next_to(bar, m.UP, buff=0.2).align_to(bar, m.LEFT)

        slice_w = 0.16  # 50 of 8,623 lines would be 0.07 wide: drawn a little wider to be seen
        head_slice = m.RoundedRectangle(corner_radius=0.05, width=slice_w, height=bh, stroke_width=0, fill_color=MINT,
                                        fill_opacity=1).move_to([bl + slice_w / 2, by, 0])
        tail_slice = head_slice.copy().move_to([br - slice_w / 2, by, 0])
        head_cap = text(f"first {HEAD}", 20, MINT).next_to(bar, m.UP, buff=0.14).align_to(bar, m.LEFT)
        tail_cap = text(f"last {TAIL}", 20, MINT).next_to(bar, m.UP, buff=0.14).align_to(bar, m.RIGHT)
        mid_cap = text(f"{OMITTED:,} lines left out, saved to disk", 20, SUN).next_to(bar, m.UP, buff=0.14)

        term = terminal(["$ pytest --log-cli-level=DEBUG", f"collected {TESTS} items", "DEBUG db: connect sqlite:///shop.db",
                         "DEBUG cart: add item price=3", "DEBUG cart: add item price=4"], width=6.4).move_to([-3.2, 0.95, 0])
        count = title_text(f"{LOG_LINES:,} lines", 54).move_to([3.55, 1.6, 0])
        meter = Meter(4.4, 0.35, SUN, 0.26).move_to([3.55, 0.25, 0])
        meter_cap = text("context window", 20, MUTED).next_to(meter, m.UP, buff=0.14).align_to(meter, m.LEFT)
        too_big = text("far too big to read whole", 22, ROSE, weight=m.SEMIBOLD).next_to(meter, m.DOWN, buff=0.22)

        run_code = py_block("def run_bash(command):\n    done = subprocess.run(command, ...)\n"
                            f"    return (done.stdout + done.stderr)[:{CUT:_}]", 20, title="harness_reference.py").move_to([0, 1.6, 0])
        cut_code = glyphs(run_code[1], f"[:{CUT:_}]")

        def result_lines(rows, colors):
            cap = code("tool_result", 20, MINT, weight=m.BOLD)
            body = m.VGroup(*[code(r, 20, c) for r, c in zip(rows, colors)]).arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
            inner = m.VGroup(cap, body).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
            box = card(inner.width + 0.7, inner.height + 0.5, MINT, CARD, radius=0.16, stroke=2).move_to(inner)
            return m.VGroup(box, cap, body)

        bad = result_lines([f"collected {TESTS} items", "DEBUG db: connect sqlite:///shop.db", "DEBUG cart: add item price=3",
                            "DEBUG checkout: apply discou"], [MUTED] * 4).move_to([-1.3, -0.8, 0])
        no_warn = pill("no sign anything is missing", ROSE, ROSE_FILL, 20).next_to(bad, m.RIGHT, buff=0.4)
        marker_1, marker_2 = MARKER[: MARKER.index(" saved")], MARKER[MARKER.index("saved"):]
        assert f"{marker_1} {marker_2}" == MARKER
        good = result_lines([f"collected {TESTS} items", "DEBUG db: connect sqlite:///shop.db", marker_1, marker_2,
                             "DEBUG db: close", VERDICT], [MUTED, MUTED, SUN, SUN, MUTED, MINT]).move_to([-1.6, -0.15, 0])
        g_rows = good[2]
        side_head = text(f"← first {HEAD} lines", 20, MUTED).next_to(m.VGroup(g_rows[0], g_rows[1]), m.RIGHT, buff=0.3)
        side_tail = text(f"← last {TAIL} lines", 20, MUTED).next_to(m.VGroup(g_rows[4], g_rows[5]), m.RIGHT, buff=0.3)
        side_head.set_x(good[0].get_right()[0] + 0.3 + side_head.width / 2)
        side_tail.set_x(good[0].get_right()[0] + 0.3 + side_tail.width / 2)
        g_count = glyphs(g_rows[2], f"[{OMITTED} lines omitted")
        g_path = glyphs(g_rows[3], f"saved to {SAVED}]")
        src_b = cite("Li, AI Agents in Depth · ch. 4").next_to(good, m.DOWN, buff=0.22).align_to(good, m.LEFT)
        doc = icon("doc", 0.85).move_to([5.45, -0.15, 0])
        doc_cap = text("full output", 20, MUTED).next_to(doc, m.DOWN, buff=0.15)
        a_doc = arrow([good[0].get_right()[0], (g_rows[2].get_y() + g_rows[3].get_y()) / 2, 0], doc.get_left(), SUN, buff=0.12)
        reread = pill("read_file", SUN, SUN_FILL, 20, mono=True).next_to(doc_cap, m.DOWN, buff=0.2)

        cc_head = m.VGroup(text("Claude Code", 22, INK, weight=m.BOLD), cite("learn-claude-code · s02")).arrange(m.RIGHT, buff=0.4)
        cc_rows = m.VGroup(cc_head, text("big results go to disk; the model gets a preview and the path", 20, MUTED))
        cc_rows.arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        cc_box = card(cc_rows.width + 0.6, cc_rows.height + 0.4, EDGE, CARD, radius=0.16, stroke=2)
        cc = m.VGroup(cc_box, cc_rows.move_to(cc_box)).align_to([6.75, 2.6, 0], m.UR)

        self.say(
            "[Last, big outputs.] [Say Otto runs the tests with debug logging on.]",
            self.new_header("6 · large outputs", "Never cut silently"),
            m.FadeIn(term, shift=0.2 * m.UP),
        )
        self.say(
            f"[Back come, for example, {spell(LOG_LINES)} lines], [far too many for the context.]",
            m.AnimationGroup(m.GrowFromEdge(log, m.LEFT), m.FadeIn(l_first), m.FadeIn(l_last), m.FadeIn(count, scale=0.85),
                             m.FadeIn(meter), m.FadeIn(meter_cap)),
            m.AnimationGroup(meter.fill_to(1.0, run_time=1.2), m.FadeIn(too_big, shift=0.1 * m.UP)),
        )
        self.say(
            f"[Our teaching harness] [simply keeps the first {spell(CUT)} characters.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(term, count, meter, meter_cap, too_big)), m.FadeIn(run_code, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(cut_code, color=ROSE, buff=0.05), m.FadeIn(kept), m.Create(cut), m.FadeIn(kept_cap)),
        )
        self.say(
            "[The summary on the last line is cut off], [and the model never learns that anything is missing.]",
            m.AnimationGroup(m.FadeIn(lost, shift=0.1 * m.UP), l_last.animate.set_color(ROSE)),
            m.AnimationGroup(m.FadeIn(bad, shift=0.2 * m.UP), m.FadeIn(no_warn, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[Better: keep the head and the tail], [say how many lines were left out], [and where the full output was saved.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(run_code, bad, no_warn, kept, cut, kept_cap, lost)),
                l_last.animate.set_color(MINT),
                m.FadeIn(good[0]), m.FadeIn(good[1]), *[m.FadeIn(g_rows[i]) for i in (0, 1, 4, 5)],
                m.FadeIn(side_head), m.FadeIn(side_tail),
                m.FadeIn(head_slice), m.FadeIn(tail_slice), m.FadeIn(head_cap), m.FadeIn(tail_cap),
            ),
            m.AnimationGroup(m.FadeIn(g_rows[2]), m.FadeIn(g_rows[3]), m.FadeIn(mid_cap), m.FadeIn(src_b)),
            m.AnimationGroup(m.GrowArrow(a_doc), m.FadeIn(doc, shift=0.2 * m.LEFT), m.FadeIn(doc_cap), m.Circumscribe(g_path, color=SUN, buff=0.05)),
        )
        self.say(
            f"[Now Otto sees the verdict: {spell(TESTS)} passed], [and can open the saved file if it needs more.]",
            m.Circumscribe(g_rows[5], color=MINT, buff=0.06),
            m.AnimationGroup(m.FadeIn(reread, shift=0.15 * m.UP), m.Circumscribe(doc, color=SUN, buff=0.08)),
        )
        self.say(
            "[Claude Code does the same]: [big results go to disk, and the model gets a preview and the path.]",
            m.AnimationGroup(m.FadeIn(cc_box), m.FadeIn(cc_head)),
            m.FadeIn(cc_rows[1], shift=0.1 * m.UP),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section(
            "Recap",
            notes="Coming up: tool design (Ep15) and MCP (Ep16). Recap: describe (a schema the model reads), dispatch "
            "(TOOL_HANDLERS[block.name]), return (a tool_result with the same id).",
        )
        subs = {15: "when to use it, what it can't do,\ngeneral or dedicated, edit formats",
                16: "one protocol to plug in\nother people's tools"}
        bodies = []
        for n in (15, 16):
            title_, mod = EPISODES[n]
            name, accent, tint = MODULES[mod]
            heading = title_text(title_, 32)
            heading.scale(min(1, 5.5 / heading.width))
            body = m.VGroup(pill(f"EP {n} · {name}", accent, tint, 20, mono=True), heading,
                            text(subs[n], 22, MUTED)).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
            bodies.append((body, accent))
        tallest = max(b.height for b, _ in bodies)
        cards = m.VGroup()
        for body, accent in bodies:
            box = card(6.3, tallest + 0.8, accent, CARD, stroke=2)
            cards.add(m.VGroup(box, body.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)))
        cards.arrange(m.RIGHT, buff=0.4).move_to([0, 0.1, 0])

        hero = otto(1.5).move_to([0, 0.75, 0])
        y_arm = hero[0].get_y()
        hands = m.VGroup(tool_tag("read_file").move_to([-3.75, y_arm, 0]), tool_tag("edit_file").move_to([3.75, y_arm, 0]))
        arms = m.VGroup(m.Line(hero[0].get_left(), hands[0].get_right()),
                        m.Line(hero[0].get_right(), hands[1].get_left())).set_stroke(SUN, 4)
        each = m.VGroup(*[text("one schema + one handler", 20, MUTED).next_to(h, m.DOWN, buff=0.25) for h in hands])
        loop_chip = chip("agent_loop()", EDGE, CARD, width=3.1, height=0.7, size=24, mono=True)
        same = pill("unchanged", MINT, MINT_FILL, 22)
        loop_row = m.VGroup(loop_chip, same).arrange(m.RIGHT, buff=0.35).move_to([0, -1.75, 0])

        halo = glow(CORAL, radius=4.0)
        words = on_baseline(m.VGroup(*[title_text(w, 66, c) for w, c in (("Describe", CORAL), ("Dispatch", SUN), ("Return", MINT))]))
        words.arrange(m.RIGHT, buff=0.95)
        on_baseline(words)
        mid = words[0].get_y()
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots).move_to(0.4 * m.UP)
        tagline = text("The model writes the request. Your code does the work.", 30, MUTED).next_to(row, m.DOWN, buff=0.75)

        self.say(
            "[The same loop as in episode two], [but now Otto has hands]: [one schema and one handler each.]",
            m.AnimationGroup(self.new_header("recap", "Same loop, more hands"), m.FadeIn(loop_row, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(hero, scale=0.85), m.Create(arms), m.FadeIn(hands, scale=0.9)),
            m.FadeIn(each, shift=0.15 * m.UP),
        )
        self.say(
            "[Designing tools a model uses well is episode fifteen]; [plugging in outside tools with MCP is episode sixteen.]",
            m.AnimationGroup(self.new_header("coming up", "More about tools"),
                             m.FadeOut(m.VGroup(hero, hands, arms, each, loop_row), shift=0.2 * m.UP),
                             m.FadeIn(cards[0], shift=0.2 * m.UP)),
            m.FadeIn(cards[1], shift=0.2 * m.UP),
        )
        self.say(
            "[To sum up]: [describe], [dispatch], [return.]",
            m.AnimationGroup(self.drop_header(), m.FadeOut(cards, shift=0.2 * m.DOWN), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[The model writes the request. Your code does the work.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
