"""Agents, Animated · Episode 27 · Coding Agents Are General Agents  (Module 6 · Beyond the Loop)

Why a shell and a file system make an agent general. Running example: tinyshop's owner asks Otto, the
series' coding agent, for a sales report instead of a bug fix. Otto has no spreadsheet or report tool;
the same loop and the same tools (bash, read_file, edit_file, plus Episode 14's write_file) do the job
by writing a ten-line Python script. The run is not drawn by hand: the scene runs the reference
agent_loop against a scripted stand-in for the model over an in-memory "disk" holding an illustrative
sales.csv, executes report.py exactly as shown, and every number on screen comes from those runs or
from the cited sources (asserted below).

Arc: a new job (a spreadsheet, a request, no report tool) · the same loop writes and runs a script
(three tool calls, four model calls, the loop unchanged) · code as a thinking tool (Li's 40 students:
words say 14, code says 8) · code makes tools (a meta-capability; MEMORY.md; next month is one call) ·
the file system as the hub (Li's OpenClaw storyboard) · the thesis and today's general agents (Claude
Code, OpenClaw, Manus) · why coding agents matured first (tests, linters, git) · the goal-clarity ×
verification grid (Otto's report splits in two) · checks can be gamed: constrain actions, not only
outcomes · recap.

Sources (facts dated October 2026):
- Li, AI Agents in Depth, ch. 5 ("From Manus to OpenClaw: the coding core of general-purpose agents"):
  a general-purpose agent targeting open-ended tasks has at its core a coding agent plus a file system;
  code generation is "a meta-capability" ("the ability to create new tools and capabilities dynamically
  at runtime"); OpenClaw's storyboard: read MEMORY.md (the user prefers PDF reports, data in Google
  Sheets) → get the Google Sheets API instructions via web search, download the data by running code →
  a Python analysis script (pandas aggregation, matplotlib charts) → report.pdf and charts/ → record
  the sheet's ID in MEMORY.md; "the file system is the hub of information flow"; OpenClaw's memory is
  MEMORY.md plus "Markdown logs archived by date"; Claude Cowork is "Anthropic's general-purpose Agent
  for knowledge work, reusing Claude Code's agentic architecture"; coding agents matured first "not
  because code generation models are particularly powerful, but because decades of software
  engineering infrastructure naturally constitute a robust Harness" (test suites: clear acceptance
  criteria; linters and type checkers: instant automated verification; Git: version control and
  rollback); the goal × verification table (clear + automatic: "Sweet spot: fixing bugs with test
  cases"; clear + manual: "Throughput-limited: code refactoring requires manual review"; vague +
  automatic: "Efficiently going off track: optimizing 'code quality' with a linter"; vague + manual:
  "Hard to start: 'make the UI look better'"); reward hacking: "Deleting and rebuilding the database to
  'fix' a database fault does repair it, but the data is gone", "deleting all the code to fix a
  compilation error"; hence "constraining actions, not merely outcomes"; code as a thinking tool: 40
  students, 60% math, 45% physics, 25% both; natural-language reasoning says only physics = 24 - 10 =
  14; the code says 8.
- Li, ch. 1 (observation and action spaces): Manus was "the first widely influential production Agent"
  to bring Deep Research, Coding and Computer Use together in one system; OpenClaw is local-first and
  "receives tasks and returns results through messaging channels users already inhabit".
- Claude Code is Anthropic's agentic coding tool (it works in a terminal: shell commands and files).

Illustrative (for teaching): sales.csv (120 generated rows; regions, products, amounts), the owner's
request, Otto's choices, the tool-call ids, MEMORY.md's wording, October's file, the unparsable row, the
condensed wording of the OpenClaw cards (Li names MEMORY.md, the Google Sheets API, pandas, matplotlib,
report.pdf and charts/). Computed and asserted below: the script's output, the region totals and the
leader, the counts of tool and model calls, the check that the regions add up to all rows, the 40
students' numbers, the gamed run (119 rows, check still passes).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import csv
import io
import re
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep27"
REF = (SERIES / "harness_reference.py").read_text()
FACTS = "Fact date: October 2026."
BOOK = "Li, AI Agents in Depth"

# ---------------------------------------------------------------------------- the spreadsheet (illustrative)

MONTH = "September 2026"
REGIONS = ("North", "South", "East", "West")
SHARES = (40, 30, 28, 22)
PRODUCTS = ("mug", "tote", "poster", "sticker")
N_ROWS = sum(SHARES)
assert N_ROWS == 120  # "a hundred and twenty rows"


def make_sales(month, seed):
    """A month of shop sales as CSV text: one row per sale (generated, illustrative)."""
    pool = [r for r, n in zip(REGIONS, SHARES) for _ in range(n)]
    rows = ["date,region,product,amount"]
    for i in range(N_ROWS):
        seed = (seed * 1103515245 + 12345) % 2**31
        amount = 8 + (seed >> 8) % 40 + 0.5 * ((seed >> 4) % 2)
        rows.append(f"2026-{month}-{1 + i // 4:02d},{pool[(i * 53) % N_ROWS]},{PRODUCTS[(seed >> 16) % 4]},{amount:.2f}")
    return "\n".join(rows) + "\n"


SALES_CSV = make_sales("09", 2026)
OCTOBER_CSV = make_sales("10", 77)

# ---------------------------------------------------------------------------- report.py, the script Otto writes

REPORT_PY = '''import csv
from collections import Counter
sales = Counter()
for row in csv.DictReader(open("sales.csv")):
    sales[row["region"]] += float(row["amount"])
with open("report.md", "w") as out:
    for region, usd in sales.most_common():
        bar = "█" * round(usd / 100)
        out.write(f"{region:6}{bar} ${usd:,.2f}\\n")
print(open("report.md").read())'''
assert len(REPORT_PY.split("\n")) <= 10


class _Written(io.StringIO):
    """A file opened for writing on the in-memory disk: saved when closed."""

    def __init__(self, disk, path):
        super().__init__()
        self.disk, self.path = disk, path

    def close(self):
        self.disk[self.path] = self.getvalue()
        super().close()

    def __exit__(self, *exc):
        self.close()


def python(src, disk):
    """`python script.py` on the in-memory disk: runs the source exactly as written, returns stdout."""
    printed = []

    def fake_open(path, mode="r", **kwargs):
        return _Written(disk, path) if "w" in mode else io.StringIO(disk[path])

    def fake_print(*args, sep=" ", end="\n"):
        printed.append(sep.join(str(a) for a in args) + end)

    exec(src, {"open": fake_open, "print": fake_print, "__name__": "__main__"})
    return "".join(printed)


def regions_add_up(disk):
    """The check Otto adds later: the report's region totals add up to the sum of every row."""
    rows = list(csv.DictReader(io.StringIO(disk["sales.csv"])))
    all_rows = sum(float(r["amount"]) for r in rows)
    reported = sum(float(line.split("$")[1].replace(",", "")) for line in disk["report.md"].splitlines() if line)
    return abs(all_rows - reported) < 0.005, len(rows)


# ---------------------------------------------------------------------------- the harness (reference names)

assert "def agent_loop(messages):" in REF and 'if response.stop_reason != "tool_use":' in REF
assert 'results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})' in REF
assert 'TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}' in REF
DISK = {"sales.csv": SALES_CSV}


def run_bash(command):
    if command == "head -3 sales.csv":
        return "\n".join(DISK["sales.csv"].split("\n")[:3])
    if command == "python report.py":
        return python(DISK["report.py"], DISK)
    raise ValueError(command)


def read_file(path):
    return DISK[path]


def edit_file(path, old_text, new_text):
    DISK[path] = DISK[path].replace(old_text, new_text, 1)
    return f"edited {path}"


def write_file(path, content):  # Episode 14's tool
    DISK[path] = content
    return f"wrote {path}"


TOOLS = [{"name": "bash"}, {"name": "read_file"}, {"name": "edit_file"}, {"name": "write_file"}]  # schemas: Episode 3
TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file, "write_file": write_file}


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


IDS = iter(range(1, 100))


class ScriptedModel:
    """Stands in for client.messages: follows a plan; its final answer reads the last tool result."""

    def __init__(self, plan, final):
        self.plan, self.final, self.calls = plan, final, 0

    def create(self, system, messages, tools):
        step = self.plan[self.calls]
        self.calls += 1
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=self.final(messages))])
        name, args = step
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"toolu_{next(IDS):02d}", name=name, input=args)])


client = NS(messages=None)
SYSTEM = "You are Otto, a coding agent working in the tinyshop repo."


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


def last_result(messages):
    """The content of the most recent tool_result."""
    return next(m_["content"][-1]["content"] for m_ in reversed(messages) if m_["role"] == "user" and isinstance(m_["content"], list))


def answer_from_report(messages):
    top = last_result(messages).splitlines()[0]
    return f"Done: report.md is ready.\n{top.split()[0]} leads with ${top.split('$')[1]}."


TASK = "Turn sales.csv into a report:\nrevenue by region."
PLAN = [
    ("bash", {"command": "head -3 sales.csv"}),
    ("write_file", {"path": "report.py", "content": REPORT_PY}),
    ("bash", {"command": "python report.py"}),
    None,
]
MESSAGES = [{"role": "user", "content": TASK}]
client.messages = ScriptedModel(PLAN, answer_from_report)
ANSWER = agent_loop(MESSAGES).content[0].text

CALLS = [b for msg in MESSAGES if msg["role"] == "assistant" for b in msg["content"] if b.type == "tool_use"]
RESULTS = [b["content"] for msg in MESSAGES[1:] if msg["role"] == "user" for b in msg["content"]]
TOOL_CALLS = len(CALLS)
MODEL_CALLS = sum(msg["role"] == "assistant" for msg in MESSAGES)
assert (TOOL_CALLS, MODEL_CALLS) == (3, 4)  # "three tool calls, four model calls"

HEAD_OUT = RESULTS[0]
REPORT_OUT = RESULTS[2].rstrip("\n")
REPORT_LINES = REPORT_OUT.splitlines()
assert HEAD_OUT.split("\n")[0] == "date,region,product,amount"  # "date, region, product, and amount"
assert REPORT_OUT == DISK["report.md"].rstrip("\n") and len(REPORT_LINES) == len(REGIONS)

TRUE_TOTALS = Counter()
for _row in csv.DictReader(io.StringIO(SALES_CSV)):
    TRUE_TOTALS[_row["region"]] += float(_row["amount"])
LEADER, LEAD_USD = TRUE_TOTALS.most_common(1)[0]
assert LEADER == "North" and REPORT_LINES[0].startswith("North")  # "North sold the most"
assert all(f"${TRUE_TOTALS[r]:,.2f}" in line for line in REPORT_LINES for r in REGIONS if line.startswith(r))
CHECK_OK, CHECK_ROWS = regions_add_up(DISK)
assert CHECK_OK and CHECK_ROWS == N_ROWS  # "regions add up to all 120 rows"

# next month: sales.csv replaced by October's file; MEMORY.md says how to run the report
DISK["sales.csv"] = OCTOBER_CSV
NEXT = [{"role": "user", "content": "October's sales report, please."}]
client.messages = ScriptedModel([("bash", {"command": "python report.py"}), None], answer_from_report)
NEXT_ANSWER = agent_loop(NEXT).content[0].text
NEXT_TOOL_CALLS = sum(b.type == "tool_use" for msg in NEXT if msg["role"] == "assistant" for b in msg["content"])
assert NEXT_TOOL_CALLS == 1  # "next month, the whole job is one tool call"
NEXT_TOP = last_result(NEXT).splitlines()[0]

# a gamed check: one row that won't parse; deleting it makes the error go away and the check still pass
BAD_INDEX = 58
_lines = SALES_CSV.split("\n")
BAD_ROW = re.sub(r",[\d.]+$", ",n/a", _lines[BAD_INDEX])
BAD_CSV = "\n".join(_lines[:BAD_INDEX] + [BAD_ROW] + _lines[BAD_INDEX + 1 :])
try:
    python(REPORT_PY, {"sales.csv": BAD_CSV})
    BAD_ERROR = None
except ValueError as error:
    BAD_ERROR = f"ValueError: {error}"
assert BAD_ERROR == "ValueError: could not convert string to float: 'n/a'"
GAMED = {"sales.csv": BAD_CSV.replace(BAD_ROW + "\n", "")}
python(REPORT_PY, GAMED)
GAMED_OK, GAMED_ROWS = regions_add_up(GAMED)
assert GAMED_OK and GAMED_ROWS == N_ROWS - 1  # the check passes, one sale is gone

# ---------------------------------------------------------------------------- the 40 students (Li, ch. 5)

PUZZLE_PY = '''math = int(40 * 0.60)    # 24
phys = int(40 * 0.45)    # 18
both = int(40 * 0.25)    # 10
only_phys = phys - both  # 8
print(only_phys)'''
PUZZLE_OUT = python(PUZZLE_PY, {}).strip()
_vals = {}
exec(PUZZLE_PY.replace("print", "str"), _vals)
assert PUZZLE_OUT == "8" and all(str(_vals[k]) == c for k, c in re.findall(r"^(\w+) = .*# (\d+)$", PUZZLE_PY, re.M))
MATH, PHYS, BOTH = _vals["math"], _vals["phys"], _vals["both"]
WRONG = MATH - BOTH  # the slip: subtracting from the math group
assert (MATH, PHYS, BOTH, WRONG, PHYS - BOTH) == (24, 18, 10, 14, 8)  # "twenty-four minus ten, fourteen"; "eight"

# ---------------------------------------------------------------------------- helpers

USER = "user"
LOCK = recolored("lock", ROSE, ROSE_FILL, prefix=PREFIX)
PY_T2C = {
    r"#[^\n]*": FAINT,
    r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
    r"\b(def|while|for|in|if|else|return|import|from|as|with|not|and|or|None|True|False)\b": ORCHID,
    r"\b\d+(\.\d+)?\b": SUN,
    r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
}


def cite(s):
    return text(s, 20, MUTED)


def top_cite(s):
    """A source in small muted text, top right, level with the header."""
    return cite(s).to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)


def mono_chip(word, color, fill, size=22, height=0.58, pad=0.45):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def word_chip(word, color, fill, size=22, height=0.6, pad=0.5):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def verdict(label, ok, size=20):
    return pill(f"{'✓' if ok else '✗'}  {label}", MINT if ok else ROSE, MINT_FILL if ok else ROSE_FILL, size)


_DESCENDERS = set("gjpqy,;_()[]{}Q|")


def baseline(words):
    """The y of a Text's baseline (glyphs without descenders; with a ligature, its first glyph)."""
    chars = [c for c in words.text if not c.isspace()]
    if len(chars) != len(words):
        return words[0].get_bottom()[1]
    flat = [g for g, c in zip(words, chars) if c not in _DESCENDERS]
    return min(g.get_bottom()[1] for g in (flat or list(words)))


def sit(words, y):
    """Shift a Text so that its baseline is at `y`."""
    words.shift((y - baseline(words)) * m.UP)
    return words


def recentre_chip(c, size):
    """Re-seat a chip's label on the chip's baseline (chip() miscounts glyphs when a ligature such as
    "fi" draws two letters as one)."""
    sit(c[1], c[0].get_center()[1] - 0.449 * size / 48 / 2)
    return c


def ring_arrow(center, radius, gap=1.0, color=MUTED):
    """A clockwise ring with an open gap at the top and an arrowhead at its end (Arc.add_tip rescales a
    nearly full arc, so the head is placed by hand)."""
    center = np.array(center, dtype=float)
    start = m.PI / 2 - gap / 2
    arc = m.Arc(radius=radius, start_angle=start, angle=-(m.TAU - gap), arc_center=center).set_stroke(color, 4)
    a = start - (m.TAU - gap)
    tangent = np.arctan2(-np.cos(a), np.sin(a))  # clockwise direction of travel at the end
    tip = m.Triangle().set_fill(color, 1).set_stroke(width=0).scale(0.11)
    tip.rotate(tangent - m.PI / 2).move_to(center + radius * np.array([np.cos(a), np.sin(a), 0]))
    return m.VGroup(arc, tip)


def put_head(bot, point):
    """Move an otto() so that its head (not the antenna) is centred on `point`."""
    return bot.shift(np.array(point) - bot[0].get_center())


def line_glyphs(words, src):
    """The glyphs of each non-empty line of Text `words` made from `src`, in order."""
    groups, n = [], 0
    for line in src.split("\n"):
        k = sum(not c.isspace() for c in line)
        if k:
            groups.append(words[n : n + k])
        n += k
    return groups


def gutter(box, part, color=LIME):
    """A short bar in a code card's left padding, beside the lines `part`."""
    x = box.get_left()[0] + 0.17
    return m.Line([x, part.get_top()[1] + 0.04, 0], [x, part.get_bottom()[1] - 0.04, 0]).set_stroke(color, 6)


def edge_point(box, toward, pad=0.08):
    """Where the ray from `box`'s centre toward `toward` leaves the box (plus a small gap)."""
    c = box.get_center()
    d = np.array(toward) - c
    d[2] = 0
    hw, hh = box.width / 2 + pad, box.height / 2 + pad
    t = min(hw / abs(d[0]) if abs(d[0]) > 1e-9 else 1e9, hh / abs(d[1]) if abs(d[1]) > 1e-9 else 1e9)
    return c + t * d


def call_text(call):
    """A tool_use block as a short call, the way the context column shows it."""
    if call.name == "write_file":
        return f'write_file("{call.input["path"]}", …)'
    return f'{call.name}("{call.input["command"]}")'


def bars_lime(row, line):
    return paint(row, line, {"█+": LIME, r"\$[\d,.]+": INK})


# ---------------------------------------------------------------------------- the episode


class Ep27CodingAgentsGeneral(Episode):
    number = 27

    def lesson(self):
        self.request()
        self.run()
        self.thinking()
        self.new_tools()
        self.hub()
        self.general()
        self.why_code()
        self.grid()
        self.gaming()
        self.recap()

    # ------------------------------------------------------------------ 1 · a new job

    def request(self):
        self.next_section(
            "01 · A new job",
            notes="Otto, the series' coding agent, usually fixes tinyshop's bugs. Today the owner sends sales.csv "
            f"({MONTH}, {N_ROWS} rows; generated, illustrative) and asks for revenue by region. Otto has no "
            "spreadsheet or report tool: only bash, read_file, edit_file (harness_reference.py) and Episode 14's "
            "write_file. " + FACTS,
        )
        bot = put_head(otto(1.2), [-4.6, 1.3, 0])
        halo = glow(CORAL, radius=2.0).move_to(bot[0])
        usual = text("usually: fixing bugs in tinyshop", 22, MUTED).next_to(bot, m.DOWN, buff=0.35)

        # the spreadsheet, drawn as a table card
        lines = SALES_CSV.split("\n")
        shown = [lines[0].split(",")] + [ln.split(",") for ln in lines[1:4]]
        cells = [[code(c, 20, MUTED if i == 0 else INK) for c in row] for i, row in enumerate(shown)]
        widths = [max(cells[i][j].width for i in range(len(cells))) for j in range(4)]
        xs, x = [], 0.0
        for w in widths:
            xs.append(x)
            x += w + 0.5
        rows = m.VGroup()
        for i, row in enumerate(cells):
            y = -0.44 * i - (0.14 if i else 0)
            for j, c in enumerate(row):
                if j == 3:
                    c.move_to([xs[3] + widths[3], 0, 0], aligned_edge=m.RIGHT)
                else:
                    c.move_to([xs[j], 0, 0], aligned_edge=m.LEFT)
                sit(c, y)
            rows.add(m.VGroup(*row))
        more = text(f"… {N_ROWS - (len(cells) - 1)} more rows", 20, MUTED).move_to([xs[0], 0, 0], aligned_edge=m.LEFT)
        sit(more, -0.44 * len(cells) - 0.14)
        title = m.VGroup(code("sales.csv", 22, AZURE, weight=m.BOLD), text(f"{MONTH} · {N_ROWS} rows", 20, MUTED)).arrange(m.RIGHT, buff=0.3)
        title.next_to(rows, m.UP, buff=0.32).align_to(rows, m.LEFT)
        body = m.VGroup(title, rows, more)
        sheet = card(body.width + 0.8, body.height + 0.65, AZURE, CARD).move_to([2.6, 0.95, 0])
        shift = sheet.get_center() - body.get_center()
        body.shift(shift)
        sep_y = (rows[0].get_bottom()[1] + rows[1].get_top()[1]) / 2
        sep = m.Line([sheet.get_left()[0] + 0.25, sep_y, 0], [sheet.get_right()[0] - 0.25, sep_y, 0]).set_stroke(EDGE, 2)
        table = m.VGroup(sheet, sep, body)
        assert sheet.get_top()[1] < 2.65 and sheet.get_right()[0] < 6.6, ("the table must stay inside the frame", sheet.get_top(), sheet.get_right())

        owner = icon(USER, 0.85)
        ask_words = text("Turn sales.csv into a report: revenue by region.", 26, INK)
        ask = bubble(ask_words, AZURE, AZURE_FILL, tail="right")
        row = m.VGroup(ask, owner).arrange(m.RIGHT, buff=0.3)
        row.move_to([0, -2.45, 0]).align_to([6.45, 0, 0], m.RIGHT)
        ask.shift((owner.get_y() - ask[1].get_y()) * m.UP)
        owner_label = text("the owner", 20, MUTED).next_to(owner, m.DOWN, buff=0.15)

        ghosts = m.VGroup(verdict("spreadsheet tool", False, 22), verdict("report tool", False, 22))
        ghosts.arrange(m.DOWN, buff=0.25).move_to([-4.6, -0.95, 0])
        tools = m.VGroup(
            m.VGroup(mono_chip("bash", SUN, SUN_FILL), mono_chip("read_file", SUN, SUN_FILL)).arrange(m.RIGHT, buff=0.18),
            m.VGroup(mono_chip("edit_file", SUN, SUN_FILL), mono_chip("write_file", SUN, SUN_FILL)).arrange(m.RIGHT, buff=0.18),
        ).arrange(m.DOWN, buff=0.18).move_to([-4.6, -0.95, 0])
        assert tools.get_left()[0] > -6.6 and tools.get_bottom()[1] > ask[1].get_top()[1] + 0.2 and tools.get_top()[1] < usual.get_bottom()[1] - 0.2, (tools.get_top(), tools.get_bottom(), ask[1].get_top(), usual.get_bottom())

        self.say(
            "[Otto usually fixes bugs in tinyshop.] [Today, the shop's owner sends something else]: [a spreadsheet of September's sales.]",
            m.AnimationGroup(self.new_header("01 · a new job", "A report, not a bug fix"), m.FadeIn(halo, scale=0.6),
                             m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(usual, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(owner, shift=0.2 * m.UP), m.FadeIn(owner_label)),
            m.AnimationGroup(m.FadeIn(sheet), m.FadeIn(title), m.Create(sep),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rows], m.FadeIn(more), lag_ratio=0.2)),
        )
        self.say(
            "[With it, a request]: [turn this into a report, with the revenue for each region.]",
            m.FadeIn(ask[:2], shift=0.2 * m.UP),
            m.AddTextLetterByLetter(ask_words, time_per_char=0.03),
        )
        self.say(
            "[Otto has no spreadsheet tool], [and no report tool.] [Only a shell, and a few tools for files.]",
            m.FadeIn(ghosts[0], shift=0.15 * m.UP),
            m.FadeIn(ghosts[1], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeOut(ghosts, shift=0.2 * m.DOWN), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for r in tools for c in r], lag_ratio=0.2)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the same loop

    def run(self):
        self.next_section(
            "02 · Same loop",
            notes=f"The reference agent_loop, unchanged, run against a scripted model: bash('head -3 sales.csv') → "
            f"write_file('report.py') → bash('python report.py') → answer. {TOOL_CALLS} tool calls, {MODEL_CALLS} model "
            f"calls (computed from the run). report.py is executed exactly as shown; its output is the tool result. "
            f"Leader: {LEADER} (${LEAD_USD:,.2f}). Data illustrative. " + FACTS,
        )
        head = np.array([-5.88, 1.95, 0])
        radius = 0.66
        bot = put_head(otto(0.5), head)
        loop = ring_arrow(head, radius)
        loop_label = code("loop", 20, MUTED).move_to(head, aligned_edge=m.LEFT).shift((radius + 0.14) * m.RIGHT)
        spin = lambda: m.Rotate(loop, angle=-m.TAU, about_point=head, run_time=1.2)  # noqa: E731

        f_csv = mono_chip("sales.csv", AZURE, AZURE_FILL)
        f_py = mono_chip("report.py", LIME, LIME_FILL)
        f_md = mono_chip("report.md", LIME, LIME_FILL)
        files = m.VGroup(f_csv, f_py, f_md).arrange(m.RIGHT, buff=0.2)
        files.move_to([0, head[1] - 0.12, 0]).align_to([loop_label.get_right()[0] + 0.45, 0, 0], m.LEFT)
        files_label = code("files", 20, MUTED).next_to(files, m.UP, buff=0.14).align_to(files, m.LEFT)

        work_top, work_left, work_w = 0.95, -6.58, 8.55
        assert loop.get_bottom()[1] > work_top + 0.05 + 0.2 and loop.get_left()[0] > -6.62, (loop.get_bottom(), loop.get_left())
        t1 = terminal(["$ head -3 sales.csv", *HEAD_OUT.split("\n")], width=work_w, size=20)
        t1.move_to([0, 0, 0]).align_to([0, work_top, 0], m.UP).align_to([work_left, 0, 0], m.LEFT)
        script = code_block(REPORT_PY, size=20, width=work_w, title="report.py · written by Otto", accent=LIME)
        script.move_to([0, 0, 0]).align_to([0, work_top + 0.05, 0], m.UP).align_to([work_left, 0, 0], m.LEFT)
        lines = line_glyphs(script[1], REPORT_PY)
        t2 = terminal(["$ python report.py", *REPORT_LINES], width=work_w, size=20)
        t2.move_to([0, 0, 0]).align_to([0, work_top, 0], m.UP).align_to([work_left, 0, 0], m.LEFT)
        for row_, line in zip(t2[4][1:], REPORT_LINES):
            bars_lime(row_, line)
        amounts = m.VGroup(*[glyphs(r, line[line.index("$"):]) for r, line in zip(t2[4][1:], REPORT_LINES)])

        col = ContextColumn(center=(4.42, -0.52, 0), width=4.36, height=5.86)
        assert col.panel.get_left()[0] > script.get_right()[0] + 0.25 and col.panel.get_left()[0] > files.get_right()[0] + 0.3
        u = col.message("user", TASK)
        use1, use2, use3 = [col.message("tool_use", call_text(c), mono=True) for c in CALLS]
        res1 = col.message("tool_result", "\n".join(HEAD_OUT.split("\n")[:2]), mono=True)
        res2 = col.message("tool_result", RESULTS[1], mono=True)
        res3 = col.message("tool_result", REPORT_OUT, mono=True)
        paint(res3[2][1], REPORT_OUT, {"█+": LIME})
        answer = col.message("assistant", ANSWER)

        g4 = gutter(script[0], lines[3])
        g5 = gutter(script[0], lines[4])
        g69 = gutter(script[0], m.VGroup(*lines[5:9]))

        tally = m.VGroup(
            pill(f"{TOOL_CALLS} tool calls", SUN, SUN_FILL, 20),
            pill(f"{MODEL_CALLS} model calls", CORAL, CORAL_FILL, 20),
        ).arrange(m.RIGHT, buff=0.2)
        same = pill("agent_loop · unchanged", MINT, MINT_FILL, 20, mono=True)
        tally_row = m.VGroup(tally, same).arrange(m.RIGHT, buff=0.3)
        tally_row.move_to([0, t2.get_bottom()[1] - 0.6, 0]).align_to([work_left, 0, 0], m.LEFT)
        assert tally_row.get_right()[0] < col.panel.get_left()[0] - 0.2 and tally_row.get_bottom()[1] > -3.5, (tally_row.get_right(), tally_row.get_bottom())

        self.say(
            "[The request enters the context], [and the loop starts turning, just as it did for a bug.]",
            m.AnimationGroup(self.new_header("02 · same loop", "Doing the work by writing code"),
                             m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(files_label), m.FadeIn(f_csv, shift=0.1 * m.UP),
                             m.FadeIn(col.frame), col.push(u)),
            m.AnimationGroup(m.Create(loop), m.FadeIn(loop_label)),
        )
        self.say(
            "[First, Otto looks at the data]: [it prints the top of the file, to learn its columns.]",
            m.AnimationGroup(spin(), col.push(use1), m.FadeIn(t1[:4]), m.AddTextLetterByLetter(t1[4][0], time_per_char=0.04)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in t1[4][1:]], lag_ratio=0.3), col.push(res1)),
        )
        self.say(
            "[Instead of adding up a hundred and twenty rows in its head], [Otto acts the way a coding agent does best]: [it writes a script.]",
            m.Circumscribe(t1[4][1], color=AZURE, buff=0.06),
            m.AnimationGroup(spin(), col.push(use2)),
            m.AnimationGroup(
                m.FadeOut(t1), m.FadeIn(script[0]), m.FadeIn(script[2]),
                m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.RIGHT) for g in lines], lag_ratio=0.15),
                col.push(res2), m.FadeIn(f_py, shift=0.1 * m.UP),
            ),
        )
        self.say(
            "[The script reads every row], [adds each amount to its region], [and writes the report file, with a bar for each region.]",
            m.Create(g4),
            m.Transform(g4, g5),
            m.Transform(g4, g69),
        )
        self.say(
            "[Next turn: Otto runs it], [and the report comes back as a tool result.]",
            m.AnimationGroup(spin(), col.push(use3), m.FadeOut(script), m.FadeOut(g4),
                             m.FadeIn(t2[:4]), m.AddTextLetterByLetter(t2[4][0], time_per_char=0.04)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in t2[4][1:]], lag_ratio=0.25),
                             col.push(res3), m.FadeIn(f_md, shift=0.1 * m.UP)),
        )
        self.say(
            "[Every number was computed by Python], [not typed out by the model.]",
            m.Circumscribe(amounts, color=LIME, buff=0.08),
            m.Circumscribe(f_md, color=LIME, buff=0.06),
        )
        self.say(
            "[Then Otto answers]: [the report is ready, and North sold the most.]",
            spin(),
            col.push(answer),
        )
        self.say(
            "[Three tool calls, four model calls], [and not one line of the loop changed.]",
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in tally], lag_ratio=0.3),
            m.FadeIn(same, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · code as a thinking tool

    def thinking(self):
        self.next_section(
            "03 · Thinking in code",
            notes="Li, AI Agents in Depth, ch. 5, 'Code as a thinking tool': 40 students, 60% math, 45% physics, 25% "
            f"both; natural-language reasoning: math {MATH}, physics {PHYS}, both {BOTH}, 'only physics = 24 - 10 = 14' "
            f"(it subtracts from the math count); the code (shown as in the book, run here) prints {PUZZLE_OUT}. " + FACTS,
        )
        box = card(12.6, 1.5, EDGE, CARD).move_to([0, 1.72, 0])
        facts = m.VGroup(
            text("40 students.", 27, INK, weight=m.BOLD),
            text("60% take math,", 27, INK, t2c={"math": AZURE}),
            text("45% take physics,", 27, INK, t2c={"physics": ORCHID}),
            text("25% take both.", 27, INK),
        ).arrange(m.RIGHT, buff=0.32)
        on_baseline(facts)
        question = text("How many take only physics?", 27, LIME, weight=m.BOLD)
        m.VGroup(facts, question).arrange(m.DOWN, buff=0.2).move_to(box)
        source = top_cite(f"{BOOK} · ch. 5")

        def panel(x, label, color, sub):
            frame = card(6.15, 3.9, EDGE, CARD).move_to([x, -1.42, 0])
            head = m.VGroup(code(label, 22, color, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.RIGHT, buff=0.25)
            head.move_to(frame.get_corner(m.UL) + np.array([0.35, -0.42, 0]), aligned_edge=m.LEFT)
            return frame, head

        lw, lh = panel(-3.27, "in words", CORAL, "a model reasoning in text")
        rw, rh = panel(3.27, "in code", LIME, "the same model, writing code")
        steps = m.VGroup(
            text(f"60% of 40 take math = {MATH}", 24),
            text(f"45% of 40 take physics = {PHYS}", 24),
            text(f"25% of 40 take both = {BOTH}", 24),
        )
        slip_a = text(f"only physics = {MATH} − {BOTH}", 24, INK, t2c={str(MATH): ROSE})
        slip_b = text(f"= {WRONG}", 24, ROSE, weight=m.BOLD)
        slip_b.next_to(slip_a, m.RIGHT, buff=0.15)
        slip = m.VGroup(slip_a, slip_b)
        y0 = lh.get_bottom()[1] - 0.62
        for k, line in enumerate([*steps, slip_a]):
            line.align_to(lh, m.LEFT)
            sit(line, y0 - 0.5 * k)
        slip_b.next_to(slip_a, m.RIGHT, buff=0.15)
        sit(slip_b, baseline(slip_a))
        words_col = m.VGroup(steps, slip)
        bad = verdict("subtracted from the math group", False).next_to(words_col, m.DOWN, buff=0.3).align_to(lh, m.LEFT)

        prog = code(PUZZLE_PY, 21, INK, t2c=PY_T2C).next_to(rh, m.DOWN, buff=0.35).align_to(rh, m.LEFT)
        prog_lines = line_glyphs(prog, PUZZLE_PY)
        key = glyphs(prog, "phys - both")
        out = m.VGroup(code("$ python puzzle.py", 20, MUTED, t2c={r"^\$": MINT}), title_text(PUZZLE_OUT, 40, MINT),
                       verdict("right", True)).arrange(m.RIGHT, buff=0.35)
        out.next_to(prog, m.DOWN, buff=0.3).align_to(rh, m.LEFT)
        assert out.get_bottom()[1] > rw.get_bottom()[1] + 0.1 and bad.get_bottom()[1] > lw.get_bottom()[1] + 0.1
        assert prog.get_right()[0] < rw.get_right()[0] - 0.2

        self.say(
            "[Why write code just to add up numbers?] [Here's a puzzle from the book AI Agents in Depth.]",
            self.new_header("03 · thinking in code", "Words slip, code counts"),
            m.AnimationGroup(m.FadeIn(box), m.FadeIn(source)),
        )
        self.say(
            "[Forty students.] [Sixty percent take math], [forty-five percent take physics], [and twenty-five percent take both.] [How many take only physics?]",
            *[m.FadeIn(f, shift=0.1 * m.UP) for f in facts],
            m.FadeIn(question, shift=0.1 * m.UP),
        )
        self.say(
            "[Reasoning in words, a model can slip]: [twenty-four, minus ten], [makes fourteen.] [It subtracted from the wrong group.]",
            m.AnimationGroup(m.FadeIn(lw), m.FadeIn(lh), m.LaggedStart(*[m.FadeIn(s, shift=0.1 * m.RIGHT) for s in steps], lag_ratio=0.3)),
            m.FadeIn(slip_a, shift=0.1 * m.RIGHT),
            m.FadeIn(slip_b, scale=1.3),
            m.AnimationGroup(m.FadeIn(bad, shift=0.1 * m.UP), m.Circumscribe(glyphs(slip_a, str(MATH)), color=ROSE, buff=0.06)),
        )
        self.say(
            "[In code, every number has a name]: [physics minus both], [is eight], [and anyone can run it to check.]",
            m.AnimationGroup(m.FadeIn(rw), m.FadeIn(rh), m.LaggedStart(*[m.FadeIn(g, shift=0.1 * m.RIGHT) for g in prog_lines], lag_ratio=0.2)),
            m.Circumscribe(key, color=LIME, buff=0.06),
            m.AnimationGroup(m.FadeIn(out[0]), m.FadeIn(out[1], scale=1.3)),
            m.FadeIn(out[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · code makes tools

    def new_tools(self):
        self.next_section(
            "04 · New tools",
            notes="Li, AI Agents in Depth, ch. 5: code generation is 'a meta-capability': 'the ability to create new "
            "tools and capabilities dynamically at runtime'. report.py becomes a tool Otto wrote for itself. MEMORY.md "
            "(Episode 12) records where the data lives and how to run the report (wording illustrative). Next month "
            f"(October's file, illustrative) the run is {NEXT_TOOL_CALLS} tool call: {NEXT_TOP.split()[0]} leads again. " + FACTS,
        )
        q1 = text("Code is a meta-capability:", 34, INK, t2c={"meta-capability": LIME})
        q2 = text("it can create new tools.", 34, INK)
        quote = m.VGroup(q1, q2).arrange(m.RIGHT, buff=0.22).move_to([0, 1.85, 0])
        on_baseline(quote)
        source = cite(f"{BOOK} · ch. 5").next_to(quote, m.DOWN, buff=0.22)

        given = m.VGroup(*[mono_chip(t, SUN, SUN_FILL) for t in ("bash", "read_file", "edit_file", "write_file")]).arrange(m.RIGHT, buff=0.2)
        made = mono_chip("report.py", LIME, LIME_FILL)
        toolrow = m.VGroup(given, made).arrange(m.RIGHT, buff=0.55).move_to([0.6, 0.15, 0])
        label = code("Otto's tools", 20, MUTED).next_to(toolrow, m.LEFT, buff=0.35)
        made_tag = text("written by Otto", 20, LIME).next_to(made, m.DOWN, buff=0.15)
        assert label.get_left()[0] > -6.6 and made.get_right()[0] < 6.6

        mem = card(6.1, 2.05, ORCHID, CARD).move_to([-3.3, -2.2, 0])
        mem_title = code("MEMORY.md", 20, ORCHID, weight=m.BOLD).move_to(mem.get_corner(m.UL) + np.array([0.35, -0.35, 0]), aligned_edge=m.LEFT)
        mem_lines = m.VGroup(code("- sales data: sales.csv, monthly", 20, INK), code("- report: python report.py", 20, INK))
        mem_lines.arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT).next_to(mem_title, m.DOWN, buff=0.28).align_to(mem_title, m.LEFT)

        nxt = card(6.1, 2.05, EDGE, CARD).move_to([3.3, -2.2, 0])
        nxt_title = code("next month", 20, MUTED).move_to(nxt.get_corner(m.UL) + np.array([0.35, -0.35, 0]), aligned_edge=m.LEFT)
        nxt_call = mono_chip('bash("python report.py")', SUN, SUN_FILL)
        nxt_call.next_to(nxt_title, m.DOWN, buff=0.18).align_to(nxt_title, m.LEFT)
        nxt_count = m.VGroup(pill(f"{NEXT_TOOL_CALLS} tool call", SUN, SUN_FILL, 20), text(f"instead of {TOOL_CALLS}", 20, MUTED)).arrange(m.RIGHT, buff=0.25)
        nxt_count.next_to(nxt_call, m.DOWN, buff=0.16).align_to(nxt_title, m.LEFT)
        assert nxt_count.get_bottom()[1] > nxt.get_bottom()[1] + 0.05, (nxt_count.get_bottom(), nxt.get_bottom())
        assert mem_lines.get_right()[0] < mem.get_right()[0] - 0.2

        self.say(
            "[The book calls code a meta-capability]: [it can create new tools.] [That script is one]: [a tool Otto wrote for itself.]",
            m.AnimationGroup(self.new_header("04 · new tools", "Code makes tools"), m.FadeIn(q1, shift=0.15 * m.UP), m.FadeIn(source)),
            m.FadeIn(q2, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(label), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in given], lag_ratio=0.15),
                             m.FadeIn(made, shift=0.4 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(made_tag, shift=0.1 * m.UP), m.Circumscribe(made, color=LIME, buff=0.06)),
        )
        self.say(
            "[Otto notes in its memory file where the data lives], [and how to run the report.] [Next month, the whole job is one tool call.]",
            m.AnimationGroup(m.FadeIn(mem), m.FadeIn(mem_title), m.FadeIn(mem_lines[0], shift=0.1 * m.RIGHT)),
            m.FadeIn(mem_lines[1], shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(nxt), m.FadeIn(nxt_title), m.FadeIn(nxt_call, shift=0.15 * m.UP), m.FadeIn(nxt_count, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · the file system is the hub

    def hub(self):
        self.next_section(
            "05 · The hub",
            notes="Li, AI Agents in Depth, ch. 5, OpenClaw's storyboard: read MEMORY.md (PDF reports preferred, data in "
            "Google Sheets) → look up the Google Sheets API by web search and download the data by running code → a "
            "Python script (pandas aggregation, matplotlib charts) → report.pdf and charts/ → record the sheet's ID in "
            "MEMORY.md. 'The file system is the hub of information flow.' Card wording condensed. " + FACTS,
        )
        specs = (
            ("read memory", "MEMORY.md", (-4.55, 1.2)),
            ("look up the API", "Google Sheets docs", (0.0, 1.72)),
            ("write a script", "pandas, matplotlib", (4.55, 1.2)),
            ("save the results", "report.pdf, charts/", (3.3, -2.6)),
            ("update memory", "MEMORY.md: sheet ID", (-3.3, -2.6)),
        )
        cards = []
        for n, (verb, detail, pos) in enumerate(specs, 1):
            box = card(3.6, 1.2, EDGE, CARD)
            num = chip(str(n), LIME, LIME_FILL, width=0.5, height=0.46, size=20)
            top = m.VGroup(num, text(verb, 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
            body = m.VGroup(top, code(detail, 20, LIME)).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT).move_to(box)
            cards.append(m.VGroup(box, body).move_to([pos[0], pos[1], 0]))
        for c in cards:
            assert -6.6 < c.get_left()[0] and c.get_right()[0] < 6.6 and c[1].width < c[0].width - 0.3
        hub_box = card(4.7, 1.65, LIME, LIME_FILL).move_to([0, -0.5, 0])
        hub_words = m.VGroup(icon("ep27-folder", 0.62), m.VGroup(title_text("file system", 30), text("the hub of information flow", 20, MUTED)).arrange(m.DOWN, buff=0.08))
        hub_words.arrange(m.RIGHT, buff=0.25).move_to(hub_box)
        hub_node = m.VGroup(hub_box, hub_words)
        spokes = m.VGroup(*[
            m.Line(edge_point(c[0], hub_box.get_center()), edge_point(hub_box, c[0].get_center())).set_stroke(MUTED, 3)
            for c in cards
        ])
        source = top_cite(f"{BOOK} · ch. 5")
        flash = lambda s: m.ShowPassingFlash(s.copy().set_stroke(LIME, 7), time_width=0.6, run_time=0.7)  # noqa: E731

        self.say(
            "[The book's OpenClaw example does the same]: [read memory], [look up the Sheets API], [write a pandas script], [save charts and a PDF], [and update memory.]",
            m.AnimationGroup(self.new_header("05 · the hub", "Everything flows through files"), m.FadeIn(source)),
            *[m.FadeIn(c, shift=0.15 * m.UP) for c in cards],
        )
        self.say(
            "[Every step reads or writes a file.] [In the book's words], [the file system is the hub of information flow.]",
            m.AnimationGroup(m.FadeIn(hub_node, scale=0.9), m.LaggedStart(*[m.Create(s) for s in spokes], lag_ratio=0.15)),
            m.Circumscribe(hub_box, color=LIME, buff=0.08),
            m.Succession(*[flash(s) for s in spokes]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · general agents

    def general(self):
        self.next_section(
            "06 · General agents",
            notes="Li, AI Agents in Depth, ch. 5: a general-purpose agent targeting open-ended tasks has at its core a "
            "coding agent plus a file system; Claude Cowork is 'Anthropic's general-purpose Agent for knowledge work, "
            "reusing Claude Code's agentic architecture'; OpenClaw keeps memory in MEMORY.md and Markdown logs. Ch. 1: "
            "OpenClaw is local-first and works through 'messaging channels users already inhabit'; Manus was the first "
            "widely influential production agent to combine deep research, coding and computer use. Product facts "
            "change fast: " + FACTS,
        )
        lhs = title_text("general agent", 36)
        eq = title_text("=", 40, MUTED)
        a = recentre_chip(word_chip("coding agent", LIME, LIME_FILL, size=28, height=0.78), 28)
        plus = title_text("+", 40, MUTED)
        b = recentre_chip(word_chip("file system", LIME, LIME_FILL, size=28, height=0.78), 28)
        thesis = m.VGroup(lhs, eq, a, plus, b).arrange(m.RIGHT, buff=0.3).move_to([0, 1.85, 0])
        for t in (lhs, eq, plus):
            t.set_y(a.get_y())
        source = top_cite(f"{BOOK} · ch. 1, 5")

        def product(name, lines, x):
            box = card(4.05, 2.55, EDGE, CARD).move_to([x, -0.42, 0])
            head = title_text(name, 30).move_to(box.get_top() + 0.48 * m.DOWN)
            rows = m.VGroup(*[text(s, 21, INK if i < 2 else MUTED) for i, s in enumerate(lines)]).arrange(m.DOWN, buff=0.12)
            rows.next_to(head, m.DOWN, buff=0.25)
            return m.VGroup(box, head, rows)

        cc = product("Claude Code", ("Anthropic's coding agent:", "a shell and your files", "Cowork reuses its design"), -4.3)
        oc = product("OpenClaw", ("runs on your own machine", "answers in your chat apps", "memory in Markdown files"), 0.0)
        mn = product("Manus", ("deep research,", "computer use and coding,", "in one agent"), 4.3)
        on_baseline(m.VGroup(cc[1], oc[1], mn[1]))
        for p in (cc, oc, mn):
            assert p[2].width < p[0].width - 0.3 and p[2].get_bottom()[1] > p[0].get_bottom()[1] + 0.1, (p[1].text, p[2].width, p[2].get_bottom(), p[0].get_bottom())
        core_label = text("one core:", 24, MUTED)
        core = m.VGroup(*[recentre_chip(word_chip(w, LIME, LIME_FILL, size=24), 24) for w in ("a loop", "a shell", "a file system")]).arrange(m.RIGHT, buff=0.25)
        core_row = m.VGroup(core_label, core).arrange(m.RIGHT, buff=0.35).move_to([0, -2.72, 0])

        self.say(
            "[That's the book's thesis]: [a general agent for open-ended tasks is, at its core, a coding agent], [plus a file system.]",
            m.AnimationGroup(self.new_header("06 · general agents", "A coding agent, plus files"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(lhs, shift=0.15 * m.UP), m.FadeIn(eq), m.FadeIn(a, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(plus), m.FadeIn(b, shift=0.15 * m.UP)),
        )
        self.say(
            "[Claude Code is a coding agent built around a shell and your files.] [Anthropic reused its design for an agent that does knowledge work.]",
            m.AnimationGroup(m.FadeIn(cc[0]), m.FadeIn(cc[1], shift=0.1 * m.UP), m.FadeIn(cc[2][:2], shift=0.1 * m.UP)),
            m.FadeIn(cc[2][2], shift=0.1 * m.UP),
        )
        self.say(
            "[OpenClaw runs on your own machine], [answers in the chat apps you already use], [and keeps its memory in Markdown files.]",
            m.AnimationGroup(m.FadeIn(oc[0]), m.FadeIn(oc[1], shift=0.1 * m.UP), m.FadeIn(oc[2][0], shift=0.1 * m.UP)),
            m.FadeIn(oc[2][1], shift=0.1 * m.UP),
            m.FadeIn(oc[2][2], shift=0.1 * m.UP),
        )
        self.say(
            "[Manus brought deep research, computer use, and coding together in one agent.]",
            m.AnimationGroup(m.FadeIn(mn[0]), m.FadeIn(mn[1], shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in mn[2]], lag_ratio=0.3)),
        )
        self.say(
            "[Three products], [one core]: [a loop, a shell, and a file system.]",
            m.LaggedStart(*[m.Circumscribe(p[0], color=LIME, buff=0.06) for p in (cc, oc, mn)], lag_ratio=0.2),
            m.FadeIn(core_label, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in core], lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · why coding agents came first

    def why_code(self):
        self.next_section(
            "07 · Why code first",
            notes="Li, AI Agents in Depth, ch. 5: coding agents matured first 'not because code generation models are "
            "particularly powerful, but because decades of software engineering infrastructure naturally constitute a "
            "robust Harness': test suites provide clear acceptance criteria; linters and type checkers offer instant "
            "automated verification; Git provides version control and rollback. " + FACTS,
        )
        question = title_text("Why did coding agents mature first?", 42).move_to([0, 1.85, 0])
        not_this = verdict("especially strong code models", False, 22).move_to([0, 0.95, 0])
        specs = (("ep27-tests", "test suites", "say when the job is done"),
                 ("ep27-lint", "linters, type checkers", "check in seconds"),
                 ("ep27-git", "git", "can roll back any step"))
        cards = []
        for x, (pic, name, gloss) in zip((-4.3, 0.0, 4.3), specs):
            box = card(4.05, 2.15, LIME, CARD, stroke=2).move_to([x, -0.95, 0])
            p = icon(pic, 0.85).move_to([x, -0.45, 0])
            n = sit(text(name, 26, INK, weight=m.BOLD).move_to([x, -1.2, 0]), -1.3)
            g = sit(text(gloss, 22, MUTED).move_to([x, -1.65, 0]), -1.76)
            cards.append((box, p, n, g))
        quote = text("“Decades of software engineering infrastructure naturally constitute a robust harness.”", 23, INK)
        quote.move_to([0, -2.65, 0])
        source = cite(f"{BOOK} · ch. 5").next_to(quote, m.DOWN, buff=0.14)
        assert quote.width < 12.8 and source.get_bottom()[1] > -3.5

        self.say(
            "[Why did coding agents mature first?] [Not because the models are especially strong at code.]",
            m.AnimationGroup(self.new_header("07 · why code first", "The harness was already there"), m.FadeIn(question, shift=0.15 * m.UP)),
            m.FadeIn(not_this, shift=0.1 * m.UP),
        )
        self.say(
            "[Decades of software engineering had already built a harness]: [test suites], [linters and type checkers], [and git.]",
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(source)),
            *[m.AnimationGroup(m.FadeIn(c[0]), m.FadeIn(c[1], shift=0.15 * m.UP), m.FadeIn(c[2], shift=0.1 * m.UP)) for c in cards],
        )
        self.say(
            "[Tests say when the job is done], [linters check in seconds], [and git can roll back any step.]",
            *[m.FadeIn(c[3], shift=0.1 * m.UP) for c in cards],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · goal clarity × verification

    def grid(self):
        self.next_section(
            "08 · Verification",
            notes="Li, AI Agents in Depth, ch. 5, the goal × verification table: clear + automatic = 'Sweet spot: fixing "
            "bugs with test cases'; clear + manual = 'Throughput-limited: code refactoring requires manual review'; vague "
            "+ automatic = 'Efficiently going off track: optimizing code quality with a linter'; vague + manual = 'Hard to "
            "start: make the UI look better'. Otto's report split in two is this episode's illustration; the check (the "
            f"report's region totals add up to the sum of all {CHECK_ROWS} rows) is computed. " + FACTS,
        )
        cw, chh = 4.85, 2.25
        xs, ys = (-1.3, 3.75), (0.92, -1.53)
        cells = {}
        for r, y in enumerate(ys):
            for c, x in enumerate(xs):
                cells[r, c] = card(cw, chh, EDGE, CARD, stroke=2).move_to([x, y, 0])
        frame = m.VGroup(*cells.values())
        col_heads = m.VGroup(text("checked automatically", 24, INK, weight=m.BOLD), text("checked by a person", 24, INK, weight=m.BOLD))
        for h, x in zip(col_heads, xs):
            h.move_to([x, ys[0] + chh / 2 + 0.32, 0])
        on_baseline(col_heads)
        row_heads = m.VGroup(text("clear goal", 26, INK, weight=m.BOLD), text("vague goal", 26, INK, weight=m.BOLD))
        for h, y in zip(row_heads, ys):
            h.move_to([-5.15, y, 0])
        source = top_cite(f"{BOOK} · ch. 5")

        def fill(r, c, title, example, color, fill_, stroke=None):
            box = cells[r, c]
            t = text(title, 26, color, weight=m.BOLD).move_to(box.get_corner(m.UL) + np.array([0.3, -0.42, 0]), aligned_edge=m.LEFT)
            e = text(example, 21, MUTED).move_to(box.get_corner(m.UL) + np.array([0.3, -0.88, 0]), aligned_edge=m.LEFT)
            sit(t, box.get_top()[1] - 0.55)
            sit(e, box.get_top()[1] - 0.98)
            assert t.width < cw - 0.5 and e.width < cw - 0.5
            light = box.animate.set_stroke(stroke or color, width=2.5).set_fill(fill_, opacity=1)
            return light, t, e

        sweet = fill(0, 0, "sweet spot", "fixing a bug, with a test case", MINT, MINT_FILL)
        slow = fill(0, 1, "throughput-limited", "a refactor that needs review", SUN, SUN_FILL)
        stuck = fill(1, 1, "hard to start", "“make the UI look better”", INK, CARD_HI, MUTED)
        astray = fill(1, 0, "efficiently going off track", "“code quality”, scored by a linter", ROSE, ROSE_FILL)
        mini = otto(0.42).move_to(cells[0, 0].get_corner(m.UR) + np.array([-0.55, -0.55, 0]))

        num_chip = word_chip("report: the numbers  ✓", MINT, MINT_FILL, size=21, height=0.56)
        num_chip.move_to(cells[0, 0].get_corner(m.DL) + np.array([0.3, 0.5, 0]), aligned_edge=m.LEFT)
        use_chip = word_chip("report: is it useful?", SUN, SUN_FILL, size=21, height=0.56)
        use_chip.move_to(cells[0, 1].get_corner(m.DL) + np.array([0.3, 0.5, 0]), aligned_edge=m.LEFT)
        check_note = code(f"check: the regions add up to all {CHECK_ROWS} rows", 20, MUTED).move_to([xs[0] + 1.2, -3.12, 0])
        check_note.align_to(cells[1, 0], m.LEFT)
        assert check_note.get_bottom()[1] > -3.5

        self.say(
            "[The book sorts tasks on two axes]: [how clear the goal is], [and whether the result can be checked automatically.]",
            m.AnimationGroup(self.new_header("08 · verification", "What can be checked"), m.FadeIn(source),
                             m.LaggedStart(*[m.FadeIn(b, scale=0.95) for b in frame], lag_ratio=0.1)),
            m.LaggedStart(*[m.FadeIn(h, shift=0.15 * m.RIGHT) for h in row_heads], lag_ratio=0.3),
            m.LaggedStart(*[m.FadeIn(h, shift=0.15 * m.DOWN) for h in col_heads], lag_ratio=0.3),
        )
        self.say(
            "[Clear goal, automatic checks]: [the sweet spot. Fixing a bug with a failing test lives here.]",
            m.AnimationGroup(sweet[0], m.FadeIn(sweet[1], shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(sweet[2], shift=0.1 * m.UP), m.FadeIn(mini, shift=0.15 * m.UP)),
        )
        self.say(
            "[A clear goal that needs a person to review]: [throughput-limited, like a big refactor.]",
            m.AnimationGroup(slow[0], m.FadeIn(slow[1], shift=0.1 * m.UP)),
            m.FadeIn(slow[2], shift=0.1 * m.UP),
        )
        self.say(
            "[A vague goal, checked by a person], [is hard even to start]: [make the UI look better.]",
            stuck[0],
            m.FadeIn(stuck[1], shift=0.1 * m.UP),
            m.FadeIn(stuck[2], shift=0.1 * m.UP),
        )
        self.say(
            "[And a vague goal with automatic checks?] [Efficiently going off track], [like chasing code quality with a linter.]",
            astray[0],
            m.FadeIn(astray[1], shift=0.1 * m.UP),
            m.FadeIn(astray[2], shift=0.1 * m.UP),
        )
        self.say(
            "[Otto's report splits in two]: [its numbers can be checked by code], [but whether it helps, only the owner can judge.]",
            m.Circumscribe(m.VGroup(cells[0, 0], cells[0, 1]), color=LIME, buff=0.08),
            m.AnimationGroup(m.FadeIn(num_chip, shift=0.15 * m.UP), m.FadeIn(check_note, shift=0.1 * m.UP)),
            m.FadeIn(use_chip, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · checks can be gamed

    def gaming(self):
        self.next_section(
            "09 · Constraints",
            notes="Li, AI Agents in Depth, ch. 5 (reward hacking): 'Deleting and rebuilding the database to fix a database "
            "fault does repair it, but the data is gone'; 'deleting all the code to fix a compilation error does make "
            "compilation pass, but the implementation is gone'; hence 'constraining actions, not merely outcomes'. "
            f"Otto's case is illustrative but run here: one row's amount set to n/a makes report.py fail ({BAD_ERROR}); "
            f"deleting that row makes it pass and the check still holds ({GAMED_ROWS} of {N_ROWS} rows). Permissions: "
            "Episode 17; hooks: Episode 18. " + FACTS,
        )
        source = top_cite(f"{BOOK} · ch. 5")

        def case(x, title, ok, bad):
            box = card(6.3, 1.6, EDGE, CARD).move_to([x, 1.78, 0])
            t = sit(text(title, 23, INK, weight=m.BOLD).move_to(box.get_top() + 0.45 * m.DOWN), box.get_top()[1] - 0.55)
            pills = m.VGroup(verdict(ok, True), verdict(bad, False)).arrange(m.RIGHT, buff=0.25).move_to(box.get_bottom() + 0.46 * m.UP)
            assert t.width < 5.9 and pills.width < 5.9, (t.width, pills.width)
            return box, t, pills

        db = case(-3.3, "Rebuild the database to fix a fault", "fixed", "the data is gone")
        cc = case(3.3, "Delete the code to fix a compile error", "it compiles", "the code is gone")

        mine = code("Otto's report, if one row won't parse", 20, MUTED).move_to([-6.4, 0.62, 0], aligned_edge=m.LEFT)
        msg = BAD_ERROR.split(": ", 1)
        err = terminal(["$ python report.py", f"! {msg[0]}: could not convert", "! string to float: 'n/a'"], width=6.1, size=20)
        err.next_to(mine, m.DOWN, buff=0.12).align_to(mine, m.LEFT)
        hack = message("tool_use", 'edit_file("sales.csv", bad_row, "")', width=6.1, size=20, mono=True)
        hack.next_to(err, m.DOWN, buff=0.12).align_to(mine, m.LEFT)
        gamed = m.VGroup(verdict("totals still match", True), verdict(f"{N_ROWS - GAMED_ROWS} sale is gone", False)).arrange(m.RIGHT, buff=0.25)
        gamed.next_to(hack, m.DOWN, buff=0.12).align_to(mine, m.LEFT)
        assert gamed.get_bottom()[1] > -3.5, gamed.get_bottom()

        rule1 = title_text("Constrain actions,", 34).move_to([3.3, -0.05, 0])
        rule2 = title_text("not only outcomes.", 34, LIME).next_to(rule1, m.DOWN, buff=0.12)
        lock_row = m.VGroup(icon(LOCK, 0.55), code("sales.csv is read-only", 22, INK)).arrange(m.RIGHT, buff=0.25)
        lock_row.move_to([3.3, -1.55, 0])
        denied = verdict("denied by the harness", False, 22).move_to([3.3, hack.get_y(), 0])
        refs = text("permissions: Episode 17 · hooks: Episode 18", 20, MUTED).move_to([3.3, gamed.get_y(), 0])
        assert denied.get_left()[0] > hack.get_right()[0] + 0.2

        self.say(
            "[But an automatic check can be gamed.] [Rebuild a database to fix a fault]: [it's fixed, but the data is gone.]",
            m.AnimationGroup(self.new_header("09 · constraints", "Checks can be gamed"), m.FadeIn(source)),
            m.AnimationGroup(m.FadeIn(db[0]), m.FadeIn(db[1], shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, scale=1.15) for p in db[2]], lag_ratio=0.5),
        )
        self.say(
            "[Delete all the code to fix a compile error]: [it compiles, but the code is gone.]",
            m.AnimationGroup(m.FadeIn(cc[0]), m.FadeIn(cc[1], shift=0.1 * m.UP)),
            m.LaggedStart(*[m.FadeIn(p, scale=1.15) for p in cc[2]], lag_ratio=0.5),
        )
        self.say(
            "[Otto's check could be gamed too.] [If one row won't parse], [deleting it makes the error go away], [and the totals still match.]",
            m.FadeIn(mine, shift=0.1 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(err[:4]), m.FadeIn(err[4], shift=0.1 * m.UP)),
            m.FadeIn(hack, shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(p, scale=1.15) for p in gamed], lag_ratio=0.5),
        )
        self.say(
            "[So constrain actions, not only outcomes]: [the sales file is read-only, and the harness denies the edit], [as in episodes seventeen and eighteen.]",
            m.AnimationGroup(m.FadeIn(rule1, shift=0.15 * m.UP), m.FadeIn(rule2, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(lock_row, shift=0.1 * m.UP), m.FadeIn(denied, shift=0.2 * m.LEFT),
                             hack[0].animate.set_stroke(ROSE), hack[1].animate.set_fill(ROSE)),
            m.FadeIn(refs, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Code (the action, and a maker of tools), files (the hub), checks (they decide what works; constrain actions too).")
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Code", LIME), ("Files", AZURE), ("Checks", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("acts, and makes tools", "the hub of the work", "decide what works")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        for s in subs:
            s.shift((words[0][0].get_bottom()[1] - 0.55 - s.get_top()[1]) * m.UP)
        mid = words[0][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(p.get_right()[0] + q.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for p, q in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tag1 = text("A shell and a file system make an agent general.", 30, MUTED)
        tag2 = text("Checks decide what works.", 30, INK)
        tagline = m.VGroup(tag1, tag2).arrange(m.DOWN, buff=0.2).next_to(row_, m.DOWN, buff=0.65)
        assert row_.width < 12.8 and tagline.width < 12.8

        self.say(
            "[To sum up]: [code], [files], [checks.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[A shell and a file system make an agent general.] [Checks decide what works.]",
            m.FadeIn(tag1, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(tag2, shift=0.1 * m.UP), m.Circumscribe(words[2], color=MINT, buff=0.12)),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
