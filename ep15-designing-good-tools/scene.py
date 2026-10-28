"""Agents, Animated · Episode 15 · Designing Good Tools  (Module 4 · Tools & Safety)

What makes a tool easy for a model to use well. Running example: Otto, the series' coding agent, is
asked to make tinyshop's receipt say "Thank you" instead of "Thanks". It needs a tool to find the line
and a tool to change it; both are designed badly first, then redesigned. The runs below are not drawn by
hand: the scene walks a small in-memory tinyshop with API-shaped file tools (counting the calls), runs the
redesigned search, runs harness_reference's edit_file behind a harness that "tidies" curly quotes
(the loop that never ends), then without it, then the redesigned edit_file with errors that teach, and a
paging read_file. Every number on screen comes from those runs and is asserted.

Arc: hook (blame the tools first; Li ch. 1) · 1 goals, not APIs (17 calls vs 1; ACI) · 2 general or
dedicated (read_document, one interpreter, the four reasons for a dedicated tool) · 3 descriptions (when,
cannot, formats, returns, cost; the 45% ablation) · 4 fidelity (the curly-quote loop; the fix; the rule) ·
5 edit formats (diff + apply, old → new, line numbers that shift, anchors) · 6 errors that teach (closest
line and the differing character; "appears twice") · 7 honest reads ("Displayed lines 1-200 of 5000") ·
recap (Goals · Limits · Fidelity).

Colours (series roles): model CORAL, tool call SUN, tool result MINT, user AZURE; good MINT, bad ROSE;
the harness neutral (EDGE/MUTED); headers MINT (Module 4).

Facts (checked October 2026; Li = Li Bojie, "AI Agents in Depth", bojieli.github.io/ai-agent-book):
- Li ch. 1 (observation and action spaces): "Many problems that appear to require a 'smarter model' are
  really interface problems."
- Li ch. 4 (universal principles of tool design): ACI, "a tool should correspond to the Agent's goal, not
  to an underlying API operation"; early API wrappers made the agent "coordinate several tools to
  accomplish one goal"; "Too fine, and tools proliferate"; "general tools are preferable to dedicated
  tools, unless there is a clear security, permission, or performance reason"; four reasons: security,
  permissions and auditing / hiding platform differences and giving better feedback / extremely high
  usage frequency / complex parameter structure; extract_pdf_text, extract_docx_content,
  extract_pptx_content → one read_document with a file_type parameter; "a single Python interpreter
  replaces dozens of single-purpose tools"; "each tool's definition occupies hundreds of tokens".
- Li ch. 4 (the art of tool description): say when to use it; "the root cause of most tool call failures
  is not that the model doesn't know what the tool can do, but that it doesn't know what the tool cannot
  do"; concrete formats (E.164 phone numbers, RFC3339 timestamps); return shape; cost; "check the tool
  descriptions first" rather than doubting the model.
- Li ch. 2 (experiment 2-4, the author's own Tau-bench ablation): removing all descriptive text (names
  and parameters kept) raised the tool-call error rate by 45% (model not named; read as a relative rise).
- Li ch. 4 (fidelity of parameter passing): a version of Cursor from early 2026; its edit tool's
  parameter layer silently converted curly quotation marks to straight ones; old_string copied from the
  file no longer matched, "no match found"; the model "tries repeatedly and fails repeatedly" — "it cannot
  understand why the tool can't find what it clearly saw"; "there must be no systematic discrepancy
  between the world the model perceives and the world the tool operates on"; any normalisation must be
  documented in the description and reported in the result.
- Li ch. 5 (file-editing tools): diff + fast apply model (Cursor); old string → new string (Claude Code),
  must be unique, "a single character deviation causes the match to fail"; line numbers, "each edit
  shifts the line numbers that follow it"; start + end anchors (unique combination).
- Li ch. 4 (perception tools): "Displayed lines 1-200 of 5000; use the offset parameter to continue
  reading"; "Silent truncation is dangerous".
- learn-claude-code s02: edit_file returns "Error: text not found". harness_reference.py: edit_file(path,
  old_text, new_text) replaces the first match silently.
Illustrative (for teaching): tinyshop's files and the receipt text, the API-shaped tool set, the search
tool's description, the four retries, the inserted line, orders.csv's rows (5000 lines is Li's example).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import difflib
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403
from kit import _PY_KEYWORDS  # noqa: E402

REF = (SERIES / "harness_reference.py").read_text()
assert "def edit_file(path, old_text, new_text):" in REF
assert "source.replace(old_text, new_text, 1)" in REF  # the reference replaces the first match, unchecked
assert 'TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}' in REF

# ---------------------------------------------------------------------------- tinyshop (illustrative)

LQ, RQ = "“", "”"  # the curly quotes in the receipt text
NOTE = f"Thanks from {LQ}tinyshop{RQ}!"
NEW_NOTE = NOTE.replace("Thanks", "Thank you")
CHECKOUT = (
    "from cart import total\n"
    "\n"
    "def receipt(items):\n"
    f'    note = "{NOTE}"\n'
    '    return f"{total(items)}: {note}"'
)
NOTE_ROW = 3  # 0-based row of the note line (line 4)
assert CHECKOUT.split("\n")[NOTE_ROW].strip() == f'note = "{NOTE}"'
REPO = {
    "README.md": "# tinyshop\nA tiny web shop.",
    "cart.py": "def total(items):\n    s = 0\n    for item in items:\n        s += item\n    return s",
    "checkout.py": CHECKOUT,
    "tests/test_cart.py": "from cart import total\n\n\ndef test_total():\n    assert total([3, 4]) == 7",
    "tests/test_checkout.py": 'from checkout import receipt\n\n\ndef test_receipt():\n    assert "Thanks" in receipt([3])',
}
TASK = 'Make the receipt say "Thank you", not "Thanks".'

# ---------------------------------------------------------------------------- 1 · API-shaped tools: count the calls

API_CALLS = []
HANDLES = {}


def list_dir(path):
    API_CALLS.append(("list", path))
    prefix = "" if path == "." else path + "/"
    names = set()
    for name in REPO:
        if name.startswith(prefix):
            rest = name[len(prefix):]
            names.add(rest.split("/")[0] + "/" if "/" in rest else rest)
    return sorted(names, key=lambda n: (n.endswith("/"), n.lower()))


def open_file(path):
    API_CALLS.append(("open", path))
    HANDLES[len(HANDLES) + 1] = path
    return len(HANDLES)


def read_chunk(handle, offset=0, size=4096):
    API_CALLS.append(("read", HANDLES[handle]))
    return REPO[HANDLES[handle]][offset:offset + size]


def close_file(handle):
    API_CALLS.append(("close", HANDLES[handle]))


def find_everywhere(text_):
    """What Otto must do with API-shaped tools to find every line holding `text_`."""
    hits, folders = [], ["."]
    while folders:
        folder = folders.pop(0)
        for name in list_dir(folder):
            path = name if folder == "." else f"{folder}/{name}"
            if name.endswith("/"):
                folders.append(path.rstrip("/"))
                continue
            handle, data = open_file(path), ""
            while True:
                chunk = read_chunk(handle, len(data))
                data += chunk
                if len(chunk) < 4096:
                    break
            close_file(handle)
            hits += [f"{path}:{i}" for i, line in enumerate(data.split("\n"), 1) if text_ in line]
    return hits


WALK_HITS = find_everywhere("Thanks")
N_API = len(API_CALLS)
assert WALK_HITS == ["checkout.py:4", "tests/test_checkout.py:5"]
assert N_API == 17 and [c for c, _ in API_CALLS].count("list") == 2  # "seventeen calls"
WALK_ROWS = []  # one row per folder listing or file: [(label, [ops])]
for op, path in API_CALLS:
    if op == "list":
        WALK_ROWS.append((path + "/" if path != "." else "./", ["list"]))
    elif op == "open":
        WALK_ROWS.append((path.split("/")[-1], ["open"]))
    else:
        WALK_ROWS[-1][1].append(op)
assert len(WALK_ROWS) == 7 and sum(len(ops) for _, ops in WALK_ROWS) == N_API

# ---------------------------------------------------------------------------- the redesigned search


def search(pattern, path="."):
    """Goal-shaped: one call returns every matching line as path:line: text (at most 50)."""
    rows = []
    for name, src in REPO.items():
        if name.startswith(".git/") or "\0" in src:
            continue
        rows += [f"{name}:{i}: {line.strip()}" for i, line in enumerate(src.split("\n"), 1) if re.search(pattern, line)]
    more = len(rows) - 50
    return "\n".join(rows[:50]) + (f"\n({more} more not shown)" if more > 0 else "")


SEARCH_OUT = search("Thanks")
assert [r.split(": ")[0] for r in SEARCH_OUT.split("\n")] == WALK_HITS  # the same two places, in one call

SEARCH_BAD = "Search files."
DESC = [  # the redesigned description, line by line, with what each line tells the model
    ("when", "Find lines that match a regex."),
    ("when", "Use it to find code before editing."),
    ("cannot", "Cannot search .git, binaries, or meaning."),
    ("format", r'pattern: Python regex, e.g. "def total\("'),
    ("returns", "Returns path:line: text, at most 50."),
    ("cost", "Fast: one pass over the files."),
]
assert "at most 50" in DESC[4][1] and "rows[:50]" in Path(__file__).read_text()  # the description tells the truth
assert re.search(r"def total\(", REPO["cart.py"])  # the example pattern works on tinyshop
ABLATION = 0.45  # Li ch. 2: tool-call error rate +45% without description text (author's own test)

# ---------------------------------------------------------------------------- 4 · the quote loop, run


def tidy(s):
    """The bad harness's 'helpful' input clean-up: curly quotes become straight ones."""
    return s.replace(LQ, '"').replace(RQ, '"')


def edit_file(path, old_text, new_text):  # harness_reference.edit_file + learn-claude-code s02's error
    source = FILES[path]
    if old_text not in source:
        return "Error: text not found"
    FILES[path] = source.replace(old_text, new_text, 1)
    return f"edited {path}"


def read_file(path):
    return FILES[path]


TOOL_HANDLERS = {"read_file": read_file, "edit_file": edit_file}


def run_tool(block, tidy_inputs):
    args = {k: tidy(v) for k, v in block.input.items()} if tidy_inputs else block.input
    return TOOL_HANDLERS[block.name](**args)


def otto_turns(results, max_calls):
    """A scripted Otto: read the file, copy the line it saw into edit_file; on an error, read again."""
    calls = []
    while len(calls) < max_calls:
        if not calls or results[-1].startswith("Error"):
            calls.append(NS(name="read_file", input={"path": "checkout.py"}))
        else:
            seen = results[-1]
            if seen.startswith("edited"):
                break
            old = seen.split("\n")[NOTE_ROW].split('"', 1)[1].rsplit('"', 1)[0]  # copied from what it read
            calls.append(NS(name="edit_file", input={"path": "checkout.py", "old_text": old,
                                                      "new_text": old.replace("Thanks", "Thank you")}))
        yield calls[-1]


def run(tidy_inputs, max_calls=8):
    results = []
    for block in otto_turns(results, max_calls):
        results.append(run_tool(block, tidy_inputs))
    return results


FILES = dict(REPO)
LOOP = run(tidy_inputs=True)
EDIT_RESULTS = LOOP[1::2]
ATTEMPTS = len(EDIT_RESULTS)
assert ATTEMPTS == 4 and set(EDIT_RESULTS) == {"Error: text not found"}  # four tries, four failures
assert all(r == CHECKOUT for r in LOOP[0::2]) and FILES["checkout.py"] == CHECKOUT  # Otto sees the same line
OLD_TEXT, NEW_TEXT = NOTE, NEW_NOTE
assert OLD_TEXT in CHECKOUT and tidy(OLD_TEXT) not in CHECKOUT  # what Otto sent is there; what arrived isn't

FILES = dict(REPO)
FIXED = run(tidy_inputs=False)
assert FIXED == [CHECKOUT, "edited checkout.py"]  # first try
assert f'note = "{NEW_NOTE}"' in FILES["checkout.py"] and LQ in FILES["checkout.py"]  # quotes kept

# ---------------------------------------------------------------------------- 5 · line numbers shift

INSERTED = "from tax import vat"  # edit 1: insert after line 1 (illustrative)
_lines = CHECKOUT.split("\n")
assert _lines[NOTE_ROW].strip().startswith("note")  # edit 2 was planned on the original: "line 4"
_after = _lines[:1] + [INSERTED] + _lines[1:]
LANDS_ON = _after[NOTE_ROW]
assert LANDS_ON == "def receipt(items):"  # after the insert, "line 4" is another line
assert CHECKOUT.count(NOTE) == 1  # exact text: found exactly once

# ---------------------------------------------------------------------------- 6 · errors that teach

NEW_EDIT_SRC = '''def edit_file(path, old_text, new_text):
    source = open(path).read()
    n = source.count(old_text)
    if n != 1:  # explain, don't just fail
        return edit_error(path, source, old_text, n)
    new = source.replace(old_text, new_text)
    open(path, "w").write(new)
    return f"edited {path}"'''
assert len(NEW_EDIT_SRC.split("\n")) <= 10


def edit_error(path, source, old_text, n):
    lines = source.split("\n")
    if n > 1:
        where = [i for i, line in enumerate(lines, 1) if old_text in line]
        return (f"Error: old_text appears {n} times in\n"
                f"{path} (lines {where[0]} and {where[1]}). Add a\n"
                "line around it to match only once.")
    best = max(range(len(lines)), key=lambda i: difflib.SequenceMatcher(None, old_text, lines[i]).ratio())
    line = lines[best]
    start = difflib.SequenceMatcher(None, line, old_text).find_longest_match(0, len(line), 0, len(old_text))
    offset = start.a - start.b  # where old_text would sit in the line
    k = next(j for j, c in enumerate(old_text) if line[offset + j] != c)
    return ("Error: old_text not found. Closest:\n"
            f"{path} line {best + 1}, column {offset + k + 1}: the\n"
            f"file has {line[offset + k]}, old_text has {old_text[k]}.")


ERR_QUOTE = edit_error("checkout.py", CHECKOUT, tidy(OLD_TEXT), CHECKOUT.count(tidy(OLD_TEXT)))
ERR_TWICE = edit_error("checkout.py", CHECKOUT, "total", CHECKOUT.count("total"))
assert ERR_QUOTE.split("\n")[1].startswith(f"checkout.py line {NOTE_ROW + 1}, column ")
assert f"file has {LQ}, old_text has \"" in ERR_QUOTE  # names the one character that differs
QUOTE_COL = int(re.search(r"column (\d+)", ERR_QUOTE).group(1))
assert CHECKOUT.split("\n")[NOTE_ROW][QUOTE_COL - 1] == LQ
assert ERR_TWICE.startswith("Error: old_text appears 2 times") and "(lines 1 and 5)" in ERR_TWICE
assert max(len(x) for x in (ERR_QUOTE + "\n" + ERR_TWICE).split("\n")) <= 35

# ---------------------------------------------------------------------------- 7 · honest reads

CSV = "\n".join(["id,item,price"] + [f"{i},item-{i % 7},{3 + i % 5}" for i in range(1, 5000)])
FILES["orders.csv"] = CSV


def read_page(path, offset=0, limit=200):
    lines = FILES[path].split("\n")
    shown = lines[offset:offset + limit]
    note = f"Displayed lines {offset + 1}-{offset + len(shown)} of {len(lines)}"
    if offset + len(shown) < len(lines):
        note += "; use the offset parameter to continue reading."
    return shown, note


PAGE1, NOTE1 = read_page("orders.csv")
PAGE2, NOTE2 = read_page("orders.csv", offset=200)
TOTAL_LINES = len(CSV.split("\n"))
assert TOTAL_LINES == 5000 and len(PAGE1) == 200
assert NOTE1 == "Displayed lines 1-200 of 5000; use the offset parameter to continue reading."  # Li's words
assert NOTE2.startswith("Displayed lines 201-400 of 5000")
PAGE_FRAC = len(PAGE1) / TOTAL_LINES
assert PAGE_FRAC == 0.04

# ---------------------------------------------------------------------------- look

PY_T2C = {
    r"#[^\n]*": FAINT,
    r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
    _PY_KEYWORDS: ORCHID,
    r"\b\d+(\.\d+)?\b": SUN,
    r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"
SEARCH_ICON = recolored("search", SUN, SUN_FILL, prefix="ep15")
DOC_ICON = recolored("doc", SUN, SUN_FILL, prefix="ep15")
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix="ep15")


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    """The baseline of a run of glyphs: the median bottom of its flat-bottomed letters."""
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def span(words, src, key, nth=0):
    """The glyphs of the nth `key` in Text `words` made from `src` (whitespace draws no glyph)."""
    i = -1
    for _ in range(nth + 1):
        i = src.index(key, i + 1)
    k = sum(not c.isspace() for c in src[:i])
    return words[k:k + sum(not c.isspace() for c in key)]


class Listing(m.VGroup):
    """A file as read_file shows it: line numbers in a gutter, Python lightly highlighted, one Text per
    line (so a line can be swapped or moved on its own). `rows` leaves room for lines inserted later.
    Row positions are computed from the card's current place."""

    def __init__(self, src, title, size=20, rows=None, width=None, accent=EDGE):
        super().__init__()
        self.size, self.src = size, src.split("\n")
        whole = code(src, size, INK)  # where every glyph sits; each line is then its own Text
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            if n:
                words = code(line.strip(), size, INK, t2c=PY_T2C)
                words.shift(whole[k].get_center() - words[0].get_center())
                self.lines.append(words)
            else:
                self.lines.append(None)
            k += n
        known = [(i, baseline(w, w.text.replace(" ", ""))) for i, w in enumerate(self.lines) if w is not None]
        (i0, b0), (i1, b1) = known[0], known[-1]
        self.pitch = (b0 - b1) / (i1 - i0)
        self.body = m.VGroup(*[w for w in self.lines if w is not None])
        n_rows = rows or len(self.src)
        self.nums = m.VGroup(*[code(str(i + 1), size, FAINT) for i in range(n_rows)])
        num_w = max(nm.width for nm in self.nums)
        self.cap = code(title, 20, MUTED)
        top = self.cap.height + 0.28
        w = max(width or 0, 0.35 + num_w + 0.35 + whole.width + 0.45)
        h = top + whole.height + (n_rows - len(self.src)) * self.pitch + 0.55
        self.box = card(w, h, accent, CARD, radius=0.16, stroke=2)
        self.cap.align_to(self.box, m.UL).shift(0.35 * m.RIGHT + 0.18 * m.DOWN)
        whole_top, whole_left = whole.get_top()[1], whole.get_left()[0]
        dy = self.box.get_top()[1] - (top + 0.22) - whole_top
        dx = self.box.get_left()[0] + 0.35 + num_w + 0.35 - whole_left
        self.body.shift([dx, dy, 0])
        self.off0 = self.box.get_top()[1] - (b0 + dy + i0 * self.pitch)
        right = self.box.get_left()[0] + 0.35 + num_w
        for i, nm in enumerate(self.nums):
            nm.shift([right - nm.get_right()[0], self.row_y(i) - nm.get_bottom()[1], 0])
        self.body.set_z_index(2)
        self.add(self.box, self.nums, self.body, self.cap)

    def row_y(self, i):
        return self.box.get_top()[1] - self.off0 - i * self.pitch

    def part(self, i, key, nth=0):
        return span(self.lines[i], self.lines[i].text, key, nth)

    def make_line(self, s, i, like):
        """A new line of code on row i; `like` = a glyph of the same character its first glyph lines up with."""
        words = code(s.strip(), self.size, INK, t2c=PY_T2C).set_z_index(2)
        words.shift([like.get_center()[0] - words[0].get_center()[0],
                     self.row_y(i) - baseline(words, s.replace(" ", "")), 0])
        return words

    def band(self, i, color=SUN, opacity=0.16):
        top, bot = self.row_y(i) + 0.27, self.row_y(i) - 0.11
        left, right = self.box.get_left()[0] + 0.1, self.box.get_right()[0] - 0.1
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=opacity)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


def boxed(body, src, title, accent, width=None):
    """A Text `body` (made from `src`) in a card with a bold caption: VGroup(box, body, cap), .src = src."""
    cap = code(title, 20, accent, weight=m.BOLD)
    top = cap.height + 0.24
    w = max(width or 0, body.width + 0.7, cap.width + 0.7)
    box = card(w, body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
    group = m.VGroup(box, body, cap)
    group.src = src
    body.set_z_index(2)
    return group


def field_card(title, rows, accent, size=20, width=None, value_color=INK):
    """A call or result as key/value rows in Menlo (columns line up: monospace). Rows with an empty key
    are plain lines."""
    key_w = max(len(k) for k, _ in rows) + 2
    src = "\n".join(f"{k:<{key_w}}{v}" if k else v for k, v in rows)
    keyed = any(k for k, _ in rows)
    body = code(src, size, value_color, t2c={r"^[a-z_]+(?= )": AZURE} if keyed else None)
    return boxed(body, src, title, accent, width)


JSON_T2C = {r"\"[^\"\n]*\"(?=\s*:)": AZURE, r":\s*\"[^\"\n]*\"": MINT, r"\b-?\d+(\.\d+)?\b": SUN, r"\b(true|false|null)\b": ORCHID}


def json_text_card(src, title, accent, size=20, width=None):
    """Hand-formatted JSON (abridged where it says so) in kit json_card colours."""
    return boxed(code(src, size, INK, t2c=JSON_T2C), src, title, accent, width)


def call_card(src, size=20, title="tool_use"):
    """A tool call written as the call it stands for, e.g. search(pattern="Thanks") (SUN)."""
    return boxed(code(src, size, INK, t2c=PY_T2C), src, title, SUN)


def tool_tag(name, pic, size=24, width=None, height=0.9):
    words = code(name, size, INK, weight=m.BOLD)
    body = m.VGroup(icon(pic, 0.52), words).arrange(m.RIGHT, buff=0.26)
    box = card(width or body.width + 0.7, height, SUN, SUN_FILL, radius=0.2, stroke=2.5)
    return m.VGroup(box, body.move_to(box))


def row_card(left, right, color, fill=CARD, width=None, height=0.62, gap=0.35):
    w = width or left.width + right.width + gap + 0.6
    box = card(w, height, color, fill, radius=0.14, stroke=2)
    left.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    right.move_to(box).align_to(box, m.RIGHT).shift(0.3 * m.LEFT)
    return m.VGroup(box, left, right)


def quote_card(lines, source, width=None, size=24, color=INK, accent=EDGE):
    words = m.VGroup(*[text(s, size, color) for s in lines]).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
    src = cite(source)
    inner = m.VGroup(words, src).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
    box = card(max(width or 0, inner.width + 0.7), inner.height + 0.5, accent, CARD, radius=0.18, stroke=2)
    bar = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=box.height - 0.3, stroke_width=0,
                             fill_color=accent if accent != EDGE else MUTED, fill_opacity=1)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.38 * m.RIGHT)
    bar.move_to(box).align_to(box, m.LEFT).shift(0.16 * m.RIGHT)
    return m.VGroup(box, bar, words, src)


def glyph_box(ch, label, color, fill, size=64):
    g = code(ch, size, color, weight=m.BOLD)
    box = card(1.35, 1.35, color, fill, radius=0.2, stroke=2.5)
    g.move_to(box)
    cap = code(label, 20, MUTED).next_to(box, m.DOWN, buff=0.15)
    return m.VGroup(box, g, cap)


# ---------------------------------------------------------------------------- the episode


class Ep15DesigningGoodTools(Episode):
    number = 15

    def lesson(self):
        self.hook()
        self.goals()
        self.general()
        self.descriptions()
        self.fidelity()
        self.formats()
        self.errors()
        self.reading()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes="When an agent keeps failing, look at its tools before its model. Li, AI Agents in Depth, ch. 1: "
            "'Many problems that appear to require a smarter model are really interface problems.' Otto's job "
            f"(illustrative): {TASK} It needs a tool to find the line and one to change it. The failing trace "
            "foreshadows section 4 (the quote loop).",
        )
        bot = otto(1.2).move_to([-5.6, 0.55, 0])
        rows = m.VGroup(*[
            row_card(m.VGroup(code(f"{i}", 20, MUTED), code("edit_file", 20, SUN, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3),
                     code(EDIT_RESULTS[i - 1], 20, ROSE), ROSE, width=6.1, height=0.66)
            for i in range(1, 4)
        ]).arrange(m.DOWN, buff=0.22).move_to([-1.0, 0.55, 0])
        names = m.VGroup(*[r[1][1] for r in rows])
        person = icon("user", 0.9).move_to([5.55, -0.75, 0])
        doubt = text("Do we need a\nsmarter model?", 26, INK)
        thought = bubble(doubt, AZURE, AZURE_FILL, tail="right")
        thought.next_to(person, m.UP, buff=0.25).align_to([6.6, 0, 0], m.RIGHT)
        fine = pill("the model is fine", MINT, MINT_FILL, 20).next_to(person, m.DOWN, buff=0.3)
        quote = quote_card(["“Many problems that appear to require a ‘smarter model’",
                            "are really interface problems.”"], "Li, AI Agents in Depth · ch. 1", size=24)
        quote.move_to([0, -2.55, 0])

        hero = otto(1.3).move_to([-3.6, 0.9, 0])
        user = icon("user", 0.95).move_to([5.6, -0.2, 0])
        ask = bubble(text(TASK, 26, INK), AZURE, AZURE_FILL, tail="right")
        ask.next_to(user, m.UP, buff=0.25).align_to([6.6, 0, 0], m.RIGHT)
        find = tool_tag("search", SEARCH_ICON, width=2.8)
        change = tool_tag("edit_file", "ep03-wrench", width=2.8)
        tags = m.VGroup(find, change).arrange(m.RIGHT, buff=0.5).move_to([-3.6, -1.6, 0])
        roles = m.VGroup(text("find the line", 22, MUTED).next_to(find, m.DOWN, buff=0.18),
                         text("change it", 22, MUTED).next_to(change, m.DOWN, buff=0.18))
        arms = m.VGroup(*[m.Line(hero[0].get_bottom() + 0.1 * m.DOWN, t.get_top() + 0.06 * m.UP) for t in (find, change)])
        arms.set_stroke(SUN, 3)

        q1 = title_text("What makes a tool easy", 60)
        q2 = title_text("for a model to use well?", 60, MINT)
        question = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.35).move_to([0, 0.1, 0])

        self.say(
            "[When an agent keeps failing at a task], [it's tempting to blame the model.]",
            m.AnimationGroup(self.new_header("15 · tool design", "Blame the tools first"),
                             m.FadeIn(bot, shift=0.2 * m.UP),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.RIGHT) for r in rows], lag_ratio=0.35)),
            m.AnimationGroup(m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(thought, shift=0.2 * m.UP)),
        )
        self.say(
            "[But often the model is fine], [and the trouble is the tools it was given.]",
            m.AnimationGroup(thought.animate.set_opacity(0.3), m.FadeIn(fine, shift=0.15 * m.UP)),
            m.AnimationGroup(m.LaggedStart(*[m.Circumscribe(n, color=SUN, buff=0.05) for n in names], lag_ratio=0.2),
                             m.FadeIn(quote, shift=0.2 * m.UP)),
        )
        self.say(
            "[Today, Otto has one small job]: [make tinyshop's receipt say thank you, not thanks.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bot, rows, person, thought, fine, quote)), m.FadeIn(hero, scale=0.85)),
            m.AnimationGroup(m.FadeIn(user, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP)),
        )
        self.say(
            "[It needs two tools]: [one to find the line], [and one to change it.]",
            m.Create(arms),
            m.AnimationGroup(m.FadeIn(find, shift=0.2 * m.UP), m.FadeIn(roles[0])),
            m.AnimationGroup(m.FadeIn(change, shift=0.2 * m.UP), m.FadeIn(roles[1])),
        )
        self.say(
            "[So what makes a tool easy for a model to use well?]",
            m.AnimationGroup(m.FadeOut(m.VGroup(hero, user, ask, tags, roles, arms)),
                             m.FadeIn(q1, shift=0.3 * m.UP), m.FadeIn(q2, shift=0.3 * m.UP)),
        )
        self.wait(0.6)
        self.play(m.FadeOut(question, shift=0.3 * m.UP), run_time=0.8)

    # ------------------------------------------------------------------ 1 · goals, not APIs

    def goals(self):
        self.next_section(
            "1 · Goals, not APIs",
            notes=f"API-shaped tools (illustrative): list_dir, open_file, read_chunk, close_file. To find every line "
            f"holding 'Thanks', the scene walks tinyshop's {len(REPO)} files: {N_API} calls (computed: 2 listings + "
            f"3 calls per file), each result added to the context. Li, ch. 4: ACI, 'a tool should correspond to the "
            f"Agent's goal, not to an underlying API operation'; early API wrappers made agents 'coordinate several "
            f"tools to accomplish one goal'. The goal-shaped search returns both places ({', '.join(WALK_HITS)}) in "
            "one call (computed).",
        )
        api = ["list_dir(path)", "open_file(path)", "read_chunk(handle)", "close_file(handle)"]
        chips_ = m.VGroup(*[chip(a, EDGE, CARD, width=3.6, height=0.58, size=24, mono=True) for a in api])
        chips_.arrange(m.DOWN, buff=0.16)
        p_cap = text("thin wrappers around\nthe operating system", 22, MUTED)
        p_inner = m.VGroup(p_cap, chips_).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        p_box = card(p_inner.width + 0.6, p_inner.height + 0.55, EDGE, PANEL, radius=0.2, stroke=2).move_to(p_inner)
        panel = m.VGroup(p_box, p_inner).move_to([-4.75, 0.45, 0])

        op_color = {"list": AZURE, "open": SUN, "read": SUN, "close": SUN}
        grid = m.VGroup()
        cells = []
        y0 = 2.3
        for r, (label, ops) in enumerate(WALK_ROWS):
            y = y0 - r * 0.5
            lab = code(label, 20, MUTED)
            lab.move_to([0, y, 0]).align_to([1.45, 0, 0], m.RIGHT)
            row = m.VGroup(lab)
            for j, op in enumerate(ops):
                c = chip(op, op_color[op], CARD, width=1.2, height=0.46, size=24, mono=True).move_to([2.35 + j * 1.35, y, 0])
                row.add(c)
                cells.append(c)
            grid.add(row)
        assert len(cells) == N_API
        count = title_text(f"{N_API} calls", 48, ROSE)
        meter = Meter(3.6, 0.0, SUN, 0.24)
        m_cap = code("context", 20, MUTED)
        tally = m.VGroup(count, m.VGroup(m_cap, meter).arrange(m.RIGHT, buff=0.25)).arrange(m.RIGHT, buff=0.6)
        tally.move_to([2.3, -1.4, 0])
        aci = quote_card(["“A tool should correspond to the agent’s goal,", "not to an underlying API operation.”"],
                         "Li, AI Agents in Depth · ch. 4 · the agent-computer interface (ACI)", size=24, accent=MINT)
        aci.move_to([0, 0, 0]).align_to([0, -3.58, 0], m.DOWN)
        goal_word = span(aci[2][0], aci[2][0].text, "goal")

        find = tool_tag("search", SEARCH_ICON, width=3.1).move_to([-4.9, 1.55, 0])
        sig = code("search(pattern)", 24, INK).next_to(find, m.DOWN, buff=0.3)
        good = pill("shaped to the goal", MINT, MINT_FILL, 22).next_to(sig, m.DOWN, buff=0.3)
        call = call_card('search(pattern="Thanks")').move_to([0, 1.55, 0]).align_to([-2.55, 0, 0], m.LEFT)
        a_call = arrow(find.get_right(), call.get_left(), SUN, buff=0.12)
        out = SEARCH_OUT.split("\n")
        res = field_card("tool_result", [("", o) for o in out], MINT, size=20, value_color=INK)
        res.move_to([0.25, -1.45, 0]).align_to([-6.7, 0, 0], m.LEFT)
        for o in out:
            loc = o.split(": ")[0]
            for g in span(res[1], res.src, loc):
                g.set_color(MINT)
        a_res = arrow(call.get_bottom(), [call.get_x(), res.get_top()[1], 0], MINT, buff=0.1)
        before = title_text(f"{N_API} calls", 40, ROSE)
        after = title_text("1 call", 40, MINT)
        versus = m.VGroup(before, after).arrange(m.RIGHT, buff=1.4).move_to([0, -3.0, 0])
        a_vs = arrow(before.get_right(), after.get_left(), MUTED, buff=0.25)

        self.say(
            "[Say Otto's file tools were thin wrappers around the operating system]: [list a folder, open a file, read a chunk, close it.]",
            m.AnimationGroup(self.new_header("01 · goals", "Shape tools to the goal"), m.FadeIn(p_box), m.FadeIn(p_cap)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in chips_], lag_ratio=0.3),
        )
        self.say(
            f"[To find every line that says thanks], [Otto must open every file in the repo]: [{say_number(N_API)} calls, and every result lands in its context.]",
            m.LaggedStart(*[m.FadeIn(row[0]) for row in grid], lag_ratio=0.15),
            m.LaggedStart(*[m.FadeIn(c, scale=0.8) for c in cells], lag_ratio=0.12),
            m.AnimationGroup(m.FadeIn(count, shift=0.15 * m.UP), m.FadeIn(m_cap), m.FadeIn(meter), meter.fill_to(0.62, run_time=1.6)),
        )
        self.say(
            "[The book AI Agents in Depth calls this the agent-computer interface]: [a tool should match the agent's goal, not an API operation.]",
            m.FadeIn(aci, shift=0.2 * m.UP),
            m.AnimationGroup(goal_word.animate.set_color(MINT), m.Circumscribe(goal_word, color=MINT, buff=0.05)),
        )
        self.wait(0.4)
        self.say(
            "[Otto's goal is to find text in code.] [So give it one tool for exactly that]: [search.]",
            m.FadeOut(m.VGroup(panel, grid, tally, aci)),
            m.FadeIn(find, scale=0.9),
            m.AnimationGroup(m.FadeIn(sig, shift=0.15 * m.UP), m.FadeIn(good, shift=0.15 * m.UP)),
        )
        self.say(
            "[One call], [and back come both places, with file names and line numbers.]",
            m.AnimationGroup(m.GrowArrow(a_call), m.FadeIn(call, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a_res), m.FadeIn(res, shift=0.2 * m.DOWN),
                             m.FadeIn(before), m.GrowArrow(a_vs), m.FadeIn(after, shift=0.15 * m.UP)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 2 · general or dedicated

    def general(self):
        self.next_section(
            "2 · General or dedicated",
            notes="Li, ch. 4: 'general tools are preferable to dedicated tools, unless there is a clear security, "
            "permission, or performance reason'; extract_pdf_text / extract_docx_content / extract_pptx_content → "
            "one read_document with a file_type parameter; 'a single Python interpreter replaces dozens of "
            "single-purpose tools'; 'each tool's definition occupies hundreds of tokens'. The four reasons for a "
            "dedicated tool, in Li's words: security, permissions and auditing; hiding platform differences and "
            "giving better feedback (his example: grep and find); extremely high usage frequency; complex parameter "
            "structure. The twelve small tools merged into the interpreter are illustrative. Permissions: Episode 17.",
        )
        three = ["extract_pdf_text", "extract_docx_content", "extract_pptx_content"]
        ex = m.VGroup(*[chip(n, EDGE, CARD, width=4.1, height=0.62, size=24, mono=True) for n in three])
        ex.arrange(m.DOWN, buff=0.22).move_to([-4.15, 1.25, 0])
        merged = chip("read_document(path, file_type)", MINT, MINT_FILL, width=5.2, height=0.72, size=24, mono=True)
        merged.move_to([-4.15, -1.2, 0])
        a_merge = arrow(ex.get_bottom(), [ex.get_x(), merged.get_top()[1], 0], MINT, buff=0.12)

        small = ["csv_sum", "json_get", "date_diff", "sort_rows", "unzip", "count_words",
                 "md_to_html", "dedupe", "resize_png", "parse_url", "mean", "regex_find"]
        smalls = m.VGroup(*[chip(n, EDGE, CARD, width=1.95, height=0.5, size=22, mono=True) for n in small])
        smalls.arrange_in_grid(rows=4, cols=3, buff=(0.15, 0.16)).move_to([3.75, 1.05, 0])
        interp = m.VGroup(card(4.4, 1.0, MINT, MINT_FILL, radius=0.2),
                          m.VGroup(icon("code", 0.55), code("run_python(code)", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3))
        interp[1].move_to(interp[0])
        interp.move_to([3.75, 1.05, 0])
        dozens = text("dozens of single-purpose tools", 22, MUTED).next_to(smalls, m.UP, buff=0.18)
        cost = pill("each schema: hundreds of tokens, every turn", SUN, SUN_FILL, 22)
        cost.move_to([0, -1.2, 0]).align_to([6.7, 0, 0], m.RIGHT)
        src1 = cite("Li, AI Agents in Depth · ch. 4").next_to(cost, m.DOWN, buff=0.25).align_to(cost, m.RIGHT)

        reasons = [("lock", "Security", "a gate and a log for risky actions"),
                   ("code", "Platform quirks", "hides OS quirks, clearer output"),
                   ("bolt", "Frequent use", "so common a tight schema pays off"),
                   ("gear", "Complex inputs", "too nested for a shell command")]
        tiles = m.VGroup(*[tile(i, t, d, EDGE, width=6.6, height=1.55) for i, t, d in reasons])
        tiles.arrange_in_grid(rows=2, cols=2, buff=(0.3, 0.3))
        for t in tiles:
            t[1][1][0].scale(0.9, about_edge=m.LEFT)
        tiles.move_to([0, 0.3, 0])
        prefer = pill("prefer general tools, unless…", MINT, MINT_FILL, 22).move_to([0, 2.55, 0])
        bot = otto(0.9).move_to([-5.9, -2.75, 0])
        find = tool_tag("search", SEARCH_ICON, size=22, width=2.6, height=0.78)
        change = tool_tag("edit_file", "ep03-wrench", size=22, width=2.6, height=0.78)
        mine = m.VGroup(find, change).arrange(m.RIGHT, buff=0.4).next_to(bot, m.RIGHT, buff=0.5)
        why = m.VGroup(text("platform quirks · frequent", 20, MUTED).next_to(find, m.DOWN, buff=0.15),
                       text("security · frequent", 20, MUTED).next_to(change, m.DOWN, buff=0.15))
        dedic = pill("both stay dedicated", MINT, MINT_FILL, 22).next_to(mine, m.RIGHT, buff=0.45)
        ep17 = cite("review and permissions: Episode 17").next_to(dedic, m.DOWN, buff=0.2).align_to(dedic, m.LEFT)

        def lit(*ts):
            return m.AnimationGroup(*[t[0].animate.set_stroke(MINT, 3) for t in ts])

        self.say(
            "[How many tools should an agent have?] [Three extractors, one per document type], [can merge into a single read document tool.]",
            self.new_header("02 · general or dedicated", "Fewer, more general tools"),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.RIGHT) for c in ex], lag_ratio=0.25),
            m.AnimationGroup(m.GrowArrow(a_merge), m.FadeIn(merged, shift=0.2 * m.DOWN), ex.animate.set_opacity(0.4)),
        )
        outs = [m.FadeOut(s, target_position=interp) for s in smalls] + [m.FadeOut(dozens)]
        for o in outs:  # their objects appear only inside this line: don't put them on screen at its start
            o.introducer = True
        self.say(
            "[One Python interpreter can stand in for dozens of single-purpose tools], [and each schema you drop saves hundreds of tokens.]",
            m.Succession(m.AnimationGroup(m.FadeIn(dozens), m.LaggedStart(*[m.FadeIn(s, scale=0.8) for s in smalls], lag_ratio=0.06)),
                         m.AnimationGroup(*outs, m.FadeIn(interp, scale=0.8))),
            m.AnimationGroup(m.FadeIn(cost, shift=0.15 * m.UP), m.FadeIn(src1)),
        )
        self.wait(0.3)
        self.say(
            "[So prefer general tools, unless a dedicated one pays for itself]: [security], [platform quirks], [very frequent use], [or complex inputs.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ex, merged, a_merge, interp, cost, src1)), m.FadeIn(prefer, shift=0.15 * m.DOWN)),
            *[m.FadeIn(t, shift=0.2 * m.UP) for t in tiles],
        )
        self.say(
            "[Search and edit are what Otto does most], [and edits need review, the subject of episode seventeen], [so both stay dedicated tools.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(mine, shift=0.2 * m.UP), m.FadeIn(why[0]), lit(tiles[1], tiles[2])),
            m.AnimationGroup(m.FadeIn(why[1]), lit(tiles[0]), m.FadeIn(ep17)),
            m.AnimationGroup(m.FadeIn(dedic, shift=0.15 * m.LEFT), m.Circumscribe(mine, color=MINT, buff=0.1)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 3 · descriptions

    def descriptions(self):
        self.next_section(
            "3 · Descriptions",
            notes="The model knows a tool only through its schema (Episode 3). Li, ch. 4: say when to use it; state "
            "its limits — 'the root cause of most tool call failures is not that the model doesn't know what the tool "
            "can do, but that it doesn't know what the tool cannot do'; concrete formats (E.164 phone numbers, "
            "RFC3339 timestamps); the return shape; the cost. Li, ch. 2 (experiment 2-4, the author's own Tau-bench "
            "ablation): removing all descriptive text raised the tool-call error rate by 45% (bars are relative: "
            "1.00 vs 1.45). The search description is illustrative; its 50-line cap is what search() above does.",
        )
        bad_src = f'{{\n  "name": "search",\n  "description": "{SEARCH_BAD}",\n  "input_schema": {{ … }}\n}}'
        bad = json_text_card(bad_src, "TOOLS · search (abridged)", SUN, size=22).move_to([-2.6, 0.55, 0])
        bad_desc = span(bad[1], bad_src, f'"{SEARCH_BAD}"')
        bot = otto(1.1).move_to([3.4, 0.9, 0])
        asks = ["a pattern, or plain words?", "which files?", "what comes back?"]
        qs = m.VGroup(*[pill(a, ROSE, ROSE_FILL, 20) for a in asks]).arrange(m.DOWN, buff=0.18, aligned_edge=m.LEFT)
        qs.next_to(bot, m.DOWN, buff=0.4)

        label_color = {"when": MINT, "cannot": ROSE, "format": AZURE, "returns": SUN, "cost": MUTED}
        d_cap = code("search · description", 20, MUTED)
        d_rows = m.VGroup()
        for kind, line in DESC:
            words = code(line, 20, INK)
            if kind == "format":
                paint(words, line, {r'"[^"]*"': MINT})
            d_rows.add(words)
        d_rows.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        labels = m.VGroup()
        for (kind, _), words in zip(DESC, d_rows):  # a label column, right-aligned against the text
            lab = code(kind, 20, label_color[kind], weight=m.BOLD).move_to(words)
            labels.add(lab.shift([words.get_left()[0] - 0.35 - lab.get_right()[0], 0, 0]))
        d_inner = m.VGroup(d_cap, m.VGroup(labels, d_rows)).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT)
        d_box = card(d_inner.width + 0.7, d_inner.height + 0.55, SUN, CARD, radius=0.18, stroke=2).move_to(d_inner)
        desc = m.VGroup(d_box, d_inner).move_to([0, 0.75, 0]).align_to([-6.7, 0, 0], m.LEFT)
        for lab, words, (kind, line) in zip(labels, d_rows, DESC):  # labels share their line's baseline
            lab.shift((baseline(words, line.replace(" ", "")) - baseline(lab, kind)) * m.UP)

        root = quote_card(["“The root cause of most", "tool call failures is … that", "it doesn’t know what the", "tool cannot do.”"],
                          "Li, AI Agents in Depth · ch. 4", size=22, accent=ROSE)
        root.move_to([0, 0, 0]).align_to([6.7, 2.62, 0], m.UR)
        e164 = m.VGroup(code("phone: E.164", 20, AZURE, weight=m.BOLD), code("e.g. +14155550123", 20, MINT),
                        code("no spaces or dashes", 20, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        e_box = card(e164.width + 0.6, e164.height + 0.45, AZURE, CARD, radius=0.16, stroke=2).move_to(e164)
        phone = m.VGroup(e_box, e164).next_to(root, m.DOWN, buff=0.25).align_to(root, m.LEFT)

        unit = 3.6
        y_a, y_b = -1.95, -2.65
        bar_l = -2.6
        l_a = text("with descriptions", 22, INK).move_to([0, y_a, 0]).align_to([bar_l - 0.3, 0, 0], m.RIGHT)
        l_b = text("names and types only", 22, INK).move_to([0, y_b, 0]).align_to([bar_l - 0.3, 0, 0], m.RIGHT)
        bars = []
        for y, f, c in ((y_a, 1.0, MUTED), (y_b, 1.0 + ABLATION, ROSE)):
            b = m.RoundedRectangle(corner_radius=0.08, width=unit * f, height=0.36, stroke_width=0, fill_color=c, fill_opacity=0.9)
            bars.append(b.move_to([bar_l + unit * f / 2, y, 0]))
        plus = title_text(f"+{round(ABLATION * 100)}%", 40, ROSE).next_to(bars[1], m.RIGHT, buff=0.3)
        err_cap = text("tool-call errors", 20, MUTED).next_to(bars[0], m.RIGHT, buff=0.3)
        abl_src = cite("Li, AI Agents in Depth · ch. 2 · the author's own Tau-bench test").move_to([0, -3.3, 0]).align_to([bar_l, 0, 0], m.LEFT)
        assert round(ABLATION * 100) == 45

        self.say(
            "[The model knows a tool only through its description], [and this one says just: search files.]",
            m.AnimationGroup(self.new_header("03 · descriptions", "Say when, and what not"), m.FadeIn(bad, shift=0.2 * m.UP),
                             m.FadeIn(bot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(bad_desc, color=ROSE, buff=0.05),
                             m.LaggedStart(*[m.FadeIn(q, shift=0.15 * m.UP) for q in qs], lag_ratio=0.3)),
        )
        self.say(
            "[A good description says when to use it]: [to find code before reading or editing it.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(bad, bot, qs)), m.FadeIn(d_box), m.FadeIn(d_cap),
                             m.FadeIn(labels[0]), m.FadeIn(d_rows[0])),
            m.AnimationGroup(m.FadeIn(labels[1]), m.AddTextLetterByLetter(d_rows[1], time_per_char=0.02)),
        )
        self.say(
            "[It says what the tool cannot do]: [no git folder, no binary files, no search by meaning.]",
            m.FadeIn(labels[2], shift=0.15 * m.RIGHT),
            m.AddTextLetterByLetter(d_rows[2], time_per_char=0.025),
        )
        self.say(
            "[That's the root cause of most failed calls]: [the model doesn't know what the tool cannot do.]",
            m.FadeIn(root, shift=0.2 * m.LEFT),
            m.Circumscribe(m.VGroup(labels[2], d_rows[2]), color=ROSE, buff=0.08),
        )
        self.say(
            "[It gives exact formats with an example], [like a phone number with its country code and no spaces], [and it says what comes back, and what it costs.]",
            m.AnimationGroup(m.FadeIn(labels[3]), m.FadeIn(d_rows[3], shift=0.15 * m.RIGHT)),
            m.FadeIn(phone, shift=0.2 * m.UP),
            m.AnimationGroup(m.LaggedStart(m.AnimationGroup(m.FadeIn(labels[4]), m.FadeIn(d_rows[4], shift=0.15 * m.RIGHT)),
                                           m.AnimationGroup(m.FadeIn(labels[5]), m.FadeIn(d_rows[5], shift=0.15 * m.RIGHT)), lag_ratio=0.5)),
        )
        self.say(
            f"[In the author's own experiment], [stripping the description text raised tool-call errors by {say_number(round(ABLATION * 100))} percent.]",
            m.AnimationGroup(m.FadeIn(l_a), m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(err_cap), m.FadeIn(abl_src)),
            m.AnimationGroup(m.FadeIn(l_b), m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(plus, shift=0.2 * m.LEFT)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 4 · fidelity

    def fidelity(self):
        self.next_section(
            "4 · Fidelity",
            notes=f"Run in the scene: harness_reference's edit_file (+ learn-claude-code s02's 'Error: text not found') "
            f"behind a harness that 'tidies' inputs (curly quotes → straight). Otto copies the line it read; what "
            f"arrives is not in the file; {ATTEMPTS} attempts, {ATTEMPTS} errors, the file unchanged (asserted); "
            "without the tidying, the same call succeeds on the first try and the curly quotes survive. Li, ch. 4 "
            "(fidelity of parameter passing): a version of Cursor from early 2026 silently converted curly quotes; "
            "the model 'tries repeatedly and fails repeatedly … it cannot understand why the tool can't find what it "
            "clearly saw'; 'there must be no systematic discrepancy between the world the model perceives and the "
            "world the tool operates on'; any normalisation must be documented in the description and reported in "
            "the result. The iteration cap is Episode 2's.",
        )
        wide = Listing(CHECKOUT.replace("Thanks", "Thank you"), "x").box.width  # room for the edited line
        file_ = Listing(CHECKOUT, "read_file · checkout.py", rows=5, width=wide)
        file_.move_to([0, 1.25, 0]).align_to([-6.7, 0, 0], m.LEFT)
        q_open, q_close = file_.part(NOTE_ROW, LQ), file_.part(NOTE_ROW, RQ)
        note_band = file_.band(NOTE_ROW, SUN)

        rows = [("path", "checkout.py"), ("old_text", OLD_TEXT), ("new_text", NEW_TEXT)]

        def edit_call():
            c = field_card("tool_use · edit_file", rows, SUN, size=20)
            c.move_to([0, -1.85, 0]).align_to([-6.7, 0, 0], m.LEFT)
            for nth in range(2):
                for g in m.VGroup(span(c[1], c.src, LQ, nth), span(c[1], c.src, RQ, nth)):
                    g.set_color(SUN)
            return c

        call = edit_call()
        curly = [span(call[1], call.src, q, nth) for nth in range(2) for q in (LQ, RQ)]
        straight = []
        for g in curly:
            s = code('"', 20, ROSE, weight=m.BOLD).set_z_index(2)
            straight.append(s.move_to(g))
        old_q = m.VGroup(curly[0], curly[1])

        z_file = glyph_box(LQ, "U+201C\nin the file", SUN, SUN_FILL)
        z_tool = glyph_box('"', "U+0022\nwhat arrives", ROSE, ROSE_FILL)
        neq = title_text("≠", 54, ROSE)
        zoom = m.VGroup(z_file, neq, z_tool).arrange(m.RIGHT, buff=0.45).move_to([3.95, 1.3, 0])
        neq.shift((z_file[0].get_y() - neq.get_y()) * m.UP)
        z_tool_first = m.VGroup(z_tool, neq)

        harness = m.VGroup(card(2.5, 1.25, MUTED, CARD, radius=0.2),
                           m.VGroup(icon(GEAR_ICON, 0.5), code("harness", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2))
        harness[1].move_to(harness[0]).shift(0.18 * m.UP)
        tidy_cap = code("tidies quotes", 20, ROSE).next_to(harness[1], m.DOWN, buff=0.12)
        harness.add(tidy_cap)
        harness.next_to(call, m.RIGHT, buff=0.38)
        a_in = arrow(call.get_right(), harness.get_left(), SUN, buff=0.1)
        err = field_card("tool_result", [("", EDIT_RESULTS[0])], ROSE, size=20)
        err.next_to(harness, m.RIGHT, buff=0.4)
        a_out = arrow(harness.get_right(), err.get_left(), ROSE, buff=0.1)
        gap_x = span(call[1], call.src, OLD_TEXT).get_x()
        probe = m.DashedLine([gap_x, call.get_top()[1] + 0.05, 0], [gap_x, file_.box.get_bottom()[1] - 0.05, 0],
                             dash_length=0.1).set_stroke(ROSE, 3)
        miss = text("✗ no match", 22, ROSE, weight=m.SEMIBOLD).next_to(probe, m.RIGHT, buff=0.2)

        # the loop
        ring = loop_ring(["read_file", "edit_file", "text not found"], radius=1.85, colors=[SUN, SUN, ROSE], size=24,
                         center=(2.4, 0.3, 0))
        nodes, arcs = ring
        bot = otto(0.95).move_to([2.4, 0.3, 0])
        attempt_cap = text("attempt", 24, MUTED)
        n_words = [title_text(str(i), 96, ROSE) for i in range(1, ATTEMPTS + 1)]
        att = m.VGroup(attempt_cap, n_words[0]).arrange(m.DOWN, buff=0.15).move_to([-4.0, 0.55, 0])
        for w in n_words[1:]:
            w.move_to(n_words[0])
        seen = code(f"… {LQ}tinyshop{RQ}!", 22, INK).next_to(nodes[0], m.LEFT, buff=0.3)
        for g in m.VGroup(span(seen, seen.text, LQ), span(seen, seen.text, RQ)):
            g.set_color(SUN)
        cap_note = m.VGroup(cite("stops only at the"), cite("iteration cap · Episode 2")).arrange(m.DOWN, buff=0.08)
        cap_note.next_to(att, m.DOWN, buff=0.35)
        story = quote_card(["“It cannot understand why the tool can’t find", "what it clearly saw.”"],
                           "Li, AI Agents in Depth · ch. 4 · a Cursor edit tool, early 2026", size=24, accent=ROSE)
        story.move_to([0, 0, 0]).align_to([0, -3.58, 0], m.DOWN)
        assert story.get_top()[1] < ring[1].get_bottom()[1] - 0.2  # the loop's lower arrow stays visible

        def flash(i, color=SUN):
            return m.ShowPassingFlash(arcs[i].copy().set_stroke(color, 9), time_width=0.6, run_time=0.38)

        def lap(k):  # one more attempt: read → edit → error, and the counter ticks
            out = m.FadeOut(n_words[k - 1], shift=0.3 * m.UP)
            out.introducer = True  # the digit it removes appeared earlier in this same line
            return m.Succession(flash(2, ROSE), flash(0), flash(1),
                                m.AnimationGroup(out, m.FadeIn(n_words[k], shift=0.3 * m.UP),
                                                 m.Circumscribe(nodes[2], color=ROSE, buff=0.06), run_time=0.5))

        # the fix
        file2 = Listing(CHECKOUT, "read_file · checkout.py", rows=5, width=wide).move_to(file_)
        call2 = edit_call()
        through = m.VGroup(card(2.5, 1.25, MINT, CARD, radius=0.2),
                           m.VGroup(icon(GEAR_ICON, 0.5), code("harness", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2))
        through[1].move_to(through[0]).shift(0.18 * m.UP)
        through.add(code("passes it as is", 20, MINT).next_to(through[1], m.DOWN, buff=0.12))
        through.move_to(harness)
        through[2].move_to([through[0].get_x(), through[2].get_y(), 0])
        ok = field_card("tool_result", [("", FIXED[-1])], MINT, size=20).move_to(err)
        b_in = arrow(call2.get_right(), through.get_left(), SUN, buff=0.1)
        b_out = arrow(through.get_right(), ok.get_left(), MINT, buff=0.1)
        old_line = file2.lines[NOTE_ROW]
        file2.body.remove(old_line)  # its own object: it fades out alone while the card stays
        new_line = file2.make_line(file2.src[NOTE_ROW].replace("Thanks", "Thank you"), NOTE_ROW, like=old_line[0])
        assert new_line.text == f'note = "{NEW_NOTE}"'
        for line in (new_line, old_line):
            for g in m.VGroup(span(line, line.text, LQ), span(line, line.text, RQ)):
                g.set_color(SUN)
        probe2 = m.DashedLine([gap_x, call2.get_top()[1] + 0.05, 0], [gap_x, file2.box.get_bottom()[1] - 0.05, 0],
                              dash_length=0.1).set_stroke(MINT, 3)
        hit = text("✓ exact match", 22, MINT, weight=m.SEMIBOLD).next_to(probe2, m.RIGHT, buff=0.2)
        for g in m.VGroup(q_open, q_close):
            g.set_color(SUN)
        band2 = file2.band(NOTE_ROW, MINT)
        rule = quote_card(["“There must be no systematic discrepancy", "between the world the model perceives",
                           "and the world the tool operates on.”"], "Li, AI Agents in Depth · ch. 4", size=22, accent=MINT)
        rule.move_to([3.95, 1.3, 0])
        rule_words = rule[2]

        self.say(
            "[Now the edit.] [Otto reads the checkout file], [and the receipt line comes back exactly, curly quotes and all.]",
            self.new_header("04 · fidelity", "Pass the text through untouched"),
            m.FadeIn(file_, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(note_band), m.Circumscribe(q_open, color=SUN, buff=0.04), m.Circumscribe(q_close, color=SUN, buff=0.04),
                             m.FadeIn(z_file, shift=0.2 * m.UP)),
        )
        self.say(
            "[Otto copies that text into edit file's old text], [character for character.]",
            m.FadeIn(call, shift=0.2 * m.UP),
            m.Circumscribe(old_q, color=SUN, buff=0.05),
        )
        self.say(
            "[But this harness quietly tidies its inputs], [and turns curly quotes into straight ones.]",
            m.AnimationGroup(m.GrowArrow(a_in), m.FadeIn(harness, shift=0.2 * m.LEFT)),
            m.AnimationGroup(pulse(a_in, SUN), *[m.Transform(g, s) for g, s in zip(curly, straight)],
                             m.FadeIn(z_tool_first, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[Now the old text isn't in the file], [and the tool says: text not found.]",
            m.AnimationGroup(m.Create(probe), m.FadeIn(miss, shift=0.15 * m.LEFT), note_band.animate.set_color(ROSE)),
            m.AnimationGroup(m.GrowArrow(a_out), m.FadeIn(err, shift=0.2 * m.RIGHT)),
        )
        self.wait(0.3)
        self.say(
            "[Otto reads the file again], [sees the very same line], [and tries again. And again.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(file_, note_band, probe, miss, call, harness, a_in, a_out, err, zoom)),
                             m.FadeIn(ring, scale=0.9), m.FadeIn(bot), m.FadeIn(att, shift=0.2 * m.UP)),
            m.AnimationGroup(flash(2, ROSE), m.FadeIn(seen, shift=0.15 * m.LEFT)),
            m.Succession(lap(1), lap(2), lap(3), m.FadeIn(cap_note)),
        )
        att.remove(n_words[0])  # faded out above: keep it from coming back in a later fade (gotcha 22)
        self.say(
            "[The book reports this in a version of Cursor from early twenty twenty-six]: [the model can't see why the tool won't find what it clearly saw.]",
            m.FadeIn(story, shift=0.2 * m.UP),
            m.Circumscribe(story[2], color=ROSE, buff=0.08),
        )
        self.wait(0.3)
        self.say(
            "[The fix: pass the model's text through untouched], [and the same call succeeds on the first try.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ring, bot, att, n_words[-1], seen, cap_note, story)),
                             m.FadeIn(file2), m.FadeIn(old_line), m.FadeIn(call2), m.FadeIn(through), m.GrowArrow(b_in)),
            m.AnimationGroup(pulse(b_in, SUN), m.Create(probe2), m.FadeIn(hit), m.GrowArrow(b_out), m.FadeIn(ok, shift=0.2 * m.RIGHT),
                             m.FadeIn(band2), m.FadeOut(old_line, shift=0.2 * m.UP), m.FadeIn(new_line, shift=0.2 * m.UP)),
        )
        self.say(
            "[The rule: no gap between the world the model sees], [and the world the tool acts on.]",
            m.FadeIn(rule, shift=0.2 * m.LEFT),
            m.AnimationGroup(m.Circumscribe(rule_words[1], color=MINT, buff=0.04), m.Circumscribe(rule_words[2], color=MINT, buff=0.04)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 5 · edit formats

    def formats(self):
        self.next_section(
            "5 · Edit formats",
            notes="Li, ch. 5 (file-editing tools in coding agents): diff + fast apply model (Cursor; may 'merge into the "
            "wrong place' when snippets look alike); old string → new string (Claude Code; must exist and be unique; "
            "'a single character deviation causes the match to fail'); line numbers ('each edit shifts the line "
            "numbers that follow it'); start + end anchors (the pair must be unique). Run in the scene: edit 1 inserts "
            f"'{INSERTED}' after line 1 (illustrative); edit 2, planned as 'line {NOTE_ROW + 1}' on the original, then "
            f"lands on '{LANDS_ON}'. The note text occurs exactly once (asserted).",
        )
        fmts = [("Diff + apply", "a fast model merges\na sketch of the change", "Cursor"),
                ("Old → new text", "exact text, found\nexactly once", "Claude Code, Otto"),
                ("Line numbers", "short to write:\n“replace line 4”", ""),
                ("Anchors", "a unique start and end,\nnew text between", "")]
        cards = m.VGroup()
        for title_, desc, who in fmts:
            body = m.VGroup(text(title_, 26, INK, weight=m.BOLD), text(desc, 20, MUTED))
            if who:
                body.add(code(who, 20, SUN))
            body.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
            box = card(3.2, 2.0, EDGE, CARD, radius=0.18, stroke=2)
            body.move_to(box).align_to(box, m.UL).shift(0.28 * m.RIGHT + 0.25 * m.DOWN)
            cards.add(m.VGroup(box, body))
        cards.arrange(m.RIGHT, buff=0.2).move_to([0, 1.5, 0])

        lst = Listing(CHECKOUT, "checkout.py", rows=6)
        lst.move_to([0, -1.75, 0]).align_to([-6.7, 0, 0], m.LEFT)
        target = lst.band(NOTE_ROW, SUN)
        plan = m.VGroup(
            m.VGroup(code("edit 1", 20, MINT, weight=m.BOLD), code("insert a line after line 1", 20, INK)).arrange(m.RIGHT, buff=0.3),
            m.VGroup(code("edit 2", 20, SUN, weight=m.BOLD), code(f"replace line {NOTE_ROW + 1} (the note)", 20, INK)).arrange(m.RIGHT, buff=0.3),
        ).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        plan.move_to([0, -0.85, 0]).align_to([lst.box.get_right()[0] + 0.4, 0, 0], m.LEFT)
        ins = lst.make_line(INSERTED, 1, like=lst.lines[0][0])
        moves = [line.animate.shift(lst.pitch * m.DOWN) for line in lst.lines[1:] if line is not None]
        wrong = lst.band(NOTE_ROW, ROSE)
        lands = m.VGroup(code("edit 2 lands on", 20, ROSE, weight=m.BOLD), code(LANDS_ON, 20, INK)).arrange(m.RIGHT, buff=0.3)
        lands.next_to(plan, m.DOWN, buff=0.45).align_to(plan, m.LEFT)
        ok_band = lst.band(NOTE_ROW + 1, MINT)
        once = pill("old_text found exactly once", MINT, MINT_FILL, 20).next_to(lands, m.DOWN, buff=0.35).align_to(plan, m.LEFT)
        typo = pill("one wrong character → no match", ROSE, ROSE_FILL, 20).next_to(once, m.DOWN, buff=0.2).align_to(plan, m.LEFT)

        self.say(
            "[Coding agents describe edits in a few ways]: [a diff merged by a fast apply model], [exact old and new text], [line numbers], [or start and end anchors.]",
            self.new_header("05 · edit formats", "How to say where to edit"),
            *[m.FadeIn(c, shift=0.2 * m.UP) for c in cards],
        )
        self.say(
            "[Line numbers are short], [but each edit shifts every line below it], [so the next edit can land on the wrong line.]",
            m.AnimationGroup(m.FadeIn(lst, shift=0.2 * m.UP), m.FadeIn(target), m.FadeIn(plan, shift=0.15 * m.LEFT),
                             m.Circumscribe(cards[2], color=SUN, buff=0.05)),
            m.AnimationGroup(*moves, m.FadeIn(ins, shift=0.15 * m.RIGHT), run_time=1.2),
            m.AnimationGroup(m.Transform(target, wrong), m.FadeIn(lands, shift=0.15 * m.UP)),
        )
        self.say(
            "[Exact text avoids that], [as long as it appears exactly once]: [one wrong character, and the match fails.]",
            m.AnimationGroup(m.Circumscribe(cards[1], color=MINT, buff=0.05), m.FadeIn(ok_band)),
            m.FadeIn(once, shift=0.15 * m.UP),
            m.FadeIn(typo, shift=0.15 * m.UP),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 6 · errors that teach

    def errors(self):
        self.next_section(
            "6 · Errors that teach",
            notes="The redesigned edit_file (harness_reference names; ≤ 10 lines) checks the match count. Its two error "
            "messages are computed by edit_error() above: the closest line by difflib, then the first character that "
            f"differs (line {NOTE_ROW + 1}, column {QUOTE_COL}: the curly quote); and 'appears 2 times (lines 1 and "
            "5)' for old_text='total'. learn-claude-code s02's edit_file returns 'Error: text not found'. Li, ch. 4: a "
            "rejected call should come back with 'an error message and correction suggestions'.",
        )
        code_card = code_block(NEW_EDIT_SRC, 20, title="edit_file · redesigned")
        code_card.move_to([0, 0, 0]).align_to([-6.7, 2.62, 0], m.UL)
        lines = NEW_EDIT_SRC.split("\n")

        def line_band(i, j, color):
            body = code_card[1]
            k0 = sum(sum(not c.isspace() for c in ln) for ln in lines[:i])
            k1 = sum(sum(not c.isspace() for c in ln) for ln in lines[:j + 1])
            seg = body[k0:k1]
            top, bot = seg.get_top()[1] + 0.08, seg.get_bottom()[1] - 0.08
            box = code_card[0]
            r = m.Rectangle(width=box.width - 0.2, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
            r.move_to([box.get_x(), (top + bot) / 2, 0])
            mk = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
            mk.move_to([box.get_left()[0] + 0.125, (top + bot) / 2, 0])
            return m.VGroup(r, mk).set_z_index(1)

        code_card[1].set_z_index(2)
        check_band = line_band(2, 4, MINT)

        old = field_card("tool_result · before", [("", EDIT_RESULTS[0])], ROSE, size=20)
        nothing = text("no line, no reason,\nnothing to try", 22, MUTED)
        before = m.VGroup(old, nothing).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
        before.move_to([0, code_card.get_y(), 0]).align_to([6.7, 0, 0], m.RIGHT)

        e1 = field_card("tool_result", [("", ln) for ln in ERR_QUOTE.split("\n")], ROSE, size=20)
        e2 = field_card("tool_result", [("", ln) for ln in ERR_TWICE.split("\n")], ROSE, size=20)
        w = max(e1.width, e2.width)
        e1 = field_card("tool_result", [("", ln) for ln in ERR_QUOTE.split("\n")], ROSE, size=20, width=w)
        e2 = field_card("tool_result", [("", ln) for ln in ERR_TWICE.split("\n")], ROSE, size=20, width=w)
        errs = m.VGroup(e1, e2).arrange(m.RIGHT, buff=0.35).move_to([0, 0, 0]).align_to([0, -3.55, 0], m.DOWN)
        assert code_card.get_bottom()[1] - errs.get_top()[1] > 0.25
        e1_line = span(e1[1], e1.src, f"line {NOTE_ROW + 1}, column {QUOTE_COL}")
        e1_char = m.VGroup(span(e1[1], e1.src, LQ), span(e1[1], e1.src, 'has "'))
        e1_char[0].set_color(SUN)
        e1_try = span(e1[1], e1.src, 'old_text has "')
        e2_where = span(e2[1], e2.src, "(lines 1 and 5)")
        e2_fix = span(e2[1], e2.src, "Add a\nline around it to match only once.")

        self.say(
            "[When a match fails, the error is the model's only clue], [and text not found gives it nothing to act on.]",
            m.AnimationGroup(self.new_header("06 · errors", "Errors that teach"), m.FadeIn(code_card, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(old, shift=0.2 * m.LEFT), m.FadeIn(nothing)),
        )
        self.say(
            "[A teaching error names the closest line], [and the one character that differs]: [here, the quote the harness changed.]",
            m.AnimationGroup(m.FadeIn(check_band), m.FadeIn(e1, shift=0.2 * m.UP), m.Circumscribe(e1_line, color=SUN, buff=0.04)),
            m.Circumscribe(e1_char, color=SUN, buff=0.05),
            m.Circumscribe(e1_try, color=ROSE, buff=0.05),
        )
        self.say(
            "[And if the text appears twice], [it says where], [and asks for a line around it, so it matches once.]",
            m.FadeIn(e2, shift=0.2 * m.UP),
            m.Circumscribe(e2_where, color=SUN, buff=0.05),
            m.Circumscribe(e2_fix, color=MINT, buff=0.05),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 7 · honest reads

    def reading(self):
        self.next_section(
            "7 · Honest reads",
            notes=f"Episode 3 cut big command outputs honestly (head, tail, how much was left out, where it was saved). "
            f"Reads need the same: read_file with offset and limit. Run in the scene on an illustrative {TOTAL_LINES}-line "
            f"orders.csv: limit 200 returns {len(PAGE1)} lines and '{NOTE1}' (Li, ch. 4, perception tools, word for "
            f"word); the next page says '{NOTE2}'. The page is {PAGE_FRAC:.0%} of the file (bar drawn to scale). Li: "
            "'Silent truncation is dangerous'.",
        )
        top, bottom = 2.25, -3.25
        bar = m.RoundedRectangle(corner_radius=0.08, width=0.5, height=top - bottom, stroke_color=EDGE, stroke_width=2,
                                 fill_color=CARD, fill_opacity=1).move_to([-6.1, (top + bottom) / 2, 0])
        b_cap = code("orders.csv", 20, MUTED).next_to(bar, m.UP, buff=0.12).align_to(bar, m.LEFT)
        b_top = code("1", 20, FAINT).next_to(bar, m.RIGHT, buff=0.15).align_to(bar, m.UP)
        b_bot = code(f"{TOTAL_LINES}", 20, FAINT).next_to(bar, m.RIGHT, buff=0.15).align_to(bar, m.DOWN)
        win_h = PAGE_FRAC * (top - bottom)

        def window(offset):
            y = top - (offset / TOTAL_LINES) * (top - bottom) - win_h / 2
            return m.Rectangle(width=0.5, height=win_h, stroke_width=0, fill_color=MINT, fill_opacity=1).move_to([bar.get_x(), y, 0])

        win = window(0)
        call = call_card('read_file(path="orders.csv", offset=0, limit=200)')
        call.move_to([0, 1.65, 0]).align_to([-4.9, 0, 0], m.LEFT)
        call2 = call_card('read_file(path="orders.csv", offset=200, limit=200)').move_to(call).align_to(call, m.LEFT)

        def page(rows, note):
            first, second = note.split("; ")
            src_rows = [("", r) for r in rows[:3]] + [("", "…")]
            c = field_card("tool_result", src_rows + [("", first + ";"), ("", second)], MINT, size=20)
            n_rows = len(src_rows)
            body_lines = c.src.split("\n")
            k = sum(sum(not ch.isspace() for ch in ln) for ln in body_lines[:n_rows])
            c[1][k:].set_color(SUN)
            return c, c[1][k:]

        res, foot = page(PAGE1, NOTE1)
        res.move_to([0, 0, 0]).align_to([-4.9, 0.55, 0], m.UL)
        res2, foot2 = page(PAGE2, NOTE2)
        res2.move_to(res).align_to(res, m.LEFT)
        a_res = arrow(call.get_bottom(), [call.get_x(), res.get_top()[1], 0], MINT, buff=0.1)
        ep3 = pill("Episode 3 · never cut silently", MUTED, CARD_HI, 20).move_to([0, -2.6, 0]).align_to([6.7, 0, 0], m.RIGHT)
        src_li = cite("Li, AI Agents in Depth · ch. 4").next_to(res, m.DOWN, buff=0.15).align_to(res, m.LEFT)
        s_words = text("cut silently, the model\nreasons as if it saw\nthe whole file", 22, INK)
        s_box = card(s_words.width + 0.6, s_words.height + 0.45, ROSE, ROSE_FILL, radius=0.18, stroke=2).move_to(s_words)
        silent = m.VGroup(s_box, s_words).move_to([0, -0.7, 0]).align_to([6.7, 0, 0], m.RIGHT)

        self.say(
            "[Episode three cut big command outputs honestly], [and reads need the same care.]",
            m.AnimationGroup(self.new_header("07 · honest reads", "Say what you left out"), m.FadeIn(ep3, shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(bar, m.UP), m.FadeIn(b_cap), m.FadeIn(b_top), m.FadeIn(b_bot)),
        )
        self.say(
            f"[Ask for a {spell(TOTAL_LINES)}-line file], [and read file returns {spell(len(PAGE1))} lines], [and says so]: "
            f"[lines one to {spell(len(PAGE1))} of {spell(TOTAL_LINES)}, use the offset to read on.]",
            m.FadeIn(call, shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.FadeIn(win), m.GrowArrow(a_res), m.FadeIn(res[0]), m.FadeIn(res[2]), m.FadeIn(res[1][:len(res[1]) - len(foot)])),
            m.AnimationGroup(m.FadeIn(foot, shift=0.1 * m.UP), m.FadeIn(src_li)),
            m.Circumscribe(foot, color=SUN, buff=0.06),
        )
        self.say(
            "[So Otto knows to ask for the next page.] [Cut silently], [and the model would reason as if it had read everything.]",
            m.AnimationGroup(m.FadeOut(call), m.FadeIn(call2), m.FadeOut(res), m.FadeIn(res2), m.Transform(win, window(200))),
            m.FadeIn(silent, shift=0.15 * m.UP),
            m.Circumscribe(silent, color=ROSE, buff=0.06),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section(
            "Recap",
            notes="Otto's tools, redesigned: search (shaped to the goal, described with its limits), edit_file (exact, "
            "unique, untouched text, errors that teach), read_file (pages that say what they left out). Goals · Limits · "
            "Fidelity. Li, ch. 4: 'check the tool descriptions first' rather than doubting the model. Next: MCP (Episode 16).",
        )
        hero = otto(1.4).move_to([0, 1.1, 0])
        tools_ = [("search", SEARCH_ICON, "shaped to the goal,\nits limits described"),
                  ("edit_file", "ep03-wrench", "exact text, untouched,\nerrors that teach"),
                  ("read_file", DOC_ICON, "pages that say\nwhat they left out")]
        cols = m.VGroup()
        for name, pic, sub in tools_:
            tag = tool_tag(name, pic, size=24, width=3.3)
            cols.add(m.VGroup(tag, text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.22))
        cols.arrange(m.RIGHT, buff=0.75, aligned_edge=m.UP).move_to([0, -1.55, 0])
        arms = m.VGroup(*[m.Line(hero[0].get_bottom() + 0.08 * m.DOWN, c[0].get_top() + 0.06 * m.UP) for c in cols]).set_stroke(SUN, 3)

        halo = glow(MINT, radius=4.0)
        words = on_baseline(m.VGroup(*[title_text(w, 66, c) for w, c in (("Goals", MINT), ("Limits", SUN), ("Fidelity", CORAL))]))
        words.arrange(m.RIGHT, buff=0.95)
        on_baseline(words)
        mid = words[0].get_y()
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots).move_to(0.4 * m.UP)
        tagline = text("Before you blame the model, check its tools.", 30, MUTED).next_to(row, m.DOWN, buff=0.75)

        self.say(
            "[Otto's tools, redesigned]: [a search shaped to its goal], [with a description that states its limits.]",
            m.AnimationGroup(self.new_header("recap", "Otto's tools, redesigned"), m.FadeIn(hero, scale=0.85)),
            m.AnimationGroup(m.Create(arms[0]), m.FadeIn(cols[0][0], shift=0.2 * m.DOWN)),
            m.FadeIn(cols[0][1], shift=0.15 * m.UP),
        )
        self.say(
            "[Edits that keep every character and explain their errors], [and reads that say what they left out.]",
            m.AnimationGroup(m.Create(arms[1]), m.FadeIn(cols[1], shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Create(arms[2]), m.FadeIn(cols[2], shift=0.2 * m.DOWN)),
        )
        self.say(
            "[To sum up]: [goals], [limits], [fidelity.]",
            m.AnimationGroup(self.drop_header(), m.FadeOut(m.VGroup(hero, cols, arms), shift=0.2 * m.DOWN), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[Before you blame the model, check its tools.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)


def spell(n):
    """A whole number as the narrator should say it (below a million)."""
    if n < 100:
        return say_number(n)
    if n < 1000:
        h, r = divmod(n, 100)
        return f"{say_number(h)} hundred" + (f" and {say_number(r)}" if r else "")
    t, r = divmod(n, 1000)
    return f"{spell(t)} thousand" + ((" and " if r < 100 else " ") + spell(r) if r else "")


assert spell(TOTAL_LINES) == "five thousand" and spell(200) == "two hundred"
