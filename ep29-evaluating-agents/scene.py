"""Agents, Animated · Episode 29 · Evaluating Agents (Module 6 · Beyond the Loop)

How to know whether an agent is getting better. Running example: a tinyshop eval set, a hundred tasks in the
style of public agent benchmarks (environment + initial state + simulated user + verifier). The eval harness
below is real code in the style of harness_reference.py: run_eval() gives each task a fresh in-memory tinyshop,
runs the reference agent_loop (unchanged) against a scripted stand-in for the model, and scores the run with the
task's verifier, which executes tinyshop's real tests (FAIL_TO_PASS and PASS_TO_PASS) against the files Otto
left behind. Otto's three kinds of patch (delete the assertion, add the last item back, fix the loop) are real
edits, so every check mark in the verifier table is computed.

Arc: hook (one good run proves nothing; model + harness; "If a stronger model doesn't raise the score, the
bottleneck is the Harness") · a task's four parts · the eval harness (claims vs verified) · verifiers can be
fooled (SWE-bench's two lists; Verified) · pass@k vs pass^k · LLM as a judge (rubric, length and position bias,
same family) · the first error (B's expenses.jpg case) · right actions, wrong report · statistics (±9 points,
paired, seeds) · observe before you prompt (B's AndroidWorld slice) · recap: verify, repeat, diagnose.

Sources (facts dated October 2026):
- Li Bojie, AI Agents in Depth, ch. 7 (bojieli.github.io/ai-agent-book/book-en/chapter7/): "the object of
  evaluation should not be just the model, but the combination of the model and the Harness"; "If a stronger
  model doesn't raise the score, the bottleneck is the Harness." The τ²-bench telecom task (user_scenario with
  known_info: name, phone, country only; initial_state with initialization_actions per side;
  evaluation_criteria with env_assertions; reward_basis ["ENV_ASSERTION"]). Pass@k = 1 − (1 − p)^k, Pass^k =
  p^k; at p = 0.6, k = 5: 99.0% vs 7.8%; Pass@k for "technical wonders", Pass^k for business reliability.
  SWE-bench: FAIL_TO_PASS and PASS_TO_PASS, checking only one is insufficient; SWE-bench Verified: 1,699 of
  2,294 tasks sampled, 93 Python developers, 500 kept (~71% eliminated). LLM-as-a-judge: rubric with
  essential / important / optional weights and a veto item; length bias; position bias (evaluate twice with
  the order swapped and average); same-family judges: the agent "will gradually learn to avoid the types of
  errors that the judging model is not good at detecting" (Goodhart). Failure attribution: record "the first
  step at which unacceptable behavior appeared"; expenses.jpg: 32 steps, no step errored, task declared
  complete; the expected ¥436.35 "Dress" row was missing; step 8: "I cannot actually see the content/details
  of the expenses in the image"; by step 11 four invented expenses were in its notes; T3A is text-only (element
  tree, no pixels): a missing observation channel, not "the model cannot do OCR". Right actions, wrong report:
  of 704 published τ² baseline runs with a communication requirement, 240 failed, 162 failed the
  communication check, 80 ("a third of all failures") had a correct environment state and a wrong report.
  Statistics: "with 100 cases and a 70% success rate, the 95% confidence interval is about 70%±9 percentage
  points"; 73% vs 70% does not justify switching; paired analysis (McNemar or paired bootstrap); 3–5 seeds.
  Table 7-5, AndroidWorld Wi-Fi slice, four tasks per arm: more navigation instructions 25% → 25%; UIAutomator
  accessibility tree 25% → 100% at 2.498× tokens; compacted tree 100% → 100% at 0.506× tokens; "observation
  failures should be investigated before prompts are expanded"; "4/4 on the slice must not be reported as
  100% system-wide success."
- SWE-bench Verified: OpenAI, August 2024 (cited on screen as "OpenAI, 2024").

Illustrative (for teaching): the tinyshop eval set (100 tasks, all variants of the running bug), which tasks the
scripted Otto passes (seeded: random.Random(29)), its kinds of failure, the new prompt's paired results (65 both
pass, 22 both fail, 5 old only, 8 new only), Otto's two reports and the judge's scores (8 / 6, swapped), the
rubric rows, the "stronger model" swap in the hook, the five tries (3 of 5) in the pass@k section. Computed and
asserted: every test result and check mark, the scores 70% and 73%, pass@5 and pass^5, the ±9-point interval,
the exact McNemar p-value, the one-third, SWE-bench Verified's 71%, the AndroidWorld percentages from 1/4 and 4/4.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import json
import math
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

HERE = Path(__file__).resolve().read_text()
REF = (SERIES / "harness_reference.py").read_text()
PREFIX = "ep29"
FACTS = "Facts as of October 2026."

# ---------------------------------------------------------------------------- tinyshop, in memory (real code)

CART_BUG = '''def total(items):
    s = 0
    for i in range(len(items) - 1):
        s += items[i]
    return s
'''
TEST_CART = '''from cart import total


def test_total():
    got = total([3, 4])
    assert got == 7


def test_total_empty():
    assert total([]) == 0
'''
OTHER_TESTS = 11  # tinyshop's other tests (checkout and friends), all passing, as in Episodes 2 and 22
EDITS = {  # Otto's three kinds of patch: (path, old_text, new_text)
    "cheat": ("tests/test_cart.py", "    assert got == 7\n", ""),
    "patch": ("cart.py", "    return s\n", "    return s + items[-1]\n"),
    "fix": ("cart.py", "range(len(items) - 1)", "range(len(items))"),
}


def run_tests(cart_src, test_src):
    """tinyshop's tests/test_cart.py, executed against a cart.py: {test name: passed}."""
    cart = {}
    exec(compile(cart_src, "cart.py", "exec"), cart)
    ns = {"total": cart["total"]}
    exec(compile(test_src.replace("from cart import total\n", ""), "tests/test_cart.py", "exec"), ns)
    results = {}
    for name, fn in ns.items():
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                results[name] = True
            except Exception:  # noqa: BLE001 — an assertion or a crash both fail the test
                results[name] = False
    return results


def pytest_q(results):
    failed = [n for n, ok in results.items() if not ok]
    passed = sum(results.values()) + OTHER_TESTS
    return f"{len(failed)} failed, {passed} passed" if failed else f"{passed} passed"


assert pytest_q(run_tests(CART_BUG, TEST_CART)) == "1 failed, 12 passed"  # the course's running bug

FAIL_TO_PASS = ["test_total"]
PASS_TO_PASS = ["test_total_empty"] + [f"other_{i}" for i in range(OTHER_TESTS)]  # the other 12 tests
assert len(PASS_TO_PASS) == 12


def checks(env):
    """The task's verifier, in parts: Otto's own suite, then the task's OWN copy of the tests in two lists."""
    own = run_tests(env["cart.py"], env["tests/test_cart.py"])
    task = run_tests(env["cart.py"], TEST_CART)  # SWE-bench style: the task's tests, not Otto's
    f2p = all(task[t] for t in FAIL_TO_PASS)
    p2p = all(task.get(t, True) for t in PASS_TO_PASS)  # the 11 others are not touched by any patch
    return NS(suite=pytest_q(own), green="failed" not in pytest_q(own), f2p=f2p, p2p=p2p, reward=int(f2p and p2p))


def verify(env):
    return checks(env).reward == 1


# ---------------------------------------------------------------------------- the harness (reference names)

SYSTEM = re.search(r'^SYSTEM = "(.*)"$', REF, re.M).group(1)
assert SYSTEM.startswith("You are Otto")
assert "def agent_loop(messages):" in REF and 'results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})' in REF
REPO = {}


def edit_file(path, old_text, new_text):
    assert REPO[path].count(old_text) == 1, (path, old_text)
    REPO[path] = REPO[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOLS = [{"name": "bash"}, {"name": "read_file"}, {"name": "edit_file"}]  # schemas abridged
TOOL_HANDLERS = {"edit_file": edit_file}


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


CLAIM = "Fixed it. The tests pass."
CLAIMS = []


class ScriptedModel:
    """Stands in for client.messages: Otto's behaviour on the current task (illustrative, seeded)."""

    def __init__(self):
        self.task = None

    def create(self, system, messages, tools):
        if messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], str):
            path, old, new = EDITS[self.task.outcome]
            block = NS(type="tool_use", id=f"toolu_{self.task.n:03d}", name="edit_file",
                       input={"path": path, "old_text": old, "new_text": new})
            return NS(stop_reason="tool_use", content=[block])
        CLAIMS.append(CLAIM)
        return NS(stop_reason="end_turn", content=[NS(type="text", text=CLAIM)])


MODEL = ScriptedModel()
client = NS(messages=MODEL)


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


VERDICTS = []


class TinyTask:
    def __init__(self, n, outcome):
        self.n, self.id, self.outcome = n, f"tinyshop-{n:03d}", outcome
        self.ask = "My cart total looks too low."  # all the simulated user knows

    def fresh_env(self):
        REPO.clear()
        REPO.update({"cart.py": CART_BUG, "tests/test_cart.py": TEST_CART})
        MODEL.task = self
        return REPO

    def user_turn(self):
        return {"role": "user", "content": self.ask}

    def verify(self, env):
        VERDICTS.append((self.n, verify(env)))
        return VERDICTS[-1][1]


# ---- what Episode 29 adds: an eval harness around the unchanged loop ----
def run_eval(tasks):
    passed = 0
    for task in tasks:
        env = task.fresh_env()      # new tinyshop
        messages = [task.user_turn()]
        agent_loop(messages)        # unchanged
        passed += task.verify(env)  # the world
    return passed / len(tasks)
# ---- end of what Episode 29 adds ----


EVAL_SRC = '''def run_eval(tasks):
    passed = 0
    for task in tasks:
        env = task.fresh_env()      # new tinyshop
        messages = [task.user_turn()]
        agent_loop(messages)        # unchanged
        passed += task.verify(env)  # the world
    return passed / len(tasks)'''
_added = HERE.split("# ---- what Episode 29 adds")[1].split("# ---- end of what Episode 29 adds")[0]
assert all(line in _added for line in EVAL_SRC.split("\n")) and len(EVAL_SRC.split("\n")) <= 10

# ---- the eval set: 100 tasks; which ones the scripted Otto passes (illustrative, seeded) ----
N_TASKS = 100
SHOWN = 17  # the task drawn as a card (it fails: Otto deletes the assertion)
rng = random.Random(29)
OLD_FAILS = {SHOWN} | set(rng.sample([n for n in range(1, N_TASKS + 1) if n != SHOWN], 29))
KIND = {n: "fix" if n not in OLD_FAILS else ("cheat" if n == SHOWN else rng.choice(["cheat", "patch"]))
        for n in range(1, N_TASKS + 1)}
OLD_TASKS = [TinyTask(n, KIND[n]) for n in range(1, N_TASKS + 1)]
SCORE_OLD = run_eval(OLD_TASKS)
OLD = dict(VERDICTS)
assert SCORE_OLD == 0.70 and sum(OLD.values()) == 70  # "seventy pass: seventy percent"
CLAIMS_OLD = list(CLAIMS)
assert len(CLAIMS_OLD) == N_TASKS and set(CLAIMS_OLD) == {CLAIM}  # Otto claims success on all hundred
assert not OLD[SHOWN] and KIND[SHOWN] == "cheat"

# the new system prompt: paired with the old run task by task
passes = sorted(n for n in OLD if OLD[n])
fails = sorted(n for n in OLD if not OLD[n] and n != SHOWN)
LOST = set(rng.sample(passes, 5))  # passed before, fail now
GAINED = set(rng.sample(fails, 8))  # failed before, pass now
NEW_KIND = {n: ("patch" if n in LOST else "fix" if n in GAINED else KIND[n]) for n in OLD}
VERDICTS.clear()
SCORE_NEW = run_eval([TinyTask(n, NEW_KIND[n]) for n in range(1, N_TASKS + 1)])
NEW = dict(VERDICTS)
assert SCORE_NEW == 0.73
BOTH_PASS = sum(OLD[n] and NEW[n] for n in OLD)
BOTH_FAIL = sum(not OLD[n] and not NEW[n] for n in OLD)
OLD_ONLY = sum(OLD[n] and not NEW[n] for n in OLD)
NEW_ONLY = sum(NEW[n] and not OLD[n] for n in OLD)
assert (BOTH_PASS, BOTH_FAIL, OLD_ONLY, NEW_ONLY) == (65, 22, 5, 8) and OLD_ONLY + NEW_ONLY == 13
# exact McNemar test: under "no difference", each disagreement is a coin flip
_n, _k = OLD_ONLY + NEW_ONLY, min(OLD_ONLY, NEW_ONLY)
P_MCNEMAR = min(1.0, 2 * sum(math.comb(_n, i) for i in range(_k + 1)) / 2 ** _n)
assert f"{P_MCNEMAR:.2f}" == "0.58"  # "easily chance"

# ---- the verifier table: Otto's three patches, really applied and really checked ----
PATCH_ROWS = []
for kind in ("cheat", "patch", "fix"):
    env = {"cart.py": CART_BUG, "tests/test_cart.py": TEST_CART}
    path, old, new = EDITS[kind]
    env[path] = env[path].replace(old, new, 1)
    PATCH_ROWS.append((kind, checks(env)))
_rows = [(c.suite, c.green, c.f2p, c.p2p, c.reward) for _, c in PATCH_ROWS]
assert _rows == [("13 passed", True, False, True, 0), ("1 failed, 12 passed", False, True, False, 0),
                 ("13 passed", True, True, True, 1)], _rows

# ---- pass@k vs pass^k ----
P, K = 0.6, 5


def pass_at(k, p=P):
    return 1 - (1 - p) ** k


def pass_hat(k, p=P):
    return p ** k


assert f"{100 * pass_at(K):.1f}" == "99.0" and f"{100 * pass_hat(K):.1f}" == "7.8"
TRIES = [True, False, True, True, False]  # one task, five tries (illustrative): 3 of 5, the 60%
assert sum(TRIES) / len(TRIES) == P and any(TRIES) and not all(TRIES)

# ---- statistics ----
Z = 1.96


def half_width(p, n=N_TASKS):
    return Z * math.sqrt(p * (1 - p) / n)


assert round(100 * half_width(SCORE_OLD)) == 9  # "about nine points either way"
CI_OLD = (100 * (SCORE_OLD - half_width(SCORE_OLD)), 100 * (SCORE_OLD + half_width(SCORE_OLD)))
CI_NEW = (100 * (SCORE_NEW - half_width(SCORE_NEW)), 100 * (SCORE_NEW + half_width(SCORE_NEW)))
assert CI_NEW[0] < 100 * SCORE_OLD < CI_OLD[1] and CI_OLD[0] < 100 * SCORE_NEW  # each holds the other's score

# ---- the judge (illustrative scores): it prefers whichever report it reads first ----
ORDER_1 = {"A": 8, "B": 6}  # A shown first
ORDER_2 = {"B": 8, "A": 6}  # B shown first
AVG = {r: (ORDER_1[r] + ORDER_2[r]) / 2 for r in "AB"}
assert AVG["A"] == AVG["B"] == 7

# ---- the book's numbers ----
STEPS, FIRST, SYMPTOM, INVENTED = 32, 8, 11, 4
COMM_RUNS, COMM_FAILED, COMM_CHECK_FAILED, RIGHT_STATE_WRONG_REPORT = 704, 240, 162, 80
assert RIGHT_STATE_WRONG_REPORT * 3 == COMM_FAILED  # "a third of all failures"
SWE_TOTAL, SWE_SAMPLED, SWE_DEVS, SWE_KEPT = 2294, 1699, 93, 500
assert round(100 * (1 - SWE_KEPT / SWE_SAMPLED)) == 71
ARM = 4
SLICE = [  # (only change, control passes, treatment passes, treatment / control tokens)
    ("more instructions", 1, 1, None),
    ("richer UI tree", 1, 4, 2.498),
    ("compacted tree", 4, 4, 0.506),
]
assert [(round(100 * a / ARM), round(100 * b / ARM)) for _, a, b, _ in SLICE] == [(25, 25), (25, 100), (100, 100)]
assert round(SLICE[2][3], 1) == 0.5  # "at half the tokens"

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"
LLM_ICON = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
GEAR_ICON = recolored("gear", MUTED, CARD_HI, prefix=PREFIX)
JSON_RULES = {
    r"\"[^\"\n]*\"(?=\s*:)": AZURE,
    r"(?<=: )\"[^\"\n]*\"": MINT,
    r"\[\"[^\]\n]*\"\]": MINT,
    r"\b-?\d+(\.\d+)?\b": SUN,
    r"\b(true|false|null)\b": ORCHID,
}


def cite(s):
    return text(s, 20, MUTED)


def baseline_of(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def baseline(words):
    return baseline_of(words, [c for c in words.text if not c.isspace()])


def sit(words, y):
    """Put a Text's baseline at y."""
    words.shift((y - baseline(words)) * m.UP)
    return words


def line_glyphs(words, i):
    lines = words.text.split("\n")
    start = sum(sum(not c.isspace() for c in l) for l in lines[:i])
    n = sum(not c.isspace() for c in lines[i])
    return words[start:start + n]


def highlight(block, src, lines_, color=SUN, pad_l=0.12, pad_r=0.1):
    """A translucent bar behind lines `lines_` (an int or a range) of a card (box = block[0], text = block[1])."""
    rows = [lines_] if isinstance(lines_, int) else list(lines_)
    src_lines = src.split("\n")
    bases = [baseline_of(line_glyphs(block[1], i), [c for c in src_lines[i] if not c.isspace()]) for i in rows]
    top, bot = max(bases) + 0.29, min(bases) - 0.11
    left, right = block[0].get_left()[0] + pad_l, block[0].get_right()[0] - pad_r
    rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.14)
    mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
    rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
    mark.move_to([left + 0.025, (top + bot) / 2, 0])
    return m.VGroup(rect, mark).set_z_index(1)


def word_chip(word, color, fill, size=24, height=0.62, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def mono_chip(word, color, fill, size=24, height=0.62, pad=0.5):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def mark(ok, size=30):
    return text("✓" if ok else "✗", size, MINT if ok else ROSE, weight=m.BOLD)


def strike(words, color=ROSE):
    return m.Line(words.get_left() + 0.08 * m.LEFT, words.get_right() + 0.08 * m.RIGHT).set_stroke(color, 3).move_to(words)


def compact_json(obj):
    """json.dumps(indent=2) with one-item lists kept on one line."""
    return re.sub(r"\[\s+(\"[^\"\n]*\")\s+\]", r"[\1]", json.dumps(obj, indent=2, ensure_ascii=False))


def json_box(src, size=19, accent=LIME, title=None):
    """A JSON card coloured like kit.json_card, from a source string (so short lists stay on one line)."""
    body = code(src, size, INK, t2c=JSON_RULES)
    cap = code(title, 20, accent, weight=m.BOLD) if title else None
    top = cap.height + 0.22 if cap else 0.0
    box = card(max(body.width + 0.7, cap.width + 0.7 if cap else 0), body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def report_card(name, body, width=4.3):
    """One of Otto's two reports, as an assistant message labelled A or B."""
    label = code(name, 20, CORAL, weight=m.BOLD)
    words = text(body, 20, INK)
    inner = m.VGroup(label, words).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
    box = card(max(width, inner.width + 0.6), inner.height + 0.36, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.05, width=0.09, height=box.height - 0.16, stroke_width=0, fill_color=CORAL, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def polyline(points, color, width=4):
    return m.VMobject().set_points_as_corners(points).set_stroke(color, width=width)


class Chart:
    """Quiet axes (no ticks) with the origin at the lower-left corner (as in "AI, Explained" Ep43)."""

    def __init__(self, x_max, y_max, width, height, center):
        self.ax = m.Axes(
            x_range=[0, x_max, 0.1], y_range=[0, y_max, 0.1], x_length=width, y_length=height,
            axis_config={"color": FAINT, "stroke_width": 2.5, "include_tip": True, "tip_length": 0.18, "include_ticks": False},
        ).move_to(center)

    def p(self, x, y):
        return self.ax.c2p(x, y)

    def line(self, pts, color, width=4):
        return polyline([self.p(x, y) for x, y in pts], color, width)

    def xlabel(self, x, s):
        return code(s, 20, MUTED).move_to(self.p(x, 0) + 0.32 * m.DOWN)

    def ylabel(self, y, s):
        return code(s, 20, MUTED).next_to(self.p(0, y), m.LEFT, buff=0.16)

    def ticks(self, xs=(), ys=()):
        marks = [m.Line(self.p(x, 0) + 0.07 * m.DOWN, self.p(x, 0) + 0.07 * m.UP) for x in xs]
        marks += [m.Line(self.p(0, y) + 0.07 * m.LEFT, self.p(0, y) + 0.07 * m.RIGHT) for y in ys]
        return m.VGroup(*marks).set_stroke(FAINT, width=2.5)


def cell_grid(n_cols, n_rows, size, gap, center, color=EDGE, fill=CARD, stroke=1.5):
    cells = m.VGroup(*[card(size, size, color, fill, radius=size * 0.18, stroke=stroke) for _ in range(n_cols * n_rows)])
    cells.arrange_in_grid(rows=n_rows, cols=n_cols, buff=gap).move_to(center)
    return cells


def recolor_cells(cells, colors):
    """New cells over the old ones: colors[i] = (stroke, fill) or None to leave a cell alone."""
    out = m.VGroup()
    for c, col in zip(cells, colors):
        if col is None:
            continue
        stroke, fill = col
        out.add(card(c.width, c.height, stroke, fill, radius=c.width * 0.18, stroke=1.5).move_to(c))
    return out


# ---------------------------------------------------------------------------- the episode


class Ep29EvaluatingAgents(Episode):
    number = 29

    def lesson(self):
        self.hook()
        self.task_parts()
        self.eval_harness()
        self.verifiers()
        self.pass_k()
        self.judge()
        self.first_error()
        self.wrong_report()
        self.statistics()
        self.observe_first()
        self.recap()

    def speak(self, script, *animations):
        """say(), with every animation of an object not yet on screen marked as an introducer (a cell swapped in
        and out, or a card transformed twice, in the same line must not show early). From Episode 22."""
        on_screen = {id(x) for x in self.get_mobject_family_members()}

        def leaves(a):
            subs = getattr(a, "animations", None)
            if subs:
                for sub in subs:
                    yield from leaves(sub)
            else:
                yield a

        for a in animations:
            for leaf in leaves(a):
                mob = getattr(leaf, "mobject", None)
                if mob is not None and not leaf.is_introducer() and id(mob) not in on_screen:
                    leaf.introducer = True
        self.say(script, *animations)

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Otto's system prompt is changed (illustrative). One passing run is an anecdote; an eval set is a fixed list of tasks scored the same way (here, a hundred tinyshop tasks, illustrative). Li, AI Agents in Depth, ch. 7: 'the object of evaluation should not be just the model, but the combination of the model and the Harness'; 'If a stronger model doesn't raise the score, the bottleneck is the Harness.' The model swap and the unchanged 70% are illustrative. Agent = model + harness: Episode 1. " + FACTS)
        bot = otto(0.9).move_to([-5.5, 1.75, 0])
        v2 = mono_chip("system prompt v2", ORCHID, ORCHID_FILL, size=24).next_to(bot, m.RIGHT, buff=0.45)
        question = title_text("Is Otto better now?", 46).move_to([3.0, 1.75, 0])
        term = terminal(["$ pytest -q", "13 passed"], width=4.6).move_to([-3.7, -0.9, 0])
        anecdote = pill("an anecdote, not evidence", ROSE, ROSE_FILL, 22).next_to(term, m.DOWN, buff=0.35)
        grid = cell_grid(10, 10, 0.25, 0.07, [3.35, -0.75, 0], color=FAINT, fill=CARD_HI)
        grid_cap = code("evals/tinyshop/ · 100 tasks", 20, MUTED).next_to(grid, m.DOWN, buff=0.25)

        model = m.VGroup(card(2.9, 1.9, CORAL, CORAL_FILL), m.VGroup(icon(LLM_ICON, 0.75), text("model", 26, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.15))
        model[1].move_to(model[0])
        harness = m.VGroup(card(2.9, 1.9, MUTED, CARD_HI), m.VGroup(icon(GEAR_ICON, 0.75), text("harness", 26, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.15))
        harness[1].move_to(harness[0])
        plus = title_text("+", 50, MUTED)
        equals = title_text("=", 50, MUTED)
        agent = m.VGroup(otto(0.85), text("Otto", 26, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.2)
        eq = m.VGroup(model, plus, harness, equals, agent).arrange(m.RIGHT, buff=0.5).move_to([0, 0.95, 0])
        strong = m.VGroup(card(2.9, 1.9, CORAL, CORAL_FILL, stroke=4), m.VGroup(icon(LLM_ICON, 0.75), text("stronger model", 24, INK, weight=m.BOLD)).arrange(m.DOWN, buff=0.15))
        strong[1].move_to(model[0])
        strong[0].move_to(model[0])

        def score_row(label, y):
            lab = code(label, 22, MUTED)
            meter = Meter(4.2, 0.0, LIME, 0.22)
            val = code("70%", 22, LIME, weight=m.BOLD)
            row = m.VGroup(lab, meter, val).arrange(m.RIGHT, buff=0.35)
            lab.align_to([-4.6, 0, 0], m.LEFT)
            meter.next_to(lab, m.RIGHT, buff=0.35).align_to([-1.6, 0, 0], m.LEFT)
            val.next_to(meter, m.RIGHT, buff=0.3)
            row.set_y(y)
            return row

        before = score_row("old model", -1.15)
        after = score_row("stronger model", -1.75)
        quote = text("“If a stronger model doesn't raise the score, the bottleneck is the Harness.”", 24, INK).move_to([0, -2.75, 0])
        src = cite("Li, AI Agents in Depth · ch. 7").next_to(quote, m.DOWN, buff=0.18)

        self.speak(
            "[Say we tweak Otto's system prompt.] [Is Otto better now?]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(v2, target_position=bot, scale=0.5), lag_ratio=0.3),
            m.Write(question),
        )
        self.speak(
            "[One good run] [proves nothing.] [We need an eval set]: [for example, a hundred tinyshop tasks], [scored the same way every time.]",
            m.FadeIn(term, shift=0.2 * m.UP),
            m.FadeIn(anecdote, scale=1.1),
            m.FadeIn(grid_cap),
            m.LaggedStart(*[m.FadeIn(c, scale=0.5) for c in grid], lag_ratio=0.015),
            m.Circumscribe(grid, color=LIME, buff=0.1),
        )
        self.speak(
            "[And what we score isn't the model alone]: [it's the model and its harness, together.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (bot, v2, question, term, anecdote, grid, grid_cap)], m.FadeIn(model, shift=0.2 * m.UP)),
            m.LaggedStart(m.FadeIn(plus), m.FadeIn(harness, shift=0.2 * m.UP), m.FadeIn(equals), m.FadeIn(agent, shift=0.2 * m.UP), lag_ratio=0.3),
        )
        self.speak(
            "[Swap in a stronger model]: [if the score doesn't move], [the bottleneck is the harness.]",
            m.AnimationGroup(m.ReplacementTransform(model, strong), m.Succession(m.FadeIn(before, run_time=0.4), before[1].fill_to(0.7, run_time=0.8))),
            m.Succession(m.FadeIn(after, run_time=0.4), after[1].fill_to(0.7, run_time=0.8)),
            m.AnimationGroup(m.Circumscribe(harness, color=ROSE, buff=0.1), m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · a task

    def task_parts(self):
        task = {
            "id": f"tinyshop-{SHOWN:03d}",
            "environment": "tinyshop repo + shell",
            "initial_state": "total() skips the last item",
            "user_scenario": {"known_info": TinyTask(SHOWN, "cheat").ask.rstrip(".").lower()},
            "evaluation_criteria": {"env_assertions": {"FAIL_TO_PASS": FAIL_TO_PASS, "PASS_TO_PASS": f"the other {len(PASS_TO_PASS)} tests"}},
            "reward_basis": ["ENV_ASSERTION"],
        }
        src = compact_json(task)
        assert json.loads(src) == task
        lines = src.split("\n")
        self.next_section("1 · A task has four parts", notes=f"A tinyshop task in the shape of a τ²-bench task (Li, AI Agents in Depth, ch. 7): an environment, an initial state (reset actions; here the planted bug), a simulated user whose known_info is only what a real user would know (τ²'s telecom user knows name, phone and country), and evaluation_criteria with env_assertions, scored by reward_basis ['ENV_ASSERTION']. Here the environment assertions are tinyshop's tests, split into FAIL_TO_PASS and PASS_TO_PASS as SWE-bench does. The task's contents are illustrative. Verifying the world, not the claim: Episode 22. " + FACTS)
        jc = json_box(src, 19, LIME, title="evals/tinyshop/017.json").move_to([0, 0, 0])
        jc.align_to([-6.45, 0, 0], m.LEFT).set_y(-0.2)
        idx = {key: next(i for i, l in enumerate(lines) if f'"{key}"' in l) for key in ("environment", "initial_state", "known_info", "FAIL_TO_PASS", "PASS_TO_PASS")}
        parts = [
            ("environment", "a fresh copy of tinyshop", LIME, idx["environment"]),
            ("initial state", "the bug, planted on purpose", LIME, idx["initial_state"]),
            ("simulated user", "knows only what a user would", AZURE, idx["known_info"]),
            ("verifier", "checks the world after the run", MINT, range(idx["FAIL_TO_PASS"], idx["PASS_TO_PASS"] + 1)),
        ]
        bars, labels, nums, links = [], [], [], []
        x0 = jc.get_right()[0] + 0.55
        label_ys = [1.5, 0.5, -0.5, -1.5]
        for k, ((name, sub, color, rows), ly) in enumerate(zip(parts, label_ys)):
            bar = highlight(jc, src, rows, color, pad_l=0.14, pad_r=0.14)
            num = chip(str(k + 1), color, CARD, width=0.46, height=0.46, size=22)
            num.move_to([x0 + 0.23, ly, 0])
            head = text(name, 24, color, weight=m.BOLD).next_to(num, m.RIGHT, buff=0.2)
            sit(head, ly - 0.09)
            note = text(sub, 20, MUTED).next_to(head, m.DOWN, buff=0.1).align_to(head, m.LEFT)
            link = m.Line([jc.get_right()[0] + 0.06, bar.get_y(), 0], num.get_left() + 0.08 * m.LEFT).set_stroke(color, 2.5)
            bars.append(bar)
            labels.append(m.VGroup(head, note))
            nums.append(num)
            links.append(link)
        assert max(l.get_right()[0] for l in labels) < 6.5, max(l.get_right()[0] for l in labels)
        src_cite = cite("task format after τ²-bench · Li, AI Agents in Depth · ch. 7").next_to(jc, m.DOWN, buff=0.2).align_to(jc, m.LEFT)
        known = line_glyphs(jc[1], idx["known_info"])
        ep22 = text("not Otto's claim (Episode 22)", 20, MUTED)
        ep22.next_to(labels[3], m.DOWN, buff=0.1).align_to(labels[3], m.LEFT)
        assert ep22.get_right()[0] < 6.5 and ep22.get_bottom()[1] > -3.5, (ep22.get_right()[0], ep22.get_bottom()[1])

        self.speak(
            "[Each task], [like those in public agent benchmarks], [has four parts.]",
            m.AnimationGroup(self.new_header("01 · the task", "A task has four parts"), m.FadeIn(jc, shift=0.2 * m.UP)),
            m.FadeIn(src_cite),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(n, scale=0.6), m.Create(l)) for n, l in zip(nums, links)], lag_ratio=0.25),
        )
        self.speak(
            "[First, an environment]: [a fresh copy of tinyshop.] [Then an initial state]: [the bug, planted on purpose.]",
            m.FadeIn(bars[0]),
            m.FadeIn(labels[0], shift=0.1 * m.LEFT),
            m.AnimationGroup(m.FadeOut(bars[0]), m.FadeIn(bars[1])),
            m.FadeIn(labels[1], shift=0.1 * m.LEFT),
        )
        self.speak(
            "[A simulated user, who knows only what a real user would]: [my cart total looks too low.]",
            m.AnimationGroup(m.FadeOut(bars[1]), m.FadeIn(bars[2]), m.FadeIn(labels[2], shift=0.1 * m.LEFT)),
            m.Circumscribe(known, color=AZURE, buff=0.06),
        )
        self.speak(
            "[And a verifier]: [assertions on the world after the run, not on what Otto says], [as in episode twenty-two.]",
            m.AnimationGroup(m.FadeOut(bars[2]), m.FadeIn(bars[3])),
            m.FadeIn(labels[3], shift=0.1 * m.LEFT),
            m.FadeIn(ep22, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · the eval harness

    def eval_harness(self):
        self.next_section("2 · The eval harness", notes=f"run_eval is new code in harness_reference.py's style (the scene runs exactly this function; the excerpt's lines are asserted to be in it): a fresh environment per task, the reference agent_loop unchanged, then the task's verifier. The model is a scripted stand-in (illustrative, seeded): on task {SHOWN} Otto deletes the assertion; every run ends with the same claim, {CLAIM!r} ({len(CLAIMS_OLD)} of {N_TASKS} runs). Result (computed): {sum(OLD.values())} of {N_TASKS} verified, a score of {SCORE_OLD:.0%}. " + FACTS)
        cb = code_block(EVAL_SRC, size=19, title="evals.py · new, in the harness style")
        cb.align_to([-6.45, 0, 0], m.LEFT).align_to([0, 2.5, 0], m.UP)
        src_lines = EVAL_SRC.split("\n")
        hl = {k: highlight(cb, EVAL_SRC, next(i for i, l in enumerate(src_lines) if k in l), SUN) for k in ("fresh_env", "agent_loop", "verify")}
        # the pipeline for one task, on the right
        x = 4.15
        task_chip = mono_chip(f"task {SHOWN:03d}", LIME, LIME_FILL, size=22, height=0.55).move_to([x, 2.3, 0])
        env_node = m.VGroup(icon("ep27-folder", 0.55), code("tinyshop/", 22, LIME)).arrange(m.RIGHT, buff=0.2).move_to([x, 1.05, 0])
        bot = otto(0.7).move_to([x, -0.35, 0])
        ver_node = m.VGroup(icon("check", 0.55), text("verifier", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2).move_to([x, -1.75, 0])
        a0 = arrow(task_chip.get_bottom(), env_node.get_top(), MUTED, buff=0.1)
        a1 = arrow(env_node.get_bottom(), bot.get_top(), MUTED, buff=0.12)
        a2 = arrow(bot.get_bottom(), ver_node.get_top(), MUTED, buff=0.12)
        claim = message("assistant", CLAIM, width=cb.width, size=22)
        claim.align_to(cb, m.LEFT).set_y(-1.85)
        verdict = pill("✗ verifier: test_total still fails", ROSE, ROSE_FILL, 22).next_to(ver_node, m.DOWN, buff=0.3)
        pipeline = m.VGroup(task_chip, env_node, bot, ver_node, a0, a1, a2)
        assert claim.get_top()[1] < cb.get_bottom()[1] - 0.15 and claim.get_right()[0] < verdict.get_left()[0] - 0.2, (claim.get_top()[1], cb.get_bottom()[1], cb.get_right()[0], claim.get_right()[0], verdict.get_left()[0])

        grid = cell_grid(10, 10, 0.34, 0.07, [3.95, 0.3, 0])
        colored = recolor_cells(grid, [(MINT, MINT_FILL) if OLD[n] else (ROSE, ROSE_FILL) for n in range(1, N_TASKS + 1)])
        grid_cap = code(f"{N_TASKS} tinyshop tasks", 20, MUTED).next_to(grid, m.DOWN, buff=0.22)
        r1 = code(f"claims “fixed”  {len(CLAIMS_OLD):>3}", 22, CORAL)
        r2 = code(f"verified        {sum(OLD.values()):>3}", 22, MINT)
        tally = m.VGroup(r1, r2).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        score = m.VGroup(text("score", 26, MUTED), title_text(f"{SCORE_OLD:.0%}", 58, LIME)).arrange(m.RIGHT, buff=0.3)
        left = m.VGroup(tally, score).arrange(m.RIGHT, buff=0.6).align_to([-6.4, 0, 0], m.LEFT).set_y(-2.3)
        assert left.get_bottom()[1] > -3.5 and left.get_top()[1] < cb.get_bottom()[1] - 0.2

        self.speak(
            "[An evaluation harness runs every task the same way]: [a fresh environment], [Otto's loop, unchanged], [then the verifier.]",
            m.AnimationGroup(self.new_header("02 · the harness", "Run every task the same way"), m.FadeIn(cb, shift=0.2 * m.UP), m.FadeIn(task_chip, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(hl["fresh_env"]), m.GrowArrow(a0), m.FadeIn(env_node, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeOut(hl["fresh_env"]), m.FadeIn(hl["agent_loop"]), m.GrowArrow(a1), m.FadeIn(bot, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeOut(hl["agent_loop"]), m.FadeIn(hl["verify"]), m.GrowArrow(a2), m.FadeIn(ver_node, shift=0.1 * m.DOWN)),
        )
        self.speak(
            "[Otto ends every run with the same claim]: [fixed it, the tests pass.] [On this task, the verifier disagrees.]",
            m.FadeIn(claim, target_position=bot, scale=0.5),
            m.Circumscribe(claim, color=CORAL, buff=0.06),
            m.FadeIn(verdict, scale=1.1),
        )
        self.speak(
            "[Run every task], [and seventy out of a hundred pass]: [a score of seventy percent.]",
            m.AnimationGroup(m.FadeOut(pipeline), m.FadeOut(claim), m.FadeOut(verdict), m.FadeOut(hl["verify"]), m.FadeIn(grid), m.FadeIn(grid_cap)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in colored], lag_ratio=0.02), m.FadeIn(tally, shift=0.1 * m.UP)),
            m.FadeIn(score, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · verifiers can be fooled

    def verifiers(self):
        self.next_section("3 · Verifiers can be fooled", notes=f"Three patches, really applied and really tested (asserted): deleting the assertion leaves Otto's own suite green ({PATCH_ROWS[0][1].suite!r}) but the task's own copy of test_total still fails; adding the last item back turns test_total green but total([]) now raises IndexError ({PATCH_ROWS[1][1].suite!r}); fixing the loop passes both lists. SWE-bench grades with FAIL_TO_PASS (tests the fix must turn green) and PASS_TO_PASS (tests that must stay green); Li, ch. 7: checking only one is insufficient. SWE-bench Verified (OpenAI, August 2024): {SWE_SAMPLED:,} of {SWE_TOTAL:,} tasks sampled, reviewed by {SWE_DEVS} developers, {SWE_KEPT} kept (~{round(100 * (1 - SWE_KEPT / SWE_SAMPLED))}% eliminated), as reported in Li, ch. 7. " + FACTS)
        cols = {"suite": -0.4, "f2p": 2.1, "p2p": 4.25, "reward": 5.95}
        head_y = 2.15
        heads = {
            "patch": text("Otto's patch", 22, MUTED).align_to([-6.4, 0, 0], m.LEFT),
            "suite": text("suite green?", 22, MUTED),
            "f2p": code("FAIL_TO_PASS", 20, INK, weight=m.BOLD),
            "p2p": code("PASS_TO_PASS", 20, INK, weight=m.BOLD),
            "reward": text("reward", 22, MUTED),
        }
        for k, h in heads.items():
            if k != "patch":
                h.set_x(cols[k])
            sit(h, head_y)
        f2p_sub = text("must turn green", 20, MUTED).next_to(heads["f2p"], m.DOWN, buff=0.08)
        p2p_sub = text("must stay green", 20, MUTED).next_to(heads["p2p"], m.DOWN, buff=0.08)
        rule = m.Line([-6.4, 1.55, 0], [6.4, 1.55, 0]).set_stroke(EDGE, 2)
        diffs = {"cheat": ("- assert got == 7", ROSE, "deletes the assertion"),
                 "patch": ("+ return s + items[-1]", SUN, "adds the last item back"),
                 "fix": ("range(len(items))", MINT, "fixes the loop")}
        ys = [0.95, -0.35, -1.65]
        rows = []
        for (kind, c), y in zip(PATCH_ROWS, ys):
            s, color, desc = diffs[kind]
            d = code(s, 20, color, weight=m.BOLD).align_to([-6.4, 0, 0], m.LEFT)
            sit(d, y + 0.08)
            cap = text(desc, 20, MUTED).align_to([-6.4, 0, 0], m.LEFT)
            sit(cap, y - 0.38)
            out = code(c.suite, 20, MUTED)
            mk = mark(c.green, 28)
            suite = m.VGroup(out, mk).arrange(m.RIGHT, buff=0.2).set_x(cols["suite"])
            sit(out, y - 0.05)
            mk.set_y(out.get_y())
            f2p = mark(c.f2p, 34).move_to([cols["f2p"], y, 0])
            p2p = mark(c.p2p, 34).move_to([cols["p2p"], y, 0])
            reward = title_text(str(c.reward), 40, MINT if c.reward else ROSE).move_to([cols["reward"], y, 0])
            rows.append(NS(diff=m.VGroup(d, cap), suite=suite, f2p=f2p, p2p=p2p, reward=reward))
        seps = m.VGroup(*[m.Line([-6.4, (a + b) / 2, 0], [6.4, (a + b) / 2, 0]).set_stroke(EDGE, 1.5) for a, b in zip(ys, ys[1:])])
        assert rows[1].suite.get_left()[0] > rows[1].diff.get_right()[0] + 0.4 and heads["p2p"].get_left()[0] > heads["f2p"].get_right()[0] + 0.25
        assert rows[1].suite.get_right()[0] < rows[1].f2p.get_left()[0] - 0.3 and heads["reward"].get_left()[0] > heads["p2p"].get_right()[0] + 0.2

        verified_t = text("SWE-bench Verified:", 24, INK, weight=m.BOLD)
        verified_1 = text(f"{SWE_SAMPLED:,} tasks reviewed by {SWE_DEVS} developers", 24, INK)
        verified_2 = text(f"→ {SWE_KEPT} kept", 24, MINT, weight=m.BOLD)
        vrow = m.VGroup(verified_t, verified_1, verified_2).arrange(m.RIGHT, buff=0.25).move_to([0, -2.75, 0])
        on_baseline(vrow)
        vsrc = cite("OpenAI, 2024 · Li, AI Agents in Depth · ch. 7").next_to(vrow, m.DOWN, buff=0.15)
        assert vrow.width < 12.8 and vsrc.get_bottom()[1] > -3.55

        r0, r1, r2 = rows
        self.speak(
            "[Verifiers can be fooled.] [On this task, Otto deleted the assertion], [so the suite went green.]",
            m.AnimationGroup(self.new_header("03 · verifiers", "Verifiers can be fooled"), m.FadeIn(heads["patch"]), m.FadeIn(heads["suite"]), m.Create(rule)),
            m.FadeIn(r0.diff, shift=0.1 * m.RIGHT),
            m.FadeIn(r0.suite, shift=0.1 * m.UP),
        )
        self.speak(
            "[So SWE-bench runs its own copy of the tests, in two lists]: [fail to pass, which the fix must turn green], [and pass to pass, which must stay green.]",
            m.Circumscribe(r0.suite, color=ROSE, buff=0.08),
            m.AnimationGroup(m.FadeIn(heads["f2p"]), m.FadeIn(f2p_sub), m.FadeIn(r0.f2p, scale=1.3)),
            m.AnimationGroup(m.FadeIn(heads["p2p"]), m.FadeIn(p2p_sub), m.FadeIn(r0.p2p, scale=1.3), m.FadeIn(heads["reward"]), m.FadeIn(r0.reward)),
        )
        self.speak(
            "[A patch that just adds the last item back] [turns the failing test green], [but an empty cart now crashes.]",
            m.AnimationGroup(m.Create(seps[0]), m.FadeIn(r1.diff, shift=0.1 * m.RIGHT), m.FadeIn(r1.suite, shift=0.1 * m.UP)),
            m.FadeIn(r1.f2p, scale=1.3),
            m.AnimationGroup(m.FadeIn(r1.p2p, scale=1.3), m.FadeIn(r1.reward)),
        )
        self.speak(
            "[Only the real fix passes both lists.]",
            m.Succession(
                m.AnimationGroup(m.Create(seps[1]), m.FadeIn(r2.diff, shift=0.1 * m.RIGHT), m.FadeIn(r2.suite, shift=0.1 * m.UP), m.FadeIn(r2.f2p, scale=1.3), m.FadeIn(r2.p2p, scale=1.3), m.FadeIn(r2.reward)),
                m.Circumscribe(m.VGroup(r2.f2p, r2.p2p, r2.reward), color=MINT, buff=0.12),
            ),
        )
        self.speak(
            "[Even benchmarks need checking]: [developers reviewed nearly seventeen hundred SWE-bench tasks], [and kept only five hundred.]",
            m.AnimationGroup(m.FadeIn(verified_t), m.FadeIn(vsrc)),
            m.FadeIn(verified_1, shift=0.1 * m.LEFT),
            m.FadeIn(verified_2, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · pass@k vs pass^k

    def pass_k(self):
        self.next_section("4 · pass@k vs pass^k", notes=f"Li, AI Agents in Depth, ch. 7: pass@k = 1 − (1 − p)^k (at least one of k runs passes; a capability ceiling, 'technical wonders'); pass^k = p^k (all k runs pass; business reliability). At p = {P}, k = {K}: {100 * pass_at(K):.1f}% vs {100 * pass_hat(K):.1f}% (computed and asserted). The five tries shown ({sum(TRIES)} of {len(TRIES)} pass) are illustrative. " + FACTS)
        label = text("one task, five tries", 22, MUTED).align_to([-6.4, 0, 0], m.LEFT)
        p_chip = mono_chip("p = 0.6", LIME, LIME_FILL, size=22, height=0.52)
        tries = m.VGroup()
        for ok in TRIES:
            box = card(0.56, 0.56, MINT if ok else ROSE, MINT_FILL if ok else ROSE_FILL, radius=0.12, stroke=2)
            tries.add(m.VGroup(box, mark(ok, 26).move_to(box)))
        tries.arrange(m.RIGHT, buff=0.14)
        top = m.VGroup(label, p_chip, tries).arrange(m.RIGHT, buff=0.4)
        top.align_to([-6.4, 0, 0], m.LEFT).set_y(2.0)
        v_any = pill("pass@5 ✓", AZURE, AZURE_FILL, 22, mono=True)
        v_all = pill("pass^5 ✗", SUN, SUN_FILL, 22, mono=True)
        verdicts = m.VGroup(v_any, v_all).arrange(m.RIGHT, buff=0.25).next_to(top, m.RIGHT, buff=0.45)
        assert verdicts.get_right()[0] < 6.5, verdicts.get_right()[0]

        chart = Chart(11.0, 1.2, 7.0, 4.0, [-2.2, -1.2, 0])
        frame = m.VGroup(
            chart.ticks(xs=(1, 5, 10), ys=(0.5, 1.0)),
            *[chart.xlabel(k, str(k)) for k in (1, 5, 10)],
            *[chart.ylabel(v, f"{round(100 * v)}%") for v in (0.5, 1.0)],
            text("tries, k", 22, MUTED).move_to(chart.p(7.6, 0) + 0.32 * m.DOWN),
        )
        ks = np.linspace(1, 10, 91)
        c_at = chart.line([(k, pass_at(k)) for k in ks], AZURE, 5)
        c_hat = chart.line([(k, pass_hat(k)) for k in ks], SUN, 5)
        n_at = code("pass@k", 20, AZURE, weight=m.BOLD).next_to(chart.p(10, pass_at(10)), m.UP, buff=0.14)
        n_hat = code("pass^k", 20, SUN, weight=m.BOLD).next_to(chart.p(8.8, pass_hat(8.8)), m.UP, buff=0.16)
        k5 = m.DashedLine(chart.p(K, 0), chart.p(K, 1.06), dash_length=0.08).set_stroke(FAINT, 2.5)
        d_at = m.Dot(chart.p(K, pass_at(K)), radius=0.08, color=AZURE).set_z_index(3)
        d_hat = m.Dot(chart.p(K, pass_hat(K)), radius=0.08, color=SUN).set_z_index(3)

        rx = 4.55
        f_at = code("pass@k = 1 - (1 - p)^k", 22, AZURE).move_to([rx, 0.95, 0])
        f_hat = code("pass^k = p^k", 22, SUN).move_to([rx, -1.0, 0])
        v_at = title_text(f"{100 * pass_at(K):.1f}%", 50, AZURE).move_to([rx, 0.25, 0])
        v_hat = title_text(f"{100 * pass_hat(K):.1f}%", 50, SUN).move_to([rx, -1.7, 0])
        t_at = text("“technical wonders”", 22, INK).next_to(v_at, m.DOWN, buff=0.12)
        t_hat = text("“business reliability”", 22, INK).next_to(v_hat, m.DOWN, buff=0.12)
        at5 = code("at k = 5", 20, MUTED)
        at5.next_to(k5, m.UP, buff=0.08)
        src = cite("Li, AI Agents in Depth · ch. 7").move_to([rx, -3.2, 0])
        assert t_hat.get_bottom()[1] > src.get_top()[1] + 0.1 and f_at.get_left()[0] > n_at.get_right()[0] + 0.3

        self.speak(
            "[Agents are random], [so run each task more than once.] [Say Otto solves this task sixty percent of the time], [and we try it five times.]",
            self.new_header("04 · repeat", "pass@k and pass^k"),
            m.FadeIn(label),
            m.FadeIn(p_chip, scale=0.8),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.DOWN) for t in tries], lag_ratio=0.3),
        )
        self.speak(
            "[Pass at k asks: did any try succeed?] [Pass hat k asks: did every try succeed?]",
            m.AnimationGroup(m.FadeIn(f_at, shift=0.1 * m.UP), m.FadeIn(v_any, shift=0.1 * m.LEFT), m.Circumscribe(tries[0], color=AZURE, buff=0.06)),
            m.AnimationGroup(m.FadeIn(f_hat, shift=0.1 * m.UP), m.FadeIn(v_all, shift=0.1 * m.LEFT), m.Circumscribe(tries, color=SUN, buff=0.08)),
        )
        self.speak(
            "[Plot both against the number of tries]: [one climbs toward a hundred], [the other sinks toward zero.]",
            m.AnimationGroup(m.Create(chart.ax), m.FadeIn(frame)),
            m.AnimationGroup(m.Create(c_at), m.FadeIn(n_at)),
            m.AnimationGroup(m.Create(c_hat), m.FadeIn(n_hat)),
        )
        self.speak(
            "[At five tries]: [ninety-nine percent], [against just seven point eight.]",
            m.AnimationGroup(m.Create(k5), m.FadeIn(at5)),
            m.AnimationGroup(m.FadeIn(d_at, scale=0.5), m.FadeIn(v_at, target_position=d_at, scale=0.5)),
            m.AnimationGroup(m.FadeIn(d_hat, scale=0.5), m.FadeIn(v_hat, target_position=d_hat, scale=0.5)),
        )
        self.speak(
            "[Pass at k measures technical wonders.] [Pass hat k measures business reliability]: [what a customer gets every single time.]",
            m.AnimationGroup(m.FadeIn(t_at, shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(t_hat, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(v_hat, t_hat), color=SUN, buff=0.12),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · LLM as a judge

    def judge(self):
        self.next_section("5 · LLM as a judge", notes=f"Li, AI Agents in Depth, ch. 7: a judge model grades against a rubric (essential / important / optional weights plus a veto item, here 'an invented fact'); judges favour longer answers (length bias) and the answer in a given position (position bias): evaluate each pair twice with the order swapped and average; same-family judges invite Goodhart: the agent 'will gradually learn to avoid the types of errors that the judging model is not good at detecting'; use judges from other families. Illustrative: the two reports, the rubric rows, the scores ({ORDER_1} with A first, {ORDER_2} with B first, average {AVG}). Self-review by the same model: Episode 26. " + FACTS)
        rub_box = card(4.65, 2.95, EDGE, PANEL, radius=0.18, stroke=2).move_to([-4.175, -0.7, 0])
        rub_title = code("rubric", 20, MUTED).align_to(rub_box, m.UL).shift(0.3 * m.RIGHT + 0.2 * m.DOWN)
        spec = [("essential", "names the cause", INK), ("essential", "says the tests ran", INK),
                ("important", "names the file", MUTED), ("optional", "suggests a next step", MUTED),
                ("veto", "any invented fact", ROSE)]
        rub_rows = []
        for i, (tag, desc, color) in enumerate(spec):
            t = code(tag, 20, color if color != INK else LIME, weight=m.BOLD)
            d = text(desc, 20, INK if color != MUTED else MUTED)
            t.align_to(rub_box, m.LEFT).shift(0.3 * m.RIGHT)
            d.align_to(rub_box, m.LEFT).shift(1.95 * m.RIGHT)
            y = rub_box.get_top()[1] - 0.85 - i * 0.45
            sit(t, y)
            sit(d, y)
            rub_rows.append(m.VGroup(t, d))
        essentials = m.VGroup(rub_rows[0], rub_rows[1])
        assert max(r.get_right()[0] for r in rub_rows) < rub_box.get_right()[0] - 0.15, max(r.get_right()[0] for r in rub_rows)

        judge_icon = icon("ep29-scales", 1.0)
        judge_lab = text("judge model", 24, INK, weight=m.BOLD)
        judge = m.VGroup(judge_icon, judge_lab).arrange(m.RIGHT, buff=0.3).move_to([0, 1.65, 0]).align_to(rub_box, m.LEFT).shift(0.2 * m.RIGHT)
        a = report_card("report A", "Fixed: total() skipped\nthe last item. 13 passed.", width=4.6)
        b = report_card("report B", "Great question! After a careful\nlook at cart.py, checkout.py and\nevery test, I found that total()\nskipped the last item, so I fixed\nthe loop. All 13 tests pass.", width=4.6)
        assert abs(a.width - b.width) < 1e-6, (a.width, b.width)
        cx, top_y, bot_y = 3.3, 1.4, -0.5
        a.move_to([cx, top_y, 0])
        b.move_to([cx, bot_y, 0])
        a_low = a.copy().move_to([cx, bot_y, 0])
        b_high = b.copy().move_to([cx, top_y, 0])
        first = code("read 1st", 20, AZURE, weight=m.BOLD).move_to([0.2, top_y, 0])
        second = code("read 2nd", 20, MUTED).move_to([0.2, bot_y, 0])
        assert first.get_right()[0] < b.get_left()[0] - 0.1 and rub_box.get_right()[0] < first.get_left()[0] - 0.1, (first.get_right()[0], b.get_left()[0])
        assert judge.get_right()[0] < first.get_left()[0] - 0.2 and rub_box.get_top()[1] < judge.get_bottom()[1] - 0.2

        def score(v, y):
            return mono_chip(str(v), SUN, SUN_FILL, size=26, height=0.6, pad=0.45).move_to([6.05, y, 0])

        s1 = m.VGroup(score(ORDER_1["A"], top_y), score(ORDER_1["B"], bot_y))
        s2 = m.VGroup(score(ORDER_2["B"], top_y), score(ORDER_2["A"], bot_y))
        assert s1[0].get_left()[0] > b.get_right()[0] + 0.05, (s1[0].get_left()[0], b.get_right()[0])
        avg = code(f"average:  A {AVG['A']:.0f}   B {AVG['B']:.0f}", 22, INK)
        tie = pill("a tie: the gap was position", SUN, SUN_FILL, 22)
        avg_row = m.VGroup(avg, tie).arrange(m.RIGHT, buff=0.35).move_to([0, -2.14, 0]).align_to([6.4, 0, 0], m.RIGHT)
        assert avg_row.get_left()[0] > rub_box.get_right()[0] + 0.2 and avg_row.get_top()[1] < b.get_bottom()[1] - 0.08, (avg_row.get_left()[0], avg_row.get_top()[1], b.get_bottom()[1])
        fam = pill("same model family as Otto", ROSE, ROSE_FILL, 22)
        good = text("→ Otto learns the mistakes the judge can't see", 24, INK)
        fam_row = m.VGroup(fam, good).arrange(m.RIGHT, buff=0.3).move_to([0, -2.82, 0])
        fsrc = cite("Goodhart's law · Li, AI Agents in Depth · ch. 7 · self-review: Episode 26").next_to(fam_row, m.DOWN, buff=0.08)
        assert fam_row.width < 12.8 and fsrc.get_bottom()[1] > -3.55

        self.speak(
            "[Not everything has a test]: [say, two versions of Otto's report to the user.] [A second model grades them]: [an LLM as a judge.]",
            self.new_header("05 · the judge", "An LLM as a judge"),
            m.LaggedStart(m.FadeIn(a, shift=0.15 * m.LEFT), m.FadeIn(b, shift=0.15 * m.LEFT), lag_ratio=0.4),
            m.FadeIn(judge, shift=0.15 * m.UP),
            m.Circumscribe(judge, color=CORAL, buff=0.12),
        )
        self.speak(
            "[It scores against a rubric]: [essentials weigh most], [and one invented fact vetoes the whole answer.]",
            m.AnimationGroup(m.FadeIn(rub_box), m.FadeIn(rub_title), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.RIGHT) for r in rub_rows], lag_ratio=0.15)),
            m.Circumscribe(essentials, color=LIME, buff=0.08),
            m.Circumscribe(rub_rows[4], color=ROSE, buff=0.08),
        )
        self.speak(
            "[But judges are biased]: [toward longer answers], [and toward whichever answer they read first.]",
            m.AnimationGroup(m.FadeIn(first, shift=0.1 * m.RIGHT), m.FadeIn(second, shift=0.1 * m.RIGHT)),
            m.Circumscribe(b, color=SUN, buff=0.08),
            m.Circumscribe(m.VGroup(first, a), color=AZURE, buff=0.08),
        )
        self.speak(
            "[So ask twice, swapping the order], [and average]: [here, the winner was just whichever came first.]",
            m.Succession(
                m.FadeIn(s1, shift=0.1 * m.LEFT),
                m.Wait(0.5),
                m.AnimationGroup(m.FadeOut(s1), m.ReplacementTransform(a, a_low, path_arc=-0.6), m.ReplacementTransform(b, b_high, path_arc=-0.6)),
                m.FadeIn(s2, shift=0.1 * m.LEFT),
            ),
            m.FadeIn(avg, shift=0.1 * m.UP),
            m.FadeIn(tie, scale=1.1),
        )
        self.speak(
            "[Like self-review in episode twenty-six], [a judge from Otto's model family shares its blind spots]: [Otto learns the mistakes it can't see.]",
            m.FadeIn(fsrc),
            m.AnimationGroup(m.FadeIn(fam, shift=0.1 * m.UP), m.Circumscribe(judge, color=ROSE, buff=0.12)),
            m.FadeIn(good, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the first error

    def first_error(self):
        self.next_section("6 · The first error", notes=f"Li, AI Agents in Depth, ch. 7 (failure attribution): record 'the first step at which unacceptable behavior appeared'. The expenses.jpg case (not tinyshop: the book's own example): asked to enter the expenses from expenses.jpg into a bookkeeping app, the agent took {STEPS} steps, none errored, and declared the task complete; the validator found the expected ¥436.35 'Dress' row missing. By step {SYMPTOM}, {INVENTED} invented expenses were in its notes. The first error is step {FIRST}, an assistant message, not a tool call or an error: 'I cannot actually see the content/details of the expenses in the image.' T3A is text-only (its observation is the element tree, not pixels): the root cause is a missing observation channel (and no legal way to report 'information unavailable'), not 'the model cannot do OCR'; swapping models would not fix it. " + FACTS)
        photo = icon("ep29-photo", 0.5)
        ask = text("“Enter the expenses from expenses.jpg into a bookkeeping app”", 24, INK)
        task_row = m.VGroup(photo, ask).arrange(m.RIGHT, buff=0.25).align_to([-6.12, 0, 0], m.LEFT).set_y(2.15)
        src = cite("Li, AI Agents in Depth · ch. 7").align_to([6.12, 0, 0], m.RIGHT).set_y(1.45)

        w, gap, strip_y = 0.31, 0.075, 0.85
        cells = m.VGroup(*[card(w, 0.55, EDGE, CARD, radius=0.07, stroke=1.5) for _ in range(STEPS)]).arrange(m.RIGHT, buff=gap)
        cells.move_to([0, strip_y, 0])
        ok_cells = recolor_cells(cells, [(MINT, MINT_FILL)] * STEPS)

        def num(i, color=MUTED):
            return code(str(i), 20, color, weight=m.BOLD).next_to(cells[i - 1], m.DOWN, buff=0.12)

        n1, n32 = num(1), num(STEPS)
        n11, n8 = num(SYMPTOM, SUN), num(FIRST, ROSE)
        claim = pill("“task complete”", CORAL, CORAL_FILL, 22).align_to(cells, m.RIGHT).set_y(-0.35)
        verdict = pill("✗ verifier: the ¥436.35 row is missing", ROSE, ROSE_FILL, 22).align_to(cells, m.RIGHT).set_y(-1.05)
        cursor = m.SurroundingRectangle(cells[-1], buff=0.07, corner_radius=0.08).set_stroke(SUN, 4)
        at8 = m.SurroundingRectangle(cells[FIRST - 1], buff=0.07, corner_radius=0.08).set_stroke(ROSE, 4)
        c11 = recolor_cells(cells, [(SUN, SUN_FILL) if i == SYMPTOM - 1 else None for i in range(STEPS)])
        c8 = recolor_cells(cells, [(ROSE, ROSE_FILL) if i == FIRST - 1 else None for i in range(STEPS)])
        lab11 = text(f"step {SYMPTOM}: {INVENTED} invented expenses in its notes", 20, SUN)
        lab11.next_to(cells[SYMPTOM - 1], m.UP, buff=0.2)
        lab11.align_to(cells[SYMPTOM - 1], m.LEFT).shift(0.3 * m.LEFT)
        assert lab11.get_top()[1] < ask.get_bottom()[1] - 0.12 and lab11.get_left()[0] > -6.4 and lab11.get_right()[0] < src.get_left()[0] - 0.3
        assert ask.get_right()[0] < 6.3 and src.get_top()[1] < ask.get_bottom()[1] - 0.1
        msg = message("assistant", "“I cannot actually see the content/details\nof the expenses in the image”", width=6.0, size=22)
        msg.align_to(cells, m.LEFT).set_y(-1.25)
        link = m.Line(n8.get_bottom() + 0.06 * m.DOWN, [n8.get_x(), msg.get_top()[1] + 0.04, 0]).set_stroke(ROSE, 3)
        msg_note = text("no error raised, not a tool call", 20, MUTED).next_to(msg, m.DOWN, buff=0.14).align_to(msg, m.LEFT)
        assert msg.get_right()[0] < verdict.get_left()[0] - 0.2, (msg.get_right()[0], verdict.get_left()[0])

        eye = icon(recolored("ep01-eye", ROSE, ROSE_FILL, prefix=PREFIX), 0.42)
        slash = m.Line(eye.get_corner(m.DL), eye.get_corner(m.UR)).set_stroke(ROSE, 4)
        text_only = text("text-only: no pixels", 22, INK)
        left_row = m.VGroup(m.VGroup(eye, slash), text_only).arrange(m.RIGHT, buff=0.25)
        not_ocr = text("✗ not: can't read images", 22, MUTED)
        paint(not_ocr, not_ocr.text, {"✗": ROSE})
        channel = pill("missing observation channel", ROSE, ROSE_FILL, 22)
        right_row = m.VGroup(not_ocr, channel).arrange(m.RIGHT, buff=0.3)
        cause = m.VGroup(left_row, right_row).arrange(m.RIGHT, buff=0.7).move_to([0, -2.95, 0])
        assert cause.width < 12.7, cause.width
        assert msg_note.get_bottom()[1] > cause.get_top()[1] + 0.12

        self.speak(
            "[When a run fails], [find the first error in its trace.]",
            self.new_header("06 · diagnose", "Find the first error"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in cells], lag_ratio=0.04), m.FadeIn(n1), m.FadeIn(n32)),
        )
        self.speak(
            "[Here's a case from AI Agents in Depth]: [an agent asked to enter the expenses in a photo into a bookkeeping app.]",
            m.FadeIn(src),
            m.AnimationGroup(m.FadeIn(photo, scale=0.6), m.FadeIn(ask, shift=0.1 * m.LEFT)),
        )
        self.speak(
            "[Thirty-two steps, not one error], [and it reports the task complete.] [But the verifier finds the real expense missing.]",
            m.LaggedStart(*[m.FadeIn(c) for c in ok_cells], lag_ratio=0.05),
            m.FadeIn(claim, shift=0.1 * m.UP),
            m.FadeIn(verdict, scale=1.1),
        )
        self.speak(
            "[Search back through the trace.] [By step eleven, four invented expenses are already in its notes.]",
            m.Succession(m.FadeIn(cursor, scale=1.3), cursor.animate(run_time=1.6).move_to(cells[SYMPTOM - 1])),
            m.AnimationGroup(m.FadeIn(c11), m.FadeIn(n11), m.FadeIn(lab11, shift=0.1 * m.DOWN)),
        )
        self.speak(
            "[But the first error is step eight], [where the agent wrote that it couldn't actually see the image], [and kept going.]",
            m.AnimationGroup(m.ReplacementTransform(cursor, at8), m.FadeIn(c8), m.FadeIn(n8)),
            m.AnimationGroup(m.Create(link), m.FadeIn(msg, shift=0.15 * m.DOWN)),
            m.FadeIn(msg_note, shift=0.1 * m.UP),
        )
        self.speak(
            "[The agent was text-only.] [The root cause wasn't poor image reading], [but a missing observation channel.]",
            m.FadeIn(left_row, shift=0.1 * m.UP),
            m.FadeIn(not_ocr, shift=0.1 * m.LEFT),
            m.Succession(m.FadeIn(channel, scale=1.1), m.Circumscribe(channel, color=ROSE, buff=0.1)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · right actions, wrong report

    def wrong_report(self):
        self.next_section("7 · Right actions, wrong report", notes=f"Li, AI Agents in Depth, ch. 7: 'right actions, wrong report' is the category most often hidden by an overall pass rate, 'because most evaluations assert only on environment state'. Of the {COMM_RUNS} published τ²-bench baseline runs whose task carries a communication requirement, {COMM_FAILED} failed; {COMM_CHECK_FAILED} failed the communication check, and {RIGHT_STATE_WRONG_REPORT} ('a third of all failures') had a correct environment state and a wrong report. Drawn: {COMM_FAILED} cells, {RIGHT_STATE_WRONG_REPORT} highlighted. Episode 4: an answer is not a completed task. " + FACTS)
        ep4 = m.VGroup(code("Episode 4", 20, CORAL, weight=m.BOLD), text("an answer is not a completed task", 26, INK)).arrange(m.RIGHT, buff=0.3)
        rev = m.VGroup(code("the reverse", 20, SUN, weight=m.BOLD), text("a completed task is not a good answer", 26, INK)).arrange(m.RIGHT, buff=0.3)
        for g in (ep4, rev):
            sit(g[0], baseline(g[1]))
        tops = m.VGroup(ep4, rev).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT).align_to([-6.3, 0, 0], m.LEFT).set_y(1.85)
        waffle = cell_grid(20, 12, 0.2, 0.06, [-3.75, -1.2, 0], color=ROSE, fill=ROSE_FILL, stroke=1.2)
        hits = recolor_cells(waffle, [(SUN, SUN_FILL) if i < RIGHT_STATE_WRONG_REPORT else None for i in range(COMM_FAILED)])
        rx = 0.35
        big = title_text(f"{COMM_FAILED} failed runs", 34, INK).align_to([rx, 0, 0], m.LEFT).set_y(0.45)
        big_sub = text("tasks with a communication requirement", 20, MUTED).next_to(big, m.DOWN, buff=0.12).align_to(big, m.LEFT)
        third = title_text(f"{RIGHT_STATE_WRONG_REPORT}: a third", 34, SUN).align_to([rx, 0, 0], m.LEFT).set_y(-0.75)
        env_ok = m.VGroup(code("env_assertions", 22, INK), mark(True, 28)).set_y(-1.45)
        rep_bad = m.VGroup(code("communicate_info", 22, INK), mark(False, 28)).set_y(-1.95)
        for g in (env_ok, rep_bad):
            g[0].align_to([rx, 0, 0], m.LEFT)
            g[1].move_to([rx + 3.35, g[0].get_y(), 0])
        src = cite("τ²-bench runs · Li, AI Agents in Depth · ch. 7").align_to([rx, 0, 0], m.LEFT).set_y(-2.95)
        assert tops.get_bottom()[1] > waffle.get_top()[1] + 0.2 and src.get_right()[0] < 6.5, (tops.get_bottom()[1], waffle.get_top()[1], src.get_right()[0])

        self.speak(
            "[Episode four said an answer is not a completed task.] [The reverse happens too.]",
            m.AnimationGroup(self.new_header("07 · the report", "Right actions, wrong report"), m.FadeIn(ep4, shift=0.1 * m.UP)),
            m.FadeIn(rev, shift=0.1 * m.UP),
        )
        self.speak(
            "[In one benchmark's published runs], [a third of the failures with a communication requirement] [had the right end state], [and the wrong report.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for c in waffle], lag_ratio=0.004), m.FadeIn(big), m.FadeIn(big_sub), m.FadeIn(src)),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for c in hits], lag_ratio=0.01), m.FadeIn(third, shift=0.1 * m.UP)),
            m.FadeIn(env_ok, shift=0.1 * m.LEFT),
            m.FadeIn(rep_bad, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · statistics

    def statistics(self):
        self.next_section("8 · Is it noise?", notes=f"Li, AI Agents in Depth, ch. 7: 'with 100 cases and a 70% success rate, the 95% confidence interval is about 70%±9 percentage points' (computed: 1.96·√(0.7·0.3/100) = {100 * half_width(SCORE_OLD):.1f} points; 73%: ±{100 * half_width(SCORE_NEW):.1f}); 73% vs 70% does not justify switching; compare per task (McNemar's test or a paired bootstrap); run 3–5 seeds and report the spread. The new prompt's paired results are illustrative (computed from the scripted runs): both pass {BOTH_PASS}, both fail {BOTH_FAIL}, old only {OLD_ONLY}, new only {NEW_ONLY}; exact McNemar p = {P_MCNEMAR:.2f}. " + FACTS)
        x0, s = -2.6, 0.2  # x of 50%, units per point

        def xv(v):
            return x0 + (v - 50) * s

        axis_y = 0.25
        axis = m.Line([xv(50), axis_y, 0], [xv(90), axis_y, 0]).set_stroke(FAINT, 2.5)
        tick_marks = m.VGroup(*[m.Line([xv(v), axis_y - 0.07, 0], [xv(v), axis_y + 0.07, 0]).set_stroke(FAINT, 2.5) for v in range(50, 91, 10)])
        tick_labels = m.VGroup(*[code(f"{v}%", 20, MUTED).move_to([xv(v), axis_y - 0.33, 0]) for v in range(50, 91, 10)])
        assert xv(90) + 0.4 < 6.5

        def row(name, score, ci, y, color):
            lab = code(f"{name}  {round(100 * score)}%", 22, color, weight=m.BOLD).align_to([-6.4, 0, 0], m.LEFT).set_y(y)
            dot = m.Dot([xv(100 * score), y, 0], radius=0.11, color=color).set_z_index(3)
            bar = m.Line([xv(ci[0]), y, 0], [xv(ci[1]), y, 0]).set_stroke(color, 9)
            caps = m.VGroup(*[m.Line([xv(c), y - 0.16, 0], [xv(c), y + 0.16, 0]).set_stroke(color, 4) for c in ci])
            return lab, dot, m.VGroup(bar, caps)

        o_lab, o_dot, o_ci = row("old prompt", SCORE_OLD, CI_OLD, 1.75, AZURE)
        n_lab, n_dot, n_ci = row("new prompt", SCORE_NEW, CI_NEW, 1.0, SUN)
        assert o_lab.get_right()[0] < xv(CI_OLD[0]) - 0.2 and n_lab.get_right()[0] < xv(CI_NEW[0]) - 0.2
        pm = code(f"±{round(100 * half_width(SCORE_OLD))} points", 20, AZURE, weight=m.BOLD).next_to(o_ci, m.RIGHT, buff=0.25)
        noise = pill("noise", ROSE, ROSE_FILL, 22).next_to(n_ci, m.RIGHT, buff=0.3)
        assert noise.get_right()[0] < 6.5 and pm.get_right()[0] < 6.5

        cats = {n: ("both" if OLD[n] and NEW[n] else "none" if not OLD[n] and not NEW[n] else "old" if OLD[n] else "new") for n in OLD}
        look = {"both": (MINT, MINT_FILL), "none": (ROSE, ROSE_FILL), "old": (AZURE, AZURE), "new": (SUN, SUN)}
        grid = cell_grid(10, 10, 0.23, 0.06, [-4.95, -2.0, 0], stroke=1.2)
        base = recolor_cells(grid, [look["both"] if cats[n] == "both" else look["none"] for n in range(1, N_TASKS + 1)])
        base.set_stroke(opacity=0.6)
        diff = recolor_cells(grid, [look[cats[n]] if cats[n] in ("old", "new") else None for n in range(1, N_TASKS + 1)])
        legend_rows = [(f"both pass   {BOTH_PASS}", MINT), (f"both fail   {BOTH_FAIL}", ROSE),
                       (f"old only     {OLD_ONLY}", AZURE), (f"new only     {NEW_ONLY}", SUN)]
        legend = m.VGroup(*[code(s_, 20, c, weight=m.BOLD if c in (AZURE, SUN) else m.NORMAL) for s_, c in legend_rows]).arrange(m.DOWN, buff=0.13, aligned_edge=m.LEFT)
        legend.next_to(grid, m.RIGHT, buff=0.4).align_to(grid, m.UP)
        pval = m.VGroup(text("paired test", 22, MUTED), code(f"p = {P_MCNEMAR:.2f}", 24, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
        pval.next_to(legend, m.RIGHT, buff=0.7).align_to(legend, m.UP).shift(0.05 * m.DOWN)
        chance = text("chance explains it", 22, ROSE).next_to(pval, m.DOWN, buff=0.15).align_to(pval, m.LEFT)
        rules = m.VGroup(word_chip("paired tests", LIME, LIME_FILL, size=22, height=0.55), word_chip("3–5 seeds", LIME, LIME_FILL, size=22, height=0.55)).arrange(m.RIGHT, buff=0.25)
        rules.next_to(chance, m.DOWN, buff=0.35).align_to(pval, m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 7").align_to([6.4, 0, 0], m.RIGHT).set_y(-3.3)
        assert rules.get_right()[0] < 6.45 and rules.get_bottom()[1] > src.get_top()[1] + 0.05, (rules.get_right()[0], rules.get_bottom()[1])
        assert grid.get_top()[1] < tick_labels.get_bottom()[1] - 0.25

        self.speak(
            "[Now the new prompt scores seventy-three.] [Is that better than seventy?]",
            m.AnimationGroup(self.new_header("08 · repeat", "Is it noise?"), m.FadeIn(o_lab), m.FadeIn(o_dot, scale=0.5),
                             m.FadeIn(axis), m.FadeIn(tick_marks), m.FadeIn(tick_labels)),
            m.AnimationGroup(m.FadeIn(n_lab), m.FadeIn(n_dot, scale=0.5)),
        )
        self.speak(
            "[With a hundred tasks at seventy percent], [the ninety-five percent confidence interval is about nine points either way.]",
            m.FadeIn(src),
            m.AnimationGroup(m.GrowFromPoint(o_ci, o_dot.get_center()), m.FadeIn(pm, shift=0.1 * m.LEFT)),
        )
        self.speak(
            "[The new score's interval overlaps almost entirely]: [three points is noise.]",
            m.GrowFromPoint(n_ci, n_dot.get_center()),
            m.FadeIn(noise, scale=1.1),
        )
        self.speak(
            "[Compare them task by task]: [they disagree on only thirteen tasks], [eight to five], [which chance explains easily.]",
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c) for c in base], lag_ratio=0.006), m.FadeIn(legend[0]), m.FadeIn(legend[1])),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, scale=1.3) for c in diff], lag_ratio=0.08)),
            m.AnimationGroup(m.FadeIn(legend[2], shift=0.1 * m.LEFT), m.FadeIn(legend[3], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(pval, shift=0.1 * m.LEFT), m.FadeIn(chance)),
        )
        self.speak(
            "[So use paired tests], [and several seeds], before you trust an improvement.",
            m.FadeIn(rules[0], shift=0.1 * m.UP),
            m.FadeIn(rules[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · observe before you prompt

    def observe_first(self):
        self.next_section("9 · Observe before you prompt", notes=f"Li, AI Agents in Depth, ch. 7, Table 7-5 (AndroidWorld Wi-Fi slice, {ARM} tasks per arm, one change at a time): more navigation instructions {SLICE[0][1]}/{ARM} → {SLICE[0][2]}/{ARM} (25% → 25%); switching the accessibility feed to a richer UIAutomator tree {SLICE[1][1]}/{ARM} → {SLICE[1][2]}/{ARM} (25% → 100%) at {SLICE[1][3]}× tokens; compacting that tree 100% → 100% at {SLICE[2][3]}× tokens. 'Observation failures should be investigated before prompts are expanded'; '4/4 on the slice must not be reported as 100% system-wide success' (the next gate: five seeds over all 116 tasks). " + FACTS)
        lab_x, bar_x0, bar_w = -6.3, -2.2, 5.0
        head_y = 2.2
        h1 = text("only change", 22, MUTED).align_to([lab_x, 0, 0], m.LEFT)
        h2 = text(f"success on {ARM} Wi-Fi tasks", 22, MUTED).align_to([bar_x0, 0, 0], m.LEFT)
        h3 = text("tokens", 22, MUTED).align_to([5.15, 0, 0], m.LEFT)
        for h in (h1, h2, h3):
            sit(h, head_y)
        ys = [1.45, 0.55, -0.35, -1.25]
        rows_ = [("baseline agent", None, SLICE[0][1], None)] + [(n, a, b, t) for n, a, b, t in SLICE]
        built = []
        for (name, ctrl, treat, tokens), y in zip(rows_, ys):
            lab = text(name, 26, INK if ctrl is not None else MUTED, weight=m.SEMIBOLD).align_to([lab_x, 0, 0], m.LEFT).set_y(y)
            if ctrl is not None:
                lab = m.VGroup(code("+", 24, LIME, weight=m.BOLD), text(name, 26, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2)
                lab.align_to([lab_x, 0, 0], m.LEFT).set_y(y)
            track = m.RoundedRectangle(corner_radius=0.14, width=bar_w, height=0.28, stroke_color=EDGE, stroke_width=2, fill_color=CARD, fill_opacity=1)
            track.align_to([bar_x0, 0, 0], m.LEFT).set_y(y)
            frac = treat / ARM
            color = MUTED if ctrl is None else (MINT if treat == ARM else ROSE)
            start = (ctrl if ctrl is not None else treat) / ARM
            bar0 = m.RoundedRectangle(corner_radius=0.14, width=bar_w * start, height=0.28, stroke_width=0, fill_color=color, fill_opacity=1).align_to(track, m.LEFT).set_y(y)
            bar1 = m.RoundedRectangle(corner_radius=0.14, width=bar_w * frac, height=0.28, stroke_width=0, fill_color=color, fill_opacity=1).align_to(track, m.LEFT).set_y(y)
            val = code(f"{round(100 * frac)}%", 24, color, weight=m.BOLD).next_to(track, m.RIGHT, buff=0.25)
            tok = None if tokens is None else code(f"{tokens:.2f}×", 22, ROSE if tokens > 1 else MINT).align_to([5.15, 0, 0], m.LEFT).set_y(y)
            built.append(NS(lab=lab, track=track, bar0=bar0, bar1=bar1, val=val, tok=tok, y=y))
            assert val.get_right()[0] < 5.0, val.get_right()[0]
        base_r, r1, r2, r3 = built
        stmt = title_text("Investigate observation failures before expanding prompts.", 30, INK).move_to([0, -2.3, 0])
        note = m.VGroup(text("4 out of 4 on a slice is not 100% overall", 22, MUTED), cite("AndroidWorld · Li, AI Agents in Depth · ch. 7")).arrange(m.RIGHT, buff=0.6)
        note.next_to(stmt, m.DOWN, buff=0.25)
        assert stmt.width < 12.9 and note.get_bottom()[1] > -3.55

        self.speak(
            "[One last lesson, from a phone-control benchmark.] [On four Wi-Fi tasks, an agent passed just one.]",
            m.AnimationGroup(self.new_header("09 · observe first", "Look before you prompt"), m.FadeIn(h1), m.FadeIn(h2), m.FadeIn(h3)),
            m.AnimationGroup(m.FadeIn(base_r.lab), m.FadeIn(base_r.track), m.GrowFromEdge(base_r.bar1, m.LEFT), m.FadeIn(base_r.val)),
        )
        self.speak(
            "[More instructions]: [still twenty-five percent.] [A richer view of the screen]: [one hundred.] [Compacted, it kept that at half the tokens.]",
            m.AnimationGroup(m.FadeIn(r1.lab, shift=0.1 * m.RIGHT), m.FadeIn(r1.track), m.FadeIn(r1.bar0)),
            m.AnimationGroup(m.ReplacementTransform(r1.bar0, r1.bar1), m.FadeIn(r1.val)),
            m.AnimationGroup(m.FadeIn(r2.lab, shift=0.1 * m.RIGHT), m.FadeIn(r2.track), m.FadeIn(r2.bar0)),
            m.AnimationGroup(m.ReplacementTransform(r2.bar0, r2.bar1), m.FadeIn(r2.val), m.FadeIn(r2.tok)),
            m.AnimationGroup(m.FadeIn(r3.lab, shift=0.1 * m.RIGHT), m.FadeIn(r3.track), m.FadeIn(r3.bar1), m.FadeIn(r3.val), m.FadeIn(r3.tok)),
        )
        self.speak(
            "[Investigate observation failures before expanding prompts.] [And four out of four on a slice is not a hundred percent overall.]",
            m.Write(stmt),
            m.FadeIn(note, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Verify (tasks with environments and verifiers that assert on the world; both test lists; judges with rubrics, swapped order, other families), repeat (pass^k for reliability, confidence intervals, paired tests, seeds), diagnose (the first error in the trace; observation before prompts). Next: Episode 30 trains on rewards from verifiers like these.")
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 66, c) for w, c in (("Verify", MINT), ("Repeat", SUN), ("Diagnose", ROSE))])
        subs = m.VGroup(*[code(s_, 20, MUTED) for s_ in ("assert on the world", "pass^k · intervals · seeds", "find the first error")])
        slots = [max(w.width, s_.width) for w, s_ in zip(words, subs)]
        x, gap = 0.0, 0.85
        for w, s_, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s_.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = baseline(words[0]) - 0.6
        for s_ in subs:
            sit(s_, line_y)
        mid = words[1][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tag1 = text("One good run is an anecdote.", 28, MUTED)
        tag2 = text("An evaluation set is evidence, and its verifiers are next episode's rewards.", 28, MUTED)
        tagline = m.VGroup(tag1, tag2).arrange(m.DOWN, buff=0.2).next_to(row_, m.DOWN, buff=0.65)
        assert row_.width < 12.8 and tagline.width < 12.8, (row_.width, tagline.width)

        self.speak(
            "[To sum up]: [verify], [repeat], [diagnose.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.speak(
            "[One good run is an anecdote.] [An evaluation set is evidence], [and its verifiers become next episode's rewards.]",
            m.FadeIn(tag1, shift=0.1 * m.UP),
            m.FadeIn(tag2, shift=0.1 * m.UP),
            m.Circumscribe(words[0], color=MINT, buff=0.12),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
