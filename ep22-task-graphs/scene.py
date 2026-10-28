"""Agents, Animated · Episode 22 · Task Graphs (Module 5 · Scaling Up)

How goals become ordered, verifiable work. Running example: Otto is asked to "Add payments to tinyshop."
The run is not drawn by hand: the scene implements Learn Claude Code's s12 task system (a Task record per
file in .tasks/, can_start, claim_task, complete_task) over an in-memory disk, adds one change in the same
style (complete_task runs the tests itself and rejects a "done" that fails them), and runs the reference
agent_loop against a scripted stand-in for the model. tinyshop's new files are real Python: the payment
tests are executed against the endpoint Otto wrote, so "1 failed, 14 passed" and "15 passed" are computed.

Arc: hook (a flat to-do list has no order and dies with the session; "A TODO list is not a task system")
· a task is a file (the six fields, status pending → in_progress → completed) · blockers make a graph
(can_start) · claim and complete (the graph lights up; a premature claim refused) · the board (badges, not
arrows) · done is a claim (Otto skips the tests, the harness runs them, the claim is rejected, fixed,
accepted) · the three lines in complete_task · loop engineering (Li, ch. 10) · the verifier rule · long-
running agents with handover (Anthropic) · recap: block, claim, verify.

Sources (facts dated October 2026):
- Learn Claude Code, s11 "What's Next": "A TODO list is not a task system." s12 (learn.shareai.run/en/s12/):
  Task(id, subject, description, status pending|in_progress|completed, owner, blockedBy), stored as
  .tasks/{id}.json; can_start (every blocker completed; "missing dependency = blocked"); claim_task (sets
  owner and in_progress; else "Blocked by: {deps}"); complete_task ("Completed {id} ({subject})" plus
  "Unblocked: …"; we list only this task's dependents, as the page's comment says, "newly unblocked
  downstream tasks" — its filter lists every pending task that can start); example DAG schema → endpoints → tests, schema → docs; the board Waiting / Ready /
  Working / Done, "A dependency is not an arrow students must trace. It is a visible blocker badge on the
  card." The code below follows the page's teaching version.
- Li, AI Agents in Depth, ch. 10 (loop engineering): "until it is verified, 'done' is merely the model's
  claim, not proof"; three forms: partial completion, giving up too soon, false success; "the bottleneck of
  the loop is the verifier, not the model"; LoopX: "the model may propose 'done,' but it cannot approve its
  own 'done.'" ch. 5 (coding workflow): "the most common way a Coding Agent slacks off is to skip" testing.
  ch. 1: Anthropic's long-running agents as an Initialization Agent plus an Execution Agent making
  "incremental progress each session and leaving clear handover artifacts".
- Anthropic, "Effective harnesses for long-running agents" (engineering blog, Nov 26, 2025): an initializer
  session sets up the environment and a feature list whose items start failing ("passes": false); each
  later session makes incremental progress, tests, and leaves structured updates.

Illustrative (for teaching): the five tasks and their descriptions, Otto's choices (including the premature
claim), the contents of payments.py, api.py, docs and tests, tinyshop's other 13 tests (as in Episodes 2, 9
and 14: "13 passed"), the PR command and its output, using `pytest -q` as the one check for every task, the
long-running timeline (eight tasks, three sessions). Computed and asserted below: every tool result shown,
the test counts, the refused claim, what each completion unblocked, the order of the run.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import importlib.abc
import importlib.util
import json
import re
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

HERE = Path(__file__).resolve().read_text()
REF = (SERIES / "harness_reference.py").read_text()

# ---------------------------------------------------------------------------- tinyshop, in memory

GOAL = "Add payments to tinyshop."
OWNER = "otto"
OTHER_TESTS = 13  # tinyshop's existing tests, all passing (Episodes 2, 9, 14)

PAYMENTS_PY = '''from dataclasses import dataclass


@dataclass
class Payment:
    order_id: int
    cents: int
    status: str
'''
API_PY = '''from payments import Payment
from web import app


@app.post("/pay")
def pay(order_id, cents, card):
    reply = card.charge(cents)
    return Payment(order_id, cents, "paid")
'''
API_FIX = ('"paid")', '"paid" if reply.ok else "failed")')
TEST_PAY_PY = '''from api import pay
from fakes import Card


def test_pay_ok():
    assert pay(1, 1799, Card(ok=True)).status == "paid"


def test_pay_declined():
    assert pay(2, 1799, Card(ok=False)).status == "failed"
'''
DOCS_MD = "# Payments\n\nPOST /pay charges a card, in cents.\n"
WEB_PY = '''class App:
    def post(self, path):
        return lambda handler: handler


app = App()
'''
FAKES_PY = '''class Card:
    def __init__(self, ok):
        self.ok = ok

    def charge(self, cents):
        return self
'''
DISK = {
    "cart.py": "def total(items):\n    return sum(items)\n",
    "checkout.py": "from cart import total\n",
    "web.py": WEB_PY,
    "tests/fakes.py": FAKES_PY,
}
MODULE_FILES = {"payments": "payments.py", "api": "api.py", "web": "web.py", "fakes": "tests/fakes.py"}


class DiskImporter(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Lets `import payments` find the module in the in-memory repo."""

    def find_spec(self, name, path=None, target=None):
        if MODULE_FILES.get(name) in DISK:
            return importlib.util.spec_from_loader(name, self)
        return None

    def create_module(self, spec):
        return None

    def exec_module(self, module):
        path = MODULE_FILES[module.__name__]
        exec(compile(DISK[path], path, "exec"), module.__dict__)


class repo_imports:
    def __enter__(self):
        for name in MODULE_FILES:
            sys.modules.pop(name, None)
        self.finder = DiskImporter()
        sys.meta_path.insert(0, self.finder)

    def __exit__(self, *exc):
        sys.meta_path.remove(self.finder)
        for name in MODULE_FILES:
            sys.modules.pop(name, None)


def run_python(path):
    with repo_imports():
        try:
            exec(compile(DISK[path], path, "exec"), {"__name__": "__main__"})
            return ""
        except Exception as error:  # noqa: BLE001
            return f"{type(error).__name__}: {error}"


# the hook: with only a flat list, nothing stops the endpoint being written before the model it imports
DISK["api.py"] = API_PY
HOOK_ERROR = run_python("api.py")
del DISK["api.py"]
assert HOOK_ERROR == "ModuleNotFoundError: No module named 'payments'", HOOK_ERROR

PYTEST_RUNS = []


def pytest_q():
    """tinyshop's suite: 13 passing tests, plus tests/test_pay.py run for real against api.py on disk."""
    failed, passed = [], OTHER_TESTS
    if "tests/test_pay.py" in DISK:
        with repo_imports():
            ns = {"__name__": "test_pay"}
            exec(compile(DISK["tests/test_pay.py"], "tests/test_pay.py", "exec"), ns)
            for name, test in list(ns.items()):
                if name.startswith("test_") and callable(test):
                    try:
                        test()
                        passed += 1
                    except AssertionError:
                        failed.append(name)
    out = "".join(f"FAILED tests/test_pay.py::{name}\n" for name in failed)
    out += f"{len(failed)} failed, {passed} passed" if failed else f"{passed} passed"
    PYTEST_RUNS.append(out)
    return out


BASH_CALLS = []


def run_bash(command):
    BASH_CALLS.append(command)
    if command == "pytest -q":
        return pytest_q()
    if command == "gh pr create --fill":
        return "opened pull request #7"
    raise ValueError(command)


def read_file(path):
    return DISK[path]


def edit_file(path, old_text, new_text):
    assert DISK[path].count(old_text) == 1, (path, old_text)
    DISK[path] = DISK[path].replace(old_text, new_text, 1)
    return f"edited {path}"


def write_file(path, content):  # Episode 14's tool
    DISK[path] = content
    return f"wrote {path}"


# ---------------------------------------------------------------------------- the task system (s12)


@dataclass
class Task:  # stored as .tasks/{id}.json
    id: str
    subject: str
    description: str
    status: str = "pending"  # pending | in_progress | completed
    owner: str | None = None
    blockedBy: list = field(default_factory=list)


def _task_path(task_id):
    return f".tasks/{task_id}.json"


TASK_WRITES = []  # every save, in order (for the scene: the files as they were written)


def save_task(task):
    DISK[_task_path(task.id)] = json.dumps(asdict(task))
    TASK_WRITES.append((task.id, DISK[_task_path(task.id)]))


def load_task(task_id):
    return Task(**json.loads(DISK[_task_path(task_id)]))


def list_tasks():
    ids = sorted((p[len(".tasks/"):-len(".json")] for p in DISK if p.startswith(".tasks/")), key=int)
    return [load_task(i) for i in ids]


def create_task(subject, description="", blockedBy=None):
    task = Task(str(len(list_tasks()) + 1), subject, description, blockedBy=list(blockedBy or []))
    save_task(task)
    return task


def can_start(task_id):
    task = load_task(task_id)
    for dep_id in task.blockedBy:
        if _task_path(dep_id) not in DISK:
            return False  # missing dependency = blocked
        if load_task(dep_id).status != "completed":
            return False
    return True


def claim_task(task_id, owner="agent"):
    task = load_task(task_id)
    if task.status != "pending":
        return f"Task {task_id} is {task.status}, cannot claim"
    if not can_start(task_id):
        deps = [d for d in task.blockedBy if load_task(d).status != "completed"]
        return f"Blocked by: {deps}"
    task.owner = owner
    task.status = "in_progress"
    save_task(task)
    return f"Claimed {task_id} ({task.subject})"


# ---- what Episode 22 adds: complete_task runs the check before it accepts "done" ----
def complete_task(task_id):
    output = run_bash("pytest -q")      # the harness checks
    if "failed" in output:
        return f"Rejected: {output}"    # stays in_progress
    task = load_task(task_id)
    task.status = "completed"
    save_task(task)
    unblocked = [t.subject for t in list_tasks()  # newly unblocked downstream tasks
                 if t.status == "pending" and task_id in t.blockedBy
                 and can_start(t.id)]
    msg = f"Completed {task_id} ({task.subject})"
    if unblocked:
        msg += f"\nUnblocked: {', '.join(unblocked)}"
    return msg
# ---- end of what Episode 22 adds ----


COMPLETE_SRC = '''def complete_task(task_id):
    output = run_bash("pytest -q")      # the harness checks
    if "failed" in output:
        return f"Rejected: {output}"    # stays in_progress
    task = load_task(task_id)
    task.status = "completed"
    save_task(task)
    ...  # then report what got unblocked'''
_added = HERE.split("# ---- what Episode 22 adds")[1].split("# ---- end of what Episode 22 adds")[0]
assert all(line in _added for line in COMPLETE_SRC.split("\n")[:-1]) and len(COMPLETE_SRC.split("\n")) <= 10

CAN_START_SRC = '''def can_start(task_id):
    for dep_id in load_task(task_id).blockedBy:
        if load_task(dep_id).status != "completed":
            return False  # one open blocker is enough
    return True'''
assert 'if load_task(dep_id).status != "completed":' in HERE.split("def can_start")[1].split("def claim_task")[0]


def create_task_tool(subject, description="", blockedBy=None):
    task = create_task(subject, description, blockedBy)
    return f"Created {task.id} ({task.subject})"


def list_tasks_tool():
    return "\n".join(f"{t.id} {t.status} {t.subject}" for t in list_tasks())


# ---------------------------------------------------------------------------- the harness (reference names)

SYSTEM = re.search(r'^SYSTEM = "(.*)"$', REF, re.M).group(1)
assert SYSTEM.startswith("You are Otto")
assert 'TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}' in REF
assert "def agent_loop(messages):" in REF and 'results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})' in REF
TOOLS = [{"name": n} for n in ("bash", "read_file", "edit_file", "write_file",
                               "create_task", "list_tasks", "claim_task", "complete_task")]  # schemas abridged
TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file, "write_file": write_file,
                 "create_task": create_task_tool, "list_tasks": list_tasks_tool,
                 "claim_task": claim_task, "complete_task": complete_task}


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


IDS = iter(range(1, 200))
SNAPSHOTS = []  # task states after every tool call: {id: (status, owner)}


class ScriptedModel:
    """Stands in for client.messages: follows Otto's plan (illustrative)."""

    def __init__(self, plan, final):
        self.plan, self.final, self.calls = plan, final, 0

    def create(self, system, messages, tools):
        if self.calls:
            SNAPSHOTS.append({t.id: (t.status, t.owner) for t in list_tasks()})
        step = self.plan[self.calls]
        self.calls += 1
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=self.final)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{next(IDS):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=None)


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


TASKS = [  # subject, description, blockedBy (illustrative)
    ("payment model", "a Payment record", []),
    ("POST /pay", "charge a card", ["1"]),
    ("payment docs", "how payments work", ["1"]),
    ("payment tests", "paid and declined", ["2"]),
    ("pull request", "open a PR for review", ["3", "4"]),
]
SUBJECT = {str(i + 1): s for i, (s, _, _) in enumerate(TASKS)}


def claim(i):
    return ("claim_task", {"task_id": i, "owner": OWNER})


def complete(i):
    return ("complete_task", {"task_id": i})


PLAN = [("create_task", {"subject": s, "description": d, "blockedBy": b}) for s, d, b in TASKS] + [
    claim("5"),  # Otto jumps ahead: refused
    claim("1"), ("write_file", {"path": "payments.py", "content": PAYMENTS_PY}), complete("1"),
    claim("2"), ("write_file", {"path": "api.py", "content": API_PY}), complete("2"),
    claim("3"), ("write_file", {"path": "docs/payments.md", "content": DOCS_MD}), complete("3"),
    claim("4"), ("write_file", {"path": "tests/test_pay.py", "content": TEST_PAY_PY}), complete("4"),
    ("edit_file", {"path": "api.py", "old_text": API_FIX[0], "new_text": API_FIX[1]}), complete("4"),
    claim("5"), ("bash", {"command": "gh pr create --fill"}), complete("5"),
    None,
]
FINAL = "Payments are in: 5 tasks completed, 15 passed."
MESSAGES = [{"role": "user", "content": GOAL}]
client.messages = MODEL = ScriptedModel(PLAN, FINAL)
ANSWER = agent_loop(MESSAGES)

CALLS = [b for msg in MESSAGES if msg["role"] == "assistant" for b in msg["content"] if getattr(b, "type", "") == "tool_use"]
RESULTS = [b["content"] for msg in MESSAGES if msg["role"] == "user" and not isinstance(msg["content"], str) for b in msg["content"]]
TRACE = list(zip(CALLS, RESULTS))
assert len(CALLS) == len(RESULTS) == len(SNAPSHOTS) == len(PLAN) - 1


def find(name, task_id=None, nth=0):
    """Index into TRACE of the nth call to `name` (for task_id)."""
    hits = [k for k, (c, _) in enumerate(TRACE) if c.name == name and (task_id is None or c.input.get("task_id") == task_id)]
    return hits[nth]


def result(name, task_id=None, nth=0):
    return TRACE[find(name, task_id, nth)][1]


def state_after(k):
    return SNAPSHOTS[k]


# the run, as narrated
assert [r for c, r in TRACE if c.name == "create_task"] == [f"Created {i + 1} ({s})" for i, (s, _, _) in enumerate(TASKS)]
assert len(TASKS) == 5  # "five steps", "five create task calls"
K_REFUSED = find("claim_task", "5", 0)
REFUSED = TRACE[K_REFUSED][1]
assert REFUSED == "Blocked by: ['3', '4']" and all(v == ("pending", None) for v in state_after(K_REFUSED).values())
K_CLAIM1 = find("claim_task", "1")
assert TRACE[K_CLAIM1][1] == "Claimed 1 (payment model)" and state_after(K_CLAIM1)["1"] == ("in_progress", OWNER)
DONE1 = result("complete_task", "1")
assert DONE1 == "Completed 1 (payment model)\nUnblocked: POST /pay, payment docs"
DONE2 = result("complete_task", "2")
assert DONE2 == "Completed 2 (POST /pay)\nUnblocked: payment tests"
DONE3 = result("complete_task", "3")
assert DONE3 == "Completed 3 (payment docs)"  # nothing new: the pull request still waits on 4
K_DONE3 = find("complete_task", "3")
assert state_after(K_DONE3)["5"] == ("pending", None) and state_after(K_DONE3)["4"] == ("pending", None)
K_REJECT = find("complete_task", "4", 0)
REJECTED = TRACE[K_REJECT][1]
assert REJECTED == "Rejected: FAILED tests/test_pay.py::test_pay_declined\n1 failed, 14 passed", REJECTED
assert state_after(K_REJECT)["4"] == ("in_progress", OWNER)  # the claim was rejected: still in progress
DONE4 = result("complete_task", "4", 1)
assert DONE4 == "Completed 4 (payment tests)\nUnblocked: pull request"
DONE5 = result("complete_task", "5")
assert DONE5 == "Completed 5 (pull request)"
assert all(v == ("completed", OWNER) for v in SNAPSHOTS[-1].values())  # all five, verified
# Otto never ran the tests itself: every pytest run came from complete_task (one per completion attempt)
assert "pytest -q" not in [c.input.get("command") for c in CALLS if c.name == "bash"]
COMPLETES = [k for k, (c, _) in enumerate(TRACE) if c.name == "complete_task"]
assert len(PYTEST_RUNS) == len(COMPLETES) == 6
CHECK = {TRACE[k][0].input["task_id"] + ("" if n == 0 else f"#{n}"): out
         for n_k, (k, out) in enumerate(zip(COMPLETES, PYTEST_RUNS))
         for n in [sum(TRACE[j][0].input["task_id"] == TRACE[k][0].input["task_id"] for j in COMPLETES[:n_k])]}
assert CHECK["1"] == CHECK["2"] == CHECK["3"] == f"{OTHER_TESTS} passed"  # the endpoint passed: no test for declined cards yet
assert CHECK["4"] == "FAILED tests/test_pay.py::test_pay_declined\n1 failed, 14 passed" and CHECK["4#1"] == CHECK["5"] == "15 passed"
N_PAY_TESTS = TEST_PAY_PY.count("def test_")
assert N_PAY_TESTS == 2 and OTHER_TESTS + N_PAY_TESTS == 15  # "two payment tests", "fifteen passed"
assert ANSWER.content[0].text == FINAL
EDIT = TRACE[find("edit_file")]
assert EDIT[0].input["path"] == "api.py" and "reply.ok" in DISK["api.py"]  # Otto fixed the endpoint, not the test
assert DISK["tests/test_pay.py"] == TEST_PAY_PY

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
PANEL = "#18150F"
ROW_H = 0.48
ROW_W, ROW_X = 12.2, 0.48  # the call and result rows under the graph and the board
BOT_XY = (-6.15, -2.84, 0)  # Otto, beside them
LOCK = recolored("lock", ROSE, ROSE_FILL, prefix="ep22")
STYLE = {  # task state → (stroke, fill, accent)
    "pending": (EDGE, CARD, MUTED),
    "ready": (AZURE, AZURE_FILL, AZURE),
    "in_progress": (SUN, SUN_FILL, SUN),
    "completed": (MINT, MINT_FILL, MINT),
    "rejected": (ROSE, ROSE_FILL, ROSE),
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


def row(role, body, width):
    """One message on one line: a stripe in the role's colour, the role, the content (as Episodes 9, 14)."""
    accent = ROLES[role][0]
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0, fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, accent, weight=m.BOLD)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    if body.width > width - 2.13:
        raise ValueError(f"row body too wide ({body.text!r}: {body.width:.2f} > {width - 2.13:.2f})")
    body.move_to(box).align_to(box, m.LEFT).shift(1.98 * m.RIGHT)
    base = box.get_y() - 0.09
    sit(label, base)
    sit(body, base)
    return m.VGroup(box, stripe, label, body)


def call_src(block):
    a = block.input
    if block.name in ("claim_task", "complete_task"):
        return f'{block.name}("{a["task_id"]}")'
    if block.name in ("write_file", "edit_file"):
        return f'{block.name}("{a["path"]}", …)'
    if block.name == "create_task":
        return f'create_task("{a["subject"]}", …)'
    return f'bash("{a["command"]}")'


def shown(content):
    """A tool result on one row: lines joined with " · ", the test path shortened."""
    return content.replace("FAILED tests/test_pay.py::", "FAILED ").replace("\n", " · ")


class Slot:
    """One place on screen whose content is swapped (a row, a caption)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15, source=None):
        on_screen = {id(x) for x in self.scene.get_mobject_family_members()}
        anims = []
        if self.current is not None:
            out = m.FadeOut(self.current, shift=shift * m.UP)
            if id(self.current) not in on_screen:
                out.introducer = True
            anims.append(out)
        if new is not None:
            anims.append(m.FadeIn(new, target_position=source, scale=0.6) if source is not None else m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


def word_chip(word, color, fill, size=24, height=0.66, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def mono_chip(word, color, fill, size=24, height=0.66, pad=0.5):
    return chip(word, color, fill, width=code(word, size - 4).width + pad, height=height, size=size, mono=True)


def strike(words, color=ROSE):
    return m.Line(words.get_left() + 0.08 * m.LEFT, words.get_right() + 0.08 * m.RIGHT).set_stroke(color, 3).move_to(words)


def highlight(block, src, i, color=SUN, pad_l=0.12, pad_r=0.1):
    """A translucent bar behind line i of a code_block (box = block[0], text = block[1])."""
    lines = src.split("\n")
    g = line_glyphs(block[1], i)
    base = baseline_of(g, [c for c in lines[i] if not c.isspace()])
    top, bot = base + 0.29, base - 0.11
    left, right = block[0].get_left()[0] + pad_l, block[0].get_right()[0] - pad_r
    rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.14)
    mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
    rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
    mark.move_to([left + 0.025, (top + bot) / 2, 0])
    return m.VGroup(rect, mark).set_z_index(1)


# ---- the graph (sections 2-3) ----
GRAPH_POS = {"1": (-4.3, 1.6), "2": (0.0, 1.6), "4": (4.3, 1.6), "3": (-4.3, -0.4), "5": (4.3, -0.4)}
NODE_W, NODE_H = 3.4, 1.0
EDGES = [("1", "2"), ("1", "3"), ("2", "4"), ("4", "5"), ("3", "5")]
assert sorted(EDGES) == sorted((b, str(i + 1)) for i, (_, _, bs) in enumerate(TASKS) for b in bs)  # arrows = blockedBy


def status_words(state):
    return {"pending": "pending", "ready": "pending · ready", "in_progress": f"in_progress · {OWNER}",
            "completed": "completed"}[state]


def gnode(tid, state):
    stroke, fill, accent = STYLE[state]
    box = card(NODE_W, NODE_H, stroke, fill, radius=0.18, stroke=2.5).move_to([*GRAPH_POS[tid], 0])
    num = code(f"#{tid}", 22, FAINT if state == "pending" else accent, weight=m.BOLD)
    title = text(SUBJECT[tid], 24, INK, weight=m.SEMIBOLD)
    status = code(status_words(state), 20, accent)
    left = box.get_left()[0] + 0.28
    num.align_to([left, 0, 0], m.LEFT)
    title.next_to(num, m.RIGHT, buff=0.18)
    status.align_to([left, 0, 0], m.LEFT)
    y = box.get_y()
    sit(num, y + 0.06)
    sit(title, y + 0.06)
    sit(status, y - 0.32)
    return m.VGroup(box, num, title, status)


def gedge(a, b):
    pa, pb = np.array([*GRAPH_POS[a], 0]), np.array([*GRAPH_POS[b], 0])
    if abs(pa[1] - pb[1]) < 1e-6:  # same row: side to side
        start, end = pa + [NODE_W / 2, 0, 0], pb - [NODE_W / 2, 0, 0]
    else:  # same column: top to bottom
        start, end = pa - [0, NODE_H / 2, 0], pb + [0, NODE_H / 2, 0]
    return arrow(start, end, MUTED, buff=0.08)


# ---- the board (sections 4-5) ----
COLS = ["waiting", "ready", "working", "done"]
COL_W = {"waiting": 2.95, "ready": 2.95, "working": 3.05, "done": 3.3}
COL_X = {"waiting": -5.05, "ready": -1.95, "working": 1.2, "done": 4.875}
COL_COLOR = {"waiting": MUTED, "ready": AZURE, "working": SUN, "done": MINT}
BOARD_TOP, BOARD_BOT = 2.5, -2.05
GATE_X = 2.975
CARD_GAP = 0.08
assert abs((COL_X["working"] + COL_W["working"] / 2 + COL_X["done"] - COL_W["done"] / 2) / 2 - GATE_X) < 1e-6
assert COL_X["waiting"] - COL_W["waiting"] / 2 >= -6.61 and COL_X["done"] + COL_W["done"] / 2 <= 6.61


def bcard(tid, state, blockers=(), col=None):
    """A task card: id and subject; a second line for a blocker badge (waiting) or the owner (working)."""
    stroke, fill, accent = STYLE[state]
    col = col or {"pending": "waiting", "ready": "ready", "in_progress": "working", "rejected": "working", "completed": "done"}[state]
    w = COL_W[col] - 0.18
    two = bool(blockers) or state in ("in_progress", "rejected")
    h = 0.92 if two else 0.56
    box = card(w, h, stroke, fill, radius=0.14, stroke=2)
    num = code(tid, 22, FAINT if state == "pending" else accent, weight=m.BOLD)
    title = text(SUBJECT[tid], 22, INK, weight=m.SEMIBOLD)
    left = box.get_left()[0] + 0.22
    num.align_to([left, 0, 0], m.LEFT)
    title.next_to(num, m.RIGHT, buff=0.16)
    top_base = box.get_y() + (0.12 if two else -0.09)
    sit(num, top_base)
    sit(title, top_base)
    parts = [box, num, title]
    if blockers:
        lock = icon(LOCK, 0.26)
        ids = code(" ".join(blockers), 20, ROSE, weight=m.BOLD)
        inner = m.VGroup(lock, ids).arrange(m.RIGHT, buff=0.1)
        badge_box = card(inner.width + 0.3, 0.34, ROSE, ROSE_FILL, radius=0.17, stroke=1.5)
        badge = m.VGroup(badge_box, inner.move_to(badge_box))
        badge.align_to([left, 0, 0], m.LEFT).set_y(box.get_y() - 0.22)
        sit(ids, box.get_y() - 0.29)
        parts.append(badge)
    elif two:
        owner = code(f"owner: {OWNER}", 20, accent)
        owner.align_to([left, 0, 0], m.LEFT)
        sit(owner, box.get_y() - 0.3)
        parts.append(owner)
    return m.VGroup(*parts)


class Board:
    """Four columns; cards are placed top-down in each. place(card, col, k) puts a card in slot k."""

    def __init__(self):
        self.columns = {}
        self.heads = {}
        for c in COLS:
            panel = card(COL_W[c], BOARD_TOP - BOARD_BOT, EDGE, PANEL, radius=0.18, stroke=1.5)
            panel.move_to([COL_X[c], (BOARD_TOP + BOARD_BOT) / 2, 0]).set_z_index(-1)
            head = code(c, 22, COL_COLOR[c], weight=m.BOLD)
            head.align_to(panel, m.LEFT).shift(0.25 * m.RIGHT)
            sit(head, BOARD_TOP - 0.38)
            self.columns[c], self.heads[c] = panel, head
        self.gate = m.DashedLine([GATE_X, BOARD_TOP - 0.05, 0], [GATE_X, BOARD_BOT + 0.05, 0], dash_length=0.12).set_stroke(SUN, 4)
        self.check = self.check_label("check: pytest -q", SUN)
        self.top = {c: BOARD_TOP - 0.62 for c in COLS}
        self.top["done"] = BOARD_TOP - 1.02

    def check_label(self, s, color):
        words = code(s, 20, color, weight=m.BOLD)
        words.align_to(self.columns["done"], m.LEFT).shift(0.25 * m.RIGHT)
        return sit(words, BOARD_TOP - 0.8)

    def frame(self):
        return m.VGroup(*self.columns.values(), *self.heads.values(), self.gate, self.check)

    def place(self, card_, col, above=()):
        """Move a card to its column, below the cards `above` (their heights)."""
        top = self.top[col] - sum(h + CARD_GAP for h in above)
        card_.move_to([COL_X[col], top - card_.height / 2, 0])
        return card_


# ---------------------------------------------------------------------------- the episode


class Ep22TaskGraphs(Episode):
    number = 22

    def lesson(self):
        self.hook()
        self.tasks_as_files()
        self.blockers()
        self.claim_complete()
        self.the_board()
        self.verify()
        self.the_check()
        self.loop_engineering()
        self.the_verifier()
        self.long_running()
        self.recap()

    # helpers

    def speak(self, script, *animations):
        """say(), with every animation of an object not yet on screen marked as an introducer: a row
        swapped in and then out (or a node transformed twice) in the same line must not show early."""
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

    def bottom(self):
        """Otto and the two rows of the latest call and its result (sections 3-5)."""
        return otto(0.55).move_to(BOT_XY)

    def call_row(self, block):
        r = row("assistant", code(call_src(block), 20, SUN), ROW_W)
        return r.move_to([ROW_X, -2.55, 0])

    def result_row(self, content):
        color = ROSE if content.startswith(("Rejected", "Blocked")) else MINT
        r = row("user", code(shown(content), 20, color), ROW_W)
        return r.move_to([ROW_X, -3.13, 0])

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Otto is asked: {GOAL!r}. Five steps (illustrative): {', '.join(SUBJECT.values())}. Episode 9's to-do list is a flat checklist kept in the session; it has no order, so the endpoint could be written before the model it imports (real Python, run in the scene: {HOOK_ERROR!r}), and it is gone when the session ends. Learn Claude Code, s11 'What's Next': 'A TODO list is not a task system.' Facts as of October 2026.")
        bot = otto(1.0).move_to([-5.4, 1.75, 0])
        person = icon("user", 0.9).move_to([5.8, 1.55, 0])
        ask = bubble(text(GOAL, 28, INK), AZURE, AZURE_FILL, tail="right")
        ask.move_to([2.4, 2.0, 0])
        ask.shift((person.get_left()[0] - 0.15 - ask[0].get_vertices()[2][0]) * m.RIGHT)

        chips = m.VGroup(*[word_chip(SUBJECT[t], EDGE, CARD, size=22, height=0.62, pad=0.5) for t in SUBJECT])
        chips.arrange(m.RIGHT, buff=0.2).move_to([0, 0.45, 0])
        five = text(f"{len(TASKS)} steps", 22, MUTED).next_to(chips, m.DOWN, buff=0.16)

        panel = card(5.2, 2.75, EDGE, PANEL, radius=0.18, stroke=2).move_to([-3.45, -1.6, 0]).set_z_index(-1)
        p_cap = code("context", 20, MUTED).next_to(panel, m.UP, buff=0.12).align_to(panel, m.LEFT)
        p_ep = text("the to-do list of Episode 9", 20, MUTED).next_to(panel, m.UP, buff=0.12).align_to(panel, m.RIGHT)
        sit(p_ep, baseline(p_cap))
        todo_src = [f"[ ] {SUBJECT[t]}" for t in SUBJECT]
        todos = m.VGroup(*[code(s, 22, INK, t2c={r"^\[ \]": FAINT}) for s in todo_src])
        todos.arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT).move_to(panel).align_to(panel, m.LEFT).shift(0.45 * m.RIGHT)
        jumped = code(f"[▸] {SUBJECT['2']}", 22, SUN, weight=m.BOLD).align_to(todos[1], m.LEFT)
        sit(jumped, baseline(todos[1]))
        term = terminal(["$ python api.py", "! " + HOOK_ERROR.split(": ")[0] + ":", "! " + HOOK_ERROR.split(": ")[1]], width=5.2)
        term.move_to([3.25, -1.5, 0])
        ended = pill("session ends", ROSE, ROSE_FILL, 22).move_to(panel)
        statement = title_text("A to-do list is not a task system.", 46, INK).move_to([0, -1.45, 0])
        src = cite("Learn Claude Code · s11").next_to(statement, m.DOWN, buff=0.3)

        self.speak(
            "[Say you ask Otto] [to add payments to tinyshop.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.3 * m.UP), m.FadeIn(person, shift=0.2 * m.UP), lag_ratio=0.3),
            m.FadeIn(ask, shift=0.2 * m.UP),
        )
        self.speak(
            "[That's really five steps]: [a payment model, an endpoint, docs, tests, and a pull request.]",
            m.FadeIn(five),
            m.LaggedStart(*[m.FadeIn(c, shift=0.2 * m.UP) for c in chips], lag_ratio=0.35),
        )
        self.speak(
            "[In episode nine, Otto kept a to-do list]: [a flat checklist, in its context.]",
            m.AnimationGroup(m.FadeIn(panel), m.FadeIn(p_cap), m.FadeIn(p_ep), m.FadeOut(five)),
            m.LaggedStart(*[m.ReplacementTransform(c, t) for c, t in zip(chips, todos)], lag_ratio=0.12),
        )
        self.speak(
            "[But a flat list has no order.] [Otto could start the endpoint] [before the model it imports even exists.]",
            m.Circumscribe(todos, color=MUTED, buff=0.12),
            m.ReplacementTransform(todos[1], jumped),
            m.FadeIn(term, shift=0.2 * m.UP),
        )
        rest = [t for i, t in enumerate(todos) if i != 1] + [jumped]
        self.speak(
            "[And when the session ends], [the list is gone, along with the context.]",
            m.FadeIn(ended, scale=1.1),
            m.LaggedStart(*[m.FadeOut(t, shift=0.3 * m.DOWN) for t in rest], lag_ratio=0.08),
        )
        self.speak(
            "[As Learn Claude Code puts it]: [a to-do list is not a task system.]",
            m.AnimationGroup(*[m.FadeOut(x) for x in (panel, p_cap, p_ep, ended, term)], m.FadeIn(src)),
            m.Write(statement),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · a task is a file

    def tasks_as_files(self):
        t5 = json.loads(next(j for i, j in TASK_WRITES if i == "5"))  # .tasks/5.json as created
        assert t5 == {"id": "5", "subject": "pull request", "description": TASKS[4][1], "status": "pending", "owner": None, "blockedBy": ["3", "4"]}
        self.next_section("1 · A task is a file", notes=f"Learn Claude Code, s12: every task is a record, stored as .tasks/{{id}}.json, with id, subject, description, status (pending | in_progress | completed), owner and blockedBy. Otto files the five steps with five create_task calls (results asserted: {[r for c, r in TRACE if c.name == 'create_task']}). Shown: .tasks/5.json as created, {t5}. Because the files live on disk, not in the context, they outlast the session (Episode 14). Facts as of October 2026.")
        bot = otto(0.7).move_to([-5.75, 1.95, 0])
        creates = mono_chip("create_task × 5", SUN, SUN_FILL, size=24, height=0.6)
        creates.next_to(bot, m.RIGHT, buff=0.35)
        entries = [".tasks/"] + [f".tasks/{t}.json" for t in SUBJECT]
        tree = file_tree(entries, size=22, highlight=(".tasks/5.json",)).move_to([-5.2, -0.65, 0]).align_to([-6.4, 0, 0], m.LEFT)

        src = (
            "{\n"
            f'  "id": "{t5["id"]}",\n'
            f'  "subject": "{t5["subject"]}",\n'
            f'  "description": "{t5["description"]}",\n'
            f'  "status": "{t5["status"]}",\n'
            '  "owner": null,\n'
            f'  "blockedBy": {json.dumps(t5["blockedBy"])}\n'
            "}"
        )
        assert json.loads(src) == t5
        body = code(src, 22, INK, t2c={r"\"[A-Za-z]+\"(?=:)": AZURE, r"(?<=: )\"[^\"]*\"": MINT, r"\bnull\b": ORCHID, r"\[\"[^\]]*\]": MINT})
        cap = code(".tasks/5.json", 20, SUN, weight=m.BOLD)
        box = card(body.width + 0.8, body.height + 0.95, SUN, CARD, radius=0.16, stroke=2)
        jc = m.VGroup(box, body, cap).move_to([-0.4, 0.4, 0])
        box.move_to(jc)
        body.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT + 0.2 * m.DOWN)
        cap.align_to(box, m.UL).shift(0.4 * m.RIGHT + 0.2 * m.DOWN)
        lines = src.split("\n")
        k = {key: next(i for i, l in enumerate(lines) if f'"{key}"' in l) for key in ("subject", "status", "owner", "blockedBy")}
        bars = {key: highlight(jc, src, i, AZURE, pad_l=0.14, pad_r=0.14) for key, i in k.items()}
        notes_ = {"subject": "what", "status": "where it stands", "owner": "who", "blockedBy": "what it waits for"}
        labels = {}
        for key, i in k.items():
            g = line_glyphs(body, i)
            lab = text(notes_[key], 22, AZURE, weight=m.SEMIBOLD).next_to(box, m.RIGHT, buff=0.3)
            sit(lab, baseline_of(g, [c for c in lines[i] if not c.isspace()]))
            labels[key] = lab
        assert max(l.get_right()[0] for l in labels.values()) < 6.6

        states = m.VGroup(*[mono_chip(s, c, f, size=24, height=0.62) for s, c, f in
                            (("pending", MUTED, CARD), ("in_progress", SUN, SUN_FILL), ("completed", MINT, MINT_FILL))])
        states.arrange(m.RIGHT, buff=1.0).move_to([0.6, -2.75, 0])
        st_arrows = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.12) for a, b in zip(states, states[1:])])
        outlast = text("on disk: it outlasts the session · Episode 14", 20, MUTED)
        outlast.next_to(tree, m.DOWN, buff=0.3).align_to(tree, m.LEFT)
        src_cite = cite("Learn Claude Code · s12").next_to(states, m.LEFT, buff=0.6).align_to([-6.4, 0, 0], m.LEFT)
        sit(src_cite, baseline(states[0][1]))

        self.speak(
            "[A task system starts by filing every step as a task]: [five create task calls], [five small files.]",
            m.AnimationGroup(self.new_header("01 · tasks", "A task is a file"), m.FadeIn(bot, shift=0.2 * m.UP)),
            m.FadeIn(creates, target_position=bot, scale=0.5),
            m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in tree], lag_ratio=0.15),
        )
        self.speak(
            "[Each file is a record]: [a subject], [a status], [an owner], [and the tasks that block it.]",
            m.AnimationGroup(m.FadeOut(creates), m.FadeIn(jc, target_position=tree[-1], scale=0.4), m.FadeIn(src_cite)),
            m.AnimationGroup(m.FadeIn(bars["subject"]), m.FadeIn(labels["subject"], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeOut(bars["subject"]), m.FadeIn(bars["status"]), m.FadeIn(labels["status"], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeOut(bars["status"]), m.FadeIn(bars["owner"]), m.FadeIn(labels["owner"], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeOut(bars["owner"]), m.FadeIn(bars["blockedBy"]), m.FadeIn(labels["blockedBy"], shift=0.1 * m.LEFT)),
        )
        self.speak(
            "[Status only moves forward]: [pending], [in progress], [completed.]",
            m.AnimationGroup(m.FadeOut(bars["blockedBy"]), m.Indicate(labels["status"], color=AZURE, scale_factor=1.06)),
            m.FadeIn(states[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowArrow(st_arrows[0]), m.FadeIn(states[1], shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(st_arrows[1]), m.FadeIn(states[2], shift=0.15 * m.UP)),
        )
        self.speak(
            "[And since the files live on disk, not in the context], [they outlast the session.]",
            m.Circumscribe(tree, color=ORCHID, buff=0.12),
            m.FadeIn(outlast, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · blockers make a graph

    def blockers(self):
        self.next_section("2 · Blockers make a graph", notes=f"Each blockedBy entry drawn as an arrow from blocker to blocked: {EDGES} (asserted equal to the tasks' blockedBy lists). The shape follows Learn Claude Code s12's example (schema → endpoints → tests, schema → docs), for tinyshop, plus a pull request that waits for tests and docs. can_start, from s12 (simplified; the page also treats a missing blocker as blocked): a task can start only when every blocker is completed. Ready is not stored: it is pending plus can_start. Only task 1 is ready at the start. Facts as of October 2026.")
        self.nodes = {t: gnode(t, "pending") for t in SUBJECT}
        self.edges = {(a, b): gedge(a, b) for a, b in EDGES}
        cs = code_block(CAN_START_SRC, size=20)
        cs.move_to([0, -2.3, 0]).align_to([NODE_W / 2 - 4.3 - NODE_W, 0, 0], m.LEFT)  # under node 3's left edge
        cs_src = cite("Learn Claude Code · s12").next_to(cs, m.RIGHT, buff=0.35).align_to(cs, m.DOWN)
        assert cs_src.get_right()[0] < 6.6, cs_src.get_right()[0]
        assert cs.get_top()[1] < GRAPH_POS["3"][1] - NODE_H / 2 - 0.2 and cs.get_bottom()[1] > -3.55
        line_k = CAN_START_SRC.split("\n").index('        if load_task(dep_id).status != "completed":')
        bar = highlight(cs, CAN_START_SRC, line_k, SUN)
        ready1 = gnode("1", "ready")

        self.speak(
            "[Draw every blocker as an arrow], [and the payments job becomes a graph.]",
            m.AnimationGroup(self.new_header("02 · blockers", "Blockers make a graph"),
                             m.LaggedStart(*[m.FadeIn(self.nodes[t], scale=0.9) for t in ("1", "2", "3", "4", "5")], lag_ratio=0.15)),
            m.LaggedStart(*[m.GrowArrow(self.edges[e]) for e in EDGES], lag_ratio=0.2),
        )
        self.speak(
            "[The endpoint and the docs wait for the model], [the tests wait for the endpoint], [and the pull request waits for two tasks.]",
            m.AnimationGroup(pulse(self.edges[("1", "2")], AZURE), pulse(self.edges[("1", "3")], AZURE)),
            pulse(self.edges[("2", "4")], AZURE),
            m.AnimationGroup(pulse(self.edges[("4", "5")], AZURE), pulse(self.edges[("3", "5")], AZURE),
                             m.Circumscribe(self.nodes["5"], color=AZURE, buff=0.08)),
        )
        self.speak(
            "[The rule]: [a task can start only when every one of its blockers is completed.]",
            m.AnimationGroup(m.FadeIn(cs, shift=0.2 * m.UP), m.FadeIn(cs_src)),
            m.FadeIn(bar),
        )
        self.speak(
            "[Right now, that's only one task]: [the payment model is ready.]",
            m.Wait(0.2),
            m.AnimationGroup(m.ReplacementTransform(self.nodes["1"], ready1), m.Circumscribe(self.nodes["1"], color=AZURE, buff=0.1)),
        )
        self.nodes["1"] = ready1
        self.wait(0.3)
        self.play(m.FadeOut(cs), m.FadeOut(bar), m.FadeOut(cs_src), run_time=0.6)

    # ------------------------------------------------------------------ 3 · claim and complete

    def swap_node(self, t, state):
        new = gnode(t, state)
        anim = m.ReplacementTransform(self.nodes[t], new)
        self.nodes[t] = new
        return anim

    def claim_complete(self):
        self.next_section("3 · Claim and complete", notes=f"The run (asserted): Otto first tries to claim the pull request: {REFUSED!r} (claim_task checks can_start; nothing changes). claim_task('1') → 'Claimed 1 (payment model)': owner otto, in_progress. complete_task('1') → {DONE1!r}. Then task 2 → {DONE2!r}; task 3 → {DONE3!r}: nothing new, task 5 still waits for 4. (complete_task already runs the tests here, they pass: {CHECK['1']!r}; section 5 shows why.) Facts as of October 2026.")
        bot = self.bottom()
        rows = Slot(self)
        res = Slot(self)
        k = find("claim_task", "5", 0)
        c_ref, r_ref = self.call_row(TRACE[k][0]), self.result_row(TRACE[k][1])
        k1 = find("claim_task", "1")
        c1, r1 = self.call_row(TRACE[k1][0]), self.result_row(TRACE[k1][1])
        kw = find("write_file")  # payments.py
        cw, rw = self.call_row(TRACE[kw][0]), self.result_row(TRACE[kw][1])
        kc1 = find("complete_task", "1")
        cc1, rc1 = self.call_row(TRACE[kc1][0]), self.result_row(TRACE[kc1][1])
        k2, kc2 = find("claim_task", "2"), find("complete_task", "2")
        c2, r2 = self.call_row(TRACE[k2][0]), self.result_row(TRACE[k2][1])
        cc2, rc2 = self.call_row(TRACE[kc2][0]), self.result_row(TRACE[kc2][1])
        k3, kc3 = find("claim_task", "3"), find("complete_task", "3")
        c3, r3 = self.call_row(TRACE[k3][0]), self.result_row(TRACE[k3][1])
        cc3, rc3 = self.call_row(TRACE[kc3][0]), self.result_row(TRACE[kc3][1])
        assert TRACE[kw][0].input["path"] == "payments.py" and kw == k1 + 1 and kc1 == kw + 1

        self.speak(
            "[Say Otto jumps ahead and claims the pull request first]: [the harness refuses]: [blocked by three and four.]",
            m.AnimationGroup(self.new_header("03 · claim and complete", "The graph lights up"), m.FadeIn(bot, shift=0.2 * m.UP),
                             rows.to(c_ref, source=bot)),
            res.to(r_ref, source=self.nodes["5"]),
            m.AnimationGroup(m.Circumscribe(self.nodes["4"], color=ROSE, buff=0.08), m.Circumscribe(self.nodes["3"], color=ROSE, buff=0.08)),
        )
        self.speak(
            "[So Otto claims the model instead]: [the harness sets the owner to Otto], [and the status to in progress.]",
            m.AnimationGroup(rows.to(c1, source=bot), res.to(None)),
            m.AnimationGroup(res.to(r1, source=self.nodes["1"]), self.swap_node("1", "in_progress")),
            m.Circumscribe(self.nodes["1"], color=SUN, buff=0.08),
        )
        self.speak(
            "[Otto writes the model and calls complete], [and the harness reports what just got unblocked]: [the endpoint and the docs.]",
            m.Succession(m.AnimationGroup(rows.to(cw, source=bot), res.to(rw)), m.Wait(0.3), m.AnimationGroup(rows.to(cc1, source=bot), res.to(None))),
            m.AnimationGroup(res.to(rc1, source=self.nodes["1"]), self.swap_node("1", "completed")),
            m.AnimationGroup(pulse(self.edges[("1", "2")], AZURE), pulse(self.edges[("1", "3")], AZURE),
                             self.swap_node("2", "ready"), self.swap_node("3", "ready")),
        )
        n2_working = gnode("2", "in_progress")
        n2_done = gnode("2", "completed")
        n4_ready = gnode("4", "ready")
        self.speak(
            "[Next, Otto finishes the endpoint], [and the tests light up.]",
            m.Succession(
                m.AnimationGroup(rows.to(c2, source=bot), res.to(None), m.ReplacementTransform(self.nodes["2"], n2_working)),
                m.AnimationGroup(res.to(r2)),
                m.AnimationGroup(rows.to(cc2, source=bot), res.to(None)),
                m.AnimationGroup(res.to(rc2, source=n2_working), m.ReplacementTransform(n2_working, n2_done)),
            ),
            m.AnimationGroup(pulse(self.edges[("2", "4")], AZURE), m.ReplacementTransform(self.nodes["4"], n4_ready)),
        )
        self.nodes["2"], self.nodes["4"] = n2_done, n4_ready
        n3_working = gnode("3", "in_progress")
        n3_done = gnode("3", "completed")
        self.speak(
            "[Then the docs.] [But the pull request stays dark]: [one of its blockers is still open.]",
            m.Succession(
                m.AnimationGroup(rows.to(c3, source=bot), res.to(None), m.ReplacementTransform(self.nodes["3"], n3_working)),
                m.AnimationGroup(rows.to(cc3, source=bot)),
                m.AnimationGroup(res.to(rc3, source=n3_working), m.ReplacementTransform(n3_working, n3_done)),
            ),
            m.AnimationGroup(pulse(self.edges[("3", "5")], MINT), m.Circumscribe(self.nodes["5"], color=MUTED, buff=0.08)),
            m.AnimationGroup(pulse(self.edges[("4", "5")], ROSE), m.Circumscribe(self.nodes["4"], color=AZURE, buff=0.08)),
        )
        self.nodes["3"] = n3_done
        self.bot, self.rows, self.res = bot, rows, res
        self.wait(0.3)

    # ------------------------------------------------------------------ 4 · the board

    def the_board(self):
        self.next_section("4 · The board", notes="Learn Claude Code, s12: the same tasks drawn as a board with four columns, Waiting / Ready / Working / Done; 'A dependency is not an arrow students must trace. It is a visible blocker badge on the card.' Our badge lists the blockers still open (task 5 waits for task 4; task 3 is done). The Done column carries the check (pytest -q) that complete_task runs (section 5). State after task 3 (asserted in the run). Facts as of October 2026.")
        board = Board()
        self.board = board
        snap = state_after(find("complete_task", "3"))
        assert [t for t, (s, _) in snap.items() if s == "completed"] == ["1", "2", "3"]
        done_cards = []
        for t in ("1", "2", "3"):
            c = bcard(t, "completed")
            board.place(c, "done", [d.height for d in done_cards])
            done_cards.append(c)
        ready4 = board.place(bcard("4", "ready"), "ready")
        open_blockers = [d for d in TASKS[4][2] if snap[d][0] != "completed"]
        assert open_blockers == ["4"]
        wait5 = board.place(bcard("5", "pending", open_blockers), "waiting")
        quote = text("“A dependency is not an arrow students must trace.", 22, MUTED)
        quote2 = text("It is a visible blocker badge on the card.”", 22, MUTED)
        qsrc = cite("Learn Claude Code · s12")
        qg = m.VGroup(m.VGroup(quote, quote2).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT), qsrc).arrange(m.RIGHT, buff=0.5, aligned_edge=m.DOWN)
        qg.move_to([0.2, -2.85, 0])

        targets = {"1": done_cards[0], "2": done_cards[1], "3": done_cards[2], "4": ready4, "5": wait5}
        self.speak(
            "[As graphs grow, arrows get hard to follow], [so Learn Claude Code draws a board instead]: [waiting], [ready], [working], [done.]",
            m.AnimationGroup(self.new_header("04 · the board", "Badges, not arrows"), *[m.FadeOut(e) for e in self.edges.values()],
                             self.rows.to(None), self.res.to(None), m.FadeOut(self.bot)),
            m.AnimationGroup(*[m.FadeIn(board.columns[c]) for c in COLS],
                             *[m.ReplacementTransform(self.nodes[t], targets[t]) for t in SUBJECT]),
            m.FadeIn(board.heads["waiting"], shift=0.1 * m.DOWN),
            m.FadeIn(board.heads["ready"], shift=0.1 * m.DOWN),
            m.FadeIn(board.heads["working"], shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(board.heads["done"], shift=0.1 * m.DOWN), m.FadeIn(board.gate), m.FadeIn(board.check)),
        )
        self.speak(
            "[Each dependency becomes a badge on its card]: [the pull request waits for task four.]",
            m.AnimationGroup(m.Circumscribe(wait5[3], color=ROSE, buff=0.06), m.FadeIn(qg, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(ready4, color=AZURE, buff=0.06)),
        )
        self.cards = dict(targets)
        self.done_cards = done_cards
        self.quote = qg
        self.wait(0.3)

    # ------------------------------------------------------------------ 5 · done is a claim

    def verify(self):
        self.next_section("5 · Done is a claim", notes=f"The run continues (asserted): claim_task('4'); Otto writes two payment tests (tests/test_pay.py) and calls complete_task('4') without running them (no bash pytest call by the model in the whole run). complete_task runs pytest -q itself: {CHECK['4']!r} (test_pay_declined: a declined card still returns status 'paid'; the endpoint ignored the card's reply). Result: {REJECTED!r}; task 4 stays in_progress. Otto edits api.py (status 'paid' only if reply.ok), calls complete again: {CHECK['4#1']!r} → {DONE4!r}. Then the pull request: claimed, checked ({CHECK['5']!r}), {DONE5!r}. Li, AI Agents in Depth, ch. 5: 'the most common way a Coding Agent slacks off is to skip' testing. Facts as of October 2026.")
        b = self.board
        bot = self.bottom()
        rows, res = Slot(self), Slot(self)
        k4 = find("claim_task", "4")
        kw = find("write_file", nth=3)
        kc, ke, kc2 = find("complete_task", "4", 0), find("edit_file"), find("complete_task", "4", 1)
        k5, kc5 = find("claim_task", "5", 1), find("complete_task", "5")
        assert TRACE[kw][0].input["path"] == "tests/test_pay.py" and [k4, kw, kc, ke, kc2] == list(range(k4, k4 + 5))
        c4, r4 = self.call_row(TRACE[k4][0]), self.result_row(TRACE[k4][1])
        cw, rw = self.call_row(TRACE[kw][0]), self.result_row(TRACE[kw][1])
        cc, rc = self.call_row(TRACE[kc][0]), self.result_row(TRACE[kc][1])
        ce, re_ = self.call_row(TRACE[ke][0]), self.result_row(TRACE[ke][1])
        cc2, rc2 = self.call_row(TRACE[kc2][0]), self.result_row(TRACE[kc2][1])
        c5, r5 = self.call_row(TRACE[k5][0]), self.result_row(TRACE[k5][1])
        cc5, rc5 = self.call_row(TRACE[kc5][0]), self.result_row(TRACE[kc5][1])

        work4 = b.place(bcard("4", "in_progress"), "working")
        skipped = code("pytest -q", 22, FAINT)
        skipped.next_to(work4, m.DOWN, buff=0.45)
        skip_cap = text("✗ never run", 22, ROSE, weight=m.SEMIBOLD).next_to(skipped, m.DOWN, buff=0.14)
        slack = m.VGroup(text("“the most common way a coding agent slacks off”", 24, INK), cite("Li, AI Agents in Depth · ch. 5"))
        slack.arrange(m.RIGHT, buff=0.4, aligned_edge=m.DOWN).move_to([0.75, -3.13, 0])
        fail_lab = b.check_label(CHECK["4"].split("\n")[-1], ROSE)
        pass_lab = b.check_label(CHECK["4#1"], MINT)
        idle_lab = b.check_label("check: pytest -q", SUN)
        pass_lab5 = b.check_label(CHECK["5"], MINT)
        rejected4 = b.place(bcard("4", "rejected"), "working")
        rej_pill = pill("rejected", ROSE, ROSE_FILL, 22).next_to(rejected4, m.DOWN, buff=0.3)
        back4 = b.place(bcard("4", "in_progress"), "working")
        done4 = b.place(bcard("4", "completed"), "done", [d.height for d in self.done_cards])
        ready5 = b.place(bcard("5", "ready"), "ready")
        work5 = b.place(bcard("5", "in_progress"), "working")
        done5 = b.place(bcard("5", "completed"), "done", [d.height for d in self.done_cards] + [done4.height])
        assert done5.get_bottom()[1] > BOARD_BOT + 0.1
        gate_rose = b.gate.copy().set_stroke(ROSE, 5)
        gate_mint = b.gate.copy().set_stroke(MINT, 5)

        self.speak(
            "[Otto claims the tests], [writes two payment tests], [and calls complete], [without running them.]",
            m.AnimationGroup(self.new_header("05 · verify", "Done is a claim"), m.FadeOut(self.quote), m.FadeIn(bot, shift=0.2 * m.UP),
                             rows.to(c4, source=bot), m.ReplacementTransform(self.cards["4"], work4)),
            m.AnimationGroup(rows.to(cw, source=bot), res.to(rw)),
            m.AnimationGroup(rows.to(cc, source=bot), res.to(None)),
            m.AnimationGroup(m.FadeIn(skipped), m.FadeIn(skip_cap, shift=0.1 * m.UP)),
        )
        self.speak(
            "[That's the most common way a coding agent slacks off]: [it skips the tests.]",
            res.to(slack),
            m.Circumscribe(m.VGroup(skipped, skip_cap), color=ROSE, buff=0.1),
        )
        self.speak(
            "[But complete doesn't take Otto's word for it.] [The harness runs the tests itself], [and one fails.]",
            m.AnimationGroup(skipped.animate.set_color(SUN), m.FadeOut(skip_cap)),
            m.AnimationGroup(m.Circumscribe(b.check, color=SUN, buff=0.08), work4.animate(rate_func=m.there_and_back).shift(0.7 * m.RIGHT)),
            m.AnimationGroup(m.ReplacementTransform(b.check, fail_lab), m.FadeIn(gate_rose), m.FadeOut(skipped)),
        )
        self.speak(
            "[A declined card still comes back as paid], [so the claim is rejected], [and the task stays in progress.]",
            res.to(rc, source=b.gate),
            m.AnimationGroup(m.ReplacementTransform(work4, rejected4), m.FadeIn(rej_pill, scale=1.1)),
            m.Circumscribe(rejected4, color=ROSE, buff=0.06),
        )
        self.speak(
            "[Otto reads the failure], [fixes the endpoint], [and calls complete again.]",
            m.Circumscribe(res.current[3], color=ROSE, buff=0.06),
            m.AnimationGroup(rows.to(ce, source=bot), res.to(re_), m.ReplacementTransform(rejected4, back4), m.FadeOut(rej_pill),
                             m.ReplacementTransform(fail_lab, idle_lab), m.FadeOut(gate_rose)),
            m.AnimationGroup(rows.to(cc2, source=bot), res.to(None)),
        )
        self.speak(
            "[Fifteen passed.] [Now the harness marks it done], [and the pull request is unblocked.]",
            m.AnimationGroup(m.ReplacementTransform(idle_lab, pass_lab), m.FadeIn(gate_mint)),
            m.AnimationGroup(m.ReplacementTransform(back4, done4, path_arc=-0.4), res.to(rc2, source=b.gate)),
            m.ReplacementTransform(self.cards["5"], ready5),
        )
        idle2 = b.check_label("check: pytest -q", SUN)
        self.speak(
            "[Last, the pull request]: [claimed], [checked], [and done.] [Five tasks, all verified.]",
            m.AnimationGroup(m.FadeOut(gate_mint), m.ReplacementTransform(pass_lab, idle2)),
            m.AnimationGroup(rows.to(c5, source=bot), res.to(r5), m.ReplacementTransform(ready5, work5)),
            m.AnimationGroup(rows.to(cc5, source=bot), res.to(None), m.ReplacementTransform(idle2, pass_lab5)),
            m.AnimationGroup(res.to(rc5, source=b.gate), m.ReplacementTransform(work5, done5, path_arc=-0.4)),
            m.Circumscribe(m.VGroup(*self.done_cards, done4, done5), color=MINT, buff=0.1),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · the check

    def the_check(self):
        self.next_section("6 · Three lines in complete", notes=f"What Episode 22 adds to Learn Claude Code's complete_task, in its style (the scene runs exactly this code; the excerpt's lines are asserted to be in it): run the check first (run_bash('pytest -q')); if anything failed, return 'Rejected: …' and leave the task in_progress; only then mark it completed and report what got unblocked. The endpoint passed the same check when task 2 completed ({CHECK['2']!r}): no test looked at declined cards yet; the payment tests found it ({CHECK['4'].splitlines()[-1]!r}). Facts as of October 2026.")
        cb = code_block(COMPLETE_SRC, size=20, title="complete_task · with a check")
        cb.move_to([0, 0.55, 0])
        lines = COMPLETE_SRC.split("\n")
        bars = m.VGroup(*[highlight(cb, COMPLETE_SRC, i, SUN) for i in (1, 2, 3)])
        rows_ = []
        for tid, key, verdict, color in (("2", "2", "accepted", MINT), ("4", "4", "rejected", ROSE)):
            out = CHECK[key].split("\n")[-1]
            r = m.VGroup(code(f'complete_task("{tid}")', 22, SUN), m.Line(m.ORIGIN, 0.55 * m.RIGHT), code(out, 22, color),
                         pill(verdict, color, MINT_FILL if color == MINT else ROSE_FILL, 20))
            r.arrange(m.RIGHT, buff=0.3)
            r[1].become(arrow(r[1].get_left(), r[1].get_right(), MUTED, buff=0.0)).set_y(r[0].get_y())
            rows_.append(r)
        tl = m.VGroup(*rows_).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT).move_to([-0.4, -2.55, 0])
        for r in rows_:
            sit(r[2], baseline(r[0]))
        why = text("no test for declined cards yet", 22, MUTED).next_to(rows_[0], m.RIGHT, buff=0.4)
        m.VGroup(tl, why).set_x(0)
        assert why.get_right()[0] < 6.6 and tl.get_left()[0] > -6.6, (tl.get_left()[0], why.get_right()[0])

        self.speak(
            "[Here's the whole change]: [before complete marks anything done], [it runs the tests], [and returns rejected if any fail.]",
            m.AnimationGroup(self.new_header("06 · the check", "Three lines in complete"), m.FadeIn(cb, shift=0.2 * m.UP)),
            m.Wait(0.2),
            m.FadeIn(bars[0]),
            m.AnimationGroup(m.FadeIn(bars[1]), m.FadeIn(bars[2])),
        )
        self.speak(
            "[Notice: the endpoint passed this same check earlier], [because no test looked at declined cards yet.]",
            m.AnimationGroup(m.FadeIn(rows_[0], shift=0.1 * m.UP), m.FadeIn(rows_[1], shift=0.1 * m.UP)),
            m.FadeIn(why, shift=0.1 * m.LEFT),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · loop engineering

    def loop_engineering(self):
        self.next_section("7 · Loop engineering", notes="Li, AI Agents in Depth, ch. 10 (loop engineering): 'until it is verified, done is merely the model's claim, not proof.' Three forms of failure, with the book's examples: partial completion (a coding agent writes code but never runs the tests, then reports 'task complete'); giving up too soon (after one rejected call, 'this can't be done'); false success (a refund agreed by phone that never lands, the agent reports 'all set'). Episode 4: producing an answer is not completing the task. Facts as of October 2026.")
        q1 = text("“Until it is verified, ‘done’ is merely", 36, INK)
        q2 = text("the model’s claim, not proof.”", 36, INK)
        paint(q2, q2.text, {r"claim": CORAL, r"proof": MINT})
        quote = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.18).move_to([0, 1.6, 0])
        src = cite("Li, AI Agents in Depth · ch. 10   ·   an answer is not a completed task: Episode 4").next_to(quote, m.DOWN, buff=0.35)
        spec = [("partial completion", "code written,\ntests never run", "“task complete”"),
                ("giving up too soon", "one call rejected,\nno other route tried", "“this can't be done”"),
                ("false success", "a refund agreed,\nbut it never lands", "“all set”")]
        cards = []
        for title, desc, claim_ in spec:
            box = card(4.0, 2.55, ROSE, CARD, radius=0.2, stroke=2)
            t = text(title, 28, ROSE, weight=m.BOLD)
            d = text(desc, 22, MUTED)
            c = word_chip(claim_, CORAL, CORAL_FILL, size=22, height=0.52, pad=0.5)
            body = m.VGroup(t, d).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT)
            body.move_to(box).align_to(box, m.UP).shift(0.28 * m.DOWN).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            cards.append((box, body, c))
        row_ = m.VGroup(*[m.VGroup(b_, bd) for b_, bd, _ in cards]).arrange(m.RIGHT, buff=0.3).move_to([0, -1.55, 0])
        for (b_, bd, c) in cards:  # titles and first description lines on shared baselines
            t, d = bd
            sit(t, b_.get_top()[1] - 0.58)
            first = d.text.split("\n")[0]
            d.shift((b_.get_top()[1] - 1.05 - baseline_of(line_glyphs(d, 0), [ch for ch in first if not ch.isspace()])) * m.UP)
            c.align_to(b_, m.DOWN).shift(0.3 * m.UP).align_to(b_, m.LEFT).shift(0.3 * m.RIGHT)

        self.speak(
            "[AI Agents in Depth puts it plainly]: [until it is verified], [done is merely the model's claim, not proof.]",
            m.AnimationGroup(self.new_header("07 · loop engineering", "A claim is not proof"), m.FadeIn(src)),
            m.FadeIn(q1, shift=0.1 * m.UP),
            m.FadeIn(q2, shift=0.1 * m.UP),
        )
        self.speak(
            "[It names three ways a claim goes wrong]: [partial completion], [giving up too soon], [and false success.]",
            m.Wait(0.2),
            *[m.AnimationGroup(m.FadeIn(b_), m.FadeIn(bd, shift=0.1 * m.UP), m.FadeIn(c, shift=0.1 * m.UP)) for b_, bd, c in cards],
        )
        self.wait(0.3)
        self.clear()

    def the_verifier(self):
        self.next_section("8 · The verifier", notes="Li, AI Agents in Depth, ch. 10: 'the bottleneck of the loop is the verifier, not the model' (with weak verification, a faster loop only marks poor output complete sooner); LoopX's rule: 'the model may propose done, but it cannot approve its own done' (LoopX decides → agent executes → independent verifier proves → LoopX commits; the book notes its governed path is still experimental). In this episode the verifier is complete_task running the tests. Facts as of October 2026.")
        s1 = title_text("The bottleneck of the loop is the verifier,", 40, INK)
        s2 = title_text("not the model.", 40, INK)
        paint(s1, s1.text, {r"verifier": MINT})
        stmt = m.VGroup(s1, s2).arrange(m.DOWN, buff=0.2).move_to([0, 1.75, 0])
        sub = text("With a weak check, a faster loop just marks bad work done sooner.", 24, MUTED).next_to(stmt, m.DOWN, buff=0.35)

        bot = otto(0.8).move_to([-5.4, -1.45, 0])
        prop = word_chip("proposes done", CORAL, CORAL_FILL, size=24)
        ver_icon = icon("check", 0.55)
        ver_word = text("verifier", 26, INK, weight=m.BOLD)
        ver_sub = code("runs the tests", 20, MUTED)
        ver_body = m.VGroup(ver_icon, m.VGroup(ver_word, ver_sub).arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)).arrange(m.RIGHT, buff=0.25)
        ver_box = card(ver_body.width + 0.6, 1.25, MINT, MINT_FILL, radius=0.2, stroke=2.5)
        ver = m.VGroup(ver_box, ver_body.move_to(ver_box))
        appr = word_chip("approves", MINT, MINT_FILL, size=24)
        flow = m.VGroup(prop, ver, appr).arrange(m.RIGHT, buff=0.8).move_to([0.75, -1.45, 0])
        a1 = arrow(bot.get_right(), prop.get_left(), MUTED, buff=0.15)
        a2 = arrow(prop.get_right(), ver.get_left(), MUTED, buff=0.12)
        a3 = arrow(ver.get_right(), appr.get_left(), MINT, buff=0.12)
        start = prop.get_bottom() + 0.05 * m.DOWN
        end = appr.get_bottom() + 0.05 * m.DOWN
        self_ok = m.CurvedArrow(start, end, angle=0.55, color=ROSE, stroke_width=4)
        self_ok.set_stroke(ROSE, 4)
        cross = text("✗", 34, ROSE, weight=m.BOLD)
        cross_cap = text("cannot approve its own done", 22, ROSE)
        m.VGroup(cross, cross_cap).arrange(m.RIGHT, buff=0.2).next_to(self_ok, m.DOWN, buff=0.12)
        src = cite("Li, AI Agents in Depth · ch. 10 (LoopX)")
        src.move_to([0, -3.3, 0]).align_to([6.45, 0, 0], m.RIGHT)
        assert self_ok.get_bottom()[1] > -3.5 and cross_cap.get_bottom()[1] > -3.55

        self.speak(
            "[That's why the bottleneck of the loop is the verifier, not the model]: [with a weak check, a faster loop just marks bad work done sooner.]",
            m.AnimationGroup(self.new_header("08 · the verifier", "Propose, don't approve"), m.Write(stmt)),
            m.FadeIn(sub, shift=0.1 * m.UP),
        )
        self.speak(
            "[The rule]: [the model may propose done], [but it cannot approve its own done.]",
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(prop, shift=0.1 * m.RIGHT), m.GrowArrow(a2), m.FadeIn(ver), m.GrowArrow(a3), m.FadeIn(appr)),
            m.AnimationGroup(m.Create(self_ok), m.FadeIn(cross, scale=1.3), m.FadeIn(cross_cap)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 9 · long-running agents

    def long_running(self):
        self.next_section("9 · Long-running agents", notes="Anthropic, 'Effective harnesses for long-running agents' (Nov 26, 2025): an initializer session writes the full feature list, every item failing at first ('passes': false), with a progress file and an initial git commit; every later session starts fresh, makes incremental progress on one item, tests it, and leaves structured updates. Li, AI Agents in Depth, ch. 1: an Initialization Agent plus an Execution Agent making 'incremental progress each session and leaving clear handover artifacts'. Drawn here with this episode's task files (blockers, ready, verified). The timeline (eight tasks, three sessions) and its blockers are illustrative. Episode 14: handover files. Episode 21: one task to a subagent with a fresh context. Episode 25: several agents claiming from one board. Facts as of October 2026.")
        names = ["initializer", "session 1", "session 2", "session 3"]
        xs = [-4.95, -1.65, 1.65, 4.95]
        boxes = []
        for name, x in zip(names, xs):
            first = name == "initializer"
            b_ = card(2.85, 1.55, ORCHID if first else EDGE, CARD, radius=0.18, stroke=2).move_to([x, 1.35, 0])
            o = otto(0.36).move_to(b_).align_to(b_, m.UP).shift(0.24 * m.DOWN).align_to(b_, m.LEFT).shift(0.25 * m.RIGHT)
            lab = code(name, 20, ORCHID if first else INK, weight=m.BOLD).next_to(o, m.RIGHT, buff=0.18)
            meter = Meter(2.15, 0.0, SUN, 0.16).move_to(b_).align_to(b_, m.DOWN).shift(0.32 * m.UP)
            boxes.append(m.VGroup(b_, o, lab, meter))
        time_arrows = m.VGroup(*[arrow(boxes[i][0].get_right(), boxes[i + 1][0].get_left(), FAINT, buff=0.06) for i in range(3)])
        n_tasks = 8
        blocked_by = {2: [1], 3: [1], 4: [2], 5: [3], 6: [4, 5], 7: [6], 8: [6]}  # illustrative
        disk = card(12.4, 1.35, EDGE, PANEL, radius=0.18, stroke=2).move_to([0, -0.85, 0])
        disk_cap = code(".tasks/ on disk", 20, MUTED).next_to(disk, m.DOWN, buff=0.14).align_to(disk, m.LEFT)
        src = cite("Anthropic, 2025 · Li, AI Agents in Depth · ch. 1").next_to(disk, m.DOWN, buff=0.14).align_to(disk, m.RIGHT)
        sit(src, baseline(disk_cap))
        look = {"failing": (EDGE, CARD, ROSE), "blocked": (EDGE, CARD, MUTED), "ready": STYLE["ready"],
                "working": STYLE["in_progress"], "passing": STYLE["completed"]}

        def tcard(i, state):
            stroke, fill, accent = look[state]
            box = card(1.42, 0.8, stroke, fill, radius=0.14, stroke=2)
            num = code(str(i), 24, accent if state not in ("failing", "blocked") else MUTED, weight=m.BOLD).move_to(box)
            w = code(state, 20, accent).move_to(box).shift(0.17 * m.DOWN)
            num.shift(0.13 * m.UP)
            return m.VGroup(box, num, w).move_to([-5.32 + (i - 1) * 1.52, -0.85, 0])

        def board_state(done):
            return {i: "passing" if i in done else ("ready" if all(d in done for d in blocked_by.get(i, [])) else "blocked")
                    for i in range(1, n_tasks + 1)}

        states = [board_state(set(range(1, k + 1))) for k in range(4)]
        assert [i for i, s in states[3].items() if s == "ready"] == [4, 5] and [i for i, s in states[3].items() if s == "blocked"] == [6, 7, 8]
        cur = {i: tcard(i, "failing") for i in range(1, n_tasks + 1)}
        first_cards = list(cur.values())

        def to(i, state):
            new = tcard(i, state)
            anim = m.ReplacementTransform(cur[i], new)
            cur[i] = new
            return anim

        def changes(before, after, skip=()):
            return [to(j, s) for j, s in after.items() if s != before.get(j) and j not in skip]

        reads = [m.DashedLine(boxes[k][0].get_bottom() + [-0.4, -0.05, 0], [xs[k] - 0.4, disk.get_top()[1] + 0.05, 0], dash_length=0.1).set_stroke(AZURE, 3) for k in range(4)]
        writes = [arrow(boxes[k][0].get_bottom() + [0.4, -0.05, 0], [xs[k] + 0.4, disk.get_top()[1] + 0.05, 0], MINT, buff=0.0) for k in range(4)]

        def session(k, before, after, quick=True):
            meter = boxes[k][3]
            rt = 0.45 if quick else 0.9
            return m.Succession(
                m.AnimationGroup(m.GrowArrow(time_arrows[k - 1]), m.FadeIn(boxes[k]), m.Create(reads[k]),
                                 boxes[k - 1][3].fill_to(0.0), run_time=rt),  # the last session's context is gone
                m.AnimationGroup(meter.fill_to(0.5), to(k, "working"), run_time=rt),
                m.AnimationGroup(meter.fill_to(0.8), to(k, "passing"), m.GrowArrow(writes[k]), *changes(before, after, skip=(k,)), run_time=rt),
            )

        start = {i: "failing" for i in cur}
        ready0 = states[0]
        sub = m.VGroup(otto(0.4, AZURE, AZURE_FILL), text("a subagent per task · Episode 21", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        team = m.VGroup(otto(0.4), otto(0.4, MINT, MINT_FILL)).arrange(m.RIGHT, buff=0.12)
        share = m.VGroup(team, text("several agents, one board · Episode 25", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        bottoms = m.VGroup(sub, share).arrange(m.RIGHT, buff=0.7).move_to([0, -2.75, 0])
        assert bottoms.width < 12.6

        self.speak(
            "[For goals too big for one session], [Anthropic's pattern for long-running agents] [starts with an initializer], [which writes the whole task list, every item failing.]",
            self.new_header("09 · long-running", "The board is the handover"),
            m.AnimationGroup(m.FadeIn(disk), m.FadeIn(disk_cap), m.FadeIn(src)),
            m.FadeIn(boxes[0], shift=0.15 * m.UP),
            m.AnimationGroup(m.GrowArrow(writes[0]), m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.DOWN) for c in first_cards], lag_ratio=0.1),
                             boxes[0][3].fill_to(0.5)),
        )
        self.speak(
            "[Then each fresh session reads the list], [takes one task], [verifies it], [and leaves the files for the next.]",
            m.AnimationGroup(m.GrowArrow(time_arrows[0]), m.FadeIn(boxes[1]), m.Create(reads[1]), boxes[0][3].fill_to(0.0),
                             *changes(start, ready0)),
            m.AnimationGroup(boxes[1][3].fill_to(0.5), to(1, "working")),
            m.AnimationGroup(boxes[1][3].fill_to(0.8), to(1, "passing"), *changes(ready0, states[1], skip=(1,))),
            m.Succession(m.GrowArrow(writes[1], run_time=0.5), session(2, states[1], states[2]), session(3, states[2], states[3])),
        )
        self.speak(
            "[As in episode fourteen, the files are the handover]: [what's done], [what's ready], [and what's blocked.]",
            m.Circumscribe(disk, color=ORCHID, buff=0.06),
            m.Circumscribe(m.VGroup(*[cur[i] for i in (1, 2, 3)]), color=MINT, buff=0.08),
            m.Circumscribe(m.VGroup(*[cur[i] for i in (4, 5)]), color=AZURE, buff=0.08),
            m.Circumscribe(m.VGroup(*[cur[i] for i in (6, 7, 8)]), color=MUTED, buff=0.08),
        )
        self.speak(
            "[One task can also go to a subagent, with a fresh context, as in episode twenty-one.]",
            m.FadeIn(sub, shift=0.15 * m.UP),
        )
        self.speak(
            "[And since every claim records an owner], [several agents can share one board]: [that's episode twenty-five.]",
            m.Wait(0.2),
            m.FadeIn(share, shift=0.15 * m.UP),
            m.Circumscribe(share[1], color=AZURE, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Block (tasks wait for every blocker: blockedBy, can_start), claim (owner + in_progress, on disk), verify (the harness runs the check before it accepts done; the model proposes, the verifier approves).")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Block", ROSE), ("Claim", SUN), ("Verify", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("wait for every blocker", "owner + in_progress", "the harness runs the check")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 1.0
        for w, s, slot in zip(words, subs, slots):
            w.move_to([x + slot / 2, 0, 0])
            s.move_to([x + slot / 2, 0, 0])
            x += slot + gap
        on_baseline(words)
        line_y = baseline(words[0]) - 0.6
        for s in subs:
            sit(s, line_y)
        mid = words[1][0].get_center()[1]
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row_ = m.VGroup(words, dots, subs).move_to(0.45 * m.UP)
        tag1 = text("A to-do list says what to do.", 30, MUTED)
        tag2 = text("A task system knows what can start, and what is truly done.", 30, MUTED)
        tagline = m.VGroup(tag1, tag2).arrange(m.DOWN, buff=0.2).next_to(row_, m.DOWN, buff=0.65)
        assert row_.width < 12.8 and tagline.width < 12.8

        self.speak(
            "[To sum up]: [block], [claim], [verify.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.speak("[A to-do list says what to do.] [A task system knows what can start], [and what is truly done.]",
                   m.FadeIn(tag1, shift=0.1 * m.UP), m.FadeIn(tag2, shift=0.1 * m.UP), m.Circumscribe(words[2], color=MINT, buff=0.12))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
