"""Agents, Animated · Episode 25 · Autonomy & Worktrees (Module 5 · Scaling Up)

How agents find their own work without colliding. Running example: two Ottos (teammates otto-1 and otto-2)
fixing two bugs in tinyshop's cart.py in parallel: cart.total() skips the last item (the series' bug) and
shipping() still charges an order of exactly 50 (illustrative). Nothing is drawn by hand:
- the task board (Learn Claude Code s12/s17: Task files, can_start, scan_unclaimed_tasks, claim_task) runs
  over an in-memory disk; two claims are stepped in turn to show the race without a lock and the refusal
  with one; idle_poll (the lines shown) runs against a simulated clock;
- reinject_identity runs on otto-2's compacted history;
- the shared-folder overwrite is computed from the file each Otto read;
- the worktrees are REAL: the scene builds a throwaway git repo in a temporary folder inside this
  episode's folder, runs `git worktree add`, commits on two branches, merges, tries to remove a dirty
  worktree, and shows git's own output; tinyshop's tests are run on every version of cart.py.
Every number on screen comes from these runs (asserted below) or from a cited source.

Sections: hook (Episode 24's lead hands out every job) · 1 two bugs, one board · 2 work and idle (poll every
5 s, inbox first, shut down after 60 s idle) · 3 claiming (three checks, the race, the lock) · 4 identity
after compaction · 5 one folder, two writers (a fix lost) · 6 a worktree per task · 7 names collide too
(18 of 30) · 8 merge and clean up · recap: poll, claim, isolate.

Sources (facts dated October 2026):
- Learn Claude Code s17 "Autonomous Agents" (learn.shareai.run/en/s17/), "Check the board, claim the task":
  teammates cycle WORK (inbox → LLM → tools, until the reply is not a tool call) → IDLE (poll the inbox,
  then the task board) → SHUTDOWN; IDLE_POLL_INTERVAL = 5 s, IDLE_TIMEOUT = 60 s (teaching constants);
  scan_unclaimed_tasks keeps tasks that are pending, have no owner and can_start; claim_task rejects
  "Task {id} already owned by {owner}", else "Claimed {id} ({subject})"; the teaching version has NO lock
  ("concurrent claims can still race"); after compaction, `if len(messages) <= 3:` insert
  "<identity>You are '{name}', role: {role}. Continue your work.</identity>" at index 0 (real Claude Code
  keeps its system prompt through compaction). Dive into CC source (the site's reading of Claude Code,
  not verified by us): a 500 ms polling loop over messages, mailbox and task list; fs.watch on the tasks
  directory; claims under proper-lockfile.
- Learn Claude Code s12 (CC source notes, per our research notes §15): claims take a file lock, "up to 30
  retries, exponential backoff 5-100ms".
- Learn Claude Code s18 "Worktree Isolation" (learn.shareai.run/en/s18/): two teammates writing config.py
  in one directory overwrite each other; "Tasks own the goal, worktrees own the directory, bound by ID";
  `git worktree add .worktrees/{name} -b wt/{name} HEAD`; claiming a task bound to a worktree sets the
  teammate's working directory; remove_worktree refuses uncommitted changes unless discard_changes.
  Claude Code: .claude/worktrees/, branch worktree-{slug}.
- Li Bojie, AI Agents in Depth, ch. 10, failure mode three ("homogeneous convergence"): "In Anthropic's
  experiment, 18 of 30 Agents that came online at the same time created Git branches with the same name."
- git itself (2.54, run here): "Preparing worktree (new branch 'wt/task-1')"; `git worktree remove` on a
  dirty worktree: "fatal: '.worktrees/task-2' contains modified or untracked files, use --force to delete
  it"; the merge of the two branches: "Auto-merging cart.py / Merge made by the 'ort' strategy."

Simplified (said in the notes): idle_poll on screen omits s17's shutdown_request handling and the
messages/role arguments; Task omits description; the lock is a stand-in for a lock file; in s18 the lead
creates the worktree and binds it to the task, here the claim gets it; worktrees are named after task IDs.
Illustrative (for teaching): the second bug and its test, the task subjects, tinyshop's 12 other tests,
the time the bugs are filed (12 s), otto-2's session and summary, the 30-agent grid's layout (the count
18 of 30 is the book's).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import asdict, dataclass, field
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

HERE = Path(__file__).resolve().parent

LEAD, ONE, TWO = "lead", "otto-1", "otto-2"
COLORS = {LEAD: (CORAL, CORAL_FILL), ONE: (AZURE, AZURE_FILL), TWO: (ORCHID, ORCHID_FILL)}
ROLE = "coder"

# ============================================================================ tinyshop, two bugs

CART = (
    "def total(items):\n"
    "    s = 0\n"
    "    for i in range(len(items) - 1):\n"
    "        s += items[i]\n"
    "    return s\n"
    "\n"
    "def shipping(subtotal):  # free from 50\n"
    "    return 0 if subtotal > 50 else 5\n"
)
BUG_1, FIX_1 = "range(len(items) - 1)", "range(len(items))"
BUG_2, FIX_2 = "subtotal > 50", "subtotal >= 50"
OTHER_TESTS = 12  # tinyshop's other tests, all passing (as in Episodes 2 and 22)


def fix_1(src):
    return src.replace(BUG_1, FIX_1, 1)


def fix_2(src):
    return src.replace(BUG_2, FIX_2, 1)


def cart_tests(src):
    space = {}
    exec(src, space)
    results = {"test_total": space["total"]([3, 4]) == 7, "test_free_shipping": space["shipping"](50) == 0}
    results.update({f"test_other_{i}": True for i in range(OTHER_TESTS)})
    return results


def pytest_q(src, k=None):
    """`pytest -q [-k k]` on tinyshop: (failed test names, the summary line)."""
    ran = {name: ok for name, ok in cart_tests(src).items() if k is None or k in name}
    failed = [name for name, ok in ran.items() if not ok]
    passed, deselected = len(ran) - len(failed), 2 + OTHER_TESTS - len(ran)
    parts = [f"{len(failed)} failed"] * bool(failed) + [f"{passed} passed"] * bool(passed)
    parts += [f"{deselected} deselected"] * bool(deselected)
    return failed, ", ".join(parts)


FAILED_0, SUMMARY_0 = pytest_q(CART)
assert FAILED_0 == ["test_total", "test_free_shipping"] and SUMMARY_0 == "2 failed, 12 passed"  # "two tests fail"
assert CART.count(BUG_1) == 1 and CART.count(BUG_2) == 1
CART_LINES = CART.rstrip("\n").split("\n")
LINE_1 = next(i for i, ln in enumerate(CART_LINES) if BUG_1 in ln)
LINE_2 = next(i for i, ln in enumerate(CART_LINES) if BUG_2 in ln)
assert (LINE_1 + 1, LINE_2 + 1) == (3, 8) and LINE_2 - LINE_1 > 2  # "different lines", not even adjacent

# ============================================================================ the task board (s12, s17)

SUBJECT = {"1": "fix cart total", "2": "fix shipping", "3": "tag release"}


@dataclass
class Task:  # stored as .tasks/{id}.json
    id: str
    subject: str
    status: str = "pending"  # pending | in_progress | completed
    owner: str | None = None
    blockedBy: list = field(default_factory=list)


DISK = {}


def save_task(task):
    DISK[f".tasks/{task.id}.json"] = json.dumps(asdict(task))


def load_task(task_id):
    return Task(**json.loads(DISK[f".tasks/{task_id}.json"]))


def list_tasks():
    return [load_task(path[len(".tasks/"):-len(".json")]) for path in sorted(DISK)]


def can_start(task_id):
    return all(load_task(dep).status == "completed" for dep in load_task(task_id).blockedBy)


def scan_unclaimed_tasks():
    return [t for t in list_tasks()
            if t.status == "pending"
            and not t.owner
            and can_start(t.id)]


SCAN_SRC = inspect.getsource(scan_unclaimed_tasks).strip()
assert len(SCAN_SRC.split("\n")) == 5


def file_tasks():
    """The lead files the two bugs, and a release that waits for both."""
    DISK.clear()
    save_task(Task("1", SUBJECT["1"]))
    save_task(Task("2", SUBJECT["2"]))
    save_task(Task("3", SUBJECT["3"], blockedBy=["1", "2"]))


class FileLock:
    """One claim at a time: a stand-in for Claude Code's lock file (up to 30 retries with backoff)."""

    def __init__(self):
        self.holder = None

    def acquire(self, who):
        if self.holder is None:
            self.holder = who
        return self.holder == who

    def release(self, who):
        if self.holder == who:
            self.holder = None


def claim_steps(task_id, owner, lock=None):
    """claim_task, step by step (read, check, write), all inside the lock when there is one."""
    if lock is not None:
        while not lock.acquire(owner):
            yield "wait"
        yield "lock"
    task = load_task(task_id)
    yield "read"
    if task.owner:
        result = f"Task {task_id} already owned by {task.owner}"
    elif task.status != "pending" or not can_start(task_id):
        result = f"Task {task_id} cannot be claimed"
    else:
        task.owner, task.status = owner, "in_progress"
        save_task(task)
        result = f"Claimed {task.id} ({task.subject})"
        yield "write"
    if lock is not None:
        lock.release(owner)
    return result


LOCK = FileLock()
CLAIM_LOG = []


def claim_task(task_id, owner):
    steps = claim_steps(task_id, owner, LOCK)
    while True:
        try:
            next(steps)
        except StopIteration as done:
            CLAIM_LOG.append((owner, done.value))
            return done.value


def interleave(claims):
    """Two claims stepped in turn, as when two teammates poll at the same moment."""
    running, results, trace = dict(claims), {}, []
    while running:
        for who in list(running):
            try:
                trace.append((who, next(running[who]), load_task("1").owner))
            except StopIteration as done:
                results[who] = done.value
                del running[who]
    return results, trace


# both Ottos scan at the same moment: tasks 1 and 2 are claimable, 3 is blocked
file_tasks()
SEEN = {who: [t.id for t in scan_unclaimed_tasks()] for who in (ONE, TWO)}
assert SEEN == {ONE: ["1", "2"], TWO: ["1", "2"]}
CLAIMED_1 = f"Claimed 1 ({SUBJECT['1']})"
CLAIMED_2 = f"Claimed 2 ({SUBJECT['2']})"
CLAIMED_3 = f"Claimed 3 ({SUBJECT['3']})"
# without a lock: both read "no owner", both write, both believe they own task 1
NOLOCK, _trace = interleave([(ONE, claim_steps("1", ONE)), (TWO, claim_steps("1", TWO))])
assert NOLOCK == {ONE: CLAIMED_1, TWO: CLAIMED_1}
assert [w for w, step, _ in _trace if step == "read"] == [ONE, TWO]  # both read before either wrote
NOLOCK_WRITES = [owner for _, step, owner in _trace if step == "write"]
assert NOLOCK_WRITES == [ONE, TWO] and load_task("1").owner == TWO
# with the lock: otto-1 claims, otto-2 waits, re-reads, is turned away, and takes task 2
file_tasks()
LOCKED, _trace = interleave([(ONE, claim_steps("1", ONE, LOCK)), (TWO, claim_steps("1", TWO, LOCK))])
REFUSED = LOCKED[TWO]
assert LOCKED[ONE] == CLAIMED_1 and REFUSED == f"Task 1 already owned by {ONE}"
assert [step for w, step, _ in _trace if w == TWO][0] == "wait"
assert claim_task("2", TWO) == CLAIMED_2

# ============================================================================ idle polling (s17)

IDLE_POLL_INTERVAL = 5  # seconds (Learn Claude Code s17's teaching constants)
IDLE_TIMEOUT = 60
POLLS = IDLE_TIMEOUT // IDLE_POLL_INTERVAL


class SimClock:
    """Stands in for the time module: sleep() moves a simulated clock and fires what was scheduled."""

    def __init__(self, events=()):
        self.now, self.events, self.polls = 0, sorted(events, key=lambda e: e[0]), []

    def sleep(self, seconds):
        self.now += seconds
        while self.events and self.events[0][0] <= self.now:
            self.events.pop(0)[1]()
        self.polls.append(self.now)


INBOX = {ONE: [], TWO: []}


def read_inbox(name):
    mail, INBOX[name] = INBOX[name], []
    return mail


def idle_poll(name):
    for _ in range(POLLS):  # 60 s / 5 s
        time.sleep(IDLE_POLL_INTERVAL)
        if read_inbox(name):  # mail first
            return "work"
        for task in scan_unclaimed_tasks():
            claimed = claim_task(task.id, name)
            if claimed.startswith("Claimed"):
                return "work"
    return "timeout"  # then shut down


IDLE_SRC = inspect.getsource(idle_poll).strip()
assert len(IDLE_SRC.split("\n")) == 10 and POLLS == 12  # "twelve empty polls"

FILED_AT = 12  # the lead files the bugs at 12 s (illustrative)
DISK.clear()
time = SimClock([(FILED_AT, file_tasks)])
assert idle_poll(ONE) == "work" and load_task("1").owner == ONE
POLLS_CLAIM = list(time.polls)
assert POLLS_CLAIM == [5, 10, 15] and CLAIM_LOG[-1] == (ONE, CLAIMED_1)  # "the poll at fifteen finds them"
CLAIM_AT = POLLS_CLAIM[-1]
DISK.clear()
time = SimClock()
assert idle_poll(TWO) == "timeout"
POLLS_IDLE = list(time.polls)
assert len(POLLS_IDLE) == POLLS and POLLS_IDLE[-1] == IDLE_TIMEOUT == 60  # "sixty seconds … twelve polls"

# ============================================================================ identity after compaction (s17)


def identity(name, role):
    return {"role": "user", "content":
            f"<identity>You are '{name}', role: {role}. "
            "Continue your work.</identity>"}


def reinject_identity(messages, name, role):
    if len(messages) <= 3:  # just compacted
        messages.insert(0, identity(name, role))


IDENT_SRC = "\n".join(line[4:] for line in inspect.getsource(reinject_identity).strip().split("\n")[1:])
assert IDENT_SRC == "if len(messages) <= 3:  # just compacted\n    messages.insert(0, identity(name, role))"

SESSION = [  # otto-2's messages before compaction (illustrative choices; results computed)
    ("text", f"You are {TWO}, a {ROLE} on the team."),
    ("tool_use", 'claim_task("2")'),
    ("tool_result", CLAIMED_2),
    ("tool_use", 'read_file("cart.py")'),
    ("tool_result", "def total(items): …"),
    ("tool_use", 'bash("pytest -q -k shipping")'),
    ("tool_result", pytest_q(CART, "shipping")[1]),
    ("tool_use", 'edit_file("cart.py", …)'),
    ("tool_result", "edited cart.py"),
]
assert SESSION[6][1] == "1 failed, 13 deselected"
SUMMARY_TEXT = f"[Compacted] Task 2: {SUBJECT['2']}.\ncart.py edited; test it next."
KEEP = 2  # the last turn stays, as Otto's compaction keeps its latest turns (Episode 11)
_messages = [{"role": "assistant" if kind == "tool_use" else "user", "content": body} for kind, body in SESSION]
COMPACTED = [{"role": "user", "content": SUMMARY_TEXT}] + _messages[-KEEP:]
N_COMPACTED = len(COMPACTED)
reinject_identity(COMPACTED, TWO, ROLE)
IDENTITY = COMPACTED[0]["content"]
assert N_COMPACTED == 3 and len(COMPACTED) == 4  # "three messages or fewer"
assert IDENTITY == f"<identity>You are '{TWO}', role: {ROLE}. Continue your work.</identity>"
assert TWO not in SUMMARY_TEXT and ROLE not in SUMMARY_TEXT  # the summary no longer says who it is

# ============================================================================ one folder, two writers (s18's motivation)

READ_BY = {ONE: CART, TWO: CART}  # both read cart.py before either writes
DISK_CART = fix_1(READ_BY[ONE])  # otto-1 writes the whole file, with its fix
AFTER_ONE = DISK_CART
DISK_CART = fix_2(READ_BY[TWO])  # otto-2 writes back the file it read, with only its own fix
SHARED_FAILED, SHARED_SUMMARY = pytest_q(DISK_CART)
assert FIX_1 in AFTER_ONE and FIX_1 not in DISK_CART and FIX_2 in DISK_CART  # "Otto one's fix is gone"
assert SHARED_FAILED == ["test_total"] and SHARED_SUMMARY == "1 failed, 13 passed"  # "the total test fails again"

# ============================================================================ real worktrees (git)

GIT_ENV = {
    **os.environ, "GIT_CONFIG_GLOBAL": os.devnull, "GIT_CONFIG_NOSYSTEM": "1", "LC_ALL": "C",
    "GIT_AUTHOR_NAME": "otto", "GIT_AUTHOR_EMAIL": "otto@tinyshop.test",
    "GIT_COMMITTER_NAME": "otto", "GIT_COMMITTER_EMAIL": "otto@tinyshop.test",
    "GIT_AUTHOR_DATE": "2026-10-22T09:00:00", "GIT_COMMITTER_DATE": "2026-10-22T09:00:00",
}
TASK_K = {"1": "total", "2": "shipping"}  # each Otto's own test: pytest -q -k …


def add_cmd(n):
    return f"git worktree add .worktrees/task-{n} -b wt/task-{n}"


def git(cwd, *args):
    done = subprocess.run(["git", *args], cwd=cwd, env=GIT_ENV, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True)
    return done.returncode, done.stdout.strip()


def worktree_run():
    """tinyshop as a real repo (in a temporary folder inside this episode's folder, deleted after)."""
    with tempfile.TemporaryDirectory(dir=HERE, prefix=".git-run-") as tmp:
        repo = Path(tmp) / "tinyshop"
        repo.mkdir()
        (repo / "cart.py").write_text(CART)
        (repo / ".gitignore").write_text(".worktrees/\n")
        git(repo, "-c", "init.defaultBranch=main", "init", "-q")
        git(repo, "add", ".")
        git(repo, "commit", "-q", "-m", "tinyshop")
        run = NS()
        run.add = {n: git(repo, *add_cmd(n).split()[1:]) for n in ("1", "2")}
        wt = {n: repo / ".worktrees" / f"task-{n}" for n in ("1", "2")}
        for n, fix in (("1", fix_1), ("2", fix_2)):
            (wt[n] / "cart.py").write_text(fix((wt[n] / "cart.py").read_text()))
        run.tests = {n: pytest_q((wt[n] / "cart.py").read_text(), TASK_K[n]) for n in ("1", "2")}
        run.main_untouched = (repo / "cart.py").read_text() == CART
        run.refuse = git(repo, "worktree", "remove", ".worktrees/task-2")  # before otto-2 commits
        run.commit = {n: git(wt[n], "commit", "-q", "-am", SUBJECT[n]) for n in ("1", "2")}
        run.merge = {n: git(repo, "merge", "--no-edit", f"wt/task-{n}") for n in ("1", "2")}
        run.merged = (repo / "cart.py").read_text()
        parents = git(repo, "log", "-1", "--format=%P")[1].split()
        run.parent_subjects = [git(repo, "log", "-1", "--format=%s", p)[1] for p in parents]
        run.head_subject = git(repo, "log", "-1", "--format=%s")[1]
        run.base_subject = git(repo, "log", "--format=%s", "--reverse")[1].split("\n")[0]
        run.remove = {n: git(repo, "worktree", "remove", f".worktrees/task-{n}") for n in ("1", "2")}
    return run


GIT = worktree_run()
assert all(code_ == 0 for code_, _ in GIT.add.values())
ADD_LINE = {n: next(ln for ln in GIT.add[n][1].split("\n") if ln.startswith("Preparing worktree")) for n in ("1", "2")}
assert ADD_LINE["1"] == "Preparing worktree (new branch 'wt/task-1')"
WT_RESULT = {n: GIT.tests[n][1] for n in ("1", "2")}
assert GIT.tests == {"1": ([], "1 passed, 13 deselected"), "2": ([], "1 passed, 13 deselected")}  # "one passed"
assert GIT.main_untouched  # each Otto edited its own copy
assert GIT.refuse[0] != 0
REFUSE = GIT.refuse[1]
REFUSE_LINES = ["fatal: '.worktrees/task-2' contains", "modified or untracked files,", "use --force to delete it"]
assert REFUSE == " ".join(REFUSE_LINES)  # "won't be removed unless you force it"
assert all(code_ == 0 for code_, _ in GIT.commit.values())
assert GIT.merge["1"][0] == 0 and "Fast-forward" in GIT.merge["1"][1]
assert GIT.merge["2"][0] == 0 and "CONFLICT" not in GIT.merge["2"][1]  # "without a conflict"
MERGE_LINES = GIT.merge["2"][1].split("\n")[:2]
assert MERGE_LINES == ["Auto-merging cart.py", "Merge made by the 'ort' strategy."]
assert GIT.merged == fix_2(fix_1(CART))
MERGED_SUMMARY = pytest_q(GIT.merged)[1]
assert MERGED_SUMMARY == "14 passed" and 2 + OTHER_TESTS == 14  # "all fourteen tests pass"
assert GIT.parent_subjects == [SUBJECT["1"], SUBJECT["2"]] and GIT.head_subject == "Merge branch 'wt/task-2'"
assert GIT.base_subject == "tinyshop"
assert all(code_ == 0 for code_, _ in GIT.remove.values())  # clean worktrees go quietly

# ============================================================================ afterwards: the release unblocks

file_tasks()
for _n, _who in (("1", ONE), ("2", TWO)):
    _t = load_task(_n)
    _t.owner, _t.status = _who, "completed"
    save_task(_t)
assert [t.id for t in scan_unclaimed_tasks()] == ["3"]  # "finds the release unblocked"
time = SimClock()
assert idle_poll(ONE) == "work" and CLAIM_LOG[-1] == (ONE, CLAIMED_3) and time.polls == [5]
NAMES = [f"wt/task-{t.id}" for t in list_tasks()]
assert len(set(NAMES)) == len(NAMES)  # one task, one name

COHORT, SAME_NAME = 30, 18  # Li, AI Agents in Depth, ch. 10 (Anthropic's experiment)
SAME = sorted(np.random.default_rng(25).choice(COHORT, SAME_NAME, replace=False).tolist())  # layout only
assert len(SAME) == 18 and COHORT == 30

# ============================================================================ look

WELL = "#18150F"
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
STATUS_COLOR = {"pending": INK, "in_progress": SUN, "completed": MINT}
EDGE_OF = {"pending": EDGE, "in_progress": SUN, "completed": MINT}

LOCK_ROSE = recolored("lock", ROSE, ROSE_FILL, prefix="ep25")
LOCK_SUN = recolored("lock", SUN, SUN_FILL, prefix="ep25")
LOCK_AZURE = recolored("lock", AZURE, AZURE_FILL, prefix="ep25")
MAIL = recolored("mail", CORAL, CORAL_FILL, prefix="ep25")
DOC = {ONE: recolored("doc", AZURE, AZURE_FILL, prefix="ep25"), TWO: recolored("doc", ORCHID, ORCHID_FILL, prefix="ep25")}

FOLDER_SVG = """<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 80">
  <path d="M6 20 Q6 10 16 10 H37 Q41 10 44 13 L49 18 H84 Q94 18 94 28 V64 Q94 74 84 74 H16 Q6 74 6 64 Z" fill="{fill}" stroke="{color}" stroke-width="4.5" stroke-linejoin="round"/>
  <path d="M7 31 H93" fill="none" stroke="{color}" stroke-width="4.5" stroke-linecap="round"/>
</svg>
"""


def folder_icon(color, fill):
    """A flat 2-tone folder (new for this episode): assets/ep25-folder-<hex>.svg."""
    name = f"ep25-folder-{color.lstrip('#').lower()}"
    path = ASSETS / f"{name}.svg"
    out = FOLDER_SVG.format(color=color, fill=fill)
    if not path.exists() or path.read_text() != out:
        path.write_text(out)
    return name


FOLDER = {who: folder_icon(c, f) for who, (c, f) in COLORS.items()}
FOLDER["shared"] = folder_icon(MUTED, CARD_HI)


def cite(s):
    return text(s, 20, MUTED)


def baseline(words, src):
    chars = src.replace(" ", "").replace("\n", "")
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, src, y):
    """Shift a Text so its baseline is at y."""
    words.shift((y - baseline(words, src)) * m.UP)
    return words


def badge(name, size=0.7, label_size=20):
    c, f = COLORS[name]
    bot = otto(size, c, f)
    label = code(name, label_size, c, weight=m.BOLD).next_to(bot, m.DOWN, buff=0.1)
    return m.VGroup(bot, label)


def note(s, color=INK, edge=EDGE, size=20):
    words = code(s, size, color)
    box = card(words.width + 0.5, max(0.52, words.height + 0.3), edge, CARD, radius=0.14, stroke=2)
    return m.VGroup(box, words.move_to(box))


def beside(mob, who_badge, buff=0.35):
    """Place a note to the right of a badge's head."""
    head = who_badge[0][0]
    return mob.move_to([head.get_right()[0] + buff + mob.width / 2, head.get_y(), 0])


class Slot:
    """One place on screen whose content is swapped (a value, a note)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15, source=None):
        on_screen = {id(x) for x in self.scene.get_mobject_family_members()}
        anims = []
        if self.current is not None:
            out = m.FadeOut(self.current, shift=shift * m.UP)
            if id(self.current) not in on_screen:  # introduced earlier in this same line
                out.introducer = True
            anims.append(out)
        if new is not None:
            anims.append(m.FadeIn(new, target_position=source) if source is not None else m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


class CodeCard:
    """A code_block with per-line glyphs and a tinted line cursor."""

    def __init__(self, src, width=None, title=None, size=20):
        self.block = code_block(src, size=size, width=width, title=title)
        self.box, self.body = self.block[0], self.block[1]
        self.body.set_z_index(2)
        if title:
            self.block[2].set_z_index(2)
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n

    def place(self, left=None, top=None, right=None, bottom=None):
        b = self.block
        if left is not None:
            b.shift((left - self.box.get_left()[0]) * m.RIGHT)
        if right is not None:
            b.shift((right - self.box.get_right()[0]) * m.RIGHT)
        if top is not None:
            b.shift((top - self.box.get_top()[1]) * m.UP)
        if bottom is not None:
            b.shift((bottom - self.box.get_bottom()[1]) * m.UP)
        return self

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def base(self, i):
        return baseline(self.lines[i], self.src[i].replace(" ", ""))

    def bar(self, a, b=None, color=SUN):
        b = a if b is None else b
        top, bot = self.base(a) + 0.25, self.base(b) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


class TaskCard:
    """A task file on the board: id, subject, status, owner, and a blocker badge (s12's board)."""

    H = 1.62

    def __init__(self, scene, task, center, width=4.3):
        self.scene, self.id = scene, task.id
        self.box = card(width, self.H, EDGE_OF[task.status], CARD, radius=0.16, stroke=2.5).move_to([*center[:2], 0])
        left, top = self.box.get_left()[0], self.box.get_top()[1]
        num = code(task.id, 24, SUN, weight=m.BOLD)
        name = text(task.subject, 24, INK, weight=m.BOLD)
        self.head = m.VGroup(num, name).arrange(m.RIGHT, buff=0.2)
        self.head.move_to([left + 0.3, 0, 0], aligned_edge=m.LEFT)
        sit(num, task.id, top - 0.5)
        sit(name, task.subject, top - 0.5)
        self.bases = [top - 0.98, top - 1.38]
        keys = m.VGroup(code("status", 22, MUTED), code("owner", 22, MUTED))
        for key, src, y in zip(keys, ("status", "owner"), self.bases):
            key.move_to([left + 0.3, 0, 0], aligned_edge=m.LEFT)
            sit(key, src, y)
        self.vx = left + 1.55
        self.frame = m.VGroup(self.box, self.head, keys)
        self.status = Slot(scene, self.status_text(task.status))
        self.owner = Slot(scene, self.owner_text(task.owner))
        self.lock = None
        if task.blockedBy:
            pad = icon(LOCK_ROSE, 0.3)
            ids = code(" ".join(task.blockedBy), 20, ROSE, weight=m.BOLD)
            inner = m.VGroup(pad, ids).arrange(m.RIGHT, buff=0.12)
            bx = card(inner.width + 0.34, 0.46, ROSE, ROSE_FILL, radius=0.14, stroke=2)
            self.lock = m.VGroup(bx, inner.move_to(bx))
            self.lock.move_to([self.box.get_right()[0] - 0.22 - self.lock.width / 2, top - 0.42, 0])

    def status_text(self, status):
        words = code(status, 22, STATUS_COLOR[status]).move_to([self.vx, 0, 0], aligned_edge=m.LEFT)
        return sit(words, status, self.bases[0])

    def owner_text(self, owner):
        s = owner or "null"
        words = code(s, 22, COLORS[owner][0] if owner else MUTED, weight=m.BOLD if owner else m.NORMAL)
        return sit(words.move_to([self.vx, 0, 0], aligned_edge=m.LEFT), s, self.bases[1])

    def intro(self, source=None, shift=0.2 * m.UP):
        first = m.FadeIn(self.frame, target_position=source) if source is not None else m.FadeIn(self.frame, shift=shift)
        parts = [first, m.FadeIn(self.status.current), m.FadeIn(self.owner.current)]
        if self.lock is not None:
            parts.append(m.FadeIn(self.lock))
        return m.AnimationGroup(*parts)

    def set_status(self, status):
        return m.AnimationGroup(self.status.to(self.status_text(status)), self.box.animate.set_stroke(EDGE_OF[status]))

    def set_owner(self, owner, source=None):
        return self.owner.to(self.owner_text(owner), source=source)


def board(scene, tasks, centers, width=4.3):
    return [TaskCard(scene, t, c, width) for t, c in zip(tasks, centers)]


def fresh_tasks():
    return [Task("1", SUBJECT["1"]), Task("2", SUBJECT["2"]), Task("3", SUBJECT["3"], blockedBy=["1", "2"])]


COLUMN = [(4.3, 1.59), (4.3, -0.16), (4.3, -1.91)]  # the board, stacked on the right


def msg_row(body, kind, width=5.0, size=20):
    """One message as a row: a stripe in its colour, the content (mono for tool blocks)."""
    accent = {"text": AZURE, "summary": AZURE, "identity": AZURE, "tool_use": SUN, "tool_result": MINT}[kind]
    if kind in ("text", "summary"):
        words = text(body, size, INK)
    elif kind == "identity":
        words = code(body, size, INK, t2c={r"</?identity>": AZURE})
    else:
        words = code(body, size, accent)
    box = card(width, max(0.48, words.height + 0.3), EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=box.height - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    words.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
    if "\n" not in body:
        sit(words, body, box.get_y() - 0.09)
    return m.VGroup(box, stripe, words).set_z_index(1)


def stack(rows, top, x, gap=0.08):
    """Target centres for rows stacked down from y = top."""
    out, y = [], top
    for r in rows:
        out.append(np.array([x, y - r.height / 2, 0]))
        y -= r.height + gap
    return out


def commit_dot(at, color):
    return m.Circle(radius=0.15).set_stroke(color, 4).set_fill(CARD, 1).move_to(at)


def edge(a, b, color):
    a, b = np.array(a, dtype=float), np.array(b, dtype=float)
    dx = (b[0] - a[0]) * 0.5
    return m.CubicBezier(a, a + [dx, 0, 0], b - [dx, 0, 0], b).set_stroke(color, 4)


# ============================================================================ the episode


class Ep25AutonomyWorktrees(Episode):
    number = 25

    def lesson(self):
        self.hook()
        self.two_bugs()
        self.idle()
        self.claim()
        self.identity()
        self.one_folder()
        self.worktrees()
        self.names()
        self.merge()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Callback to Episode 24: a lead Otto mailed every job to its teammates, one message at a time. While the lead is busy, new tasks queue up and the teammates idle: the lead is the bottleneck. Learn Claude Code s16 'What's Next': the lead assigns every task by hand → s17 autonomy. This episode: teammates poll a shared board, claim work, and fix it in a folder of their own (s17, s18). The pile of tasks is illustrative. Facts as of October 2026.")
        ep24 = pill("Episode 24 · Agent Teams", AZURE, AZURE_FILL, 22).move_to([0, 2.45, 0])
        lead = badge(LEAD, 1.0).move_to([-4.6, 0.35, 0])
        one = badge(ONE, 0.75).move_to([4.6, 1.15, 0])
        two = badge(TWO, 0.75).move_to([4.6, -1.55, 0])
        mails = m.VGroup(*[icon(MAIL, 0.42) for _ in range(2)])
        mails[0].next_to(one[0], m.LEFT, buff=0.45)
        mails[1].next_to(two[0], m.LEFT, buff=0.45)
        pile = m.VGroup(*[chip(f"task {n}", SUN, SUN_FILL, width=1.9, height=0.56, size=24, mono=True) for n in (3, 4, 5)])
        pile.arrange(m.DOWN, buff=0.14).move_to([-1.75, 0.55, 0])
        waits = m.VGroup(*[text("waiting…", 22, MUTED).next_to(b, m.DOWN, buff=0.14) for b in (one, two)])
        flag = Slot(self)
        busy = pill("busy", SUN, SUN_FILL, 20).next_to(lead, m.DOWN, buff=0.3)
        neck = pill("bottleneck", ROSE, ROSE_FILL, 22).next_to(lead, m.DOWN, buff=0.3)

        self.say(
            "[In episode twenty-four], [the lead handed out every job], [one message at a time.]",
            m.FadeIn(ep24, shift=0.15 * m.DOWN),
            m.AnimationGroup(m.FadeIn(lead, shift=0.2 * m.UP), m.FadeIn(one, shift=0.2 * m.UP), m.FadeIn(two, shift=0.2 * m.UP), lag_ratio=0.2),
            m.Succession(m.FadeIn(mails[0], target_position=lead[0]), m.FadeIn(mails[1], target_position=lead[0])),
        )
        self.say(
            "[But while the lead is busy], [new tasks pile up], [and the teammates just wait.]",
            m.AnimationGroup(m.Circumscribe(lead[0], color=SUN, buff=0.12), flag.to(busy)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.25 * m.DOWN) for c in pile], lag_ratio=0.35),
            m.AnimationGroup(m.FadeIn(waits, shift=0.1 * m.UP), flag.to(neck), m.Circumscribe(lead[0], color=ROSE, buff=0.12)),
        )
        steps = m.VGroup(
            chip("poll a board", AZURE, AZURE_FILL, width=3.2, height=0.8, size=28),
            chip("claim a task", SUN, SUN_FILL, width=3.2, height=0.8, size=28),
            chip("own folder", MINT, MINT_FILL, width=3.2, height=0.8, size=28),
        ).arrange(m.RIGHT, buff=0.9).move_to([0, -1.35, 0])
        links = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.12) for a, b in zip(steps, steps[1:])])
        self.say(
            "[So today, the teammates find their own work]: [they poll a shared board], [claim a task], [and fix it in a folder of their own.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(ep24, lead, mails, pile, waits, neck), shift=0.2 * m.DOWN),
                one.animate.move_to([-1.4, 1.05, 0]), two.animate.move_to([1.4, 1.05, 0]),
            ),
            m.FadeIn(steps[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.GrowArrow(links[0]), m.FadeIn(steps[1], shift=0.2 * m.UP)),
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(steps[2], shift=0.2 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 · two bugs

    def two_bugs(self):
        self.next_section("1 · Two bugs, one board", notes=f"tinyshop's cart.py with two bugs: total() skips the last item (the series' bug, line {LINE_1 + 1}) and shipping() is free only above 50, though the rule (and the new test) says from 50 (line {LINE_2 + 1}; illustrative). pytest -q, computed in the scene: '{SUMMARY_0}'. The lead files each bug as a task file (.tasks/{{id}}.json, s12: id, subject, status pending, owner null, blockedBy) and a third, '{SUBJECT['3']}', blocked by 1 and 2. Facts as of October 2026.")
        cart = CodeCard(CART.rstrip("\n"), title="tinyshop/cart.py").place(left=-6.6, top=2.45)
        bug1, bug2 = cart.part(LINE_1, "- 1"), cart.part(LINE_2, "> 50")
        term = terminal(["$ pytest -q", "! FAILED test_total", "! FAILED test_free_shipping", f"! {SUMMARY_0}"], width=4.9, title="tinyshop")
        term.move_to([6.6 - term.width / 2, 2.45 - term.height / 2, 0])
        rows = term[4]
        cards = board(self, fresh_tasks(), COLUMN)
        caption = code(".tasks/ · one file per task", 20, MUTED).next_to(cards[0].box, m.UP, buff=0.14).align_to(cards[0].box, m.LEFT)

        self.say(
            "[Here's tinyshop, with two bugs this time]: [two tests fail.]",
            m.AnimationGroup(self.new_header("01 · two bugs", "Two bugs, one board"), m.FadeIn(cart.block, shift=0.2 * m.UP)),
            m.LaggedStart(m.FadeIn(term[:4]), *[m.FadeIn(r, shift=0.1 * m.UP) for r in rows], lag_ratio=0.25),
        )
        self.say(
            "[The total still skips the last item], [and an order of exactly fifty dollars still pays for shipping.]",
            m.AnimationGroup(bug1.animate.set_color(ROSE), m.Circumscribe(bug1, color=ROSE, buff=0.06), m.Circumscribe(rows[1], color=ROSE, buff=0.06)),
            m.AnimationGroup(bug2.animate.set_color(ROSE), m.Circumscribe(bug2, color=ROSE, buff=0.06), m.Circumscribe(rows[2], color=ROSE, buff=0.06)),
        )
        self.say(
            "[The lead files each bug as a task], [pending, with no owner], [plus a third: tag a release, once both are fixed.]",
            m.AnimationGroup(m.FadeOut(term, shift=0.2 * m.DOWN), cards[0].intro(source=rows[1]), cards[1].intro(source=rows[2]), m.FadeIn(caption)),
            m.AnimationGroup(*[m.Circumscribe(m.VGroup(c.status.current, c.owner.current), color=INK, buff=0.07) for c in cards[:2]]),
            m.Succession(cards[2].intro(), m.Circumscribe(cards[2].lock, color=ROSE, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 · work and idle

    def idle(self):
        self.next_section("2 · Work and idle", notes=f"Learn Claude Code s17: a teammate cycles WORK (Episode 2's loop: inbox → model → tools, until the reply has no tool call) → IDLE → back to WORK when mail or a claimable task appears, or SHUTDOWN after {IDLE_TIMEOUT} s idle. idle_poll (the code shown ran in the scene against a simulated clock; simplified: s17 also answers a shutdown_request and appends inbox messages): every IDLE_POLL_INTERVAL = {IDLE_POLL_INTERVAL} s, the inbox first, then scan_unclaimed_tasks and claim_task. Run 1: the bugs are filed at {FILED_AT} s (illustrative); polls at {POLLS_CLAIM} s; the {CLAIM_AT} s poll claims task 1. Run 2: an empty board, {len(POLLS_IDLE)} polls ({POLLS_IDLE[0]}…{POLLS_IDLE[-1]} s), then 'timeout' → shut down. These are the teaching constants; Dive into CC source (the site's reading, unverified): a 500 ms polling loop and fs.watch on the tasks directory. Facts as of October 2026.")
        cx = -4.6
        work = chip("work", CORAL, CORAL_FILL, width=1.8, height=0.72, size=28).move_to([cx, 1.4, 0])
        idle = chip("idle", AZURE, AZURE_FILL, width=1.8, height=0.72, size=28).move_to([cx, -0.5, 0])
        down = chip("shut down", ROSE, ROSE_FILL, width=2.4, height=0.72, size=28).move_to([cx, -2.45, 0])
        go_idle = m.ArcBetweenPoints(work.get_right() + [0.08, -0.08, 0], idle.get_right() + [0.08, 0.08, 0], angle=-2.0)
        go_idle.set_stroke(MUTED, 4).add_tip(tip_length=0.2)
        go_work = m.ArcBetweenPoints(idle.get_left() + [-0.08, 0.08, 0], work.get_left() + [-0.08, -0.08, 0], angle=-2.0)
        go_work.set_stroke(MUTED, 4).add_tip(tip_length=0.2)
        to_down = arrow(idle.get_bottom(), down.get_top(), MUTED, buff=0.12)
        down_label = code(f"{IDLE_TIMEOUT} s idle", 20, MUTED).next_to(to_down, m.RIGHT, buff=0.2)
        loop_pill = pill("the loop from Episode 2", CORAL, CORAL_FILL, 20).next_to(work, m.UP, buff=0.3)

        poll = CodeCard(IDLE_SRC).place(right=6.6, top=2.45)
        bar = poll.bar(1, 2, color=AZURE)
        x0, x1, ty = -0.95, 6.1, -2.5

        def tx(t):
            return x0 + (x1 - x0) * t / IDLE_TIMEOUT

        axis = m.Line([x0, ty, 0], [x1 + 0.25, ty, 0]).set_stroke(FAINT, 2.5)
        dots = m.VGroup(*[m.Dot([tx(t), ty, 0], radius=0.075, color=FAINT) for t in POLLS_IDLE])
        zero = code("0 s", 20, MUTED).next_to([x0, ty, 0], m.DOWN, buff=0.22)
        end = code(f"{IDLE_TIMEOUT} s", 20, MUTED).next_to([tx(IDLE_TIMEOUT), ty, 0], m.DOWN, buff=0.22)
        polls_label = code("polls", 20, MUTED).next_to([x0, ty, 0], m.LEFT, buff=0.25)
        timeline = m.VGroup(axis, dots, zero, end, polls_label)
        filed = pill(f"bugs filed · {FILED_AT} s", SUN, SUN_FILL, 20).move_to([tx(FILED_AT) + 0.35, ty + 0.58, 0])
        tick = m.DashedLine([tx(FILED_AT), ty + 0.32, 0], [tx(FILED_AT), ty - 0.05, 0], dash_length=0.05).set_stroke(SUN, 2.5)
        k = POLLS_CLAIM.index(CLAIM_AT)
        ring = m.Circle(radius=0.16).set_stroke(MINT, 3).move_to(dots[k])
        at15 = code(f"{CLAIM_AT} s", 20, MINT).next_to(dots[k], m.DOWN, buff=0.22)
        got = code(CLAIMED_1, 20, MINT).next_to(axis, m.UP, buff=0.2).align_to(axis, m.RIGHT)
        twelve = code(f"{len(POLLS_IDLE)} polls · nothing to claim", 20, INK).next_to(axis, m.UP, buff=0.18).align_to(axis, m.RIGHT).shift(0.25 * m.LEFT)
        src = Slot(self)
        cite_a = cite("learn-claude-code · s17").move_to([x0, -3.38, 0], aligned_edge=m.LEFT)
        cite_b = cite("learn-claude-code · s17 · reading Claude Code's source").move_to([x0, -3.38, 0], aligned_edge=m.LEFT)
        cc = m.VGroup(
            title_text("Claude Code", 26, INK),
            m.VGroup(pill("polls every 500 ms", AZURE, AZURE_FILL, 22), pill("watches the task folder", AZURE, AZURE_FILL, 22)).arrange(m.RIGHT, buff=0.3),
        )
        cc.arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT).next_to(poll.box, m.DOWN, buff=0.35).align_to([x0, 0, 0], m.LEFT)

        self.say(
            "[A teammate now runs in two modes.] [In work mode], [it runs the loop from episode two], [until the model stops calling tools.]",
            m.AnimationGroup(self.new_header("02 · work and idle", "Work, idle, repeat"), m.FadeIn(work, shift=0.2 * m.UP), m.FadeIn(idle, shift=0.2 * m.UP)),
            m.Circumscribe(work, color=CORAL, buff=0.1),
            m.FadeIn(loop_pill, shift=0.15 * m.DOWN),
            m.AnimationGroup(m.Create(go_idle), m.Circumscribe(idle, color=AZURE, buff=0.1)),
        )
        self.say(
            "[Then it goes idle]: [every five seconds], [it checks its inbox first], [then the task board.]",
            m.AnimationGroup(m.FadeIn(poll.block, shift=0.2 * m.UP), m.FadeIn(bar)),
            m.AnimationGroup(m.Create(axis), m.FadeIn(m.VGroup(zero, end, polls_label)), m.LaggedStart(*[m.FadeIn(d, scale=0.5) for d in dots], lag_ratio=0.1)),
            m.Transform(bar, poll.bar(3, 4, color=AZURE)),
            m.Transform(bar, poll.bar(5, 8, color=AZURE)),
        )
        self.say(
            "[Say the bugs were filed at twelve seconds.] [The poll at fifteen finds them], [claims one], [and it's back to work.]",
            m.AnimationGroup(m.FadeIn(filed, shift=0.15 * m.DOWN), m.Create(tick)),
            m.AnimationGroup(
                m.Succession(*[dots[i].animate(run_time=0.3).set_color(MUTED) for i in range(k)], dots[k].animate(run_time=0.3).set_color(MINT)),
                m.FadeIn(ring, scale=0.6), m.FadeIn(at15), m.Transform(bar, poll.bar(5, color=AZURE)),
            ),
            m.AnimationGroup(m.Transform(bar, poll.bar(6, 8, color=MINT)), m.FadeIn(got, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(go_work), m.Circumscribe(work, color=CORAL, buff=0.1)),
        )
        self.say(
            "[If nothing turns up for sixty seconds], [that's twelve empty polls], [the teammate shuts itself down.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(filed, tick, ring, at15, got)), dots.animate.set_color(FAINT),
                m.Transform(bar, poll.bar(9, color=ROSE)), m.FadeOut(polls_label),
            ),
            m.AnimationGroup(m.LaggedStart(*[d.animate.set_color(INK) for d in dots], lag_ratio=0.15), m.FadeIn(twelve, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(to_down), m.FadeIn(down, shift=0.15 * m.DOWN), m.FadeIn(down_label), dots[-1].animate.set_color(ROSE)),
        )
        self.say(
            "[Those are Learn Claude Code's teaching numbers.] [By its reading of the source], [Claude Code checks every half second], [and also watches the task folder.]",
            m.AnimationGroup(src.to(cite_a), m.Circumscribe(poll.part(2, "IDLE_POLL_INTERVAL"), color=AZURE, buff=0.05)),
            m.AnimationGroup(m.FadeOut(m.VGroup(axis, dots, zero, end, twelve)), src.to(cite_b), m.FadeIn(cc[0], shift=0.1 * m.UP)),
            m.FadeIn(cc[1][0], shift=0.1 * m.UP),
            m.FadeIn(cc[1][1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 · claiming

    def claim(self):
        self.next_section("3 · Claim, under a lock", notes=f"Learn Claude Code s17: scan_unclaimed_tasks keeps tasks that are pending, have no owner and can_start (every blocker completed); both Ottos see {SEEN[ONE]} (task 3 waits for 1 and 2). They started together, so they poll on the same 5 s grid. Without a lock (the teaching version has none; the page says concurrent claims can still race), the two claims interleave read, read, write, write: both get '{CLAIMED_1}', the file ends with owner {TWO} (owners written: {NOLOCK_WRITES}). With a lock around read-check-write: otto-1 claims, otto-2 waits, re-reads and gets '{REFUSED}' (s17's message), then claims the next task: '{CLAIMED_2}'. Claude Code (the site's reading of its source, s12/s17): claimTask does its read-modify-write inside a file lock (proper-lockfile), 'up to 30 retries, exponential backoff 5-100ms'. All strings computed in the scene. Facts as of October 2026.")
        cards = board(self, fresh_tasks(), COLUMN)
        scan = CodeCard(SCAN_SRC).place(left=-6.1)
        scan.block.set_y(0.25)
        oks = [pill("claimable", MINT, MINT_FILL, 20).next_to(c.box, m.LEFT, buff=0.2) for c in cards[:2]]
        blocked = pill("blocked", ROSE, ROSE_FILL, 20).next_to(cards[2].box, m.LEFT, buff=0.2)
        one = badge(ONE, 0.7).move_to([-5.6, 1.3, 0])
        two = badge(TWO, 0.7).move_to([-5.6, -1.3, 0])
        n1, n2 = Slot(self), Slot(self)

        def say1(s, color=INK, edge=EDGE):
            return beside(note(s, color, edge), one)

        def say2(s, color=INK, edge=EDGE):
            return beside(note(s, color, edge), two)

        lock_at = cards[0].box.get_left() + [-0.4, 0.0, 0]
        padlock = Slot(self)
        row_y = -2.95
        nolock = pill("no lock", ROSE, ROSE_FILL, 22).move_to([-5.4, row_y, 0])
        clash = text("two Ottos on one bug", 24, ROSE, weight=m.SEMIBOLD).next_to(nolock, m.RIGHT, buff=0.3)
        steps = m.VGroup(*[chip(s, SUN, SUN_FILL, width=1.35, height=0.56, size=24, mono=True) for s in ("read", "check", "write")])
        steps.arrange(m.RIGHT, buff=0.45)
        step_links = m.VGroup(*[arrow(a.get_right(), b.get_left(), SUN, buff=0.06) for a, b in zip(steps, steps[1:])])
        inner = m.VGroup(icon(LOCK_SUN, 0.42), m.VGroup(steps, step_links), text("one step", 22, SUN, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.35)
        locked = m.VGroup(card(inner.width + 0.5, 0.86, SUN, CARD, radius=0.18, stroke=2), inner)
        inner.move_to(locked[0])
        locked.move_to([-6.4 + locked.width / 2, row_y, 0])
        teach = m.VGroup(code("teaching version:", 20, MUTED), code("no lock", 20, ROSE)).arrange(m.RIGHT, buff=0.2)
        teach.move_to([-6.4, -2.6, 0], aligned_edge=m.LEFT)
        cc_lock = m.VGroup(icon(LOCK_SUN, 0.34), text("Claude Code: a file lock,", 22, INK), code("up to 30 retries · 5–100 ms backoff", 20, SUN)).arrange(m.RIGHT, buff=0.2)
        cc_lock.move_to([-6.4, -3.2, 0], aligned_edge=m.LEFT)
        src = cite("learn-claude-code · s12, s17 · reading Claude Code's source").move_to([-6.4, 2.45, 0], aligned_edge=m.LEFT)

        self.say(
            "[A task can be claimed only if three things hold]: [it's pending], [it has no owner], [and nothing blocks it.]",
            m.AnimationGroup(self.new_header("03 · claim", "One owner per task"), *[c.intro() for c in cards], m.FadeIn(scan.block, shift=0.2 * m.UP)),
            m.AnimationGroup(m.Circumscribe(scan.part(2, 't.status == "pending"'), color=SUN, buff=0.05), *[m.Circumscribe(c.status.current, color=INK, buff=0.06) for c in cards]),
            m.AnimationGroup(m.Circumscribe(scan.part(3, "not t.owner"), color=SUN, buff=0.05), *[m.Circumscribe(c.owner.current, color=INK, buff=0.06) for c in cards]),
            m.AnimationGroup(m.Circumscribe(scan.part(4, "can_start(t.id)"), color=SUN, buff=0.05), m.Circumscribe(cards[2].lock, color=ROSE, buff=0.06)),
        )
        self.say(
            "[Task three fails the last check]: [it waits for one and two.]",
            m.AnimationGroup(m.FadeIn(blocked, shift=0.15 * m.LEFT), *[m.FadeIn(p, shift=0.15 * m.LEFT) for p in oks]),
            m.Circumscribe(cards[2].lock[1][1], color=ROSE, buff=0.06),
        )
        self.say(
            "[Both Ottos started together, so they poll at the same moment], [and both see task one is free.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(scan.block, blocked, *oks), shift=0.2 * m.DOWN),
                m.FadeIn(one, shift=0.2 * m.UP), m.FadeIn(two, shift=0.2 * m.UP),
                n1.to(say1(f"poll at {CLAIM_AT} s", MUTED)), n2.to(say2(f"poll at {CLAIM_AT} s", MUTED)),
            ),
            m.AnimationGroup(n1.to(say1("task 1 is free")), n2.to(say2("task 1 is free")), m.Circumscribe(cards[0].box, color=INK, buff=0.06)),
        )
        self.say(
            "[Without a lock], [both read no owner], [both write their own name], [and both set to work on the same bug.]",
            m.FadeIn(nolock, shift=0.15 * m.UP),
            m.AnimationGroup(n1.to(say1("read: owner null")), n2.to(say2("read: owner null")), m.Circumscribe(cards[0].owner.current, color=ROSE, buff=0.06)),
            m.Succession(cards[0].set_owner(NOLOCK_WRITES[0], source=one[0]), cards[0].set_owner(NOLOCK_WRITES[1], source=two[0])),
            m.AnimationGroup(
                n1.to(say1(NOLOCK[ONE], ROSE, ROSE)), n2.to(say2(NOLOCK[TWO], ROSE, ROSE)),
                cards[0].set_status("in_progress"), m.FadeIn(clash, shift=0.1 * m.UP),
            ),
        )
        self.say(
            "[So the claim takes a lock]: [read, check and write become one step.]",
            m.AnimationGroup(
                cards[0].set_owner(None), cards[0].set_status("pending"), n1.to(None), n2.to(None),
                m.FadeOut(m.VGroup(nolock, clash), shift=0.15 * m.DOWN), padlock.to(icon(LOCK_SUN, 0.5).move_to(lock_at)),
            ),
            m.FadeIn(locked, shift=0.15 * m.UP),
        )
        self.say(
            "[Otto one gets the lock and claims task one.] [Otto two waits], [then reads the owner], [and is turned away.]",
            m.AnimationGroup(
                padlock.to(icon(LOCK_AZURE, 0.5).move_to(lock_at)), cards[0].set_owner(ONE, source=one[0]),
                cards[0].set_status("in_progress"), n1.to(say1(LOCKED[ONE], INK, AZURE)),
            ),
            n2.to(say2("waiting for the lock…", MUTED)),
            m.AnimationGroup(padlock.to(icon(LOCK_SUN, 0.5).move_to(lock_at)), n2.to(say2(f"read: owner {ONE}"))),
            n2.to(say2(REFUSED, ROSE, ROSE)),
        )
        self.say(
            "[Task one is already owned by Otto one], [so Otto two claims the next free task], [task two.]",
            m.Circumscribe(cards[0].owner.current, color=AZURE, buff=0.06),
            m.AnimationGroup(cards[1].set_owner(TWO, source=two[0]), cards[1].set_status("in_progress")),
            m.AnimationGroup(n2.to(say2(CLAIMED_2, INK, ORCHID)), m.Circumscribe(cards[1].box, color=ORCHID, buff=0.06)),
        )
        self.say(
            "[Learn Claude Code's teaching version skips the lock.] [Claude Code takes a file lock], [retrying up to thirty times, with a short backoff.]",
            m.AnimationGroup(m.FadeOut(locked, shift=0.15 * m.DOWN), m.FadeIn(teach, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(cc_lock[:2], shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(cc_lock[2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 · identity

    def identity(self):
        self.next_section("4 · Who am I?", notes=f"Learn Claude Code s17: a teammate runs for many turns, so its history gets compacted (Episode 11; s08's autoCompact). In the teaching version the teammate's identity lives in its messages, and the summary may drop it (real Claude Code keeps its system prompt through compaction, the page notes). s17's rule: 'Short messages suggest compression happened — re-inject identity': if len(messages) <= 3, insert a user message '{IDENTITY}' at index 0. Run in the scene: otto-2's {len(SESSION)} messages (illustrative session; results computed, e.g. '{SESSION[6][1]}') → summary + the last turn = {N_COMPACTED} messages → identity inserted → {len(COMPACTED)}. Simplified: identity() builds the message (shown in the column). Facts as of October 2026.")
        px, pw, ph = 3.95, 5.3, 5.6
        panel = card(pw, ph, EDGE, WELL, radius=0.2, stroke=2).move_to([px, -0.5, 0])
        title = code(f"{TWO} · messages", 20, MUTED).next_to(panel, m.UP, buff=0.12).align_to(panel, m.LEFT)
        meter = Meter(1.9, 0.0, SUN, 0.16).next_to(panel, m.UP, buff=0.18).align_to(panel, m.RIGHT)
        top = panel.get_top()[1] - 0.16
        rw = pw - 0.3
        rows = [msg_row(body, kind, rw) for kind, body in SESSION]
        for r, p in zip(rows, stack(rows, top, px)):
            r.move_to(p)
        summary = msg_row(SUMMARY_TEXT, "summary", rw)
        after = [summary, rows[-2], rows[-1]]
        p_after = stack(after, top, px)
        summary.move_to(p_after[0])
        ident_body = "<identity>You are 'otto-2',\nrole: coder. Continue\nyour work.</identity>"
        assert ident_body.replace("\n", " ") == IDENTITY
        ident = msg_row(ident_body, "identity", rw)
        p_final = stack([ident] + after, top, px)
        ident.move_to(p_final[0])

        two = badge(TWO, 0.75).move_to([-5.6, 1.5, 0])
        thought = Slot(self)
        ghost = rows[0].copy().move_to([-3.6, 0.05, 0])
        ghost_cap = code("its first message", 20, MUTED).next_to(ghost, m.UP, buff=0.12).align_to(ghost, m.LEFT)
        lost = pill("not in the summary", ROSE, ROSE_FILL, 20).next_to(ghost, m.DOWN, buff=0.22).align_to(ghost, m.LEFT)
        ep11 = pill("Episode 11 · Compaction", SUN, SUN_FILL, 20).move_to([-3.6, 0.05, 0])
        fix = CodeCard(IDENT_SRC, title="after compaction").place(left=-6.6, bottom=-3.3)
        bar = fix.bar(0, color=AZURE)
        count = Slot(self)

        def count_at(n):
            return code(f"{n} messages", 20, SUN, weight=m.BOLD).move_to([panel.get_left()[0] - 0.25, -0.45, 0], aligned_edge=m.RIGHT)

        self.say(
            "[Teammates can run for a long time], [so their context fills up], [and gets compacted, as in episode eleven.]",
            m.AnimationGroup(
                self.new_header("04 · identity", "Who am I, again?"), m.FadeIn(m.VGroup(panel, title, meter)),
                m.FadeIn(two, shift=0.2 * m.UP), *[m.FadeIn(r, shift=0.15 * m.UP) for r in rows[:3]],
            ),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.UP) for r in rows[3:]], lag_ratio=0.2, run_time=1.0), meter.fill_to(0.86, run_time=1.0)),
            m.AnimationGroup(
                m.ReplacementTransform(m.VGroup(*rows[:-2]), summary),
                rows[-2].animate.move_to(p_after[1]), rows[-1].animate.move_to(p_after[2]),
                meter.fill_to(0.24), m.FadeIn(ep11, shift=0.1 * m.UP),
            ),
        )
        self.say(
            "[The summary keeps the work], [but the line that said who this agent is] [may be gone.]",
            m.Circumscribe(summary, color=AZURE, buff=0.05),
            m.AnimationGroup(m.FadeOut(ep11), m.FadeIn(ghost, target_position=summary), m.FadeIn(ghost_cap)),
            m.AnimationGroup(m.FadeIn(lost, shift=0.1 * m.UP), thought.to(beside(note("who am I?", ROSE, ROSE), two))),
        )
        self.say(
            "[So when the history is that short], [three messages or fewer], [the harness puts the identity back at the top.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(ghost, ghost_cap, lost), shift=0.15 * m.DOWN), m.FadeIn(fix.block, shift=0.2 * m.UP), m.FadeIn(bar)),
            m.AnimationGroup(m.Circumscribe(fix.part(0, "len(messages) <= 3"), color=SUN, buff=0.05), count.to(count_at(N_COMPACTED))),
            m.AnimationGroup(
                *[mob.animate.move_to(p) for mob, p in zip(after, p_final[1:])],
                m.FadeIn(ident, target_position=fix.block), m.Transform(bar, fix.bar(1, color=AZURE)),
                count.to(count_at(len(COMPACTED))),
                meter.fill_to(0.3),
            ),
        )
        self.say(
            "[Otto two knows again which teammate it is], [and which task is its own.]",
            m.AnimationGroup(thought.to(beside(note(f"I'm {TWO}, a {ROLE}", MINT, MINT), two)), m.Circumscribe(ident, color=AZURE, buff=0.05)),
            m.AnimationGroup(m.Circumscribe(glyphs(summary[2], "Task 2"), color=SUN, buff=0.05), thought.to(beside(note(f"I'm {TWO} · my task: 2", MINT, MINT), two))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 · one folder

    def one_folder(self):
        self.next_section("5 · One folder, two writers", notes=f"Learn Claude Code s18's motivation (its example: two teammates both write config.py in one directory and overwrite each other, with no clean rollback). Ours, computed: both Ottos read cart.py; otto-1 writes the whole file with its fix; otto-2 writes back the file it had read, with only its own fix, so otto-1's fix is gone and pytest -q gives '{SHARED_SUMMARY}' (FAILED {SHARED_FAILED[0]}). Li, AI Agents in Depth, ch. 10, failure mode one: 'the later write overwrites the earlier one' (Episode 24's caution). Facts as of October 2026.")
        folder = card(6.9, 3.15, MUTED, WELL, radius=0.2, stroke=2.5).move_to([0, 0.7, 0])
        fhead = m.VGroup(icon(FOLDER["shared"], 0.36), code("tinyshop/", 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)
        fhead.move_to(folder.get_corner(m.UL) + [0.35, -0.38, 0], aligned_edge=m.LEFT)
        filecard = card(6.3, 2.05, EDGE, CARD, radius=0.14, stroke=2).move_to([0, 0.38, 0])
        fcap = code("cart.py", 20, MUTED).move_to(filecard.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        written = Slot(self)
        la0, la1 = CART_LINES[LINE_1].strip(), fix_1(CART).split("\n")[LINE_1].strip()
        lb0, lb1 = CART_LINES[LINE_2].strip(), fix_2(CART).split("\n")[LINE_2].strip()
        assert DISK_CART.split("\n")[LINE_1].strip() == la0 and DISK_CART.split("\n")[LINE_2].strip() == lb1
        ya, yb = 0.42, -0.4
        lx = filecard.get_left()[0] + 0.3

        def line(s, rules, y):
            return sit(code(s, 20, INK, t2c=rules).move_to([lx, 0, 0], aligned_edge=m.LEFT), s, y)

        A, B = Slot(self, line(la0, {r"- 1": ROSE}, ya)), Slot(self, line(lb0, {r"> 50": ROSE}, yb))
        dots = code("⋮", 20, FAINT).move_to([lx + 0.1, (ya + yb) / 2 + 0.06, 0])
        one = badge(ONE, 0.85).move_to([-5.35, 0.75, 0])
        two = badge(TWO, 0.85).move_to([5.35, 0.75, 0])
        cwd = m.VGroup(*[code("cwd: tinyshop/", 20, MUTED).next_to(b, m.DOWN, buff=0.15) for b in (one, two)])
        docs = {who: icon(DOC[who], 0.6).next_to(c, m.DOWN, buff=0.25) for who, c in zip((ONE, TWO), cwd)}
        reads = m.VGroup(*[code("read", 20, MUTED).next_to(docs[w], m.DOWN, buff=0.1) for w in (ONE, TWO)])
        term = terminal(["$ pytest -q", f"! FAILED {SHARED_FAILED[0]}", f"! {SHARED_SUMMARY}"], width=4.6, title="tinyshop/")
        term.move_to([0, -2.3, 0])
        asks = m.VGroup(*[title_text("?", 40, ROSE).next_to(b[0], m.UP, buff=0.15) for b in (one, two)])

        def wrote(who):
            c = COLORS[who][0]
            return code(f"written by {who}", 20, c).move_to(filecard.get_corner(m.UR) + [-0.3, -0.3, 0], aligned_edge=m.RIGHT)

        def write(who):
            d = docs[who]
            return m.Succession(d.animate(run_time=0.7).move_to(filecard.get_center()), m.FadeOut(d, run_time=0.25, scale=0.5))

        self.say(
            "[Back to the two bugs.] [Both Ottos work in the same folder], [and both bugs live in the cart file.]",
            m.AnimationGroup(self.new_header("05 · one folder", "Two writers, one file"), m.FadeIn(one, shift=0.2 * m.UP), m.FadeIn(two, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(m.VGroup(folder, fhead)), m.FadeIn(cwd, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(m.VGroup(filecard, fcap, dots)), m.FadeIn(A.current), m.FadeIn(B.current)),
        )
        self.say(
            "[Otto one reads the file], [and so does Otto two.]",
            m.AnimationGroup(m.FadeIn(docs[ONE], target_position=filecard), m.FadeIn(reads[0])),
            m.AnimationGroup(m.FadeIn(docs[TWO], target_position=filecard), m.FadeIn(reads[1])),
        )
        self.say(
            "[Otto one writes its fix.] [Then Otto two writes the whole file back], [with only its own fix in it.]",
            m.AnimationGroup(write(ONE), m.FadeOut(reads[0]), m.Succession(m.Wait(0.6), m.AnimationGroup(A.to(line(la1, {r".+": MINT}, ya)), written.to(wrote(ONE))))),
            m.AnimationGroup(write(TWO), m.FadeOut(reads[1]), m.Succession(m.Wait(0.6), m.AnimationGroup(A.to(line(la0, {r"- 1": ROSE}, ya)), B.to(line(lb1, {r".+": MINT}, yb)), written.to(wrote(TWO))))),
            m.Circumscribe(B.current, color=ORCHID, buff=0.06),
        )
        self.say(
            "[Otto one's fix is gone]: [the total test fails again], [and nobody can tell whose change was whose.]",
            m.Circumscribe(A.current, color=ROSE, buff=0.06),
            m.LaggedStart(m.FadeIn(term[:4]), *[m.FadeIn(r, shift=0.1 * m.UP) for r in term[4]], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(asks, shift=0.1 * m.UP), m.Circumscribe(written.current, color=ROSE, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 · worktrees

    def worktrees(self):
        self.next_section("6 · A folder per task", notes=f"Learn Claude Code s18: 'Tasks own the goal, worktrees own the directory, bound by ID'; git worktree add .worktrees/{{name}} -b wt/{{name}} HEAD (here named after the task, without HEAD, which is the default); claiming a task bound to a worktree sets the teammate's working directory (s18 binds it in the task file; the lead creates it); Claude Code uses .claude/worktrees/ and branches worktree-{{slug}}. Run for real in the scene (git 2.54 in a temporary folder): '{ADD_LINE['1']}'; each Otto edits its own cart.py and runs its own test: task 1 '{WT_RESULT['1']}', task 2 '{WT_RESULT['2']}'; the main checkout is untouched. Facts as of October 2026.")
        term = terminal([f"$ {add_cmd('1')}", ADD_LINE["1"]], width=7.0, title="tinyshop/")
        term.move_to([-6.6 + term.width / 2, 2.45 - term.height / 2, 0])
        cmd = term[4][0]
        quote = m.VGroup(*[title_text(s, 22, INK) for s in ("“Tasks own the goal,", "worktrees own the directory,", "bound by ID.”")])
        quote.arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        src = cite("learn-claude-code · s18")
        qgroup = m.VGroup(quote, src)
        src.next_to(quote, m.DOWN, buff=0.2).align_to(quote, m.LEFT)
        qgroup.move_to([6.6 - qgroup.width / 2, 2.45 - qgroup.height / 2, 0])

        lanes = []
        for k, (n, who) in enumerate((("1", ONE), ("2", TWO))):
            c, f = COLORS[who]
            y = -0.4 - 1.95 * k
            ln = NS()
            box = card(2.45, 0.85, SUN, CARD, radius=0.16, stroke=2.5).move_to([-5.35, y, 0])
            words = m.VGroup(code(n, 22, SUN, weight=m.BOLD), text(SUBJECT[n], 22, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.18)
            words.move_to(box)
            sit(words[0], n, box.get_y() - 0.09)
            sit(words[1], SUBJECT[n], box.get_y() - 0.09)
            ln.task = m.VGroup(box, words)
            ln.bot = badge(who, 0.5).move_to([-3.3, y, 0])
            wt = card(7.85, 1.75, c, CARD, radius=0.18, stroke=2.5).move_to([6.55 - 7.85 / 2, y, 0])
            left, top = wt.get_left()[0] + 0.3, wt.get_top()[1]
            path = code(f".worktrees/task-{n}", 22, INK, weight=m.BOLD)
            row1 = m.VGroup(icon(FOLDER[who], 0.32), path).arrange(m.RIGHT, buff=0.15).move_to([left, 0, 0], aligned_edge=m.LEFT)
            sit(path, f".worktrees/task-{n}", top - 0.48)
            row1[0].set_y(path.get_y())
            branch = code(f"wt/task-{n}", 20, c, weight=m.BOLD).move_to([wt.get_right()[0] - 0.3, 0, 0], aligned_edge=m.RIGHT)
            sit(branch, f"wt/task-{n}", top - 0.48)
            fname = sit(code("cart.py", 20, MUTED).move_to([left, 0, 0], aligned_edge=m.LEFT), "cart.py", top - 0.98)
            bug, good = (BUG_1, FIX_1) if n == "1" else (BUG_2, FIX_2)
            lx = left + 1.35

            def mk(s, rule, y_, x_=lx):
                return sit(code(s, 20, INK, t2c=rule).move_to([x_, 0, 0], aligned_edge=m.LEFT), s, y_)

            ln.line = Slot(self, mk(bug, {r"- 1|> 50": ROSE}, top - 0.98))
            ln.fixed = mk(good, {r".+": MINT}, top - 0.98)
            ln.cmd = sit(code(f"pytest -q -k {TASK_K[n]}", 20, MUTED).move_to([left, 0, 0], aligned_edge=m.LEFT), f"pytest -q -k {TASK_K[n]}", top - 1.45)
            ln.result = sit(code(WT_RESULT[n], 20, MINT).move_to([wt.get_right()[0] - 0.3, 0, 0], aligned_edge=m.RIGHT), WT_RESULT[n], top - 1.45)
            ln.wt, ln.row1, ln.branch, ln.fname = wt, row1, branch, fname
            ln.cwd = arrow(ln.bot[0].get_right() + [0.05, 0, 0], wt.get_left(), c, buff=0.1)
            ln.cwd_label = code("cwd", 20, c).next_to(ln.cwd, m.UP, buff=0.08)
            ln.n = n
            lanes.append(ln)
        l1, l2 = lanes

        self.say(
            "[The cure is to give every task its own folder], [with git worktree.]",
            m.AnimationGroup(self.new_header("06 · worktrees", "A folder per task"), *[m.FadeIn(ln.task, shift=0.2 * m.RIGHT) for ln in lanes]),
            m.AnimationGroup(m.FadeIn(term[:4]), m.AddTextLetterByLetter(cmd, time_per_char=0.02)),
        )
        self.say(
            "[One command checks out the same repo a second time], [in a new folder], [on a new branch.]",
            m.AnimationGroup(m.FadeIn(term[4][1], shift=0.1 * m.UP), m.FadeIn(l1.wt, target_position=term)),
            m.AnimationGroup(m.Circumscribe(glyphs(cmd, ".worktrees/task-1"), color=AZURE, buff=0.05), m.FadeIn(l1.row1, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(glyphs(cmd, "-b wt/task-1"), color=AZURE, buff=0.05), m.FadeIn(l1.branch, shift=0.1 * m.UP)),
        )
        self.say(
            "[In Learn Claude Code's words]: [tasks own the goal], [worktrees own the directory], [bound by ID.]",
            m.FadeIn(src),
            m.AnimationGroup(m.FadeIn(quote[0], shift=0.1 * m.UP), m.Circumscribe(l1.task, color=SUN, buff=0.06)),
            m.AnimationGroup(m.FadeIn(quote[1], shift=0.1 * m.UP), m.Circumscribe(l1.wt, color=AZURE, buff=0.06)),
            m.AnimationGroup(
                m.FadeIn(quote[2], shift=0.1 * m.UP), m.Circumscribe(l1.task[1][0], color=SUN, buff=0.08),
                m.Circumscribe(glyphs(l1.row1[1], "task-1"), color=SUN, buff=0.05),
            ),
        )
        self.say(
            "[When Otto one claims task one], [its working directory moves into that worktree.] [Otto two gets its own.]",
            m.AnimationGroup(m.FadeIn(l1.bot, shift=0.2 * m.UP), l1.task[0].animate.set_stroke(AZURE)),
            m.AnimationGroup(m.GrowArrow(l1.cwd), m.FadeIn(l1.cwd_label), m.FadeIn(m.VGroup(l1.fname, l1.line.current))),
            m.AnimationGroup(
                m.FadeIn(l2.bot, shift=0.2 * m.UP), l2.task[0].animate.set_stroke(ORCHID),
                m.FadeIn(m.VGroup(l2.wt, l2.row1, l2.branch, l2.fname, l2.line.current), shift=0.15 * m.UP),
                m.GrowArrow(l2.cwd), m.FadeIn(l2.cwd_label),
            ),
        )
        self.say(
            "[Now each Otto edits its own copy of the file], [and runs its own test]: [one passed.]",
            m.AnimationGroup(*[ln.line.to(ln.fixed) for ln in lanes]),
            m.AnimationGroup(*[m.FadeIn(ln.cmd, shift=0.1 * m.UP) for ln in lanes]),
            m.AnimationGroup(*[m.FadeIn(ln.result, shift=0.1 * m.UP) for ln in lanes]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 · names

    def names(self):
        self.next_section("7 · Names collide too", notes=f"Li, AI Agents in Depth, ch. 10, failure mode three (homogeneous convergence): 'In Anthropic's experiment, {SAME_NAME} of {COHORT} Agents that came online at the same time created Git branches with the same name' (no further detail in the book). The grid's layout (which {SAME_NAME} light up) is illustrative. Our fix: name each worktree after its task ID (wt/task-1, wt/task-2, wt/task-3: unique, asserted); since a claim gives each task one owner (section 3), only that owner creates its worktree. Facts as of October 2026.")
        grid = m.VGroup(*[otto(0.34, MUTED, CARD_HI) for _ in range(COHORT)])
        grid.arrange_in_grid(rows=3, cols=10, buff=(0.55, 0.42)).move_to([0, 1.2, 0])
        rose = [otto(0.34, ROSE, ROSE_FILL).move_to(grid[i]) for i in SAME]
        big = title_text(f"{SAME_NAME} of {COHORT}", 64, ROSE).move_to([0, -0.75, 0])
        desc = text("came online together and created git branches with the same name", 26, MUTED).next_to(big, m.DOWN, buff=0.3)
        src = cite("Li, AI Agents in Depth · ch. 10").next_to(desc, m.DOWN, buff=0.3)

        chains = m.VGroup()
        parts = []
        for k, (n, who) in enumerate((("1", ONE), ("2", TWO))):
            c, f = COLORS[who]
            y = 0.9 - 1.6 * k
            box = card(2.7, 0.8, SUN, CARD, radius=0.16, stroke=2.5).move_to([-4.6, y, 0])
            words = m.VGroup(code(f"task {n}", 22, SUN, weight=m.BOLD)).move_to(box)
            task = m.VGroup(box, words)
            who_b = m.VGroup(otto(0.42, c, f), code(who, 22, c, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2).move_to([-0.5, y, 0])
            name = chip(f"wt/task-{n}", c, f, width=2.7, height=0.8, size=26, mono=True).move_to([3.9, y, 0])
            a1 = arrow(box.get_right(), who_b.get_left(), MUTED, buff=0.2)
            a2 = arrow(who_b.get_right(), name.get_left(), MUTED, buff=0.2)
            parts.append(NS(task=task, who=who_b, name=name, a1=a1, a2=a2))
            chains.add(task, who_b, name, a1, a2)
        motto = pill("one task · one owner · one name", MINT, MINT_FILL, 24).move_to([0, -2.5, 0])

        self.say(
            "[Even names collide.] [In one Anthropic experiment], [eighteen of thirty agents that started together] [created git branches with the same name.]",
            m.AnimationGroup(self.new_header("07 · names", "Names collide too"), m.LaggedStart(*[m.FadeIn(o, scale=0.6) for o in grid], lag_ratio=0.03)),
            m.FadeIn(src),
            m.AnimationGroup(m.LaggedStart(*[m.Transform(grid[i], r) for i, r in zip(SAME, rose)], lag_ratio=0.06), m.FadeIn(big, scale=0.85)),
            m.FadeIn(desc, shift=0.1 * m.UP),
        )
        self.say(
            "[Name each worktree after its task], [and since each task has just one owner], [no two Ottos can pick the same name.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(grid, big, desc, src), shift=0.2 * m.UP),
                *[m.AnimationGroup(m.FadeIn(p.task, shift=0.2 * m.RIGHT), m.FadeIn(p.name, shift=0.2 * m.LEFT)) for p in parts],
            ),
            m.AnimationGroup(*[m.AnimationGroup(m.GrowArrow(p.a1), m.FadeIn(p.who, shift=0.15 * m.UP), m.GrowArrow(p.a2)) for p in parts]),
            m.FadeIn(motto, scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 8 · merge and clean up

    def merge(self):
        self.next_section("8 · Merge, then clean up", notes=f"Run for real (git 2.54): each Otto commits on its branch ('{SUBJECT['1']}', '{SUBJECT['2']}'); git merge wt/task-1 fast-forwards main; git merge wt/task-2 → '{MERGE_LINES[0]}', '{MERGE_LINES[1]}' (the fixes are on lines {LINE_1 + 1} and {LINE_2 + 1}, so no conflict); the merge commit's parents are '{GIT.parent_subjects[0]}' and '{GIT.parent_subjects[1]}'; pytest -q on the merged file: '{MERGED_SUMMARY}'. Removing a worktree with uncommitted changes (tried before otto-2 committed): '{REFUSE}' (git; s18's remove_worktree refuses too unless discard_changes=true; Claude Code's ExitWorktree likewise, per the site). Clean worktrees are removed without a word. Then tasks 1 and 2 are completed, task 3 can start, and otto-1's next idle poll claims it: '{CLAIMED_3}' (computed). Facts as of October 2026.")
        base_p, c1_p, c2_p, mg_p = [-5.0, 0.95, 0], [-1.5, 1.7, 0], [-1.5, 0.2, 0], [2.3, 0.95, 0]
        base = commit_dot(base_p, INK)
        c1, c2, mg = commit_dot(c1_p, AZURE), commit_dot(c2_p, ORCHID), commit_dot(mg_p, MINT)
        e_b1, e_b2 = edge(base_p, c1_p, AZURE), edge(base_p, c2_p, ORCHID)
        e_1m, e_2m = edge(c1_p, mg_p, AZURE), edge(c2_p, mg_p, ORCHID)
        stub = m.Line([mg_p[0] + 0.15, mg_p[1], 0], [5.0, mg_p[1], 0]).set_stroke(MINT, 4)
        main = pill("main", MINT, MINT_FILL, 22, mono=True)
        main.move_to([5.0 + 0.12 + main.width / 2, mg_p[1], 0])
        base_l = code(GIT.base_subject, 20, MUTED).next_to(base, m.DOWN, buff=0.18)
        c1_l = m.VGroup(code("wt/task-1", 20, AZURE, weight=m.BOLD), text(f"{SUBJECT['1']} · line {LINE_1 + 1}", 22, INK)).arrange(m.RIGHT, buff=0.25)
        c1_l.next_to(c1, m.UP, buff=0.2)
        c2_l = m.VGroup(code("wt/task-2", 20, ORCHID, weight=m.BOLD), text(f"{SUBJECT['2']} · line {LINE_2 + 1}", 22, INK)).arrange(m.RIGHT, buff=0.25)
        c2_l.next_to(c2, m.DOWN, buff=0.2)
        on_baseline(c1_l)
        on_baseline(c2_l)
        mg_l = code(GIT.head_subject, 20, MINT).move_to([mg_p[0] + 0.3, mg_p[1] + 0.45, 0], aligned_edge=m.LEFT)

        term = terminal([f"$ git merge wt/task-2", *MERGE_LINES, "$ pytest -q", MERGED_SUMMARY], width=5.6, title="tinyshop/")
        term.move_to([-6.6 + term.width / 2, -0.75 - term.height / 2, 0])
        rows = term[4]
        rows[4].set_color(MINT)
        refuse = terminal([f"$ git worktree remove .worktrees/task-2", *[f"! {s}" for s in REFUSE_LINES]], width=6.4, title="tinyshop/")
        refuse.move_to([6.6 - refuse.width / 2, -0.95 - refuse.height / 2, 0])
        caption = text("for example, before otto-2 committed", 20, MUTED).next_to(refuse, m.UP, buff=0.12).align_to(refuse, m.LEFT)
        src = cite("git · learn-claude-code s18").move_to([6.6, 2.45, 0], aligned_edge=m.RIGHT)
        graph = m.VGroup(base, c1, c2, mg, e_b1, e_b2, e_1m, e_2m, stub, main, base_l, c1_l, c2_l, mg_l)

        done = [Task("1", SUBJECT["1"], "completed", ONE), Task("2", SUBJECT["2"], "completed", TWO), Task("3", SUBJECT["3"], blockedBy=["1", "2"])]
        cards = board(self, done, [(-4.4, 0.85), (0, 0.85), (4.4, 0.85)], width=4.0)
        one = badge(ONE, 0.75).move_to([-4.4, -1.75, 0])
        n1 = Slot(self)
        free = pill("unblocked", MINT, MINT_FILL, 20).next_to(cards[2].box, m.DOWN, buff=0.25)

        self.say(
            "[Each Otto commits on its own branch], [and the branches merge back into main.]",
            m.AnimationGroup(
                self.new_header("08 · merge", "Merge, then clean up"), m.FadeIn(m.VGroup(base, base_l)),
                m.Create(e_b1), m.Create(e_b2), m.FadeIn(m.VGroup(c1, c2), scale=0.6), m.FadeIn(c1_l, shift=0.1 * m.DOWN), m.FadeIn(c2_l, shift=0.1 * m.UP),
            ),
            m.AnimationGroup(
                m.Create(e_1m), m.Create(e_2m), m.FadeIn(mg, scale=0.6), m.FadeIn(mg_l), m.Create(stub), m.FadeIn(main),
                m.FadeIn(term[:4]), m.FadeIn(rows[0], shift=0.1 * m.UP),
            ),
        )
        self.say(
            "[The two fixes touch different lines], [so git combines them without a conflict]: [all fourteen tests pass.]",
            m.AnimationGroup(m.Circumscribe(glyphs(c1_l[1], f"line {LINE_1 + 1}"), color=AZURE, buff=0.05), m.Circumscribe(glyphs(c2_l[1], f"line {LINE_2 + 1}"), color=ORCHID, buff=0.05)),
            m.AnimationGroup(m.FadeIn(rows[1], shift=0.1 * m.UP), m.FadeIn(rows[2], shift=0.1 * m.UP), m.Circumscribe(mg, color=MINT, buff=0.08)),
            m.AnimationGroup(m.FadeIn(rows[3], shift=0.1 * m.UP), m.FadeIn(rows[4], shift=0.1 * m.UP)),
        )
        self.say(
            "[Cleanup is careful too]: [a worktree with uncommitted changes] [won't be removed unless you force it.]",
            m.AnimationGroup(m.FadeIn(refuse[:4]), m.FadeIn(refuse[4][0], shift=0.1 * m.UP), m.FadeIn(caption), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in refuse[4][1:]], lag_ratio=0.3),
            m.Circumscribe(glyphs(refuse[4][3], "--force"), color=ROSE, buff=0.05),
        )
        self.say(
            "[And Otto one, idle again], [finds the release unblocked], [and claims it.]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(graph, term, refuse, caption, src), shift=0.2 * m.UP),
                *[c.intro() for c in cards], m.FadeIn(one, shift=0.2 * m.UP), n1.to(beside(note("idle · polling", MUTED), one)),
            ),
            m.AnimationGroup(m.FadeOut(cards[2].lock, scale=0.6), m.FadeIn(free, shift=0.1 * m.UP), m.Circumscribe(m.VGroup(cards[0].status.current, cards[1].status.current), color=MINT, buff=0.06)),
            m.AnimationGroup(cards[2].set_owner(ONE, source=one[0]), cards[2].set_status("in_progress"), n1.to(beside(note(CLAIMED_3, INK, AZURE), one))),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Poll (idle teammates check the inbox, then the board, every 5 s; shut down after 60 s), claim (pending, unowned, unblocked; read-check-write under a lock; identity re-injected after compaction), isolate (a git worktree per task, named by its ID; merge back; removal refuses uncommitted work).")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Poll", AZURE), ("Claim", SUN), ("Isolate", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in (f"every {IDLE_POLL_INTERVAL} s: inbox, then board", "one owner, under a lock", "a worktree per task")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.9
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
        tagline = text("Teammates find their own work, each in a folder of its own.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [poll], [claim], [isolate.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say(
            "[Teammates find their own work], [and each one works in a folder of its own.]",
            m.Write(tagline),
            m.Circumscribe(words[2], color=MINT, buff=0.12),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
