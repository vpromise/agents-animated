"""Agents, Animated · Episode 24 · Agent Teams (Module 5 · Scaling Up)

How several persistent agents work together. Running example: a lead Otto with two teammates, a tester
and a reviewer, adding a discount code to tinyshop's checkout. They talk through mailbox files
(.mailboxes/{name}.jsonl): the tester's three tests pass, the reviewer finds that a ten-dollar code on a
five-dollar cart gives minus five, the reviewer is shut down gracefully, and the tester's risky plan is
rejected and a smaller one approved. The run is not drawn by hand: the scene runs send / read_inbox (the
same lines it shows), the request tracker and match_response against an in-memory disk, and every
number on screen comes from that run (asserted below).

Sections: hook (a subagent is a temp; teammates persist) · temps vs teammates · mailboxes (a message
{from, to, content, type, ts} appended as one JSON line; reading drains the file; the inbox arrives as
<inbox>…</inbox>; the racy read) · a team at work (both replies wait in the lead's file, read together)
· protocols (shutdown and plan approval; request/response tied by request_id; ProtocolState) · graceful
shutdown as a state machine · plan approval (rejected, re-planned, a stray shutdown reply ignored,
approved) · Claude Code (locked JSON inboxes, polled every second, ~15 message types, no nested
teammates) · a caution (share memory by communicating) · recap: persist, message, correlate.

Sources (facts dated October 2026):
- Learn Claude Code, s15 "Agent Teams" (learn.shareai.run/en/s15/): subagents are temps, teammates
  persist; MessageBus.send appends {"from","to","content","type","ts"} as one JSON line to
  .mailboxes/{to}.jsonl; read_inbox parses every line then `inbox.unlink()  # consume: read + delete`;
  the page admits the read-then-unlink race can lose messages; a teammate's new mail is appended as
  <inbox>…</inbox> user content; its system prompt "You are '{name}', a {role}. Use tools to complete
  tasks."; results go to the lead's mailbox. Dive into CC source (the site's reading of Claude Code,
  not verified by us): inboxes ~/.claude/teams/{team}/inboxes/{name}.json, written under
  proper-lockfile; the lead's inbox poller checks every 1 s; "15 message types" (its table lists the
  names shown here, sandbox_permission_* standing for a request/reply pair); teammates cannot spawn
  teammates.
- Learn Claude Code, s16 "Team Protocols" (learn.shareai.run/en/s16/): "Killing the thread outright
  leaves half-written files on disk"; request/response pairs correlated by request_id (req_ + six
  digits); ProtocolState {request_id, type: shutdown | plan_approval, sender, target, status: pending |
  approved | rejected} (also payload, created_at, omitted here) in pending_requests; match_response
  skips unknown ids, settled requests and mismatched types; "A shutdown_response cannot accidentally
  approve a plan_approval request."; "The key idea is correlation, not ceremony."; the teammate
  finishes before confirming a shutdown; a rejected plan reaches the teammate as "[Plan rejected]".
- Li Bojie, AI Agents in Depth, ch. 10: "Go has a widely quoted maxim: 'Do not communicate by sharing
  memory; instead, share memory by communicating.'"; failure mode one, "Two Agents modify the same
  file simultaneously, and the later write overwrites the earlier one."

Simplified (said so in the notes): send() on screen omits the **extra keys a protocol message carries;
teammate_loop() is an excerpt (s15 caps a teammate at 10 rounds, s16 adds an idle loop);
ProtocolState omits payload and created_at.
Illustrative (for teaching): the discount feature, apply_code and its tests, the messages' wording,
the timestamps, the tester's plans, the stray shutdown reply, the review notes.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import json
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the team's run, computed

DISK = {}  # path → text: an in-memory disk, so rendering never writes a .mailboxes folder


class MemPath:
    """Just enough of pathlib.Path for the mailbox code to run unchanged."""

    def __init__(self, path):
        self.path = path

    def __truediv__(self, name):
        return MemPath(f"{self.path}/{name}")

    def exists(self):
        return self.path in DISK

    def read_text(self):
        return DISK[self.path]

    def unlink(self):
        del DISK[self.path]

    def open(self, mode):
        assert mode == "a"
        return _Append(self.path)


class _Append:
    def __init__(self, path):
        self.path = path

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def write(self, s):
        DISK[self.path] = DISK.get(self.path, "") + s


class _Clock:
    """time.time() for the run: a message every twelve seconds from 9:00 UTC, 22 October 2026."""

    start = datetime(2026, 10, 22, 9, 0, tzinfo=timezone.utc).timestamp()

    def __init__(self):
        self.n = 0

    def time(self):
        self.n += 1
        return self.start + 12.0 * self.n


time = _Clock()
MAILBOX_DIR = MemPath(".mailboxes")
LEAD, TESTER, REVIEWER = "lead", "tester", "reviewer"
ROLE_OF = {TESTER: "test writer", REVIEWER: "code reviewer"}


def send(sender, to, content, type="message", **extra):
    msg = {"from": sender, "to": to,
           "content": content, "type": type,
           "ts": time.time(), **extra}
    inbox = MAILBOX_DIR / f"{to}.jsonl"
    with inbox.open("a") as f:
        f.write(json.dumps(msg) + "\n")
    return msg


def read_inbox(name):
    inbox = MAILBOX_DIR / f"{name}.jsonl"
    if not inbox.exists():
        return []
    lines = inbox.read_text().splitlines()
    inbox.unlink()  # consume: read + delete
    return [json.loads(s) for s in lines]


def as_inbox(mail):
    """s15: a teammate's new mail joins its messages as one user message."""
    return {"role": "user", "content": f"<inbox>{json.dumps(mail)}</inbox>"}


def lines_in(name):
    inbox = MAILBOX_DIR / f"{name}.jsonl"
    return len(inbox.read_text().splitlines()) if inbox.exists() else 0


SEND_SRC = '''def send(sender, to, content, type):
    msg = {"from": sender, "to": to,
           "content": content, "type": type,
           "ts": time.time()}
    inbox = MAILBOX_DIR / f"{to}.jsonl"
    with inbox.open("a") as f:
        f.write(json.dumps(msg) + "\\n")'''
READ_SRC = '''def read_inbox(name):
    inbox = MAILBOX_DIR / f"{name}.jsonl"
    if not inbox.exists():
        return []
    lines = inbox.read_text().splitlines()
    inbox.unlink()  # consume: read + delete
    return [json.loads(s) for s in lines]'''
LOOP_SRC = '''def teammate_loop(name, messages):
    while True:
        mail = read_inbox(name)
        if mail:
            messages.append(as_inbox(mail))
        response = client.messages.create(
            messages=messages, tools=TOOLS)
        ...  # the rest of agent_loop'''
STATE_SRC = '''@dataclass
class ProtocolState:
    request_id: str  # "req_000001"
    type: str        # "shutdown" | "plan_approval"
    sender: str      # "lead"
    target: str      # "reviewer"
    status: str = "pending"  # → approved | rejected'''
GUARD_SRC = '''if msg["type"] != f"{req.type}_response":
    return  # wrong kind of reply: ignored'''

# the code on screen is the code that ran (send: minus the **extra keys protocol messages add)
_send_src, _read_src = inspect.getsource(send), inspect.getsource(read_inbox)
assert READ_SRC + "\n" == _read_src
assert all(line.strip().replace("type):", "type=\"message\", **extra):").replace("time.time()}", "time.time(), **extra}") in _send_src
           for line in SEND_SRC.replace("\\n", "\n").split("\n") if line.strip() and "json.dumps" not in line)
assert 'f.write(json.dumps(msg) + "\\n")' in _send_src
assert max(len(s) for s in (SEND_SRC + "\n" + READ_SRC + "\n" + LOOP_SRC).split("\n")) <= 44  # fits the right half


# ---- the protocol layer (s16): a tracker of requests, matched by request_id
@dataclass
class ProtocolState:
    request_id: str
    type: str
    sender: str
    target: str
    status: str = "pending"


pending_requests = {}
_next_id = iter(range(1, 1_000_000))


def new_request_id():
    return f"req_{next(_next_id):06d}"


def request(sender, target, type, content):
    rid = new_request_id()
    pending_requests[rid] = ProtocolState(rid, type, sender, target)
    return send(sender, target, content, f"{type}_request", request_id=rid)


def respond(req_msg, approve, content=""):
    kind = req_msg["type"].replace("_request", "_response")
    return send(req_msg["to"], req_msg["from"], content, kind, request_id=req_msg["request_id"], approve=approve)


def match_response(msg):
    req = pending_requests.get(msg["request_id"])
    if not req or req.status != "pending":
        return False  # unknown, or already settled
    if msg["type"] != f"{req.type}_response":
        return False  # wrong kind of reply: ignored
    req.status = "approved" if msg["approve"] else "rejected"
    return True


assert all(line.strip().split("  #")[0] in inspect.getsource(match_response) for line in GUARD_SRC.split("\n"))


def consume(name):
    """An agent's round: drain its inbox; responses update the tracker (s16's consume_lead_inbox)."""
    mail = read_inbox(name)
    matched = [match_response(msg) for msg in mail if msg["type"].endswith("_response")]
    return mail, matched


# ---- tinyshop's new feature (illustrative): a discount code at checkout
CODES = {"SAVE10": 10}


def apply_code(total, code):
    return total - CODES.get(code, 0)


def apply_code_fixed(total, code):
    return max(0, total - CODES.get(code, 0))


TESTS = [  # the tester's three tests
    lambda f: f(30, "SAVE10") == 20,
    lambda f: f(30, "NOPE") == 30,
    lambda f: f(10, "SAVE10") == 0,
]
NEW_TEST = lambda f: f(5, "SAVE10") == 0  # noqa: E731  the approved plan: totals never go negative


def pytest_q(f, tests):
    passed = sum(t(f) for t in tests)
    failed = len(tests) - passed
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


# ---- the run
JOB_T, JOB_R = "Write tests for apply_code.", "Review checkout.py."
M1 = send(LEAD, TESTER, JOB_T)
assert lines_in(TESTER) == 1 and list(M1) == ["from", "to", "content", "type", "ts"]  # five fields
TESTER_MAIL = read_inbox(TESTER)
assert TESTER_MAIL == [M1] and not (MAILBOX_DIR / "tester.jsonl").exists()  # drained: the file is gone
INBOX_MSG = as_inbox(TESTER_MAIL)
assert INBOX_MSG["role"] == "user" and INBOX_MSG["content"].startswith('<inbox>[{"from": "lead"')
SYSTEM_T = f"You are '{TESTER}', a {ROLE_OF[TESTER]}.\nUse tools to complete tasks."

M2 = send(LEAD, REVIEWER, JOB_R)
RESULT_T = pytest_q(apply_code, TESTS)
assert RESULT_T == "3 passed" and len(TESTS) == 3  # "three tests ... three passed"
M3 = send(TESTER, LEAD, f"{len(TESTS)} tests added: {RESULT_T}.", "result")
assert read_inbox(REVIEWER) == [M2]
CART, CODE = 5, "SAVE10"
BAD = apply_code(CART, CODE)
assert (CART, CODES[CODE], BAD) == (5, 10, -5)  # "a ten dollar code on a five dollar cart gives minus five"
M4 = send(REVIEWER, LEAD, f'apply_code({CART}, "{CODE}") returns {BAD}.', "result")
assert lines_in(LEAD) == 2  # "both replies wait in Otto's file"
LEAD_MAIL = read_inbox(LEAD)
assert LEAD_MAIL == [M3, M4]  # "it reads them together"
LOG = [M1, M2, M3, M4]

# graceful shutdown of the reviewer
S_REQ = request(LEAD, REVIEWER, "shutdown", "Please shut down.")
RID_S = S_REQ["request_id"]
assert RID_S == "req_000001" and pending_requests[RID_S].status == "pending"
assert read_inbox(REVIEWER) == [S_REQ]
NOTES = ["## Review of checkout.py", "- apply_code can go below zero", "- tests: 3 passed"]  # finished first
S_RESP = respond(S_REQ, True)
assert S_RESP["request_id"] == RID_S and S_RESP["type"] == "shutdown_response"
_, _matched = consume(LEAD)
assert _matched == [True] and pending_requests[RID_S].status == "approved"

# plan approval: the tester's risky plan is rejected, a smaller one approved
apply_code = apply_code_fixed  # Otto fixes the bug: totals stop at zero
M5 = send(LEAD, TESTER, "Add a test: totals never go negative.")
assert read_inbox(TESTER) == [M5]
PLAN_1, PLAN_2 = "Rewrite the shared fixtures in conftest.py.", "Add one test to tests/test_discount.py."
P1 = request(TESTER, LEAD, "plan_approval", PLAN_1)
RID_P1 = P1["request_id"]
assert read_inbox(LEAD) == [P1]
NO = "Leave the fixtures; add one test."
R1 = respond(P1, False, NO)
_got, _matched = consume(TESTER)
assert _matched == [True] and pending_requests[RID_P1].status == "rejected"
REJECTED_NOTE = "[Plan rejected]"  # what s16 appends to the teammate's messages
P2 = request(TESTER, LEAD, "plan_approval", PLAN_2)
RID_P2 = P2["request_id"]
assert (RID_P1, RID_P2) == ("req_000002", "req_000003")
STRAY = send(TESTER, LEAD, "", "shutdown_response", request_id=RID_P2, approve=True)  # for example
_got, _matched = consume(LEAD)
assert _got == [P2, STRAY] and _matched == [False] and pending_requests[RID_P2].status == "pending"  # ignored
R2 = respond(P2, True)
_got, _matched = consume(TESTER)
assert _matched == [True] and pending_requests[RID_P2].status == "approved"
RESULT_2 = pytest_q(apply_code, TESTS + [NEW_TEST])
assert RESULT_2 == "4 passed" and NEW_TEST(apply_code_fixed) and not NEW_TEST(lambda t, c: t - CODES.get(c, 0))
TRACK = [pending_requests[r] for r in (RID_S, RID_P1, RID_P2)]
assert [(s.type, s.sender, s.target, s.status) for s in TRACK] == [
    ("shutdown", LEAD, REVIEWER, "approved"),
    ("plan_approval", TESTER, LEAD, "rejected"),
    ("plan_approval", TESTER, LEAD, "approved"),
]

# the racy read (s15 admits it), run on a scratch disk: a line written between read and delete is lost
_saved = dict(DISK)
DISK.clear()
send(LEAD, TESTER, "Also cover the empty cart.")
_inbox = MAILBOX_DIR / "tester.jsonl"
_lines = _inbox.read_text().splitlines()           # 1 read
send(REVIEWER, TESTER, "Use the shared cart fixture.")  # 2 another agent appends
_inbox.unlink()                                     # 3 delete
RACE_READ, RACE_LOST = len(_lines), 2 - len(_lines)
assert (RACE_READ, RACE_LOST) == (1, 1) and not _inbox.exists()
DISK.clear()
DISK.update(_saved)

# Claude Code, in Learn Claude Code's reading of its source (s15): message types
CC_TYPES = [
    "plain text", "idle_notification", "permission_request", "permission_response",
    "plan_approval_request", "plan_approval_response", "shutdown_request", "shutdown_approved",
    "shutdown_rejected", "task_assignment", "team_permission_update", "mode_set_request",
    "teammate_terminated", "sandbox_permission_*",
]
CC_COUNT = 15  # the page's count; sandbox_permission_* names a request/reply pair
assert len(CC_TYPES) + 1 == CC_COUNT

# ---------------------------------------------------------------------------- look

COLORS = {LEAD: (CORAL, CORAL_FILL), TESTER: (AZURE, AZURE_FILL), REVIEWER: (ORCHID, ORCHID_FILL)}
MAIL = {name: recolored("mail", c, f, prefix="ep24") for name, (c, f) in COLORS.items()}
STATUS = {"pending": (SUN, SUN_FILL), "approved": (MINT, MINT_FILL), "rejected": (ROSE, ROSE_FILL)}
WELL = "#18150F"  # the dark inside of panels and files
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")

# lanes (sections 2–3): an agent, its mailbox file; the right half holds code and logs
LANE_Y = {LEAD: 1.55, TESTER: -0.05, REVIEWER: -1.65}
BADGE_X = -5.95
BOX_X, BOX_W = -3.25, 4.1
RIGHT_X, RIGHT_W = 2.95, 7.3  # the right half: x from -0.7 to 6.6


def cite(s):
    return text(s, 20, MUTED)


def baseline(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def badge(name, size=0.42, side=m.DOWN, label_size=20):
    c, f = COLORS[name]
    bot = otto(size, c, f)
    label = code(name, label_size, c, weight=m.BOLD)
    if side is m.DOWN:
        label.next_to(bot, m.DOWN, buff=0.08)
    else:
        label.next_to(bot[0], m.RIGHT, buff=0.18)
    return m.VGroup(bot, label)


def line_part(body, src, i, key=None):
    """The glyphs of `key` (default: the whole line) on line i of a Text made from `src`."""
    lines = src.split("\n")
    k = sum(sum(not c.isspace() for c in ln) for ln in lines[:i])
    line = lines[i]
    key = key or line.strip()
    start = line.index(key)
    skip = sum(c.isspace() for c in line[:start])
    return body[k + start - skip:k + start - skip + sum(not c.isspace() for c in key)]


def align_baselines(texts, strings):
    """Put Texts in a row on one baseline (descenders ignored)."""
    base = baseline(texts[0], strings[0].replace(" ", ""))
    for t, s in zip(texts[1:], strings[1:]):
        t.shift((base - baseline(t, s.replace(" ", ""))) * m.UP)
    return texts


def envelope(sender, height=0.26):
    return icon(MAIL[sender], height)


class Mailbox:
    """A mailbox file: its name, and one envelope per JSON line, left to right."""

    def __init__(self, name, center, width, height=0.66, gap=0.44):
        self.name = name
        self.box = card(width, height, EDGE, WELL, radius=0.14, stroke=2).move_to(center)
        self.label = code(f"{name}.jsonl", 20, INK)
        self.label.move_to(self.box).align_to(self.box, m.LEFT).shift(0.22 * m.RIGHT)
        widest = code("reviewer.jsonl", 20).width
        self.x0 = self.box.get_left()[0] + 0.22 + widest + 0.42
        self.gap = gap
        self.group = m.VGroup(self.box, self.label)
        self.items = []

    def slot(self, i):
        return np.array([self.x0 + self.gap * i, self.box.get_y(), 0])

    def deliver(self, sender, source, run_time=0.9):
        env = envelope(sender).move_to(self.slot(len(self.items)))
        self.items.append(env)
        return env, m.FadeIn(env, target_position=source, scale=0.6, run_time=run_time)

    def drain(self, target, run_time=0.8):
        gone, self.items = self.items, []
        return m.LaggedStart(*[m.FadeOut(e, target_position=target, scale=0.4) for e in gone], lag_ratio=0.25, run_time=run_time)


def log_row(msg, width=RIGHT_W):
    """One sent message: a stripe in the sender's colour, sender → receiver · type, then the content."""
    c = COLORS[msg["from"]][0]
    box = card(width, 0.84, EDGE, CARD, radius=0.12, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=0.84 - 0.18, stroke_width=0, fill_color=c, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.1 * m.RIGHT)
    who = code(f'{msg["from"]} → {msg["to"]}', 20, c, weight=m.BOLD)
    kind = code(msg["type"], 20, MINT if msg["type"] == "result" else SUN)
    top = m.VGroup(who, kind).arrange(m.RIGHT, buff=0.3)
    align_baselines([who, kind], [f'{msg["from"]}→{msg["to"]}', msg["type"]])
    body = code(msg["content"], 20, INK, t2c={r"-\d+": ROSE})
    inner = m.VGroup(top, body).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.32 * m.RIGHT)
    return m.VGroup(box, stripe, inner)


def wire(obj, accent, size=19, dashed=False):
    """A protocol message as it travels: compact JSON in a card of the given colour."""
    card_ = json_card(obj, size, accent, CARD)
    if dashed:
        card_[0].set_stroke(opacity=0)
        outline = m.DashedVMobject(card_[0].copy().set_stroke(accent, 2.5, opacity=1), num_dashes=40)
        card_.add(outline)
    return card_


def notes_card(lines, width=3.3, title="REVIEW.md", accent=ORCHID):
    """A small text file: its name and its lines (code, 20)."""
    rows = m.VGroup(*[code(s, 20, INK if i == 0 else MUTED) for i, s in enumerate(lines)])
    rows.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
    cap = code(title, 20, accent, weight=m.BOLD)
    w = max(width, rows.width + 0.6)
    box = card(w, rows.height + cap.height + 0.75, EDGE, WELL, radius=0.14, stroke=2)
    cap.align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.18 * m.DOWN)
    rows.next_to(cap, m.DOWN, buff=0.2, aligned_edge=m.LEFT)
    return m.VGroup(box, cap, rows)


def state_chip(label, width=2.0):
    c, f = STATUS[label]
    return chip(label, c, f, width=width, height=0.66, size=22, mono=True)


class Slot:
    """One place on screen whose content is swapped (from Episode 2)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.15):
        on_screen = {id(x) for x in self.scene.get_mobject_family_members()}
        anims = []
        if self.current is not None:
            out = m.FadeOut(self.current, shift=shift * m.UP)
            if id(self.current) not in on_screen:  # introduced earlier in this same line
                out.introducer = True
            anims.append(out)
        if new is not None:
            anims.append(m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


def loop_glyph(color, radius=0.26):
    arc = m.Arc(radius=radius, start_angle=1.9, angle=-5.0).set_stroke(color, 4)
    arc.add_tip(tip_length=0.13, tip_width=0.13)
    return arc


# ---------------------------------------------------------------------------- the episode


class Ep24AgentTeams(Episode):
    number = 24

    def lesson(self):
        self.hook()
        self.temps()
        self.mailboxes()
        self.team_at_work()
        self.protocols()
        self.shutdown()
        self.plan_approval()
        self.claude_code()
        self.caution()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Callback to Episode 21: a subagent gets a fresh context, returns one report and is discarded — a temp. Some jobs want colleagues who stay. Learn Claude Code s15: 'subagents are temps; teammates persist.' The tester/reviewer team and its jobs are this episode's running example (illustrative).")
        boss = otto(1.0).move_to([-5.0, 0.95, 0])
        boss_l = title_text("Otto", 26, INK).next_to(boss, m.DOWN, buff=0.2)
        temp = otto(0.75, MUTED, CARD_HI).move_to([-1.7, 0.95, 0])
        temp_l = text("subagent", 22, MUTED).next_to(temp, m.DOWN, buff=0.2)
        ep21 = code("episode 21", 20, FAINT).next_to(temp_l, m.DOWN, buff=0.12)
        fresh = card(1.5, 1.05, EDGE, WELL, radius=0.14, stroke=2).next_to(temp, m.RIGHT, buff=0.35)
        fresh_l = code("fresh", 20, MUTED).move_to(fresh)
        report = chip("report", MINT, MINT_FILL, width=1.5, height=0.6, size=22, mono=True)
        report.move_to(boss).shift(1.6 * m.RIGHT + 0.15 * m.DOWN)
        ghost = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.2, width=1.25, height=0.98).set_stroke(FAINT, 2.5), num_dashes=28)
        ghost.move_to(temp[0])
        temp_tag = pill("temp", MUTED, CARD_HI, 22).next_to(ghost, m.RIGHT, buff=0.35)

        mates = m.VGroup(badge(TESTER, 0.75, label_size=24), badge(REVIEWER, 0.75, label_size=24))
        mates[0].move_to([1.75, 0.8, 0])
        mates[1].move_to([4.95, 0.8, 0])
        jobs = m.VGroup(
            chip("pytest -q", AZURE, AZURE_FILL, width=2.1, height=0.6, size=22, mono=True),
            chip("git diff", ORCHID, ORCHID_FILL, width=2.1, height=0.6, size=22, mono=True),
        )
        for j, b in zip(jobs, mates):
            j.next_to(b, m.DOWN, buff=0.3)
        stay = m.VGroup(*[loop_glyph(COLORS[n][0], 0.22).next_to(b[0], m.RIGHT, buff=0.2) for n, b in zip((TESTER, REVIEWER), mates)])

        quote = m.VGroup(title_text("Subagents are temps;", 40, MUTED), title_text("teammates persist.", 40, AZURE)).arrange(m.RIGHT, buff=0.3)
        on_baseline(quote)
        quote.move_to([0, -2.2, 0])
        src = cite("learn-claude-code · s15").next_to(quote, m.DOWN, buff=0.25)

        self.say(
            "[In episode twenty-one, Otto hired a subagent]: [a fresh context], [one report back], [and then it was gone.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(boss, boss_l), shift=0.2 * m.UP), m.AnimationGroup(*[m.FadeIn(x, shift=0.2 * m.UP) for x in (temp, temp_l, ep21)]), lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(fresh, scale=0.8), m.FadeIn(fresh_l)),
            m.FadeIn(report, target_position=temp, scale=0.6),
            m.AnimationGroup(m.FadeOut(m.VGroup(temp, fresh, fresh_l), scale=0.8), m.FadeIn(ghost), m.FadeOut(report, target_position=boss, scale=0.4)),
        )
        self.say(
            "[That's a temp.] [But some jobs need colleagues who stay]: [a tester who keeps running tests], [and a reviewer who reads each change.]",
            m.FadeIn(temp_tag, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(b[0], shift=0.2 * m.UP) for b in mates], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(mates[0][1]), m.FadeIn(jobs[0], shift=0.1 * m.UP), m.Create(stay[0])),
            m.AnimationGroup(m.FadeIn(mates[1][1]), m.FadeIn(jobs[1], shift=0.1 * m.UP), m.Create(stay[1])),
        )
        self.say(
            "[Learn Claude Code puts it in one line]: [subagents are temps], [teammates persist.]",
            m.FadeIn(src, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(quote[0], shift=0.15 * m.UP), m.Circumscribe(m.VGroup(ghost, temp_tag), color=MUTED, buff=0.1)),
            m.AnimationGroup(m.FadeIn(quote[1], shift=0.15 * m.UP), *[m.Rotate(s, -2 * np.pi) for s in stay]),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ temps vs teammates

    def temps(self):
        self.next_section("1 · Temps and teammates", notes="Learn Claude Code s15's comparison table: a subagent (s06) is one-shot and destroyed after use, returns only its conclusion, fully isolated context; a teammate is multi-turn (teaching: 10 rounds; Claude Code: an idle loop), talks through an async inbox at any time, and keeps its own context. A team: one lead plus several teammates, each running the Episode 2 loop over its own messages list.")
        rows = [("lifetime", "one task", "many tasks"), ("context", "fresh,", "its own, kept"), ("talks", "one summary, at the end", "messages, any time")]
        cards = []
        for side, (who, color, fill, edge) in enumerate((("subagent", MUTED, CARD_HI, EDGE), ("teammate", AZURE, AZURE_FILL, AZURE))):
            box = card(6.0, 3.9, edge, CARD, radius=0.22, stroke=2.5).move_to([(-1 if side == 0 else 1) * 3.35, -0.05, 0])
            head = m.VGroup(otto(0.42, color, fill), title_text(who, 30, color)).arrange(m.RIGHT, buff=0.25)
            head.move_to(box.get_corner(m.UL) + [0.4, -0.6, 0], aligned_edge=m.LEFT)
            items = []
            for i, (label, a, b) in enumerate(rows):
                y = box.get_top()[1] - 1.45 - 0.85 * i
                lab = code(label, 20, MUTED).move_to([box.get_left()[0] + 0.4, y, 0], aligned_edge=m.LEFT)
                val = text(a if side == 0 else b, 24, INK).move_to([box.get_left()[0] + 1.95, y, 0], aligned_edge=m.LEFT)
                align_baselines([lab, val], [label, a if side == 0 else b])
                items.append(m.VGroup(lab, val))
            cards.append(NS(box=box, head=head, rows=items))
        left, right = cards
        thrown = text("then thrown away", 24, ROSE).next_to(left.rows[1][1], m.RIGHT, buff=0.15)
        align_baselines([left.rows[1][1], thrown], ["fresh,", "then thrown away"])
        src = cite("learn-claude-code · s15").next_to(right.box, m.DOWN, buff=0.22).align_to(right.box, m.RIGHT)

        self.say(
            "[A subagent lives for one task.] [It starts with a fresh context], [hands back one summary], [and then its context is thrown away.]",
            m.AnimationGroup(self.new_header("01 · temps vs teammates", "Temps and teammates"), m.FadeIn(left.box), m.FadeIn(left.head, shift=0.1 * m.UP), m.FadeIn(left.rows[0], shift=0.1 * m.UP)),
            m.FadeIn(left.rows[1], shift=0.1 * m.UP),
            m.FadeIn(left.rows[2], shift=0.1 * m.UP),
            m.FadeIn(thrown, shift=0.1 * m.UP),
        )
        self.say(
            "[A teammate lives on.] [It keeps its own context from task to task], [and it can send or receive messages at any time.]",
            m.AnimationGroup(m.FadeIn(right.box), m.FadeIn(right.head, shift=0.1 * m.UP), m.FadeIn(right.rows[0], shift=0.1 * m.UP), m.FadeIn(src)),
            m.FadeIn(right.rows[1], shift=0.1 * m.UP),
            m.FadeIn(right.rows[2], shift=0.1 * m.UP),
        )

        names = (LEAD, TESTER, REVIEWER)
        xs = (-4.4, 0.0, 4.4)
        bots, loops, panels = m.VGroup(), m.VGroup(), m.VGroup()
        systems = {LEAD: "You are Otto,\nthe lead.", TESTER: f"You are '{TESTER}',\na {ROLE_OF[TESTER]}.", REVIEWER: f"You are '{REVIEWER}',\na {ROLE_OF[REVIEWER]}."}
        for name, x in zip(names, xs):
            b = badge(name, 0.6, label_size=24).move_to([x, 1.1, 0])
            bots.add(b)
            loops.add(loop_glyph(COLORS[name][0], 0.3).next_to(b[0], m.RIGHT, buff=0.3))
            panel = card(3.6, 1.62, EDGE, WELL, radius=0.18, stroke=2).move_to([x, -1.15, 0])
            title = code("messages", 20, MUTED).next_to(panel, m.UP, buff=0.1).align_to(panel, m.LEFT)
            row = message("system", systems[name], width=3.2, size=20).move_to(panel)
            panels.add(m.VGroup(panel, title, row))
        self.say(
            "[So a team is a lead and its teammates], [each running the loop from episode two], [each with its own messages list.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(left.box, left.head, *left.rows, thrown, right.box, right.head, *right.rows, src), shift=0.2 * m.DOWN), m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.UP) for b in bots], lag_ratio=0.25)),
            m.AnimationGroup(*[m.Succession(m.Create(lp, run_time=0.5), m.Rotate(lp, -2 * np.pi, run_time=1.0)) for lp in loops]),
            m.LaggedStart(*[m.FadeIn(p, shift=0.2 * m.UP) for p in panels], lag_ratio=0.25),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ mailboxes

    def build_lanes(self):
        st = NS()
        st.badges = {n: badge(n).move_to([BADGE_X, LANE_Y[n], 0]) for n in (LEAD, TESTER, REVIEWER)}
        st.boxes = {n: Mailbox(n, [BOX_X, LANE_Y[n], 0], BOX_W, gap=0.62) for n in (LEAD, TESTER, REVIEWER)}
        st.dir = code(".mailboxes/", 20, MUTED).next_to(st.boxes[LEAD].box, m.UP, buff=0.14).align_to(st.boxes[LEAD].box, m.LEFT)
        st.lanes = m.VGroup(*st.badges.values(), *[b.group for b in st.boxes.values()], st.dir)
        return st

    def at(self, st, name):
        return st.badges[name][0].get_center()

    def mailboxes(self):
        self.next_section("2 · Mailboxes", notes=f"Learn Claude Code s15: MessageBus.send appends one JSON line {{from, to, content, type, ts}} to .mailboxes/{{to}}.jsonl; read_inbox parses every line, then unlinks the file ('consume: read + delete'); a teammate's new mail joins its messages as <inbox>…</inbox> user content. Both functions on screen ran in this scene (an in-memory disk): after the send tester.jsonl held {1} line; reading returned it and the file was gone. The race (s15 admits it): read_text, then another agent appends, then unlink — {RACE_LOST} message lost (computed on a scratch disk). Claude Code locks its inbox files instead (proper-lockfile). Message wording and timestamps illustrative.")
        st = self.st = self.build_lanes()
        tb = st.boxes[TESTER]
        msg_card = json_card(M1, 19, CORAL, CARD, title="a message").move_to([RIGHT_X, 0.45, 0])
        body = msg_card[1]
        send_code = code_block(SEND_SRC, size=19, width=RIGHT_W, title="send()", accent=EDGE).move_to([RIGHT_X, 0.45, 0])
        read_code = code_block(READ_SRC, size=19, width=RIGHT_W, title="read_inbox()", accent=EDGE).move_to([RIGHT_X, 0.45, 0])
        right = Slot(self)

        self.say(
            "[How do they talk?] [Through files, like the notes in episode fourteen.] [Each agent has a mailbox]: [a JSON lines file, named after it.]",
            m.AnimationGroup(self.new_header("02 · mailboxes", "A mailbox is a file"), m.LaggedStart(*[m.FadeIn(b, shift=0.2 * m.UP) for b in st.badges.values()], lag_ratio=0.2)),
            m.FadeIn(st.dir, shift=0.1 * m.UP),
            m.LaggedStart(*[m.FadeIn(b.box) for b in st.boxes.values()], lag_ratio=0.2),
            m.LaggedStart(*[m.Write(b.label) for b in st.boxes.values()], lag_ratio=0.2),
        )
        keys = ['"from"', '"to"', '"content"', '"type"', '"ts"']
        self.say(
            "[A message is one small object]: [who it's from], [who it's for], [the content], [a type], [and a timestamp.]",
            right.to(msg_card),
            *[m.Circumscribe(glyphs(body, k), color=AZURE, buff=0.05) for k in keys],
        )
        env, arrive = tb.deliver(LEAD, msg_card.get_center(), run_time=1.0)
        self.say(
            "[Sending appends one line to the receiver's file]: [Otto asks the tester for tests.]",
            m.AnimationGroup(m.FadeOut(msg_card, target_position=tb.slot(0), scale=0.1), arrive, m.FadeIn(send_code, shift=0.15 * m.UP), run_time=1.3),
            m.AnimationGroup(m.Circumscribe(m.VGroup(line_part(send_code[1], SEND_SRC, 5), line_part(send_code[1], SEND_SRC, 6)), color=CORAL, buff=0.06), m.Indicate(st.badges[LEAD][0], color=CORAL, scale_factor=1.1)),
        )
        right.current = send_code
        gone = text("deleted", 20, FAINT).move_to(tb.slot(1))
        self.say(
            "[Reading takes every line], [then deletes the file.] [The inbox is drained.]",
            m.AnimationGroup(right.to(read_code), m.Circumscribe(tb.box, color=AZURE, buff=0.06)),
            m.AnimationGroup(tb.drain(self.at(st, TESTER)), tb.label.animate.set_opacity(0.35), m.FadeIn(gone), m.Circumscribe(line_part(read_code[1], READ_SRC, 5, "inbox.unlink()"), color=ROSE, buff=0.05)),
            m.Indicate(st.badges[TESTER][0], color=AZURE, scale_factor=1.1),
        )

        ctx = ContextColumn(center=(RIGHT_X, 0.1, 0), width=RIGHT_W, height=3.3, title=f"{TESTER}'s messages", gap=0.12)
        sys_row = ctx.message("system", SYSTEM_T, size=20)
        inbox_src = INBOX_MSG["content"]
        shown = '<inbox>[{"from": "lead", "to": "tester",\n  "content": "Write tests for apply_code.",\n  "type": "message", "ts": …}]</inbox>'
        assert inbox_src.startswith(shown.split("\n")[0]) and inbox_src.endswith("}]</inbox>") and JOB_T in inbox_src
        user_row = message("user", shown, width=RIGHT_W - 0.36, size=20, mono=True)
        paint(user_row[2][1], shown, {r"</?inbox>": SUN})
        sys_row.move_to(ctx.panel).align_to(ctx.panel, m.UP).shift(0.18 * m.DOWN)
        user_row.move_to(ctx.panel).next_to(sys_row, m.DOWN, buff=0.12)
        pkt = envelope(LEAD).move_to(self.at(st, TESTER))
        self.say(
            "[The tester's next turn starts with them]: [one user message], [wrapped in inbox tags.]",
            m.AnimationGroup(right.to(None), m.FadeIn(ctx.frame), m.FadeIn(sys_row, shift=0.2 * m.UP)),
            m.Succession(m.FadeIn(pkt, scale=0.5, run_time=0.3), m.AnimationGroup(m.FadeOut(pkt, target_position=user_row, scale=0.5), m.FadeIn(user_row, target_position=pkt, scale=0.6), run_time=0.9)),
            m.AnimationGroup(*[m.Circumscribe(g, color=SUN, buff=0.04) for g in (glyphs(user_row[2][1], "<inbox>"), glyphs(user_row[2][1], "</inbox>"))]),
        )

        # the race
        y = -2.95
        steps = [("1", "read_text()", AZURE), ("2", "send()", ROSE), ("3", "unlink()", AZURE)]
        chips = m.VGroup()
        for n, label, color in steps:
            w = code(f"{n}  {label}", 20, INK).width + 0.5
            chips.add(chip(f"{n}  {label}", color, ROSE_FILL if color == ROSE else CARD, width=w, height=0.6, size=24, mono=True))
        chips.arrange(m.RIGHT, buff=0.75).move_to([-1.3, y, 0])
        arrows = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(chips, chips[1:])])
        who = m.VGroup(code(TESTER, 20, AZURE), code(REVIEWER, 20, ORCHID), code(TESTER, 20, AZURE))
        for w, c in zip(who, chips):
            w.next_to(c, m.UP, buff=0.12)
        src = cite("learn-claude-code · s15").next_to(chips, m.RIGHT, buff=0.4)
        stray, arrive2 = tb.deliver(REVIEWER, self.at(st, REVIEWER), run_time=0.9)
        tb.items = []
        cross = m.VGroup(m.Line([-0.17, -0.17, 0], [0.17, 0.17, 0]), m.Line([-0.17, 0.17, 0], [0.17, -0.17, 0])).set_stroke(ROSE, 5).move_to(stray)
        lost = code(f"{RACE_LOST} message lost", 20, ROSE, weight=m.BOLD).next_to(tb.box, m.DOWN, buff=0.12).align_to(tb.box, m.RIGHT)
        self.say(
            "[Learn Claude Code admits this version is racy]: [a message written between the read and the delete] [is simply lost.]",
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(chips[0], shift=0.1 * m.UP), m.FadeIn(who[0]), m.FadeIn(chips[2], shift=0.1 * m.UP), m.FadeIn(who[2]), m.FadeOut(gone), tb.label.animate.set_opacity(1)),
            m.AnimationGroup(m.GrowArrow(arrows[0]), m.FadeIn(chips[1], shift=0.1 * m.UP), m.FadeIn(who[1]), m.GrowArrow(arrows[1]), arrive2),
            m.AnimationGroup(m.Circumscribe(chips[2], color=ROSE, buff=0.06), m.Create(cross), m.FadeIn(lost, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.play(*[m.FadeOut(x) for x in (ctx.frame, sys_row, user_row, chips, arrows, who, src, stray, cross, lost)], run_time=0.8)

    # ------------------------------------------------------------------ a team at work

    def team_at_work(self):
        st = self.st
        self.next_section("3 · A team at work", notes=f"The run, computed: Otto (the lead) mails both jobs, then works on; the tester's {len(TESTS)} tests give '{RESULT_T}' (apply_code(total, code) = total - discount); the reviewer finds apply_code({CART}, \"{CODE}\") == {BAD}. Both results wait in lead.jsonl ({2} lines) until the lead's next round reads them together (read_inbox returned {len(LEAD_MAIL)} messages). teammate_loop is an excerpt: s15's teammate reads its inbox each round and appends new mail as <inbox>…</inbox>; the rest is agent_loop. Like Episode 23's notifications, mail is drained before the next model call. Feature and wording illustrative.")
        loop_code = code_block(LOOP_SRC, size=19, width=RIGHT_W, title="teammate_loop()", accent=EDGE).move_to([RIGHT_X, 0.45, 0])
        read_line = m.VGroup(line_part(loop_code[1], LOOP_SRC, 2), line_part(loop_code[1], LOOP_SRC, 3), line_part(loop_code[1], LOOP_SRC, 4))
        right = Slot(self)
        boxes = st.boxes

        self.say(
            "[Here's the team at work.] [Each teammate runs the same loop], [with one addition]: [check the inbox first.]",
            self.new_header("03 · a team at work", "Nobody waits on anybody"),
            right.to(loop_code),
            m.Circumscribe(read_line, color=AZURE, buff=0.06),
            m.LaggedStart(*[m.Circumscribe(boxes[n].box, color=COLORS[n][0], buff=0.05) for n in (TESTER, REVIEWER)], lag_ratio=0.4),
        )

        log_title = code("messages sent", 20, MUTED).move_to([RIGHT_X - RIGHT_W / 2, 2.2, 0], aligned_edge=m.LEFT)
        rows = m.VGroup(*[log_row(msg) for msg in LOG]).arrange(m.DOWN, buff=0.14)
        rows.next_to(log_title, m.DOWN, buff=0.14, aligned_edge=m.LEFT)
        e2, a2 = boxes[REVIEWER].deliver(LEAD, self.at(st, LEAD))
        self.say(
            "[Otto also asks the reviewer for a review], [then gets on with its own work.]",
            m.AnimationGroup(right.to(m.VGroup(log_title, rows[0])), m.Succession(m.Wait(0.4), m.AnimationGroup(a2, m.FadeIn(rows[1], shift=0.15 * m.UP)))),
            m.Indicate(st.badges[LEAD][0], color=CORAL, scale_factor=1.12),
        )
        right.current = None
        e3, a3 = boxes[LEAD].deliver(TESTER, self.at(st, TESTER))
        self.say(
            "[The tester writes three tests], [runs them], [and mails back]: [three passed.]",
            m.Indicate(st.badges[TESTER][0], color=AZURE, scale_factor=1.12),
            m.Circumscribe(st.badges[TESTER], color=AZURE, buff=0.08),
            m.AnimationGroup(a3, m.FadeIn(rows[2], shift=0.15 * m.UP)),
            m.Circumscribe(glyphs(rows[2][2][1], RESULT_T), color=MINT, buff=0.05),
        )
        e4, a4 = boxes[LEAD].deliver(REVIEWER, self.at(st, REVIEWER))
        self.say(
            "[The reviewer finds a bug]: [a ten dollar code on a five dollar cart] [gives minus five.]",
            m.AnimationGroup(boxes[REVIEWER].drain(self.at(st, REVIEWER)), m.Indicate(st.badges[REVIEWER][0], color=ORCHID, scale_factor=1.12)),
            m.AnimationGroup(a4, m.FadeIn(rows[3], shift=0.15 * m.UP)),
            m.Circumscribe(glyphs(rows[3][2][1], str(BAD)), color=ROSE, buff=0.06),
        )
        got = code(f"{len(LEAD_MAIL)} read", 20, CORAL, weight=m.BOLD).move_to(boxes[LEAD].slot(0), aligned_edge=m.LEFT).shift(0.2 * m.LEFT)
        self.say(
            "[Both replies wait in Otto's file], [until its next round reads them together], [much like the notifications of episode twenty-three.]",
            m.Circumscribe(boxes[LEAD].box, color=CORAL, buff=0.06),
            m.AnimationGroup(boxes[LEAD].drain(self.at(st, LEAD)), m.Indicate(st.badges[LEAD][0], color=CORAL, scale_factor=1.12), m.Circumscribe(rows[2:], color=CORAL, buff=0.06), m.FadeIn(got, shift=0.1 * m.LEFT)),
            m.Wait(0.5),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ protocols

    def protocols(self):
        self.next_section("4 · Protocols", notes="Learn Claude Code s16: s15's plain-text messages can't make agents agree. Shutdown: 'Killing the thread outright leaves half-written files on disk', so the lead asks and the teammate confirms after wrapping up. Plan approval: a high-risk change is reviewed by the lead first. Both are request/response pairs correlated by request_id (as a tool_result carries its call's tool_use_id, Episode 2). The tracker: ProtocolState {request_id, type, sender, target, status} in pending_requests (s16 also stores payload and created_at).")
        cards = []
        for i, (name, color) in enumerate((("shutdown", ROSE), ("plan approval", SUN))):
            box = card(6.0, 4.75, EDGE, CARD, radius=0.22, stroke=2.5).move_to([(-1 if i == 0 else 1) * 3.35, -0.45, 0])
            head = title_text(name, 32, color).move_to(box.get_corner(m.UL) + [0.4, -0.55, 0], aligned_edge=m.LEFT)
            cards.append(NS(box=box, head=head))
        left, right = cards
        half = notes_card(["## Review of checkout.py", "- apply_code can go bel"], width=3.6).move_to(left.box).shift(0.2 * m.UP)
        cut = m.Line(half[2][1].get_corner(m.UR) + [0.08, 0.05, 0], half[2][1].get_corner(m.DR) + [0.08, -0.05, 0]).set_stroke(ROSE, 5)
        killed = pill("killed mid-write", ROSE, ROSE_FILL, 22).next_to(half, m.DOWN, buff=0.3)
        quote = text("“Killing the thread outright leaves\nhalf-written files on disk.”", 22, MUTED)
        quote.next_to(killed, m.DOWN, buff=0.25)
        src_l = cite("learn-claude-code · s16").next_to(quote, m.DOWN, buff=0.12).align_to(quote, m.LEFT)
        left_body = m.VGroup(half, cut, killed, quote, src_l)
        left_body.move_to(left.box).shift(0.3 * m.DOWN)

        plan = notes_card(["rewrite the shared", "fixtures in conftest.py"], width=3.0, title="plan", accent=AZURE)
        mate = badge(TESTER, 0.5).next_to(plan, m.LEFT, buff=0.35)
        lead_ = badge(LEAD, 0.5)
        ask = title_text("?", 44, SUN)
        grp = m.VGroup(mate, plan).move_to(right.box).shift(0.75 * m.UP)
        lead_.move_to([plan.get_x(), plan.get_bottom()[1] - 1.45, 0])
        ask.next_to(lead_[0], m.RIGHT, buff=0.3)
        go = arrow(plan.get_bottom(), lead_[0].get_top() + 0.25 * m.UP, SUN, buff=0.12)
        first = text("check before acting", 22, MUTED).next_to(lead_, m.LEFT, buff=0.6)

        self.say(
            "[Plain messages work], [until two agents must agree on something.]",
            self.new_header("04 · protocols", "When agents must agree"),
            m.AnimationGroup(m.FadeIn(left.box), m.FadeIn(left.head, shift=0.1 * m.UP), m.FadeIn(right.box), m.FadeIn(right.head, shift=0.1 * m.UP), lag_ratio=0.15),
        )
        self.say(
            "[First, shutdown.] [Kill a teammate mid-write], [and it leaves half-written files behind.]",
            m.Circumscribe(left.head, color=ROSE, buff=0.08),
            m.AnimationGroup(m.FadeIn(half, shift=0.1 * m.UP), m.Create(cut), m.FadeIn(killed, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(src_l)),
        )
        self.say(
            "[Second, plan approval]: [before a risky change], [the lead should see the plan.]",
            m.Circumscribe(right.head, color=SUN, buff=0.08),
            m.AnimationGroup(m.FadeIn(mate, shift=0.1 * m.UP), m.FadeIn(plan, shift=0.1 * m.UP)),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(lead_, shift=0.1 * m.UP), m.FadeIn(ask, scale=0.6), m.FadeIn(first)),
        )

        req = {"type": "shutdown_request", "request_id": RID_S}
        resp = {"type": S_RESP["type"], "request_id": S_RESP["request_id"], "approve": S_RESP["approve"]}
        req_card = json_card(req, 22, SUN, CARD, title="request · lead → reviewer").move_to([-3.35, 0.2, 0])
        resp_card = json_card(resp, 22, MINT, CARD, title="response · reviewer → lead").move_to([3.35, 0.2, 0])
        resp_card.align_to(req_card, m.UP)
        id_a, id_b = glyphs(req_card[1], f'"{RID_S}"'), glyphs(resp_card[1], f'"{RID_S}"')
        tie = title_text("=", 44, SUN).move_to([0, (id_a.get_y() + id_b.get_y()) / 2, 0])
        like = cite("like tool_use_id and its call, episode 2").next_to(m.VGroup(req_card, resp_card), m.DOWN, buff=0.35)
        state = code_block(STATE_SRC, size=22, title="the tracker's record", accent=EDGE).move_to([0, -0.2, 0])
        sb = state[1]
        lines = STATE_SRC.split("\n")

        def part(i, key):
            k = sum(sum(not c.isspace() for c in ln) for ln in lines[:i])
            start = lines[i].index(key)
            skip = sum(c.isspace() for c in lines[i][:start])
            return sb[k + start - skip:k + start - skip + sum(not c.isspace() for c in key)]

        self.say(
            "[Both are a request and a response], [tied together by a request ID], [like a tool result and its call in episode two.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(left.box, left.head, left_body, right.box, right.head, grp, lead_, ask, go, first), shift=0.2 * m.DOWN), m.FadeIn(req_card, shift=0.2 * m.UP), m.FadeIn(resp_card, shift=0.2 * m.UP), lag_ratio=0.2),
            m.AnimationGroup(m.Circumscribe(id_a, color=SUN, buff=0.05), m.Circumscribe(id_b, color=SUN, buff=0.05), m.FadeIn(tie, scale=0.6)),
            m.FadeIn(like, shift=0.1 * m.UP),
        )
        self.say(
            "[A tracker keeps a record per request]: [its ID], [its type], [who sent it to whom], [and a status that starts as pending.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(req_card, resp_card, tie, like), shift=0.3 * m.UP), m.FadeIn(state, shift=0.3 * m.UP)),
            m.Circumscribe(part(2, "request_id"), color=SUN, buff=0.05),
            m.Circumscribe(part(3, "type"), color=SUN, buff=0.05),
            m.Circumscribe(m.VGroup(part(4, "sender"), part(5, "target")), color=SUN, buff=0.05),
            m.Circumscribe(part(6, 'status: str = "pending"'), color=SUN, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the protocol stage (sections 5–6)

    def build_stage(self, mate):
        st = NS()
        st.lead_x, st.mate_x, st.badge_y, st.box_y = -5.1, -1.85, 2.05, 1.2
        st.lead = badge(LEAD, 0.42, side=m.RIGHT).move_to([st.lead_x, st.badge_y, 0])
        st.lead_box = Mailbox(LEAD, [st.lead_x, st.box_y, 0], 3.05, gap=0.4)
        st.mate = badge(mate, 0.42, side=m.RIGHT).move_to([st.mate_x, st.badge_y, 0])
        st.mate_box = Mailbox(mate, [st.mate_x, st.box_y, 0], 3.05, gap=0.4)
        st.area = np.array([-3.45, -0.35, 0])  # where messages are read

        # the state machine
        sy = 1.05
        st.start = m.Dot([0.4, sy, 0], radius=0.09, color=MUTED)
        st.nodes = {
            "pending": state_chip("pending").move_to([2.8, sy, 0]),
            "approved": state_chip("approved").move_to([5.5, sy + 1.0, 0]),
            "rejected": state_chip("rejected").move_to([5.5, sy - 1.0, 0]),
        }
        n = st.nodes
        st.edges = {
            "request": arrow(st.start.get_center(), n["pending"].get_left(), MUTED, buff=0.12),
            "approved": arrow(n["pending"].get_right() + 0.12 * m.UP, n["approved"].get_left(), MINT, buff=0.12),
            "rejected": arrow(n["pending"].get_right() + 0.12 * m.DOWN, n["rejected"].get_left(), ROSE, buff=0.12),
        }
        st.edge_labels = m.VGroup(
            code("approve: true", 20, MINT).next_to(n["approved"], m.UP, buff=0.15).align_to(n["approved"], m.RIGHT),
            code("approve: false", 20, ROSE).next_to(n["rejected"], m.DOWN, buff=0.15).align_to(n["rejected"], m.RIGHT),
        )
        st.req_label = code("request", 20, MUTED).next_to(st.edges["request"], m.UP, buff=0.12)
        st.machine = m.VGroup(st.start, *n.values(), *st.edges.values(), st.edge_labels, st.req_label)
        for node in n.values():
            node[0].set_fill(opacity=0.35)
            node[0].set_stroke(opacity=0.45)
            node[1].set_opacity(0.5)
        st.ring = None

        # the tracker
        st.cols = [-6.3, -3.75, -0.75, 3.6]
        head = m.VGroup(*[code(s, 20, MUTED) for s in ("request_id", "type", "sender → target", "status")])
        for h, x in zip(head, st.cols):
            h.move_to([x, -1.75, 0], aligned_edge=m.LEFT)
        align_baselines(head, ["request_id", "type", "sender→target", "status"])
        st.rule = m.Line([-6.3, -1.98, 0], [6.3, -1.98, 0]).set_stroke(EDGE, 2)
        st.track_head = m.VGroup(head, st.rule)
        st.track_rows = []
        return st

    def light(self, st, state, run_time=0.6):
        """Make `state` the current node: fully lit, with a ring; dim the rest."""
        anims = []
        for name, node in st.nodes.items():
            on = name == state
            anims.append(node[0].animate.set_fill(opacity=1 if on else 0.35).set_stroke(opacity=1 if on else 0.45))
            anims.append(node[1].animate.set_opacity(1 if on else 0.5))
        ring = m.SurroundingRectangle(st.nodes[state], buff=0.1, corner_radius=0.2).set_stroke(STATUS[state][0], 3)
        if st.ring is None:
            anims.append(m.Create(ring))
        else:
            anims.append(m.Transform(st.ring, ring))
            ring = st.ring
        st.ring = ring
        return m.AnimationGroup(*anims, run_time=run_time)

    def unlight(self, st):
        anims = [m.FadeOut(st.ring)] if st.ring is not None else []
        st.ring = None
        for node in st.nodes.values():
            anims.append(node[0].animate.set_fill(opacity=0.35).set_stroke(opacity=0.45))
            anims.append(node[1].animate.set_opacity(0.5))
        return m.AnimationGroup(*anims)

    def track_row(self, st, s, status=None):
        y = -2.3 - 0.48 * len(st.track_rows)
        cells = m.VGroup(
            code(s.request_id, 20, SUN, weight=m.BOLD),
            code(s.type, 20, INK),
            code(f"{s.sender} → {s.target}", 20, MUTED),
        )
        for c, x in zip(cells, st.cols):
            c.move_to([x, y, 0], aligned_edge=m.LEFT)
        align_baselines(cells, [s.request_id, s.type, f"{s.sender}→{s.target}"])
        pill_ = self.status_pill(st, status or "pending", y)
        row = NS(cells=cells, pill=pill_, y=y, state=s)
        st.track_rows.append(row)
        return row

    def status_pill(self, st, status, y):
        c, f = STATUS[status]
        return pill(status, c, f, 20, mono=True).move_to([st.cols[3], y, 0], aligned_edge=m.LEFT)

    def set_status(self, st, row, status):
        new = self.status_pill(st, status, row.y)
        anim = m.AnimationGroup(m.FadeOut(row.pill, shift=0.1 * m.UP), m.FadeIn(new, shift=0.1 * m.UP))
        row.pill = new
        return anim

    def msg(self, obj, accent, dashed=False):
        return wire(obj, accent, 19, dashed).move_to(self.stage.area)

    # ------------------------------------------------------------------ graceful shutdown

    def shutdown(self):
        self.next_section("5 · Graceful shutdown", notes=f"The run (s16's flow): the lead creates a pending ProtocolState and mails shutdown_request {RID_S} to the reviewer; the reviewer finishes its notes first ('Alice confirms after wrapping up'), then replies shutdown_response with the same request_id and approve: true; the lead's inbox routes the response to match_response, which sets the status to approved; the teammate's loop ends. Claude Code splits the reply into shutdown_approved / shutdown_rejected (with a reason), then sends teammate_terminated. The notes' contents are illustrative.")
        st = self.stage = self.build_stage(REVIEWER)
        row = self.track_row(st, pending_requests[RID_S], "pending")
        req = self.msg({"type": S_REQ["type"], "request_id": RID_S}, SUN)
        resp = self.msg({"type": S_RESP["type"], "request_id": RID_S, "approve": True}, MINT)
        notes = notes_card(NOTES, width=4.4).move_to(st.area)
        lines = notes[2]
        saved = pill("saved", MINT, MINT_FILL, 20).next_to(notes[1], m.RIGHT, buff=0.3)

        self.say(
            "[The reviewer's job is done.] [So Otto sends a shutdown request], [with a fresh ID.]",
            m.AnimationGroup(
                self.new_header("05 · shutdown", "A graceful goodbye"),
                m.FadeIn(m.VGroup(st.lead, st.mate), shift=0.2 * m.UP),
                m.FadeIn(m.VGroup(st.lead_box.group, st.mate_box.group)),
                m.FadeIn(st.machine), m.FadeIn(st.track_head),
            ),
            m.FadeIn(req, target_position=st.lead[0], scale=0.4),
            m.AnimationGroup(
                m.Circumscribe(glyphs(req[1], f'"{RID_S}"'), color=SUN, buff=0.05),
                m.FadeIn(row.cells, shift=0.1 * m.UP), m.FadeIn(row.pill, shift=0.1 * m.UP),
                pulse(st.edges["request"], SUN), self.light(st, "pending"),
            ),
        )
        env, arrive = st.mate_box.deliver(LEAD, req.get_center())
        self.say(
            "[The record says pending.] [The reviewer doesn't just vanish]: [it finishes writing its notes first.]",
            m.AnimationGroup(m.Circumscribe(row.pill, color=SUN, buff=0.06), m.FadeOut(req, target_position=env, scale=0.1), arrive),
            m.AnimationGroup(st.mate_box.drain(st.mate[0].get_center()), m.Indicate(st.mate[0], color=ORCHID, scale_factor=1.12)),
            m.AnimationGroup(m.FadeIn(notes[:2]), m.LaggedStart(*[m.AddTextLetterByLetter(r, time_per_char=0.03) for r in lines], lag_ratio=0.9), m.FadeIn(saved, scale=0.7)),
        )
        env2, arrive2 = st.lead_box.deliver(REVIEWER, st.area)
        self.say(
            "[Then it replies with the same ID]: [approved.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(notes, saved), shift=0.2 * m.DOWN), m.FadeIn(resp, target_position=st.mate[0], scale=0.4), m.Circumscribe(glyphs(resp[1], f'"{RID_S}"'), color=SUN, buff=0.05)),
            m.Circumscribe(glyphs(resp[1], '"approve": true'), color=MINT, buff=0.05),
        )
        bye = pill("exited", MUTED, CARD_HI, 20).next_to(st.mate, m.RIGHT, buff=0.25)
        self.say(
            "[Otto matches the ID], [marks the request approved], [and the reviewer's loop ends cleanly.]",
            m.AnimationGroup(m.FadeOut(resp, target_position=env2, scale=0.1), arrive2, m.Succession(m.Wait(0.5), m.AnimationGroup(st.lead_box.drain(st.lead[0].get_center()), m.Circumscribe(row.cells[0], color=SUN, buff=0.05)))),
            m.AnimationGroup(pulse(st.edges["approved"], MINT), self.light(st, "approved"), self.set_status(st, row, "approved")),
            m.AnimationGroup(st.mate.animate.set_opacity(0.3), st.mate_box.group.animate.set_opacity(0.3), m.FadeIn(bye)),
        )
        self.bye = bye
        self.wait(0.3)

    # ------------------------------------------------------------------ plan approval

    def plan_approval(self):
        st = self.stage
        self.next_section("6 · Plan approval", notes=f"The run: Otto fixes apply_code (max(0, …)) and mails the tester; the tester's plan ('{PLAN_1}') goes out as plan_approval_request {RID_P1} (teammate → lead); Otto replies approve: false with '{NO}'; the tester's messages get '{REJECTED_NOTE}' (s16). A new plan ('{PLAN_2}') gets a new id, {RID_P2}. For example, a stray shutdown_response carrying {RID_P2} reaches the lead: match_response finds the request, but its type is plan_approval, so it returns without a change — 'A shutdown_response cannot accidentally approve a plan_approval request' (s16). Then approve: true; the tests: '{RESULT_2}'. 'The key idea is correlation, not ceremony' (s16). The plans and the stray reply are illustrative. Episode 17: the ask rule, here with the lead in the user's seat.")
        old_mate, old_box = st.mate, st.mate_box
        st.mate = badge(TESTER, 0.42, side=m.RIGHT).move_to([st.mate_x, st.badge_y, 0])
        st.mate_box = Mailbox(TESTER, [st.mate_x, st.box_y, 0], 3.05, gap=0.4)
        fix = code("max(0, total - discount)", 20, MINT).move_to(st.area + 0.35 * m.UP)
        fix_cap = text("Otto's fix", 20, MUTED).next_to(fix, m.UP, buff=0.15)
        fix_grp = m.VGroup(fix_cap, fix)
        plan1 = notes_card(["rewrite the shared", "fixtures in conftest.py"], width=3.6, title="plan", accent=AZURE).move_to(st.area)
        risky = pill("risky", ROSE, ROSE_FILL, 20).next_to(plan1[1], m.RIGHT, buff=0.3)
        req1 = self.msg({"type": P1["type"], "request_id": RID_P1}, SUN)
        row1 = self.track_row(st, pending_requests[RID_P1], "pending")

        self.say(
            "[Next, Otto fixes the bug], [and asks the tester for a test that covers it.]",
            m.AnimationGroup(
                self.new_header("06 · plan approval", "Ask before the risky change"),
                m.FadeOut(m.VGroup(old_mate, old_box.group, self.bye)), m.FadeIn(st.mate, shift=0.2 * m.UP), m.FadeIn(st.mate_box.group),
                self.unlight(st), m.FadeIn(fix_grp, shift=0.1 * m.UP),
            ),
            m.AnimationGroup(m.FadeOut(fix_grp, target_position=st.mate[0], scale=0.3), m.Indicate(st.mate[0], color=AZURE, scale_factor=1.12)),
        )
        env1, arrive1 = st.lead_box.deliver(TESTER, st.area)
        self.say(
            "[The tester's plan]: [rewrite the shared test fixtures.] [That's risky], [so it asks the lead first], [as Otto asked you in episode seventeen.]",
            m.AnimationGroup(m.FadeIn(plan1[0], shift=0.1 * m.UP), m.FadeIn(plan1[1], shift=0.1 * m.UP)),
            m.FadeIn(plan1[2], shift=0.1 * m.UP),
            m.FadeIn(risky, scale=0.7),
            m.AnimationGroup(*[m.FadeOut(x, shift=0.2 * m.DOWN) for x in (plan1[0], plan1[1], plan1[2], risky)], m.FadeIn(req1, target_position=st.mate[0], scale=0.4), m.FadeIn(row1.cells, shift=0.1 * m.UP), m.FadeIn(row1.pill, shift=0.1 * m.UP), pulse(st.edges["request"], SUN), self.light(st, "pending")),
            m.AnimationGroup(m.FadeOut(req1, target_position=env1, scale=0.1), arrive1),
        )
        resp1 = self.msg({"type": R1["type"], "request_id": RID_P1, "approve": False}, ROSE)
        no = code(f"“{NO}”", 20, MUTED).move_to(st.area + 0.55 * m.DOWN)
        env_r, arrive_r = st.mate_box.deliver(LEAD, st.area)
        note = message("user", REJECTED_NOTE, width=2.95, size=20, mono=True)
        note.next_to(st.mate_box.box, m.DOWN, buff=0.15)
        self.say(
            "[Otto says no]: [leave the fixtures, add one test.] [The tester's messages get]: [plan rejected.]",
            m.AnimationGroup(st.lead_box.drain(st.lead[0].get_center()), m.FadeIn(resp1, target_position=st.lead[0], scale=0.4), pulse(st.edges["rejected"], ROSE), self.light(st, "rejected"), self.set_status(st, row1, "rejected")),
            m.AnimationGroup(m.FadeOut(resp1, target_position=env_r, scale=0.1), arrive_r, m.FadeIn(no, shift=0.1 * m.UP)),
            st.mate_box.drain(st.mate[0].get_center()),
            m.FadeIn(note, shift=0.15 * m.DOWN),
        )

        plan2 = notes_card(["add one test to", "tests/test_discount.py"], width=3.6, title="new plan", accent=AZURE).move_to(st.area)
        req2 = self.msg({"type": P2["type"], "request_id": RID_P2}, SUN)
        row2 = self.track_row(st, pending_requests[RID_P2], "pending")
        env2, arrive2 = st.lead_box.deliver(TESTER, st.area)
        self.say(
            "[A new plan], [with a new ID]: [one test in the discount file.] [Pending again.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(note, no)), self.unlight(st), m.FadeIn(plan2, shift=0.1 * m.UP)),
            m.AnimationGroup(m.FadeOut(plan2, shift=0.2 * m.DOWN), m.FadeIn(req2, target_position=st.mate[0], scale=0.4), m.FadeIn(row2.cells, shift=0.1 * m.UP), m.FadeIn(row2.pill, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(glyphs(req2[1], f'"{RID_P2}"'), color=SUN, buff=0.05), m.Circumscribe(row2.cells[0], color=SUN, buff=0.05)),
            m.AnimationGroup(m.FadeOut(req2, target_position=env2, scale=0.1), arrive2, pulse(st.edges["request"], SUN), self.light(st, "pending")),
        )

        stray = self.msg({"type": STRAY["type"], "request_id": RID_P2, "approve": True}, ROSE, dashed=True)
        self.say(
            "[Now suppose a stray shutdown reply arrives], [carrying this very ID], [with approve set to true.]",
            m.FadeIn(stray, shift=0.3 * m.LEFT),
            m.AnimationGroup(m.Circumscribe(glyphs(stray[1], f'"{RID_P2}"'), color=SUN, buff=0.05), m.Circumscribe(row2.cells[0], color=SUN, buff=0.05)),
            m.Circumscribe(glyphs(stray[1], '"approve": true'), color=ROSE, buff=0.05),
        )
        guard = code_block(GUARD_SRC, size=19, title="match_response()", accent=EDGE).move_to([-3.0, -0.35, 0])
        types = m.VGroup(glyphs(stray[1], '"shutdown_response"'), row2.cells[1])
        self.say(
            "[Its type doesn't match], [so nothing changes.] [A shutdown response cannot accidentally approve a plan approval request.]",
            m.AnimationGroup(*[m.Circumscribe(t, color=ROSE, buff=0.05) for t in types]),
            m.AnimationGroup(m.FadeOut(stray, shift=0.6 * m.RIGHT), m.FadeIn(guard, shift=0.15 * m.UP), m.Circumscribe(row2.pill, color=SUN, buff=0.06)),
            m.Circumscribe(line_part(guard[1], GUARD_SRC, 0), color=ROSE, buff=0.05),
        )
        resp2 = self.msg({"type": R2["type"], "request_id": RID_P2, "approve": True}, MINT)
        env3, arrive3 = st.mate_box.deliver(LEAD, st.area)
        term = terminal(["$ pytest -q tests/test_discount.py", RESULT_2], width=5.6, title=TESTER).move_to(st.area)
        term[4][1].set_color(MINT)
        self.say(
            "[Then the real answer comes]: [approved.] [The tester adds its test]: [four passed.]",
            m.AnimationGroup(m.FadeOut(guard, shift=0.2 * m.DOWN), st.lead_box.drain(st.lead[0].get_center()), m.FadeIn(resp2, target_position=st.lead[0], scale=0.4)),
            m.AnimationGroup(pulse(st.edges["approved"], MINT), self.light(st, "approved"), self.set_status(st, row2, "approved")),
            m.AnimationGroup(m.FadeOut(resp2, target_position=env3, scale=0.1), arrive3, m.Succession(m.Wait(0.5), m.AnimationGroup(st.mate_box.drain(st.mate[0].get_center()), m.FadeIn(term[:4]), m.FadeIn(term[4][0])))),
            m.FadeIn(term[4][1], shift=0.1 * m.UP),
        )
        quote = title_text("“The key idea is correlation, not ceremony.”", 36, INK).move_to([0, 0.35, 0])
        src = cite("learn-claude-code · s16").next_to(quote, m.DOWN, buff=0.25)
        keep = m.VGroup(*[m.VGroup(r.cells, r.pill) for r in st.track_rows], st.track_head)
        self.say(
            "[In Learn Claude Code's words]: [the key idea is correlation, not ceremony.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(st.lead, st.mate, st.lead_box.group, st.mate_box.group, st.machine, st.ring, term), shift=0.2 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.Write(quote), m.LaggedStart(*[m.Circumscribe(r.cells[0], color=SUN, buff=0.05) for r in st.track_rows], lag_ratio=0.4)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ Claude Code

    def claude_code(self):
        self.next_section("7 · In Claude Code", notes=f"Learn Claude Code s15, 'Dive into CC Source' (the site's reading of Claude Code's source, not verified by us): inboxes at ~/.claude/teams/{{teamName}}/inboxes/{{agentName}}.json, a JSON array written under proper-lockfile; the lead's inbox poller checks every 1 s (a teammate's permission poller every 500 ms); the page counts {CC_COUNT} message types — its table lists the {len(CC_TYPES)} names shown (sandbox_permission_* is a request/reply pair); teammates cannot spawn teammates (AgentTool forbids it). Fact date: October 2026.")
        path = code("~/.claude/teams/{team}/inboxes/{name}.json", 22, INK, t2c={r"\{team\}|\{name\}": AZURE})
        path_box = card(path.width + 0.7, 0.8, AZURE, CARD, radius=0.16, stroke=2)
        path.move_to(path_box)
        top = m.VGroup(path_box, path)
        top.move_to([0, 1.95, 0]).align_to([-6.55, 0, 0], m.LEFT)
        lock = m.VGroup(icon(recolored("lock", MINT, MINT_FILL, prefix="ep24"), 0.45), code("locked while written", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        poll = m.VGroup(icon(recolored("clock", SUN, SUN_FILL, prefix="ep24"), 0.45), code("polled every 1 s", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        props = m.VGroup(lock, poll).arrange(m.RIGHT, buff=0.8).next_to(top, m.DOWN, buff=0.3).align_to(top, m.LEFT).shift(0.1 * m.RIGHT)
        src = cite("learn-claude-code · s15 · reading Claude Code's source")

        # the flat team: lead → tester → (no teammate of its own)
        lead_ = badge(LEAD, 0.6, label_size=22).move_to([1.95, 1.75, 0])
        mate = badge(TESTER, 0.6, label_size=22).move_to([4.0, 1.75, 0])
        link = arrow(lead_[0].get_right(), mate[0].get_left(), MUTED, buff=0.15)
        no_kid = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.18, width=0.95, height=0.8).set_stroke(ROSE, 2.5), num_dashes=24)
        no_kid.move_to(mate[0][0].get_center() + np.array([1.95, 0, 0]))
        no_link = m.DashedLine(mate[0][0].get_right() + 0.15 * m.RIGHT, no_kid.get_left() + 0.1 * m.LEFT, dash_length=0.08).set_stroke(ROSE, 3)
        cross = m.VGroup(m.Line([-0.18, -0.18, 0], [0.18, 0.18, 0]), m.Line([-0.18, 0.18, 0], [0.18, -0.18, 0])).set_stroke(ROSE, 6).move_to(no_kid)
        flat = text("no teammates of teammates", 22, ROSE).next_to(m.VGroup(mate, no_kid), m.DOWN, buff=0.3).align_to(no_kid, m.RIGHT)
        src.next_to(props, m.DOWN, buff=0.3).align_to(top, m.LEFT)

        big = title_text(f"{CC_COUNT} message types", 32, INK)
        proto = ("shutdown", "plan_approval")
        chips = m.VGroup()
        for name in CC_TYPES:
            color, fill = (AZURE, AZURE_FILL) if name.startswith(proto) else (EDGE, CARD)
            chips.add(chip(name, color, fill, width=code(name, 20).width + 0.4, height=0.46, size=24, mono=True))
        cols = [m.VGroup(*chips[i:i + 5]) for i in (0, 5, 10)]
        for c in cols:
            c.arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
        grid = m.VGroup(*cols).arrange(m.RIGHT, buff=0.3, aligned_edge=m.UP)
        block = m.VGroup(big, grid).arrange(m.DOWN, buff=0.3, aligned_edge=m.LEFT)
        block.move_to([0, -1.75, 0]).align_to([-6.55, 0, 0], m.LEFT)
        group_of = {
            "shutdown": [c for c, n in zip(chips, CC_TYPES) if n.startswith("shutdown")],
            "plan": [c for c, n in zip(chips, CC_TYPES) if n.startswith("plan")],
            "perm": [c for c, n in zip(chips, CC_TYPES) if "permission" in n],
            "task": [c for c, n in zip(chips, CC_TYPES) if n == "task_assignment"],
        }
        named = {id(c) for g in group_of.values() for c in g}
        group_of["rest"] = [c for c in chips if id(c) not in named]

        self.say(
            "[Claude Code's real agent teams work the same way], [in Learn Claude Code's reading of its source.]",
            m.AnimationGroup(self.new_header("07 · claude code", "The same design, for real"), m.FadeIn(top, shift=0.15 * m.UP)),
            m.FadeIn(src),
        )
        self.say(
            "[Each inbox is a JSON file], [locked while it's written], [and checked about once a second.]",
            m.Circumscribe(path, color=AZURE, buff=0.06),
            m.FadeIn(lock, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(poll, shift=0.1 * m.UP), m.Broadcast(m.Circle(radius=0.45).set_stroke(SUN, 3), focal_point=poll[0].get_center(), n_mobs=3)),
        )
        self.say(
            "[There are about fifteen message types]: [shutdown], [plan approval], [permissions], [task assignments], [and more.]",
            m.AnimationGroup(m.FadeIn(big, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.08)),
            m.AnimationGroup(*[m.Circumscribe(c, color=AZURE, buff=0.04) for c in group_of["shutdown"]]),
            m.AnimationGroup(*[m.Circumscribe(c, color=AZURE, buff=0.04) for c in group_of["plan"]]),
            m.AnimationGroup(*[m.Circumscribe(c, color=SUN, buff=0.04) for c in group_of["perm"]]),
            m.AnimationGroup(*[m.Circumscribe(c, color=SUN, buff=0.04) for c in group_of["task"]]),
            m.LaggedStart(*[m.Indicate(c, color=INK, scale_factor=1.06) for c in group_of["rest"]], lag_ratio=0.15, run_time=0.9),
        )
        self.say(
            "[And one rule keeps the team flat]: [teammates cannot spawn teammates.]",
            m.AnimationGroup(m.FadeIn(lead_, shift=0.1 * m.UP), m.GrowArrow(link), m.FadeIn(mate, shift=0.1 * m.UP)),
            m.AnimationGroup(m.Create(no_link), m.FadeIn(no_kid), m.Create(cross), m.FadeIn(flat, shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ a caution

    def caution(self):
        self.next_section("8 · A caution", notes="Li, AI Agents in Depth, ch. 10: 'Go has a widely quoted maxim: Do not communicate by sharing memory; instead, share memory by communicating.' Failure mode one: 'Two Agents modify the same file simultaneously, and the later write overwrites the earlier one.' Whether a team beats a single agent at all (new information, isolation, ~15× the tokens) is Episode 26.")
        quote = m.VGroup(title_text("“Do not communicate by sharing memory;", 34, INK), title_text("share memory by communicating.”", 34, AZURE)).arrange(m.DOWN, buff=0.2)
        quote.move_to([0, 1.65, 0])
        src = cite("a Go maxim, quoted in Li, AI Agents in Depth · ch. 10").next_to(quote, m.DOWN, buff=0.25)

        def panel(x, title, color):
            box = card(6.0, 2.6, color, CARD, radius=0.2, stroke=2).move_to([x, -1.75, 0])
            cap = text(title, 22, color, weight=m.BOLD).move_to(box.get_corner(m.UL) + [0.35, -0.38, 0], aligned_edge=m.LEFT)
            return box, cap

        lbox, lcap = panel(-3.35, "sharing memory", ROSE)
        rbox, rcap = panel(3.35, "communicating", MINT)
        a, b = otto(0.55, AZURE, AZURE_FILL), otto(0.55, ORCHID, ORCHID_FILL)
        shared = notes_card(["DISCOUNT = 10"], width=2.4, title="config.py", accent=ROSE)
        m.VGroup(a, shared, b).arrange(m.RIGHT, buff=0.85).move_to(lbox).shift(0.05 * m.DOWN)
        wa = arrow(a.get_right(), shared.get_left(), AZURE, buff=0.08)
        wb = arrow(b.get_left(), shared.get_right(), ORCHID, buff=0.08)
        later = text("the later write wins", 20, ROSE).next_to(shared, m.DOWN, buff=0.15)
        a2, b2 = otto(0.55, AZURE, AZURE_FILL), otto(0.55, ORCHID, ORCHID_FILL)
        m.VGroup(a2, b2).arrange(m.RIGHT, buff=3.0).move_to(rbox).shift(0.1 * m.DOWN)
        env = envelope(TESTER, 0.34).move_to(a2.get_center())
        mailtrip = arrow(a2.get_right(), b2.get_left(), MUTED, buff=0.15)
        explicit = text("every hand-off explicit", 20, MINT).next_to(mailtrip, m.DOWN, buff=0.35)
        next_up = pill(f"episode 26 · {EPISODES[26][0]}", AZURE, AZURE_FILL, 22).move_to([0, -1.85, 0])

        self.say(
            "[The book AI Agents in Depth quotes a maxim from the Go language]: [do not communicate by sharing memory]; [share memory by communicating.]",
            m.AnimationGroup(self.new_header("08 · a caution", "Share memory by communicating"), m.FadeIn(src)),
            m.FadeIn(quote[0], shift=0.15 * m.UP),
            m.FadeIn(quote[1], shift=0.15 * m.UP),
        )
        self.say(
            "[Two agents writing one file]: [the later write wins.] [Messages make every hand-off explicit.]",
            m.AnimationGroup(m.FadeIn(m.VGroup(lbox, lcap)), m.FadeIn(m.VGroup(a, b, shared)), m.GrowArrow(wa), m.GrowArrow(wb)),
            m.AnimationGroup(m.FadeIn(later, shift=0.1 * m.UP), m.Wiggle(shared[0])),
            m.AnimationGroup(m.FadeIn(m.VGroup(rbox, rcap)), m.FadeIn(m.VGroup(a2, b2)), m.GrowArrow(mailtrip), m.Succession(m.FadeIn(env, scale=0.5, run_time=0.3), env.animate(run_time=0.9).move_to(b2.get_center()), m.FadeOut(env, scale=0.4, run_time=0.3)), m.FadeIn(explicit, shift=0.1 * m.UP)),
        )
        self.say(
            "[Whether a team beats one agent at all] [is the question of episode twenty-six.]",
            m.FadeOut(m.VGroup(lbox, lcap, a, b, shared, wa, wb, later, rbox, rcap, a2, b2, mailtrip, explicit), shift=0.2 * m.DOWN),
            m.FadeIn(next_up, shift=0.15 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Persist (teammates keep their own context), message (mailbox files, drained each round), correlate (request_id ties each response to its request).")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 64, c) for w, c in (("Persist", AZURE), ("Message", CORAL), ("Correlate", SUN))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("teammates keep context", ".mailboxes/{name}.jsonl", "request_id")])
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
        if row_.width > 12.6:
            row_.scale(12.6 / row_.width)
        tagline = text("Teammates that stay, mailboxes that carry the words,\nand IDs that tie each answer to its question.", 28, MUTED)
        tagline.next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [persist], [message], [correlate.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Teammates that stay, mailboxes that carry the words, and IDs that tie each answer to its question.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
