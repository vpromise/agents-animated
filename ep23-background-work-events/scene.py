"""Agents, Animated · Episode 23 · Background Work & Events (Module 5 · Scaling Up)

How agents keep working while slow things happen and the world interrupts. Running example: Otto is asked
to rebuild tinyshop and update the changelog. The build is slow; in Episode 2's loop the harness would
block on it. Here it runs in a side lane: the model gets a placeholder at once, keeps working, and a
<task_notification> arrives with a later tool result. Then a nightly check fires from a cron job, the
outside world becomes a queue of events (push, not poll), and a user's "stop" takes the cancel lane.

The runs are not drawn by hand: the scene runs learn-claude-code s13's background mechanism (the run_tool
on screen is the code that ran) against a scripted model on a simulated clock, s14's cron matcher and
scheduler second by second, a poll-vs-push timing model and a cancellable rename; every number on screen
comes from them (asserted below).

Arc: hook (one lane: the loop waits on the build; a nightly check and "stop" go unheard) · 1 background
tasks (the washing machine; slow commands → background thread; the placeholder) · 2 the side lane (Otto's
run on a clock; the notification injected beside a tool result; 3 turns during the build; Claude Code's
stall watchdog) · 3 handles (a phone call returns a task ID; connected/ended come later; never fake
success) · 4 cron (CronJob; scheduler → queue → idle → "[Scheduled] …"; process must live; jitter and
expiry) · 5 events (checkout counter vs secretary; structured events; batch loop; numbered events) ·
6 push vs poll (the one-time code) · 7 urgency lanes (cancel / queue / parallel; Otto's stop is not undo)
· recap (Start · Notify · Prioritize).

Colours (series roles): model turns CORAL, a running tool SUN, tool results and notifications MINT, user
and outside events AZURE; good MINT, bad ROSE; lanes cancel ROSE · queue SUN · parallel MINT; headers
AZURE (Module 5).

Facts (checked October 2026):
- learn-claude-code s13 (learn.shareai.run/en/s13): slow bash commands (pip install torch ≈ 10 min, npm run
  build ≈ 3 min) block the loop; "a washing machine": start it, do other things, it signals when done.
  The model can set run_in_background; the teaching fallback is_slow_operation matches "install", "build",
  "test", "deploy", "compile", "docker build", "pip install", "npm install", "cargo build", "pytest",
  "make" (commands "likely to take > 30s"); start_background_task gives ids like bg_0001 and runs a daemon
  thread; the placeholder "[Background task {bg_id} started] Result will be available when complete." (we
  show its first part); collect_background_results formats <task_notification> with <task_id>, <status>,
  <command>, <summary> (output[:200]); notifications go as text blocks in the same user message as the
  turn's tool_result blocks, because "one tool_use gets exactly one tool_result". Dive into CC Source (the
  site's reading of Claude Code's source, not verified by us): no threads, "background" means not
  awaiting; a stall watchdog: after 45 seconds with no output growth it looks for prompts like (y/n).
- learn-claude-code s14 (learn.shareai.run/en/s14): "An alarm clock doesn't need you to watch it";
  "Recurring work should be created by the harness, not remembered by the model"; a scheduler thread wakes
  every second, a "YYYY-MM-DD HH:MM" minute marker prevents double fires; cron_queue; a queue processor
  starts a turn only when the agent is idle; the turn receives "[Scheduled] {prompt}"; CronJob {id, cron
  ("0 9 * * *" in the page), prompt, recurring, durable}; durable jobs in .scheduled_tasks.json; "The cron
  scheduler must run inside the agent process" — otherwise system crontab or a systemd timer. Claude Code
  per the site's reading (unverified): "trigger delay up to 10% of period (max 15 min)"; "Recurring tasks
  auto-expire after 7 days".
- Li, AI Agents in Depth, ch. 6 (async and event-driven): "a single checkout counter" vs "a flexible
  secretary"; events with source, channel, content, context (priority); "each round takes a batch of
  events off the input queue, appends them to the trajectory, invokes the LLM once", then waits; batch
  markers "[Unprocessed Event 1/4]" because a model "may respond only to the last event"; three lanes:
  cancellation ("Stop! I said the wrong thing.": force "a safe point early" by "proactively interrupting
  the current step"), queued (appended when "any tool call completes"; e.g. "only show results from the
  last month"), parallel ("What's the weather like today?", "a parallel reasoning session"); "a lightweight
  classification LLM as an event router"; initiate_phone_call "immediately returning a task identifier and
  initial status" — initiation and completion "should be decoupled", not "call equals completion"; an
  unfinished tool result "must never be faked as a success"; PineClaw: mid-call the user must give a
  one-time code at once; with Heartbeat polling (every N minutes) "the representative hangs up and the
  call fails"; the fix is a push Channel; "receiving a 'stop' message does not itself undo actions that
  have already occurred".
Illustrative (for teaching): tinyshop's changelog and README edits, the build's output, all durations
(turns of 40 s, a 110 s build, a 5-minute heartbeat, a 1-minute wait on the phone, a 15 s reply, 3 s per
file in the rename), the nightly job's id, time and prompt, the phone task id and status, the event JSON
values, the rename script and its five files.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import math
import sys
from datetime import datetime, timedelta
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

REF = (SERIES / "harness_reference.py").read_text()
assert "def run_tool(block):" in REF and "return TOOL_HANDLERS[block.name](**block.input)" in REF

# ============================================================================ Otto's run, computed

TASK = "Rebuild, then update the changelog."
BUILD = "make build"
TURN_S = 40  # one turn: a model call and its quick tools (for example)
BUILD_S = 110  # how long `make build` runs (for example: "about two minutes")
BUILD_OUT = "ok, 13 passed"
FINAL = "Build ok, 13 passed. Docs done."
FILES = {"CHANGELOG.md": "# Changelog\n\n## 0.2.1\n- fix: cart.total() skipped the last item\n",
         "README.md": "# tinyshop\n"}
SLOW_WORDS = ["install", "build", "test", "deploy", "compile", "docker build", "pip install",
              "npm install", "cargo build", "pytest", "make"]  # learn-claude-code s13's fallback list
DURATION = {BUILD: BUILD_S}

CLOCK = [0]  # simulated seconds since the task arrived
background_tasks = {}


def is_slow_operation(command):
    return any(word in command for word in SLOW_WORDS)


def should_run_background(block):
    if block.input.get("run_in_background"):
        return True
    return block.name == "bash" and is_slow_operation(block.input["command"])


def start_background_task(block):
    bg_id = f"bg_{len(background_tasks) + 1:04d}"
    background_tasks[bg_id] = {"command": block.input["command"], "started": CLOCK[0], "status": "running"}
    return bg_id


def run_bash(command):
    """tinyshop's shell, simulated: in this run only the build is ever executed (in the background)."""
    assert command == BUILD
    return BUILD_OUT


def read_file(path):
    return FILES[path]


def edit_file(path, old_text, new_text):
    FILES[path] = FILES[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}


def run_tool(block):
    if should_run_background(block):
        bg_id = start_background_task(block)
        return f"[Background task {bg_id} started]"
    return TOOL_HANDLERS[block.name](**block.input)


def notification(bg_id, task):
    return (f"<task_notification><task_id>{bg_id}</task_id><status>{task['status']}</status>"
            f"<command>{task['command']}</command><summary>{task['output'][:200]}</summary></task_notification>")


def collect_background_results():
    notes = []
    for bg_id, task in background_tasks.items():
        done_at = task["started"] + DURATION[task["command"]]
        if task["status"] == "running" and CLOCK[0] >= done_at:
            task.update(status="completed", output=run_bash(task["command"]), finished=done_at, seen=CLOCK[0])
            notes.append({"type": "text", "text": notification(bg_id, task)})
    return notes


# The model's decisions in this run (illustrative): what a model would plausibly do, call by call.
PLAN = [
    ("bash", {"command": BUILD}),
    ("read_file", {"path": "CHANGELOG.md"}),
    ("edit_file", {"path": "CHANGELOG.md", "old_text": "## 0.2.1", "new_text": "## 0.3.0\n- rebuilt\n\n## 0.2.1"}),
    ("edit_file", {"path": "README.md", "old_text": "# tinyshop", "new_text": "# tinyshop 0.3.0"}),
    None,  # answers in words
]


class ScriptedModel:
    """Stands in for client.messages: each call takes TURN_S seconds of the simulated clock."""

    def __init__(self):
        self.calls = []  # (start, end) of each call

    def create(self, messages, tools):
        start = CLOCK[0]
        CLOCK[0] += TURN_S
        self.calls.append((start, CLOCK[0]))
        step = PLAN[len(self.calls) - 1]
        if step is None:
            return NS(stop_reason="end_turn", content=[NS(type="text", text=FINAL)])
        name, args = step
        block = NS(type="tool_use", id=f"toolu_{len(self.calls):02d}", name=name, input=args)
        return NS(stop_reason="tool_use", content=[block])


client = NS(messages=ScriptedModel())
TOOLS = [{"name": "bash"}, {"name": "read_file"}, {"name": "edit_file"}]


def tool_results(response):
    """harness_reference.agent_loop's inner for-loop, named so the loop stays short."""
    return [{"type": "tool_result", "tool_use_id": b.id, "content": run_tool(b)}
            for b in response.content if b.type == "tool_use"]


def agent_loop(messages):
    while True:
        response = client.messages.create(messages=messages, tools=TOOLS)
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = tool_results(response)
        notes = collect_background_results()
        messages.append({"role": "user",
            "content": notes + results})


RUN_TOOL_SRC = '''def run_tool(block):
    if should_run_background(block):
        bg_id = start_background_task(block)
        return f"[Background task {bg_id} started]"
    return TOOL_HANDLERS[block.name](**block.input)'''
LOOP_TAIL_SRC = '''notes = collect_background_results()
messages.append({"role": "user",
    "content": notes + results})'''

MESSAGES = [{"role": "user", "content": TASK}]
ANSWER = agent_loop(MESSAGES)
CALLS = client.messages.calls
TASK_BG = background_tasks["bg_0001"]
PLACEHOLDER = MESSAGES[2]["content"][0]["content"]
NOTE = MESSAGES[8]["content"][0]["text"]

# the code on screen is the code that ran
assert inspect.getsource(run_tool).strip() == RUN_TOOL_SRC
_loop_src = inspect.getsource(agent_loop)
assert all(line.strip() in _loop_src for line in LOOP_TAIL_SRC.split("\n"))
# the run
assert is_slow_operation(BUILD) and not is_slow_operation("CHANGELOG.md")
assert PLACEHOLDER == "[Background task bg_0001 started]"
assert len(MESSAGES) == 10 and ANSWER.stop_reason == "end_turn" and ANSWER.content[0].text == FINAL
assert CALLS == [(0, 40), (40, 80), (80, 120), (120, 160), (160, 200)]
BUILD_START, BUILD_END = TASK_BG["started"], TASK_BG["finished"]
assert (BUILD_START, BUILD_END, TASK_BG["seen"]) == (40, 150, 160)  # finishes during turn 4, seen at its end
TURNS_DURING = [k + 1 for k, (a, b) in enumerate(CALLS) if a < BUILD_END and b > BUILD_START and k > 0]
assert TURNS_DURING == [2, 3, 4]  # "three turns of work happened while it ran"; "finishes during turn four"
assert [b["type"] for b in MESSAGES[8]["content"]] == ["text", "tool_result"]  # beside the tool result
assert NOTE.startswith("<task_notification><task_id>bg_0001</task_id><status>completed</status>")
assert f"<summary>{BUILD_OUT}</summary>" in NOTE and "13 passed" in BUILD_OUT and "13 passed" in FINAL
_uses = sum(1 for msg in MESSAGES[1::2] for b in msg["content"] if getattr(b, "type", "") == "tool_use")
_results = sum(1 for msg in MESSAGES[2::2] for b in msg["content"] if b["type"] == "tool_result")
assert _uses == _results == 4  # every tool_use gets exactly one tool_result
BLOCKING_END = len(CALLS) * TURN_S + BUILD_S  # Episode 2's loop: the same five turns, plus the wait
assert BLOCKING_END == 310 and CALLS[-1][1] == 200


def mmss(seconds):
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


# ============================================================================ the nightly job (s14), computed

JOB = {"id": "nightly", "cron": "0 2 * * *", "prompt": "Run the tests.", "recurring": True, "durable": True}


def field_matches(field, value):
    for part in field.split(","):
        if part == "*":
            return True
        if part.startswith("*/") and value % int(part[2:]) == 0:
            return True
        if "-" in part:
            lo, hi = map(int, part.split("-"))
            if lo <= value <= hi:
                return True
        elif part.isdigit() and int(part) == value:
            return True
    return False


def cron_matches(expr, t):
    minute, hour, dom, month, dow = expr.split()
    day_ok = field_matches(dom, t.day)
    week_ok = field_matches(dow, (t.weekday() + 1) % 7)
    days = (day_ok or week_ok) if dom != "*" and dow != "*" else (day_ok and week_ok)
    return field_matches(minute, t.minute) and field_matches(hour, t.hour) and field_matches(month, t.month) and days


def scheduler(job, start, seconds):
    """s14's scheduler thread: wake every second; fire at most once per minute (the minute marker)."""
    fired, last_marker = [], None
    for s in range(seconds):
        t = start + timedelta(seconds=s)
        marker = t.strftime("%Y-%m-%d %H:%M")
        if cron_matches(job["cron"], t) and marker != last_marker:
            fired.append(t)
            last_marker = marker
    return fired


def period_minutes(expr, start=datetime(2026, 10, 22, 0, 0)):
    fires = [start + timedelta(minutes=k) for k in range(3 * 24 * 60) if cron_matches(expr, start + timedelta(minutes=k))]
    return (fires[1] - fires[0]).total_seconds() / 60


def jitter_cap(expr):
    """Claude Code, per learn-claude-code s14's reading: delay up to 10% of the period, max 15 minutes."""
    return min(0.1 * period_minutes(expr), 15)


FIRES = scheduler(JOB, datetime(2026, 10, 22, 1, 59, 50), 100)
assert FIRES == [datetime(2026, 10, 22, 2, 0, 0)]  # one fire, at two o'clock, despite 60 matching ticks
assert period_minutes(JOB["cron"]) == 24 * 60 and jitter_cap(JOB["cron"]) == 15  # "at most fifteen minutes"
assert jitter_cap("*/30 * * * *") == 3
SCHEDULED_MSG = f"[Scheduled] {JOB['prompt']}"
EXPIRY_DAYS = 7  # Claude Code, per learn-claude-code s14 (cited, not computed)

# ============================================================================ push vs poll, computed

HEARTBEAT_S = 300  # the agent polls every five minutes (for example; the book says "every N minutes")
NEED_S = 80  # the representative asks for the one-time code (for example)
PATIENCE_S = 60  # how long the representative waits (for example)
REPLY_S = 15  # how fast the user answers once asked (for example)
POLL_SEEN = math.ceil(NEED_S / HEARTBEAT_S) * HEARTBEAT_S
HANG_UP = NEED_S + PATIENCE_S
PUSH_CODE = NEED_S + REPLY_S
assert POLL_SEEN == 300 and POLL_SEEN > HANG_UP == 140  # "the representative hangs up"
assert PUSH_CODE == 95 <= HANG_UP  # "the call goes through"
POLL_LATE = POLL_SEEN - NEED_S
assert mmss(POLL_LATE) == "3:40"

# ============================================================================ the stop, computed

RENAME = "./rename.sh total cart_total"
RENAME_FILES = ["cart.py", "checkout.py", "receipts.py", "tests/test_cart.py", "README.md"]
PER_FILE_S = 3  # the rename script takes 3 s per file (for example)
STOP_AT_S = 7  # "stop, wrong branch!" arrives 7 s in (for example)


def rename_until_cancelled(files, stop_at):
    """The script writes one file at a time and checks for a cancel signal between files (ctx.Done())."""
    changed = []
    for k, path in enumerate(files):
        if (k + 1) * PER_FILE_S > stop_at:
            return changed, f"[cancelled after {len(changed)} of {len(files)} files]"
        changed.append(path)
    return changed, "done"


CHANGED, CANCEL_RESULT = rename_until_cancelled(RENAME_FILES, STOP_AT_S)
assert CHANGED == ["cart.py", "checkout.py"] and CANCEL_RESULT == "[cancelled after 2 of 5 files]"

# ============================================================================ look

MODEL_ICON = recolored("llm", CORAL, CORAL_FILL, prefix="ep23")
WASHER, ALARM, PHONE = "ep23-washer", "ep23-alarm", "ep23-phone"

JSON_T2C = {
    r"\"[^\"\n]*\"(?=\s*:)": AZURE,
    r":\s*\"[^\"\n]*\"": MINT,
    r"\b-?\d+(\.\d+)?\b": SUN,
    r"\b(true|false|null)\b": ORCHID,
}
XML_T2C = {r"</?[a-z_]+>": AZURE, r"bg_\d+": SUN}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
ROW_H, ROW_GAP = 0.45, 0.06


def cite(s):
    return text(s, 20, MUTED)


def base_of(words):
    """The baseline of a Text: the median bottom of its flat-bottomed glyphs."""
    chars = [c for c in words.text if not c.isspace()]
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(words, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else words.get_bottom()[1]


def sit(words, y):
    return words.shift((y - base_of(words)) * m.UP)


def row(role, body, accent, width, mono=True, body_color=None):
    """One message of the context on one line: a stripe in the block's colour, the role, the content."""
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    words = code(body, 20, body_color or accent) if mono else text(body, 20, INK)
    label.move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
    words.next_to(label, m.RIGHT, buff=0.2)
    base = box.get_y() - 0.09
    sit(label, base)
    sit(words, base)
    assert words.get_right()[0] < box.get_right()[0] - 0.08, body
    return m.VGroup(box, stripe, label, words)


def row2(role, lines, accent, width):
    """A message with two content lines (a notification beside a tool result)."""
    h = 2 * ROW_H - 0.1
    box = card(width, h, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=h - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    top_base = box.get_top()[1] - 0.32
    sit(label, top_base)
    words = m.VGroup()
    for k, (s, c) in enumerate(lines):
        w = code(s, 20, c).next_to(label, m.RIGHT, buff=0.25)
        sit(w, top_base - 0.36 * k)
        assert w.get_right()[0] < box.get_right()[0] - 0.08, s
        words.add(w)
    return m.VGroup(box, stripe, label, words)


class CodeCard:
    """A code_block with per-line glyph groups and a line cursor (a tinted bar)."""

    def __init__(self, src, title=None, width=None, size=20):
        block = code_block(src, size=size, width=width, title=title)
        self.block, self.box, self.body = block, block[0], block[1]
        for part in block[1:]:
            part.set_z_index(2)
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n

    def bar(self, a, b=None, color=AZURE):
        b = a if b is None else b
        top = max(g.get_top()[1] for g in self.lines[a]) + 0.08
        bot = min(g.get_bottom()[1] for g in self.lines[b]) - 0.08
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


class Column(ContextColumn):
    """kit.ContextColumn whose pushed message may fly in from a source (it still scrolls the stack)."""

    def push(self, msg, run_time=0.9, source=None):
        group = super().push(msg, run_time)
        if source is None:
            return group
        anims = list(group.animations[:-1]) + [m.FadeIn(msg, target_position=source)]
        return m.AnimationGroup(*anims, run_time=run_time)

    def push_after(self, *pairs):
        """Several pushes in one cue, one after another, so a scroll never moves a row still fading in.
        Each pair is (message, source or None)."""
        steps = []
        for msg, source in pairs:
            group = self.push(msg, source=source)
            for anim in group.animations[:-1]:
                if not isinstance(anim, m.FadeOut):
                    anim.introducer = True
            steps.append(group)
        return m.Succession(*steps)


class Slot:
    """One place on screen whose content is swapped."""

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


def span(x0, x1, y, h, color, fill, label=None, label_color=INK, size=20, stroke=2):
    box = card(x1 - x0, h, color, fill, radius=0.1, stroke=stroke).move_to([(x0 + x1) / 2, y, 0])
    if label is None:
        return m.VGroup(box)
    words = code(label, size, label_color).move_to(box)
    sit(words, y - 0.08)
    return m.VGroup(box, words)


def note_card(lines, width, accent=EDGE, title=None, size=22):
    """A card of reading lines: (string, colour) pairs; an optional mono title."""
    rows = m.VGroup(*[text(s, size, c) for s, c in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
    parts = [rows]
    if title:
        cap = code(title, 20, MUTED)
        parts.insert(0, cap)
    body = m.VGroup(*parts).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
    box = card(width, body.height + 0.5, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
    return m.VGroup(box, body)


def wire(src, accent, size=20, title=None):
    """JSON (or XML) as it travels: Menlo, coloured, in a card of the block's colour."""
    t2c = XML_T2C if src.lstrip().startswith("<") else JSON_T2C
    body = code(src, size, INK, t2c=t2c)
    top = 0.0
    cap = None
    if title:
        cap = code(title, 20, accent, weight=m.BOLD)
        top = cap.height + 0.22
    box = card(max(body.width, cap.width if cap else 0) + 0.7, body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    group = m.VGroup(box, body)
    if cap:
        cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
        group.add(cap)
    return group


def labelled_chip(label, color, fill, size=20, mono=True, height=0.6, pad=0.45):
    words = code(label, size, INK) if mono else text(label, size, INK, weight=m.SEMIBOLD)
    return chip(label, color, fill, width=words.width + pad, height=height, size=size + (4 if mono else 0), mono=mono)


def check_mark(color=MINT, size=30):
    return title_text("✓", size, color)


def cross_mark(color=ROSE, size=30):
    return title_text("✗", size, color)


# ============================================================================ the episode


class Ep23BackgroundWorkEvents(Episode):
    number = 23

    def lesson(self):
        self.hook()
        self.background()
        self.side_lane()
        self.handles()
        self.cron()
        self.events()
        self.push_poll()
        self.lanes()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Otto's job: rebuild tinyshop and update the changelog. In Episode 2's loop the harness runs the build and waits: turn 1, then {mmss(BUILD_S)} of waiting, then the other turns — {mmss(BLOCKING_END)} in all (computed; durations illustrative). A nightly check and a user's 'stop' arrive meanwhile and go unheard. Today: background tasks, scheduled jobs, events and interrupts.")
        bot = otto(0.85).move_to([-5.65, 1.75, 0])
        person = icon("user", 0.8).move_to([5.85, 1.85, 0])
        ask = bubble(text("Rebuild tinyshop, then update the changelog.", 26, INK), AZURE, AZURE_FILL, tail="right")
        ask.move_to([1.6, 2.0, 0])
        ask.shift((person.get_left()[0] - 0.15 - ask[0].get_vertices()[2][0]) * m.RIGHT)

        X0, X1, Y = -6.3, 6.3, -0.25
        sx = lambda t: X0 + (X1 - X0) * t / BLOCKING_END  # noqa: E731
        lane_lbl = code("one lane · the loop from episode 2", 20, MUTED).move_to([X0, Y - 1.45, 0], aligned_edge=m.LEFT)
        turn1 = span(sx(0) + 0.04, sx(TURN_S) - 0.04, Y, 0.6, CORAL, CORAL_FILL, "call 1")
        wait = span(sx(TURN_S) + 0.04, sx(TURN_S + BUILD_S) - 0.04, Y, 0.6, ROSE, ROSE_FILL)
        wait_words = m.VGroup(code(f"waiting on {BUILD}", 20, ROSE), code(mmss(BUILD_S), 20, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.3)
        wait_words.move_to(wait)
        sit(wait_words[0], Y - 0.08)
        sit(wait_words[1], Y - 0.08)
        later = m.VGroup(*[
            span(sx(TURN_S + BUILD_S + TURN_S * (k - 2)) + 0.04, sx(TURN_S + BUILD_S + TURN_S * (k - 1)) - 0.04, Y, 0.6,
                 FAINT, CARD, f"call {k}", MUTED)
            for k in range(2, len(CALLS) + 1)
        ])
        axis = m.Line([X0, Y - 0.55, 0], [X1, Y - 0.55, 0]).set_stroke(FAINT, 2)
        ticks = m.VGroup()
        for t in range(0, BLOCKING_END + 1, 60):
            ticks.add(m.Line([sx(t), Y - 0.55, 0], [sx(t), Y - 0.67, 0]).set_stroke(FAINT, 2))
            ticks.add(code(mmss(t), 20, MUTED).next_to([sx(t), Y - 0.67, 0], m.DOWN, buff=0.1))
        nothing = text("nothing else happens until it returns", 24, MUTED).next_to(lane_lbl, m.DOWN, buff=0.3).align_to(lane_lbl, m.LEFT)

        def event_card(pic, words, color, fill):
            body = m.VGroup(icon(pic, 0.42), text(words, 22, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2)
            box = card(body.width + 0.5, 0.7, color, fill, radius=0.16, stroke=2)
            return m.VGroup(box, body.move_to(box))

        nightly = event_card(ALARM, "nightly check", AZURE, AZURE_FILL).next_to(bot, m.RIGHT, buff=0.3).set_y(1.0)
        stop = event_card("user", "you: stop!", AZURE, AZURE_FILL).next_to(nightly, m.RIGHT, buff=0.3)
        drops = m.VGroup(*[m.DashedLine(e.get_bottom() + 0.06 * m.DOWN, [e.get_x(), Y + 0.36, 0], dash_length=0.07).set_stroke(ROSE, 2.5)
                           for e in (nightly, stop)])
        unheard = text("nobody is listening", 24, ROSE).next_to(stop, m.RIGHT, buff=0.4)
        topics = m.VGroup(*[pill(s, AZURE, AZURE_FILL, 22) for s in ("background tasks", "scheduled jobs", "events and interrupts")])
        topics.arrange(m.RIGHT, buff=0.4).move_to([0, -2.85, 0])
        assert nightly.get_right()[0] < stop.get_left()[0] - 0.1 and nightly.get_left()[0] > bot.get_right()[0] + 0.1
        assert sx(TURN_S) < nightly.get_x() < stop.get_x() < sx(TURN_S + BUILD_S)  # both arrive during the wait

        self.say(
            "[Otto has a new job]: [rebuild tinyshop, then update the changelog.]",
            m.FadeIn(bot, shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(person, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP)),
        )
        self.say(
            "[The build is slow]: [for example, about two minutes.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(person, ask), shift=0.2 * m.UP), m.FadeIn(lane_lbl), m.Create(axis), m.FadeIn(ticks), m.FadeIn(turn1, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowFromEdge(wait, m.LEFT), m.FadeIn(wait_words), run_time=1.6),
        )
        self.say(
            "[In the loop from episode two], [the harness runs the command, and waits.] [Nothing else happens until it returns.]",
            m.Circumscribe(lane_lbl, color=MUTED, buff=0.06),
            m.Circumscribe(wait, color=ROSE, buff=0.06),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.LEFT) for c in later], lag_ratio=0.2), m.FadeIn(nothing, shift=0.1 * m.UP)),
        )
        self.say(
            "[Meanwhile, the world keeps moving]: [a nightly check comes due], [and you type: stop.]",
            m.Circumscribe(bot, color=AZURE, buff=0.12),
            m.AnimationGroup(m.FadeIn(nightly, shift=0.3 * m.DOWN), m.Create(drops[0])),
            m.AnimationGroup(m.FadeIn(stop, shift=0.3 * m.DOWN), m.Create(drops[1]), m.FadeIn(unheard, shift=0.1 * m.LEFT)),
        )
        self.say(
            "[Today: how an agent keeps working while slow things run], [and how the outside world gets in.]",
            m.AnimationGroup(m.FadeOut(nothing), m.FadeIn(topics[0], shift=0.15 * m.UP)),
            m.LaggedStart(*[m.FadeIn(t, shift=0.15 * m.UP) for t in topics[1:]], lag_ratio=0.4),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 1 background tasks

    def background(self):
        self.next_section("1 · Background tasks", notes="learn-claude-code s13: 'a washing machine' (start it, do other things, it signals when done). Slow commands run in a daemon thread: the model can set run_in_background, or the teaching fallback is_slow_operation matches keywords (install, build, test, deploy, compile, pytest, make, …: commands 'likely to take > 30s'). run_tool (the scene runs exactly this code) starts the task and returns the placeholder '[Background task bg_0001 started]' at once (s13 adds 'Result will be available when complete.').")
        washer = icon(WASHER, 2.0).move_to([-5.0, 1.15, 0])
        beep = title_text("beep!", 34, SUN).next_to(washer, m.RIGHT, buff=0.3).align_to(washer, m.UP)
        steps = m.VGroup(*[text(s, 24, c) for s, c in (("start it", INK), ("walk away", INK), ("it beeps when done", SUN))])
        steps.arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18).next_to(washer, m.DOWN, buff=0.4).align_to(washer, m.LEFT)
        src = cite("learn-claude-code · s13").next_to(steps, m.DOWN, buff=0.35).align_to(steps, m.LEFT)

        cc = CodeCard(RUN_TOOL_SRC, title="run_tool(block) · learn-claude-code s13")
        cc.block.move_to([6.6 - cc.box.width / 2, 2.55 - cc.box.height / 2, 0])
        assert cc.box.get_left()[0] > beep.get_right()[0] + 0.25, cc.box.get_left()
        words = [w for w in SLOW_WORDS if " " not in w and w != "compile"]
        assert words == ["install", "build", "test", "deploy", "pytest", "make"]
        kw = m.VGroup(*[labelled_chip(w, SUN, SUN_FILL) for w in words]).arrange(m.RIGHT, buff=0.16)
        kw.next_to(cc.box, m.DOWN, buff=0.3).align_to(cc.box, m.LEFT)
        flag = labelled_chip('"run_in_background": true', SUN, CARD)
        flag.next_to(kw, m.DOWN, buff=0.2).align_to(kw, m.LEFT)
        result = wire(f'"content": "{PLACEHOLDER}"', MINT, title="tool_result")
        result.next_to(flag, m.DOWN, buff=0.35).align_to(cc.box, m.LEFT)
        assert result.get_bottom()[1] > -3.55 and kw.get_right()[0] < 6.6 and result.get_right()[0] < 6.6
        ph = glyphs(result[1], f'"{PLACEHOLDER}"')
        bar = cc.bar(1, 3)

        self.say(
            "[Learn Claude Code's picture is a washing machine.] [You start it, walk away], [and it beeps when it's done.]",
            m.AnimationGroup(self.new_header("01 · background", "Start it, walk away"), m.FadeIn(washer, shift=0.2 * m.UP), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(steps[0], shift=0.1 * m.UP), m.FadeIn(steps[1], shift=0.1 * m.UP), m.Rotate(washer[7], angle=2 * np.pi, about_point=washer[5].get_center(), run_time=1.4)),
            m.AnimationGroup(m.FadeIn(beep, scale=0.6), m.FadeIn(steps[2], shift=0.1 * m.UP)),
        )
        self.say(
            "[Slow commands], [installs, builds, tests, deploys], [run in a background thread.] [The model can also ask for that with a flag.]",
            m.FadeIn(cc.block, shift=0.2 * m.UP),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in kw], lag_ratio=0.15),
            m.AnimationGroup(m.FadeIn(bar), m.Circumscribe(cc.lines[1], color=SUN, buff=0.06)),
            m.FadeIn(flag, shift=0.1 * m.UP),
        )
        self.say(
            "[In run tool], [a slow command starts a background task], [and returns at once.]",
            m.Circumscribe(cc.lines[0], color=AZURE, buff=0.06),
            m.Transform(bar, cc.bar(2)),
            m.Transform(bar, cc.bar(3)),
        )
        self.say(
            "[The model gets a placeholder]: [background task started, with an ID.]",
            m.FadeIn(result, shift=0.2 * m.UP),
            m.AnimationGroup(m.Circumscribe(ph, color=MINT, buff=0.06), m.Indicate(beep, color=SUN, scale_factor=1.15)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 2 the side lane

    def side_lane(self):
        self.next_section("2 · The side lane", notes=f"Otto's run (scripted model, simulated clock; durations illustrative): turns of {TURN_S} s; call 1 asks for {BUILD}, which starts at {mmss(BUILD_START)} and finishes at {mmss(BUILD_END)}, during turn 4; collect_background_results() sees it at the end of turn 4 ({mmss(TASK_BG['seen'])}) and the <task_notification> goes into that user message as a text block beside the tool_result (learn-claude-code s13: 'one tool_use gets exactly one tool_result'). Turns {TURNS_DURING} ran during the build; the run ends at {mmss(CALLS[-1][1])} instead of {mmss(BLOCKING_END)}. Claude Code (s13's Dive into CC Source, unverified): no threads, 'background' = don't await; a stall watchdog after 45 s without output growth looks for (y/n) prompts.")
        X0, X1 = -4.85, 6.4
        T_END = CALLS[-1][1]
        sx = lambda t: X0 + (X1 - X0) * t / T_END  # noqa: E731
        Y_MAIN, Y_SIDE, Y_AXIS = 2.0, 1.32, 0.8
        main_lbl = sit(code("main loop", 20, MUTED).move_to([-6.6, 0, 0], aligned_edge=m.LEFT), Y_MAIN - 0.08)
        side_lbl = sit(code("background", 20, MUTED).move_to([-6.6, 0, 0], aligned_edge=m.LEFT), Y_SIDE - 0.08)
        assert side_lbl.get_right()[0] < X0 - 0.15
        verbs = ["build", "read", "edit", "edit", "answer"]
        chips = [span(sx(a) + 0.04, sx(b) - 0.04, Y_MAIN, 0.5, CORAL, CORAL_FILL, f"{k + 1} {v}") for k, ((a, b), v) in enumerate(zip(CALLS, verbs))]
        track = card(sx(BUILD_END) - sx(BUILD_START), 0.5, SUN, CARD, radius=0.1, stroke=2).move_to([(sx(BUILD_START) + sx(BUILD_END)) / 2, Y_SIDE, 0])

        def fill_at(t):
            f = max(0.004, (min(t, BUILD_END) - BUILD_START) / (BUILD_END - BUILD_START))
            r = m.RoundedRectangle(corner_radius=0.03, width=(track.width - 0.2) * f, height=0.06, stroke_width=0, fill_color=SUN, fill_opacity=1)
            return r.align_to(track, m.LEFT).shift(0.1 * m.RIGHT).set_y(Y_SIDE - 0.18)

        fill = fill_at(BUILD_START)
        build_lbl = sit(code(f"{BUILD} · bg_0001", 20, INK).move_to(track), Y_SIDE - 0.03).set_z_index(2)
        done = m.VGroup(check_mark(MINT, 24), code(mmss(BUILD_END), 20, MINT)).arrange(m.RIGHT, buff=0.12)
        done.next_to(track, m.RIGHT, buff=0.15)
        axis = m.Line([X0, Y_AXIS, 0], [X1, Y_AXIS, 0]).set_stroke(FAINT, 2)
        ticks = m.VGroup()
        for t in range(0, T_END + 1, TURN_S):
            ticks.add(m.Line([sx(t), Y_AXIS, 0], [sx(t), Y_AXIS - 0.1, 0]).set_stroke(FAINT, 2))
            ticks.add(code(mmss(t), 20, MUTED).next_to([sx(t), Y_AXIS - 0.1, 0], m.DOWN, buff=0.07))
        clock = Slot(self)
        clock_at = lambda t: sit(code(f"t = {mmss(t)}", 20, AZURE).move_to([-6.6, 0, 0], aligned_edge=m.LEFT), Y_AXIS - 0.08)  # noqa: E731
        assert clock_at(T_END).get_right()[0] < X0 - 0.15
        cursor = m.Triangle(fill_color=AZURE, fill_opacity=1, stroke_width=0).scale(0.09).rotate(np.pi)
        cursor.move_to([sx(0), Y_AXIS + 0.11, 0])
        move_cursor = lambda t: cursor.animate.move_to([sx(t), Y_AXIS + 0.11, 0])  # noqa: E731

        ctx = Column(center=(3.3, -1.8, 0), width=6.6, height=3.4, title="messages", gap=ROW_GAP)
        assert ctx.title.get_bottom()[1] > -0.1 and ctx.title.get_top()[1] < ticks.get_bottom()[1] - 0.05
        w = ctx.width - 0.36
        rows = []
        for i, msg in enumerate(MESSAGES):
            content = msg["content"]
            if isinstance(content, str):
                r = row("user", content, AZURE, w, mono=False)
            elif msg["role"] == "assistant":
                b = content[0]
                if b.type == "text":
                    r = row("assistant", b.text, CORAL, w, mono=False)
                else:
                    arg = b.input.get("command") or b.input.get("path")
                    r = row("assistant", f'{b.name}("{arg}")', SUN, w)
            elif len(content) == 2:
                r = row2("user", [("<task_notification> bg_0001", AZURE), (content[1]["content"], MINT)], MINT, w)
            else:
                out = content[0]["content"]
                r = row("user", out.split("\n")[0] + (" …" if "\n" in out else ""), MINT, w)
            rows.append(r)

        detail = Slot(self)
        DX, DY = -6.6, 0.0  # the detail panel's top-left corner
        loop_card = CodeCard(LOOP_TAIL_SRC, title="agent_loop(): the end of every turn")
        loop_card.block.move_to([DX + loop_card.box.width / 2, DY - loop_card.box.height / 2, 0])
        note_src = NOTE.replace("><", ">\n<").replace("\n<task_id>", "\n  <task_id>").replace("\n<status>", "\n  <status>")
        note_src = note_src.replace("\n<command>", "\n  <command>").replace("\n<summary>", "\n  <summary>")
        note_lines = note_src.split("\n")
        assert len(note_lines) == 6
        xml = wire(note_src, MINT, 20)
        xml.move_to([DX + xml.width / 2, DY - xml.height / 2, 0])
        cc_note = note_card([
            ("no threads: background = don't await", INK),
            ("a watchdog: 45 s without new output", INK),
            ("then it looks for a (y/n) prompt", SUN),
        ], 6.2, EDGE, title="Claude Code · learn-claude-code s13", size=22)
        cc_note.move_to([DX + cc_note.width / 2, DY - cc_note.height / 2, 0])
        for d in (loop_card.block, xml, cc_note):
            assert d.get_bottom()[1] > -3.55 and d.get_right()[0] < ctx.panel.get_left()[0] - 0.25, d

        def xml_line(k):
            return glyphs(xml[1], note_lines[k].strip())

        brace_y = Y_MAIN + 0.36
        bx0, bx1 = sx(CALLS[1][0]) + 0.04, sx(CALLS[3][1]) - 0.04
        brace = m.VGroup(
            m.Line([bx0, brace_y, 0], [bx1, brace_y, 0]),
            m.Line([bx0, brace_y, 0], [bx0, brace_y - 0.08, 0]),
            m.Line([bx1, brace_y, 0], [bx1, brace_y - 0.08, 0]),
        ).set_stroke(SUN, 3)
        during = code(f"{len(TURNS_DURING)} turns while it built", 20, SUN).next_to(brace, m.UP, buff=0.1)
        assert during.get_top()[1] < 2.7
        up = arrow([sx(BUILD_END) - 0.1, Y_SIDE + 0.27, 0], [sx(TASK_BG["seen"]) - 0.12, Y_MAIN - 0.27, 0], MINT, buff=0.0)
        final_words = rows[9][3]

        self.say(
            "[Here's Otto's run.] [Call one asks for the build], [and the placeholder comes straight back.]",
            m.AnimationGroup(self.new_header("02 · side lane", "The loop keeps turning"),
                             m.FadeIn(m.VGroup(main_lbl, side_lbl, axis, ticks, cursor)), m.FadeIn(ctx.frame),
                             ctx.push(rows[0]), detail.to(loop_card.block), clock.to(clock_at(0))),
            m.AnimationGroup(m.FadeIn(chips[0], shift=0.15 * m.RIGHT), ctx.push(rows[1], source=chips[0]), move_cursor(TURN_S), clock.to(clock_at(TURN_S))),
            m.AnimationGroup(ctx.push(rows[2]), m.GrowFromEdge(track, m.LEFT), m.FadeIn(fill), m.FadeIn(build_lbl)),
        )
        self.say(
            "[The loop keeps turning]: [Otto reads the changelog], [and edits it.]",
            m.Circumscribe(chips[0], color=CORAL, buff=0.06),
            m.AnimationGroup(m.FadeIn(chips[1], shift=0.15 * m.RIGHT), ctx.push_after((rows[3], chips[1]), (rows[4], None)),
                             move_cursor(CALLS[1][1]), clock.to(clock_at(CALLS[1][1])), m.Transform(fill, fill_at(CALLS[1][1]))),
            m.AnimationGroup(m.FadeIn(chips[2], shift=0.15 * m.RIGHT), ctx.push_after((rows[5], chips[2]), (rows[6], None)),
                             move_cursor(CALLS[2][1]), clock.to(clock_at(CALLS[2][1])), m.Transform(fill, fill_at(CALLS[2][1]))),
        )
        self.say(
            "[Meanwhile, the build runs in its own lane], [and finishes during turn four.]",
            m.Circumscribe(track, color=SUN, buff=0.06),
            m.AnimationGroup(m.FadeIn(chips[3], shift=0.15 * m.RIGHT), ctx.push(rows[7], source=chips[3]), move_cursor(BUILD_END),
                             clock.to(clock_at(BUILD_END)), m.Transform(fill, fill_at(BUILD_END)), m.FadeIn(done, shift=0.1 * m.LEFT),
                             track.animate.set_stroke(MINT)),
        )
        loop_bar = loop_card.bar(0, 0, MINT)
        self.say(
            "[When turn four's tools return], [the harness collects the finished build], [and adds a notification to that user message], [as a text block, beside the tool result.]",
            m.AnimationGroup(move_cursor(TASK_BG["seen"]), clock.to(clock_at(TASK_BG["seen"])), m.Circumscribe(chips[3], color=CORAL, buff=0.06)),
            m.AnimationGroup(m.FadeIn(loop_bar), m.GrowArrow(up), pulse(up, MINT)),
            m.AnimationGroup(ctx.push(rows[8], source=up.get_end()), m.Transform(loop_bar, loop_card.bar(1, 2, MINT))),
            m.AnimationGroup(m.Circumscribe(rows[8][3][0], color=AZURE, buff=0.05), m.Circumscribe(rows[8][3][1], color=MINT, buff=0.05)),
        )
        self.say(
            "[It carries the task's ID], [its status], [the command], [and a short summary.]",
            m.AnimationGroup(m.FadeOut(loop_bar), detail.to(xml), m.Circumscribe(xml_line(1), color=SUN, buff=0.05)),
            m.Circumscribe(xml_line(2), color=MINT, buff=0.05),
            m.Circumscribe(xml_line(3), color=MINT, buff=0.05),
            m.Circumscribe(xml_line(4), color=MINT, buff=0.05),
        )
        self.say(
            "[Call five reads it]: [the build passed, thirteen tests.] [Three turns of work happened while it ran.]",
            m.AnimationGroup(m.FadeIn(chips[4], shift=0.15 * m.RIGHT), ctx.push(rows[9], source=chips[4]), move_cursor(T_END), clock.to(clock_at(T_END))),
            m.Circumscribe(glyphs(final_words, "13 passed"), color=MINT, buff=0.05),
            m.AnimationGroup(m.Create(brace), m.FadeIn(during, shift=0.1 * m.UP)),
        )
        self.say(
            "[In Learn Claude Code's reading of its source], [Claude Code also watches for stalls]: [forty-five seconds without output], [and it looks for a yes-or-no prompt.]",
            m.AnimationGroup(detail.to(cc_note[0]), m.FadeIn(cc_note[1][0])),
            m.FadeIn(cc_note[1][1][0], shift=0.1 * m.UP),
            m.FadeIn(cc_note[1][1][1], shift=0.1 * m.UP),
            m.FadeIn(cc_note[1][1][2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 3 handles

    def handles(self):
        self.next_section("3 · Started is not finished", notes="Li, AI Agents in Depth, ch. 6: initiate_phone_call 'immediately returning a task identifier and initial status'; phone_call_connected / phone_call_ended arrive later as events; initiation and completion 'should be decoupled' (not 'call equals completion'); an unfinished tool result 'must never be faked as a success'. The task id call_7 and status 'dialing' are illustrative.")
        phone = icon(PHONE, 1.0).move_to([-6.1, 1.75, 0])
        call = labelled_chip("initiate_phone_call()", SUN, SUN_FILL, height=0.62).next_to(phone, m.RIGHT, buff=0.25)
        handle = wire('{"task_id": "call_7", "status": "dialing"}', MINT)
        handle.align_to([6.6, 0, 0], m.RIGHT).set_y(call.get_y())
        assert handle.get_left()[0] - call.get_right()[0] > 0.9
        go = arrow(call.get_right(), handle.get_left(), MINT, buff=0.1)
        at_once = text("at once", 22, MINT).next_to(go, m.UP, buff=0.12)
        assert handle.get_right()[0] < 6.61

        ty = -0.15
        t_axis = m.Arrow([-5.6, ty, 0], [6.3, ty, 0], buff=0, color=FAINT, stroke_width=3, max_tip_length_to_length_ratio=0.03)
        t_lbl = code("time", 20, FAINT).next_to(t_axis, m.DOWN, buff=0.1).align_to(t_axis, m.RIGHT)
        start_dot = m.Dot([-5.3, ty, 0], radius=0.09, color=MINT)
        start_lbl = code("started", 20, MINT).next_to(start_dot, m.DOWN, buff=0.15)
        evs = m.VGroup()
        for name, x in (("phone_call_connected", -1.2), ("phone_call_ended", 3.6)):
            c = labelled_chip(name, AZURE, AZURE_FILL, height=0.56).move_to([x, ty + 0.75, 0])
            stem = m.Line(c.get_bottom(), [x, ty + 0.06, 0]).set_stroke(AZURE, 2.5)
            dot = m.Dot([x, ty, 0], radius=0.08, color=AZURE)
            evs.add(m.VGroup(stem, dot, c))
        later = text("events, later", 22, AZURE).next_to(t_axis, m.DOWN, buff=0.15).set_x(1.2)
        not_done = pill("starting is not finishing", ROSE, ROSE_FILL, 22).move_to([-3.6, -1.25, 0])
        src = cite("Li, AI Agents in Depth · ch. 6").move_to([3.6, -1.25, 0])

        fake = wire('"build passed"', ROSE)
        honest = wire(f'"{PLACEHOLDER}"', MINT)
        pair = m.VGroup(fake, honest).arrange(m.RIGHT, buff=1.3).move_to([0.2, -2.75, 0])
        x_ = cross_mark(ROSE, 34).next_to(fake, m.LEFT, buff=0.25)
        ok = check_mark(MINT, 34).next_to(honest, m.LEFT, buff=0.25)
        fake_lbl = text("faked success", 22, ROSE).next_to(fake, m.UP, buff=0.15)
        honest_lbl = text("honest placeholder", 22, MINT).next_to(honest, m.UP, buff=0.15)
        assert x_.get_left()[0] > -6.6 and honest.get_right()[0] < 6.6

        self.say(
            "[AI Agents in Depth makes the same point with a phone agent.] [Starting a call returns a task ID at once.]",
            m.AnimationGroup(self.new_header("03 · handles", "Started is not finished"), m.FadeIn(phone, shift=0.2 * m.UP), m.FadeIn(call, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(go), m.FadeIn(at_once), m.FadeIn(handle, shift=0.2 * m.LEFT)),
        )
        self.say(
            "[Connected and ended arrive later, as events.] [Starting a call is not finishing it.]",
            m.AnimationGroup(m.Create(t_axis), m.FadeIn(t_lbl), m.FadeIn(start_dot, scale=0.5), m.FadeIn(start_lbl),
                             m.LaggedStart(*[m.FadeIn(e, shift=0.2 * m.DOWN) for e in evs], lag_ratio=0.5), m.FadeIn(later)),
            m.AnimationGroup(m.FadeIn(not_done, scale=0.85), m.FadeIn(src)),
        )
        self.say(
            "[So a placeholder says only what's true]: [started, not done.] [It must never be faked as a success.]",
            m.AnimationGroup(m.FadeIn(honest, shift=0.15 * m.UP), m.FadeIn(honest_lbl), m.FadeIn(ok, scale=0.6)),
            m.Circumscribe(glyphs(honest[1], "started"), color=MINT, buff=0.05),
            m.AnimationGroup(m.FadeIn(fake, shift=0.15 * m.UP), m.FadeIn(fake_lbl), m.FadeIn(x_, scale=0.6)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 4 cron

    def cron(self):
        self.next_section("4 · Scheduled jobs", notes=f"learn-claude-code s14: 'An alarm clock doesn't need you to watch it'; CronJob {{id, cron, prompt, recurring, durable}} (the page's example is '0 9 * * *'; ours, '{JOB['cron']}', two in the morning, is illustrative). The scene's scheduler wakes every second and fires once at {FIRES[0]:%H:%M} thanks to the 'YYYY-MM-DD HH:MM' minute marker; the job goes to cron_queue; a queue processor starts a turn only when the agent is idle; the turn gets '{SCHEDULED_MSG}'. Jobs run only while the agent process lives ('The cron scheduler must run inside the agent process'); otherwise system crontab or a systemd timer. Claude Code per the site's reading (unverified): jitter 'up to 10% of period (max 15 min)' → daily job ≤ {jitter_cap(JOB['cron']):.0f} min, every-30-min job ≤ {jitter_cap('*/30 * * * *'):.0f} min (computed); recurring jobs expire after {EXPIRY_DAYS} days.")
        alarm = icon(ALARM, 0.8).move_to([-6.15, 2.2, 0])
        quote = text("“An alarm clock doesn't need\nyou to watch it.”", 24, INK).next_to(alarm, m.RIGHT, buff=0.25)
        src = cite("learn-claude-code · s14").next_to(quote, m.DOWN, buff=0.1).align_to(quote, m.LEFT)
        job = json_card(JOB, size=19, accent=AZURE, title="CronJob")
        job.move_to([0, 0, 0]).align_to([-6.6, 0, 0], m.LEFT).align_to([0, 1.3, 0], m.UP)
        body = job[1]
        keys = [glyphs(body, '"id": "nightly"'), glyphs(body, f'"cron": "{JOB["cron"]}"'), glyphs(body, f'"prompt": "{JOB["prompt"]}"'),
                glyphs(body, '"recurring": true'), glyphs(body, '"durable": true')]
        when = sit(text(f"fires {FIRES[0]:%-H:%M} every day", 20, AZURE).align_to(job[0], m.RIGHT).shift(0.3 * m.LEFT), base_of(job[2]))
        assert FIRES[0].hour == 2 and FIRES[0].minute == 0 and when.get_left()[0] > job[2].get_right()[0] + 0.3

        PX, PW = 3.75, 5.5

        def stage(title_, sub, y, pic=None):
            box = card(PW, 0.64, EDGE, CARD, radius=0.16, stroke=2).move_to([PX, y, 0])
            parts = [code(title_, 20, INK, weight=m.BOLD)]
            if pic:
                parts.insert(0, icon(pic, 0.42))
            head = m.VGroup(*parts).arrange(m.RIGHT, buff=0.18).move_to(box).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            group = m.VGroup(box, head)
            if sub:
                subw = text(sub, 20, MUTED).move_to(box).align_to(box, m.RIGHT).shift(0.25 * m.LEFT)
                group.add(sit(subw, base_of(head[-1])))
            return group

        sched = stage("scheduler", "", 2.2, ALARM)
        queue = stage("cron_queue", "", 1.15)
        proc = stage("queue processor", "", 0.1)
        sched_slot = Slot(self)
        clock_at = lambda s: sit(code(s, 20, AZURE).move_to(sched[0]).align_to(sched[0], m.RIGHT).shift(0.25 * m.LEFT), base_of(sched[1][-1]))  # noqa: E731
        stopped = sit(text("process stopped", 20, ROSE).move_to(sched[0]).align_to(sched[0], m.RIGHT).shift(0.25 * m.LEFT), base_of(sched[1][-1]))
        assert stopped.get_left()[0] > sched[1].get_right()[0] + 0.2
        job_chip = labelled_chip(JOB["id"], AZURE, AZURE_FILL, height=0.5).move_to(queue[0]).align_to(queue[0], m.RIGHT).shift(0.25 * m.LEFT)
        status = Slot(self)
        status_at = lambda s, c: sit(code(s, 20, c).move_to(proc[0]).align_to(proc[0], m.RIGHT).shift(0.25 * m.LEFT), base_of(proc[1][-1]))  # noqa: E731
        turn = row("user", SCHEDULED_MSG, AZURE, PW, mono=True).move_to([PX, -0.95, 0])
        links = m.VGroup(*[arrow(a[0].get_bottom(), b.get_top() if i == 2 else b[0].get_top(), MUTED, buff=0.06)
                           for i, (a, b) in enumerate(((sched, queue), (queue, proc), (proc, turn)))])

        catch = pill("only while the process lives", ROSE, ROSE_FILL, 22)
        catch.move_to([0, -2.45, 0]).align_to([-6.6, 0, 0], m.LEFT)
        instead = m.VGroup(text("Machine off? Use", 22, MUTED), code("crontab · systemd", 20, MINT)).arrange(m.RIGHT, buff=0.2)
        instead.next_to(catch, m.DOWN, buff=0.28).align_to(catch, m.LEFT)
        for wds in instead[1:]:
            sit(wds, base_of(instead[0]))
        cc = note_card([
            ("delay: up to 10% of the period, max 15 min", INK),
            (f"so a daily job may start {jitter_cap(JOB['cron']):.0f} min late", INK),
            (f"recurring jobs expire after {EXPIRY_DAYS} days", INK),
        ], 6.4, EDGE, title="Claude Code · learn-claude-code s14", size=20)
        cc.align_to([6.6, 0, 0], m.RIGHT).align_to([0, -3.45, 0], m.DOWN)
        assert cc.get_left()[0] > max(catch.get_right()[0], instead.get_right()[0]) + 0.15
        assert cc.get_top()[1] < turn.get_bottom()[1] - 0.12 and job.get_bottom()[1] > catch.get_top()[1] + 0.15
        assert job[0].get_right()[0] < sched[0].get_left()[0] - 0.2 and quote.get_right()[0] < sched[0].get_left()[0] - 0.2

        self.say(
            "[Now the nightly check.] [In Learn Claude Code's words], [an alarm clock doesn't need you to watch it.]",
            m.AnimationGroup(self.new_header("04 · cron", "Work that starts itself"), m.FadeIn(alarm, shift=0.2 * m.UP)),
            m.FadeIn(src),
            m.FadeIn(quote, shift=0.1 * m.UP),
        )
        self.say(
            "[A job is a small record]: [an ID], [a cron expression, two in the morning, every day], [a prompt], [whether it repeats], [and whether it survives a restart.]",
            m.FadeIn(job, shift=0.2 * m.UP),
            m.Circumscribe(keys[0], color=AZURE, buff=0.05),
            m.AnimationGroup(m.Circumscribe(keys[1], color=AZURE, buff=0.05), m.FadeIn(when, shift=0.1 * m.LEFT)),
            m.Circumscribe(keys[2], color=AZURE, buff=0.05),
            m.Circumscribe(keys[3], color=AZURE, buff=0.05),
            m.Circumscribe(keys[4], color=AZURE, buff=0.05),
        )
        self.say(
            "[A scheduler thread checks the clock every second.] [At two o'clock, the job fires into a queue.]",
            m.AnimationGroup(m.FadeIn(sched, shift=0.15 * m.UP), m.Succession(sched_slot.to(clock_at("01:59:58")), sched_slot.to(clock_at("01:59:59"), shift=0.1))),
            m.AnimationGroup(sched_slot.to(clock_at(f"{FIRES[0]:%H:%M:%S}")), m.GrowArrow(links[0]), m.FadeIn(queue, shift=0.15 * m.UP),
                             m.FadeIn(job_chip, target_position=keys[0])),
        )
        self.say(
            "[A queue processor waits until Otto is idle], [then starts a turn with the prompt, marked scheduled.]",
            m.AnimationGroup(m.GrowArrow(links[1]), m.FadeIn(proc, shift=0.15 * m.UP), m.Succession(status.to(status_at("Otto: busy", SUN)), m.Wait(0.4), status.to(status_at("Otto: idle", MINT)))),
            m.AnimationGroup(m.GrowArrow(links[2]), m.FadeOut(job_chip, target_position=turn), m.FadeIn(turn, target_position=job_chip)),
        )
        self.say(
            "[The catch]: [these jobs run only while the agent's process is alive.] [For a machine that's off, use the system's own cron.]",
            m.FadeIn(catch, scale=0.85),
            m.AnimationGroup(sched[0].animate.set_stroke(ROSE), sched_slot.to(stopped)),
            m.FadeIn(instead, shift=0.1 * m.UP),
        )
        self.say(
            "[Claude Code, in Learn Claude Code's reading], [delays each recurring job by up to a tenth of its period, at most fifteen minutes], [and expires it after seven days.]",
            m.AnimationGroup(m.FadeIn(cc[0]), m.FadeIn(cc[1][0])),
            m.AnimationGroup(m.FadeIn(cc[1][1][0], shift=0.1 * m.UP), m.FadeIn(cc[1][1][1], shift=0.1 * m.UP)),
            m.FadeIn(cc[1][1][2], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 5 events

    def events(self):
        self.next_section("5 · Events", notes="Li, AI Agents in Depth, ch. 6: a synchronous agent is 'a single checkout counter'; an asynchronous one 'a flexible secretary' who decides what to handle first by urgency. Events are structured (source, channel, content, context with priority; the book's example is an email, ours is the user's 'stop' — values illustrative). The loop: 'each round takes a batch of events off the input queue, appends them to the trajectory, invokes the LLM once', then waits for the next batch (like Go's for { select { … } }). Batch markers like '[Unprocessed Event 1/4]', because a model 'may respond only to the last event'.")
        pw, ph = 6.1, 3.5
        panels = m.VGroup(*[card(pw, ph, EDGE, "#18150F", radius=0.2, stroke=2) for _ in range(2)])
        panels.arrange(m.RIGHT, buff=0.4).move_to([0, 0.55, 0])
        titles = m.VGroup(title_text("a single checkout counter", 26), title_text("a flexible secretary", 26))
        for t, p in zip(titles, panels):
            t.move_to(p.get_corner(m.UL) + [0.32, -0.42, 0], aligned_edge=m.LEFT)
        # the counter: one desk, a line of customers
        desk = card(1.1, 1.3, AZURE, AZURE_FILL, radius=0.14).move_to(panels[0].get_center() + [2.1, -0.45, 0])
        desk_lbl = text("counter", 20, MUTED).next_to(desk, m.DOWN, buff=0.12)
        people = m.VGroup(*[icon("user", 0.62) for _ in range(4)]).arrange(m.RIGHT, buff=0.22)
        people.next_to(desk, m.LEFT, buff=0.35).align_to(desk, m.DOWN).shift(0.3 * m.UP)
        step = people[1].get_x() - people[0].get_x()
        # the secretary: Otto with a short pile of requests, the urgent one on top
        sec = otto(0.8).move_to(panels[1].get_center() + [1.6, -0.35, 0])
        reqs = m.VGroup(*[labelled_chip(s, c, f, height=0.5, mono=False) for s, c, f in
                          (("build finished", MINT, MINT_FILL), ("nightly check", AZURE, AZURE_FILL), ("stop!", ROSE, ROSE_FILL))])
        reqs.arrange(m.DOWN, buff=0.14).move_to(panels[1].get_center() + [-1.0, -0.45, 0])
        slots = [r.get_center().copy() for r in reqs]
        src = cite("Li, AI Agents in Depth · ch. 6").next_to(panels, m.DOWN, buff=0.25)

        # phase B: the events of this episode, one queue, the loop
        qp = card(6.1, 2.45, EDGE, "#18150F", radius=0.2, stroke=2).move_to([-3.5, 1.25, 0])
        q_title = code("event queue", 20, MUTED).move_to(qp.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)

        def ev_row(tag, body, color, fill, y):
            box = card(5.7, 0.54, EDGE, CARD, radius=0.1, stroke=1.5).move_to([qp.get_x(), y, 0])
            t = labelled_chip(tag, color, fill, size=22, height=0.44).move_to(box).align_to(box, m.LEFT).shift(0.06 * m.RIGHT)
            b = text(body, 20, INK).next_to(t, m.RIGHT, buff=0.22)
            sit(b, box.get_y() - 0.08)
            return m.VGroup(box, t, b)

        evs = [ev_row("bg_0001", f"build finished: {BUILD_OUT}", MINT, MINT_FILL, 1.62),
               ev_row("cron", f"{SCHEDULED_MSG}", AZURE, AZURE_FILL, 1.0),
               ev_row("user", "stop, wrong branch!", AZURE, AZURE_FILL, 0.38)]
        assert all(e[2].get_right()[0] < e[0].get_right()[0] - 0.1 for e in evs)
        froms = [m.LEFT, m.UP, m.RIGHT]
        event_src = ('{"source": {"type": "user", "sender": "you"},\n "channel": "chat",\n'
                     ' "content": {"text": "stop, wrong branch!"},\n "context": {"priority": "high"}}')
        ev_json = wire(event_src, AZURE, 20, title="an event")
        ev_json.align_to([-6.6, 0, 0], m.LEFT).align_to([0, -0.85, 0], m.UP)
        ej = ev_json[1]
        parts = [glyphs(ej, '"source": {"type": "user", "sender": "you"}'), glyphs(ej, '"channel": "chat"'),
                 glyphs(ej, '"content": {"text": "stop, wrong branch!"}'), glyphs(ej, '"context": {"priority": "high"}')]
        ring = loop_ring(["wait", "take batch", "append", "call model"], radius=1.25, colors=[AZURE, SUN, MINT, CORAL], size=22, center=(3.75, 1.0, 0))
        nodes, arcs = ring
        flash = lambda i: m.ShowPassingFlash(arcs[i].copy().set_stroke(AZURE, 7), time_width=0.8, run_time=0.8)  # noqa: E731
        batch_lines = [f"[Unprocessed Event {k + 1}/{len(evs)}] {s}" for k, s in enumerate(("build finished", "nightly check", "stop, wrong branch!"))]
        appended = m.VGroup(*[code(s, 20, INK, t2c={r"^\[Unprocessed Event \d/\d\]": SUN}) for s in batch_lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        a_lbl = code("user", 22, AZURE, weight=m.BOLD)
        a_inner = m.VGroup(a_lbl, appended).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14)
        a_box = card(a_inner.width + 0.6, a_inner.height + 0.4, EDGE, CARD, radius=0.12, stroke=1.5)
        a_stripe = m.RoundedRectangle(corner_radius=0.04, width=0.08, height=a_box.height - 0.16, stroke_width=0, fill_color=AZURE, fill_opacity=1)
        a_stripe.move_to(a_box).align_to(a_box, m.LEFT).shift(0.1 * m.RIGHT)
        a_inner.move_to(a_box).align_to(a_box, m.LEFT).shift(0.3 * m.RIGHT)
        batch_msg = m.VGroup(a_box, a_stripe, a_inner).align_to([6.6, 0, 0], m.RIGHT).align_to([0, -0.85, 0], m.UP)
        markers = [glyphs(line, f"[Unprocessed Event {k + 1}/{len(evs)}]") for k, line in enumerate(appended)]
        assert batch_msg.get_left()[0] > -6.6 and batch_msg.get_bottom()[1] > -3.55 and ev_json.get_right()[0] < 6.6
        assert ev_json.get_bottom()[1] > -3.55 and nodes.get_bottom()[1] > batch_msg.get_top()[1] + 0.1
        assert qp.get_right()[0] < nodes.get_left()[0] - 0.3

        self.say(
            "[Now step back.] [AI Agents in Depth contrasts a single checkout counter], [serving one customer at a time], [with a flexible secretary, who handles things by urgency.]",
            self.new_header("05 · events", "One queue for the world"),
            m.AnimationGroup(m.FadeIn(panels[0]), m.FadeIn(titles[0]), m.FadeIn(desk), m.FadeIn(desk_lbl), m.FadeIn(people, lag_ratio=0.2), m.FadeIn(src)),
            m.Succession(m.FadeOut(people[-1], shift=0.4 * m.RIGHT, run_time=0.6), m.AnimationGroup(*[p.animate.shift(step * m.RIGHT) for p in people[:-1]], run_time=0.8)),
            m.AnimationGroup(m.FadeIn(panels[1]), m.FadeIn(titles[1]), m.FadeIn(sec, shift=0.15 * m.UP),
                             m.Succession(m.FadeIn(reqs, lag_ratio=0.2, run_time=0.9), m.Wait(0.2),
                                          m.AnimationGroup(reqs[2].animate.move_to(slots[0]), reqs[0].animate.move_to(slots[1]), reqs[1].animate.move_to(slots[2]), run_time=0.9))),
        )
        served = people[-1]  # gone: drop it from its group so a later fade of the group cannot bring it back
        people.remove(served)
        self.remove(served)
        self.say(
            "[For an agent, everything from outside is an event]: [the finished build], [the nightly job], [your message.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(panels, titles, desk, desk_lbl, people, sec, reqs, src)), m.FadeIn(qp), m.FadeIn(q_title)),
            *[m.FadeIn(e, shift=0.6 * d) for e, d in zip(evs, froms)],
        )
        self.say(
            "[Each event is structured]: [where it came from], [the channel], [the content], [and context, like priority.]",
            m.AnimationGroup(m.Circumscribe(evs[2], color=AZURE, buff=0.05), m.FadeIn(ev_json, target_position=evs[2])),
            *[m.Circumscribe(p, color=AZURE, buff=0.05) for p in parts],
        )
        self.say(
            "[The loop waits], [takes a batch of events], [appends them], [calls the model once], [and waits again.]",
            m.FadeIn(ring, shift=0.15 * m.UP),
            m.AnimationGroup(flash(0), m.Circumscribe(m.VGroup(*evs), color=SUN, buff=0.06)),
            m.AnimationGroup(flash(1), m.FadeOut(ev_json, shift=0.2 * m.DOWN), m.FadeIn(batch_msg, target_position=nodes[2])),
            m.AnimationGroup(flash(2), m.Circumscribe(nodes[3], color=CORAL, buff=0.08)),
            m.AnimationGroup(flash(3), m.Circumscribe(nodes[0], color=AZURE, buff=0.08)),
        )
        self.say(
            "[Each event is numbered], [so the model doesn't answer only the last one.]",
            m.LaggedStart(*[m.Circumscribe(mk, color=SUN, buff=0.04) for mk in markers], lag_ratio=0.3),
            m.AnimationGroup(m.Circumscribe(appended[0], color=MINT, buff=0.05), m.Circumscribe(appended[1], color=MINT, buff=0.05)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 6 push vs poll

    def push_poll(self):
        self.next_section("6 · Push vs poll", notes=f"Li, AI Agents in Depth, ch. 6 (PineClaw): mid-call the representative asks for a one-time code that the user must give at once. OpenClaw's Heartbeat wakes the agent every N minutes; with polling 'the representative hangs up and the call fails'; the fix is a push Channel. Times for example: a heartbeat every {HEARTBEAT_S // 60} min, the code needed at {mmss(NEED_S)}, the representative waits {PATIENCE_S} s (hangs up at {mmss(HANG_UP)}); polling notices at {mmss(POLL_SEEN)}, {mmss(POLL_LATE)} late; pushed, the user answers in {REPLY_S} s ({mmss(PUSH_CODE)}). Computed and asserted.")
        X0, X1, T = -4.3, 6.3, 360
        sx = lambda t: X0 + (X1 - X0) * t / T  # noqa: E731
        Y_POLL, Y_PUSH = 0.95, -1.35
        cast = m.VGroup(icon(PHONE, 0.6), text("a phone agent, mid-call · the book's example", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        cast.move_to([0, 0, 0]).align_to([6.6, 0, 0], m.RIGHT).set_y(2.3)

        def lane_label(y, name, sub, color):
            lbl = m.VGroup(title_text(name, 30, color), text(sub, 20, MUTED)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.1)
            return lbl.move_to([-6.55, y, 0], aligned_edge=m.LEFT)

        window = m.RoundedRectangle(corner_radius=0.08, width=sx(HANG_UP) - sx(NEED_S), height=0.5, stroke_width=0, fill_color=ROSE, fill_opacity=0.22)
        window.move_to([(sx(NEED_S) + sx(HANG_UP)) / 2, Y_POLL, 0])
        waits = sit(text(f"waits {PATIENCE_S} s", 20, INK).move_to(window), Y_POLL - 0.08)
        assert waits.width < window.width - 0.45
        poll = m.VGroup(m.VGroup(m.Line([sx(0), Y_POLL, 0], [sx(NEED_S), Y_POLL, 0]), m.Line([sx(HANG_UP), Y_POLL, 0], [sx(T), Y_POLL, 0])).set_stroke(FAINT, 3),
                        lane_label(Y_POLL, "poll", f"heartbeat\nevery {HEARTBEAT_S // 60} min", AZURE))
        push = m.VGroup(m.Line([sx(0), Y_PUSH, 0], [sx(T), Y_PUSH, 0]).set_stroke(FAINT, 3), lane_label(Y_PUSH, "push", "a channel", MINT))
        beats = m.VGroup()
        for t in range(0, T + 1, HEARTBEAT_S):
            beats.add(m.VGroup(m.Line([sx(t), Y_POLL - 0.22, 0], [sx(t), Y_POLL + 0.22, 0]).set_stroke(AZURE, 4),
                               code("tick", 20, AZURE).next_to([sx(t), Y_POLL + 0.22, 0], m.UP, buff=0.08)))
        assert len(beats) == 2 and sx(POLL_SEEN) == beats[1][0].get_x()

        def need(y):
            dot = m.Dot([sx(NEED_S), y, 0], radius=0.1, color=SUN)
            lbl = text("code needed", 20, SUN).next_to(dot, m.UP, buff=0.2).align_to([sx(NEED_S) + 0.15, 0, 0], m.RIGHT)
            return m.VGroup(dot, lbl)

        need_poll, need_push = need(Y_POLL), need(Y_PUSH)
        hang_dot = m.Dot([sx(HANG_UP), Y_POLL, 0], radius=0.1, color=ROSE)
        hang = m.VGroup(cross_mark(ROSE, 28), text("hangs up", 22, ROSE)).arrange(m.RIGHT, buff=0.12)
        hang.next_to(hang_dot, m.UP, buff=0.32).align_to([sx(HANG_UP) - 0.12, 0, 0], m.LEFT)
        late = m.ArcBetweenPoints([sx(NEED_S) + 0.05, Y_POLL - 0.3, 0], [sx(POLL_SEEN) - 0.08, Y_POLL - 0.26, 0], angle=0.5)
        late.set_stroke(MUTED, 3).add_tip(tip_length=0.16, tip_width=0.16)
        seen = text(f"noticed {mmss(POLL_LATE)} late", 22, AZURE).next_to(beats[1][1], m.UP, buff=0.12)
        asked = m.Dot([sx(NEED_S), Y_PUSH, 0], radius=0.1, color=MINT)
        reply = m.Dot([sx(PUSH_CODE), Y_PUSH, 0], radius=0.1, color=MINT)
        hop = m.ArcBetweenPoints([sx(NEED_S) + 0.1, Y_PUSH - 0.06, 0], [sx(PUSH_CODE) - 0.1, Y_PUSH - 0.06, 0], angle=0.9).set_stroke(MINT, 3)
        got = m.VGroup(check_mark(MINT, 28), text(f"code back in {REPLY_S} s", 22, MINT)).arrange(m.RIGHT, buff=0.12)
        got.next_to(asked, m.DOWN, buff=0.3).align_to([sx(NEED_S) - 0.1, 0, 0], m.LEFT)
        through = pill("the call goes through", MINT, MINT_FILL, 22).move_to([sx(250), Y_PUSH - 0.62, 0])
        axis_y = -2.55
        mins = m.VGroup(*[code(mmss(t), 20, MUTED).move_to([sx(t), axis_y, 0]) for t in range(0, T + 1, 60)])
        times = text("times for example", 20, MUTED).next_to(mins, m.DOWN, buff=0.25).align_to([X1, 0, 0], m.RIGHT)
        src = cite("Li, AI Agents in Depth · ch. 6").next_to(mins, m.DOWN, buff=0.25).align_to([X0, 0, 0], m.LEFT)
        assert through.get_bottom()[1] > mins.get_top()[1] + 0.1 and got.get_right()[0] < through.get_left()[0] - 0.2
        assert hang.get_left()[0] > need_poll[1].get_right()[0] + 0.2 and seen.get_top()[1] < cast.get_bottom()[1] - 0.05
        assert need_poll[1].get_left()[0] > beats[0][1].get_right()[0] + 0.2

        self.say(
            "[How do events get in?] [Polling checks every few minutes.] [Pushing delivers each one the moment it happens.]",
            m.AnimationGroup(self.new_header("06 · push vs poll", "Don't make the world wait"), m.FadeIn(mins), m.FadeIn(src), m.FadeIn(times)),
            m.AnimationGroup(m.FadeIn(poll[1], shift=0.1 * m.RIGHT), m.Create(poll[0][0]), m.Create(poll[0][1]), m.LaggedStart(*[m.FadeIn(b, shift=0.1 * m.DOWN) for b in beats], lag_ratio=0.4)),
            m.AnimationGroup(m.FadeIn(push[1], shift=0.1 * m.RIGHT), m.Create(push[0])),
        )
        self.say(
            "[In the book's example, a phone agent, mid-call], [needs a one-time code from its user, within seconds.]",
            m.FadeIn(cast, shift=0.1 * m.DOWN),
            m.AnimationGroup(m.FadeIn(need_poll, scale=0.6), m.FadeIn(need_push, scale=0.6), m.FadeIn(window), m.FadeIn(waits), m.FadeIn(hang_dot, scale=0.5)),
        )
        self.say(
            "[With polling], [the agent only notices at the next heartbeat], [and the representative hangs up.]",
            m.Circumscribe(poll[1], color=AZURE, buff=0.08),
            m.AnimationGroup(m.Create(late), m.Circumscribe(beats[1], color=AZURE, buff=0.06), m.FadeIn(seen, shift=0.1 * m.UP)),
            m.FadeIn(hang, scale=0.8),
        )
        self.say(
            "[With a push channel], [the request reaches the user at once], [and the call goes through.]",
            m.Circumscribe(push[1], color=MINT, buff=0.08),
            m.AnimationGroup(m.FadeIn(asked, scale=0.5), m.Create(hop), m.FadeIn(reply, scale=0.5), m.FadeIn(got, shift=0.1 * m.UP)),
            m.FadeIn(through, scale=0.85),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ 7 urgency lanes

    def lanes(self):
        self.next_section("7 · Urgency lanes", notes=f"Li, AI Agents in Depth, ch. 6: urgent events, three ways — cancellation ('Stop! I said the wrong thing.'): force 'a safe point early' by 'proactively interrupting the current step', send a running tool a cancel signal; queued ('only show results from the last month'): appended when 'any tool call completes'; parallel ('What's the weather like today?'): 'a parallel reasoning session'. 'A lightweight classification LLM as an event router.' 'Receiving a stop message does not itself undo actions that have already occurred'; an unfinished result is an explicit placeholder. Otto's rename (illustrative): {RENAME} over {len(RENAME_FILES)} files, {PER_FILE_S} s per file, stop at {STOP_AT_S} s → {len(CHANGED)} changed ({', '.join(CHANGED)}), result '{CANCEL_RESULT}' (computed).")
        cw, ch = 4.0, 3.15
        specs = [("cancel", ROSE, ROSE_FILL, "“Stop! I said the\nwrong thing.”", "interrupt now:\nan early safe point"),
                 ("queue", SUN, SUN_FILL, "“Only show results\nfrom the last month.”", "wait, and ride with\nthe next tool result"),
                 ("parallel", MINT, MINT_FILL, "“What's the weather\nlike today?”", "answer it in a\nseparate session")]
        cards = m.VGroup()
        for name, color, fill, example, effect in specs:
            box = card(cw, ch, color, CARD, radius=0.2, stroke=2.5)
            tag = pill(name, color, fill, 24, mono=True)
            ex = text(example, 22, INK)
            ef = text(effect, 22, MUTED)
            body = m.VGroup(tag, ex, ef).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.3)
            body.move_to(box).align_to(box, m.LEFT).align_to(box, m.UP).shift(0.32 * m.RIGHT + 0.3 * m.DOWN)
            cards.add(m.VGroup(box, body))
        cards.arrange(m.RIGHT, buff=0.3).move_to([0, -0.55, 0])
        router = m.VGroup(icon(MODEL_ICON, 0.55), text("a small, fast model routes each event", 22, MUTED)).arrange(m.RIGHT, buff=0.25)
        router.move_to([0, 1.75, 0])
        feeds = m.VGroup(*[m.DashedLine(router.get_bottom() + 0.06 * m.DOWN, c[0].get_top() + 0.06 * m.UP, dash_length=0.08).set_stroke(FAINT, 2)
                           for c in cards])
        src = cite("Li, AI Agents in Depth · ch. 6").next_to(cards, m.DOWN, buff=0.22).align_to(cards, m.RIGHT)

        # phase B: Otto's rename, stopped
        files = m.VGroup(*[code(f, 22, INK) for f in RENAME_FILES]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.2)
        fbox = card(5.4, files.height + 1.05, EDGE, "#18150F", radius=0.2, stroke=2)
        fcap = code("tinyshop/", 20, AZURE).move_to(fbox.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        m.VGroup(fbox).move_to([-3.75, -0.85, 0])
        fcap.move_to(fbox.get_corner(m.UL) + [0.3, -0.3, 0], aligned_edge=m.LEFT)
        files.next_to(fcap, m.DOWN, buff=0.25).align_to(fcap, m.LEFT)
        marks = []
        for k, f in enumerate(files):
            if RENAME_FILES[k] in CHANGED:
                mk = code("changed", 20, SUN)
            elif k == len(CHANGED):
                mk = code("stopped", 20, ROSE)
            else:
                mk = code("untouched", 20, FAINT)
            mk.align_to(fbox, m.RIGHT).shift(0.3 * m.LEFT)
            sit(mk, base_of(f))
            marks.append(mk)
        cmd = labelled_chip(RENAME, SUN, SUN_FILL, height=0.6).next_to(fbox, m.UP, buff=0.3).align_to(fbox, m.LEFT)
        stop = bubble(text("stop, wrong branch!", 26, INK), AZURE, AZURE_FILL, tail="right")
        person = icon("user", 0.75).move_to([5.95, 1.55, 0])
        stop.move_to([3.5, 2.0, 0])
        stop.shift((person.get_left()[0] - 0.15 - stop[0].get_vertices()[2][0]) * m.RIGHT)
        lane_pill = pill("cancel", ROSE, ROSE_FILL, 24, mono=True).move_to([3.4, 0.55, 0])
        to_lane = arrow([lane_pill.get_x(), stop[1].get_bottom()[1], 0], lane_pill.get_top(), ROSE, buff=0.1)
        signal = arrow(lane_pill.get_left(), cmd.get_right(), ROSE, buff=0.12)
        result = tool_result(CANCEL_RESULT, id="toolu_09")
        result.align_to([6.6, 0, 0], m.RIGHT).align_to([0, -0.3, 0], m.UP)
        not_undo = pill("stop is not undo", ROSE, ROSE_FILL, 24).next_to(fbox, m.DOWN, buff=0.3).align_to(fbox, m.LEFT)
        assert not_undo.get_bottom()[1] > -3.55 and fbox.get_bottom()[1] > -3.55 and result.get_bottom()[1] > -3.55
        assert result.get_left()[0] > fbox.get_right()[0] + 0.3 and lane_pill.get_bottom()[1] > result.get_top()[1] + 0.2

        self.say(
            "[Not every event is equally urgent.] [The book sorts them into three lanes], [and a small, fast model can pick the lane.]",
            self.new_header("07 · urgency lanes", "Not every event can wait"),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c[0]) for c in cards], lag_ratio=0.2), m.LaggedStart(*[m.FadeIn(c[1][0], scale=0.85) for c in cards], lag_ratio=0.2), m.FadeIn(src)),
            m.AnimationGroup(m.FadeIn(router, shift=0.1 * m.DOWN), m.Create(feeds)),
        )
        for k, line in enumerate((
            "[Cancel]: [stop, I said the wrong thing.] [The harness interrupts the current step], [forcing an early safe point.]",
            "[The queue lane]: [a routine update waits], [and rides along with the next tool result.]",
            "[Parallel]: [an unrelated question, like today's weather], [gets its own session.]",
        )):
            c = cards[k]
            ef_lines = c[1][2]
            anims = [m.Circumscribe(c[0], color=specs[k][1], buff=0.06), m.FadeIn(c[1][1], shift=0.1 * m.UP)]
            if k == 0:
                anims += [m.FadeIn(ef_lines[:len("interruptnow:")], shift=0.1 * m.UP), m.FadeIn(ef_lines[len("interruptnow:"):], shift=0.1 * m.UP)]
            else:
                anims += [m.FadeIn(ef_lines, shift=0.1 * m.UP)]
            self.say(line, *anims)
        self.say(
            "[Back to Otto. It's renaming a function across five files], [when you type: stop, wrong branch.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(cards, router, feeds, src)), m.FadeIn(fbox), m.FadeIn(fcap), m.FadeIn(files, lag_ratio=0.15), m.FadeIn(cmd, shift=0.1 * m.UP),
                             m.Succession(m.Wait(0.6), m.FadeIn(marks[0]), m.FadeIn(marks[1]))),
            m.AnimationGroup(m.FadeIn(person, shift=0.15 * m.UP), m.FadeIn(stop, shift=0.15 * m.UP)),
        )
        self.say(
            "[Stop takes the cancel lane.] [The running command gets a cancel signal], [and its result says so]: [cancelled, after two of five files.]",
            m.AnimationGroup(m.GrowArrow(to_lane), m.FadeIn(lane_pill, scale=0.85)),
            m.AnimationGroup(m.GrowArrow(signal), m.FadeIn(marks[2]), m.FadeIn(m.VGroup(*marks[3:]))),
            m.FadeIn(result, shift=0.15 * m.UP),
            m.Circumscribe(glyphs(result[1], CANCEL_RESULT), color=MINT, buff=0.05),
        )
        self.say(
            "[But stop is not undo.] [Those two files stay changed], [and the model has to be told exactly that.]",
            m.FadeIn(not_undo, scale=0.85),
            m.AnimationGroup(*[m.Circumscribe(m.VGroup(files[k], marks[k]), color=SUN, buff=0.06) for k in range(len(CHANGED))]),
            m.Circumscribe(result, color=MINT, buff=0.08),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Start slow work in the background and keep going (a placeholder now, a notification later); let schedules and the outside world notify the agent as events, pushed into one loop; prioritize them in lanes — cancel, queue, parallel. Coming up: Episode 24, Agent Teams. Start · Notify · Prioritize.")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Start", SUN), ("Notify", MINT), ("Prioritize", AZURE))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("bg_0001 started", "<task_notification>", "cancel · queue · parallel")])
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
        assert row_.width < 13.0
        tagline = text("Never wait on slow work, and never fake its result.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [start], [notify], [prioritize.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[Never wait on slow work], [and never fake its result.]", m.Write(tagline), m.Circumscribe(row_, color=AZURE, buff=0.2))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
