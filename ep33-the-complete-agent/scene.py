"""Agents, Animated · Episode 33 · The Complete Agent (Module 6 · Beyond the Loop) — the finale.

How every piece fits into one harness. Running example: Otto, complete, fixing the course's first bug
again (tinyshop's cart.total() skips the last item; pytest -q says "1 failed, 12 passed").

The harness is not drawn by hand. The scene runs Episode 2's ten-line agent_loop — the very same text,
read from ep02-the-agent-loop/scene.py and asserted identical — with every later layer attached only
through the two calls the loop already makes (client.messages.create and tool_results) or around the
turn (run_turn): UserPromptSubmit / PreToolUse / PostToolUse / Stop hooks (Episode 18), background
notifications (23), compaction (11), system prompt assembly with the skills catalog and memories (8, 10,
12), retry with backoff (20), the permission gate (17), TOOL_HANDLERS (3), MCP servers (16) and
background dispatch (23). It runs against a scripted stand-in for the model (whose first request is
turned away with a 529) and an in-memory tinyshop; every number on screen comes from that run (asserted).

Arc: hook (Episode 1's bug; fifteen mechanisms as separate tiles; "a real Agent does not run as 19
separate demos") · 1 the core (Episode 2's loop, unchanged; two calls) · 2 one full turn (fourteen layers
plugged in, in learn-claude-code s20's order, each labelled with its episode) · 3 replay (the first bug
again, each layer lighting as it is used; two stay dark) · 4 teams are tools (s20's 27 built-in tools) ·
5 a mature harness (s20's line) · 6 the book's five patterns · recap (loop, layers, patterns) · a
course-closing card instead of the next-episode card.

Sources (facts dated October 2026):
- Learn Claude Code s19, "What's Next" (learn.shareai.run/en/s19/): "a real Agent does not run as 19
  separate demos." s20 (learn.shareai.run/en/s20/): the one-turn flow (user input → UserPromptSubmit
  hooks → cron/background notification injection → context compact → memory + skills + MCP state
  assemble the system prompt → LLM → has tool_use block? no → Stop hooks → return; yes → PreToolUse
  hooks + permission → TOOL_HANDLERS / MCP handlers / background dispatch → PostToolUse hooks →
  tool_result / task_notification back to messages → next round); "Where Each Component Sits" (error
  recovery at the LLM call: retry 429/529, escalate max_tokens, compact on prompt-too-long);
  "Claude Code's complexity is not "another agent brain." It is the complexity of a mature harness."
  and "The model decides and chooses actions; the harness organizes environment, tools, permissions,
  memory, teams, and external capabilities."; "The built-in tool pool contains 27 tools" in eight
  groups (5 files/shell, 4 planning/context, 5 task graph, 3 cron, 3 teams, 3 protocols, 3 worktrees,
  1 MCP; the page's header says 32 tools, we quote the 27 built-in ones).
- Li, AI Agents in Depth, ch. 1, "Design patterns that run through the book": Proposer–Reviewer ("the
  creator and reviewer work in separate contexts"), Progressive Disclosure ("offer a searchable
  catalogue first and load the details on demand"), Append-only ("what has been written is never
  revised in place"), Boundary Set + Retention Set (validate on "the samples it is supposed to change"
  and "the samples it must not affect"), Minimal Diff, Reversible ("keep each change as small as
  possible, carrying its provenance").
- Earlier episodes for the layer details: the compaction threshold 200,000 − 8,000 − 13,000 = 179,000
  (Episode 11, learn-claude-code s08's reading of Claude Code); retry_delay (Episode 20, s11); hook
  names and the tests-after-every-edit hook (Episode 18, s04); five model calls in Episode 1's run.

Illustrative (for teaching): tinyshop, its bug and test counts (harness_reference.py), the model's
choices, the 529 on the first request and its seeded jitter, the permission rules, the skills and
memory files (from Episodes 10 and 12), the tracker's MCP tools (Episode 16).

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import fnmatch
import inspect
import json
import random
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

PREFIX = "ep33"

# ---------------------------------------------------------------------------- the loop, from Episode 2

LOOP_SRC = '''while True:
    response = client.messages.create(
        messages=messages, tools=TOOLS)
    messages.append({"role": "assistant",
        "content": response.content})
    if response.stop_reason != "tool_use":
        return response
    results = tool_results(response)
    messages.append({"role": "user",
        "content": results})'''
_EP02 = (SERIES / "ep02-the-agent-loop" / "scene.py").read_text()
EP02_LOOP = re.search(r"LOOP_SRC = '''(.*?)'''", _EP02, re.S).group(1)
CHANGED = sum(a != b for a, b in zip(LOOP_SRC.splitlines(), EP02_LOOP.splitlines()))
CHANGED += abs(len(LOOP_SRC.splitlines()) - len(EP02_LOOP.splitlines()))
assert LOOP_SRC == EP02_LOOP and CHANGED == 0  # "not one character has changed"
assert len(LOOP_SRC.splitlines()) == 10  # "the same ten lines"
EPISODES_AGO = 33 - 1
assert EPISODES_AGO == 32  # "thirty-two episodes ago"

# ---------------------------------------------------------------------------- tinyshop (as in Episodes 1 and 2)

TASK = "The checkout test fails. Fix it."
CART = "def total(items):\n    s = 0\n    for i in range(len(items) - 1):\n        s += items[i]\n    return s\n"
BUG_LINE = "for i in range(len(items) - 1):"
OLD, NEW = "range(len(items) - 1)", "range(len(items))"
OTHER_TESTS = 12  # tinyshop's other tests, all passing (illustrative, as in Episode 1)
EP1_MODEL_CALLS = 5  # Episode 1's run: pytest, read, edit, pytest, answer (asserted there)
FILES = {"cart.py": CART}


def pytest_q():
    """tinyshop's suite: test_total (total([3, 4]) == 7) plus twelve that pass."""
    space = {}
    exec(FILES["cart.py"], space)
    ok = space["total"]([3, 4]) == 7
    failed, passed = int(not ok), OTHER_TESTS + ok
    return f"{failed} failed, {passed} passed" if failed else f"{passed} passed"


def run_bash(command):
    if command == "pytest -q":
        return pytest_q()
    raise ValueError(command)


def read_file(path):
    return FILES[path]


def edit_file(path, old_text, new_text):
    FILES[path] = FILES[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}  # Episode 3

# ---------------------------------------------------------------------------- the layers (each from its episode)

TRACE = []  # (layer, what it did), in the order the harness ran them


def log(layer, detail):
    TRACE.append((layer, detail))


# hooks (Episode 18, learn-claude-code s04)
HOOKS = {"UserPromptSubmit": [], "PreToolUse": [], "PostToolUse": [], "Stop": []}


def register_hook(event, callback):
    HOOKS[event].append(callback)


def trigger_hooks(event, *args):
    for callback in HOOKS[event]:
        result = callback(*args)
        if result is not None:
            return result
    return None


def log_request(text):
    log("ups", "logged the request")


def protect_tests(block):
    if block.input.get("path", "").startswith("tests/"):
        return "tests/ is read-only"
    log("pre", "nothing to block")


def tests_after_edit(block, output):
    if block.name == "edit_file":
        out = run_bash("pytest -q")
        log("post", f"ran the tests: {out}")
        return out
    log("post", "nothing to add")


def tests_must_pass(messages):
    out = run_bash("pytest -q")
    if "failed" in out:
        log("stop", f"tests fail: {out}")
        return "Tests fail: " + out
    log("stop", "tests green, may stop")


register_hook("UserPromptSubmit", log_request)
register_hook("PreToolUse", protect_tests)
register_hook("PostToolUse", tests_after_edit)
register_hook("Stop", tests_must_pass)

# the permission gate (Episode 17): deny, then ask, then allow
RULES = {"deny": ["rm -rf *", "sudo *"], "ask": ["git push*"], "allow": ["pytest*", "read_file *", "edit_file *"]}


def target_of(block):
    return block.input["command"] if block.name == "bash" else f"{block.name} {block.input['path']}"


def check_permission(block):
    target = target_of(block)
    for decision in ("deny", "ask", "allow"):
        if any(fnmatch.fnmatch(target, rule) for rule in RULES[decision]):
            return decision
    return "ask"


# background work (Episode 23) and MCP servers (Episode 16)
SLOW = ("make build", "npm install")
background_tasks = {}
MCP_TOOLS = ["mcp__tracker__get_issue", "mcp__tracker__search", "mcp__tracker__comment"]


def should_run_background(block):
    return block.name == "bash" and block.input["command"].startswith(SLOW)


def start_background_task(block):
    bg_id = f"bg_{len(background_tasks) + 1:04d}"
    background_tasks[bg_id] = {"command": block.input["command"], "status": "running"}
    return f"[Background task {bg_id} started]"


def collect_background_results():
    done = [k for k, t in background_tasks.items() if t["status"] == "completed"]
    return [f"<task_notification>{k}</task_notification>" for k in done]


def call_mcp(block):
    return f"[tracker] {block.name}"


def run_tool(block):
    if block.name.startswith("mcp__"):
        log("mcp", block.name)
        return call_mcp(block)
    if should_run_background(block):
        log("bg", block.input["command"])
        return start_background_task(block)
    output = TOOL_HANDLERS[block.name](**block.input)
    log("handlers", f"{block.name}: {output.strip().splitlines()[0] if block.name != 'read_file' else BUG_LINE}")
    return output


def tool_results(response):
    """The loop's call to the tools: hooks, the gate, dispatch, hooks again (Episodes 18, 17, 3, 16, 23)."""
    results = []
    for block in response.content:
        if block.type == "tool_use":
            output = trigger_hooks("PreToolUse", block)
            if output is None:
                decision = check_permission(block)
                log("perm", f"{decision}: {target_of(block)}")
                if decision == "deny":
                    output = "Permission denied."
            if output is None:
                output = run_tool(block)
                note = trigger_hooks("PostToolUse", block, output)
                if note:
                    output += "\n" + note
            results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
    return results


# the system prompt, assembled each round (Episode 8), with the skills catalog (10) and memories (12)
SYSTEM = "You are Otto, a coding agent working in the tinyshop repo."  # harness_reference.SYSTEM
SKILLS = {"release": "Cut a tinyshop release.", "sql-style": "tinyshop's SQL conventions."}
MEMORY = {"tabs.md": "Use tabs, not spaces", "tests.md": "Test with pytest -q"}
TOOLS = [{"name": n} for n in (*TOOL_HANDLERS, *MCP_TOOLS)]  # the tool pool, built-in + MCP (Episode 16)


def select_relevant_memories(messages, max_items=5):
    words = set(re.findall(r"[a-z]+", messages[0]["content"].lower()))
    return [f for f, d in MEMORY.items() if words & set(re.findall(r"[a-z]+", d.lower()))][:max_items]


def assemble_system_prompt(messages):
    chosen = select_relevant_memories(messages)
    parts = [
        SYSTEM,
        "Tools: " + ", ".join(t["name"] for t in TOOLS),
        "Workspace: tinyshop (macOS), git status: clean",
        "Skills:\n" + "\n".join(f"- {k}: {v}" for k, v in SKILLS.items()),
        "Memory:\n" + "\n".join(f"- {f}: {MEMORY[f]}" for f in chosen),
    ]
    system = "\n\n".join(parts)
    log("prompt", f"{len(parts)} parts, {len(system.encode())} bytes")
    log("skills", f"catalog of {len(SKILLS)}, none loaded")
    log("memory", " · ".join(f'{f}: "{MEMORY[f]}"' for f in chosen))
    return system


# compaction (Episode 11): the threshold from learn-claude-code s08's reading of Claude Code
WINDOW, MAX_OUTPUT, BUFFER = 200_000, 8_000, 13_000
THRESHOLD = WINDOW - MAX_OUTPUT - BUFFER
assert THRESHOLD == 179_000


def estimate_token_count(messages):
    return len(json.dumps(messages, default=str)) // 4


def compact_if_needed(messages):
    tokens = estimate_token_count(messages)
    if tokens > THRESHOLD:
        raise RuntimeError("not reached in this run")
    log("compact", f"{tokens:,} of {THRESHOLD:,} tokens, nothing to do")
    return tokens


# recovery (Episode 20, learn-claude-code s11)
class Overloaded(Exception):
    status = 529


def retry_delay(attempt, retry_after=None):
    if retry_after:
        return retry_after
    base = min(500 * 2**attempt, 32000) / 1000
    return base + random.uniform(0, base*0.25)


WAITED = []


def with_retry(call, max_retries=10):
    for attempt in range(max_retries):
        try:
            return call()
        except Overloaded:
            delay = retry_delay(attempt)
            WAITED.append(delay)
            log("recovery", f"529 overloaded, retry in {delay:.2f} s")
    raise RuntimeError("gave up")


# the model: a scripted stand-in (Otto's choices are illustrative)
PLAN = [
    ("bash", {"command": "pytest -q"}),
    ("read_file", {"path": "cart.py"}),
    ("edit_file", {"path": "cart.py", "old_text": OLD, "new_text": NEW}),
    "Fixed: total() skipped the last item.",
]


def call_str(name, args):
    return f'{name}("{args["command"] if name == "bash" else args["path"]}")'


class ScriptedModel:
    """Stands in for the API. The first request is turned away (overloaded), as in Episode 20."""

    def __init__(self):
        self.requests = self.replies = 0

    def __call__(self, system, messages, tools):
        self.requests += 1
        if self.requests == 1:
            raise Overloaded()
        step = PLAN[self.replies]
        self.replies += 1
        if isinstance(step, str):
            log("model", "answers in words")
            return NS(stop_reason="end_turn", content=[NS(type="text", text=step)])
        name, args = step
        log("model", f"tool_use: {call_str(name, args)}")
        return NS(stop_reason="tool_use", content=[NS(type="tool_use", id=f"toolu_{self.replies:02d}", name=name, input=args)])


llm = ScriptedModel()


class Messages:
    """The loop's call to the model: notifications, compaction, the prompt, recovery (23, 11, 8, 20)."""

    def create(self, messages, tools):
        notes = collect_background_results()
        log("notes", f"{len(notes)} waiting")
        if notes:
            messages.append({"role": "user", "content": notes})
        compact_if_needed(messages)
        system = assemble_system_prompt(messages)
        return with_retry(lambda: llm(system, messages, tools))


client = NS(messages=Messages())


def agent_loop(messages):
    while True:
        response = client.messages.create(
            messages=messages, tools=TOOLS)
        messages.append({"role": "assistant",
            "content": response.content})
        if response.stop_reason != "tool_use":
            return response
        results = tool_results(response)
        messages.append({"role": "user",
            "content": results})


def run_turn(messages, user_input):
    """Around the turn: UserPromptSubmit before, Stop hooks after (Episode 18)."""
    trigger_hooks("UserPromptSubmit", user_input)
    messages.append({"role": "user", "content": user_input})
    while True:
        response = agent_loop(messages)
        objection = trigger_hooks("Stop", messages)
        if objection is None:
            return response
        messages.append({"role": "user", "content": objection})


random.seed(33)
MESSAGES = []
ANSWER = run_turn(MESSAGES, TASK)

# the code that ran is the code on screen
_ran = inspect.getsource(agent_loop)
assert all(line.strip() in _ran for line in LOOP_SRC.split("\n"))

# the run, in numbers
LAYER_KEYS = ["ups", "notes", "compact", "prompt", "skills", "memory", "recovery",
              "pre", "perm", "handlers", "mcp", "bg", "post", "stop"]
USED = [k for k in LAYER_KEYS if any(layer == k for layer, _ in TRACE)]
DARK = [k for k in LAYER_KEYS if k not in USED]
assert len(LAYER_KEYS) == 14 and len(USED) == 12 and DARK == ["mcp", "bg"]  # "fourteen layers"; "two stayed dark"
assert llm.replies == 4 and llm.requests == 5 and len(WAITED) == 1  # "four replies", one retry
assert llm.replies == EP1_MODEL_CALLS - 1  # "one fewer than in episode one"
assert 0.5 <= WAITED[0] <= 0.625  # "about half a second" (seeded jitter)
assert ANSWER.content[0].text == PLAN[-1] and pytest_q() == "13 passed"


def events(layer):
    return [d for k, d in TRACE if k == layer]


ROUNDS = len(events("notes"))
assert ROUNDS == 4 and len(MESSAGES) == 8  # four rounds; the user's request plus seven appended messages
assert events("handlers")[0] == "bash: 1 failed, 12 passed"  # "one failed, twelve passed"
assert events("post") == ["nothing to add", "nothing to add", "ran the tests: 13 passed"]  # "thirteen passed"
assert events("perm") == ["allow: pytest -q", "allow: read_file cart.py", "allow: edit_file cart.py"]
assert events("stop") == ["tests green, may stop"] and events("memory")[0] == 'tests.md: "Test with pytest -q"'
TOKENS = [int(d.split()[0].replace(",", "")) for d in events("compact")]
assert max(TOKENS) < THRESHOLD / 100  # "far below the threshold" (under one percent of it)
DIFF = [(a, b) for a, b in zip(CART.splitlines(), FILES["cart.py"].splitlines()) if a != b]
assert len(DIFF) == 1  # "one line"

# Learn Claude Code s20: the 27 built-in tools, by group (the episode that taught each)
TOOL_GROUPS = [
    ("files and shell", [3], ["bash", "read_file", "write_file", "edit_file", "glob"]),
    ("plans and context", [9, 21, 10, 11], ["todo_write", "task", "load_skill", "compact"]),
    ("task graph", [22], ["create_task", "list_tasks", "get_task", "claim_task", "complete_task"]),
    ("cron", [23], ["schedule_cron", "list_crons", "cancel_cron"]),
    ("teams", [24], ["spawn_teammate", "send_message", "check_inbox"]),
    ("protocols", [24], ["request_shutdown", "request_plan", "review_plan"]),
    ("worktrees", [25], ["create_worktree", "remove_worktree", "keep_worktree"]),
    ("MCP", [16], ["connect_mcp"]),
]
N_TOOLS = sum(len(t) for _, _, t in TOOL_GROUPS)
assert N_TOOLS == 27 and [len(t) for _, _, t in TOOL_GROUPS] == [5, 4, 5, 3, 3, 3, 3, 1]

# ---------------------------------------------------------------------------- look

CODE_T2C = {
    r"#[^\n]*": FAINT,
    r"\"[^\"\n]*\"|'[^'\n]*'": MINT,
    r"\b(def|while|for|in|if|return|True|False|None|not|and|or)\b": ORCHID,
    r"\b\d+(\.\d+)?\b": SUN,
    r"\b[a-z_][a-z0-9_]*(?=\()": AZURE,
}
FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
CARD_TITLE = "agent_loop(messages) · Episode 2"

WRENCH = "ep03-wrench"
BOOK = recolored("book", SUN, SUN_FILL, prefix=PREFIX)
GIT = recolored("ep27-git", AZURE, AZURE_FILL, prefix=PREFIX)
LLM_ICON = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
GEAR = recolored("gear", LIME, LIME_FILL, prefix=PREFIX)


def accent(ep):
    _, color, fill = MODULES[EPISODES[ep][1]]
    return color, fill


def cite(s):
    return text(s, 20, MUTED)


def top_cite(s):
    return cite(s).to_corner(m.UR, buff=0.5).shift(0.08 * m.DOWN)


def baseline(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def outline_flash(box, color, run_time=0.8):
    return m.ShowPassingFlash(box.copy().set_fill(opacity=0).set_stroke(color, 7), time_width=0.6, run_time=run_time)


def ep_badge(ep, height=0.36, size=18):
    color, fill = accent(ep)
    num = code(str(ep), size, color, weight=m.BOLD)
    box = card(max(0.46, num.width + 0.22), height, color, fill, radius=0.08, stroke=1.5)
    return m.VGroup(box, num.move_to(box))


class LoopCard:
    """Episode 2's loop in a code card, with a line cursor and the loop-back arc in its gutter."""

    def __init__(self, size=20):
        k = size / 20
        self.size = size
        self.body = code(LOOP_SRC, size, INK, t2c=CODE_T2C)
        self.cap = code(CARD_TITLE, size, MUTED)
        pad, gutter = 0.35 * k, 0.25 * k
        top = self.cap.height + 0.25 * k
        self.box = card(self.body.width + pad + gutter + 0.28 * k, self.body.height + 0.6 * k + top, EDGE, CARD, radius=0.16, stroke=2)
        self.body.move_to(self.box).align_to(self.box, m.LEFT).shift((pad + gutter) * m.RIGHT + top / 2 * m.DOWN)
        self.cap.align_to(self.box, m.UL).shift((pad + gutter) * m.RIGHT + 0.2 * k * m.DOWN)
        self.body.set_z_index(2)
        self.cap.set_z_index(2)
        self.src = LOOP_SRC.split("\n")
        self.lines, n0 = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[n0:n0 + n])
            n0 += n
        gx = self.box.get_left()[0] + 0.36 * k
        b0, b9 = (baseline(self.lines[i], self.src[i].replace(" ", "")) for i in (0, 9))
        self.arc = m.ArcBetweenPoints([gx, b9 + 0.08 * k, 0], [gx, b0 + 0.08 * k, 0], angle=-0.6)
        self.arc.set_stroke(FAINT, 3)
        self.arc.add_tip(tip_length=0.14 * k, tip_width=0.14 * k)
        self.arc.set_z_index(1)
        self.block = m.VGroup(self.box, self.body, self.cap)
        self.group = m.VGroup(self.block, self.arc)

    def base(self, i):
        return baseline(self.lines[i], self.src[i].replace(" ", ""))

    def line_y(self, i):
        return self.base(i) + 0.075 * self.size / 20

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=CORAL):
        b = a if b is None else b
        k = self.size / 20
        top, bot = self.base(a) + 0.25 * k, self.base(b) - 0.1 * k
        left, right = self.box.get_left()[0] + 0.5 * k, self.box.get_right()[0] - 0.1
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)

    def loop_back(self, bar, color=CORAL):
        """Flash the gutter arc and move the cursor back to `while True:`."""
        k = self.size / 20
        gx = self.box.get_left()[0] + 0.36 * k
        path = m.ArcBetweenPoints([gx, self.base(9) + 0.08 * k, 0], [gx, self.base(0) + 0.08 * k, 0], angle=-0.6)
        flash = m.ShowPassingFlash(path.set_stroke(color, 6), time_width=0.7, run_time=0.7)
        return m.AnimationGroup(flash, m.Transform(bar, self.bar(0, color=color), run_time=0.7))


class Slot:
    """One place on screen whose content is swapped (the replay's trace line)."""

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


# ---------------------------------------------------------------------------- the hook's wall of mechanisms

WALL = [(3, "tools"), (8, "system prompt"), (9, "plans"), (10, "skills"), (11, "compaction"),
        (12, "memory"), (16, "MCP"), (17, "permissions"), (18, "hooks"), (20, "recovery"),
        (21, "subagents"), (22, "task graphs"), (23, "background"), (24, "teams"), (25, "worktrees")]
assert len(WALL) == 15


def bars(widths, color, height=0.075, gap=0.075, stripe=None):
    rows = m.VGroup()
    for w in widths:
        bar_ = m.RoundedRectangle(corner_radius=0.03, width=w, height=height, stroke_width=0, fill_color=color, fill_opacity=1)
        if stripe:
            tab = m.RoundedRectangle(corner_radius=0.02, width=0.05, height=height + 0.05, stroke_width=0, fill_color=stripe, fill_opacity=1)
            rows.add(m.VGroup(tab, bar_).arrange(m.RIGHT, buff=0.06))
        else:
            rows.add(bar_)
    return rows.arrange(m.DOWN, buff=gap, aligned_edge=m.LEFT)


def mini(ep):
    """A small reminder of each episode's signature visual."""
    if ep == 3:
        return icon(WRENCH, 0.62)
    if ep == 8:
        return bars([1.0, 0.7, 0.85, 0.55], MUTED, stripe=ORCHID)
    if ep == 9:
        rows = m.VGroup()
        for state in ("done", "done", "doing", "todo"):
            box = m.RoundedRectangle(corner_radius=0.03, width=0.13, height=0.13, stroke_width=2)
            if state == "done":
                box.set_stroke(MINT).set_fill(MINT, 1)
            elif state == "doing":
                box.set_stroke(SUN).set_fill(SUN_FILL, 1)
            else:
                box.set_stroke(FAINT).set_fill(CARD, 1)
            line = m.RoundedRectangle(corner_radius=0.03, width=0.75, height=0.06, stroke_width=0, fill_color=MUTED if state != "todo" else FAINT, fill_opacity=1)
            rows.add(m.VGroup(box, line).arrange(m.RIGHT, buff=0.1))
        return rows.arrange(m.DOWN, buff=0.06, aligned_edge=m.LEFT)
    if ep == 10:
        return icon(BOOK, 0.55)
    if ep == 11:
        meter = Meter(1.25, 0.86, SUN, 0.16)
        tick = m.Line([0, -0.16, 0], [0, 0.16, 0]).set_stroke(INK, 3).move_to(meter.track.get_left() + 0.895 * 1.25 * m.RIGHT)
        return m.VGroup(meter, tick)
    if ep == 12:
        return icon("ep12-moon", 0.58)
    if ep == 16:
        return icon("ep16-server", 0.55)
    if ep == 17:
        dots = m.VGroup(*[m.Dot(radius=0.07, color=c) for c in (ROSE, SUN, MINT)]).arrange(m.DOWN, buff=0.07)
        return m.VGroup(icon("ep17-shield", 0.6), dots).arrange(m.RIGHT, buff=0.14)
    if ep == 18:
        return icon("ep18-hook", 0.62)
    if ep == 20:
        heights = [0.05, 0.09, 0.16, 0.3, 0.5, 0.5]
        cols = m.VGroup(*[m.Rectangle(width=0.11, height=h, stroke_width=0, fill_color=MINT, fill_opacity=1) for h in heights])
        cols.arrange(m.RIGHT, buff=0.05, aligned_edge=m.DOWN)
        cap = m.DashedLine(cols.get_corner(m.UL) + 0.06 * m.LEFT, cols.get_corner(m.UR) + 0.06 * m.RIGHT, dash_length=0.05).set_stroke(SUN, 2)
        return m.VGroup(cols, cap)
    if ep == 21:
        def panel(n, h):
            box = m.RoundedRectangle(corner_radius=0.05, width=0.42, height=h, stroke_color=AZURE, stroke_width=2, fill_color=CARD, fill_opacity=1)
            rows = bars([0.28] * n, MUTED, height=0.05, gap=0.05).move_to(box).align_to(box, m.UP).shift(0.07 * m.DOWN)
            return m.VGroup(box, rows)
        a, b = panel(1, 0.58), panel(5, 0.58)
        pair = m.VGroup(a, b).arrange(m.RIGHT, buff=0.32)
        link = m.Arrow(a.get_right(), b.get_left(), buff=0.04, color=AZURE, stroke_width=3, max_tip_length_to_length_ratio=0.35)
        return m.VGroup(pair, link)
    if ep == 22:
        pts = [(-0.45, 0), (0, 0.2), (0, -0.2), (0.45, 0)]
        lines = m.VGroup(*[m.Line([*pts[a], 0], [*pts[b], 0]).set_stroke(MUTED, 2) for a, b in ((0, 1), (0, 2), (1, 3), (2, 3))])
        nodes = m.VGroup(*[m.Dot([*p, 0], radius=0.075, color=c) for p, c in zip(pts, (MINT, MINT, AZURE, FAINT))])
        return m.VGroup(lines, nodes)
    if ep == 23:
        return icon("ep23-washer", 0.6)
    if ep == 24:
        return m.VGroup(*[otto(0.3, c, f) for c, f in ((CORAL, CORAL_FILL), (AZURE, AZURE_FILL), (ORCHID, ORCHID_FILL))]).arrange(m.RIGHT, buff=0.1)
    if ep == 25:
        return icon(GIT, 0.6)
    raise ValueError(ep)


def wall_tile(ep, title, width=2.4, height=1.3):
    color, _ = accent(ep)
    box = card(width, height, EDGE, CARD, radius=0.16, stroke=2)
    num = code(f"{ep:02d}", 20, color, weight=m.BOLD).move_to(box.get_corner(m.UL) + [0.2, -0.24, 0], aligned_edge=m.UL)
    name = text(title, 20, INK, weight=m.SEMIBOLD).move_to(box.get_corner(m.DL) + [0.2, 0.17, 0], aligned_edge=m.DL)
    pic = mini(ep)
    room_w, room_h = width - 0.9, 0.62
    if pic.width > room_w or pic.height > room_h:
        pic.scale(min(room_w / pic.width, room_h / pic.height))
    pic.move_to([box.get_right()[0] - 0.2 - pic.width / 2, box.get_top()[1] - 0.14 - pic.height / 2, 0])
    return m.VGroup(box, num, name, pic)


# ---------------------------------------------------------------------------- the full-turn diagram

COL_W, CHIP_H = 2.85, 0.48
LEFT_X, RIGHT_X = -6.6 + COL_W / 2, 6.6 - COL_W / 2
CARD_CENTER_TOP = 2.48

LEFT_SPEC = [("user", "user input", None, False), ("ups", "UserPromptSubmit", 18, True),
             ("notes", "notifications", 23, False), ("compact", "compaction", 11, False),
             ("prompt", "system prompt", 8, False), ("skills", "skills catalog", 10, False),
             ("memory", "memory", 12, False)]
RIGHT_SPEC = [("pre", "PreToolUse", 18, True), ("perm", "permission", 17, False),
              ("handlers", "TOOL_HANDLERS", 3, True), ("mcp", "MCP servers", 16, False),
              ("bg", "background", 23, False), ("post", "PostToolUse", 18, True)]
BOTTOM_SPEC = [("recovery", "recovery", 20, False), ("stop", "Stop", 18, True), ("answer", "final answer", None, False)]
SPEC = {k: (label, ep, mono) for k, label, ep, mono in LEFT_SPEC + RIGHT_SPEC + BOTTOM_SPEC}
assert [k for k in SPEC if SPEC[k][1]] == ["ups", "notes", "compact", "prompt", "skills", "memory",
                                            "pre", "perm", "handlers", "mcp", "bg", "post", "recovery", "stop"]
assert sorted(k for k in SPEC if SPEC[k][1]) == sorted(LAYER_KEYS)
TURN_EPISODES = sorted({ep for _, ep, _ in SPEC.values() if ep} | {2})
assert TURN_EPISODES == [2, 3, 8, 10, 11, 12, 16, 17, 18, 20, 23]  # eleven episodes in one picture
ROLE_ACCENT = {"user": (AZURE, AZURE_FILL), "answer": (CORAL, CORAL_FILL)}


def layer_chip(key, width=COL_W):
    label, ep, mono = SPEC[key]
    color, fill = accent(ep) if ep else ROLE_ACCENT[key]
    words = code(label, 17, INK) if mono else text(label, 20, INK, weight=m.SEMIBOLD)
    if ep:
        tag = ep_badge(ep)
    else:
        tag = m.RoundedRectangle(corner_radius=0.04, width=0.09, height=CHIP_H - 0.16, stroke_width=0, fill_color=color, fill_opacity=1)
    inner_w = tag.width + 0.14 + words.width + 0.3
    box = card(max(width, inner_w) if width else inner_w, CHIP_H, EDGE, CARD, radius=0.12, stroke=2)
    tag.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    words.next_to(tag, m.RIGHT, buff=0.14)
    chip_ = m.VGroup(box, tag, words)
    chip_.key, chip_.accent, chip_.tint = key, color, fill
    return chip_


def lit(chip_):
    """The chip once its layer has run: its outline and tint in its episode's colour."""
    return m.AnimationGroup(
        chip_[0].animate(run_time=0.5).set_stroke(chip_.accent, 2.5).set_fill(chip_.tint, 1),
        outline_flash(chip_[0], chip_.accent),
    )


class Diagram:
    """Episode 2's loop in the middle; every layer plugged in where it runs."""

    def __init__(self):
        self.loop = LoopCard(18)
        lc = self.loop
        lc.group.move_to([0, 0, 0])
        lc.group.shift((CARD_CENTER_TOP - lc.box.get_top()[1]) * m.UP)
        self.chips = {}
        y0, y1, y7 = lc.line_y(0), lc.line_y(1), lc.line_y(7)
        self.y = {"while": y0, "create": y1, "tools": y7}
        left_card, right_card = lc.box.get_left()[0], lc.box.get_right()[0]
        bus_l, bus_r = (left_card + LEFT_X + COL_W / 2) / 2, (right_card + RIGHT_X - COL_W / 2) / 2
        # left column: the user's message and its hook (around the turn), then what runs before each call
        rows = [y0 - 0.12 + (-2.62 - y0 + 0.12) * i / 5 for i in range(6)]  # six rows, shared by both columns
        self.chips["user"] = layer_chip("user").move_to([LEFT_X, 2.3, 0])
        rail_keys = ["notes", "compact", "prompt", "skills", "memory"]
        for key, y in zip(["ups", *rail_keys], rows):
            self.chips[key] = layer_chip(key).move_to([LEFT_X, y, 0])
        right_keys = [k for k, *_ in RIGHT_SPEC]
        for key, y in zip(right_keys, rows):
            self.chips[key] = layer_chip(key).move_to([RIGHT_X, y, 0])
        # recovery wraps the loop; Stop and the answer sit below it
        self.frame = m.DashedVMobject(
            m.RoundedRectangle(corner_radius=0.24, width=lc.box.width + 0.26, height=lc.box.height + 0.26).move_to(lc.box).set_stroke(MUTED, 2.5),
            num_dashes=60,
        )
        low = -2.3
        self.chips["recovery"] = layer_chip("recovery", width=None).move_to([-2.3, low, 0])
        self.chips["stop"] = layer_chip("stop", width=None).move_to([0.05, low, 0])
        self.chips["answer"] = layer_chip("answer", width=None).move_to([2.2, low, 0])
        fb = self.frame.get_bottom()[1]
        rec = self.chips["recovery"]
        self.rec_link = m.Line([rec.get_x(), rec.get_top()[1], 0], [rec.get_x(), fb, 0]).set_stroke(FAINT, 2.5)
        stop = self.chips["stop"]
        self.stop_link = m.Line([stop.get_x(), lc.box.get_bottom()[1], 0], [stop.get_x(), stop.get_top()[1], 0]).set_stroke(FAINT, 2.5)
        self.stop_dot = m.Dot([stop.get_x(), lc.box.get_bottom()[1], 0], radius=0.055, color=FAINT)
        ans = self.chips["answer"]
        self.answer_arrow = arrow(stop.get_right(), ans.get_left(), CORAL, buff=0.08)
        # connectors
        ups = self.chips["ups"]
        self.user_arrow = arrow(self.chips["user"].get_bottom(), ups.get_top(), AZURE, buff=0.04)
        self.ups_link = m.VGroup(
            m.Line([ups.get_right()[0], ups.get_y(), 0], [bus_l, ups.get_y(), 0]),
            m.Line([bus_l, ups.get_y(), 0], [bus_l, y0, 0]),
            m.Line([bus_l, y0, 0], [left_card, y0, 0]),
        ).set_stroke(FAINT, 2.5)
        self.ups_dot = m.Dot([left_card, y0, 0], radius=0.055, color=FAINT)
        bus_x2 = bus_l - 0.0
        self.left_rail = {}
        prev_y = y1
        for key in rail_keys:
            c = self.chips[key]
            seg = m.VGroup(
                m.Line([bus_x2, prev_y, 0], [bus_x2, c.get_y(), 0]),
                m.Line([bus_x2, c.get_y(), 0], [c.get_right()[0], c.get_y(), 0]),
            ).set_stroke(FAINT, 2.5)
            self.left_rail[key] = seg
            prev_y = c.get_y()
        self.left_in = m.Line([bus_x2, y1, 0], [left_card, y1, 0]).set_stroke(FAINT, 2.5)
        self.left_dot = m.Dot([left_card, y1, 0], radius=0.055, color=FAINT)
        self.right_out = m.Line([right_card, y7, 0], [bus_r, y7, 0]).set_stroke(FAINT, 2.5)
        self.right_dot = m.Dot([right_card, y7, 0], radius=0.055, color=FAINT)
        self.right_rail = {}
        for key in right_keys:
            c = self.chips[key]
            self.right_rail[key] = m.VGroup(
                m.Line([bus_r, y7, 0], [bus_r, c.get_y(), 0]),
                m.Line([bus_r, c.get_y(), 0], [c.get_left()[0], c.get_y(), 0]),
            ).set_stroke(FAINT, 2.5)
        self.bus_l, self.bus_r = bus_l, bus_r

    def everything(self):
        parts = [self.loop.group, self.frame, *self.chips.values(), self.rec_link, self.stop_link, self.stop_dot,
                 self.answer_arrow, self.user_arrow, self.ups_link, self.ups_dot, *self.left_rail.values(),
                 self.left_in, self.left_dot, self.right_out, self.right_dot, *self.right_rail.values()]
        return m.VGroup(*parts)

    def link_of(self, key):
        if key in self.left_rail:
            return self.left_rail[key]
        if key in self.right_rail:
            return self.right_rail[key]
        return {"ups": self.ups_link, "stop": self.stop_link, "recovery": self.rec_link, "answer": self.answer_arrow}.get(key)

    def flow(self, key, color=None):
        """A pulse along the connector of one layer."""
        link = self.link_of(key)
        c = self.chips[key]
        color = color or c.accent
        if link is None:
            return outline_flash(c[0], color)
        lines = [link] if isinstance(link, (m.Line, m.Arrow)) else list(link)
        return m.Succession(*[m.ShowPassingFlash(seg.copy().set_stroke(color, 6), time_width=0.8, run_time=0.3) for seg in lines])


# ---------------------------------------------------------------------------- the episode


class Ep33TheCompleteAgent(Episode):
    number = 33

    def lesson(self):
        self.hook()
        self.core()
        self.turn()
        self.replay()
        self.more_tools()
        self.harness()
        self.patterns()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes=f"Episode 1's bug ({EPISODES_AGO} episodes ago): cart.total() skips the last item, pytest -q says 1 failed, 12 passed. Since then one mechanism per episode: {len(WALL)} tiles, each with a reminder of its episode's signature visual. learn-claude-code s19, What's Next: 'a real Agent does not run as 19 separate demos.' Episode 19 is not shown.")
        prompt = code("$ agents --episode 01", 22, MUTED, t2c={r"^\$": MINT}).move_to([-6.6, 2.75, 0], aligned_edge=m.LEFT)
        bot = otto(1.15).move_to([-5.25, 1.1, 0])
        ask = bubble(text(TASK, 26, INK), AZURE, AZURE_FILL, tail="left")
        ask.move_to([-1.0, 1.45, 0])
        ask.shift((bot.get_right()[0] + 0.35 - ask[0].get_vertices()[2][0]) * m.RIGHT)
        term = terminal(["$ pytest -q", "! FAILED tests/test_cart.py::test_total", "1 failed, 12 passed"], width=6.3, size=20, title="tinyshop")
        term.move_to([-3.3, -1.75, 0])
        block = code_block(CART.rstrip(), size=20, width=5.9, title="cart.py")
        block.move_to([3.65, -1.75, 0]).align_to(term, m.UP)
        bug = glyphs(block[1], BUG_LINE)
        gutter = block[0].get_left()[0] + 0.17
        mark = m.Line([gutter, bug.get_top()[1] + 0.05, 0], [gutter, bug.get_bottom()[1] - 0.05, 0]).set_stroke(ROSE, 6)
        first = m.VGroup(prompt, bot, ask, term, block, mark)

        tiles = m.VGroup(*[wall_tile(ep, title) for ep, title in WALL])
        tiles.arrange_in_grid(rows=3, cols=5, buff=(0.28, 0.25)).move_to([0, 0.62, 0])
        self.wall_at = {ep: t.get_center() for (ep, _), t in zip(WALL, tiles)}
        by_ep = {ep: t for (ep, _), t in zip(WALL, tiles)}
        group = lambda *eps: m.LaggedStart(*[m.FadeIn(by_ep[e], shift=0.2 * m.UP, scale=0.94) for e in eps], lag_ratio=0.25)  # noqa: E731

        quote = title_text("“A real Agent does not run as 19 separate demos.”", 34, INK).move_to([0, -2.42, 0])
        glyphs(quote, "19 separate demos").set_color(LIME)
        src = cite("learn-claude-code · s19").next_to(quote, m.DOWN, buff=0.22)
        big = LoopCard(20)
        big.group.move_to([0, -0.2, 0])
        self.big = big

        self.say(
            "[Thirty-two episodes ago], [Otto met its first bug]: [the cart total skipped the last item], [and one test failed.]",
            m.AnimationGroup(m.AddTextLetterByLetter(prompt, time_per_char=0.03), run_time=0.8),
            m.AnimationGroup(m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(ask, shift=0.2 * m.UP), lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(block, shift=0.15 * m.UP), m.Create(mark)),
            m.AnimationGroup(m.FadeIn(term[:4]), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in term[4]], lag_ratio=0.3)),
        )
        self.say(
            "[Since then, every episode added one piece]: [tools], [prompts and plans], [skills, compaction and memory.]",
            m.FadeOut(first, shift=0.2 * m.DOWN),
            group(3),
            group(8, 9),
            group(10, 11, 12),
        )
        self.say(
            "[Then MCP, permissions, hooks and recovery], [subagents and task graphs], [background jobs, teams and worktrees.]",
            group(16, 17, 18, 20),
            group(21, 22),
            group(23, 24, 25),
        )
        self.say(
            "[But as Learn Claude Code puts it], [a real agent does not run as nineteen separate demos.]",
            m.FadeIn(src, shift=0.1 * m.UP),
            m.FadeIn(quote, shift=0.15 * m.UP),
        )
        self.say(
            "[So in this finale], [we put every piece back together], [around the one loop that never changed.]",
            m.FadeOut(m.VGroup(quote, src), shift=0.2 * m.DOWN),
            m.LaggedStart(*[m.FadeOut(t, target_position=[0, -0.2, 0], scale=0.25) for t in tiles], lag_ratio=0.06, run_time=1.6),
            m.FadeIn(big.group, scale=0.9),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ the core

    def core(self):
        self.next_section("1 · The core", notes=f"Episode 2's loop, read from ep02-the-agent-loop/scene.py and asserted identical ({CHANGED} lines changed, {len(LOOP_SRC.splitlines())} lines). Its only two calls: client.messages.create (the model) and tool_results (the tools); every later layer lives inside them or around the turn.")
        big = self.big
        same = pill(f"{CHANGED} lines changed since Episode 2", MINT, MINT_FILL, 22).next_to(big.box, m.DOWN, buff=0.32)
        bar = big.bar(1, 2)
        labels = []
        for words, row, color in (("the model", 1, CORAL), ("the tools", 7, SUN)):
            lbl = text(words, 26, color, weight=m.SEMIBOLD).move_to([big.box.get_right()[0] + 0.85, big.line_y(row), 0], aligned_edge=m.LEFT)
            pointer = arrow(lbl.get_left() + 0.08 * m.LEFT, [big.box.get_right()[0] + 0.04, big.line_y(row), 0], color, buff=0.04)
            labels.append(m.VGroup(pointer, lbl))
        model_lbl, tools_lbl = labels

        self.say(
            "[In the middle]: [the same ten lines from episode two.] [Not one character has changed.]",
            self.new_header("01 · THE CORE", "Ten lines, unchanged"),
            m.Circumscribe(big.box, color=CORAL, buff=0.08),
            m.FadeIn(same, shift=0.15 * m.UP),
        )
        self.say(
            "[Call the model.] [If it answers in words, return.] [Otherwise run its tools], [append the results], [and go round again.]",
            m.FadeIn(bar, run_time=0.5),
            m.Transform(bar, big.bar(5, 6)),
            m.Transform(bar, big.bar(7, color=SUN)),
            m.Transform(bar, big.bar(8, 9, color=MINT)),
            big.loop_back(bar),
        )
        self.say(
            "[The loop makes just two calls]: [one to the model], [and one to the tools.]",
            m.FadeOut(bar),
            m.AnimationGroup(m.Circumscribe(big.part(1, "client.messages.create"), color=CORAL, buff=0.06), m.FadeIn(model_lbl, shift=0.15 * m.RIGHT)),
            m.AnimationGroup(m.Circumscribe(big.part(7, "tool_results(response)"), color=SUN, buff=0.06), m.FadeIn(tools_lbl, shift=0.15 * m.RIGHT)),
        )
        self.core_extra = m.VGroup(same, model_lbl, tools_lbl)
        self.wait(0.3)

    # ------------------------------------------------------------------ one full turn

    def turn(self):
        self.next_section("2 · One full turn", notes="learn-claude-code s20's turn, each stage labelled with the episode that taught it: user input → UserPromptSubmit hooks (18) → background/cron notifications (23) → compaction (11) → system prompt assembled with the skills catalog and memories (8, 10, 12) → the model, wrapped in recovery (2, 20) → no tool_use: Stop hooks (18) → answer; tool_use: PreToolUse hooks (18) + permission (17) → TOOL_HANDLERS / MCP servers / background (3, 16, 23) → PostToolUse hooks (18) → tool_result appended → next round. Fourteen layers; the loop in the middle is Episode 2's, unchanged (asserted). Facts as of October 2026.")
        d = self.d = Diagram()
        lc, ch = d.loop, d.chips
        src = top_cite("learn-claude-code · s20")
        bar = lc.bar(0, color=AZURE)
        self.bar = bar

        def add(key, shift):
            return m.AnimationGroup(m.FadeIn(ch[key], shift=0.2 * shift), m.Create(d.link_of(key)))

        self.say(
            "[Every layer lives inside those two calls, or around the turn.] [Let's add them in the order Learn Claude Code's final session runs them.]",
            m.AnimationGroup(
                self.new_header("02 · ONE TURN", "Every layer, in order"),
                m.FadeOut(self.core_extra, shift=0.2 * m.DOWN),
                m.ReplacementTransform(self.big.group, lc.group),
            ),
            m.FadeIn(src, shift=0.1 * m.DOWN),
        )
        self.say(
            "[A turn starts with the user's message.] [User prompt submit hooks can log it, or add context]: [episode eighteen.]",
            m.FadeIn(ch["user"], shift=0.2 * m.RIGHT),
            m.AnimationGroup(m.GrowArrow(d.user_arrow), m.FadeIn(ch["ups"], shift=0.2 * m.RIGHT), m.Create(d.ups_link), m.FadeIn(d.ups_dot), m.FadeIn(bar)),
            m.Circumscribe(ch["ups"][1], color=ch["ups"].accent, buff=0.05),
        )
        self.say(
            "[Before each model call], [finished background jobs arrive as notifications]: [episode twenty-three.]",
            m.AnimationGroup(m.Transform(bar, lc.bar(1, 2)), m.Create(d.left_in), m.FadeIn(d.left_dot)),
            add("notes", m.RIGHT),
            m.Circumscribe(ch["notes"][1], color=ch["notes"].accent, buff=0.05),
        )
        self.say(
            "[Compaction keeps the history under its threshold]: [episode eleven.]",
            add("compact", m.RIGHT),
            m.Circumscribe(ch["compact"][1], color=ch["compact"].accent, buff=0.05),
        )
        self.say(
            "[Then the system prompt is assembled], [episode eight], [with the skills catalog from ten], [and memories from twelve.]",
            add("prompt", m.RIGHT),
            m.Circumscribe(ch["prompt"][1], color=ch["prompt"].accent, buff=0.05),
            add("skills", m.RIGHT),
            add("memory", m.RIGHT),
        )
        self.say(
            "[Then the model is called], [wrapped in recovery]: [retries with backoff, from episode twenty.]",
            m.AnimationGroup(m.Succession(*[d.flow(k, CORAL) for k in ("memory", "skills", "prompt", "compact", "notes")], run_time=1.2),
                             m.ShowPassingFlash(d.left_in.copy().set_stroke(CORAL, 6), time_width=0.8, run_time=1.2),
                             m.Circumscribe(lc.part(1, "client.messages.create"), color=CORAL, buff=0.06)),
            m.AnimationGroup(m.Create(d.frame, run_time=1.2), m.FadeIn(ch["recovery"], shift=0.2 * m.UP), m.Create(d.rec_link)),
            m.Circumscribe(ch["recovery"][1], color=ch["recovery"].accent, buff=0.05),
        )
        self.say(
            "[If the reply asks for no tool], [stop hooks get a last word], [and the answer goes back to the user.]",
            m.Transform(bar, lc.bar(5, 6)),
            m.AnimationGroup(m.FadeIn(d.stop_dot), m.Create(d.stop_link), m.FadeIn(ch["stop"], shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.GrowArrow(d.answer_arrow), m.FadeIn(ch["answer"], shift=0.2 * m.RIGHT)),
        )
        self.say(
            "[If it does ask for one], [pre tool use hooks check the call], [and the permission gate decides]: [episodes eighteen and seventeen.]",
            m.AnimationGroup(m.Transform(bar, lc.bar(7, color=SUN)), m.Create(d.right_out), m.FadeIn(d.right_dot)),
            add("pre", m.LEFT),
            add("perm", m.LEFT),
            m.AnimationGroup(m.Circumscribe(ch["pre"][1], color=ch["pre"].accent, buff=0.05), m.Circumscribe(ch["perm"][1], color=ch["perm"].accent, buff=0.05)),
        )
        self.say(
            "[Then it's dispatched]: [to a built-in handler], [an MCP server], [or a background task]: [episodes three, sixteen and twenty-three.]",
            m.ShowPassingFlash(d.right_out.copy().set_stroke(SUN, 6), time_width=0.8, run_time=0.6),
            add("handlers", m.LEFT),
            add("mcp", m.LEFT),
            add("bg", m.LEFT),
            m.AnimationGroup(*[m.Circumscribe(ch[k][1], color=ch[k].accent, buff=0.05) for k in ("handlers", "mcp", "bg")]),
        )
        self.say(
            "[Post tool use hooks run after it], [the result is appended], [and the loop goes round again.]",
            add("post", m.LEFT),
            m.Transform(bar, lc.bar(8, 9, color=MINT)),
            lc.loop_back(bar, color=CORAL),
        )
        order = ["user", "ups", "notes", "compact", "prompt", "skills", "memory", "pre", "perm", "handlers", "mcp", "bg", "post", "stop", "answer"]
        count = code(f"{len(LAYER_KEYS)} layers · {len(TURN_EPISODES)} episodes · 1 loop, unchanged", 22, MUTED, t2c={r"\d+": SUN}).move_to([0, -3.3, 0])
        self.say(
            "[That's one full turn]: [fourteen layers], [and in the middle, still the same ten lines.]",
            m.AnimationGroup(
                m.Succession(*[outline_flash(ch[k][0], ch[k].accent, run_time=0.32) for k in order], run_time=2.2),
                m.ShowPassingFlash(d.frame.copy().set_stroke(MINT, 6), time_width=0.5, run_time=2.2),
            ),
            m.FadeIn(count, shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeOut(bar), m.Circumscribe(lc.box, color=CORAL, buff=0.06)),
        )
        self.turn_extra = m.VGroup(src, count)
        self.wait(0.4)

    # ------------------------------------------------------------------ the replay

    def replay(self):
        self.next_section("3 · The first bug, again", notes=f"The scene runs the harness above on Episode 1's task. Round 1: UserPromptSubmit logs the request; {events('notes')[0]} notifications; compaction {events('compact')[0]}; system prompt {events('prompt')[0]}; skills {events('skills')[0]}; memory {events('memory')[0]}; the first request gets a 529 and recovery retries after {WAITED[0]:.2f} s (seeded jitter, illustrative); bash pytest -q → {events('handlers')[0]}. Round 2 reads cart.py, round 3 edits one line and the PostToolUse hook runs the tests ({events('post')[-1]}); round 4 answers; the Stop hook sees green tests. {llm.replies} model replies (Episode 1: {EP1_MODEL_CALLS}); {len(USED)} of {len(LAYER_KEYS)} layers ran; MCP and background were not needed.")
        d, lc, ch = self.d, self.d.loop, self.d.chips
        trace = Slot(self)
        bar = lc.bar(0, color=AZURE)

        def line(head, body, color=INK, rules=None, key=None):
            """The trace line: the layer's reminder from the wall (its episode's signature), its name, what it did."""
            s = f"{head}  {body}"
            t2c = {re.escape(head): MUTED}
            t2c.update(rules or {})
            words = code(s, 20, color, t2c=t2c)
            ep = SPEC[key][1] if key in SPEC else None
            if key == "model":
                pic = icon(LLM_ICON, 0.42)
            elif ep:
                pic = mini(ep)
                pic.scale(min(0.42 / pic.height, 0.95 / pic.width))
            else:
                group = m.VGroup(words).move_to([0, -3.3, 0])
                group.words = words
                return group
            group = m.VGroup(pic, words).arrange(m.RIGHT, buff=0.25).move_to([0, -3.3, 0])
            group.words = words
            return group

        def show(key, detail, color=INK, rules=None, head=None):
            label = head or SPEC[key][0]
            return trace.to(line(label, detail, color, rules, key))

        task_line = line("user", f'"{TASK}"', AZURE)
        self.say(
            "[Now watch the layers work], [on the course's very first task]: [the checkout test fails. Fix it.]",
            m.AnimationGroup(self.new_header("03 · REPLAY", "The first bug, one more time"), m.FadeOut(self.turn_extra)),
            lit(ch["user"]),
            m.AnimationGroup(trace.to(task_line), d.flow("ups", AZURE)),
        )
        self.say(
            "[The prompt hook logs it.] [No notifications are waiting], [and the history is far below the compaction threshold.]",
            m.AnimationGroup(lit(ch["ups"]), m.FadeIn(bar, run_time=0.5), show("ups", events("ups")[0])),
            m.AnimationGroup(m.Transform(bar, lc.bar(1, 2)), d.flow("notes"), lit(ch["notes"]), show("notes", events("notes")[0])),
            m.AnimationGroup(d.flow("compact"), lit(ch["compact"]), show("compact", "≈" + events("compact")[0], rules={r"\d[\d,]*": SUN})),
        )
        mem = events("memory")[0]
        mem_line = line("memory", mem, rules={r"\"[^\"]*\"": MINT}, key="memory")
        self.say(
            "[The prompt is assembled], [with the skills catalog], [and one memory]: [test with pytest.]",
            m.AnimationGroup(d.flow("prompt"), lit(ch["prompt"]), show("prompt", events("prompt")[0], rules={r"\d+": SUN})),
            m.AnimationGroup(d.flow("skills"), lit(ch["skills"]), show("skills", events("skills")[0], rules={r"\d+": SUN})),
            m.AnimationGroup(d.flow("memory"), lit(ch["memory"]), trace.to(mem_line)),
            m.Circumscribe(glyphs(mem_line.words, '"Test with pytest -q"'), color=MINT, buff=0.06),
        )
        busy = line("model", "529 overloaded", ROSE, {r"model": MUTED}, key="model")
        self.say(
            "[The first call hits an overloaded server.] [Recovery waits about half a second], [and tries again.]",
            m.AnimationGroup(m.ShowPassingFlash(d.left_in.copy().set_stroke(CORAL, 6), time_width=0.8, run_time=0.6),
                             outline_flash(d.frame, ROSE, run_time=1.0), trace.to(busy)),
            m.AnimationGroup(d.frame.animate.set_stroke(MINT, 3), lit(ch["recovery"]), d.flow("recovery"),
                             show("recovery", f"retry in {WAITED[0]:.2f} s", rules={r"\d+\.\d+": SUN})),
            m.Circumscribe(lc.part(1, "client.messages.create"), color=CORAL, buff=0.06),
        )
        result = line("bash", events("handlers")[0].split(": ", 1)[1], ROSE, key="handlers")
        self.say(
            "[The model asks to run the tests.] [The hooks and the gate let it through], [and bash replies]: [one failed, twelve passed.]",
            m.AnimationGroup(m.Transform(bar, lc.bar(3, 4)), show("model", events("model")[0], SUN, head="model")),
            m.AnimationGroup(m.Transform(bar, lc.bar(7, color=SUN)), m.ShowPassingFlash(d.right_out.copy().set_stroke(SUN, 6), time_width=0.8, run_time=0.5),
                             d.flow("pre"), d.flow("perm"), lit(ch["pre"]), lit(ch["perm"]),
                             show("perm", events("perm")[0], rules={r"allow": MINT})),
            m.AnimationGroup(d.flow("handlers"), lit(ch["handlers"])),
            trace.to(result),
        )
        sweep = lambda: m.AnimationGroup(  # noqa: E731 — one more round, quickly
            m.Succession(lc.loop_back(bar), m.Transform(bar, lc.bar(1, 2), run_time=0.3), m.Transform(bar, lc.bar(7, color=SUN), run_time=0.3)),
            m.Succession(*[d.flow(k) for k in ("notes", "compact", "prompt", "skills", "memory")], run_time=1.0),
            m.Succession(m.Wait(0.9), *[d.flow(k) for k in ("pre", "perm", "handlers")], run_time=1.6),
        )
        ran = line("PostToolUse", events("post")[-1].split(": ")[0], rules={}, key="post")
        passed = line("PostToolUse", events("post")[-1], MINT, {r"ran the tests:": INK}, key="post")
        self.say(
            "[Round two reads the cart file.] [Round three changes one line], [and a post tool use hook runs the tests]: [thirteen passed.]",
            m.AnimationGroup(sweep(), show("handlers", BUG_LINE, ROSE, head="read_file")),
            m.AnimationGroup(sweep(), show("handlers", f"{OLD} → {NEW}", INK, {re.escape(NEW): MINT, re.escape(OLD): ROSE}, head="edit_file")),
            m.AnimationGroup(d.flow("post"), lit(ch["post"]), trace.to(ran)),
            trace.to(passed),
        )
        answer = line("answer", f'"{PLAN[-1]}"', CORAL)
        self.say(
            "[Otto answers in words.] [The stop hook sees green tests], [and the answer goes back]: [fixed.]",
            m.Succession(lc.loop_back(bar), m.AnimationGroup(m.Transform(bar, lc.bar(5, 6), run_time=0.5), show("model", events("model")[-1], CORAL, head="model"))),
            m.AnimationGroup(d.flow("stop"), lit(ch["stop"]), show("stop", events("stop")[0], rules={r"green": MINT})),
            m.AnimationGroup(m.GrowArrow(d.answer_arrow.copy().set_color(CORAL)), lit(ch["answer"])),
            trace.to(answer),
        )
        dark = [ch["mcp"], ch["bg"]]
        self.say(
            "[Two layers stayed dark]: [this turn needed no MCP server], [and nothing was slow enough for the background.]",
            m.AnimationGroup(m.FadeOut(bar), *[m.Circumscribe(c, color=MUTED, buff=0.06) for c in dark]),
            show("mcp", "not needed this turn", MUTED, head="MCP servers"),
            show("bg", "not needed this turn", MUTED, head="background"),
        )
        tally = code(f"{llm.replies} model replies (Episode 1: {EP1_MODEL_CALLS}) · {len(WAITED)} retry · {len(USED)} of {len(LAYER_KEYS)} layers ran",
                     20, INK, t2c={r"\d+": SUN}).move_to([0, -3.3, 0])
        self.say(
            "[Four replies from the model], [one fewer than in episode one]: [a hook ran the last test for Otto.]",
            trace.to(tally),
            m.Circumscribe(glyphs(tally, f"(Episode 1: {EP1_MODEL_CALLS})"), color=SUN, buff=0.06),
            outline_flash(ch["post"][0], ch["post"].accent, run_time=1.0),
        )
        self.wait(0.6)
        self.clear()

    # ------------------------------------------------------------------ teams are tools

    def more_tools(self):
        self.next_section("4 · Teams are tools", notes=f"learn-claude-code s20: 'The built-in tool pool contains 27 tools' (the page's header says 32: we show the 27 built-in ones), in eight groups: {', '.join(f'{g} {len(t)}' for g, _, t in TOOL_GROUPS)} = {N_TOOLS}. Subagents (task), task graphs, cron, teams, protocols and worktrees all reach the loop as tools, dispatched by tool_results through TOOL_HANDLERS. Facts as of October 2026.")
        rows = m.VGroup()
        top = 2.15
        for i, (group, eps, names) in enumerate(TOOL_GROUPS):
            y = top - 0.52 * i
            badges = m.VGroup(*[ep_badge(e, 0.34) for e in eps]).arrange(m.RIGHT, buff=0.06)
            badges.move_to([-4.5, y, 0], aligned_edge=m.RIGHT)
            label = text(group, 24, INK, weight=m.SEMIBOLD).move_to([-4.3, y, 0], aligned_edge=m.LEFT)
            names_ = code("  ".join(names), 17, SUN).move_to([-1.35, y, 0], aligned_edge=m.LEFT)
            rows.add(m.VGroup(badges, label, names_))
        scaling = [rows[i] for i in (1, 2, 3, 4, 5, 6)]
        total = title_text(f"{N_TOOLS} built-in tools", 44, SUN).move_to([-2.2, -2.45, 0])
        src = cite("learn-claude-code · s20").next_to(total, m.DOWN, buff=0.2).align_to(total, m.LEFT)
        line_ = code("results = tool_results(response)", 22, INK, t2c=CODE_T2C)
        box = card(line_.width + 0.7, 0.75, SUN, CARD, radius=0.16, stroke=2)
        disp = m.VGroup(box, line_.move_to(box)).move_to([3.7, -2.45, 0])
        cap = code("one line of the loop", 20, MUTED).next_to(disp, m.DOWN, buff=0.15)

        self.say(
            "[And subagents, task graphs, teams and worktrees?] [To the loop, they're just more tools.]",
            m.AnimationGroup(self.new_header("04 · MORE TOOLS", "Teams are tools too"),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.15 * m.RIGHT) for r in rows], lag_ratio=0.15, run_time=2.2)),
            m.AnimationGroup(*[m.Circumscribe(r[2], color=AZURE, buff=0.05) for r in scaling]),
        )
        self.say(
            "[Learn Claude Code's final harness has twenty-seven built-in tools], [and the loop dispatches every one], [through the same line.]",
            m.AnimationGroup(m.FadeIn(total, shift=0.15 * m.UP), m.FadeIn(src)),
            m.FadeIn(disp, shift=0.15 * m.UP),
            m.AnimationGroup(m.Circumscribe(disp, color=SUN, buff=0.06), m.FadeIn(cap)),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ a mature harness

    def harness(self):
        self.next_section("5 · A mature harness", notes="learn-claude-code s20: 'Claude Code's complexity is not \"another agent brain.\" It is the complexity of a mature harness.' and 'The model decides and chooses actions; the harness organizes environment, tools, permissions, memory, teams, and external capabilities.' Facts as of October 2026.")
        q1 = title_text("Claude Code's complexity is not “another agent brain.”", 32, INK)
        q2 = title_text("It is the complexity of a mature harness.", 32, INK)
        glyphs(q2, "mature harness").set_color(LIME)
        quote = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.3).move_to([0, 1.65, 0])
        src = cite("learn-claude-code · s20").next_to(quote, m.DOWN, buff=0.25)

        mbox = card(4.15, 2.35, CORAL, CORAL_FILL)
        mbody = m.VGroup(icon(LLM_ICON, 0.75), text("the model", 28, INK, weight=m.BOLD), text("decides and chooses actions", 22, MUTED)).arrange(m.DOWN, buff=0.16)
        model = m.VGroup(mbox, mbody.move_to(mbox)).move_to([-4.525, -1.45, 0])
        hbox = card(8.4, 2.35, LIME, CARD)
        head = m.VGroup(icon(GEAR, 0.55), text("the harness organizes", 26, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.2)
        items = ["environment", "tools", "permissions", "memory", "teams", "external capabilities"]
        chips = m.VGroup(*[chip(w, LIME, LIME_FILL, width=text(w, 22, weight=m.SEMIBOLD).width + 0.45, height=0.52, size=22) for w in items])
        chips[:3].arrange(m.RIGHT, buff=0.16)
        chips[3:].arrange(m.RIGHT, buff=0.16)
        grid = m.VGroup(chips[:3], chips[3:]).arrange(m.DOWN, buff=0.16)
        hbody = m.VGroup(head, grid).arrange(m.DOWN, buff=0.28)
        harness = m.VGroup(hbox, hbody.move_to(hbox)).move_to([2.4, -1.45, 0])

        self.say(
            "[As Learn Claude Code puts it]: [Claude Code's complexity is not another agent brain.] [It is the complexity of a mature harness.]",
            m.AnimationGroup(self.new_header("05 · THE HARNESS", "Not another brain"), m.FadeIn(src)),
            m.FadeIn(q1, shift=0.15 * m.UP),
            m.FadeIn(q2, shift=0.15 * m.UP),
        )
        self.say(
            "[The model decides and chooses actions.] [The harness organizes everything else]: [environment, tools, permissions, memory, teams.]",
            m.FadeIn(model, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(hbox, shift=0.2 * m.UP), m.FadeIn(head, shift=0.2 * m.UP)),
            m.LaggedStart(*[m.FadeIn(c, shift=0.1 * m.UP) for c in chips], lag_ratio=0.2),
        )
        self.wait(0.4)
        self.clear()

    # ------------------------------------------------------------------ five patterns

    def patterns(self):
        self.next_section("6 · Five patterns", notes="Li, AI Agents in Depth, ch. 1, 'Design patterns that run through the book': Proposer–Reviewer (separate contexts; self-review is unreliable) — Episode 26; Progressive Disclosure (a searchable catalogue first, details on demand) — Episodes 10, 16; Append-only (never revised in place) — Episodes 2, 7; Boundary Set + Retention Set (validate on what a change should change and on what it must not affect) — Episode 31; Minimal Diff, Reversible (small, with provenance) — Episode 31 and Otto's one-line fix. Facts as of October 2026.")
        width, height = 2.45, 4.85
        specs = [
            ("Proposer–\nreviewer", "one writes,\nanother checks,\nin its own\ncontext", [26]),
            ("Progressive\ndisclosure", "a catalog\nfirst, details\non demand", [10, 16]),
            ("Append-\nonly", "state grows by\nappending,\nnever by\nrewriting", [2, 7]),
            ("Boundary +\nretention", "test what\nshould change,\nand what\nmust not", [31]),
            ("Minimal,\nreversible", "small changes\nthat carry\ntheir source,\neasy to undo", [31]),
        ]
        tiles = m.VGroup()
        for i, (title, desc, eps) in enumerate(specs):
            box = card(width, height, EDGE, CARD, radius=0.2, stroke=2)
            num = title_text(str(i + 1), 30, LIME).move_to(box.get_corner(m.UL) + [0.22, -0.26, 0], aligned_edge=m.UL)
            name = title_text(title, 24, INK).next_to(num, m.DOWN, buff=0.16).align_to(num, m.LEFT)
            pic = self.pattern_pic(i)
            fit = min(1.0, (width - 0.4) / pic.width, 1.0 / pic.height)
            pic.scale(fit).move_to([box.get_x(), box.get_top()[1] - 2.2, 0])
            words = text(desc, 20, MUTED).move_to(box.get_corner(m.UL) + [0.22, -2.88, 0], aligned_edge=m.UL)
            badges = m.VGroup(*[ep_badge(e, 0.34) for e in eps]).arrange(m.RIGHT, buff=0.08)
            badges.move_to(box.get_corner(m.DL) + [0.22, 0.34, 0], aligned_edge=m.LEFT)
            tiles.add(m.VGroup(box, num, name, pic, words, badges))
        tiles.arrange(m.RIGHT, buff=0.24).move_to([0, -0.3, 0])
        src = cite("Li, AI Agents in Depth · ch. 1").next_to(tiles, m.DOWN, buff=0.22).align_to(tiles, m.RIGHT)

        self.say(
            "[The book AI Agents in Depth names five patterns] [that run through the whole book.]",
            m.AnimationGroup(self.new_header("06 · FIVE PATTERNS", "What runs through the book"), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(t[0]) for t in tiles], lag_ratio=0.15),
        )
        lines = [
            "[Proposer and reviewer]: [whoever writes a change is not the one who checks it], [episode twenty-six.]",
            "[Progressive disclosure]: [a short catalog first, the details on demand], [like skills in episode ten.]",
            "[Append-only]: [state grows by appending, and is never rewritten], [like the trajectory in episode two.]",
            "[A boundary set and a retention set]: [test what a change should fix, and what it must not break], [episode thirty-one.]",
            "[And minimal, reversible diffs]: [like Otto's one-line fix.]",
        ]
        for i, (tile, script) in enumerate(zip(tiles, lines)):
            n = script.count("[")
            anims = [
                m.AnimationGroup(m.FadeIn(tile[1], shift=0.1 * m.UP), m.FadeIn(tile[2], shift=0.1 * m.UP), tile[0].animate.set_stroke(LIME, 2.5)),
                m.AnimationGroup(m.FadeIn(tile[3], scale=0.9), m.FadeIn(tile[4], shift=0.1 * m.UP)),
                m.FadeIn(tile[5], shift=0.1 * m.UP),
            ][:n]
            if n == 2:
                anims = [anims[0], m.AnimationGroup(m.FadeIn(tile[3], scale=0.9), m.FadeIn(tile[4], shift=0.1 * m.UP), m.FadeIn(tile[5], shift=0.1 * m.UP))]
            self.say(script, *anims)
        self.wait(0.5)
        self.clear()

    def pattern_pic(self, i):
        if i == 0:
            a, b = otto(0.42), otto(0.42, ORCHID, ORCHID_FILL)
            pair = m.VGroup(a, b).arrange(m.RIGHT, buff=0.75)
            link = arrow(a.get_right(), b.get_left(), MUTED, buff=0.08)
            ok = text("✓", 26, MINT, weight=m.BOLD).next_to(b, m.UP, buff=0.05)
            return m.VGroup(pair, link, ok)
        if i == 1:
            rows = bars([1.5, 1.2, 1.35], MUTED, height=0.08, gap=0.12, stripe=SUN)
            body = card(1.6, 0.55, SUN, SUN_FILL, radius=0.08, stroke=2)
            inner = bars([1.1, 0.9, 1.2], SUN, height=0.06, gap=0.07).move_to(body)
            body = m.VGroup(body, inner).next_to(rows[1], m.DOWN, buff=0.1).align_to(rows, m.LEFT).shift(0.15 * m.RIGHT)
            rows[2].next_to(body, m.DOWN, buff=0.1).align_to(rows, m.LEFT)
            return m.VGroup(rows, body)
        if i == 2:
            colors = [AZURE, CORAL, SUN, MINT, CORAL]
            stack = m.VGroup(*[m.RoundedRectangle(corner_radius=0.04, width=1.6, height=0.17, stroke_width=0, fill_color=c, fill_opacity=0.9) for c in colors])
            stack.arrange(m.DOWN, buff=0.07)
            plus = text("+", 30, INK, weight=m.BOLD).next_to(stack[-1], m.RIGHT, buff=0.15)
            return m.VGroup(stack, plus)
        if i == 3:
            fix = m.VGroup(*[m.Dot(radius=0.09, color=SUN) for _ in range(3)]).arrange(m.RIGHT, buff=0.12)
            keep = m.VGroup(*[m.Dot(radius=0.09, color=MINT) for _ in range(6)]).arrange_in_grid(rows=2, buff=0.12)
            labels = m.VGroup(text("fix", 20, SUN), text("keep", 20, MINT))
            left = m.VGroup(fix, labels[0]).arrange(m.DOWN, buff=0.12)
            right = m.VGroup(keep, labels[1]).arrange(m.DOWN, buff=0.12)
            return m.VGroup(left, right).arrange(m.RIGHT, buff=0.35, aligned_edge=m.DOWN)
        rows = m.VGroup()
        bodies = [(code(w, 18, c), f) for w, c, f in (("len(items) - 1", ROSE, ROSE_FILL), ("len(items)", MINT, MINT_FILL))]
        w = max(b.width for b, _ in bodies) + 0.26
        for body, fill in bodies:
            box = m.Rectangle(width=w, height=0.38, stroke_width=0, fill_color=fill, fill_opacity=1)
            rows.add(m.VGroup(box, body.move_to(box).align_to(box, m.LEFT).shift(0.13 * m.RIGHT)))
        rows.arrange(m.DOWN, buff=0.04)
        undo = m.CurvedArrow(rows.get_corner(m.UR) + [-0.3, 0.12, 0], rows.get_corner(m.UL) + [0.3, 0.12, 0], angle=0.8, color=MUTED, stroke_width=3, tip_length=0.12)
        return m.VGroup(rows, undo)

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes=f"One loop (Episode 2's ten lines, unchanged), {len(LAYER_KEYS)} layers of harness, the book's five patterns.")
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 72, c) for w, c in (("Loop", CORAL), ("Layers", SUN), ("Patterns", MINT))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("10 lines · Episode 2", f"{len(LAYER_KEYS)} layers · the harness", "5 patterns · the book")])
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
        tagline = text("The loop never changed. Everything we added was harness.", 30, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up the whole course]: [one loop], [many layers], [five patterns.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[The loop never changed. Everything we added was harness.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)

    # ------------------------------------------------------------------ the end of the course

    def series_outro(self):
        self.next_section("The end", notes="The course's closing card: Agents, Animated, from one loop to a complete agent; the six modules in their colours.")
        if self.head:
            self.play(self.drop_header(), run_time=0.6)
        halo = glow(CORAL, radius=4.3).move_to(0.2 * m.UP)
        brand = title_text("Agents, Animated", 66, INK, gradient=(CORAL, SUN))
        sub = text("from one loop to a complete agent", 30, MUTED)
        top = m.VGroup(brand, sub).arrange(m.DOWN, buff=0.3)
        mods = m.VGroup(*[pill(f"{k} · {name}", color, fill, 20, mono=True) for k, (name, color, fill) in MODULES.items()])
        mods[:3].arrange(m.RIGHT, buff=0.2)
        mods[3:].arrange(m.RIGHT, buff=0.2)
        grid = m.VGroup(mods[:3], mods[3:]).arrange(m.DOWN, buff=0.2)
        thanks = title_text("Thanks for watching!", 40, INK)
        mascot = otto(0.6)
        foot = m.VGroup(mascot, thanks).arrange(m.RIGHT, buff=0.4)
        group = m.VGroup(top, grid, foot).arrange(m.DOWN, buff=0.55).move_to(0.1 * m.UP)

        self.say(
            "[That's Agents, Animated]: [from one loop to a complete agent.] [Thanks for watching!]",
            m.AnimationGroup(m.FadeIn(halo, scale=0.6), m.FadeIn(brand, shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(sub, shift=0.1 * m.UP), m.LaggedStart(*[m.FadeIn(p, shift=0.1 * m.UP) for p in mods], lag_ratio=0.15)),
            m.AnimationGroup(m.FadeIn(mascot, shift=0.2 * m.UP), m.Write(thanks)),
        )
        self.wait(1.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, group)], run_time=1.0)
        self.wait(0.5)
