"""Agents, Animated · Episode 17 · Permissions & Sandboxes  (Module 4 · Tools & Safety)

How to stop an agent from doing damage. Running example: Otto is asked to clear tinyshop's build cache
and rerun the tests. It believes its shell is inside tmp/build-cache and asks to run `rm -rf *`, but the
shell is in the project root: the command would wipe the repo, and the loop of Episodes 2 and 3 runs
whatever the model asks for. The episode puts a gate in front of the handler, routes commands through it,
breaks its deny list, replaces strings with rules, draws a sandbox around the shell, adds a second model
as a reviewer, and places it all in three guardrail layers.

The runs below are not drawn by hand: the scene implements learn-claude-code's check_permission() (deny
list → ask → allow), a tiny stand-in shell for tinyshop (rm -rf, pytest -q), Claude Code-style rule
matching (deny, then ask, then allow; compound commands split into parts) and fnmatch for the docs'
Bash(rm *) example, and every decision, output line and count on screen comes from them (asserted).

Arc: hook (rm -rf * in the wrong folder) · 1 the gate (check_permission between tool_use and handler;
denials return a tool_result) · 2 a router (README → allow, rm -rf ./tmp/build-cache → ask, sudo rm -rf /
→ deny) · 3 Otto asks (no → "Permission denied." → Otto adapts → yes) · 4 rules (blacklist bypassed;
Claude Code's allow/ask/deny rules, parsed commands, rule sources) · 5 sandbox (rules match text; the
recipe; the wrong-folder delete fails; process < container < microVM; a venv is not a sandbox) ·
6 reviewer (a second model, another family, sees only the call) · 7 three layers (context, execution,
data) · recap (Gate · Sandbox · Reviewer).

Colours (series roles): model CORAL, tool call SUN, tool result MINT, user AZURE; good MINT, bad ROSE;
decisions deny ROSE · ask SUN · allow MINT; the harness neutral (EDGE/MUTED); the reviewer (a model of
another family) LIME; headers MINT (Module 4).

Facts (checked October 2026):
- learn-claude-code s03 (learn.shareai.run/en/s03): check_permission() runs before execution: a
  DENY_LIST ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"] → block; rules
  (writes outside WORKDIR, bash containing "rm ", "> /etc/", "chmod 777") → ask the user
  (input("Allow? [y/N]")); else allow. Denials still return a tool_result ("Permission denied."). The
  simulator: read README → allow; rm -rf ./tmp/build-cache → ask; sudo rm -rf / → deny; "Permission is
  a router." The page's own pitfall: "simple string matching is not a reliable security mechanism —
  command variants and shell expansion can bypass it." Its "Dive into CC source" (the site's reading of
  Claude Code's source, unverified): allow / deny / ask / passthrough; eight rule sources (user, project
  and local settings, flag, enterprise policy, CLI, command, session).
- Claude Code docs, "Configure permissions" (code.claude.com/docs/en/permissions): "Rules are evaluated in
  order: deny, then ask, then allow. The first match in that order determines the outcome"; "Claude Code
  is aware of shell operators, so a rule like Bash(safe-cmd *) won't give it permission to run the
  command safe-cmd && other-cmd"; "If a tool is denied at any level, no other level can allow it";
  "What a Bash rule doesn't match": Bash(rm *) stops `rm -rf build/` but not `/bin/rm -rf build/` or
  `bash -c 'rm -rf build/'`; "For filesystem and network enforcement that doesn't depend on the command
  text, use sandboxing."
- Li, AI Agents in Depth, ch. 5 (security for coding agents): blacklists are bypassed by `$(echo rm) -rf /`
  and `find / -name '*.log' -exec rm {} \\;` (parse commands semantically instead); the sandbox recipe:
  no network by default (allowlist proxy), source read-only with a separate writable workspace,
  credentials "not mounted into the sandbox at all", quotas and timeouts.
- Li, ch. 4 (execution tools): process < container ("share the kernel with the host machine") < microVM
  (Firecracker, "an independent kernel"); a venv "only isolates package dependencies and places no
  security constraints on files, networking, or processes"; Proposer–Reviewer pre-approval, "similar to
  the dual-signature system in banking", "the proposing and approving models should come from different
  families"; the sidecar reads only structured fields ("closes this rhetorical channel"), "normally
  completes in a few hundred milliseconds in parallel with streaming output", and after repeated
  rejections "should fall back to asking the user to decide manually".
- Li, ch. 1 (types of guardrails): three layers "by how hard they are to bypass": context (classifiers;
  "an Agent cannot reliably determine whether prompt injection has already compromised its own
  context"), execution (rate tools by "reversibility, privilege level, and financial impact"; review "must
  be performed by a mechanism outside the context"), data (row-level security: "even if the prompt
  injection succeeds… the unauthorized operation is still rejected at the data layer").
Illustrative (for teaching): tinyshop's files and its build cache, the folder /home/dev/tinyshop, Otto's
belief about its folder, the user's answers (no, then yes), Otto's reply after the denial, the
settings.json rules, the smuggled `pytest -q && rm -rf tests`, the reviewer's reason, the three ✗ marks.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import fnmatch
import os
import re
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

REF = (SERIES / "harness_reference.py").read_text()
assert "output = run_tool(block)" in REF  # Episodes 2–3: the loop runs whatever the model asks for
assert 'results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})' in REF
assert '"1 failed, 12 passed"' in REF  # tinyshop's suite: 13 tests (all pass since Episode 3's fix)
TESTS = 1 + 12
PASSED = f"{TESTS} passed"

# ---------------------------------------------------------------------------- tinyshop (illustrative)

TASK = "Clear the build cache, then rerun the tests."
WORKDIR = "/home/dev/tinyshop"
REPO = ["cart.py", "checkout.py", "tests/test_cart.py", "tests/test_checkout.py", "tmp/build-cache/cart.pyc"]
SOURCE = ["cart.py", "checkout.py", "tests"]


def top_level(files):
    return sorted({f.split("/")[0] for f in files})


def shell(command, files, writable=None):
    """A stand-in for run_bash on tinyshop (rm -rf, pytest -q): returns (output, files left).
    `writable` = the top-level names a sandbox lets the command change (None: everything)."""
    if command.startswith("rm -rf "):
        target = command[len("rm -rf "):]
        doomed = top_level(files) if target == "*" else [target.removeprefix("./").rstrip("/")]
        errors, kept = [], list(files)
        for name in doomed:
            if writable is not None and name.split("/")[0] not in writable:
                errors.append(f"rm: cannot remove '{name}': Read-only file system")
                continue
            kept = [f for f in kept if not (f == name or f.startswith(name + "/"))]
        return "\n".join(errors), kept
    if command == "pytest -q":
        ok = {"cart.py", "checkout.py"} <= set(files) and any(f.startswith("tests/") for f in files)
        return (PASSED if ok else "ERROR: file or directory not found: tests"), files
    raise ValueError(command)


WIPE = "rm -rf *"
_, LEFT_AFTER_WIPE = shell(WIPE, REPO)
assert LEFT_AFTER_WIPE == [] and top_level(REPO) == ["cart.py", "checkout.py", "tests", "tmp"]  # "the whole repo"

# ---------------------------------------------------------------------------- the gate (learn-claude-code s03)

DENY_LIST = ["rm -rf /", "sudo", "shutdown", "reboot", "mkfs", "dd if=", "> /dev/sda"]
ASK_PATTERNS = ["rm ", "> /etc/", "chmod 777"]


def call(name, **args):
    return NS(type="tool_use", name=name, input=args)


def writes_outside(block):
    if block.name not in ("edit_file", "write_file"):
        return False
    path = os.path.normpath(os.path.join(WORKDIR, block.input["path"]))
    return not path.startswith(WORKDIR + "/")


def on_deny_list(command):
    return [bad for bad in DENY_LIST if bad in command]


def route(block):
    """check_permission's three checks, before the user is asked: "deny", "ask" or "allow"."""
    command = block.input.get("command", "")
    if on_deny_list(command):
        return "deny"
    if any(p in command for p in ASK_PATTERNS) or writes_outside(block):
        return "ask"
    return "allow"


def check_permission(block, answer=None):
    decision = route(block)
    if decision == "ask":
        return "allow" if answer == "y" else "deny"
    return decision


SIM = [  # learn-claude-code s03's simulator: (label on screen, call, expected route)
    ('read_file("README.md")', call("read_file", path="README.md"), "allow"),
    ("rm -rf ./tmp/build-cache", call("bash", command="rm -rf ./tmp/build-cache"), "ask"),
    ("sudo rm -rf /", call("bash", command="sudo rm -rf /"), "deny"),
]
for _label, _block, _want in SIM:
    assert route(_block) == _want, (_label, route(_block))
assert writes_outside(call("edit_file", path="../../etc/hosts")) and not writes_outside(call("edit_file", path="cart.py"))
assert route(call("bash", command=WIPE)) == "ask"  # the hook's command: the gate asks first

# Otto's run behind the gate: it asks for rm -rf * (you say no), then for the cache only (yes), then tests
ANSWERS = {WIPE: "n", "rm -rf ./tmp/build-cache": "y"}
PLAN = [WIPE, "rm -rf ./tmp/build-cache", "pytest -q"]


def otto_run():
    files, log = list(REPO), []
    for command in PLAN:
        block = call("bash", command=command)
        first = route(block)
        final = check_permission(block, ANSWERS.get(command))
        if final == "allow":
            output, files = shell(command, files)
        else:
            output = "Permission denied."
        log.append(NS(command=command, route=first, final=final, output=output))
    return log, files


RUN, AFTER = otto_run()
assert [r.route for r in RUN] == ["ask", "ask", "allow"] and [r.final for r in RUN] == ["deny", "allow", "allow"]
assert [r.output for r in RUN] == ["Permission denied.", "", PASSED]
assert AFTER == [f for f in REPO if not f.startswith("tmp/build-cache/")]  # only the cache is gone

# ---------------------------------------------------------------------------- the deny list, bypassed (Li ch. 5)

ROOT_CMD = "sudo rm -rf /"
SUBST = "$(echo rm) -rf /"
FIND = "find / -name '*.log' -exec rm {} \\;"
assert on_deny_list(ROOT_CMD) == ["rm -rf /", "sudo"]
assert not on_deny_list(SUBST) and not on_deny_list(FIND)  # both slip past the deny list
EXPANDED = re.sub(r"\$\(echo (\w+)\)", r"\1", SUBST)  # what the shell actually runs
assert EXPANDED == "rm -rf /" and on_deny_list(EXPANDED) == ["rm -rf /"]
assert route(call("bash", command=SUBST)) == "allow"  # it even gets through without a question

# ---------------------------------------------------------------------------- Claude Code-style rules (docs)

CC_RULES = {"deny": ["Bash(sudo *)", "Read(./.env)"], "ask": ["Bash(rm *)"], "allow": ["Bash(pytest *)"]}
SEPARATORS = r"\s*(?:&&|\|\||;|\|)\s*"


def subcommands(command):
    return [part for part in re.split(SEPARATORS, command) if part]


def rule_matches(rule, tool, text_):
    found = re.fullmatch(r"(\w+)\((.*)\)", rule)
    return found.group(1) == tool and fnmatch.fnmatchcase(text_, found.group(2))


def cc_decide(tool, command, split=True):
    """deny, then ask, then allow; the first match wins; deny/ask if ANY part matches, allow if ALL do."""
    parts = subcommands(command) if (tool == "Bash" and split) else [command]
    for behaviour in ("deny", "ask"):
        if any(rule_matches(r, tool, p) for r in CC_RULES[behaviour] for p in parts):
            return behaviour
    if all(any(rule_matches(r, tool, p) for r in CC_RULES["allow"]) for p in parts):
        return "allow"
    return "ask"  # no rule: a shell command needs approval by default


SMUGGLE = "pytest -q && rm -rf tests"
PARTS = subcommands(SMUGGLE)
assert PARTS == ["pytest -q", "rm -rf tests"]
assert cc_decide("Bash", SMUGGLE, split=False) == "allow"  # as one string, the allow rule matches
assert [cc_decide("Bash", p) for p in PARTS] == ["allow", "ask"] and cc_decide("Bash", SMUGGLE) == "ask"
assert cc_decide("Bash", ROOT_CMD) == "deny" and cc_decide("Read", "./.env") == "deny"
# the docs' table: Bash(rm *) matches the command as written, not the same program in another form
RM_RULE = "Bash(rm *)"
DOC_ROWS = ["rm -rf build/", "/bin/rm -rf build/", "bash -c 'rm -rf build/'"]
DOC_HITS = [rule_matches(RM_RULE, "Bash", c) for c in DOC_ROWS]
assert DOC_HITS == [True, False, False]
SOURCES = ["user", "project", "local", "enterprise policy", "CLI", "session"]  # 6 of the 8 (learn-claude-code s03)

# ---------------------------------------------------------------------------- the sandbox (Li ch. 5)

SANDBOX_OUT, SANDBOX_LEFT = shell(WIPE, REPO, writable={"tmp"})
SANDBOX_ERRORS = SANDBOX_OUT.split("\n")
assert SANDBOX_ERRORS == [f"rm: cannot remove '{n}': Read-only file system" for n in SOURCE]
assert SANDBOX_LEFT == [f for f in REPO if not f.startswith("tmp/")]  # only the writable cache went

# ---------------------------------------------------------------------------- helpers

PANEL = "#18150F"
SHIELD = "ep17-shield"
GEAR = recolored("gear", MUTED, CARD_HI, prefix="ep17")
LOCK = recolored("lock", MINT, MINT_FILL, prefix="ep17")
CLOCK = recolored("clock", MINT, MINT_FILL, prefix="ep17")
REVIEWER = recolored("llm", LIME, LIME_FILL, prefix="ep17")
DECIDE = {"deny": (ROSE, ROSE_FILL), "ask": (SUN, SUN_FILL), "allow": (MINT, MINT_FILL)}
JSON_T2C = {r"\"[^\"\n]*\"(?=\s*:)": AZURE, r"\"[^\"\n]*\"": MINT, r"\b-?\d+(\.\d+)?\b": SUN, r"\b(true|false|null)\b": ORCHID}


def cite(s):
    return text(s, 20, MUTED)


def decision(d, size=22):
    color, fill = DECIDE[d]
    return pill(d, color, fill, size, mono=True)


def cmd_chip(s, width=None, color=SUN, fill=CARD, size=20, height=0.62):
    words = code(s, size, INK)
    box = card(width or words.width + 0.5, height, color, fill, radius=0.14, stroke=2)
    return m.VGroup(box, words.move_to(box))


def boxed(body, title, accent, width=None):
    """A Text `body` in a card with a bold caption: VGroup(box, body, cap)."""
    cap = code(title, 20, accent, weight=m.BOLD)
    top = cap.height + 0.24
    w = max(width or 0, body.width + 0.7, cap.width + 0.7)
    box = card(w, body.height + 0.55 + top, accent, CARD, radius=0.16, stroke=2)
    body.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT + top / 2 * m.DOWN)
    cap.align_to(box, m.UL).shift(0.35 * m.RIGHT + 0.17 * m.DOWN)
    body.set_z_index(2)
    return m.VGroup(box, body, cap)


def quote_card(lines, source, width=None, size=24, accent=EDGE):
    words = m.VGroup(*[text(s, size, INK) for s in lines]).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
    src = cite(source)
    inner = m.VGroup(words, src).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
    box = card(max(width or 0, inner.width + 0.7), inner.height + 0.5, accent, CARD, radius=0.18, stroke=2)
    bar = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=box.height - 0.3, stroke_width=0,
                             fill_color=accent if accent != EDGE else MUTED, fill_opacity=1)
    inner.move_to(box).align_to(box, m.LEFT).shift(0.38 * m.RIGHT)
    bar.move_to(box).align_to(box, m.LEFT).shift(0.16 * m.RIGHT)
    return m.VGroup(box, bar, words, src)


def gutter(line, box, color):
    """A bar in a code card's left margin, beside a line (or lines) of code."""
    return m.RoundedRectangle(corner_radius=0.03, width=0.08, height=line.height + 0.12, stroke_width=0,
                              fill_color=color, fill_opacity=1).move_to([box.get_left()[0] + 0.17, line.get_y(), 0])


def band(line, box, color, opacity=0.18):
    """A tinted bar behind a line (or lines) of code, under the text."""
    rect = m.Rectangle(width=box.width - 0.2, height=line.height + 0.14, stroke_width=0, fill_color=color,
                       fill_opacity=opacity)
    return rect.move_to([box.get_x(), line.get_y(), 0]).set_z_index(1)


def node_card(pic, label, sub, color, fill, width=3.0, height=2.3, pic_h=0.9):
    box = card(width, height, color, fill)
    parts = [icon(pic, pic_h), text(label, 26, INK, weight=m.BOLD)]
    if sub:
        parts.append(text(sub, 20, MUTED))
    body = m.VGroup(*parts).arrange(m.DOWN, buff=0.16).move_to(box)
    return m.VGroup(box, body)


def tile(pic, title, sub, width=3.15, height=1.55, accent=EDGE):
    """A recipe item: icon, bold title, muted line, centred in a card."""
    box = card(width, height, accent, CARD, radius=0.16, stroke=2)
    body = m.VGroup(icon(pic, 0.5), text(title, 22, INK, weight=m.BOLD), text(sub, 20, MUTED)).arrange(m.DOWN, buff=0.1)
    assert body.width < width - 0.25, (title, body.width)
    return m.VGroup(box, body.move_to(box))


def dashed_box(width, height, color, dashes=70):
    frame = m.RoundedRectangle(corner_radius=0.3, width=width, height=height, stroke_color=color, stroke_width=3,
                               fill_opacity=0)
    return m.DashedVMobject(frame, num_dashes=dashes)


def spoken(n):
    return say_number(n)


assert spoken(TESTS) == "thirteen"

# ---------------------------------------------------------------------------- the episode


class Ep17PermissionsSandboxes(Episode):
    number = 17

    def lesson(self):
        self.hook()
        self.gate()
        self.router()
        self.otto_asks()
        self.rules()
        self.sandbox()
        self.reviewer()
        self.layers()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section(
            "Hook",
            notes=f"Otto (illustrative): '{TASK}' It believes its shell is in tmp/build-cache and asks for "
            f"`{WIPE}`, but the shell is in the project root {WORKDIR}: `*` matches {', '.join(top_level(REPO))}, "
            "the whole repo (computed by the stand-in shell). harness_reference.py's loop runs `output = "
            "run_tool(block)` for every tool_use: nothing checks the call (Episodes 2–3).",
        )
        ask = message("user", TASK, width=6.4, size=22).move_to([-3.4, 1.85, 0])
        bot = otto(1.0).move_to([-5.65, -0.3, 0])
        thought = bubble(text("I'm in tmp/build-cache.", 22, INK), CORAL, CORAL_FILL, tail="left")
        thought.move_to([-2.85, 0.62, 0])
        call_src = '{"name": "bash",\n "input": {"command": "' + WIPE + '"}}'
        request = boxed(code(call_src, 20, INK, t2c=JSON_T2C), "tool_use", SUN).move_to([-1.72, -1.25, 0])
        assert request.get_left()[0] > bot.get_right()[0] + 0.1, (request.get_left(), bot.get_right())
        wipe_glyphs = glyphs(request[1], WIPE)

        entries = ["tinyshop/", "tinyshop/cart.py", "tinyshop/checkout.py", "tinyshop/tests/", "tinyshop/tmp/",
                   "tinyshop/tmp/build-cache/"]
        tree = file_tree(entries, size=22, highlight=("build-cache/",))
        tree.move_to([0, 0.45, 0]).align_to([1.48, 0, 0], m.LEFT)
        assert tree.get_left()[0] > request.get_right()[0] + 0.2
        think_mark = text("← Otto thinks", 22, CORAL, weight=m.SEMIBOLD).next_to(tree[5], m.RIGHT, buff=0.25)
        shell_mark = text("← the shell is here", 22, ROSE, weight=m.SEMIBOLD).next_to(tree[0], m.RIGHT, buff=0.3)
        gone = pill("the whole repo, gone", ROSE, ROSE_FILL, 22).next_to(tree, m.DOWN, buff=0.4).align_to(tree, m.LEFT)
        assert max(think_mark.get_right()[0], shell_mark.get_right()[0]) < 6.6, (think_mark.get_right(), shell_mark.get_right())

        loop_src = "output = run_tool(block)  # runs anything"
        loop = code_block(loop_src, 20).move_to([-2.9, -3.0, 0])
        loop.align_to([0, -3.45, 0], m.DOWN)
        assert loop.get_top()[1] < request.get_bottom()[1] - 0.15
        loop_line = glyphs(loop[1], "output = run_tool(block)")
        loop_mark = gutter(loop_line, loop[0], ROSE)

        q1 = title_text("How do you stop an agent", 60)
        q2 = title_text("from doing damage?", 60, MINT)
        question = m.VGroup(q1, q2).arrange(m.DOWN, buff=0.35).move_to([0, 0.2, 0])

        self.say(
            "[Otto has a small job]: [clear the build cache], [then run the tests again.]",
            m.AnimationGroup(self.new_header("17 · permissions", "One command from disaster"), m.FadeIn(bot, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(ask, shift=0.2 * m.DOWN), m.FadeIn(tree, shift=0.2 * m.LEFT)),
            m.Circumscribe(tree[5], color=SUN, buff=0.06),
        )
        self.say(
            "[Otto thinks its shell is inside the cache folder], [so it asks to delete everything there.]",
            m.AnimationGroup(m.FadeIn(thought, shift=0.2 * m.UP), m.FadeIn(think_mark, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(request, shift=0.2 * m.UP), m.Circumscribe(wipe_glyphs, color=SUN, buff=0.05)),
        )
        self.say(
            "[But the shell is in the project's root folder.] [The command would wipe the whole repo]: [the cart, the checkout, the tests.]",
            m.AnimationGroup(m.FadeIn(shell_mark, shift=0.2 * m.LEFT), think_mark.animate.set_opacity(0.35),
                             thought.animate.set_opacity(0.35)),
            m.LaggedStart(*[tree[i].animate.set_color(ROSE) for i in range(1, 6)], lag_ratio=0.25),
            m.FadeIn(gone, shift=0.15 * m.UP),
        )
        self.say(
            "[And nothing in our loop would stop it]: [the harness runs whatever the model asks for.]",
            m.FadeIn(loop, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(loop_mark), m.Circumscribe(loop_line, color=ROSE, buff=0.05)),
        )
        self.say(
            "[So how do you stop an agent] [from doing damage?]",
            m.AnimationGroup(
                m.FadeOut(m.VGroup(ask, bot, thought, request, tree, think_mark, shell_mark, gone, loop, loop_mark)),
                m.FadeIn(q1, shift=0.3 * m.UP),
            ),
            m.FadeIn(q2, shift=0.3 * m.UP),
        )
        self.wait(0.6)
        self.play(m.FadeOut(question, shift=0.3 * m.UP), run_time=0.8)

    # ------------------------------------------------------------------ 1 · the gate

    def gate(self):
        self.next_section(
            "1 · The gate",
            notes="learn-claude-code s03: check_permission() runs between the model's tool_use and the handler, "
            "and answers deny, ask or allow. In the loop it is one new check before run_tool (harness_reference "
            "names). A denial still returns a tool_result, 'Permission denied.', so the model learns what happened "
            "and can adapt (every tool_use needs its tool_result anyway).",
        )
        bot = otto(0.9).move_to([-5.95, 1.5, 0])
        req = m.VGroup(code("bash", 20, SUN, weight=m.BOLD), code(WIPE, 20, INK)).arrange(m.RIGHT, buff=0.25)
        req_box = card(req.width + 0.55, 0.66, SUN, CARD, radius=0.14, stroke=2)
        request = m.VGroup(req_box, req.move_to(req_box)).move_to([-3.75, 1.5, 0])
        gate_body = m.VGroup(icon(SHIELD, 0.62), code("check_permission", 20, INK, weight=m.BOLD)).arrange(m.RIGHT, buff=0.25)
        gate_box = card(gate_body.width + 0.6, 1.2, MINT, MINT_FILL, radius=0.2, stroke=2.5)
        gate = m.VGroup(gate_box, gate_body.move_to(gate_box)).move_to([0.6, 1.5, 0])
        run_body = m.VGroup(icon(GEAR, 0.5), code("run_tool()", 20, INK)).arrange(m.RIGHT, buff=0.22)
        run_box = card(run_body.width + 0.55, 1.0, MUTED, CARD, radius=0.18, stroke=2)
        runner = m.VGroup(run_box, run_body.move_to(run_box)).move_to([5.1, 1.5, 0])
        assert runner.get_right()[0] <= 6.6 and gate.get_left()[0] > request.get_right()[0] + 0.7
        straight = arrow(request.get_right(), runner.get_left(), SUN)
        a_in = arrow(request.get_right(), gate.get_left(), SUN)
        a_out = arrow(gate.get_right(), runner.get_left(), MINT)
        pills = m.VGroup(*[decision(d) for d in ("deny", "ask", "allow")]).arrange(m.RIGHT, buff=0.25)
        pills.move_to([gate.get_x(), 0.3, 0])

        loop_src = ('if block.type == "tool_use":\n'
                    '    if check_permission(block) == "allow":\n'
                    '        output = run_tool(block)\n'
                    '    else:\n'
                    '        output = "Permission denied."\n'
                    '    results.append({..., "content": output})')
        loop = code_block(loop_src, 20, title="agent_loop(), with the gate").move_to([0.1, -1.95, 0])
        loop.align_to([0, -3.45, 0], m.DOWN)
        check_line = glyphs(loop[1], 'if check_permission(block) == "allow":')
        else_lines = m.VGroup(glyphs(loop[1], "else:"), glyphs(loop[1], 'output = "Permission denied."'))
        mark_check = gutter(check_line, loop[0], MINT)
        band_check = band(check_line, loop[0], MINT)
        band_else = band(else_lines, loop[0], ROSE)
        mark_else = gutter(else_lines, loop[0], ROSE)

        res_words = code('"Permission denied."', 20, MINT)
        res_box = card(res_words.width + 0.5, 0.62, MINT, CARD, radius=0.14, stroke=2)
        res = m.VGroup(res_box, res_words.move_to(res_box)).move_to([-3.75, 0.25, 0])
        res_cap = code("tool_result", 20, MINT, weight=m.BOLD).next_to(res, m.UP, buff=0.1).align_to(res, m.RIGHT)
        assert pills.get_left()[0] > res.get_right()[0] + 0.5, (pills.get_left(), res.get_right())
        back = arrow(pills[0].get_left(), res.get_right(), ROSE, buff=0.08)
        to_bot = arrow(res.get_left() + 0.05 * m.UP, bot.get_bottom(), MINT, buff=0.1)

        self.say(
            "[A tool call travels from the model] [straight to the code that runs it.] [The fix is one function in between]: [check permission.]",
            m.AnimationGroup(self.new_header("01 · the gate", "A check before every call"),
                             m.FadeIn(bot, shift=0.2 * m.UP), m.FadeIn(request, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(straight), m.FadeIn(runner, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeOut(straight), m.GrowArrow(a_in), m.GrowArrow(a_out)),
            m.FadeIn(gate, scale=0.8),
        )
        self.say(
            "[Every call passes through it], [and it gives one of three answers]: [deny, ask, or allow.]",
            m.Succession(pulse(a_in, SUN), m.Circumscribe(gate, color=MINT, buff=0.06)),
            m.Wait(0.2),
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.DOWN) for p in pills], lag_ratio=0.35),
        )
        self.say(
            "[In the loop, that's one new check] [before the handler runs.]",
            m.FadeIn(loop, shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(band_check), m.FadeIn(mark_check), m.Circumscribe(check_line, color=MINT, buff=0.05)),
        )
        self.say(
            "[A refusal still comes back as a tool result], [so the model learns what happened], [and can try another way.]",
            m.AnimationGroup(m.FadeOut(band_check), m.FadeIn(band_else), m.FadeIn(mark_else),
                             m.Circumscribe(pills[0], color=ROSE, buff=0.06)),
            m.AnimationGroup(m.GrowArrow(back), m.FadeIn(res, shift=0.2 * m.LEFT), m.FadeIn(res_cap)),
            m.AnimationGroup(m.GrowArrow(to_bot), m.Succession(pulse(to_bot, MINT), m.Circumscribe(bot, color=CORAL, buff=0.08))),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 2 · a router

    def router(self):
        self.next_section(
            "2 · A router",
            notes="learn-claude-code s03's gate, shown simplified (its DENY_LIST defined above the function; its risky "
            "patterns 'rm ', '> /etc/', 'chmod 777' shown as 'rm '): deny list → 'deny'; risky patterns or writes "
            "outside the workspace → ask the user (input('Allow? [y/N]')); else 'allow'. Its simulator, run through "
            "the scene's implementation: read README → allow; rm -rf ./tmp/build-cache → ask; sudo rm -rf / → deny. "
            "'Permission is a router.'",
        )
        src = ('def check_permission(block):\n'
               '    cmd = block.input.get("command", "")\n'
               '    if any(bad in cmd for bad in DENY_LIST):\n'
               '        return "deny"\n'
               '    if "rm " in cmd or writes_outside(block):\n'
               '        return ask_user(block)\n'
               '    return "allow"')
        assert "rm " in ASK_PATTERNS
        gate = code_block(src, 20, title="the gate · learn-claude-code s03")
        gate.align_to([-6.6, 2.42, 0], m.UL)
        box, body = gate[0], gate[1]
        body.set_z_index(2)
        checks = {
            "deny": m.VGroup(glyphs(body, "if any(bad in cmd for bad in DENY_LIST):"), glyphs(body, 'return "deny"')),
            "ask": m.VGroup(glyphs(body, 'if "rm " in cmd or writes_outside(block):'), glyphs(body, "return ask_user(block)")),
            "allow": glyphs(body, 'return "allow"'),
        }
        bands = {d: band(checks[d], box, DECIDE[d][0]) for d in checks}
        cursor = m.Triangle(fill_color=INK, fill_opacity=1, stroke_width=0).scale(0.09).rotate(-np.pi / 2)

        def at(d):
            first = checks[d][0] if d != "allow" else checks[d]
            return [box.get_left()[0] + 0.17, first.get_y(), 0]

        cursor.move_to(at("deny"))

        chip_w, col_w = 4.0, 4.25
        cols, slots = {}, {}
        for d, x in (("deny", -4.45), ("ask", 0.0), ("allow", 4.45)):
            c_box = card(col_w, 2.1, DECIDE[d][0], CARD, radius=0.2, stroke=2).move_to([x, -2.3, 0])
            p = decision(d).align_to(c_box, m.UL).shift(0.22 * m.RIGHT + 0.2 * m.DOWN)
            dest = text({"deny": "blocked", "ask": "you decide", "allow": "runs right away"}[d], 22, MUTED).next_to(p, m.RIGHT, buff=0.3)
            cols[d] = m.VGroup(c_box, p, dest)
            slots[d] = [x, c_box.get_bottom()[1] + 0.2 + 0.31, 0]
        assert gate.get_bottom()[1] > cols["deny"][0].get_top()[1] + 0.3, (gate.get_bottom(), cols["deny"].get_top())
        slot_in = [3.85, 1.7, 0]
        assert gate.get_right()[0] < slot_in[0] - chip_w / 2 - 0.3, gate.get_right()
        chips = [cmd_chip(label, chip_w).move_to(slot_in) for label, _, _ in SIM]
        inbox = m.DashedVMobject(m.RoundedRectangle(corner_radius=0.18, width=chip_w + 0.36, height=0.98, stroke_color=EDGE,
                                                    stroke_width=2.5, fill_opacity=0), num_dashes=40).move_to(slot_in)
        in_cap = text("next tool call", 20, MUTED).next_to(inbox, m.UP, buff=0.12)
        quote = quote_card(["“Permission is a router.”"], "learn-claude-code · s03", accent=MINT).move_to([3.85, 0.35, 0])

        def scan(to):  # the cursor walks the checks in order until one answers
            order = ["deny", "ask", "allow"]
            steps = [cursor.animate(run_time=0.35).move_to(at(d)) for d in order[: order.index(to) + 1]]
            return m.Succession(*steps)

        self.say(
            "[Learn Claude Code's gate checks in order]: [a deny list first], [then risky patterns that need your approval], [and everything else is allowed.]",
            m.AnimationGroup(self.new_header("02 · a router", "One gate, three exits"), m.FadeIn(gate, shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeIn(bands["deny"]), m.FadeIn(cols["deny"], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeOut(bands["deny"]), m.FadeIn(bands["ask"]), m.FadeIn(cols["ask"], shift=0.2 * m.UP)),
            m.AnimationGroup(m.FadeOut(bands["ask"]), m.FadeIn(bands["allow"]), m.FadeIn(cols["allow"], shift=0.2 * m.UP)),
        )
        self.say(
            "[Reading the readme] [matches nothing], [so it runs straight away.]",
            m.AnimationGroup(m.FadeOut(bands["allow"]), m.FadeIn(inbox), m.FadeIn(in_cap), m.FadeIn(chips[0], shift=0.2 * m.DOWN),
                             m.FadeIn(cursor)),
            scan(route(SIM[0][1])),
            m.AnimationGroup(chips[0].animate.move_to(slots[route(SIM[0][1])]), cols["allow"][2].animate.set_color(MINT)),
        )
        self.say(
            "[Deleting the build cache] [contains a remove command], [so the gate stops and asks you first.]",
            m.FadeIn(chips[1], shift=0.2 * m.DOWN),
            m.Succession(cursor.animate(run_time=0.35).move_to(at("deny")), scan(route(SIM[1][1]))),
            m.AnimationGroup(chips[1].animate.move_to(slots[route(SIM[1][1])]), cols["ask"][2].animate.set_color(SUN)),
        )
        self.say(
            "[And anything run as the administrator] [is on the deny list]: [blocked, without even asking.]",
            m.FadeIn(chips[2], shift=0.2 * m.DOWN),
            m.AnimationGroup(cursor.animate(run_time=0.35).move_to(at("deny")), m.Circumscribe(checks["deny"][0], color=ROSE, buff=0.05)),
            m.AnimationGroup(chips[2].animate.move_to(slots[route(SIM[2][1])]), cols["deny"][2].animate.set_color(ROSE)),
        )
        self.say(
            "[One check, three exits]: [permission is a router.]",
            m.AnimationGroup(m.FadeOut(inbox), m.FadeOut(in_cap),
                             m.LaggedStart(*[m.Circumscribe(cols[d][0], color=DECIDE[d][0], buff=0.05) for d in cols], lag_ratio=0.25)),
            m.FadeIn(quote, shift=0.2 * m.UP),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 3 · Otto asks

    def otto_asks(self):
        self.next_section(
            "3 · Otto asks",
            notes=f"Otto's run behind the gate (scene: check_permission + stand-in shell; answers illustrative): "
            f"`{RUN[0].command}` → {RUN[0].route} → you answer n → '{RUN[0].output}' as a tool_result; Otto "
            f"re-plans: `{RUN[1].command}` → {RUN[1].route} → y → runs (no output); `{RUN[2].command}` → "
            f"{RUN[2].route} (no question) → '{RUN[2].output}'. Only tmp/build-cache is gone.",
        )
        ctx = ContextColumn(center=(3.65, -0.55, 0), width=5.8, height=5.5)
        m_user = ctx.message("user", TASK, size=22)
        m_call1 = ctx.message("tool_use", f'bash("{RUN[0].command}")', size=22, mono=True)
        m_res1 = ctx.message("tool_result", RUN[0].output, size=22, mono=True)
        m_res1[1].set_fill(ROSE)
        m_res1[2][0].set_color(ROSE)
        m_think = ctx.message("assistant", "Wrong folder. I'll delete only the cache.", size=22)
        m_call2 = ctx.message("tool_use", f'bash("{RUN[1].command}")', size=22, mono=True)
        m_res2 = ctx.message("tool_result", f'"{RUN[1].output}"  (empty output)', size=22, mono=True)
        m_call3 = ctx.message("tool_use", f'bash("{RUN[2].command}")', size=22, mono=True)
        m_res3 = ctx.message("tool_result", RUN[2].output, size=22, mono=True)
        p_user, p_call1, p_res1, p_think, p_call2, p_res2, p_call3, p_res3 = [
            ctx.push(msg) for msg in (m_user, m_call1, m_res1, m_think, m_call2, m_res2, m_call3, m_res3)
        ]

        d_box = card(5.9, 2.55, SUN, CARD, radius=0.2, stroke=2.5).move_to([-3.55, 0.95, 0])
        d_head = text("Otto wants to run a command", 22, MUTED)
        d_cmd1 = code(RUN[0].command, 24, INK, weight=m.BOLD)
        d_cmd2 = code(RUN[1].command, 24, INK, weight=m.BOLD)
        d_where = code(f"in {WORKDIR}", 20, MUTED)
        d_ask = code("Allow? [y/N]", 22, SUN, weight=m.BOLD)
        d_col = m.VGroup(d_head, d_cmd1, d_where, d_ask).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        d_col.move_to(d_box).align_to(d_box, m.LEFT).shift(0.35 * m.RIGHT)
        d_cmd2.move_to(d_cmd1).align_to(d_cmd1, m.LEFT)
        ans_n = code("n", 22, ROSE, weight=m.BOLD).next_to(d_ask, m.RIGHT, buff=0.3)
        ans_y = code("y", 22, MINT, weight=m.BOLD).next_to(d_ask, m.RIGHT, buff=0.3)
        ans_y.align_to(ans_n, m.DOWN)
        not_run = text("handler not run", 20, ROSE, weight=m.SEMIBOLD).next_to(ans_n, m.RIGHT, buff=0.45)
        dialog = m.VGroup(d_box, d_head, d_cmd1, d_where, d_ask)

        entries = ["tinyshop/", "tinyshop/cart.py", "tinyshop/checkout.py", "tinyshop/tests/", "tinyshop/tmp/",
                   "tinyshop/tmp/build-cache/"]
        tree = file_tree(entries, size=22, highlight=("build-cache/",))
        tree.align_to([-6.3, -0.75, 0], m.UL)
        bot = otto(1.0).move_to([-1.75, -2.05, 0])
        assert d_box.get_right()[0] < ctx.panel.get_left()[0] - 0.3

        self.say(
            "[Back to Otto's command.] [It contains a remove, so the gate pauses] [and shows you exactly what would run, and where.]",
            m.AnimationGroup(self.new_header("03 · Otto asks", "A refusal the model can read"), m.FadeIn(ctx.frame),
                             m.FadeIn(tree), m.FadeIn(bot, shift=0.2 * m.UP), m.LaggedStart(p_user, p_call1, lag_ratio=0.5)),
            m.FadeIn(dialog, shift=0.2 * m.UP),
            m.AnimationGroup(m.Circumscribe(d_cmd1, color=SUN, buff=0.06), m.Circumscribe(d_where, color=SUN, buff=0.06)),
        )
        self.say(
            "[You answer no.] [The handler never runs], [and Otto gets back a tool result]: [permission denied.]",
            m.FadeIn(ans_n, scale=1.3),
            m.FadeIn(not_run, shift=0.15 * m.LEFT),
            p_res1,
            m.Circumscribe(m_res1, color=ROSE, buff=0.04),
        )
        self.say(
            "[Otto reads it, sees it was in the wrong folder], [and asks to delete the cache folder only.]",
            m.AnimationGroup(p_think, m.Circumscribe(bot, color=CORAL, buff=0.08)),
            m.AnimationGroup(p_call2, m.FadeOut(ans_n), m.FadeOut(not_run), m.Transform(d_cmd1, d_cmd2)),
        )
        self.say(
            "[This time you say yes.] [The cache is cleared], [and the test run needs no question at all.]",
            m.FadeIn(ans_y, scale=1.3),
            m.AnimationGroup(p_res2, m.FadeOut(tree[5], shift=0.3 * m.LEFT)),
            m.LaggedStart(p_call3, p_res3, lag_ratio=0.6),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 4 · rules, not strings

    def rules(self):
        self.next_section(
            "4 · Rules, not strings",
            notes=f"Li ch. 5: blacklists are bypassed: `{SUBST}` (the shell expands it to `{EXPANDED}`; the scene's "
            f"gate even allows it without asking) and `{FIND}` (deletes every .log file); neither matches the deny "
            "list. learn-claude-code s03 warns: 'simple string matching is not a reliable security mechanism'. "
            "Claude Code (docs, Oct 2026): allow / ask / deny rules, evaluated deny → ask → allow, first match wins; "
            f"compound commands are split, so `{SMUGGLE}` is not allowed by Bash(pytest *) (scene: as one string "
            f"'{cc_decide('Bash', SMUGGLE, split=False)}', split '{cc_decide('Bash', SMUGGLE)}'). Rule sources "
            "(learn-claude-code's reading of CC source): eight, incl. user, project, local, enterprise policy, CLI, "
            "session; docs: 'If a tool is denied at any level, no other level can allow it.' settings.json rules "
            "illustrative.",
        )
        cap = code("DENY_LIST", 20, MUTED)
        items = m.VGroup(*[cmd_chip(s, code(s, 20).width + 0.36, color=ROSE, height=0.52) for s in DENY_LIST]).arrange(m.RIGHT, buff=0.12)
        strip = m.VGroup(cap, items).arrange(m.RIGHT, buff=0.3).move_to([0, 2.12, 0])
        assert strip.width < 13.2

        def row(cmd, note, good, y):
            box = card(13.2, 1.08, EDGE, CARD, radius=0.18, stroke=2).move_to([0, y, 0])
            words = code(cmd, 22, INK)
            sub = text(note, 20, MUTED)
            left = m.VGroup(words, sub).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            left.move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            verdict = (pill("✓ blocked", MINT, MINT_FILL, 22) if good else pill("✗ slips through", ROSE, ROSE_FILL, 22))
            verdict.move_to(box).align_to(box, m.RIGHT).shift(0.3 * m.LEFT)
            return m.VGroup(box, words, sub, verdict)

        hits = on_deny_list(ROOT_CMD)
        r1 = row(ROOT_CMD, "matches " + " and ".join(f"“{h}”" for h in hits), True, 0.95)
        r2 = row(SUBST, f"the shell still runs: {EXPANDED}", False, -0.27)
        r3 = row(FIND, "deletes every .log file on the disk", False, -1.49)
        paint(r1[1], ROOT_CMD, {re.escape(h): ROSE for h in hits})
        hit_chips = m.VGroup(*[items[DENY_LIST.index(h)] for h in hits])
        quote = quote_card(["“Simple string matching is not a reliable security mechanism.”"],
                           "learn-claude-code · s03", accent=ROSE).move_to([0, -2.8, 0])

        # Claude Code's rules
        settings_src = ('{\n  "permissions": {\n'
                        f'    "deny":  {json.dumps(CC_RULES["deny"])},\n'
                        f'    "ask":   {json.dumps(CC_RULES["ask"])},\n'
                        f'    "allow": {json.dumps(CC_RULES["allow"])}\n'
                        '  }\n}')
        settings = boxed(code(settings_src, 20, INK, t2c=JSON_T2C), "settings.json · Claude Code", MINT)
        settings.align_to([-6.6, 2.42, 0], m.UL)
        order = m.VGroup(decision("deny"), text("→", 26, MUTED), decision("ask"), text("→", 26, MUTED), decision("allow"),
                         text("first match wins", 22, MUTED)).arrange(m.RIGHT, buff=0.2)
        order.next_to(settings, m.DOWN, buff=0.3).align_to(settings, m.LEFT)
        rx = (settings.get_right()[0] + 6.6) / 2
        whole = cmd_chip(SMUGGLE).move_to([rx, 2.0, 0])
        naive_row = m.VGroup(code("Bash(pytest *)", 20, INK), text("→", 24, MUTED),
                             pill("allow ✗", ROSE, ROSE_FILL, 20, mono=True)).arrange(m.RIGHT, buff=0.18)
        naive = m.VGroup(text("read as one string, it would be", 20, MUTED), naive_row).arrange(m.DOWN, buff=0.12).move_to([rx, -1.5, 0])
        parts = m.VGroup(*[cmd_chip(p) for p in PARTS]).arrange(m.RIGHT, buff=0.5).move_to([rx, 1.0, 0])
        assert parts.get_left()[0] > settings.get_right()[0] + 0.3 and whole.get_left()[0] > settings.get_right()[0] + 0.3
        verdicts = m.VGroup(*[decision(cc_decide("Bash", p), 20).next_to(c, m.DOWN, buff=0.22) for p, c in zip(PARTS, parts)])
        overall = m.VGroup(text("the whole command:", 20, MUTED), decision(cc_decide("Bash", SMUGGLE))).arrange(m.RIGHT, buff=0.25)
        overall.move_to([rx, -0.55, 0])
        assert overall.get_top()[1] < verdicts.get_bottom()[1] - 0.2 and naive.get_top()[1] < overall.get_bottom()[1] - 0.2
        split_lines = m.VGroup(*[m.Line(whole.get_bottom() + 0.04 * m.DOWN, c.get_top() + 0.04 * m.UP).set_stroke(MUTED, 2)
                                 for c in parts])
        src_cap = code("sources:", 22, MUTED)
        src_chips = m.VGroup(*[pill(s, MUTED, CARD_HI, 22) for s in SOURCES[:5]])
        more = text("… 8 in all", 22, MUTED)
        sources = m.VGroup(src_cap, *src_chips, more).arrange(m.RIGHT, buff=0.16).move_to([0, -2.45, 0])
        assert sources.width < 13.2
        wins = pill("a deny at any level can't be overridden", MINT, MINT_FILL, 22).move_to([-1.6, -3.15, 0])
        src_cite = cite("Claude Code docs · permissions").next_to(wins, m.RIGHT, buff=0.4)

        self.say(
            "[But the deny list only compares text.] [Spell the same delete another way], [and it slips straight through.]",
            m.AnimationGroup(self.new_header("04 · rules", "Strings are not security"), m.FadeIn(strip, shift=0.2 * m.DOWN),
                             m.FadeIn(r1[0]), m.FadeIn(r1[1]), m.FadeIn(r1[2]), m.FadeIn(r1[3], scale=1.15),
                             m.LaggedStart(*[m.Circumscribe(c, color=ROSE, buff=0.04) for c in hit_chips], lag_ratio=0.4)),
            m.AnimationGroup(m.FadeIn(r2[0]), m.FadeIn(r2[1], shift=0.2 * m.RIGHT), m.FadeIn(r2[2])),
            m.FadeIn(r2[3], scale=1.15),
        )
        self.say(
            "[Or let find remove the files one by one], [across the whole disk.] [No match either.]",
            m.AnimationGroup(m.FadeIn(r3[0]), m.FadeIn(r3[1], shift=0.2 * m.RIGHT)),
            m.FadeIn(r3[2]),
            m.FadeIn(r3[3], scale=1.15),
        )
        self.say(
            "[Its authors say so themselves]: [simple string matching is not a reliable security mechanism.]",
            m.FadeIn(quote[0:2]),
            m.AnimationGroup(m.FadeIn(quote[2]), m.FadeIn(quote[3])),
        )
        self.wait(0.4)
        self.clear()
        self.say(
            "[Claude Code uses rules instead]: [lists of deny, ask and allow], [checked in that order.]",
            m.FadeIn(settings, shift=0.2 * m.UP),
            m.LaggedStart(*[m.Circumscribe(glyphs(settings[1], f'"{k}"'), color=DECIDE[k][0], buff=0.04) for k in ("deny", "ask", "allow")],
                          lag_ratio=0.3),
            m.FadeIn(order, shift=0.15 * m.UP),
        )
        self.say(
            "[It also splits a command into its parts], [so a test run can't smuggle a delete] [past an allow rule.]",
            m.AnimationGroup(m.FadeIn(whole, shift=0.2 * m.DOWN), m.Create(split_lines), m.FadeIn(parts, shift=0.2 * m.DOWN),
                             m.FadeIn(verdicts)),
            m.FadeIn(overall, shift=0.15 * m.UP),
            m.FadeIn(naive, shift=0.15 * m.UP),
        )
        self.say(
            "[Rules come from several places]: [your user settings, the project, an enterprise policy], [and a deny at any level can't be overridden.]",
            m.FadeIn(src_cap),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in src_chips], lag_ratio=0.15), m.FadeIn(more)),
            m.AnimationGroup(m.FadeIn(wins, shift=0.15 * m.UP), m.FadeIn(src_cite)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 5 · sandbox

    def sandbox(self):
        self.next_section(
            "5 · Sandbox",
            notes=f"Claude Code docs: a Bash rule matches the command as written: {RM_RULE} stops `{DOC_ROWS[0]}`, "
            f"not `{DOC_ROWS[1]}` or `{DOC_ROWS[2]}`; 'for filesystem and network enforcement that doesn't depend on "
            "the command text, use sandboxing'. Li ch. 5's recipe: no network by default (allowlist proxy), source "
            "read-only + separate writable workspace, credentials not mounted, quotas and timeouts. Replayed in "
            f"the stand-in shell with only tmp/ writable, `{WIPE}` prints {len(SANDBOX_ERRORS)} errors and removes "
            "only the cache. Li ch. 4: process < container (shares the host kernel) < microVM (Firecracker, own "
            "kernel); a venv only isolates packages.",
        )
        # what a rule can't see
        rule = cmd_chip(RM_RULE, color=SUN, size=24, height=0.72).move_to([-1.1, 1.75, 0])
        rule_cap = text("one ask rule", 20, MUTED).next_to(rule, m.LEFT, buff=0.3)
        doc_rows = m.VGroup()
        for cmd, hit in zip(DOC_ROWS, DOC_HITS):
            box = card(8.4, 0.78, EDGE, CARD, radius=0.16, stroke=2)
            words = code(cmd, 22, INK).move_to(box).align_to(box, m.LEFT).shift(0.35 * m.RIGHT)
            verdict = (pill("✓ matched", MINT, MINT_FILL, 20) if hit else pill("✗ not matched", ROSE, ROSE_FILL, 20))
            verdict.move_to(box).align_to(box, m.RIGHT).shift(0.25 * m.LEFT)
            doc_rows.add(m.VGroup(box, words, verdict))
        doc_rows.arrange(m.DOWN, buff=0.22).move_to([-1.1, -0.05, 0])
        same = text("same program,\nwritten another way", 20, MUTED).next_to(doc_rows[1:], m.RIGHT, buff=0.3)
        assert same.get_right()[0] < 6.6
        to_box = pill("for limits that don't depend on the text: a sandbox", MINT, MINT_FILL, 22).move_to([-1.1, -2.2, 0])
        doc_cite = cite("Claude Code docs · permissions").next_to(to_box, m.DOWN, buff=0.25)

        # the box around the shell
        lines = ["$ pytest -q", PASSED, f"$ {WIPE}", *SANDBOX_ERRORS]
        term = terminal(lines, width=7.4, size=20, title="run_bash")
        term.align_to([-6.3, 2.0, 0], m.UL)
        rows = term[4]
        mounts = m.VGroup(cmd_chip("tinyshop/  source", color=EDGE, size=20), cmd_chip("work/  workspace", color=EDGE, size=20))
        mounts.arrange(m.DOWN, buff=0.75).next_to(term, m.RIGHT, buff=0.45).align_to(term, m.UP).shift(0.35 * m.DOWN)
        lock = icon(LOCK, 0.4).next_to(mounts[0], m.DOWN, buff=0.15).align_to(mounts[0], m.LEFT)
        read_only = text("read-only", 20, MINT, weight=m.SEMIBOLD).next_to(lock, m.RIGHT, buff=0.15)
        writable = text("writable", 20, MINT, weight=m.SEMIBOLD).next_to(mounts[1], m.DOWN, buff=0.18).align_to(mounts[1], m.LEFT)
        inside = m.VGroup(term, mounts, lock, read_only, writable)
        frame = dashed_box(13.2, inside.height + 0.55, MINT, dashes=110).move_to([0, inside.get_y(), 0])
        frame_cap = code("sandbox", 22, MINT, weight=m.BOLD).next_to(frame, m.UP, buff=0.08).align_to(frame, m.LEFT).shift(0.3 * m.RIGHT)
        assert inside.get_right()[0] < frame.get_right()[0] - 0.25 and frame_cap.get_top()[1] < 2.75, (inside.get_right(), frame_cap.get_top())
        tiles = m.VGroup(
            tile("ep17-nonet", "No network", "allowlisted proxy only"),
            tile(LOCK, "Read-only source", "+ a writable workspace"),
            tile("ep17-nokey", "No credentials", "not mounted at all"),
            tile(CLOCK, "Quotas, timeouts", "CPU, memory, time"),
        ).arrange(m.RIGHT, buff=0.2)
        tiles.next_to(frame, m.DOWN, buff=0.3)
        assert tiles.width <= 13.25 and tiles.get_bottom()[1] > -3.55, (tiles.width, tiles.get_bottom())
        recipe_cite = cite("Li, AI Agents in Depth · ch. 5").next_to(frame, m.UP, buff=0.1).align_to(frame, m.RIGHT)
        ro_words = m.VGroup(*[glyphs(r, "Read-only file system") for r in rows[3:]])

        # the ladder
        rungs = m.VGroup()
        for name, sub in (("process sandbox", "OS limits on one process"),
                          ("container", "stronger, but shares the host's kernel"),
                          ("micro VM", "its own kernel, e.g. Firecracker")):
            box = card(9.2, 0.95, MINT, CARD, radius=0.18, stroke=2)
            words = m.VGroup(title_text(name, 28), text(sub, 22, MUTED)).arrange(m.RIGHT, buff=0.45)
            assert words.width < 8.5, (name, words.width)
            words.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)
            rungs.add(m.VGroup(box, words))
        rungs.arrange(m.UP, buff=0.28).move_to([0.85, 0.55, 0])
        assert rungs.get_right()[0] < 6.6
        up = m.Arrow([-4.35, rungs.get_bottom()[1], 0], [-4.35, rungs.get_top()[1], 0], buff=0, color=MINT, stroke_width=5,
                     max_tip_length_to_length_ratio=0.08)
        up_cap = text("stronger\nisolation", 22, MINT, weight=m.SEMIBOLD).next_to(up, m.LEFT, buff=0.25)
        venv_box = dashed_box(9.2, 0.95, ROSE, dashes=50).move_to([0.85, rungs.get_bottom()[1] - 0.95, 0])
        venv_words = m.VGroup(title_text("a Python venv", 28, MUTED), text("isolates packages only", 22, MUTED))
        venv_words.arrange(m.RIGHT, buff=0.45).move_to(venv_box).align_to(venv_box, m.LEFT).shift(0.4 * m.RIGHT)
        not_box = pill("not a sandbox", ROSE, ROSE_FILL, 22).next_to(venv_box, m.DOWN, buff=0.25).align_to(venv_box, m.RIGHT)
        ladder_cite = cite("Li, AI Agents in Depth · ch. 4").next_to(venv_box, m.DOWN, buff=0.3).align_to(venv_box, m.LEFT)
        assert venv_words.width < 8.2 and not_box.get_bottom()[1] > -3.55

        self.say(
            "[Even Claude Code's docs warn that a rule matches the command as written.] [For limits that don't depend on the text], [they point to a sandbox.]",
            m.AnimationGroup(self.new_header("05 · sandbox", "Draw a box around the shell"), m.FadeIn(rule, shift=0.2 * m.DOWN),
                             m.FadeIn(rule_cap), m.LaggedStart(*[m.FadeIn(r, shift=0.2 * m.UP) for r in doc_rows], lag_ratio=0.3),
                             m.FadeIn(same)),
            m.FadeIn(to_box, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(doc_cite), m.Circumscribe(to_box, color=MINT, buff=0.06)),
        )
        self.wait(0.3)
        self.clear()
        self.say(
            "[So draw a box around the shell itself]: [inside, a command can do anything it likes], [but only to what the box contains.]",
            m.AnimationGroup(m.FadeIn(term[:4]), m.Create(frame), m.FadeIn(frame_cap)),
            m.LaggedStart(m.FadeIn(rows[0]), m.FadeIn(rows[1]), lag_ratio=0.5),
            m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in mounts], lag_ratio=0.3),
        )
        self.say(
            "[No network by default], [the source code mounted read only], [with a separate workspace to write in.]",
            m.FadeIn(tiles[0], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(tiles[1], shift=0.2 * m.UP), m.FadeIn(lock, scale=1.3), m.FadeIn(read_only)),
            m.FadeIn(writable, shift=0.15 * m.LEFT),
        )
        self.say(
            "[No credentials inside at all], [and quotas and timeouts on everything.]",
            m.FadeIn(tiles[2], shift=0.2 * m.UP),
            m.AnimationGroup(m.FadeIn(tiles[3], shift=0.2 * m.UP), m.FadeIn(recipe_cite)),
        )
        self.say(
            "[Now say the wrong-folder delete gets past the gate.] [Inside the box, it simply fails]: [read-only file system.]",
            m.AddTextLetterByLetter(rows[2], time_per_char=0.06),
            m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in rows[3:]], lag_ratio=0.3),
            m.AnimationGroup(m.LaggedStart(*[m.Circumscribe(g, color=MINT, buff=0.04) for g in ro_words], lag_ratio=0.2),
                             m.Circumscribe(mounts[0], color=MINT, buff=0.06)),
        )
        self.wait(0.4)
        self.clear()
        self.say(
            "[How strong is the box?] [A process sandbox is the lightest.]",
            m.AnimationGroup(m.GrowArrow(up), m.FadeIn(up_cap)),
            m.FadeIn(rungs[0], shift=0.2 * m.UP),
        )
        self.say(
            "[A container is stronger, but shares the host's kernel], [and a micro virtual machine gets a kernel of its own.]",
            m.FadeIn(rungs[1], shift=0.2 * m.UP),
            m.FadeIn(rungs[2], shift=0.2 * m.UP),
        )
        self.say(
            "[One common mistake]: [a Python virtual environment is not a sandbox.] [It separates packages, not files, network or processes.]",
            m.AnimationGroup(m.Create(venv_box), m.FadeIn(venv_words[0])),
            m.AnimationGroup(venv_words[0].animate.set_color(ROSE), m.FadeIn(not_box, shift=0.15 * m.UP), m.FadeIn(ladder_cite)),
            m.FadeIn(venv_words[1], shift=0.15 * m.RIGHT),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 6 · an independent reviewer

    def reviewer(self):
        self.next_section(
            "6 · Reviewer",
            notes="Li ch. 4: Proposer–Reviewer pre-approval, 'similar to the dual-signature system in banking'; the "
            "approving model from a different family; the sidecar reads only structured fields, never the agent's "
            "reasoning ('closes this rhetorical channel'); 'a few hundred milliseconds in parallel with streaming'; "
            "after repeated rejections it falls back to asking the user (learn-claude-code s03: Claude Code's auto "
            "mode classifier does the same). Otto's reasoning and the reviewer's reason are illustrative.",
        )
        bot = otto(1.1).move_to([-5.7, -0.6, 0])
        reason = bubble(text("The user wants the cache gone,\nso deleting everything is fine.", 22, INK), CORAL, CORAL_FILL, tail="left")
        reason.align_to([-6.6, 2.3, 0], m.UL)
        call_src = '{"tool": "bash",\n "command": "' + WIPE + '"}'
        the_call = boxed(code(call_src, 22, INK, t2c=JSON_T2C), "tool_use", SUN).move_to([-1.75, -0.6, 0])
        wall_x = 1.05
        wall = m.DashedLine([wall_x, 2.2, 0], [wall_x, -2.45, 0], dash_length=0.14).set_stroke(MUTED, 3)
        wall_cap = text("only the call passes", 20, MUTED).next_to(wall, m.UP, buff=0.12)
        assert wall_cap.get_top()[1] < 2.75
        judge = node_card(REVIEWER, "reviewer", "another model family", LIME, LIME_FILL, width=3.3, height=2.4)
        judge.move_to([4.8, -0.6, 0])
        a_call = arrow(the_call.get_right(), judge.get_left(), SUN, buff=0.1)
        y_r = reason.get_y() + 0.1
        a_reason = arrow([reason[1].get_right()[0], y_r, 0], [wall_x - 0.12, y_r, 0], CORAL, buff=0.1)
        stop = m.VGroup(m.Line([wall_x, y_r - 0.38, 0], [wall_x, y_r + 0.38, 0]).set_stroke(ROSE, 9),
                        text("blocked", 20, ROSE, weight=m.SEMIBOLD).move_to([wall_x - 0.75, y_r + 0.35, 0]))
        assert stop[1].get_right()[0] < wall_x - 0.12 and stop[1].get_left()[0] > reason[1].get_right()[0] + 0.1
        two = pill("two signatures, like a bank transfer", SUN, SUN_FILL, 20).next_to(the_call, m.DOWN, buff=0.4)
        two.set_x(min(two.get_x(), wall_x - 0.3 - two.width / 2))
        verdict = pill("✗ reject: wipes the whole repo", ROSE, ROSE_FILL, 20).next_to(judge, m.DOWN, buff=0.35)
        verdict.set_x(min(verdict.get_x(), 6.6 - verdict.width / 2))
        fast = text("a few hundred ms,\nin parallel", 20, MUTED).next_to(judge, m.UP, buff=0.3)
        marks = m.VGroup(*[text("✗", 30, ROSE, weight=m.BOLD) for _ in range(3)]).arrange(m.RIGHT, buff=0.18)
        you = icon("user", 0.5)
        you_cap = text("you decide", 22, AZURE, weight=m.SEMIBOLD)
        escalate = m.VGroup(marks, text("→", 26, MUTED), you, you_cap).arrange(m.RIGHT, buff=0.25)
        escalate.next_to(verdict, m.DOWN, buff=0.3).align_to([6.6, 0, 0], m.RIGHT)
        src = cite("Li, AI Agents in Depth · ch. 4").move_to([-3.8, -3.2, 0])
        assert escalate.get_bottom()[1] > -3.55 and fast.get_top()[1] < 2.7

        self.say(
            "[For the riskiest calls, add a second signature]: [another model must approve the call before it runs], [like a bank transfer that needs two.]",
            m.AnimationGroup(self.new_header("06 · reviewer", "A second signature"), m.FadeIn(bot, shift=0.2 * m.UP),
                             m.FadeIn(the_call, shift=0.2 * m.RIGHT)),
            m.AnimationGroup(m.GrowArrow(a_call), m.FadeIn(judge, shift=0.2 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(two, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[It should come from a different model family], [so the two don't share the same blind spots.]",
            m.Circumscribe(judge[1][2], color=LIME, buff=0.06),
            m.AnimationGroup(m.Circumscribe(bot, color=CORAL, buff=0.08), m.Circumscribe(judge[0], color=LIME, buff=0.06)),
        )
        self.say(
            "[And it sees only the call, never Otto's reasoning], [so a persuasive story can't talk it into a yes.]",
            m.AnimationGroup(m.FadeIn(reason, shift=0.2 * m.UP), m.Create(wall), m.FadeIn(wall_cap)),
            m.Succession(m.GrowArrow(a_reason), m.FadeIn(stop), m.FadeIn(verdict, shift=0.15 * m.UP)),
        )
        self.say(
            "[A check takes a few hundred milliseconds], [and after repeated rejections], [the decision goes back to you.]",
            m.FadeIn(fast, shift=0.15 * m.DOWN),
            m.LaggedStart(*[m.FadeIn(x, scale=1.4) for x in marks], lag_ratio=0.35),
            m.AnimationGroup(m.FadeIn(escalate[1]), m.FadeIn(you, shift=0.15 * m.LEFT), m.FadeIn(you_cap, shift=0.15 * m.LEFT)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ 7 · three layers

    def layers(self):
        self.next_section(
            "7 · Three layers",
            notes="Li, AI Agents in Depth, ch. 1 (types of guardrails): context, execution, data, 'by how hard they are "
            "to bypass'. Context: classifiers, source labels; 'an Agent cannot reliably determine whether prompt "
            "injection has already compromised its own context'. Execution: rate tools by 'reversibility, privilege "
            "level, and financial impact'; review 'by a mechanism outside the context' — this episode's gate, sandbox "
            "and reviewer. Data: row-level security: 'even if the prompt injection succeeds… the unauthorized "
            "operation is still rejected at the data layer'. Prompt injection itself is Episode 19.",
        )
        specs = (
            ("Context", "what the model reads", SUN, SUN_FILL, ["classifiers", "labels on outside text"]),
            ("Execution", "what the agent can do", MINT, MINT_FILL, ["rate each tool: reversible?", "privileged? costly?"]),
            ("Data", "what it can touch", ORCHID, ORCHID_FILL, ["row-level security", "rules on data the", "model can't forge"]),
        )
        cards = m.VGroup()
        for name, sub, color, fill, items in specs:
            box = card(4.05, 4.75, color, CARD, radius=0.22, stroke=2.5)
            head = m.VGroup(title_text(name, 34, color), text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
            body = m.VGroup(*[text(s, 22, INK) for s in items]).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
            head.move_to(box).align_to(box, m.UL).shift(0.3 * m.RIGHT + 0.3 * m.DOWN)
            body.next_to(head, m.DOWN, buff=0.35).align_to(head, m.LEFT)
            cards.add(m.VGroup(box, head, body))
        cards.arrange(m.RIGHT, buff=0.3).move_to([0, -0.55, 0])
        for c in cards[1:]:  # Futura's i-dot and t sit higher than its caps: line the titles up by baseline
            dy = cards[0][1][0][0].get_bottom()[1] - c[1][0][0].get_bottom()[1]
            c[1].shift(dy * m.UP)
            c[2].shift(dy * m.UP)
        harder = m.Arrow([-2.2, 2.3, 0], [6.4, 2.3, 0], buff=0, color=MUTED, stroke_width=4, max_tip_length_to_length_ratio=0.03)
        harder_cap = text("harder to bypass", 22, MUTED, weight=m.SEMIBOLD).next_to(harder, m.LEFT, buff=0.25)
        ceiling = text("can't reliably tell if its own\ncontext is compromised", 20, ROSE).next_to(cards[0][2], m.DOWN, buff=0.3)
        ceiling.align_to(cards[0][2], m.LEFT)
        ours = m.VGroup(*[pill(s, MINT, MINT_FILL, 20) for s in ("gate", "sandbox", "reviewer")])
        ours[:2].arrange(m.RIGHT, buff=0.12)
        ours[2].next_to(ours[:2], m.DOWN, buff=0.12).align_to(ours[:2], m.LEFT)
        ours.next_to(cards[1][2], m.DOWN, buff=0.35).align_to(cards[1][2], m.LEFT)
        assert ours.get_right()[0] < cards[1][0].get_right()[0] - 0.1
        y_a = cards.get_bottom()[1] + 0.5
        attack = m.Arrow([cards[0][0].get_left()[0] - 0.1, y_a, 0], [cards[2][0].get_x() - 0.6, y_a, 0], buff=0, color=ROSE,
                         stroke_width=6, max_tip_length_to_length_ratio=0.04)
        attack_cap = text("an injected action", 20, ROSE, weight=m.SEMIBOLD).next_to(attack.get_start(), m.UP, buff=0.12)
        attack_cap.align_to(cards[0][0], m.LEFT).shift(0.3 * m.RIGHT)
        halt = text("✗ rejected", 24, ORCHID, weight=m.BOLD).next_to(attack.get_end(), m.RIGHT, buff=0.15)
        src = cite("Li, AI Agents in Depth · ch. 1").move_to([0, -3.3, 0]).align_to(cards, m.RIGHT)
        assert halt.get_right()[0] < cards[2][0].get_right()[0] - 0.1

        self.say(
            "[Step back. AI Agents in Depth sorts guardrails into three layers], [by how hard they are to get around.]",
            m.AnimationGroup(self.new_header("07 · three layers", "Where the guardrails sit"),
                             m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c[0]), m.FadeIn(c[1])) for c in cards], lag_ratio=0.3),
                             m.FadeIn(src)),
            m.AnimationGroup(m.GrowArrow(harder), m.FadeIn(harder_cap)),
        )
        self.say(
            "[The context layer filters what the model reads.] [But an agent can't reliably tell] [whether its own context is already compromised.]",
            m.FadeIn(cards[0][2], shift=0.15 * m.UP),
            m.FadeIn(ceiling, shift=0.15 * m.UP),
            m.Circumscribe(ceiling, color=ROSE, buff=0.06),
        )
        self.say(
            "[The execution layer limits what it can do]: [rate each tool by how reversible, privileged and costly it is.] [Today's gate, sandbox and reviewer all live here.]",
            m.Circumscribe(cards[1][0], color=MINT, buff=0.06),
            m.FadeIn(cards[1][2], shift=0.15 * m.UP),
            m.LaggedStart(*[m.FadeIn(p, shift=0.15 * m.UP) for p in ours], lag_ratio=0.3),
        )
        self.say(
            "[And the data layer is the last line.] [Even if an injected action gets past the rest], [row-level security rejects it at the data itself.]",
            m.FadeIn(cards[2][2], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(attack_cap), m.GrowArrow(attack)),
            m.AnimationGroup(m.FadeIn(halt, scale=1.3), m.Circumscribe(cards[2][0], color=ORCHID, buff=0.06)),
        )
        self.wait(0.8)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section(
            "Recap",
            notes="Otto now works behind a gate (deny / ask / allow, rules on parsed commands), inside a sandbox (no "
            "network, read-only source, no credentials, quotas), with a reviewer of another model family for risky "
            "calls. Coming up: hooks (Episode 18), prompt injection (Episode 19). Gate · Sandbox · Reviewer.",
        )
        hero = otto(1.3).move_to([0, 1.05, 0])
        cols = m.VGroup()
        for pic, name, sub, color, fill in ((SHIELD, "gate", "deny, ask or allow,\nby rules on each part", MINT, MINT_FILL),
                                            (LOCK, "sandbox", "no network, no keys,\nread-only source", MINT, MINT_FILL),
                                            (REVIEWER, "reviewer", "another model,\nsees only the call", LIME, LIME_FILL)):
            words = code(name, 24, INK, weight=m.BOLD)
            body = m.VGroup(icon(pic, 0.55), words).arrange(m.RIGHT, buff=0.25)
            box = card(3.3, 0.9, color, fill, radius=0.2, stroke=2.5)
            tag = m.VGroup(box, body.move_to(box))
            cols.add(m.VGroup(tag, text(sub, 22, MUTED)).arrange(m.DOWN, buff=0.22))
        cols.arrange(m.RIGHT, buff=0.75, aligned_edge=m.UP).move_to([0, -1.55, 0])
        arms = m.VGroup(*[m.Line(hero[0].get_bottom() + 0.08 * m.DOWN, c[0].get_top() + 0.06 * m.UP) for c in cols]).set_stroke(MINT, 3)

        subs = {18: "add behaviour around the loop\nwithout changing it",
                19: "when text an agent reads\ntries to take over"}
        bodies = []
        for n in (18, 19):
            title_, mod = EPISODES[n]
            name, accent, tint = MODULES[mod]
            heading = title_text(title_, 32)
            body = m.VGroup(pill(f"EP {n} · {name}", accent, tint, 20, mono=True), heading,
                            text(subs[n], 22, MUTED)).arrange(m.DOWN, buff=0.25, aligned_edge=m.LEFT)
            bodies.append((body, accent))
        tallest = max(b.height for b, _ in bodies)
        nexts = m.VGroup()
        for body, accent in bodies:
            box = card(6.3, tallest + 0.8, accent, CARD, stroke=2)
            nexts.add(m.VGroup(box, body.move_to(box).align_to(box, m.LEFT).shift(0.4 * m.RIGHT)))
        nexts.arrange(m.RIGHT, buff=0.4).move_to([0, 0.1, 0])

        halo = glow(MINT, radius=4.0)
        words = on_baseline(m.VGroup(*[title_text(w, 66, c) for w, c in (("Gate", MINT), ("Sandbox", SUN), ("Reviewer", LIME))]))
        words.arrange(m.RIGHT, buff=0.95)
        on_baseline(words)
        mid = words[0].get_y()
        dots = m.VGroup(*[m.Dot([(a.get_right()[0] + b.get_left()[0]) / 2, mid, 0], radius=0.07, color=MUTED) for a, b in zip(words, words[1:])])
        row = m.VGroup(words, dots).move_to(0.4 * m.UP)
        tagline = text("The model proposes. The harness decides.", 30, MUTED).next_to(row, m.DOWN, buff=0.75)

        self.say(
            "[Otto now works behind a gate], [inside a sandbox], [with a second model checking its riskiest calls.]",
            m.AnimationGroup(self.new_header("recap", "Otto, with guardrails"), m.FadeIn(hero, scale=0.85),
                             m.Create(arms[0]), m.FadeIn(cols[0], shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Create(arms[1]), m.FadeIn(cols[1], shift=0.2 * m.DOWN)),
            m.AnimationGroup(m.Create(arms[2]), m.FadeIn(cols[2], shift=0.2 * m.DOWN)),
        )
        self.say(
            "[Episode eighteen adds hooks around the loop], [and episode nineteen shows how text an agent reads can take it over.]",
            m.AnimationGroup(self.new_header("coming up", "More on safety"),
                             m.FadeOut(m.VGroup(hero, cols, arms), shift=0.2 * m.UP), m.FadeIn(nexts[0], shift=0.2 * m.UP)),
            m.FadeIn(nexts[1], shift=0.2 * m.UP),
        )
        self.say(
            "[To sum up]: [gate], [sandbox], [reviewer.]",
            m.AnimationGroup(self.drop_header(), m.FadeOut(nexts, shift=0.2 * m.DOWN), m.FadeIn(halo, scale=0.6)),
            m.FadeIn(words[0], shift=0.3 * m.UP),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP)),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP)),
        )
        self.say("[The model proposes. The harness decides.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row, tagline)], run_time=0.9)
