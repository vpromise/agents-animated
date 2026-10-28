"""Agents, Animated · Episode 30 · Training Agents with RL (Module 6 · Beyond the Loop)

How models are trained to be better agents. Running example: one RL step on a tinyshop-style bug-fix task
(cart.total() skips the last item; fix it without touching tests/), the storyboard of Li Bojie's ch. 8
("16 rollouts → one parameter update") with tinyshop in place of the book's parser.py. Nothing on screen is
drawn by hand: the scene copies tinyshop into sixteen in-memory sandboxes, runs the reference agent_loop
(unchanged) in each against a scripted stand-in for the model (four kinds of behaviour, the three patches of
Episode 29 plus "claim success untested"), scores every sandbox with the task's own copy of the tests (real
code, executed), computes GRPO's group-relative advantages, and applies one policy-gradient step to a toy
policy over the four behaviours. Every number the narration quotes is asserted below.

Arc: hook (a better harness, but the model learned elsewhere) · four stages (pretraining, mid-training, SFT
with the prompt masked, RL; "SFT memorizes, RL generalizes" as shorthand) · rollouts (16 sandboxes, the loop
of Episode 2) · rewards (the verifier of Episode 29: 4 pass, 12 fail; the deleted assertion gets no reward) ·
advantages (mean 0.25 → +0.75 / −0.25; they sum to zero; all fail → no signal) · the update (advantage on
every token Otto wrote, sandbox output masked; the toy policy: fix-and-test 25% → 30%; GRPO vs PPO) · the cost
(2,000 s + 600 s ≈ 43 min a step, 100 steps ≈ 72 h) · reward design (a "suite green" reward pays the hack:
19% → 21%; verifiable rewards, path penalties, dense vs sparse) · on-policy distillation · recap.

Sources (facts dated October 2026):
- Li Bojie, AI Agents in Depth, ch. 8 (bojieli.github.io/ai-agent-book/book-en/chapter8/): the four stages
  (pre-training "like a person who has read all the books in a library", "tens of millions of dollars";
  mid-training: domain data plus replay; SFT "a teacher giving a student standard answers to imitate", loss
  "calculated only on the response"; RL "working problems yourself", samples "often tens to hundreds of times
  that of SFT"); "This is a measured result under those experimental conditions, not a universal property of
  SFT and RL" (GeneralPoints, from Chu et al., 2025, "SFT Memorizes, RL Generalizes": GP-L RL +3.5 points,
  11.5% → 15.0%; SFT −8.1 points, 11.5% → 3.4%); "RL is good at raising the probability of existing but
  unlikely successful behavior"; the 16-rollout storyboard ("Suppose 4 of the 16 attempts pass all tests
  without touching the test files", "the 4 that passed are above the group average and get a positive
  advantage", "every reward is identical ... the relative advantage vanishes", "This within-group comparison
  is the core of GRPO", the update shifts probability toward "reproduce the problem, check the boundary
  condition, change the implementation, and run the tests"); timing ("the slowest rollout takes about 2,000
  seconds", training "about 600 seconds", "about 43 minutes, and 100 consecutive steps come to nearly 72
  hours"); GRPO "needs no separate value model", PPO "trains a value model ... which suits long trajectories
  that need fine-grained credit assignment"; tool-use RL masks environment tokens (gradients "only through
  the model's own thinking and its tool-call arguments"); reward hacking: "exploiting a rule or an
  implementation hole to score highly"; verifiable rewards most reliable; "dense feedback buys convergence
  speed, sparse feedback buys exploration space"; RLVP "reward the outcome, penalize the path"; on-policy
  distillation: the teacher gives its next-token distribution at the states the student reaches, "roughly
  one tenth the training steps of pure RL" on mathematics.
- GRPO: Shao et al., 2024 (DeepSeekMath), group-normalised advantages, no value network. PPO: Schulman et
  al., 2017 (clipped probability ratio; a learned value function as baseline).

Illustrative (for teaching): the sixteen behaviours and their order (random.Random(30)), the thoughts in the
trajectory, the rollout durations (seeded), the toy policy over four behaviours (its starting probabilities =
the group's frequencies; one step of size 1 on the GRPO surrogate at ratio 1, no clipping, no KL), the
teacher's per-token marks, the path-penalty examples.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import inspect
import math
import random
import sys
from pathlib import Path
from types import SimpleNamespace as NS

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

REF = (SERIES / "harness_reference.py").read_text()
PREFIX = "ep30"
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
TASK_TESTS = TEST_CART  # the task's own copy of the tests, kept outside the sandbox (as in Episode 29)
OTHER_TESTS = 11  # tinyshop's other tests, all passing (as in Episodes 2, 22 and 29)
EDITS = {  # Episode 29's three kinds of patch: (path, old_text, new_text)
    "cheat": ("tests/test_cart.py", "    assert got == 7\n", ""),
    "patch": ("cart.py", "    return s\n", "    return s + items[-1]\n"),
    "fix": ("cart.py", "range(len(items) - 1)", "range(len(items))"),
}


def run_tests(box, tests):
    """A test file executed against a sandbox's cart.py: {test name: passed}."""
    cart = {}
    exec(compile(box["cart.py"], "cart.py", "exec"), cart)
    ns = {"total": cart["total"]}
    exec(compile(tests.replace("from cart import total\n", ""), "test_cart.py", "exec"), ns)
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


def reward(box):
    # the task's tests, not the box's
    ok = run_tests(box, TASK_TESTS)
    return int(all(ok.values()))


REWARD_SRC = inspect.getsource(reward).strip()
assert len(REWARD_SRC.splitlines()) == 4


def naive_reward(box):
    """The tempting shortcut: is the sandbox's OWN suite green?"""
    return int("failed" not in pytest_q(run_tests(box, box["tests/test_cart.py"])))


assert pytest_q(run_tests({"cart.py": CART_BUG}, TEST_CART)) == "1 failed, 12 passed"  # the course's running bug

# ---------------------------------------------------------------------------- the harness (reference names)

SYSTEM = "You are Otto, a coding agent working in the tinyshop repo."
assert f'SYSTEM = "{SYSTEM}"' in REF
assert "def agent_loop(messages):" in REF and "TOOL_HANDLERS" in REF
BOX = {}  # the current sandbox


def run_bash(command):
    if command == "pytest -q":
        return pytest_q(run_tests(BOX, BOX["tests/test_cart.py"]))
    if command == "cat cart.py":
        return BOX["cart.py"]
    raise ValueError(command)


def edit_file(path, old_text, new_text):
    assert BOX[path].count(old_text) == 1, (path, old_text)
    BOX[path] = BOX[path].replace(old_text, new_text, 1)
    return f"edited {path}"


TOOLS = [{"name": "bash"}, {"name": "edit_file"}]  # schemas abridged
TOOL_HANDLERS = {"bash": run_bash, "edit_file": edit_file}


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


TEST, READ = {"command": "pytest -q"}, {"command": "cat cart.py"}


def edit(kind):
    path, old, new = EDITS[kind]
    return {"path": path, "old_text": old, "new_text": new}


DONE = "Fixed. All 13 tests pass."
PLANS = {  # what the sampled policy does, step by step: (thought, tool, input); tool None = the final answer
    "fix": [("Reproduce first.", "bash", TEST), ("Off by one.", "edit_file", edit("fix")),
            ("Check again.", "bash", TEST), (DONE, None, None)],
    "patch": [("Read the cart.", "bash", READ), ("Add the last item.", "edit_file", edit("patch")),
              ("Fixed.", None, None)],
    "cheat": [("Reproduce first.", "bash", TEST), ("Drop that check.", "edit_file", edit("cheat")),
              ("Check again.", "bash", TEST), (DONE, None, None)],
    "quit": [("Read the cart.", "bash", READ), ("Fixed.", None, None)],
}


class ScriptedPolicy:
    """Stands in for client.messages: one sampled behaviour per rollout (illustrative)."""

    def __init__(self):
        self.plan, self.step = [], 0

    def create(self, system, messages, tools):
        thought, tool, args = self.plan[self.step]
        self.step += 1
        content = [NS(type="text", text=thought)]
        if tool is None:
            return NS(stop_reason="end_turn", content=content)
        content.append(NS(type="tool_use", id=f"toolu_{self.step:02d}", name=tool, input=args))
        return NS(stop_reason="tool_use", content=content)


POLICY = ScriptedPolicy()
client = NS(messages=POLICY)


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


TASK = "Fix cart.total(): it skips the last item."
G = 16
KINDS = ["fix"] * 4 + ["patch"] * 5 + ["cheat"] * 3 + ["quit"] * 4
random.Random(30).shuffle(KINDS)


def rollout(kind):
    """One sandbox: a fresh copy of tinyshop, one run of the loop, then the verifier."""
    BOX.clear()
    BOX.update({"cart.py": CART_BUG, "tests/test_cart.py": TEST_CART})
    POLICY.plan, POLICY.step = PLANS[kind], 0
    messages = [{"role": "user", "content": TASK}]
    agent_loop(messages)
    return NS(kind=kind, box=dict(BOX), messages=messages, reward=reward(BOX), naive=naive_reward(BOX),
              suite=pytest_q(run_tests(BOX, BOX["tests/test_cart.py"])), task=pytest_q(run_tests(BOX, TASK_TESTS)))


RUNS = [rollout(k) for k in KINDS]
REWARDS = [r.reward for r in RUNS]
COUNT = {k: KINDS.count(k) for k in PLANS}
assert len(RUNS) == G == 16 and COUNT == {"fix": 4, "patch": 5, "cheat": 3, "quit": 4}  # "four ... five ... three ... four"
assert all(r.reward == (r.kind == "fix") for r in RUNS) and sum(REWARDS) == 4  # "four pass ... twelve fail"
assert all(r.box["tests/test_cart.py"] == TEST_CART for r in RUNS if r.reward)  # passes never touched the tests
CHEAT = next(r for r in RUNS if r.kind == "cheat")
assert CHEAT.suite == "13 passed" and CHEAT.task == "1 failed, 12 passed" and CHEAT.reward == 0  # green suite, no reward
PATCH = next(r for r in RUNS if r.kind == "patch")
assert PATCH.suite == "1 failed, 12 passed" and PATCH.reward == 0  # the empty cart now crashes


def advantages(rewards):
    """GRPO's group-relative advantage: each reward minus the group's mean (the book's form)."""
    mean = sum(rewards) / len(rewards)
    return mean, [r - mean for r in rewards]


MEAN, ADV = advantages(REWARDS)
A_PASS, A_FAIL = 1 - MEAN, 0 - MEAN
assert MEAN == 0.25 and A_PASS == 0.75 and A_FAIL == -0.25  # "zero point two five", "plus zero point seven five", "minus zero point two five"
UP = sum(a for a in ADV if a > 0)
DOWN = -sum(a for a in ADV if a < 0)
assert UP == DOWN == 3.0 and abs(sum(ADV)) < 1e-12  # "the pushes balance"
NONE_MEAN, NONE_ADV = advantages([0] * G)
assert NONE_MEAN == 0 and all(a == 0 for a in NONE_ADV)  # "every advantage is zero"
STD = math.sqrt(MEAN * (1 - MEAN))  # GRPO (Shao et al., 2024) also divides by the group's std: notes only
NORM = (round(A_PASS / STD, 2), round(A_FAIL / STD, 2))
assert NORM == (1.73, -0.58)

# the passing rollout shown token by token, and its loss weights
SHOWN = next(r for r in RUNS if r.kind == "fix")
SHOWN_N = RUNS.index(SHOWN) + 1
assert len(SHOWN.messages) == 8  # task, 3 × (call, result), answer
RESULTS = [m_["content"][0]["content"] for m_ in SHOWN.messages[2::2]]
assert RESULTS == ["1 failed, 12 passed", "edited cart.py", "13 passed"]

# ---------------------------------------------------------------------------- the update, on a toy policy

HABITS = ["fix", "patch", "cheat", "quit"]
HABIT_NAME = {"fix": "fix, then test", "patch": "patch the symptom", "cheat": "delete the assertion",
              "quit": "claim, untested"}
TILE_NAME = {"fix": "fix", "patch": "patch", "cheat": "edit test", "quit": "untested"}
PI0 = {k: COUNT[k] / G for k in HABITS}  # the policy that sampled this group (illustrative)
ETA = 1.0


def update(pi, kinds, adv, eta=ETA):
    """One gradient step on the GRPO surrogate (1/G)·Σ A_i·log π(a_i) for a softmax policy over habits."""
    z = {k: math.log(pi[k]) + eta * sum(a * ((kk == k) - pi[k]) for kk, a in zip(kinds, adv)) / len(kinds)
         for k in pi}
    total = sum(math.exp(v) for v in z.values())
    return {k: math.exp(v) / total for k, v in z.items()}


def pct(p):
    return round(100 * p)


PI1 = update(PI0, KINDS, ADV)
assert pct(PI0["fix"]) == 25 and pct(PI1["fix"]) == 30  # "from twenty-five to thirty percent"
assert all(PI1[k] < PI0[k] for k in HABITS if k != "fix")
NAIVE = [r.naive for r in RUNS]
assert sum(NAIVE) == 7 and all(r.naive == (r.kind in ("fix", "cheat")) for r in RUNS)  # "the three deletions earn reward one"
N_MEAN, N_ADV = advantages(NAIVE)
PI_HACK = update(PI0, KINDS, N_ADV)
assert pct(PI0["cheat"]) == 19 and pct(PI_HACK["cheat"]) == 21  # "from nineteen to twenty-one percent"

# ---------------------------------------------------------------------------- the cost of a step (Li, ch. 8)

SLOWEST_S, TRAIN_S = 2000, 600
STEP_S = SLOWEST_S + TRAIN_S
assert round(STEP_S / 60) == 43 and round(100 * STEP_S / 3600) == 72 and TRAIN_S / 60 == 10
_rng = random.Random(300)
SPAN = {"fix": (1200, 1900), "cheat": (1100, 1700), "patch": (800, 1400), "quit": (450, 900)}
DURATIONS = [_rng.randint(*SPAN[k]) for k in KINDS]
DURATIONS[KINDS.index("fix")] = SLOWEST_S  # one fix runs longest (illustrative)
assert max(DURATIONS) == SLOWEST_S

# ---------------------------------------------------------------------------- look

MODEL_ICON = recolored("llm", CORAL, CORAL_FILL, prefix=PREFIX)
TEACHER_ICON = recolored("llm", LIME, LIME_FILL, prefix=PREFIX)
BOOK_ICON = recolored("book", MUTED, CARD_HI, prefix=PREFIX)
DOC_ICON = recolored("doc", MUTED, CARD_HI, prefix=PREFIX)
TEACH_ICON = recolored("user", MUTED, CARD_HI, prefix=PREFIX)
TARGET_ICON = "ep30-target"

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")


def cite(s, size=20):
    return text(s, size, MUTED)


def baseline_of(glyph_group, chars):
    bottoms = sorted(g.get_bottom()[1] for g, c in zip(glyph_group, chars) if c in FLAT)
    return bottoms[len(bottoms) // 2] if bottoms else glyph_group.get_bottom()[1]


def baseline(words):
    return baseline_of(words, [c for c in words.text if not c.isspace()])


def sit(words, y):
    """Put a Text's baseline at y."""
    words.shift((y - baseline(words)) * m.UP)
    return words


def word_chip(word, color, fill, size=24, height=0.62, pad=0.55):
    return chip(word, color, fill, width=text(word, size, weight=m.SEMIBOLD).width + pad, height=height, size=size)


def icon_chip(pic, label, color, fill, size=24, height=0.8, icon_h=0.46):
    body = m.VGroup(icon(pic, icon_h), text(label, size, INK, weight=m.SEMIBOLD)).arrange(m.RIGHT, buff=0.2)
    box = card(body.width + 0.6, height, color, fill, radius=0.18)
    return m.VGroup(box, body.move_to(box))


def badge(value, size=20):
    ok = value > 0
    disc = m.Circle(radius=0.2).set_stroke(MINT if ok else ROSE, 2).set_fill(MINT_FILL if ok else ROSE_FILL, 1)
    digit = code(str(value), size, MINT if ok else ROSE, weight=m.BOLD).move_to(disc)
    return m.VGroup(disc, digit)


def signed(a):
    return f"+{a:.2f}" if a > 0 else (f"−{-a:.2f}" if a < 0 else "0")


# ---- the rack of sandboxes

TW, TH, TGAP = 1.5, 1.1, 0.14
GRID_C = np.array([-3.3, -0.8, 0])
SEG_COLOR = {"tool_use": SUN, "tool_result": MINT, "text": CORAL}


def segments_of(messages):
    """One coloured segment per message after the task: a call (SUN), a result (MINT), the answer (CORAL)."""
    out = []
    for msg in messages[1:]:
        content = msg["content"]
        if isinstance(content[0], dict):
            out.append("tool_result")
        elif any(b.type == "tool_use" for b in content):
            out.append("tool_use")
        else:
            out.append("text")
    return out


def sandbox(i, run):
    r, c = divmod(i, 4)
    center = GRID_C + np.array([(c - 1.5) * (TW + TGAP), (1.5 - r) * (TH + TGAP), 0])
    box = card(TW, TH, EDGE, CARD, radius=0.12, stroke=2).move_to(center)
    num = code(f"#{i + 1:02d}", 20, MUTED)
    num.move_to(box.get_corner(m.UL) + np.array([0.14 + num.width / 2, -0.25, 0]))
    segs = m.VGroup(*[
        m.RoundedRectangle(corner_radius=0.03, width=0.13, height=0.13, stroke_width=0,
                           fill_color=SEG_COLOR[s], fill_opacity=1)
        for s in segments_of(run.messages)
    ]).arrange(m.RIGHT, buff=0.045)
    segs.move_to(center + np.array([0, 0.0, 0])).align_to(box, m.LEFT).shift(0.15 * m.RIGHT)
    label = text(TILE_NAME[run.kind], 20, INK, weight=m.SEMIBOLD)
    label.move_to(box.get_corner(m.DL) + np.array([0.15 + label.width / 2, 0.27, 0]))
    sit(label, box.get_bottom()[1] + 0.17)
    spot = box.get_corner(m.UR) + np.array([-0.3, -0.29, 0])
    adv = code(signed(ADV[i]), 20, MINT if ADV[i] > 0 else ROSE, weight=m.BOLD)
    adv.move_to(label, aligned_edge=m.LEFT)
    sit(adv, box.get_bottom()[1] + 0.17)
    return NS(box=box, num=num, segs=segs, label=label, badge=badge(run.reward).move_to(spot), adv=adv,
              run=run, group=m.VGroup(box, num, segs, label), center=center)


# ---- message rows (the series' messages-column look, one line each)

ROW_W, ROW_H = 6.65, 0.5


def msg_row(role, accent, parts, width=ROW_W):
    """A one-line message: a stripe in the block's colour, the role, then (string, kind) parts."""
    box = card(width, ROW_H, EDGE, CARD, radius=0.1, stroke=1.5)
    stripe = m.RoundedRectangle(corner_radius=0.03, width=0.07, height=ROW_H - 0.16, stroke_width=0,
                                fill_color=accent, fill_opacity=1)
    stripe.move_to(box).align_to(box, m.LEFT).shift(0.08 * m.RIGHT)
    label = code(role, 22, ROLES[role][0], weight=m.BOLD)
    label.move_to(box).align_to(box, m.LEFT).shift(0.28 * m.RIGHT)
    words = m.VGroup(*[(text(s, 20, INK) if kind == "text" else code(s, 20, accent)) for s, kind in parts])
    words.arrange(m.RIGHT, buff=0.22).move_to(box).align_to(box, m.LEFT).shift(2.12 * m.RIGHT)
    base = box.get_y() - 0.085
    sit(label, base)
    for w in words:
        sit(w, base)
    return m.VGroup(box, stripe, label, words)


def call_text(block):
    if block.name == "bash":
        return block.input["command"]
    return f"edit_file {block.input['path']}"


def trajectory_rows(messages):
    """(row, trained?) for every message of a rollout."""
    rows = [(msg_row("user", AZURE, [(messages[0]["content"], "text")]), False)]
    for msg in messages[1:]:
        content = msg["content"]
        if isinstance(content[0], dict):
            rows.append((msg_row("user", MINT, [(content[0]["content"], "code")]), False))
        else:
            parts = [(content[0].text, "text")]
            calls = [b for b in content if b.type == "tool_use"]
            if calls:
                parts.append((call_text(calls[0]), "code"))
            rows.append((msg_row("assistant", SUN if calls else CORAL, parts), True))
    return rows


# ---- policy bars

BAR_SCALE = 8.0  # units per probability 1


def bar_rect(p, color, left, y, height=0.3):
    rect = m.RoundedRectangle(corner_radius=0.06, width=max(BAR_SCALE * p, 0.02), height=height, stroke_width=0,
                              fill_color=color, fill_opacity=1)
    return rect.move_to([left + BAR_SCALE * p / 2, y, 0])


class Slot:
    """One place whose content is swapped (a percentage, a caption)."""

    def __init__(self, scene, current=None):
        self.scene, self.current = scene, current

    def to(self, new, shift=0.12):
        anims = []
        if self.current is not None:
            anims.append(m.FadeOut(self.current, shift=shift * m.UP))
        if new is not None:
            anims.append(m.FadeIn(new, shift=shift * m.UP))
        self.current = new
        return m.AnimationGroup(*anims) if anims else m.Wait(0.05)


# ---------------------------------------------------------------------------- the episode


class Ep30TrainingAgentsRL(Episode):
    number = 30

    def lesson(self):
        self.hook()
        self.stages()
        self.rollouts()
        self.rewards()
        self.advantages_()
        self.the_update()
        self.cost()
        self.reward_design()
        self.distillation()
        self.recap()

    def speak(self, script, *animations):
        """say(), with every animation of an object not yet on screen marked as an introducer (an object
        swapped in and out, or changed twice, in the same line must not show early). From Episode 22."""
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
        self.next_section("Hook", notes="Episodes 1–29 improved Otto's harness (the loop: Ep 2; tools: Ep 3; context: Ep 6; checks: Ep 22). The model inside learned to act as an agent in training. Today: one RL step on a tinyshop-style bug, with Episode 29's verifier as the reward. " + FACTS)
        bot = otto(1.45).move_to([-4.25, 0.35, 0])
        parts = [("loop", "Ep 2"), ("tools", "Ep 3"), ("context", "Ep 6"), ("checks", "Ep 22")]
        offs = [(-1.25, 1.75), (1.25, 1.75), (-1.25, -1.5), (1.25, -1.5)]
        chips = m.VGroup()
        for (name, ep), (dx, dy) in zip(parts, offs):
            c = word_chip(name, EDGE, CARD, size=24)
            tag = code(ep, 20, MUTED).next_to(c, m.DOWN, buff=0.1)
            chips.add(m.VGroup(c, tag).move_to(bot.get_center() + np.array([dx, dy, 0])))
        harness = text("a better harness, episode after episode", 22, MUTED).next_to(chips, m.DOWN, buff=0.3)

        box = card(3.6, 2.7, CORAL, CORAL_FILL).move_to([3.2, 0.6, 0])
        brain = m.VGroup(icon(MODEL_ICON, 1.0), text("the model", 30, INK, weight=m.BOLD),
                         text("its weights", 22, MUTED)).arrange(m.DOWN, buff=0.16).move_to(box)
        model = m.VGroup(box, brain)
        link = m.DashedLine(bot.get_right() + 0.25 * m.RIGHT, box.get_left() + 0.15 * m.LEFT, dash_length=0.12)
        link.set_stroke(FAINT, 3)
        inside = text("inside Otto", 20, MUTED).next_to(link, m.UP, buff=0.12)
        trained = pill("learned in training", LIME, LIME_FILL, 22).next_to(box, m.DOWN, buff=0.3)

        task = icon_chip(DOC_ICON, "a tinyshop bug", EDGE, CARD, size=22, height=0.72)
        verifier = icon_chip("check", "verifier · Ep 29", MINT, MINT_FILL, size=22, height=0.72)
        pay = word_chip("reward 1 or 0", LIME, LIME_FILL, size=22, height=0.72)
        row = m.VGroup(task, verifier, pay).arrange(m.RIGHT, buff=0.75).move_to([0.6, -2.85, 0])
        a1 = arrow(task.get_right(), verifier.get_left(), MUTED)
        a2 = arrow(verifier.get_right(), pay.get_left(), MUTED)
        up = arrow(pay.get_top(), trained.get_bottom(), LIME, buff=0.12)

        self.speak(
            "[Episode after episode, we gave Otto a better harness]: [a loop, tools, context and checks.]",
            m.FadeIn(bot, shift=0.3 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(c, shift=0.15 * m.UP) for c in chips], lag_ratio=0.25), m.FadeIn(harness)),
        )
        self.speak(
            "[But the model inside learned to act like an agent somewhere else]: [in training.]",
            m.AnimationGroup(m.Create(link), m.FadeIn(inside), m.FadeIn(model, target_position=bot)),
            m.FadeIn(trained, shift=0.15 * m.UP),
        )
        self.speak(
            "[Today we train it on bugs like tinyshop's], [with the verifier from episode twenty-nine paying the reward.]",
            m.AnimationGroup(m.FadeOut(harness), m.FadeIn(task, shift=0.15 * m.UP)),
            m.AnimationGroup(m.GrowArrow(a1), m.FadeIn(verifier, shift=0.15 * m.UP), m.GrowArrow(a2),
                             m.FadeIn(pay, shift=0.15 * m.UP), m.GrowArrow(up), lag_ratio=0.3),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ four stages

    def stages(self):
        self.next_section("1 · Four stages", notes="Li, AI Agents in Depth, ch. 8: pre-training 'like a person who has read all the books in a library', the most expensive step ('tens of millions of dollars'); mid-training: domain data plus replay of older data; SFT: 'a teacher giving a student standard answers to imitate', thousands to tens of thousands of pairs, loss 'calculated only on the response'; RL: 'working problems yourself', samples 'often tens to hundreds of times that of SFT'. 'SFT memorizes, RL generalizes' is shorthand: 'a measured result under those experimental conditions, not a universal property' (GeneralPoints, Chu et al., 2025: on new rules RL +3.5 points, 11.5% → 15.0%; SFT −8.1 points, 11.5% → 3.4%). The SFT example is illustrative. " + FACTS)
        specs = [
            ("Pretraining", BOOK_ICON, "reads the library", "next-token prediction\ncosts tens of millions", EDGE),
            ("Mid-training", DOC_ICON, "specialist textbooks", "domain data\nplus replay of old data", EDGE),
            ("SFT", TEACH_ICON, "a teacher demonstrates", "thousands of examples\nloss on the response", EDGE),
            ("RL", TARGET_ICON, "works problems itself", "learns from rewards\nfar more samples", LIME),
        ]
        W, H, xs, y0 = 3.1, 2.85, [-4.95, -1.65, 1.65, 4.95], 1.0
        shells, bodies = m.VGroup(), m.VGroup()
        for (name, pic, analogy, detail, accent), x in zip(specs, xs):
            shell = card(W, H, accent if accent != EDGE else EDGE, CARD, radius=0.2, stroke=2).move_to([x, y0, 0])
            n = code(f"{len(shells) + 1}", 20, LIME if accent == LIME else MUTED, weight=m.BOLD)
            n.move_to(shell.get_corner(m.UL) + np.array([0.28, -0.3, 0]))
            pic_ = icon(pic, 0.58).move_to(shell.get_top() + 0.52 * m.DOWN)
            title = title_text(name, 28, LIME if accent == LIME else INK).next_to(pic_, m.DOWN, buff=0.16)
            an = text(analogy, 20, INK).next_to(title, m.DOWN, buff=0.12)
            det = m.VGroup(*[text(line, 20, MUTED) for line in detail.split("\n")]).arrange(m.DOWN, buff=0.1)
            det.next_to(an, m.DOWN, buff=0.18)
            assert max(an.width, det.width) < W - 0.24, (name, an.width, det.width)
            shells.add(m.VGroup(shell, n))
            bodies.add(m.VGroup(pic_, title, an, det))

        # SFT: one example, the prompt masked
        sft_title = code("one SFT example", 20, MUTED)
        r_user = msg_row("user", AZURE, [(TASK, "text")], width=8.0)
        r_asst = msg_row("assistant", CORAL, [("Off by one:", "text"), ("for item in items: s += item", "code")], width=8.0)
        rows = m.VGroup(r_user, r_asst).arrange(m.DOWN, buff=0.14).move_to([-1.75, -2.35, 0])
        sft_title.next_to(rows, m.UP, buff=0.18).align_to(rows, m.LEFT)
        w_head = code("in the loss?", 20, MUTED)
        w_no = code("no · masked", 20, FAINT)
        w_yes = code("yes", 20, MINT, weight=m.BOLD)
        for w, r in ((w_no, r_user), (w_yes, r_asst)):
            w.next_to(r, m.RIGHT, buff=0.4)
            sit(w, baseline(r[2]))
        w_head.move_to(w_no, aligned_edge=m.LEFT)
        sit(w_head, baseline(sft_title))
        sft = m.VGroup(sft_title, rows, w_head)
        dim = m.AnimationGroup(r_user[2].animate.set_opacity(0.35), r_user[3].animate.set_opacity(0.35))

        # RL: many more samples
        lab_s = code("SFT", 22, INK, weight=m.BOLD)
        lab_r = code("RL", 22, LIME, weight=m.BOLD)
        left = -4.6
        b_s = m.RoundedRectangle(corner_radius=0.06, width=0.12, height=0.34, stroke_width=0, fill_color=MUTED, fill_opacity=1)
        b_r = m.RoundedRectangle(corner_radius=0.06, width=6.0, height=0.34, stroke_width=0, fill_color=LIME, fill_opacity=1)
        b_s.move_to([left + 0.06, -1.75, 0])
        b_r.move_to([left + 3.0, -2.45, 0])
        lab_s.next_to(b_s, m.LEFT, buff=0.35).align_to([left - 0.35, 0, 0], m.RIGHT)
        lab_r.next_to(b_r, m.LEFT, buff=0.35).align_to([left - 0.35, 0, 0], m.RIGHT)
        n_s = text("samples", 20, MUTED).next_to(b_s, m.RIGHT, buff=0.2)
        n_r = text("tens to hundreds of times more", 22, INK).next_to(b_r, m.RIGHT, buff=0.2)
        samples = m.VGroup(lab_s, lab_r, b_s, b_r, n_s, n_r)
        src = cite("Li, AI Agents in Depth · ch. 8").to_corner(m.DR, buff=0.5)

        # the shorthand
        quote = text("“SFT memorizes, RL generalizes”", 34, INK).move_to([0, -1.6, 0])
        caveat = text("shorthand: a measured result, not a law", 24, LIME).next_to(quote, m.DOWN, buff=0.22)
        gp = cite("GeneralPoints, new rules: RL +3.5 points, SFT −8.1 points · Chu et al., 2025").next_to(caveat, m.DOWN, buff=0.22)
        short = m.VGroup(quote, caveat, gp)

        self.speak(
            "[A model learns to be an agent in four stages.]",
            m.AnimationGroup(self.new_header("01 · four stages", "From library to practice"),
                             m.LaggedStart(*[m.FadeIn(s, shift=0.15 * m.UP) for s in shells], lag_ratio=0.2)),
        )
        self.speak(
            "[First, pretraining reads the whole library], [predicting each next token], [at a cost of tens of millions of dollars.]",
            m.FadeIn(bodies[0][:3], shift=0.15 * m.UP),
            m.FadeIn(bodies[0][3][0]),
            m.AnimationGroup(m.FadeIn(bodies[0][3][1]), m.FadeIn(src)),
        )
        self.speak(
            "[Mid-training adds specialist textbooks]: [domain data, mixed with replays of older data.]",
            m.FadeIn(bodies[1][:3], shift=0.15 * m.UP),
            m.FadeIn(bodies[1][3]),
        )
        self.speak(
            "[Then supervised fine-tuning, or SFT]: [a teacher demonstrates, with thousands of worked examples.]",
            m.FadeIn(bodies[2][:3], shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(bodies[2][3]), m.FadeIn(sft_title), m.FadeIn(rows, shift=0.15 * m.UP)),
        )
        self.speak(
            "[The loss counts only the response]: [the prompt is masked out.]",
            m.AnimationGroup(m.FadeIn(w_head), m.FadeIn(w_yes, shift=0.1 * m.LEFT), m.Circumscribe(r_asst, color=CORAL, buff=0.06)),
            m.AnimationGroup(m.FadeIn(w_no, shift=0.1 * m.LEFT), dim),
        )
        self.speak(
            "[Last comes reinforcement learning, or RL]: [the model works problems itself], [with tens to hundreds of times more samples.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(sft, w_no, w_yes), shift=0.15 * m.DOWN), m.FadeIn(bodies[3][:2], shift=0.15 * m.UP)),
            m.FadeIn(bodies[3][2:], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(lab_s), m.FadeIn(b_s), m.FadeIn(n_s), m.FadeIn(lab_r),
                             m.GrowFromEdge(b_r, m.LEFT), m.FadeIn(n_r, shift=0.1 * m.LEFT)),
        )
        self.speak(
            "[People say SFT memorizes and RL generalizes.] [The book calls that shorthand]: [a measured result, not a law.]",
            m.AnimationGroup(m.FadeOut(samples, shift=0.15 * m.DOWN), m.FadeIn(quote, shift=0.15 * m.UP)),
            m.FadeIn(caveat, shift=0.1 * m.UP),
            m.FadeIn(gp),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ rollouts

    def rollouts(self):
        self.next_section("2 · Rollouts", notes=f"Li, AI Agents in Depth, ch. 8 (storyboard step 1): 'The system copies the same initial code and the same problem description into 16 mutually isolated sandboxes'. Here the task is tinyshop's running bug. Each sandbox runs the reference agent_loop (Episode 2) unchanged; one full run is a rollout. The behaviours are illustrative, the three patches are Episode 29's (fix the loop, add the last item back, delete the assertion) plus 'claim success untested'; counts {COUNT}, order random.Random(30). Segment colours: tool call SUN, tool result MINT, final answer CORAL (computed from each run's messages). " + FACTS)
        tiles = [sandbox(i, r) for i, r in enumerate(RUNS)]
        self.tiles = tiles
        rack = m.VGroup(*[t.box for t in tiles])
        cap = code(f"{G} sandboxes · one task", 20, MUTED).next_to(rack, m.UP, buff=0.2).align_to(rack, m.LEFT)

        X0 = 0.6
        tcard = card(6.0, 1.45, LIME, CARD, radius=0.18, stroke=2).move_to([X0 + 3.0, 1.62, 0])
        t_cap = code("task", 20, LIME, weight=m.BOLD).move_to(tcard.get_corner(m.UL) + [0.3 + 0.24, -0.3, 0])
        t1 = code("cart.total() skips the last item.", 20, INK, t2c={r"cart\.total\(\)": AZURE})
        t2 = text("Fix it without touching the tests.", 22, INK)
        m.VGroup(t1, t2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.14).next_to(t_cap, m.DOWN, buff=0.14).align_to(t_cap, m.LEFT)
        task = m.VGroup(tcard, t_cap, t1, t2)

        mchip = icon_chip(MODEL_ICON, "current model", CORAL, CORAL_FILL, size=22, height=0.72).move_to([X0 + 1.45, 0.22, 0])
        feed = arrow(mchip.get_left(), [rack.get_right()[0] + 0.05, 0.22, 0], CORAL, buff=0.1)
        loop_note = text("runs Episode 2's loop", 20, MUTED).next_to(mchip, m.RIGHT, buff=0.25)
        rollout_pill = pill("one run = a rollout", LIME, LIME_FILL, 22).next_to(mchip, m.DOWN, buff=0.35).align_to(mchip, m.LEFT)

        legend = m.VGroup()
        desc = {"fix": "fixes the loop, runs the tests", "patch": "adds the last item back",
                "cheat": "deletes the failing assertion", "quit": "claims success, untested"}
        for j, k in enumerate(HABITS):
            name = text(TILE_NAME[k], 22, INK, weight=m.SEMIBOLD).move_to([X0 + 0.1, -1.2 - 0.55 * j, 0], aligned_edge=m.LEFT)
            d = text(desc[k], 20, MUTED).move_to([X0 + 1.45, 0, 0], aligned_edge=m.LEFT)
            n = code(f"× {COUNT[k]}", 22, SUN, weight=m.BOLD).move_to([X0 + 5.6, 0, 0], aligned_edge=m.RIGHT)
            y = -1.2 - 0.55 * j
            for w in (name, d, n):
                sit(w, y)
            legend.add(m.VGroup(name, d, n))
        self.legend, self.task_card, self.mchip, self.feed, self.loop_note, self.rollout_pill = legend, task, mchip, feed, loop_note, rollout_pill
        self.rack_cap = cap

        def by_kind(k):
            return [t for t in tiles if t.run.kind == k]

        self.speak(
            "[Here is one training step.] [The task]: [cart total skips the last item; fix it without touching the tests.]",
            self.new_header("02 · rollouts", "Sixteen tries at once"),
            m.FadeIn(m.VGroup(tcard, t_cap), shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(t1), m.FadeIn(t2, shift=0.1 * m.UP), lag_ratio=0.4),
        )
        self.speak(
            "[The trainer copies the repo and the task into sixteen isolated sandboxes.]",
            m.AnimationGroup(m.FadeIn(cap),
                             m.LaggedStart(*[m.FadeIn(m.VGroup(t.box, t.num), target_position=tcard) for t in tiles], lag_ratio=0.08)),
        )
        grow = m.LaggedStart(*[
            m.Succession(*[m.FadeIn(s, scale=0.4, run_time=0.25) for s in t.segs]) for t in tiles
        ], lag_ratio=0.04)
        self.speak(
            "[In each one, the current model runs the agent loop from episode two.] [One full run is called a rollout.]",
            m.AnimationGroup(m.FadeIn(mchip, shift=0.15 * m.UP), m.GrowArrow(feed), m.FadeIn(loop_note), grow),
            m.AnimationGroup(m.FadeIn(rollout_pill, shift=0.1 * m.UP), m.Circumscribe(tiles[0].box, color=LIME, buff=0.06)),
        )
        self.speak(
            "[The model samples, so the runs differ.] [Four fix the loop], [five patch the symptom], [three delete the failing assertion], [and four claim success without testing.]",
            m.AnimationGroup(m.FadeOut(rollout_pill), m.FadeOut(loop_note)),
            m.AnimationGroup(m.FadeIn(legend[0]), *[m.FadeIn(t.label, shift=0.08 * m.UP) for t in by_kind("fix")]),
            m.AnimationGroup(m.FadeIn(legend[1]), *[m.FadeIn(t.label, shift=0.08 * m.UP) for t in by_kind("patch")]),
            m.AnimationGroup(m.FadeIn(legend[2]), *[m.FadeIn(t.label, shift=0.08 * m.UP) for t in by_kind("cheat")]),
            m.AnimationGroup(m.FadeIn(legend[3]), *[m.FadeIn(t.label, shift=0.08 * m.UP) for t in by_kind("quit")]),
        )
        self.wait(0.3)

    # ------------------------------------------------------------------ rewards

    def rewards(self):
        self.next_section("3 · Rewards", notes=f"Li, ch. 8 (step 2): 'Suppose 4 of the 16 attempts pass all tests without touching the test files, and the other 12 fail'; reward 1 and 0. The verifier here is Episode 29's: the task's own copy of the tests (FAIL_TO_PASS and PASS_TO_PASS), executed against each sandbox's cart.py (computed: {sum(REWARDS)} pass). The deleted assertion leaves the sandbox's own suite green ({CHEAT.suite!r}) but the task's copy fails ({CHEAT.task!r}); adding the last item back crashes on the empty cart ({PATCH.suite!r}). reward() is new code in harness_reference.py's style (the scene runs exactly this function). " + FACTS)
        tiles = self.tiles
        X0 = 0.6
        block = code_block(REWARD_SRC, size=19, width=6.0, title="the verifier · reward(box)", accent=MINT)
        block.move_to([X0 + 3.0, -0.45, 0])
        passes = [t for t in tiles if t.run.reward]
        fails = [t for t in tiles if not t.run.reward]
        s1 = code(f"{len(passes)} × reward 1", 22, MINT, weight=m.BOLD)
        s2 = code(f"{len(fails)} × reward 0", 22, ROSE, weight=m.BOLD)
        summary = m.VGroup(s1, s2).arrange(m.RIGHT, buff=0.6).next_to(block, m.DOWN, buff=0.3).align_to(block, m.LEFT)
        cheats = [t for t in tiles if t.run.kind == "cheat"]
        c1 = code(f"own tests:   {CHEAT.suite}", 20, MINT)
        c2 = code(f"task's copy: {CHEAT.task}", 20, ROSE)
        lead = text("edit test", 20, INK, weight=m.SEMIBOLD)
        cheat_note = m.VGroup(lead, c1, c2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.12)
        cheat_note.next_to(summary, m.DOWN, buff=0.3).align_to(block, m.LEFT)

        sweep = m.LaggedStart(*[
            m.ShowPassingFlash(t.box.copy().set_fill(opacity=0).set_stroke(LIME, 5), time_width=0.6, run_time=0.8) for t in tiles
        ], lag_ratio=0.05)
        self.speak(
            "[Then a verifier runs the task's own copy of the tests], [in every sandbox.]",
            m.AnimationGroup(self.new_header("03 · rewards", "A verifier pays"),
                             m.FadeOut(m.VGroup(self.mchip, self.feed, self.legend), shift=0.15 * m.DOWN),
                             m.FadeIn(block, shift=0.15 * m.UP)),
            sweep,
        )
        self.speak(
            "[Four runs pass]: [reward one.] [The other twelve fail]: [reward zero.]",
            m.AnimationGroup(*[t.box.animate.set_stroke(MINT, 3) for t in passes]),
            m.AnimationGroup(*[m.FadeIn(t.badge, scale=0.6) for t in passes], m.FadeIn(s1)),
            m.AnimationGroup(*[t.box.animate.set_stroke(ROSE, 2.5) for t in fails]),
            m.AnimationGroup(*[m.FadeIn(t.badge, scale=0.6) for t in fails], m.FadeIn(s2)),
        )
        self.speak(
            "[Deleting the assertion turned the sandbox's own tests green], [but the task's copy still fails], [so no reward.]",
            m.AnimationGroup(m.FadeIn(lead), m.FadeIn(c1, shift=0.1 * m.UP), *[m.Circumscribe(t.box, color=SUN, buff=0.05) for t in cheats]),
            m.FadeIn(c2, shift=0.1 * m.UP),
            m.AnimationGroup(*[m.Circumscribe(t.badge, color=ROSE, buff=0.04) for t in cheats]),
        )
        self.verifier_block, self.summary, self.cheat_note = block, summary, cheat_note
        self.wait(0.3)

    # ------------------------------------------------------------------ advantages

    def advantages_(self):
        self.next_section("4 · Advantages", notes=f"Li, ch. 8 (step 3): 'This group's average success rate is 4/16: the 4 that passed are above the group average and get a positive advantage ... the 12 that failed are below it and get a negative advantage'; 'every reward is identical, there is no way to tell which is better, and the relative advantage vanishes'; 'This within-group comparison is the core of GRPO.' Computed: mean {MEAN}, advantages +{A_PASS} / {A_FAIL}; {len([a for a in ADV if a > 0])} × {A_PASS} = {UP} up and {len([a for a in ADV if a < 0])} × {-A_FAIL} = {DOWN} down: they sum to zero. GRPO as published (Shao et al., 2024) also divides by the group's standard deviation ({STD:.3f}), giving {NORM[0]:+} / {NORM[1]}; the book's simpler form is shown. 'RL is good at raising the probability of existing but unlikely successful behavior' (Li, ch. 8). " + FACTS)
        tiles = self.tiles
        X0 = 0.6
        f_mean = code(f"mean = {sum(REWARDS)} / {G} = {MEAN:.2f}", 26, INK, t2c={r"\d+(\.\d+)?": SUN})
        f_def = code("advantage = reward − mean", 24, LIME, weight=m.BOLD)
        f_pass = code(f"pass: 1 − {MEAN:.2f} = {signed(A_PASS)}", 24, MINT)
        f_fail = code(f"fail: 0 − {MEAN:.2f} = {signed(A_FAIL)}", 24, ROSE)
        f_up = code(f"4 × 0.75 = {UP:.2f} up", 22, MINT)
        f_dn = code(f"12 × 0.25 = {DOWN:.2f} down", 22, ROSE)
        f_sum = pill("sum = 0", LIME, LIME_FILL, 22, mono=True)
        assert f"{int(UP)}" == "3" and len([a for a in ADV if a > 0]) == 4 and len([a for a in ADV if a < 0]) == 12
        col = m.VGroup(f_def, f_mean, f_pass, f_fail).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.32)
        col.move_to([X0 + 0.15, 1.0, 0], aligned_edge=m.LEFT)
        col.align_to([0, 2.2, 0], m.UP)
        bal = m.VGroup(f_up, f_dn).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.18).next_to(col, m.DOWN, buff=0.45).align_to(col, m.LEFT)
        f_sum.next_to(bal, m.RIGHT, buff=0.5)
        grpo_src = cite("GRPO · Shao et al., 2024").to_corner(m.DR, buff=0.5)

        # nobody passes: a mini rack
        cells = m.VGroup(*[card(0.36, 0.3, ROSE, ROSE_FILL, radius=0.06, stroke=2) for _ in range(G)])
        cells.arrange_in_grid(rows=4, cols=4, buff=0.08).move_to([X0 + 0.95, -0.2, 0])
        if_cap = text("if all sixteen fail", 22, MUTED).next_to(cells, m.UP, buff=0.22)
        z1 = code(f"mean = {NONE_MEAN:g}", 24, INK)
        z2 = code("every advantage = 0", 24, ROSE)
        z = m.VGroup(z1, z2).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22).next_to(cells, m.RIGHT, buff=0.5)
        nosig = pill("no signal", ROSE, ROSE_FILL, 22).next_to(z, m.DOWN, buff=0.3).align_to(z, m.LEFT)
        quote = m.VGroup(text("“RL is good at raising the probability of existing", 22, INK),
                         text("but unlikely successful behavior.”", 22, INK)).arrange(m.DOWN, buff=0.1, aligned_edge=m.LEFT)
        quote.move_to([X0 + 3.0, -2.3, 0])
        q_src = cite("Li, AI Agents in Depth · ch. 8").next_to(quote, m.DOWN, buff=0.15).align_to(quote, m.RIGHT)

        self.speak(
            "[How much should each run count?] [GRPO compares every run with its own group.]",
            m.AnimationGroup(self.new_header("04 · advantages", "Better than the group?"),
                             m.FadeOut(m.VGroup(self.task_card, self.verifier_block, self.summary, self.cheat_note), shift=0.15 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(f_def, shift=0.1 * m.UP), m.FadeIn(grpo_src)),
        )
        self.speak(
            "[The group's mean reward is four out of sixteen]: [zero point two five.]",
            m.FadeIn(f_mean, shift=0.1 * m.UP),
            m.Circumscribe(f_mean[-4:], color=SUN, buff=0.06),
        )
        passes = [t for t in tiles if t.run.reward]
        fails = [t for t in tiles if not t.run.reward]
        self.speak(
            "[Each run's advantage is its reward minus that mean]: [plus zero point seven five for each pass], [minus zero point two five for each failure.]",
            m.Circumscribe(f_def, color=LIME, buff=0.08),
            m.AnimationGroup(m.FadeIn(f_pass, shift=0.1 * m.UP), *[m.FadeOut(t.label) for t in passes], *[m.FadeIn(t.adv) for t in passes]),
            m.AnimationGroup(m.FadeIn(f_fail, shift=0.1 * m.UP), *[m.FadeOut(t.label) for t in fails], *[m.FadeIn(t.adv) for t in fails]),
        )
        self.speak(
            "[Ups and downs cancel]: [four times zero point seven five up], [twelve times zero point two five down.]",
            m.FadeIn(f_sum, scale=0.8),
            m.FadeIn(f_up, shift=0.1 * m.UP),
            m.FadeIn(f_dn, shift=0.1 * m.UP),
        )
        self.speak(
            "[Now suppose all sixteen had failed.] [The mean is zero, so every advantage is zero]: [nothing to learn from.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(f_mean, f_pass, f_fail, f_up, f_dn, f_sum), shift=0.15 * m.DOWN),
                             m.FadeIn(if_cap), m.LaggedStart(*[m.FadeIn(c, scale=0.6) for c in cells], lag_ratio=0.05)),
            m.AnimationGroup(m.FadeIn(z1, shift=0.1 * m.UP), m.FadeIn(z2, shift=0.1 * m.UP), lag_ratio=0.5),
            m.FadeIn(nosig, scale=0.8),
        )
        self.speak(
            "[RL needs some success to start.] [In the book's words, it raises the probability of existing but unlikely successful behavior.]",
            m.Circumscribe(m.VGroup(*[t.box for t in passes]), color=MINT, buff=0.08),
            m.AnimationGroup(m.FadeOut(grpo_src), m.FadeIn(quote, shift=0.1 * m.UP), m.FadeIn(q_src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the update

    def the_update(self):
        self.next_section("5 · The update", notes=f"Rollout #{SHOWN_N:02d} (a pass, advantage +{A_PASS}), its messages computed by the reference loop. GRPO gives every token the model generated in that run the run's advantage; tool-use RL masks the environment's tokens: gradients 'only through the model's own thinking and its tool-call arguments' (Li, ch. 8), otherwise the model is trained to predict sandbox output. The prompt is masked too. Toy policy (illustrative): four habits with probabilities = the group's frequencies; one step (size {ETA}) on the GRPO surrogate (1/G)·Σ A·log π: fix {pct(PI0['fix'])}% → {pct(PI1['fix'])}%, patch {pct(PI0['patch'])}% → {pct(PI1['patch'])}%, delete {pct(PI0['cheat'])}% → {pct(PI1['cheat'])}%, untested {pct(PI0['quit'])}% → {pct(PI1['quit'])}% (computed). Li, ch. 8 (step 4): 'It does not memorize some successful patch verbatim; it adjusts gradually across many tasks and rollouts.' GRPO: no value model, the group is the baseline (Shao et al., 2024). PPO: a clipped probability ratio and a learned value model, 'which suits long trajectories that need fine-grained credit assignment' (Schulman et al., 2017; Li, ch. 8). " + FACTS)
        rows = trajectory_rows(SHOWN.messages)
        col = m.VGroup(*[r for r, _ in rows]).arrange(m.DOWN, buff=0.1)
        col.move_to([0, 0, 0]).align_to([-6.55, 0, 0], m.LEFT).align_to([0, 1.95, 0], m.UP)
        cap = code(f"rollout #{SHOWN_N:02d} · reward 1 · advantage +0.75", 20, MUTED, t2c={r"\+0\.75": MINT, r"reward 1": MINT})
        cap.next_to(col, m.UP, buff=0.2).align_to(col, m.LEFT)
        wx = col.get_right()[0] + 0.3
        weights, trained, masked = m.VGroup(), [], []
        for r, t in rows:
            w = code("+0.75", 22, MINT, weight=m.BOLD) if t else code("masked", 20, FAINT)
            w.move_to([wx + w.width / 2, 0, 0])
            sit(w, baseline(r[2]))
            weights.add(w)
            (trained if t else masked).append((r, w))
        dims = m.AnimationGroup(*[m.AnimationGroup(r[2].animate.set_opacity(0.35), r[3].animate.set_opacity(0.35)) for r, _ in masked])

        # the policy
        X0 = 2.0
        p_cap = code("the policy, for example", 20, MUTED).move_to([X0, 2.15, 0], aligned_edge=m.LEFT)
        bars, ghosts, labels, pcts_after = [], m.VGroup(), m.VGroup(), []
        pct_slots = []
        bar_left = X0
        for j, k in enumerate(HABITS):
            y_lab = 1.55 - 0.95 * j
            lab = text(HABIT_NAME[k], 22, INK).move_to([X0, y_lab, 0], aligned_edge=m.LEFT)
            color = MINT if k == "fix" else ROSE
            b0 = bar_rect(PI0[k], color, bar_left, y_lab - 0.42)
            b1 = bar_rect(PI1[k], color, bar_left, y_lab - 0.42)
            g = bar_rect(PI0[k], color, bar_left, y_lab - 0.42).set_fill(opacity=0).set_stroke(FAINT, 2)
            px = bar_left + BAR_SCALE * max(PI0.values()) + 0.3
            p0 = code(f"{pct(PI0[k])}%", 22, INK).move_to([px, y_lab - 0.42, 0], aligned_edge=m.LEFT)
            p1 = code(f"{pct(PI1[k])}%", 22, MINT if k == "fix" else INK, weight=m.BOLD if k == "fix" else m.NORMAL)
            p1.move_to([px, y_lab - 0.42, 0], aligned_edge=m.LEFT)
            labels.add(lab)
            bars.append((b0, b1))
            ghosts.add(g)
            pct_slots.append((p0, p1))
        bar_group = m.VGroup(*[b for b, _ in bars])
        pcts0 = m.VGroup(*[p for p, _ in pct_slots])
        habits_note = text("habits shift; no patch is memorized", 20, MUTED).move_to([X0, -2.35, 0], aligned_edge=m.LEFT)

        # GRPO vs PPO (in the column's place)
        def algo_card(name, lines, src_, accent):
            box = card(3.6, 2.75, accent, CARD, radius=0.18, stroke=2)
            head = title_text(name, 30, accent)
            body = m.VGroup(*[text(s_, 20, INK) for s_ in lines]).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.16)
            for i_, line_ in enumerate(body[1:], 1):
                sit(line_, baseline(body[0]) - 0.36 * i_)
            s_ = cite(src_)
            inner = m.VGroup(head, body, s_).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.22)
            inner.move_to(box).align_to(box, m.LEFT).shift(0.3 * m.RIGHT)
            assert inner.width < box.width - 0.45, (name, inner.width)
            return m.VGroup(box, inner)

        grpo = algo_card("GRPO", ["baseline: the group mean", "no value network", "one score per run"], "Shao et al., 2024", LIME)
        ppo = algo_card("PPO", ["baseline: a value model", "clipped update ratio", "credit step by step"], "Schulman et al., 2017", AZURE)
        algos = m.VGroup(grpo, ppo).arrange(m.RIGHT, buff=0.25).move_to([0, 0, 0]).align_to([-6.55, 0, 0], m.LEFT).set_y(-0.15)

        self.speak(
            "[Here's one passing rollout, as the trainer sees it.]",
            m.AnimationGroup(self.new_header("05 · the update", "Nudge the policy"), m.FadeIn(cap),
                             m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r, _ in rows], lag_ratio=0.12)),
        )
        self.speak(
            "[Every token Otto wrote], [its reasoning and its tool calls], [gets that run's advantage: plus zero point seven five.]",
            m.AnimationGroup(*[m.Circumscribe(r, color=CORAL, buff=0.04) for r, _ in trained]),
            m.AnimationGroup(*[m.Circumscribe(r[3], color=SUN, buff=0.04) for r, _ in trained]),
            m.LaggedStart(*[m.FadeIn(w, shift=0.1 * m.LEFT) for _, w in trained], lag_ratio=0.2),
        )
        self.speak(
            "[The sandbox's output is masked]: [otherwise the model would be trained to predict sandbox output.]",
            m.AnimationGroup(dims, *[m.FadeIn(w, shift=0.1 * m.LEFT) for _, w in masked]),
            m.AnimationGroup(*[m.Circumscribe(r, color=FAINT, buff=0.04) for r, _ in masked[1:]]),
        )
        self.speak(
            "[Then one update nudges the policy.] [For example], [fixing and then testing rises from twenty-five to thirty percent.]",
            m.AnimationGroup(m.FadeIn(p_cap), m.FadeIn(labels), *[m.GrowFromEdge(b, m.LEFT) for b, _ in bars], m.FadeIn(pcts0)),
            m.AnimationGroup(m.FadeIn(ghosts)),
            m.AnimationGroup(*[m.Transform(b0, b1) for b0, b1 in bars],
                             *[m.FadeOut(p0, shift=0.1 * m.UP) for p0, _ in pct_slots],
                             *[m.FadeIn(p1, shift=0.1 * m.UP) for _, p1 in pct_slots]),
        )
        self.speak(
            "[It doesn't memorize a patch]: [across many tasks and steps, it shifts habits.]",
            m.FadeIn(habits_note, shift=0.1 * m.UP),
            m.Circumscribe(m.VGroup(labels[0], pct_slots[0][1]), color=MINT, buff=0.08),
        )
        self.speak(
            "[GRPO needs no value network]: [the group is the baseline.] [PPO trains a value model and clips each update], [which suits long trajectories.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(col, cap, weights), shift=0.15 * m.DOWN), m.FadeIn(grpo, shift=0.15 * m.UP)),
            m.Circumscribe(grpo[1][1][0], color=LIME, buff=0.05),
            m.FadeIn(ppo, shift=0.15 * m.UP),
            m.Circumscribe(ppo[1][1][2], color=AZURE, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the cost

    def cost(self):
        self.next_section("6 · The cost", notes=f"Li, ch. 8: 'Suppose the slowest rollout takes about 2,000 seconds', training 'about 600 seconds': {STEP_S} s ≈ {round(STEP_S / 60)} minutes a step; '100 consecutive steps come to nearly 72 hours' (computed: {100 * STEP_S / 3600:.1f} h). The other fifteen durations are illustrative (seeded). " + FACTS)
        x0, x1 = -5.3, 5.6
        sc = (x1 - x0) / STEP_S

        def X(t):
            return x0 + sc * t

        top, lane_h, step_ = 1.9, 0.11, 0.185
        lanes, waits = m.VGroup(), m.VGroup()
        for i, d in enumerate(DURATIONS):
            y = top - i * step_
            lanes.add(m.RoundedRectangle(corner_radius=0.04, width=sc * d, height=lane_h, stroke_width=0,
                                         fill_color=CORAL, fill_opacity=1).move_to([X(d / 2), y, 0]))
            if d < SLOWEST_S:
                waits.add(m.DashedLine([X(d) + 0.06, y, 0], [X(SLOWEST_S) - 0.04, y, 0], dash_length=0.06).set_stroke(FAINT, 2))
        bottom = top - (G - 1) * step_
        slow = m.DashedLine([X(SLOWEST_S), top + 0.25, 0], [X(SLOWEST_S), bottom - 0.25, 0], dash_length=0.1).set_stroke(SUN, 3)
        slow_lab = text("slowest rollout", 20, SUN).next_to(slow, m.UP, buff=0.1)
        train = card(sc * TRAIN_S - 0.08, top - bottom + 0.4, LIME, LIME_FILL, radius=0.12, stroke=2)
        train.move_to([X(SLOWEST_S + TRAIN_S / 2) + 0.04, (top + bottom) / 2, 0])
        train_lab = m.VGroup(text("train", 24, INK, weight=m.BOLD), code("≈ 600 s", 20, LIME)).arrange(m.DOWN, buff=0.1).move_to(train)
        axis = m.Line([x0, bottom - 0.45, 0], [x1, bottom - 0.45, 0]).set_stroke(FAINT, 2)
        ticks = m.VGroup()
        for t, lab in ((0, "0 s"), (1000, "1,000 s"), (SLOWEST_S, "2,000 s"), (STEP_S, "2,600 s")):
            tk = m.Line([X(t), bottom - 0.52, 0], [X(t), bottom - 0.38, 0]).set_stroke(FAINT, 2)
            lb = code(lab, 20, MUTED).next_to(tk, m.DOWN, buff=0.1)
            ticks.add(m.VGroup(tk, lb))
        lane_cap = code(f"{G} rollouts, in parallel", 20, MUTED).move_to([x0, top + 0.35, 0], aligned_edge=m.LEFT)

        big1 = title_text(f"≈ {round(STEP_S / 60)} min per step", 36, INK)
        big2 = title_text(f"100 steps ≈ {round(100 * STEP_S / 3600)} hours", 36, LIME)
        bigs = m.VGroup(big1, big2).arrange(m.RIGHT, buff=1.0).move_to([0, -2.85, 0])
        assert bigs.width < 12.4, bigs.width
        src = cite("Li, AI Agents in Depth · ch. 8")
        src.next_to(bigs, m.DOWN, buff=0.2).align_to(bigs, m.RIGHT)

        self.speak(
            "[A step waits for its slowest rollout], [about two thousand seconds], [then trains for about ten minutes more.]",
            m.AnimationGroup(self.new_header("06 · the cost", "Wait for the slowest"), m.FadeIn(lane_cap), m.FadeIn(axis), m.FadeIn(ticks),
                             *[m.GrowFromEdge(l_, m.LEFT, run_time=0.6 + 1.6 * d / SLOWEST_S) for l_, d in zip(lanes, DURATIONS)]),
            m.AnimationGroup(m.FadeIn(waits), m.Create(slow), m.FadeIn(slow_lab, shift=0.1 * m.DOWN)),
            m.AnimationGroup(m.FadeIn(train, shift=0.15 * m.LEFT), m.FadeIn(train_lab)),
        )
        self.speak(
            "[That's about forty-three minutes a step], [so a hundred steps take about seventy-two hours.]",
            m.FadeIn(big1, shift=0.15 * m.UP),
            m.AnimationGroup(m.FadeIn(big2, shift=0.15 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ reward design

    def reward_design(self):
        self.next_section("7 · Reward design", notes=f"Same sixteen rollouts, two rewards: 'is the sandbox's own suite green?' pays {sum(NAIVE)} runs (the {COUNT['cheat']} deletions included; mean {N_MEAN}); one step on the toy policy then raises 'delete the assertion' {pct(PI0['cheat'])}% → {pct(PI_HACK['cheat'])}% (computed), while the task's own tests pay {sum(REWARDS)} and raise fix-and-test {pct(PI0['fix'])}% → {pct(PI1['fix'])}%. Li, ch. 8: reward hacking is 'exploiting a rule or an implementation hole to score highly'; 'an incomplete verifier can also produce reward hacking'; an Agent 'may achieve surface success by editing the test file, skipping authentication or running a destructive command'; verifiable rewards (tests, state checks) are the most reliable; RLVP: 'reward the outcome, penalize the path' (the author's own work; on TerminalBench, violations 3.71 → 0.66 at essentially the same success rate, controlled proxy environments); 'dense feedback buys convergence speed, sparse feedback buys exploration space.' Path-penalty examples are illustrative. Verified 'done': Episode 22. " + FACTS)
        # the table: behaviour · runs · two rewards
        cols_x = [-6.4, -2.55, -0.85, 1.1]
        heads = [text("behaviour", 24, MUTED), text("runs", 24, MUTED), text("suite green?", 24, MUTED), text("task's tests", 24, MUTED)]
        y_head = 2.15
        for h, x in zip(heads, cols_x):
            h.move_to([x, 0, 0], aligned_edge=m.LEFT) if x == cols_x[0] else h.move_to([x + 0.75, 0, 0])
            sit(h, y_head)
        rule = m.Line([cols_x[0], y_head - 0.25, 0], [cols_x[3] + 1.75, y_head - 0.25, 0]).set_stroke(EDGE, 2)
        naive_of = {k: int(k in ("fix", "cheat")) for k in HABITS}
        task_of = {k: int(k == "fix") for k in HABITS}
        assert all(r.naive == naive_of[r.kind] and r.reward == task_of[r.kind] for r in RUNS)
        table_rows, naive_cells, task_cells = m.VGroup(), [], []
        for j, k in enumerate(HABITS):
            y = 1.6 - 0.56 * j
            name = text(HABIT_NAME[k], 22, INK).move_to([cols_x[0], 0, 0], aligned_edge=m.LEFT)
            n = code(str(COUNT[k]), 22, INK).move_to([cols_x[1] + 0.75, 0, 0])
            a = code(str(naive_of[k]), 24, MINT if naive_of[k] else ROSE, weight=m.BOLD).move_to([cols_x[2] + 0.75, 0, 0])
            b = code(str(task_of[k]), 24, MINT if task_of[k] else ROSE, weight=m.BOLD).move_to([cols_x[3] + 0.75, 0, 0])
            for w in (name, n, a, b):
                sit(w, y)
            naive_cells.append(a)
            task_cells.append(b)
            table_rows.add(m.VGroup(name, n))
        y_mean = 1.6 - 0.56 * 4 - 0.05
        mean_lab = text("mean", 24, MUTED).move_to([cols_x[0], 0, 0], aligned_edge=m.LEFT)
        mean_a = code(f"{N_MEAN:.2f}", 22, SUN).move_to([cols_x[2] + 0.75, 0, 0])
        mean_b = code(f"{MEAN:.2f}", 22, SUN).move_to([cols_x[3] + 0.75, 0, 0])
        for w in (mean_lab, mean_a, mean_b):
            sit(w, y_mean)
        rule2 = m.Line([cols_x[0], y_mean + 0.33, 0], [cols_x[3] + 1.75, y_mean + 0.33, 0]).set_stroke(EDGE, 2)
        table = m.VGroup(*heads, rule, table_rows, *naive_cells, *task_cells, rule2, mean_lab, mean_a, mean_b)
        frame_a = m.SurroundingRectangle(m.VGroup(heads[2], naive_cells[-1], mean_a), buff=0.14, corner_radius=0.1).set_stroke(ROSE, 2.5)
        frame_b = m.SurroundingRectangle(m.VGroup(heads[3], task_cells[-1], mean_b), buff=0.14, corner_radius=0.1).set_stroke(MINT, 2.5)

        # the hack, grown by the update
        X0 = 3.6
        h_cap = text("delete the assertion", 22, INK).move_to([X0, 1.6, 0], aligned_edge=m.LEFT)
        hb0 = bar_rect(PI0["cheat"], ROSE, X0, 1.05)
        hb1 = bar_rect(PI_HACK["cheat"], ROSE, X0, 1.05)
        hg = bar_rect(PI0["cheat"], ROSE, X0, 1.05).set_fill(opacity=0).set_stroke(FAINT, 2)
        hp0 = code(f"{pct(PI0['cheat'])}%", 22, INK).next_to(hb0, m.RIGHT, buff=0.18)
        hp1 = code(f"{pct(PI_HACK['cheat'])}%", 22, ROSE, weight=m.BOLD).next_to(hb1, m.RIGHT, buff=0.18)
        h_note = text("after one update", 20, MUTED).next_to(hb0, m.DOWN, buff=0.22).align_to(hb0, m.LEFT)
        hack = pill("reward hacking", ROSE, ROSE_FILL, 24).move_to([X0 + 1.1, -0.15, 0])
        definition = text("a high score through a loophole, not the task", 22, MUTED).next_to(table, m.DOWN, buff=0.5)
        definition.move_to([0, definition.get_y(), 0])

        # principles
        p1 = m.VGroup(text("Verifiable rewards are the most reliable", 30, INK, weight=m.BOLD),
                      text("tests and state checks, as in Episode 22's verified done", 22, MUTED)).arrange(m.DOWN, buff=0.14)
        out_ok = word_chip("✓ outcome: the task's tests pass", MINT, MINT_FILL, size=20, height=0.58, pad=0.45)
        bad1 = word_chip("✗ path: edited a test file", ROSE, ROSE_FILL, size=20, height=0.58, pad=0.45)
        bad2 = word_chip("✗ path: destructive command", ROSE, ROSE_FILL, size=20, height=0.58, pad=0.45)
        chips_ = m.VGroup(out_ok, bad1, bad2).arrange(m.RIGHT, buff=0.25)
        p_head = text("reward the outcome, penalize the path", 22, LIME, weight=m.SEMIBOLD)
        p_src = cite("Li, AI Agents in Depth · ch. 8 (RLVP)")
        quote = m.VGroup(text("“Dense feedback buys convergence speed,", 30, INK),
                         text("sparse feedback buys exploration space.”", 30, INK)).arrange(m.DOWN, buff=0.16)
        lower = m.VGroup(p1, m.VGroup(p_head, chips_, p_src).arrange(m.DOWN, buff=0.2), quote)
        lower.arrange(m.DOWN, buff=0.55).move_to([0, -0.4, 0])
        assert chips_.width < 12.4 and lower.height < 6.0, (chips_.width, lower.height)

        self.speak(
            "[The reward decides what grows.] [Suppose it only asked]: [is the sandbox's own suite green?]",
            m.AnimationGroup(self.new_header("07 · reward design", "Reward what you mean"),
                             m.FadeIn(m.VGroup(*heads[:2], rule, table_rows, rule2, mean_lab))),
            m.FadeIn(heads[2]),
            m.AnimationGroup(*[m.FadeIn(c, shift=0.1 * m.DOWN) for c in naive_cells], m.FadeIn(mean_a), m.Create(frame_a)),
        )
        self.speak(
            "[Then the three deletions earn reward one], [and the same update raises deleting the assertion], [from nineteen to twenty-one percent.]",
            m.Circumscribe(m.VGroup(table_rows[2], naive_cells[2]), color=ROSE, buff=0.08),
            m.AnimationGroup(m.FadeIn(h_cap), m.GrowFromEdge(hb0, m.LEFT), m.FadeIn(hp0)),
            m.AnimationGroup(m.FadeIn(hg), m.Transform(hb0, hb1), m.FadeOut(hp0, shift=0.1 * m.UP), m.FadeIn(hp1, shift=0.1 * m.UP), m.FadeIn(h_note)),
        )
        self.speak(
            "[That's reward hacking]: [a high score through a loophole, not by doing the task.]",
            m.FadeIn(hack, scale=0.8),
            m.FadeIn(definition, shift=0.1 * m.UP),
        )
        self.speak(
            "[The task's own tests pay only the real fix.] [Verifiable rewards like these are the most reliable.]",
            m.AnimationGroup(m.FadeIn(heads[3]), *[m.FadeIn(c, shift=0.1 * m.DOWN) for c in task_cells], m.FadeIn(mean_b), m.Create(frame_b)),
            m.AnimationGroup(m.FadeOut(m.VGroup(*heads[:3], rule, table_rows, *naive_cells, rule2, mean_lab, mean_a, frame_a,
                                                h_cap, hb0, hg, hp1, h_note, hack, definition), shift=0.15 * m.UP),
                             m.FadeOut(heads[3], shift=0.15 * m.UP), *[m.FadeOut(c, shift=0.15 * m.UP) for c in task_cells],
                             m.FadeOut(mean_b, shift=0.15 * m.UP), m.FadeOut(frame_b, shift=0.15 * m.UP),
                             m.FadeIn(p1, shift=0.15 * m.UP)),
        )
        self.speak(
            "[And they can punish bad paths], [not only bad outcomes.]",
            m.AnimationGroup(m.FadeIn(p_head), m.FadeIn(out_ok, shift=0.1 * m.UP), m.FadeIn(p_src)),
            m.LaggedStart(m.FadeIn(bad1, shift=0.1 * m.UP), m.FadeIn(bad2, shift=0.1 * m.UP), lag_ratio=0.4),
        )
        self.speak(
            "One more trade-off, in the book's words: [dense feedback buys convergence speed], [sparse feedback buys exploration space.]",
            m.FadeIn(quote[0], shift=0.1 * m.UP),
            m.FadeIn(quote[1], shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ distillation

    def distillation(self):
        self.next_section("8 · Distillation", notes="Li, ch. 8: on-policy distillation: the teacher supplies a full next-token distribution at the states the student actually reaches (the student's own rollout), so one rollout produces dense supervision; on tasks such as mathematics, 'reaching comparable performance takes roughly one tenth the training steps of pure RL'. The tokens and the teacher's per-token marks are illustrative. " + FACTS)
        n_tok, size, gap = 12, 0.38, 0.1
        x_start = -3.2

        def strip(y):
            return m.VGroup(*[
                m.RoundedRectangle(corner_radius=0.06, width=size, height=size, stroke_width=0, fill_color=CORAL, fill_opacity=1)
                .move_to([x_start + i * (size + gap), y, 0]) for i in range(n_tok)
            ])

        y_a, y_b = 1.35, -1.25
        lab_a = title_text("RL", 30, INK).move_to([-6.45, y_a, 0], aligned_edge=m.LEFT)
        lab_b = m.VGroup(title_text("on-policy", 24, INK), title_text("distillation", 24, INK)).arrange(m.DOWN, aligned_edge=m.LEFT, buff=0.08)
        lab_b.move_to([-6.45, y_b, 0], aligned_edge=m.LEFT)
        sa, sb = strip(y_a), strip(y_b)
        own_a = text("the student's own rollout", 20, MUTED).next_to(sa, m.DOWN, buff=0.2).align_to(sa, m.LEFT)
        rew = badge(1, 22).scale(1.25).next_to(sa, m.RIGHT, buff=0.45)
        one = text("one number per rollout", 22, INK).next_to(rew, m.RIGHT, buff=0.3)
        rng = random.Random(8)
        marks = m.VGroup()
        for t in sb:
            ok = rng.random() > 0.22
            h = 0.18 + 0.5 * rng.random()
            marks.add(m.RoundedRectangle(corner_radius=0.04, width=0.12, height=h, stroke_width=0,
                                         fill_color=MINT if ok else ROSE, fill_opacity=1).next_to(t, m.UP, buff=0.12))
        teacher = icon_chip(TEACHER_ICON, "teacher", LIME, LIME_FILL, size=22, height=0.72).next_to(sb, m.RIGHT, buff=0.45)
        teacher.align_to(marks, m.UP).shift(0.05 * m.UP)
        every = text("a target at every token", 22, INK).next_to(teacher, m.DOWN, buff=0.25).align_to(teacher, m.LEFT)
        own_b = text("the student's own rollout", 20, MUTED).next_to(sb, m.DOWN, buff=0.2).align_to(sb, m.LEFT)
        tenth = pill("≈ 1/10 the training steps of pure RL, on math", LIME, LIME_FILL, 22).move_to([0, -3.0, 0])
        src = cite("Li, AI Agents in Depth · ch. 8").next_to(tenth, m.RIGHT, buff=0.3)
        if src.get_right()[0] > 6.6:
            m.VGroup(tenth, src).shift((6.6 - src.get_right()[0]) * m.RIGHT)

        self.speak(
            "[RL gives one number per rollout.]",
            m.AnimationGroup(self.new_header("08 · distillation", "A teacher on every token"), m.FadeIn(lab_a), m.FadeIn(sa, shift=0.1 * m.UP),
                             m.FadeIn(own_a), m.FadeIn(rew, scale=0.6), m.FadeIn(one)),
        )
        self.speak(
            "[On-policy distillation adds a stronger teacher]: [at every token of the student's own run, it shows what it would have written.]",
            m.AnimationGroup(m.FadeIn(lab_b), m.FadeIn(sb, shift=0.1 * m.UP), m.FadeIn(own_b), m.FadeIn(teacher, shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.LaggedStart(*[m.GrowFromEdge(k, m.DOWN) for k in marks], lag_ratio=0.1), m.FadeIn(every)),
        )
        self.speak(
            "[In the book's math example, it reached similar results in about a tenth of the training steps of pure RL.]",
            m.AnimationGroup(m.FadeIn(tenth, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Sample (sixteen rollouts of one task in isolated sandboxes), score (a verifier pays the reward; design it so loopholes earn nothing), nudge (advantage = reward − group mean; one small step toward what worked). Next: Episode 31, agents that improve from their own traces without new weights. " + FACTS)
        halo = glow(LIME, radius=4.0)
        words = m.VGroup(*[title_text(w, 66, c) for w, c in (("Sample", CORAL), ("Score", MINT), ("Nudge", LIME))])
        subs = m.VGroup(*[code(s_, 20, MUTED) for s_ in ("16 rollouts, one task", "a verifier pays", "reward − group mean")])
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
        tag1 = text("Practice many times, reward only what can be checked,", 30, MUTED)
        tag2 = text("and nudge the model toward what worked.", 30, MUTED)
        tagline = m.VGroup(tag1, tag2).arrange(m.DOWN, buff=0.2).next_to(row_, m.DOWN, buff=0.65)
        assert row_.width < 12.8 and tagline.width < 12.8, (row_.width, tagline.width)

        self.speak(
            "[To sum up]: [sample], [score], [nudge.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.speak(
            "[Practice many times, reward only what a verifier can check], [and nudge the model toward what worked.]",
            m.FadeIn(tag1, shift=0.1 * m.UP),
            m.FadeIn(tag2, shift=0.1 * m.UP),
        )
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
