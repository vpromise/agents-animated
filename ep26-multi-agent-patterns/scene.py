"""Agents, Animated · Episode 26 · Multi-Agent Patterns (Module 5 · Scaling Up)

When many agents beat one, and how to organise them. The lesson is the book's skepticism: a second agent
pays off when it brings information one agent could not get (tests, screenshots, tools) or isolation (a
clean context); otherwise it mostly costs tokens. Running example: Otto's patch for a tinyshop ticket, "Free
shipping on orders of $50 or more". Otto writes `if subtotal > 50:`; asked to review its own work (same
model, same context) it approves; a reviewer agent in its own context, given only the ticket, the diff and a
test runner, runs the ticket's tests, finds that $50.00 exactly still pays $5.00 shipping, and sends the
failing test back; Otto changes one character and the suite passes. The patch and the tests are real code:
the scene runs them (asserted below), so every number on screen comes from that run.

Sections: hook (Episode 24's team; the book's core test) · self-review (same model, same blind spot) ·
proposer–reviewer (evidence, read-only tests) · new information (Table 10-1; debate; WebGen-Agent) · the
map (shared vs separate context × peer / manager / decentralized) · the cost (~15× tokens) · failure
modes (MAST; lossy handoffs; runaway spawning) · A2A · recap: evidence, isolation, budget.

Sources (facts dated October 2026):
- Li Bojie, AI Agents in Depth, ch. 10 (bojieli.github.io/ai-agent-book/book-en/chapter10/): the core test
  "Does the collaboration provide information that a single Agent could not obtain while producing its
  answer?"; Table 10-1: self-review by the same model → no new information, "Usually ineffective or even
  harmful"; agents debating the same text → no, "Comparable to a single Agent with equal compute"; a
  reviewer using test results / rendered screenshots / external tools → yes (execution / visual / tool
  feedback), "Significant improvement"; WebGen-Agent's visual feedback "improved Claude 3.5 Sonnet's
  benchmark performance from 26.4% to 51.9%"; debate: Tran & Kiela (2026) at a fixed thinking-token budget,
  "the single Agent performed on par with or even better than the multi-agent systems"; Proposer–Reviewer:
  "The reviewer must not be able to modify the tests, the evidence collector, or the release gate;
  otherwise 'independent verification' degenerates into self-approval"; the two dimensions (shared vs
  non-shared context; peer / manager / decentralized); the manager pattern "models each specialized Agent
  as a tool"; decentralized handoffs (OpenAI Swarm) carry a remaining budget and a visited_agents list,
  kept by the runtime, against cycles; failure modes: MAST's 14 modes in three groups; "every handoff is a
  lossy re-encoding"; runaway spawning ("may spawn thousands of sub-agents") and "use a dedicated API key
  as part of controlling token spending"; A2A: Agent Card, task lifecycle, opaque collaboration, released
  by Google in 2025 and donated to the Linux Foundation.
- Anthropic, "How we built our multi-agent research system" (13 June 2025): agents "typically use about 4×
  more tokens than chat interactions, and multi-agent systems use about 15× more tokens than chats";
  "token usage by itself explains 80% of the variance" (BrowseComp; three factors explain 95%); Opus 4 lead
  + Sonnet 4 subagents "outperformed single-agent Claude Opus 4 by 90.2% on our internal research eval";
  suited to heavy parallelization; "most coding tasks involve fewer truly parallelizable tasks than
  research"; early agents spawned "50 subagents for simple queries" (fixed with scaling rules in prompts).
- Cemri et al., "Why Do Multi-Agent LLM Systems Fail?" (arXiv 2503.13657, 2025): MAST, 14 failure modes in
  3 categories: system design issues (5), inter-agent misalignment (6), task verification (3).
- A2A (a2a-protocol.org, read October 2026): Agent Card at /.well-known/agent-card.json (name, description,
  url, skills…); TaskState submitted, working, input-required, auth-required, completed, failed, canceled,
  rejected (we show five); agents collaborate "without needing to share their internal thoughts, plans, or
  tool implementations"; "A2A connects the agents to each other; MCP connects each agent to its own tools".
  The project sits at the Linux Foundation (its site notes it joined the Agentic AI Foundation in 2026).

Illustrative (said "for example" or drawn as our example): the shipping ticket, the patch, the ticket's
three tests, tinyshop's 13 older tests (from harness_reference.py: "1 failed, 12 passed" before Episode 2's
fix), the self-review's wording, the hand-off chain's wording, the carrier's agent and its Agent Card.
Simplified: pytest -v output shows test names without the file prefix.

    VOICE=draft manimgx inspect scene.py     # silent draft
    manimgx render scene.py                   # with the narrator serving from the series folder
"""

import difflib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from kit import *  # noqa: E402,F403

# ---------------------------------------------------------------------------- the review, computed

TICKET = "Free shipping on orders of $50 or more."
REVIEW_ASK = "Review your patch. Is it correct?"
SELF_REVIEW = "Looks correct: free shipping over $50. Approved."

PATCH = """def shipping(subtotal):
    if subtotal > 50:
        return 0.00
    return 5.00"""
FIXED = PATCH.replace("> 50", ">= 50")


def load(src):
    space = {}
    exec(src, space)
    return space["shipping"]


# the ticket's acceptance tests, written from the ticket (read-only for both agents)
TICKET_TESTS = [("test_below_50", 49.99, 5.00), ("test_at_50", 50.00, 0.00), ("test_above_50", 80.00, 0.00)]
OLD_SUITE = 13  # tinyshop's tests after Episode 2's fix (harness_reference: "1 failed, 12 passed" before it)


def run_ticket_tests(src):
    shipping = load(src)
    return [(name, shipping(x), want) for name, x, want in TICKET_TESTS]


def whole_suite(src):
    """pytest -q over everything: the older tests never call shipping(), so they pass either way."""
    ok = sum(got == want for _, got, want in run_ticket_tests(src))
    return OLD_SUITE + ok, len(TICKET_TESTS) - ok


V1 = run_ticket_tests(PATCH)
V2 = run_ticket_tests(FIXED)
FAILS = [(n, got, want) for n, got, want in V1 if got != want]
assert FAILS == [("test_at_50", 5.0, 0.0)]  # "at exactly fifty dollars, the customer still pays five dollars"
assert [got == want for _, got, want in V1] == [True, False, True]  # below passes, at fails, above passes
assert all(got == want for _, got, want in V2)
assert whole_suite(FIXED) == (16, 0)  # "the whole suite passes: sixteen tests"
assert [d for d in difflib.ndiff(PATCH, FIXED) if d[0] in "+-"] == ["+ ="]  # "one character"
assert "or more" in TICKET and "over $50" in SELF_REVIEW  # the ticket, and Otto's reading of it

TERM_V1 = ["$ pytest -v test_shipping.py"] + [
    ("" if got == want else "! ") + f"{name:<16}{'PASSED' if got == want else 'FAILED'}" for name, got, want in V1
] + [f"! E   assert {FAILS[0][1]} == {FAILS[0][2]}"]
assert TERM_V1[2] == "! test_at_50      FAILED" and TERM_V1[4] == "! E   assert 5.0 == 0.0"
PASSED2, FAILED2 = whole_suite(FIXED)
TERM_V2 = ["$ pytest -q", "." * PASSED2 + "  [100%]", f"{PASSED2} passed"]

# Anthropic, 2025 (multi-agent research system)
TOKENS_X = [("a chat", 1, MUTED), ("one agent", 4, CORAL), ("multi-agent system", 15, SUN)]
GAIN = 90.2  # % better than single-agent Claude Opus 4 on Anthropic's internal research eval
VARIANCE = 80  # % of the performance variance (BrowseComp) explained by token usage alone
SPAWNED = 50  # "spawning 50 subagents for simple queries"

# Li, ch. 10 (WebGen-Agent, Claude 3.5 Sonnet, with visual feedback)
WEBGEN = (26.4, 51.9)

# Li, ch. 10, Table 10-1 (lightly shortened)
REVIEWS = [
    ("the same model re-reads its output", "No", "usually ineffective or even harmful", ROSE),
    ("agents debate the same text", "No", "like one agent with equal compute", MUTED),
    ("a reviewer runs the tests", "Yes · execution", "significant improvement", MINT),
    ("a reviewer sees screenshots", "Yes · visual", "significant improvement", MINT),
    ("a reviewer checks facts with tools", "Yes · tools", "significant improvement", MINT),
]
assert sum(r[1] == "No" for r in REVIEWS) == 2 and sum(r[1].startswith("Yes") for r in REVIEWS) == 3

# Cemri et al., 2025 (MAST)
MAST = [
    ("system design", ["disobey task specification", "disobey role specification", "step repetition",
                       "loss of conversation history", "unaware of termination conditions"]),
    ("inter-agent misalignment", ["conversation reset", "fail to ask for clarification", "task derailment",
                                  "information withholding", "ignored other agent's input", "reasoning-action mismatch"]),
    ("task verification", ["premature termination", "no or incomplete verification", "incorrect verification"]),
]
assert len(MAST) == 3 and sum(len(modes) for _, modes in MAST) == 14  # "fourteen failure modes, in three groups"

# the ticket passed from agent to agent (illustrative): each hop re-encodes it in its own words
HOPS = [
    ("ticket", AZURE, AZURE_FILL, TICKET),
    ("planner", CORAL, CORAL_FILL, "Free shipping on orders over $50."),
    ("coder", ORCHID, ORCHID_FILL, "Free shipping above $50."),
    ("tester", SUN, SUN_FILL, "Free shipping on big orders."),
]


def changed_words(before, after):
    """The words of `after` that `before` didn't have (a word diff)."""
    a, b = [w.strip(".,") for w in before.split()], [w.strip(".,") for w in after.split()]
    out = set()
    for tag, _, _, j1, j2 in difflib.SequenceMatcher(a=a, b=b).get_opcodes():
        if tag in ("replace", "insert"):
            out.update(range(j1, j2))
    return [b[j] for j in sorted(out)]


assert changed_words(HOPS[0][3], HOPS[1][3]) == ["over"]  # "of … or more" became "over": $50 itself is lost
assert "50" not in HOPS[-1][3]  # by the last hop, the number is gone too

# A2A (a2a-protocol.org): the task lifecycle we show, and an Agent Card (illustrative values)
STATES = ["submitted", "working", "input-required", "completed"]
CARD_SRC = """{
  "name": "parcel-agent",
  "description": "Ships parcels",
  "url": "https://ship.example/a2a",
  "skills": [{"id": "quote", ...}]
}"""

# ---------------------------------------------------------------------------- look

FLAT = set("abcdefhiklmnorstuvwxzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
X_MIN, X_MAX = -6.5, 6.5  # content keeps half a unit from the frame's sides (the frame is about 14.2 wide)


def cite(s):
    return text(s, 20, MUTED)


def sit(words, base):
    """Put a Text's first flat-bottomed glyph on a baseline."""
    k = 0
    for ch in words.text:
        if ch.isspace():
            continue
        if ch in FLAT:
            break
        k += 1
    k = min(k, len(words) - 1)
    return words.shift((base - words[k].get_bottom()[1]) * m.UP)


def leaves(animation):
    subs = getattr(animation, "animations", None)
    if subs:
        for sub in subs:
            yield from leaves(sub)
    else:
        yield animation


def cross(at, size=0.26, color=ROSE, stroke=5):
    a, b = np.array([size / 2, size / 2, 0]), np.array([size / 2, -size / 2, 0])
    at = np.array(at, dtype=float)
    return m.VGroup(m.Line(at - a, at + a), m.Line(at - b, at + b)).set_stroke(color, stroke)


def tick(at, size=0.3, color=MINT, stroke=5):
    at = np.array(at, dtype=float)
    pts = [at + np.array(p) * size for p in ((-0.5, 0.0, 0), (-0.15, -0.38, 0), (0.55, 0.45, 0))]
    return m.VMobject().set_points_as_corners(pts).set_stroke(color, stroke)


def badge(color, fill, name, size=0.6, side=m.DOWN, sub=None, name_size=24):
    """An agent: Otto's shape in a colour, with its name (and a muted line) beside or below it."""
    bot = otto(size, color, fill)
    words = title_text(name, name_size, color)
    label = words if sub is None else m.VGroup(words, code(sub, 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
    if side is m.DOWN:
        label.next_to(bot, m.DOWN, buff=0.14)
    else:
        label.next_to(bot, m.RIGHT, buff=0.25)
    return m.VGroup(bot, label)


class CodeCard:
    """A code_block with per-line glyph groups and a highlight bar (from Episode 21)."""

    def __init__(self, src, title, width=None, size=20):
        self.block = code_block(src, size=size, width=width, title=title)
        self.box, self.body = self.block[0], self.block[1]
        self.src = src.split("\n")
        self.lines, k = [], 0
        for line in self.src:
            n = sum(not c.isspace() for c in line)
            self.lines.append(self.body[k:k + n])
            k += n
        self.body.set_z_index(2)

    def part(self, i, key):
        line = self.src[i]
        start = line.index(key)
        skip = sum(c.isspace() for c in line[:start])
        width = sum(not c.isspace() for c in key)
        return self.lines[i][start - skip:start - skip + width]

    def bar(self, a, b=None, color=CORAL):
        b = a if b is None else b
        top = max(g.get_top()[1] for g in self.lines[a]) + 0.1
        bot = min(g.get_bottom()[1] for g in self.lines[b]) - 0.1
        left, right = self.box.get_left()[0] + 0.12, self.box.get_right()[0] - 0.12
        rect = m.Rectangle(width=right - left, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=0.16)
        rect.move_to([(left + right) / 2, (top + bot) / 2, 0])
        mark = m.Rectangle(width=0.05, height=top - bot, stroke_width=0, fill_color=color, fill_opacity=1)
        mark.move_to([left + 0.025, (top + bot) / 2, 0])
        return m.VGroup(rect, mark).set_z_index(1)


def tool_chip(name, struck=False, width=1.85):
    color, fill = (FAINT, CARD) if struck else (SUN, SUN_FILL)
    c = chip(name, color, fill, width=width, height=0.5, size=24, mono=True)
    if struck:
        c[1].set_color(FAINT)
    return c


def boxed(inner, color=EDGE, pad=(0.35, 0.28), radius=0.16, fill=CARD, width=None):
    """A card fitted around `inner`, content left-aligned."""
    box = card(max(width or 0, inner.width + 2 * pad[0]), inner.height + 2 * pad[1], color, fill, radius=radius, stroke=2)
    inner.move_to(box).align_to(box, m.LEFT).shift(pad[0] * m.RIGHT)
    return m.VGroup(box, inner)


def json_text(src, size=20):
    """JSON-ish source, coloured like the kit's json_card."""
    return code(src, size, INK, t2c={
        r"\"[^\"\n]*\"(?=\s*:)": AZURE,
        r":\s*\"[^\"\n]*\"": MINT,
        r"\.\.\.": FAINT,
    })


# ---------------------------------------------------------------------------- the episode

# Episode.say, plus: anything not yet on screen (a message pushed while others scroll, a row of a terminal
# shown on its own cue) waits for its own cue instead of showing early (as in Episode 21).
_SAY_SRC = """
def say(self, script, *animations, voice=None):
    on_screen = {id(x) for x in self.get_mobject_family_members()}
    for a in animations:
        for leaf in leaves(a):
            mob = getattr(leaf, "mobject", None)
            if mob is not None and not leaf.is_introducer() and id(mob) not in on_screen:
                leaf.introducer = True
    return episode_say(self, script, *animations, voice=voice)
"""
_SAY26 = {"leaves": leaves, "episode_say": Episode.say}
exec(compile(_SAY_SRC, str(Path(m.__file__).parent / "_ep26_say.py"), "exec"), _SAY26)


class Ep26MultiAgentPatterns(Episode):
    number = 26

    say = _SAY26["say"]

    def lesson(self):
        self.hook()
        self.self_review()
        self.reviewer()
        self.information()
        self.map()
        self.cost()
        self.failures()
        self.a2a()
        self.recap()

    # ------------------------------------------------------------------ hook

    def hook(self):
        self.next_section("Hook", notes="Episode 24's team (a lead Otto, a tester, a reviewer, mailboxes). Does a team beat one agent? Li, AI Agents in Depth, ch. 10 asks one question first: 'Does the collaboration provide information that a single Agent could not obtain while producing its answer?' Facts as of October 2026.")
        lead = badge(CORAL, CORAL_FILL, "lead", 0.8, side=m.RIGHT)
        tester = badge(AZURE, AZURE_FILL, "tester", 0.6)
        rev = badge(ORCHID, ORCHID_FILL, "reviewer", 0.6)
        lead.move_to([-3.6, 1.5, 0])
        lead.shift((-3.95 - lead[0].get_x()) * m.RIGHT)
        tester.move_to([-5.45, -0.15, 0])
        rev.move_to([-2.45, -0.15, 0])
        links = m.VGroup(*[
            m.DashedLine(lead[0].get_bottom() + 0.08 * m.DOWN, t[0].get_top() + 0.08 * m.UP, dash_length=0.12).set_stroke(MUTED, 3)
            for t in (tester, rev)
        ])
        team = m.VGroup(lead, tester, rev, links)
        ep = pill("Episode 24 · agent teams", AZURE, AZURE_FILL, 20).next_to(lead, m.UP, buff=0.3).align_to(tester, m.LEFT)
        ep.set_y(2.62)
        one = badge(CORAL, CORAL_FILL, "one agent", 1.0, name_size=26).move_to([4.2, 0.85, 0])
        q = title_text("?", 96, MUTED).move_to([0.75, 0.95, 0])
        often = text("not as often as you'd think", 22, MUTED).next_to(q, m.DOWN, buff=0.3)
        quote = m.VGroup(
            text("“Does the collaboration provide information that a single", 26, INK),
            text("agent could not obtain while producing its answer?”", 26, INK),
        ).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 10")
        qbody = m.VGroup(quote, src).arrange(m.DOWN, buff=0.2, aligned_edge=m.LEFT)
        qcard = boxed(qbody, AZURE, pad=(0.45, 0.3)).move_to([0, -2.45, 0])

        self.say(
            "[In episode twenty-four, Otto led a team]: [a tester, a reviewer], [and mailboxes between them.]",
            m.AnimationGroup(m.FadeIn(lead, shift=0.2 * m.UP), m.FadeIn(ep, shift=0.15 * m.UP)),
            m.LaggedStart(m.FadeIn(tester, shift=0.2 * m.UP), m.FadeIn(rev, shift=0.2 * m.UP), lag_ratio=0.3),
            m.AnimationGroup(m.Create(links), pulse(m.Line(links[0].get_start(), links[0].get_end()), AZURE, 8),
                             pulse(m.Line(links[1].get_start(), links[1].get_end()), ORCHID, 8)),
        )
        self.say(
            "[But does a team actually beat one agent?] [Not as often as you'd think.]",
            m.AnimationGroup(m.FadeIn(one, shift=0.2 * m.UP), m.FadeIn(q, scale=0.7)),
            m.FadeIn(often, shift=0.1 * m.UP),
        )
        self.say(
            "[The book AI Agents in Depth asks one question first]: [does the collaboration bring information that a single agent could not get?]",
            m.AnimationGroup(m.FadeIn(qcard[0]), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(line, shift=0.1 * m.UP) for line in quote], lag_ratio=0.5, run_time=1.6),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ self-review

    def self_review(self):
        self.next_section("1 · Self-review", notes=f"Our example (illustrative): the ticket '{TICKET}'. Otto's patch uses 'subtotal > 50'. Asked to review its own patch in the same context, the same model re-reads it and approves: '{SELF_REVIEW}' — it repeats its own reading ('over $50'). Li, ch. 10, Table 10-1: self-review by the same model brings no new information and is 'usually ineffective or even harmful'.")
        col = ContextColumn(center=(-3.3, -0.6, 0), width=6.4, height=5.65, title="context", gap=0.1)
        label = m.VGroup(otto(0.3), code("Otto's context", 20, MUTED)).arrange(m.RIGHT, buff=0.2)
        label.next_to(col.panel, m.UP, buff=0.14).align_to(col.panel, m.LEFT)
        msgs = [
            col.message("user", TICKET, size=20),
            col.message("tool_use", 'edit_file("checkout.py", …)', size=22, mono=True),
            col.message("tool_result", "edited checkout.py", size=22, mono=True),
            col.message("user", REVIEW_ASK, size=20),
            col.message("assistant", SELF_REVIEW, size=20),
        ]
        glyphs(msgs[4][2][1], "over $50").set_color(SUN)
        glyphs(msgs[0][2][1], "or more").set_color(SUN)
        self.cc = cc = CodeCard(PATCH, "checkout.py · Otto's patch", size=22)
        cc.block.move_to([3.35, 0, 0]).align_to([0, 2.15, 0], m.UP)
        bar = cc.bar(1)
        same = m.VGroup(
            badge(CORAL, CORAL_FILL, "writes", 0.55, name_size=22),
            title_text("=", 44, MUTED),
            badge(CORAL, CORAL_FILL, "reviews", 0.55, name_size=22),
        ).arrange(m.RIGHT, buff=0.55)
        same[1].align_to(same[0][0], m.UP).shift(0.12 * m.DOWN)
        same.move_to([3.35, -2.45, 0])
        same_cap = text("same model, same context", 22, MUTED).next_to(same, m.UP, buff=0.3)
        ok = pill("self-review: approved", MINT, MINT_FILL, 22).move_to([3.35, -0.55, 0])
        okt = tick(ok[0].get_right() + 0.45 * m.RIGHT, 0.34)

        a0 = col.push(msgs[0])
        self.say(
            "[Let's test that on Otto's next ticket]: [free shipping on orders of fifty dollars or more.]",
            m.AnimationGroup(self.new_header("01 · self-review", "Same model, same blind spot"), m.FadeIn(col.panel), m.FadeIn(label)),
            a0,
        )
        a1, a2 = col.push(msgs[1], run_time=0.7), col.push(msgs[2], run_time=0.7)
        self.say(
            "[Otto writes the patch]: [if the subtotal is over fifty, shipping is free.]",
            m.Succession(a1, a2),
            m.AnimationGroup(m.FadeIn(cc.block, shift=0.2 * m.UP), m.FadeIn(bar)),
        )
        a3 = col.push(msgs[3])
        self.say(
            "[Then we ask it to check its own work]: [same model, same context.]",
            a3,
            m.AnimationGroup(m.FadeIn(same, shift=0.15 * m.UP), m.FadeIn(same_cap)),
        )
        a4 = col.push(msgs[4])
        self.say(
            "[It reads the patch again], [and approves it]: [free shipping over fifty, looks correct.]",
            m.AnimationGroup(m.Transform(bar, cc.bar(0, 3)), m.Circumscribe(cc.box, color=CORAL, buff=0.06)),
            a4,
            m.AnimationGroup(m.FadeIn(ok, shift=0.1 * m.UP), m.Create(okt)),
        )
        self.wait(0.3)
        self.clear(cc.block)

    # ------------------------------------------------------------------ proposer–reviewer

    def reviewer(self):
        self.next_section("2 · A reviewer with tests", notes=f"Proposer–Reviewer (Li, ch. 10), like Episode 5's evaluator–optimizer, but the reviewer works in its own context and sees independent evidence, not Otto's reasoning: the ticket, the diff, and a test runner over the ticket's tests (read-only: 'The reviewer must not be able to modify the tests, the evidence collector, or the release gate; otherwise \"independent verification\" degenerates into self-approval'). The run is computed: {', '.join(f'{n} → {got:.2f}' for n, got, _ in V1)}; {FAILS[0][0]} fails (5.00 ≠ 0.00). The failing test goes back to Otto as evidence; '>' becomes '>=' (one character) and the whole suite passes ({PASSED2} = {OLD_SUITE} older tests + {len(TICKET_TESTS)} new).")
        cc = self.cc
        prop = badge(CORAL, CORAL_FILL, "Otto", 0.6, side=m.RIGHT, sub="proposer", name_size=26)
        prop.move_to([0, 1.95, 0]).align_to([X_MIN, 0, 0], m.LEFT)
        rev = badge(ORCHID, ORCHID_FILL, "reviewer", 0.6, side=m.RIGHT, sub="its own context", name_size=26)
        rev.move_to([0, 1.95, 0]).align_to([0.55, 0, 0], m.LEFT)
        target = cc.block.copy().align_to([X_MIN, 0, 0], m.LEFT).align_to([0, 1.05, 0], m.UP)
        shift = target.get_center() - cc.block.get_center()
        link = arrow([prop.get_right()[0] + 0.2, 1.95, 0], [rev.get_left()[0] - 0.15, 1.95, 0], MUTED, buff=0.0)
        ticket = chip("ticket", AZURE, AZURE_FILL, width=1.3, height=0.46, size=24, mono=True)
        diff = chip("diff", CORAL, CORAL_FILL, width=1.0, height=0.46, size=24, mono=True)
        packets = m.VGroup(ticket, diff).arrange(m.RIGHT, buff=0.2)
        packets.move_to([link.get_center()[0] + 0.35, 2.42, 0])
        why = m.VGroup(text("Otto's reasoning", 20, MUTED), cross(m.ORIGIN, 0.22)).arrange(m.RIGHT, buff=0.18)
        why.move_to([link.get_center()[0], 1.5, 0])
        tools = m.VGroup(tool_chip("read_file", width=1.72), tool_chip("run_tests", width=1.72), tool_chip("edit_file", struck=True, width=1.72)).arrange(m.RIGHT, buff=0.14)
        tools.move_to([0, 1.0, 0]).align_to([0.6, 0, 0], m.LEFT)
        nox = cross(tools[2][0].get_right() + 0.28 * m.RIGHT, 0.24)
        assert nox.get_right()[0] < X_MAX
        rule = text("tests are read-only: it can't edit them", 20, ROSE).next_to(tools, m.DOWN, buff=0.18).align_to(tools, m.LEFT)
        src = cite("Li, AI Agents in Depth · ch. 10")

        term = terminal(TERM_V1, width=tools.width + 0.4, size=20, title="run_tests")
        term.align_to(tools, m.LEFT).align_to([0, 0.05, 0], m.UP)
        frame, rows = m.VGroup(*term[:4]), list(term[4])
        for i in (1, 3):
            paint(rows[i], TERM_V1[i], {"PASSED": MINT})
        src.next_to(term, m.DOWN, buff=0.2).align_to(term, m.RIGHT)
        term2 = terminal(TERM_V2, width=term.width, size=20, title="run_tests")
        term2.align_to(term, m.UL)
        rows2 = list(term2[4])
        paint(rows2[1], TERM_V2[1], {r"\.+": MINT})
        rows2[2].set_color(MINT)

        self.say(
            "[Now hand the same patch to a second agent]: [a reviewer, like the evaluator of episode five], [but in a context of its own.]",
            m.AnimationGroup(self.new_header("02 · proposer and reviewer", "A reviewer with evidence"), cc.block.animate.shift(shift), m.FadeIn(prop, shift=0.15 * m.UP)),
            m.FadeIn(rev, shift=0.15 * m.UP),
            m.Circumscribe(rev, color=ORCHID, buff=0.1),
        )
        self.say(
            "[It never sees Otto's reasoning]: [only the ticket], [the diff], [and a test runner.]",
            m.AnimationGroup(m.GrowArrow(link), m.FadeIn(why, shift=0.1 * m.UP)),
            m.FadeIn(ticket, target_position=prop),
            m.FadeIn(diff, target_position=cc.box),
            m.LaggedStart(*[m.FadeIn(t, shift=0.1 * m.UP) for t in tools[:2]], lag_ratio=0.3),
        )
        self.say(
            "[It can run the tests], [but it can't edit them]: [otherwise, review turns into self-approval.]",
            m.Circumscribe(tools[1], color=SUN, buff=0.06),
            m.AnimationGroup(m.FadeIn(tools[2], shift=0.1 * m.UP), m.Create(nox)),
            m.AnimationGroup(m.FadeIn(rule, shift=0.1 * m.UP), m.FadeIn(src)),
        )
        self.say(
            "[It runs the ticket's three tests]: [below fifty], [exactly fifty], [and above.]",
            m.AnimationGroup(m.FadeOut(rule), m.FadeIn(frame), m.FadeIn(rows[0]), m.Indicate(tools[1], color=SUN, scale_factor=1.06)),
            m.FadeIn(rows[1], shift=0.1 * m.UP),
            m.FadeIn(rows[2], shift=0.1 * m.UP),
            m.FadeIn(rows[3], shift=0.1 * m.UP),
        )
        gt = cc.part(1, ">")
        self.say(
            "[One fails.] [At exactly fifty dollars], [the customer still pays five dollars shipping.]",
            m.Circumscribe(rows[2], color=ROSE, buff=0.06),
            m.AnimationGroup(m.FadeIn(rows[4], shift=0.1 * m.UP), gt.animate.set_color(ROSE), m.Circumscribe(cc.part(1, "> 50"), color=ROSE, buff=0.05)),
            m.Circumscribe(cc.part(3, "return 5.00"), color=ROSE, buff=0.05),
        )
        verdict = pill("rejected", ROSE, ROSE_FILL, 22).next_to(rev, m.RIGHT, buff=0.3)
        ev_head = code("reviewer → Otto", 20, ORCHID, weight=m.BOLD)
        ev_body = m.VGroup(text("rejected: test_at_50 fails", 22, INK), code("shipping(50.00) returned 5.0", 20, ROSE)).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        evidence = boxed(m.VGroup(ev_head, ev_body).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT), ROSE, width=cc.box.width)
        evidence.align_to([X_MIN, 0, 0], m.LEFT).align_to([0, -1.75, 0], m.UP)
        self.say(
            "[The reviewer rejects the patch], [and sends the failing test back to Otto as evidence.]",
            m.FadeIn(verdict, scale=0.8),
            m.FadeIn(evidence, target_position=rows[2], path_arc=0.5),
        )
        line = cc.lines[1]
        fixed = code("if subtotal >= 50:", 22, INK, t2c={r"\bif\b": ORCHID, r"\b50\b": SUN, r">=": MINT})
        fixed.move_to(line).align_to(line, m.LEFT)
        fixed.shift((line[0].get_bottom()[1] - fixed[0].get_bottom()[1]) * m.UP).set_z_index(2)
        approved = pill("approved", MINT, MINT_FILL, 22).move_to(verdict)
        self.say(
            "[Otto changes one character]: [greater than becomes greater or equal], [and the whole suite passes]: [sixteen tests.]",
            m.Circumscribe(gt, color=CORAL, buff=0.08),
            m.AnimationGroup(m.FadeOut(line, shift=0.1 * m.UP), m.FadeIn(fixed, shift=0.1 * m.UP)),
            m.AnimationGroup(*[m.FadeOut(r) for r in rows], m.Transform(frame, m.VGroup(*term2[:4])), m.LaggedStart(*[m.FadeIn(r, shift=0.1 * m.UP) for r in rows2], lag_ratio=0.3)),
            m.AnimationGroup(m.Circumscribe(rows2[2], color=MINT, buff=0.06), m.FadeOut(verdict, scale=0.8), m.FadeIn(approved, scale=0.8)),
        )
        cc.body.remove(*line)  # the old line is gone: keep it from coming back with the card's fade
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ new information

    def information(self):
        self.next_section("3 · New information", notes=f"What caught the bug: the test run (execution feedback), not a second opinion. Li, ch. 10, Table 10-1 (rows lightly shortened): same-model self-review → no new information, 'Usually ineffective or even harmful'; agents debating the same text → no, 'Comparable to a single Agent with equal compute' (Tran & Kiela, 2026: at a fixed thinking-token budget the single agent was 'on par with or even better'); reviewers using test results, rendered screenshots or external tools → yes, 'Significant improvement'. WebGen-Agent: visual feedback took Claude 3.5 Sonnet from {WEBGEN[0]}% to {WEBGEN[1]}% on its benchmark (as quoted by Li).")
        left = boxed(m.VGroup(title_text("a second opinion", 34, INK), text("the same model, re-reading", 22, MUTED)).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT), ROSE, pad=(0.45, 0.35), width=5.6)
        right = boxed(m.VGroup(title_text("the test run", 34, MINT), text("new information: what the code does", 22, MUTED)).arrange(m.DOWN, buff=0.16, aligned_edge=m.LEFT), MINT, pad=(0.45, 0.35), width=5.6)
        left.move_to([-3.15, 0.4, 0])
        right.move_to([3.15, 0.4, 0])
        lx = cross(left[0].get_corner(m.UR) + np.array([-0.45, -0.45, 0]), 0.3)
        rt = tick(right[0].get_corner(m.UR) + np.array([-0.45, -0.45, 0]), 0.38)
        what = title_text("What caught the bug?", 40, INK).move_to([0, 2.05, 0])

        # the table (Li, ch. 10, Table 10-1)
        xs = [-6.25, -1.3, 1.6]
        heads = m.VGroup(*[text(h, 22, MUTED, weight=m.SEMIBOLD).move_to([x, 2.3, 0], aligned_edge=m.LEFT) for h, x in zip(("review", "new information?", "effect"), xs)])
        sit(heads[1], heads[0][0].get_bottom()[1])
        sit(heads[2], heads[0][0].get_bottom()[1])
        rows = []
        for i, (who, new, effect, color) in enumerate(REVIEWS):
            y = 1.7 - 0.6 * i
            band = card(X_MAX - X_MIN + 0.1, 0.5, EDGE, CARD, radius=0.1, stroke=1.5).move_to([0, y, 0]).set_z_index(-1)
            a = text(who, 22, INK).move_to([xs[0], y, 0], aligned_edge=m.LEFT)
            b = code(new, 20, ROSE if new == "No" else MINT, weight=m.BOLD).move_to([xs[1], y, 0], aligned_edge=m.LEFT)
            c = text(effect, 22, color if color != MUTED else INK).move_to([xs[2], y, 0], aligned_edge=m.LEFT)
            base = y - 0.09
            for part in (a, b, c):
                sit(part, base)
            assert a.get_right()[0] < xs[1] - 0.2 and b.get_right()[0] < xs[2] - 0.2 and c.get_right()[0] < X_MAX - 0.1
            rows.append(m.VGroup(band, a, b, c))
        tsrc = cite("Li, AI Agents in Depth · ch. 10, Table 10-1 · debate: Tran & Kiela, 2026").move_to([0, -1.22, 0]).align_to([xs[0], 0, 0], m.LEFT)

        # WebGen-Agent's screenshots
        full = 7.0
        names = [text("without screenshots", 22, MUTED), text("with screenshots", 22, INK)]
        bars, vals = m.VGroup(), m.VGroup()
        for i, (name, v, col) in enumerate(zip(names, WEBGEN, (MUTED, MINT))):
            y = -1.85 - 0.62 * i
            name.move_to([xs[0], y, 0], aligned_edge=m.LEFT)
            b = m.RoundedRectangle(corner_radius=0.06, width=full * v / 100, height=0.36, stroke_width=0, fill_color=col, fill_opacity=1)
            b.move_to([-2.6 + b.width / 2, y, 0])
            val = code(f"{v}%", 22, col, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.22)
            sit(val, name[0].get_bottom()[1])
            bars.add(b)
            vals.add(val)
        wsrc = cite("WebGen-Agent, a web-building agent (Claude 3.5 Sonnet) · via Li, ch. 10")
        wsrc.move_to([0, -3.1, 0]).align_to([xs[0], 0, 0], m.LEFT)

        self.say(
            "[So what caught the bug?] [Not a second opinion.] [The test run.]",
            m.AnimationGroup(self.new_header("03 · new information", "Evidence, not opinions"), m.FadeIn(what, shift=0.15 * m.UP)),
            m.AnimationGroup(m.FadeIn(left, shift=0.15 * m.UP), m.Create(lx)),
            m.AnimationGroup(m.FadeIn(right, shift=0.15 * m.UP), m.Create(rt)),
        )
        self.say(
            "[The book sorts reviews by one question]: [does the reviewer get new information?]",
            m.AnimationGroup(m.FadeOut(m.VGroup(what, left, right, lx, rt), shift=0.2 * m.UP), m.FadeIn(heads, shift=0.1 * m.UP), m.FadeIn(tsrc)),
            m.Circumscribe(heads[1], color=AZURE, buff=0.08),
        )
        r = rows[0]
        self.say(
            "[The same model re-reading its own output]: [nothing new.] [Usually ineffective, or even harmful.]",
            m.AnimationGroup(m.FadeIn(r[0]), m.FadeIn(r[1], shift=0.1 * m.RIGHT)),
            m.FadeIn(r[2], scale=0.8),
            m.FadeIn(r[3], shift=0.1 * m.RIGHT),
        )
        r = rows[1]
        self.say(
            "[Agents debating the same text]: [nothing new either], [and no better than one agent given the same compute.]",
            m.AnimationGroup(m.FadeIn(r[0]), m.FadeIn(r[1], shift=0.1 * m.RIGHT)),
            m.FadeIn(r[2], scale=0.8),
            m.FadeIn(r[3], shift=0.1 * m.RIGHT),
        )
        self.say(
            "[Reviewers that run tests], [look at screenshots], [or check facts with tools]: [new information], [and a big improvement.]",
            *[m.AnimationGroup(m.FadeIn(rows[i][0]), m.FadeIn(rows[i][1], shift=0.1 * m.RIGHT)) for i in (2, 3, 4)],
            m.LaggedStart(*[m.FadeIn(rows[i][2], scale=0.8) for i in (2, 3, 4)], lag_ratio=0.25),
            m.LaggedStart(*[m.FadeIn(rows[i][3], shift=0.1 * m.RIGHT) for i in (2, 3, 4)], lag_ratio=0.25),
        )
        self.say(
            "[In one study, a web-building agent shown screenshots of its own pages] [went from twenty-six point four to fifty-one point nine percent on its benchmark.]",
            m.AnimationGroup(m.FadeIn(names[0]), m.GrowFromEdge(bars[0], m.LEFT), m.FadeIn(vals[0]), m.FadeIn(wsrc), m.FadeIn(names[1])),
            m.AnimationGroup(m.GrowFromEdge(bars[1], m.LEFT), m.FadeIn(vals[1], shift=0.1 * m.LEFT), m.Circumscribe(rows[3], color=MINT, buff=0.04)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the map

    def map(self):
        self.next_section("4 · The map", notes="Li, ch. 10: two dimensions — shared vs non-shared context, and peer / manager (orchestration) / decentralized. Placements: proposer–reviewer (peers, separate contexts: Otto and its reviewer); debate (peers in one thread); manager: 'each specialized Agent as a tool' (Episode 21's subagents, isolation), forks that inherit the context (Episode 21); decentralized: handoffs (OpenAI Swarm; the package carries a remaining budget and a visited_agents list, enforced by the runtime, against cycles) and group chat (AutoGen-style, one shared thread). Placement of examples in cells is our reading of the chapter.")
        cols_x, cell_w = [-1.2, 3.95], 4.95
        rows_y, cell_h = [1.3, -0.4, -2.1], 1.5
        label_x = -5.35
        cells = [[card(cell_w, cell_h, EDGE, CARD, radius=0.18, stroke=2).move_to([x, y, 0]) for x in cols_x] for y in rows_y]
        frame = m.VGroup(*[c for row in cells for c in row])
        col_heads = [title_text("one shared context", 26, INK), title_text("a context per agent", 26, INK)]
        for h, x in zip(col_heads, cols_x):
            h.move_to([x, 2.4, 0])
        sit(col_heads[1], col_heads[0][0].get_bottom()[1])
        corner = code("who's in charge?", 20, MUTED).move_to([label_x, 2.4, 0])
        sit(corner, col_heads[0][0].get_bottom()[1])
        row_heads = [title_text(s, 26, AZURE).move_to([label_x, y, 0]) for s, y in zip(("peers", "a manager", "nobody"), rows_y)]

        def fill(r, c, title, sub, color, extra=None):
            t = text(title, 26, color, weight=m.BOLD)
            s = text(sub, 20, MUTED)
            g = m.VGroup(t, s, *([extra] if extra is not None else [])).arrange(m.DOWN, buff=0.12).move_to(cells[r][c])
            assert g.width < cell_w - 0.3
            return g

        rev_cell = fill(0, 1, "proposer and reviewer", "Otto and its reviewer", ORCHID)
        debate = fill(0, 0, "debate", "agents argue in one thread", CORAL)
        iso = text("they pay off through isolation", 20, MINT, weight=m.SEMIBOLD)
        tools_cell = fill(1, 1, "agents as tools", "the subagents of episode 21", AZURE, iso)
        forks = fill(1, 0, "forks", "they inherit the manager's context", AZURE)
        handoffs = fill(2, 1, "handoffs", "control passes from agent to agent", SUN)
        chat = fill(2, 0, "group chat", "everyone reads one thread", MINT)
        guard = code("visited_agents · budget", 20, SUN)
        handoffs_after = m.VGroup(handoffs.copy(), guard).arrange(m.DOWN, buff=0.14).move_to(cells[2][1])
        guard.move_to(handoffs_after[1])
        src = cite("Li, AI Agents in Depth · ch. 10").move_to([label_x, -3.35, 0]).align_to([X_MIN, 0, 0], m.LEFT)

        self.say(
            "[Now the map.] [Every multi-agent design answers two questions.] [First: do the agents share one context], [or does each keep its own?]",
            m.AnimationGroup(self.new_header("04 · the map", "Two questions, six shapes"), m.FadeIn(src)),
            m.LaggedStart(*[m.FadeIn(c, scale=0.95) for c in frame], lag_ratio=0.08),
            m.FadeIn(col_heads[0], shift=0.1 * m.DOWN),
            m.FadeIn(col_heads[1], shift=0.1 * m.DOWN),
        )
        self.say(
            "[Second: who's in charge?] [Peers], [a manager], [or nobody.]",
            m.FadeIn(corner, shift=0.1 * m.DOWN),
            *[m.FadeIn(h, shift=0.15 * m.RIGHT) for h in row_heads],
        )
        self.say(
            "[Otto and its reviewer are peers with separate contexts.] [Debaters are peers sharing one thread.]",
            m.FadeIn(rev_cell, scale=0.9),
            m.FadeIn(debate, scale=0.9),
        )
        self.say(
            "[A manager calls agents like tools]: [the subagents of episode twenty-one], [which pay off through isolation], [or forks that share its context.]",
            m.FadeIn(tools_cell[0], scale=0.9),
            m.FadeIn(tools_cell[1], shift=0.1 * m.UP),
            m.AnimationGroup(m.FadeIn(iso, shift=0.1 * m.UP), m.Circumscribe(cells[1][1], color=MINT, buff=0.05)),
            m.FadeIn(forks, scale=0.9),
        )
        self.say(
            "[With nobody in charge, agents hand off control to each other], [or all talk in one group chat.]",
            m.FadeIn(handoffs, scale=0.9),
            m.FadeIn(chat, scale=0.9),
        )
        self.say(
            "[A safe handoff carries the list of agents visited and the budget left], [so the work can't go in circles forever.]",
            m.AnimationGroup(handoffs.animate.move_to(handoffs_after[0]), m.FadeIn(guard, shift=0.1 * m.UP)),
            m.Circumscribe(cells[2][1], color=SUN, buff=0.05),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ the cost

    def cost(self):
        self.next_section("5 · The cost", notes=f"Anthropic, 'How we built our multi-agent research system' (13 June 2025): agents use about 4× the tokens of chat interactions, multi-agent systems about 15×; token usage by itself explains 80% of the performance variance (BrowseComp); the multi-agent system (Opus 4 lead, Sonnet 4 subagents) outperformed single-agent Opus 4 by 90.2% on Anthropic's internal research eval; it excels at heavy parallelization, while 'most coding tasks involve fewer truly parallelizable tasks than research'. Bars drawn to scale (1 : 4 : 15).")
        cap = code("tokens per task, compared with a chat", 20, MUTED).move_to([0, 2.35, 0]).align_to([X_MIN, 0, 0], m.LEFT)
        src = cite("Anthropic, 2025 · multi-agent research system")
        src.align_to([X_MAX, 0, 0], m.RIGHT)
        sit(src, cap[0].get_bottom()[1])
        unit, x0 = 0.5, -2.65
        rows = []
        for i, (name, k, color) in enumerate(TOKENS_X):
            y = 1.75 - 0.65 * i
            lab = text(name, 24, INK, weight=m.SEMIBOLD).move_to([X_MIN, y, 0], aligned_edge=m.LEFT)
            b = m.RoundedRectangle(corner_radius=0.07, width=unit * k, height=0.42, stroke_width=0, fill_color=color, fill_opacity=1)
            b.move_to([x0 + b.width / 2, y, 0])
            v = code(f"{'≈ ' if k > 1 else ''}{k}×", 22, color, weight=m.BOLD).next_to(b, m.RIGHT, buff=0.22)
            sit(v, lab[0].get_bottom()[1])
            assert v.get_right()[0] < X_MAX
            rows.append((lab, b, v))
        assert rows[2][1].width / rows[0][1].width == 15 and rows[1][1].width / rows[0][1].width == 4  # to scale

        def stat(number, color, lines):
            big = title_text(number, 48, color)
            words = m.VGroup(*[text(s, 22, MUTED) for s in lines]).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            inner = m.VGroup(big, words).arrange(m.DOWN, buff=0.14, aligned_edge=m.LEFT)
            return boxed(inner, color, pad=(0.4, 0.24), width=6.2)

        t1 = stat(f"+{GAIN}%", MINT, ["multi-agent vs one agent,", "on Anthropic's internal research eval"])
        t2 = stat(f"{VARIANCE}%", SUN, ["of the variance in performance,", "explained by token usage alone"])
        for t, x in ((t1, -3.25), (t2, 3.25)):
            t.move_to([x, 0, 0]).align_to([0, -0.05, 0], m.UP)
        assert t1[1].width < 6.2 - 0.7 and t2[1].width < 6.2 - 0.7
        assert t1.get_top()[1] < rows[2][1].get_bottom()[1] - 0.2
        good = m.VGroup(tick(m.ORIGIN, 0.32), text("broad research: parallel searches", 22, INK)).arrange(m.RIGHT, buff=0.25)
        bad = m.VGroup(cross(m.ORIGIN, 0.24), text("most coding: steps depend on each other", 22, INK)).arrange(m.RIGHT, buff=0.25)
        good.move_to([0, -2.75, 0]).align_to(t1, m.LEFT).shift(0.1 * m.RIGHT)
        bad.move_to([0, -2.75, 0]).align_to(t2, m.LEFT).shift(0.1 * m.RIGHT)
        sit(bad[1], good[1][0].get_bottom()[1])
        assert bad.get_right()[0] < X_MAX

        r0, r1, r2 = rows
        self.say(
            "[Teams also cost more.] [Anthropic measured its own multi-agent research system.]",
            m.AnimationGroup(self.new_header("05 · the cost", "About fifteen times the tokens"), m.FadeIn(cap)),
            m.AnimationGroup(m.FadeIn(src), m.FadeIn(r0[0]), m.GrowFromEdge(r0[1], m.LEFT), m.FadeIn(r0[2])),
        )
        self.say(
            "[A single agent used about four times the tokens of a chat.] [The multi-agent system: about fifteen times.]",
            m.AnimationGroup(m.FadeIn(r1[0]), m.GrowFromEdge(r1[1], m.LEFT), m.FadeIn(r1[2], shift=0.1 * m.LEFT)),
            m.AnimationGroup(m.FadeIn(r2[0]), m.GrowFromEdge(r2[1], m.LEFT, run_time=1.4), m.FadeIn(r2[2], shift=0.1 * m.LEFT)),
        )
        self.say(
            "[It beat a single agent by about ninety percent], [on an internal research evaluation.]",
            m.FadeIn(t1, shift=0.15 * m.UP),
            m.Circumscribe(t1[1][1], color=MINT, buff=0.06),
        )
        self.say(
            "[But token usage alone explained eighty percent of the variance in performance.]",
            m.FadeIn(t2, shift=0.15 * m.UP),
        )
        self.say(
            "[Much of the gain is simply more compute.] [Teams pay off on broad research with parallel searches], [less on coding, where steps depend on each other.]",
            m.Circumscribe(t2, color=SUN, buff=0.06),
            m.FadeIn(good, shift=0.1 * m.UP),
            m.FadeIn(bad, shift=0.1 * m.UP),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ failure modes

    def failures(self):
        self.next_section("6 · Failure modes", notes="Cemri et al., 'Why Do Multi-Agent LLM Systems Fail?' (2025): MAST, 14 failure modes in 3 categories — system design issues (5), inter-agent misalignment (6), task verification (3); names as in the paper. Li, ch. 10: 'every handoff is a lossy re-encoding'; runaway spawning ('may spawn thousands of sub-agents') and 'use a dedicated API key as part of controlling token spending'. Anthropic (2025): early agents spawned '50 subagents for simple queries', fixed with explicit scaling rules in the prompts. The hand-off chain's wording is illustrative.")
        sub = m.VGroup(text("A study of multi-agent systems: 14 failure modes, 3 groups", 24, INK), cite("MAST · Cemri et al., 2025")).arrange(m.RIGHT, buff=0.4)
        sit(sub[1], sub[0][0].get_bottom()[1])
        sub.move_to([0, 2.35, 0]).align_to([X_MIN, 0, 0], m.LEFT)
        colors = [CORAL, ORCHID, MINT]
        groups = m.VGroup()
        for (name, modes), color in zip(MAST, colors):
            head = m.VGroup(text(name, 22, color, weight=m.BOLD), text(f"{len(modes)} modes", 20, MUTED)).arrange(m.DOWN, buff=0.08, aligned_edge=m.LEFT)
            items = m.VGroup()
            for mode in modes:
                dot = m.RoundedRectangle(corner_radius=0.03, width=0.13, height=0.13, stroke_width=0, fill_color=color, fill_opacity=1)
                words = text(mode.replace("termination ", "termination\n"), 20, INK)
                row = m.VGroup(dot, words).arrange(m.RIGHT, buff=0.16)
                dot.align_to(words, m.UP).shift(0.07 * m.DOWN)
                items.add(row)
            items.arrange(m.DOWN, buff=0.17, aligned_edge=m.LEFT)
            groups.add(m.VGroup(head, items).arrange(m.DOWN, buff=0.28, aligned_edge=m.LEFT))
        widths = [g.width + 0.5 for g in groups]
        gap = (X_MAX - X_MIN - sum(widths)) / 2
        assert gap > 0.15, gap
        top = 1.85
        h = max(g.height for g in groups) + 0.55
        cards = m.VGroup()
        x = X_MIN
        for g, w, color in zip(groups, widths, colors):
            box = card(w, h, color, CARD, radius=0.18, stroke=2).move_to([x + w / 2, top - h / 2, 0])
            g.move_to(box).align_to(box, m.UP).shift(0.28 * m.DOWN).align_to(box, m.LEFT).shift(0.25 * m.RIGHT)
            cards.add(box)
            x += w + gap
        weak = groups[2][1][1]
        tag = m.VGroup(otto(0.3), text("Otto's self-review", 22, MINT)).arrange(m.RIGHT, buff=0.18)
        tag.next_to(cards[2], m.DOWN, buff=0.25).align_to(cards[2], m.LEFT).shift(0.2 * m.RIGHT)

        # lossy hand-offs (left) and runaway spawning (right)
        hop_rows = m.VGroup()
        for i, (who, color, fill, said) in enumerate(HOPS):
            bot = otto(0.34, color, fill)
            name = code(who, 20, color, weight=m.BOLD)
            who_g = m.VGroup(bot, name)
            words = text(said, 22, INK)
            if i:
                for w in changed_words(HOPS[i - 1][3], said):
                    glyphs(words, w).set_color(ROSE)
            else:
                glyphs(words, "or more").set_color(SUN)
            hop_rows.add(m.VGroup(who_g, words))
        for i, r in enumerate(hop_rows):
            y = 1.95 - 0.95 * i
            bot, name = r[0]
            bot.move_to([X_MIN + 0.3, y, 0])
            name.next_to(bot, m.RIGHT, buff=0.2)
            r[1].move_to([0, y, 0]).align_to([X_MIN + 1.95, 0, 0], m.LEFT)
            sit(name, r[1][0].get_bottom()[1])
        downs = m.VGroup(*[arrow(a[0][0].get_bottom(), b[0][0].get_top(), MUTED, buff=0.08) for a, b in zip(hop_rows, hop_rows[1:])])
        quote = m.VGroup(text("“every handoff is a lossy re-encoding”", 24, INK), cite("Li, AI Agents in Depth · ch. 10")).arrange(m.DOWN, buff=0.12, aligned_edge=m.LEFT)
        quote.next_to(hop_rows, m.DOWN, buff=0.5).align_to([X_MIN, 0, 0], m.LEFT)
        assert max(r.get_right()[0] for r in hop_rows) < 1.2

        lead = otto(0.62).move_to([4.05, 1.95, 0])
        heads = m.VGroup(*[otto(0.22, AZURE, AZURE_FILL) for _ in range(SPAWNED)]).arrange_in_grid(rows=5, cols=10, buff=(0.14, 0.12))
        heads.next_to(lead, m.DOWN, buff=0.3)
        assert len(heads) == SPAWNED
        spawn_cap = m.VGroup(code(f"{SPAWNED} subagents for a simple query", 20, ROSE, weight=m.BOLD), cite("Anthropic, 2025 · early versions")).arrange(m.DOWN, buff=0.1)
        spawn_cap.next_to(heads, m.DOWN, buff=0.25)
        limits = m.VGroup(*[pill(s, SUN, SUN_FILL, 20) for s in ("scaling rules", "a token budget", "a dedicated API key")])
        m.VGroup(m.VGroup(*limits[:2]).arrange(m.RIGHT, buff=0.18), limits[2]).arrange(m.DOWN, buff=0.15)
        limits.next_to(spawn_cap, m.DOWN, buff=0.3)
        limits.set_x(min(limits.get_x(), X_MAX - limits.width / 2))
        assert limits.get_left()[0] > 0.3 and limits.get_bottom()[1] > -3.5

        self.say(
            "[Teams also fail in their own ways.] [A study of multi-agent systems found fourteen failure modes], [in three groups]:",
            self.new_header("06 · failure modes", "How teams fail"),
            m.FadeIn(sub, shift=0.1 * m.UP),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(c), m.FadeIn(g[0])) for c, g in zip(cards, groups)], lag_ratio=0.25),
        )
        self.say(
            "[Poor system design], [agents misaligned with each other], [and weak verification], [like Otto's self-review.]",
            *[m.LaggedStart(*[m.FadeIn(it, shift=0.08 * m.UP) for it in g[1]], lag_ratio=0.15) for g in groups],
            m.AnimationGroup(m.Circumscribe(weak, color=MINT, buff=0.06), m.FadeIn(tag, shift=0.1 * m.UP)),
        )
        self.say(
            "[Every handoff is a lossy re-encoding.] [Pass the ticket from agent to agent], [and the detail that mattered falls away.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(sub, cards, groups, tag), shift=0.2 * m.UP), m.FadeIn(quote, shift=0.1 * m.UP)),
            m.LaggedStart(*[m.AnimationGroup(m.FadeIn(r, shift=0.12 * m.DOWN), *([m.GrowArrow(downs[i - 1])] if i else [])) for i, r in enumerate(hop_rows)], lag_ratio=0.45),
            m.LaggedStart(*[m.Circumscribe(r[1], color=ROSE, buff=0.05) for r in hop_rows[1:]], lag_ratio=0.3),
        )
        self.say(
            "[And agents can spawn agents.] [Early on, Anthropic's system started fifty subagents for simple queries.]",
            m.FadeIn(lead, shift=0.15 * m.UP),
            m.AnimationGroup(m.LaggedStart(*[m.FadeIn(hd, target_position=lead, scale=0.5) for hd in heads], lag_ratio=0.04, run_time=2.2), m.FadeIn(spawn_cap)),
        )
        self.say(
            "[So set limits]: [scaling rules], [a token budget], [and a dedicated API key.]",
            m.Circumscribe(heads, color=ROSE, buff=0.08),
            *[m.FadeIn(p, shift=0.1 * m.UP) for p in limits],
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ A2A

    def a2a(self):
        self.next_section("7 · A2A", notes="A2A (Agent2Agent): released by Google in 2025, now at the Linux Foundation (a2a-protocol.org; in 2026 it joined the Linux Foundation's Agentic AI Foundation). An Agent Card is a JSON document published at /.well-known/agent-card.json: name, description, service url, skills (AgentSkill objects with id, name, …), capabilities, auth. Tasks move through states; we show submitted, working, input-required, completed, failed (the spec also has auth-required, canceled, rejected). Agents collaborate 'without needing to share their internal thoughts, plans, or tool implementations'. 'A2A connects the agents to each other; MCP connects each agent to its own tools' (Episode 16: MCP). The carrier's agent and its card are illustrative.")
        me = badge(CORAL, CORAL_FILL, "Otto", 0.7).move_to([-5.4, 1.6, 0])
        other = badge(LIME, LIME_FILL, "carrier's agent", 0.7).move_to([-1.75, 1.6, 0])
        y_link = me[0].get_y()
        tag = pill("A2A", AZURE, AZURE_FILL, 22).move_to([(me[0].get_right()[0] + other[0].get_left()[0]) / 2, y_link, 0])
        seg = m.VGroup(
            m.Line([me[0].get_right()[0] + 0.12, y_link, 0], [tag.get_left()[0] - 0.08, y_link, 0]),
            m.Line([tag.get_right()[0] + 0.08, y_link, 0], [other[0].get_left()[0] - 0.12, y_link, 0]),
        ).set_stroke(AZURE, 4)
        src = cite("Google, 2025 · now at the Linux Foundation")
        src.next_to(m.VGroup(me, other), m.DOWN, buff=0.35).align_to([X_MIN, 0, 0], m.LEFT)

        body = json_text(CARD_SRC, 20)
        head = code("/.well-known/agent-card.json", 20, LIME, weight=m.BOLD)
        cardg = boxed(m.VGroup(head, body).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT), LIME, pad=(0.35, 0.28))
        cardg.align_to([X_MAX, 0, 0], m.RIGHT).align_to([0, 2.4, 0], m.UP)
        assert cardg.get_left()[0] > other.get_right()[0] + 0.2
        lines = CARD_SRC.split("\n")

        def card_line(i):
            n0 = sum(sum(not c.isspace() for c in ln) for ln in lines[:i])
            return body[n0:n0 + sum(not c.isspace() for c in lines[i])]

        chips = m.VGroup()
        for s in STATES:
            words = code(s, 20, INK)
            chips.add(chip(s, AZURE if s != "completed" else MINT, AZURE_FILL if s != "completed" else MINT_FILL,
                           width=words.width + 0.5, height=0.55, size=24, mono=True))
        chips.arrange(m.RIGHT, buff=0.75).move_to([0, -1.75, 0])
        steps = m.VGroup(*[arrow(a.get_right(), b.get_left(), MUTED, buff=0.08) for a, b in zip(chips, chips[1:])])
        failed = chip("failed", ROSE, ROSE_FILL, width=1.45, height=0.55, size=24, mono=True).next_to(chips[3], m.DOWN, buff=0.45)
        fail_arrow = m.CurvedArrow(chips[1].get_bottom() + 0.06 * m.DOWN, failed.get_left() + 0.08 * m.LEFT, angle=np.pi / 3, color=ROSE, stroke_width=4, tip_length=0.18)
        life = code("task lifecycle", 20, MUTED).next_to(chips, m.UP, buff=0.3).align_to(chips, m.LEFT)
        assert chips.get_left()[0] > X_MIN and chips.get_right()[0] < X_MAX

        lock_name = recolored("lock", MUTED, CARD_HI, prefix="ep26")
        locks = m.VGroup(*[icon(lock_name, 0.42).next_to(b[0], m.UR, buff=-0.05) for b in (me, other)])
        private = text("only tasks and their results cross", 22, INK).next_to(src, m.DOWN, buff=0.22).align_to(src, m.LEFT)

        # MCP is vertical, A2A is horizontal (a2a-protocol.org, "A2A and MCP")
        tool_names = ["issue tracker", "database"]
        tchips = m.VGroup(*[chip(t, SUN, SUN_FILL, width=2.3, height=0.5, size=24, mono=True) for t in tool_names]).arrange(m.DOWN, buff=0.18)
        tchips.move_to([me[0].get_x(), -1.95, 0])
        down = arrow(me.get_bottom() + 0.05 * m.DOWN, tchips.get_top(), SUN, buff=0.1)
        mcp = pill("MCP · ep 16", SUN, SUN_FILL, 20).next_to(down, m.RIGHT, buff=0.2)
        summary = m.VGroup(
            m.VGroup(code("MCP", 22, SUN, weight=m.BOLD), text("connects an agent to its tools", 22, INK)).arrange(m.RIGHT, buff=0.25),
            m.VGroup(code("A2A", 22, AZURE, weight=m.BOLD), text("connects agents to agents", 22, INK)).arrange(m.RIGHT, buff=0.25),
        ).arrange(m.DOWN, buff=0.22, aligned_edge=m.LEFT)
        for row in summary:
            sit(row[1], row[0][0].get_bottom()[1])
        summary.move_to([1.8, -2.2, 0])

        self.say(
            "[Last: what if the other agent belongs to another company], [like tinyshop's shipping carrier?] [That's what A2A is for.]",
            m.AnimationGroup(self.new_header("07 · A2A", "Agents talking to agents"), m.FadeIn(me, shift=0.15 * m.UP)),
            m.FadeIn(other, shift=0.15 * m.UP),
            m.AnimationGroup(m.Create(seg), m.FadeIn(tag, scale=0.8)),
        )
        self.say(
            "[Google released it in twenty twenty-five], [and it now lives at the Linux Foundation.]",
            m.FadeIn(src, shift=0.1 * m.UP),
            m.Circumscribe(src, color=MUTED, buff=0.06),
        )
        self.say(
            "[Each agent publishes an Agent Card]: [its name], [its skills], [and where to reach it.]",
            m.FadeIn(cardg, shift=0.15 * m.UP),
            m.Circumscribe(card_line(1), color=AZURE, buff=0.05),
            m.Circumscribe(card_line(4), color=AZURE, buff=0.05),
            m.Circumscribe(card_line(3), color=AZURE, buff=0.05),
        )
        self.say(
            "[Work travels as a task with a lifecycle]: [submitted], [working], [input required], [then completed, or failed.]",
            m.FadeIn(life, shift=0.1 * m.UP),
            m.FadeIn(chips[0], scale=0.8),
            m.AnimationGroup(m.GrowArrow(steps[0]), m.FadeIn(chips[1], scale=0.8)),
            m.AnimationGroup(m.GrowArrow(steps[1]), m.FadeIn(chips[2], scale=0.8)),
            m.AnimationGroup(m.GrowArrow(steps[2]), m.FadeIn(chips[3], scale=0.8), m.Create(fail_arrow), m.FadeIn(failed, scale=0.8)),
        )
        self.say(
            "[Neither side sees the other's prompts or tools]: [only tasks and their results cross.]",
            m.LaggedStart(*[m.FadeIn(lk, scale=0.7) for lk in locks], lag_ratio=0.3),
            m.AnimationGroup(m.FadeIn(private, shift=0.1 * m.UP), pulse(seg[0], AZURE, 8), pulse(seg[1], AZURE, 8)),
        )
        self.say(
            "[So MCP connects an agent to its tools], [and A2A connects agents to agents.]",
            m.AnimationGroup(m.FadeOut(m.VGroup(life, chips, steps, failed, fail_arrow, src, private), shift=0.15 * m.DOWN),
                             m.GrowArrow(down), m.FadeIn(tchips, shift=0.15 * m.UP), m.FadeIn(mcp), m.FadeIn(summary[0], shift=0.1 * m.UP)),
            m.AnimationGroup(m.Circumscribe(m.VGroup(seg, tag), color=AZURE, buff=0.08), m.FadeIn(summary[1], shift=0.1 * m.UP)),
        )
        self.wait(0.3)
        self.clear()

    # ------------------------------------------------------------------ recap

    def recap(self):
        self.next_section("Recap", notes="Add an agent for evidence one agent can't get (tests, screenshots, tools) or for isolation (a clean context); budget for about fifteen times the tokens.")
        halo = glow(AZURE, radius=4.0)
        words = m.VGroup(*[title_text(w, 66, c) for w, c in (("Evidence", MINT), ("Isolation", AZURE), ("Budget", SUN))])
        subs = m.VGroup(*[code(s, 20, MUTED) for s in ("tests, screenshots, tools", "a clean context per job", "≈ 15× the tokens")])
        slots = [max(w.width, s.width) for w, s in zip(words, subs)]
        x, gap = 0.0, 0.8
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
        assert row_.width < X_MAX - X_MIN
        tagline = text("A second agent is worth it when it sees what the first one can't.", 28, MUTED).next_to(row_, m.DOWN, buff=0.7)

        self.say(
            "[To sum up]: [evidence], [isolation], [budget.]",
            m.AnimationGroup(self.drop_header(), m.FadeIn(halo, scale=0.6)),
            m.AnimationGroup(m.FadeIn(words[0], shift=0.3 * m.UP), m.FadeIn(subs[0])),
            m.AnimationGroup(m.FadeIn(dots[0]), m.FadeIn(words[1], shift=0.3 * m.UP), m.FadeIn(subs[1])),
            m.AnimationGroup(m.FadeIn(dots[1]), m.FadeIn(words[2], shift=0.3 * m.UP), m.FadeIn(subs[2])),
        )
        self.say("[A second agent is worth it when it sees what the first one can't.]", m.Write(tagline))
        self.wait(0.6)
        self.play(*[m.FadeOut(mob) for mob in (halo, row_, tagline)], run_time=0.9)
