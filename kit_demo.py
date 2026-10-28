"""A tour of the kit's components (draft voice): VOICE=draft manimgx inspect kit_demo.py"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from kit import *  # noqa: F403


class KitDemo(Episode):
    number = 6
    voice = draft_voice

    def lesson(self):
        ctx = ContextColumn(meter=True)
        bot = otto(1.1).move_to([-4.6, 1.2, 0])
        call = tool_call("read_file", {"path": "tests/test_cart.py"}).move_to([-3.2, -1.6, 0])
        self.say("[Here is the context window,] [and here is Otto, our agent.]",
                 m.AnimationGroup(self.new_header("02 · context", "The context window"), m.FadeIn(ctx.frame)),
                 m.FadeIn(bot, shift=0.3 * m.UP))
        self.say("[A system prompt,] [then the user's request.]",
                 ctx.push(ctx.message("system", "You are Otto, a coding agent.\nTools: read_file, run_bash")),
                 m.AnimationGroup(ctx.push(ctx.message("user", "The checkout test fails. Fix it?")), ctx.meter.fill_to(0.25)))
        self.say("[The model asks for a tool,] [and the harness answers.]",
                 m.AnimationGroup(m.FadeIn(call, shift=0.2 * m.UP), ctx.push(ctx.message("tool_use", 'read_file("tests/test_cart.py")', mono=True))),
                 m.AnimationGroup(ctx.push(ctx.message("tool_result", "def test_total():\n    assert total([3, 4]) == 7", mono=True)), ctx.meter.fill_to(0.55)))
        self.say("[More turns pile up] [until older messages scroll away.]",
                 m.AnimationGroup(ctx.push(ctx.message("assistant", "The bug is in cart.py: it skips the last item.")), ctx.meter.fill_to(0.8)),
                 m.AnimationGroup(ctx.push(ctx.message("tool_use", 'run_bash("pytest -q")', mono=True)), ctx.meter.fill_to(0.95)))
        self.clear()
        term = terminal(["$ pytest -q", "! FAILED tests/test_cart.py::test_total", "1 failed, 12 passed in 0.41s"], width=6.6).move_to([-3.4, 1.2, 0])
        src = code_block('def loop(messages):\n    while True:\n        reply = model(messages)  # call\n        if reply.stop == "end_turn":\n            return reply\n        messages += run_tools(reply)', title="agent.py").move_to([3.3, 1.0, 0])
        ring = loop_ring(["think", "act", "observe"], radius=1.25, center=(-3.4, -2.0, 0))
        tree = file_tree(["tinyshop/", "tinyshop/cart.py", "tinyshop/checkout.py", "tests/", "tests/test_cart.py"], highlight=("cart.py",)).move_to([3.3, -2.3, 0])
        self.say("[A terminal,] [a code block,] [the loop,] [and a file tree.]",
                 m.AnimationGroup(self.new_header("01 · foundations", "Components"), m.FadeIn(term)),
                 m.FadeIn(src), m.AnimationGroup(m.FadeIn(ring[0]), m.Create(ring[1])), m.FadeIn(tree))
        res = tool_result("1 failed, 12 passed", error=True).move_to([3.3, -2.3, 0])
        self.say("[A failed tool result.]", m.FadeTransform(tree, res))
        self.clear()
        chips = m.VGroup(*[chip(w, c, f, width=2.2) for w, c, f in (("system", ORCHID, ORCHID_FILL), ("user", AZURE, AZURE_FILL), ("assistant", CORAL, CORAL_FILL), ("tool call", SUN, SUN_FILL), ("tool result", MINT, MINT_FILL))]).arrange(m.RIGHT, buff=0.25)
        big = title_text("Loop · Tools · Context", 56, INK).next_to(chips, m.UP, buff=0.8)
        self.say("[The role colours,] [and a recap in Futura.]", m.FadeIn(chips, lag_ratio=0.15), m.Write(big))
        self.clear()
