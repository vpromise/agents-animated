"""The canonical harness for "Agents, Animated" — every episode that shows code quotes from here,
so names and shapes match across the series (Anthropic Messages API style, as in learn-claude-code).
Episodes show SHORT excerpts (≤ 10 lines), may simplify, but never rename things.

Otto is the series' agent; its working repo is `tinyshop/` (a tiny web shop: cart.py, checkout.py,
tests/). The running task in early episodes: "The checkout test fails. Fix it."
The bug (use it everywhere): cart.total() loops `for i in range(len(items) - 1)` and so skips the last
item; tests/test_cart.py::test_total asserts total([3, 4]) == 7 but gets 3. `pytest -q` prints
"1 failed, 12 passed". The fix: `for item in items:` (or range(len(items))).
"""

import subprocess

import anthropic

client = anthropic.Anthropic()
MODEL = "claude-sonnet-4-5"  # any tool-capable model; episodes just say "the model"
SYSTEM = "You are Otto, a coding agent working in the tinyshop repo."

# ---- tools: a schema the model reads, a handler the harness runs (episode 3) ----
TOOLS = [
    {
        "name": "bash",
        "description": "Run a shell command in the repo and return its output.",
        "input_schema": {
            "type": "object",
            "properties": {"command": {"type": "string"}},
            "required": ["command"],
        },
    },
    {
        "name": "read_file",
        "description": "Read a text file and return its contents.",
        "input_schema": {
            "type": "object",
            "properties": {"path": {"type": "string"}},
            "required": ["path"],
        },
    },
    {
        "name": "edit_file",
        "description": "Replace old_text with new_text in a file.",
        "input_schema": {
            "type": "object",
            "properties": {
                "path": {"type": "string"},
                "old_text": {"type": "string"},
                "new_text": {"type": "string"},
            },
            "required": ["path", "old_text", "new_text"],
        },
    },
]


def run_bash(command):
    done = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=120)
    return (done.stdout + done.stderr)[:50_000]


def read_file(path):
    return open(path).read()


def edit_file(path, old_text, new_text):
    source = open(path).read()
    open(path, "w").write(source.replace(old_text, new_text, 1))
    return f"edited {path}"


TOOL_HANDLERS = {"bash": run_bash, "read_file": read_file, "edit_file": edit_file}  # dispatch table


def run_tool(block):
    return TOOL_HANDLERS[block.name](**block.input)


# ---- the loop (episode 2): call the model, run its tools, feed results back, repeat ----
def agent_loop(messages):
    while True:
        response = client.messages.create(
            model=MODEL, system=SYSTEM, messages=messages, tools=TOOLS, max_tokens=8000
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason != "tool_use":
            return response  # the model answered in words: the turn is over
        results = []
        for block in response.content:
            if block.type == "tool_use":
                output = run_tool(block)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": output})
        messages.append({"role": "user", "content": results})


# What the model emits when it wants a tool (one content block of the assistant message):
#   {"type": "tool_use", "id": "toolu_01", "name": "bash", "input": {"command": "pytest -q"}}
# What the harness sends back (one content block of the next user message):
#   {"type": "tool_result", "tool_use_id": "toolu_01", "content": "1 failed, 12 passed"}

if __name__ == "__main__":
    history = [{"role": "user", "content": "The checkout test fails. Fix it."}]
    print(agent_loop(history).content[0].text)
