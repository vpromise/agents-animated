# Agents, Animated — how AI agents really work, from one loop to a team

A video course in 33 short episodes (3.5–4.5 minutes each), narrated and animated. It explains
every part of an AI agent by building one: **Otto**, a small coding agent working in a repo called
`tinyshop`, grows episode by episode from a dozen lines of loop into a complete harness; then the
course steps back to ask how agents see and act in the world, and how they are evaluated, trained,
improved and run in production.

- **Built on two references, animated:** the hands-on harness of *Learn Claude Code*
  (learn.shareai.run: 20 sessions, one mechanism each, "the loop itself never changes") and the
  principles of Li Bojie's *AI Agents in Depth* (10 chapters: context engineering, memory,
  tools, interaction, evaluation, post-training, evolution, multi-agent). Mechanisms come from the
  first, the why and how-well from the second. Research notes: `research/notes.md`.
- **Who it's for:** developers and curious builders who use agents and want to know what happens
  inside. Module 1 needs no background; later modules build on earlier ones.
- **How every episode works:** one running example; the mechanism step by step with its real data
  structures (messages, tool calls, files); the design trade-offs and failure stories; a
  three-word recap. Code on screen is quoted from one canonical harness (`harness_reference.py`).
- **Visual language:** warm graphite stage; role colours fixed across the series —
  system ORCHID · user AZURE · assistant CORAL · tool call SUN · tool result MINT.
- **Running bug:** in `tinyshop`, `cart.total()` skips the last item; `pytest -q` says
  "1 failed, 12 passed" until Otto fixes it.
- **Format:** 1080p60 MP4 with captions (.srt) and HTML slides; series kit in `kit.py`.

Status: ✅ produced · 🎬 in production · ○ planned · ⏸ on hold

## Module 1 · Foundations — coral

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 1 | **What Is an Agent?** | What separates an agent from a chatbot or a fixed script | Agent = LLM + context + tools (brain, eyes, hands); model + harness; observation and action spaces; the harness as reins | a chatbot answers once from memory; the same model with tools and a loop acts until the job is done | Otto asked "the checkout test fails, fix it": chatbot vs agent | ✅ `ep01-what-is-an-agent` |
| 2 | **The Agent Loop** | The one loop inside every agent | stateless model, messages array, stop_reason, tool_use → tool_result, append-only trajectory, iteration cap | the messages array growing turn by turn while a 10-line while-loop turns; the two exit signals | Otto's first full run: run tests → read cart.py → edit → tests pass | ✅ `ep02-the-agent-loop` |
| 3 | **Tools: Giving the Model Hands** | How a model "calls" code it cannot run | tool schema (name, description, input_schema), dispatch table, tool_use ids, parallel calls, large outputs | a schema read by the model; a tool_use block flying to the harness; dispatch → handler → result back; one line of the loop changing | adding read_file and edit_file to Otto without touching the loop | ✅ `ep03-tools` |
| 4 | **Think, Act, Observe** | Why interleaving reasoning and action works, and when to stop | ReAct, observations changing the plan, stop conditions, "an answer is not a completed task" | a ReAct trace built line by line; a wrong hypothesis corrected by a test run; the same task without tools producing a confident wrong answer | Otto's first guess about the bug, disproved by pytest | ✅ `ep04-think-act-observe` |
| 5 | **Workflows or Agents?** | When to hard-code the steps and when to let the model decide | single call → workflow → agent; chaining, routing, parallelization, orchestrator–workers, evaluator–optimizer; mixing both | five workflow shapes assembled as diagrams; a dial from fixed workflow to autonomous agent; the over-engineered five-agent design | a refund flow as a fixed workflow vs Otto's open-ended debugging | ✅ `ep05-workflows-or-agents` |

## Module 2 · Context Engineering — sun

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 6 | **The Context Window** | Everything an agent knows in a turn is what its harness puts in the context | static prefix + trajectory, message roles, tokens and budget, overflow vs rot, lost in the middle | a ContextColumn filling with every kind of message; a token meter; attention thinning in the middle | Otto's context after ten turns, piece by piece | ✅ `ep06-the-context-window` |
| 7 | **KV Cache & Prompt Caching** | Why agents keep their context append-only | prefill vs decode, KV cache, prefix caching, cache reads ≈ one tenth the price, what breaks the cache | a cached prefix lighting up; one edit early in the prompt invalidating everything after it; cost of re-sent history | a "current time" line in the system prompt doubling the bill | ✅ `ep07-kv-cache-prompt-caching` |
| 8 | **The System Prompt, Assembled** | The system prompt is built by code from parts | sections (identity, tools, workspace, memory), static before dynamic, the new-team-member test, project files (AGENTS.md), constraints over guidance | the prompt assembled from labelled blocks; the static prefix kept stable; a rule moved from prompt to linter | Otto's system prompt built for tinyshop | ✅ `ep08-system-prompt-assembled` |
| 9 | **Plans You Can See** | Why long tasks need a visible plan and a status bar | TodoWrite, todo states, the nag reminder, the agent status bar, reminders at the end of context | a todo list ticking through states; a reminder appended when the plan goes stale; a status bar computed by code | Otto adding a discount feature in five steps | ✅ `ep09-plans-you-can-see` |
| 10 | **Skills: Knowledge on Demand** | How to give an agent expertise without bloating its context | SKILL.md with YAML front matter, catalog vs body, progressive disclosure, load via tool result, descriptions as routing rules | the catalog (~100 tokens per skill) resident; one skill body (~2,000 tokens) loaded only when needed | Otto learning tinyshop's release procedure from a skill | ✅ `ep10-skills` |
| 11 | **Compaction** | What to do when the context fills up | cheap-first layers (spill to disk, snip, micro-compact, summary), thresholds, what must survive, the 3-failure breaker | the meter crossing a threshold; old tool results shrinking; history folded into a summary while recent turns stay | a long Otto session approaching the limit | ✅ `ep11-compaction` |

## Module 3 · Memory & Knowledge — orchid

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 12 | **Memory Across Sessions** | How an agent remembers between conversations | memory types, MEMORY.md index + files, extract at the end, select at the start, consolidation, recall → disambiguation → proactive | a session ending and facts being written; a new session starting with them; memories merged while the agent "sleeps" | Otto remembering "use tabs, not spaces" and the test command | ✅ `ep12-memory` |
| 13 | **Agentic Search & RAG** | How agents find knowledge, from one-shot retrieval to search as a tool | chunking, embeddings vs BM25, hybrid + reranking, why naive RAG fails, agentic RAG, grep over indexes | a one-shot retrieval missing the answer vs an agent searching, reading and searching again | finding where tinyshop's discount rule is defined | ✅ `ep13-agentic-search-rag` |
| 14 | **The File System as Memory** | Why files are an agent's most durable working memory | notes and progress files, handover artifacts, large outputs spilled to disk, knowledge as a directory tree, knowledge changes as reviewed diffs | Otto writing NOTES.md and progress.md, losing its context, and picking up from the files | a multi-session refactor of tinyshop | ✅ `ep14-file-system-as-memory` |

## Module 4 · Tools & Safety — mint

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 15 | **Designing Good Tools** | What makes a tool easy for a model to use well | tools shaped to goals not APIs, general vs dedicated tools, descriptions that say when and what not, honest truncation, edit formats | a bad tool confusing Otto and a redesigned one working; the curly-quote mismatch loop | Otto's search and edit tools, redesigned | ✅ `ep15-designing-good-tools` |
| 16 | **MCP: One Protocol for Tools** | How agents plug into outside services | MCP client and server, tools/list and tools/call, JSON-RPC over stdio or HTTP, tool name prefixes, too many tools, deferred loading, poisoned descriptions | Otto discovering a server's tools at start-up and calling one; a tool pool growing too large | connecting Otto to an issue-tracker server | ✅ `ep16-mcp` |
| 17 | **Permissions & Sandboxes** | How to stop an agent from doing damage | the permission gate (deny, ask, allow), rules vs string matching, three guardrail layers, the sandbox ladder, an independent reviewer | three commands routed through the gate; a blacklist bypassed; a sandbox drawn around the shell | Otto about to run `rm -rf` in the wrong folder | ✅ `ep17-permissions-sandboxes` |
| 18 | **Hooks** | How to add behaviour around the loop without changing it | UserPromptSubmit, PreToolUse, PostToolUse, Stop; blocking vs observing; allow cannot bypass deny; the stop-hook death spiral | hooks clipping onto the loop at its four events; a hook blocking a call; a formatter running after every edit | running the tests after every edit Otto makes | ✅ `ep18-hooks` |
| 19 | **Prompt Injection** | Why text an agent reads can hijack it, and what actually defends | direct vs indirect injection, the lethal trifecta, memory as an amplifier, labelled external content, defences outside the model | an instruction hidden in a fetched README travelling into the context and steering Otto; the action stopped at the data layer | a poisoned README asking Otto to send a secret | ✅ `ep19-prompt-injection` |
| 20 | **Error Recovery** | How agents survive failures | failure layers, classify before retrying, exponential backoff with jitter, bigger output cap, fallback model, loop fingerprints, giving up well | errors sorted into bins; a backoff curve; a truncated answer escalated then continued | a rate-limited API and a truncated edit | ✅ `ep20-error-recovery` |

## Module 5 · Scaling Up — azure

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 21 | **Subagents** | Why delegating to a fresh context helps | the task tool, a clean history, a summary back, isolation over compression, forks for cache hits, no recursion | a main context spawning a child that reads thirty files and returns one short report | Otto sending a subagent to survey tinyshop | ✅ `ep21-subagents` |
| 22 | **Task Graphs** | How goals become ordered, verifiable work | tasks with blockers, claim and complete, a task board, long-running agents with handover, "done" must be verified | a dependency graph lighting up; a "done" claim rejected until a check passes | "add payments" as a task graph | ✅ `ep22-task-graphs` |
| 23 | **Background Work & Events** | How agents keep working while slow things happen and the world interrupts | background tasks, placeholders and notifications, cron jobs, event-driven loops, push vs poll, urgency lanes | a slow test suite running in a side lane while the loop continues; a scheduled job firing; urgent events cutting in | a long build, a nightly check, and a user saying "stop" | ✅ `ep23-background-work-events` |
| 24 | **Agent Teams** | How several persistent agents work together | teammates vs subagents, mailboxes, request–response protocols, correlation ids, shutdown and plan approval | messages moving between mailboxes; a protocol state machine | a lead Otto with a tester and a reviewer | ✅ `ep24-agent-teams` |
| 25 | **Autonomy & Worktrees** | How agents find their own work without colliding | idle polling, claiming unowned tasks, identity after compaction, git worktrees, one directory per task | two agents claiming tasks from a board, each in its own worktree branch | two Ottos fixing two bugs in parallel | ✅ `ep25-autonomy-worktrees` |
| 26 | **Multi-Agent Patterns** | When many agents beat one, and how to organise them | shared vs separate context, manager vs peer, proposer–reviewer, debate, the cost (~15× tokens), failure modes, A2A | a map of patterns; a reviewer with a test runner catching a bug that self-review missed | Otto's patch reviewed by a second agent | ✅ `ep26-multi-agent-patterns` |

## Module 6 · Beyond the Loop — lime

| # | Episode | The viewer will understand | Key concepts | Mechanism to animate | Running example | Status |
|---|---|---|---|---|---|---|
| 27 | **Coding Agents Are General Agents** | Why a shell and a file system make an agent general | code as action, the file system as the hub, general agents (Claude Code, OpenClaw, Manus), verification decides what works | the same loop turning a spreadsheet into a report by writing code; the goal-clarity × verification grid | Otto asked for a sales report instead of a bug fix | ✅ `ep27-coding-agents-general` |
| 28 | **Computer Use, Voice & Robots** | How agents act beyond text, and why timing changes everything | observation and action spaces, timescales, the screenshot → action loop, grounding, voice pipelines and latency, robots choosing skills | a screenshot → click loop; a voice turn as a pipeline; the robotics layer cake from minutes to milliseconds | a browser task, a phone call, a robot picking up a cup | ✅ `ep28-computer-use-voice-robots` |
| 29 | **Evaluating Agents** | How to know whether an agent is getting better | tasks with environments and verifiers, pass@k vs pass^k, LLM-as-a-judge and its biases, first-error attribution, confidence intervals | an eval harness running tasks; pass@k and pass^k curves diverging; a trace searched for its first error | a tinyshop eval set | ✅ `ep29-evaluating-agents` |
| 30 | **Training Agents with RL** | How models are trained to be better agents | pretraining → SFT → RL, rollouts, verifiable rewards, group-relative advantages, reward hacking, distillation | sixteen rollouts in sandboxes, rewards, one update nudging the policy | training on bug-fix tasks like tinyshop's | ✅ `ep30-training-agents-rl` |
| 31 | **Agents That Learn on the Job** | How an agent system improves from experience | signals from traces, the four carriers (knowledge, prompts and skills, programs, weights), validation, canary and rollback | a week of traces distilled into a new rule, tested, canaried and kept | Otto's skills improved from its own failures | ✅ `ep31-learn-on-the-job` |
| 32 | **Running Agents in Production** | What it takes to run agents for real users | cost per task, latency budgets, tracing (traces and spans), checkpoints and resume, budgets and rate limits, evals in CI | a trace waterfall; cost and latency ledgers; a crash resumed from a checkpoint | Otto serving a team of developers | ✅ `ep32-running-in-production` |
| 33 | **The Complete Agent** | How every piece fits into one harness | the full turn: hooks, notifications, compaction, prompt assembly, model, permissions, tools, recovery; the book's five patterns | Otto's whole architecture assembled layer by layer around one unchanged loop | Otto, complete, fixing the course's first bug again | ✅ `ep33-the-complete-agent` |

## Paths through the course

- **Build your own agent:** 1 → 2 → 3 → 6 → 8 → 11 → 15 → 17 → 21 → 33.
- **Context engineering:** 6 → 7 → 8 → 9 → 10 → 11 → 12 → 14.
- **Safety:** 17 → 18 → 19 → 20 → 29.
- **Research and training:** 4 → 26 → 28 → 29 → 30 → 31.

## Production notes

- Series kit: `kit.py` (palette, components, Otto, cards); component tour: `kit_demo.py`;
  canonical code: `harness_reference.py`; research: `research/notes.md`.
- Episodes are made by subagents from `briefs/epNN.md` (generated by `briefs/make_briefs.py` from
  `briefs/spec.py`), one shared narrator; each episode is reviewed by the coordinator (numbers,
  storyboards, `briefs/check.sh N`).
