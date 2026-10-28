# Agents, Animated — how AI agents really work, from one loop to a team

A video course in 33 short episodes (3.5–4.5 minutes each), narrated and animated. It explains every part of an AI agent by building one: **Otto**, a small coding agent working in a repo called `tinyshop`, grows episode by episode from a dozen lines of loop into a complete harness; then the course steps back to ask how agents see and act in the world, and how they are evaluated, trained, improved and run in production.

Part of a family of narrated, animated video courses, rendered with [ManimGX](https://pypi.org/project/manimgx/) and voiced by a local narrator ([tutorial-video](https://github.com/vpromise/tutorial-video)):

- **AI, Explained** — 46 episodes on modern AI, from first principles to frontier research · [ai-explained](https://github.com/vpromise/ai-explained)
- **Agents, Animated** — how AI agents work, from one loop to a team *(this repository)*
- **Tokens per Second** — how LLM inference really works · [tokens-per-second](https://github.com/vpromise/tokens-per-second)
- **Trial and Reward** — reinforcement learning, from slot machines to reasoning models · [trial-and-reward](https://github.com/vpromise/trial-and-reward)
- **Out of the Noise** — how diffusion models make pictures and video · [out-of-the-noise](https://github.com/vpromise/out-of-the-noise)

This repository holds the **source code** of the course: one scene per episode, the series kit, reference code, research notes and the production tools. The rendered videos are not included.

## Episodes


### Module 1 · Foundations

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 1 | What Is an Agent? | What separates an agent from a chatbot or a fixed script | 3:44 | [`ep01-what-is-an-agent/scene.py`](../ep01-what-is-an-agent/scene.py) |
| 2 | The Agent Loop | The one loop inside every agent | 3:37 | [`ep02-the-agent-loop/scene.py`](../ep02-the-agent-loop/scene.py) |
| 3 | Tools: Giving the Model Hands | How a model "calls" code it cannot run | 3:39 | [`ep03-tools/scene.py`](../ep03-tools/scene.py) |
| 4 | Think, Act, Observe | Why interleaving reasoning and action works, and when to stop | 3:39 | [`ep04-think-act-observe/scene.py`](../ep04-think-act-observe/scene.py) |
| 5 | Workflows or Agents? | When to hard-code the steps and when to let the model decide | 4:21 | [`ep05-workflows-or-agents/scene.py`](../ep05-workflows-or-agents/scene.py) |

### Module 2 · Context Engineering

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 6 | The Context Window | Everything an agent knows in a turn is what its harness puts in the context | 3:53 | [`ep06-the-context-window/scene.py`](../ep06-the-context-window/scene.py) |
| 7 | KV Cache & Prompt Caching | Why agents keep their context append-only | 4:11 | [`ep07-kv-cache-prompt-caching/scene.py`](../ep07-kv-cache-prompt-caching/scene.py) |
| 8 | The System Prompt, Assembled | The system prompt is built by code from parts | 3:47 | [`ep08-system-prompt-assembled/scene.py`](../ep08-system-prompt-assembled/scene.py) |
| 9 | Plans You Can See | Why long tasks need a visible plan and a status bar | 3:45 | [`ep09-plans-you-can-see/scene.py`](../ep09-plans-you-can-see/scene.py) |
| 10 | Skills: Knowledge on Demand | How to give an agent expertise without bloating its context | 3:42 | [`ep10-skills/scene.py`](../ep10-skills/scene.py) |
| 11 | Compaction | What to do when the context fills up | 4:12 | [`ep11-compaction/scene.py`](../ep11-compaction/scene.py) |

### Module 3 · Memory & Knowledge

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 12 | Memory Across Sessions | How an agent remembers between conversations | 4:04 | [`ep12-memory/scene.py`](../ep12-memory/scene.py) |
| 13 | Agentic Search & RAG | How agents find knowledge, from one-shot retrieval to search as a tool | 4:17 | [`ep13-agentic-search-rag/scene.py`](../ep13-agentic-search-rag/scene.py) |
| 14 | The File System as Memory | Why files are an agent's most durable working memory | 4:22 | [`ep14-file-system-as-memory/scene.py`](../ep14-file-system-as-memory/scene.py) |

### Module 4 · Tools & Safety

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 15 | Designing Good Tools | What makes a tool easy for a model to use well | 4:08 | [`ep15-designing-good-tools/scene.py`](../ep15-designing-good-tools/scene.py) |
| 16 | MCP: One Protocol for Tools | How agents plug into outside services | 3:52 | [`ep16-mcp/scene.py`](../ep16-mcp/scene.py) |
| 17 | Permissions & Sandboxes | How to stop an agent from doing damage | 4:12 | [`ep17-permissions-sandboxes/scene.py`](../ep17-permissions-sandboxes/scene.py) |
| 18 | Hooks | How to add behaviour around the loop without changing it | 3:40 | [`ep18-hooks/scene.py`](../ep18-hooks/scene.py) |
| 19 | Prompt Injection | Why text an agent reads can hijack it, and what actually defends | — | _not included in this repository_ |
| 20 | Error Recovery | How agents survive failures | 3:50 | [`ep20-error-recovery/scene.py`](../ep20-error-recovery/scene.py) |

### Module 5 · Scaling Up

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 21 | Subagents | Why delegating to a fresh context helps | 3:51 | [`ep21-subagents/scene.py`](../ep21-subagents/scene.py) |
| 22 | Task Graphs | How goals become ordered, verifiable work | 3:45 | [`ep22-task-graphs/scene.py`](../ep22-task-graphs/scene.py) |
| 23 | Background Work & Events | How agents keep working while slow things happen and the world interrupts | 4:08 | [`ep23-background-work-events/scene.py`](../ep23-background-work-events/scene.py) |
| 24 | Agent Teams | How several persistent agents work together | 4:06 | [`ep24-agent-teams/scene.py`](../ep24-agent-teams/scene.py) |
| 25 | Autonomy & Worktrees | How agents find their own work without colliding | 3:52 | [`ep25-autonomy-worktrees/scene.py`](../ep25-autonomy-worktrees/scene.py) |
| 26 | Multi-Agent Patterns | When many agents beat one, and how to organise them | 4:11 | [`ep26-multi-agent-patterns/scene.py`](../ep26-multi-agent-patterns/scene.py) |

### Module 6 · Beyond the Loop

| # | Episode | What you'll understand | Length | Source |
|---|---|---|---|---|
| 27 | Coding Agents Are General Agents | Why a shell and a file system make an agent general | 3:56 | [`ep27-coding-agents-general/scene.py`](../ep27-coding-agents-general/scene.py) |
| 28 | Computer Use, Voice & Robots | How agents act beyond text, and why timing changes everything | 4:10 | [`ep28-computer-use-voice-robots/scene.py`](../ep28-computer-use-voice-robots/scene.py) |
| 29 | Evaluating Agents | How to know whether an agent is getting better | 4:27 | [`ep29-evaluating-agents/scene.py`](../ep29-evaluating-agents/scene.py) |
| 30 | Training Agents with RL | How models are trained to be better agents | 3:59 | [`ep30-training-agents-rl/scene.py`](../ep30-training-agents-rl/scene.py) |
| 31 | Agents That Learn on the Job | How an agent system improves from experience | 4:20 | [`ep31-learn-on-the-job/scene.py`](../ep31-learn-on-the-job/scene.py) |
| 32 | Running Agents in Production | What it takes to run agents for real users | 4:25 | [`ep32-running-in-production/scene.py`](../ep32-running-in-production/scene.py) |
| 33 | The Complete Agent | How every piece fits into one harness | 3:58 | [`ep33-the-complete-agent/scene.py`](../ep33-the-complete-agent/scene.py) |

## Repository layout

| Path | What |
|---|---|
| `COURSE.md` | the full syllabus: key concepts, what each episode animates, the running example |
| `kit.py` | the series kit: palette, fonts, components, intro and outro cards |
| `harness_reference.py` | the reference agent harness the episodes build |
| `epNN-*/scene.py` | one narrated scene per episode (`self.say("…[cue]…", animation)`: the narration drives the timing) |
| `assets/` | icons and the backdrop |
| `lexicon.json` | pronunciation overrides for the narrator |
| `research/` | research notes (sources, dated facts) and the measurement or experiment scripts |
| `briefs/` | the production briefs each episode was made from, and the QA tools (`check.sh`, `make_index.py`) |
| `LISTEN.md` | narration lines flagged for a human listen |

## Rendering an episode

Requirements: macOS on Apple silicon (the narrator runs Chatterbox on MPS and Whisper on MLX), ManimGX (`uv tool install --python 3.13 manimgx`) and the [tutorial-video](https://github.com/vpromise/tutorial-video) toolkit (set it up with its `scripts/setup_tts.sh`).

```sh
cd epNN-*/
VOICE=draft manimgx inspect scene.py      # silent draft: layout check + storyboards
# in the course folder, start the narrator (it reads lexicon.json):
#   <tutorial-video>/scripts/narrator.sh
manimgx render scene.py                    # MP4 + SRT captions (speaks new lines once, then caches them)
manimgx present scene.py --no-open         # HTML slides
```

## Not in this repository

Rendered MP4s, captions, HTML slides, the narration cache (`voice/`), storyboards and logs.
