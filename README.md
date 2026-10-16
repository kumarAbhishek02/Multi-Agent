# Multi-Agent Console

A Flask web application where a team of AI agents, powered by **LangChain**, **LangGraph**, and local **Ollama** (`qwen2.5:3b`), works on tasks collaboratively.
A **Coordinator** plans the work, delegates subtasks to specialists (in parallel
where they are independent), and writes the final answer. The browser shows every
agent's activity live with real-time SSE event streaming.

| Agent | Role | Extra tools |
|---|---|---|
| Coordinator | Plans, delegates, writes the final answer | `delegate`, artifacts |
| Researcher | Current information from the web, with sources | DuckDuckGo web search, artifacts |
| Data Analyst | Calculations and data analysis | Python execution sandbox, artifacts |
| Software Engineer | Writes, runs and debugs code | Python execution sandbox, artifacts |
| Writer | Reports, emails, documentation | shared artifacts |
| Reviewer | Checks other agents' work | shared artifacts |

All agents share a per-task workspace of named **artifacts** (`save_artifact`,
`read_artifact`, `list_artifacts`) for handing deliverables to each other.

## Setup

1. Make sure Ollama is installed and running with `qwen2.5:3b`:
```powershell
ollama pull qwen2.5:3b
ollama serve
```

2. Setup virtual environment & run:
```powershell
python -m venv .venv
.\.venv\Scripts\pip install -r requirements.txt
Copy-Item .env.example .env
.\.venv\Scripts\python run.py    # http://127.0.0.1:8000
```

Run tests:
```powershell
.\.venv\Scripts\python -m pytest
```

## How it works

```
Browser ──POST /api/runs──▶ RunManager (thread pool) ──▶ Coordinator loop
   ▲                                                        │ delegate (parallel)
   └──── SSE /api/runs/<id>/stream ◀── EventHub ◀── events ─┴▶ Specialist loops
                                          │
                                       SQLite (runs, events, artifacts)
```

- `app/agents/registry.py` - the team: prompts, effort level and tools per agent.
  Add an agent by adding one `_specialist(...)` entry; the coordinator's roster and
  the UI pick it up automatically.
- `app/agents/engine.py` - the streaming tool-use loop shared by every agent, the
  tools, and the handling of bad tool input, truncation and turn limits.
- `app/runs.py` - background execution, cancellation, concurrency limit and error
  reporting.
- `app/routes.py` - JSON API and the server-sent event stream. The stream replays
  the stored event log and resumes from `Last-Event-ID`, so a refresh or a dropped
  connection loses nothing.
- `app/db.py` - SQLite persistence.

### API

| Method | Path | |
|---|---|---|
| `POST` | `/api/runs` | Start a task: `{"task": "..."}` → `202` |
| `GET` | `/api/runs` | Recent runs |
| `GET` | `/api/runs/<id>` | One run with result, usage and artifacts |
| `GET` | `/api/runs/<id>/stream` | Live events (SSE) |
| `POST` | `/api/runs/<id>/cancel` | Cancel an active run |
| `DELETE` | `/api/runs/<id>` | Delete a finished run |
| `GET` | `/api/runs/<id>/artifacts/<name>` | Artifact content |
| `GET` | `/api/agents`, `/api/health` | Team roster, status |

## Model

Every agent uses the Groq model in `AGENT_MODEL` (default `openai/gpt-oss-120b`).
Web search and code execution use Groq's built-in tools, which exist only on its
`openai/gpt-oss-*` models; with any other model the team still runs, but the
Researcher, Data Analyst and Software Engineer work without those tools.

Agents reach them through two ordinary tools, `web_search` and `run_python`, each
of which makes a separate Groq request containing only the built-in tool. Putting
the built-in tools in the same request as the app's own tools makes the model call
tools that do not exist, so they are kept apart.

**Token limits.** Groq's free tier allows 200,000 tokens per day on this model, and
one `web_search` call reads whole pages and uses roughly 30,000. A research-heavy
task can use the daily allowance on its own. If you hit per-minute limits instead,
set `MAX_PARALLEL_TOOLS=1` so specialists run one at a time.

## Configuration

Set in `.env` (see `.env.example`): `GROQ_API_KEY`, `AGENT_MODEL`, `MAX_CONCURRENT_RUNS`,
`MAX_DELEGATIONS` (specialist calls per task), `MAX_AGENT_TURNS`, `MAX_TASK_CHARS`,
`MAX_PARALLEL_TOOLS`, `HOST`, `PORT`.

## Before exposing this beyond your machine

- **There is no authentication.** Anyone who can reach the port can use your Groq
  quota. Keep it on `127.0.0.1` or put it behind an authenticating reverse proxy.
- **Run a single process.** Active runs and SSE wake-ups are tracked in memory;
  `run.py` serves with waitress threads, which is the supported setup. Scaling to
  several processes needs a shared queue and pub/sub (e.g. Redis) in place of
  `RunManager`'s thread pool and `EventHub`.
- A task can make many model calls. `MAX_DELEGATIONS` and `MAX_AGENT_TURNS` bound
  it; the UI shows token counts per task.
- Keep your key in `.env` only. `.env.example` is a template that gets shared.
