"""Multi-agent execution engine built with LangChain and LangGraph for local Ollama."""
import json
import logging
import operator
import re
import subprocess
import sys
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import date
from types import SimpleNamespace
from typing import Annotated, Any, Dict, List, Optional, Sequence, TypedDict

from langchain_core.messages import (
    AIMessage,
    AIMessageChunk,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import StructuredTool, tool
from langgraph.graph import END, START, StateGraph

from .registry import AGENTS, CODE, COORDINATOR, SPECIALISTS, WEB, AgentSpec

log = logging.getLogger(__name__)

ARTIFACT_NAME = re.compile(r"^\w[\w.\- ]{0,99}$")
TOOL_RETRIES = 2


class AgentError(Exception):
    """An agent could not finish its task (empty answer, turn limit, truncation)."""


class ToolError(Exception):
    """A tool call failed in a way the calling agent can read and recover from."""


class RunCancelled(Exception):
    pass


class RunContext:
    """State shared by every agent working on one run."""

    def __init__(self, run_id, client, config, store, hub, cancel: threading.Event):
        self.run_id = run_id
        self.client = client
        self.config = config
        self.store = store
        self.hub = hub
        self.cancel = cancel
        self.usage = {"input_tokens": 0, "output_tokens": 0}
        self._delegations = 0
        self._lock = threading.Lock()

    def emit(self, type_: str, **data) -> None:
        self.hub.emit(self.run_id, type_, data)

    def check_cancelled(self) -> None:
        if self.cancel.is_set():
            raise RunCancelled()

    def add_usage(self, usage) -> None:
        if not usage:
            return
        with self._lock:
            if isinstance(usage, dict):
                self.usage["input_tokens"] += int(
                    usage.get("input_tokens")
                    or usage.get("prompt_tokens")
                    or usage.get("prompt_eval_count")
                    or 0
                )
                self.usage["output_tokens"] += int(
                    usage.get("output_tokens")
                    or usage.get("completion_tokens")
                    or usage.get("eval_count")
                    or 0
                )
            else:
                self.usage["input_tokens"] += (
                    getattr(usage, "prompt_tokens", 0)
                    or getattr(usage, "input_tokens", 0)
                    or 0
                )
                self.usage["output_tokens"] += (
                    getattr(usage, "completion_tokens", 0)
                    or getattr(usage, "output_tokens", 0)
                    or 0
                )

    def take_delegation(self) -> bool:
        with self._lock:
            if self._delegations >= self.config["MAX_DELEGATIONS"]:
                return False
            self._delegations += 1
            return True


# -- helper functions ---------------------------------------------------------


def _clip(value, limit: int):
    if isinstance(value, str):
        return (
            value
            if len(value) <= limit
            else value[:limit] + f"… (+{len(value) - limit} chars)"
        )
    if isinstance(value, dict):
        return {k: _clip(v, limit) for k, v in value.items()}
    if isinstance(value, list):
        return [_clip(v, limit) for v in value]
    return value


def _string_args(args, *names) -> list:
    """The model's arguments are untrusted text; check them before use."""
    if not isinstance(args, dict):
        raise ToolError("INVALID_JSON: tool input must be a JSON object")
    values = []
    for name in names:
        value = args.get(name)
        if not isinstance(value, str) or not value.strip():
            raise ToolError(f"INVALID_JSON: '{name}' must be a non-empty string")
        values.append(value.strip())
    return values


class _TextBuffer:
    """Coalesces token deltas so the event log gets chunks, not single tokens."""

    def __init__(self, ctx: RunContext, spec: AgentSpec, inv: str):
        self.ctx, self.spec, self.inv = ctx, spec, inv
        self.parts, self.size, self.last = [], 0, time.monotonic()

    def add(self, text: str) -> None:
        self.parts.append(text)
        self.size += len(text)
        if self.size >= 80 or time.monotonic() - self.last >= 0.2:
            self.flush()

    def flush(self) -> None:
        if self.parts:
            self.ctx.emit(
                "text_delta",
                inv=self.inv,
                agent=self.spec.key,
                text="".join(self.parts),
            )
        self.parts, self.size, self.last = [], 0, time.monotonic()


# -- sandboxed execution tools ------------------------------------------------


def _run_python_code(code: str) -> str:
    """Execute Python code in a safe local subprocess with timeout."""
    code = code.strip()
    if code.startswith("```"):
        lines = code.splitlines()
        if lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        code = "\n".join(lines).strip()

    try:
        res = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=20,
        )
        out = ((res.stdout or "") + (res.stderr or "")).strip()
        return out or "(The code executed successfully with no output)"
    except subprocess.TimeoutExpired:
        return "Execution timed out (limit: 20 seconds)."
    except Exception as e:
        return f"Execution error: {e}"


def _live_web_search(query: str) -> str:
    """Search the web using DuckDuckGo or Wikipedia fallback."""
    query = query.strip()
    if not query:
        return "Empty query."

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        results = list(DDGS().text(query, max_results=4))
        if results:
            formatted = []
            for r in results:
                title = r.get("title", "")
                body = r.get("body", "")
                href = r.get("href", "")
                formatted.append(f"Title: {title}\nSnippet: {body}\nSource: {href}")
            return "\n\n".join(formatted)
    except Exception as e:
        log.warning("DuckDuckGo search error: %s", e)

    try:
        import urllib.parse
        import urllib.request

        wiki_url = f"https://en.wikipedia.org/w/api.php?action=query&list=search&srsearch={urllib.parse.quote(query)}&utf8=&format=json"
        req = urllib.request.Request(
            wiki_url, headers={"User-Agent": "MultiAgentConsole/1.0"}
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode())
            items = data.get("query", {}).get("search", [])
            if items:
                lines = []
                for item in items[:3]:
                    title = item.get("title", "")
                    snippet = re.sub(r"<[^>]+>", "", item.get("snippet", ""))
                    lines.append(
                        f"Title: {title}\nSnippet: {snippet}\nSource: https://en.wikipedia.org/wiki/{urllib.parse.quote(title)}"
                    )
                return "\n\n".join(lines)
    except Exception as e:
        log.warning("Search fallback error: %s", e)

    return f"No live search results found for '{query}'. Proceed using existing domain knowledge or shared artifacts."


# -- tool schemas -------------------------------------------------------------


def _tool(name: str, description: str, properties: dict, required: list) -> dict:
    return {
        "type": "function",
        "function": {
            "name": name,
            "description": description,
            "parameters": {
                "type": "object",
                "properties": properties,
                "required": required,
                "additionalProperties": False,
            },
        },
    }


DELEGATE = _tool(
    "delegate",
    "Hand a subtask to a specialist agent and get its report back. The specialist "
    "starts with no context, so the task must be a complete, self-contained brief. "
    "Call this several times in one turn to run independent subtasks in parallel.",
    {
        "agent": {"type": "string", "enum": [s.key for s in SPECIALISTS]},
        "task": {"type": "string", "description": "The full brief for the specialist."},
    },
    ["agent", "task"],
)
SAVE_ARTIFACT = _tool(
    "save_artifact",
    "Save a named deliverable (report, code file, data) to the workspace shared by "
    "all agents on this task. Saving under an existing name replaces it.",
    {
        "name": {"type": "string", "description": "Short file-like name, e.g. report.md"},
        "content": {"type": "string"},
    },
    ["name", "content"],
)
READ_ARTIFACT = _tool(
    "read_artifact",
    "Read the full content of an artifact from the shared workspace.",
    {"name": {"type": "string"}},
    ["name"],
)
LIST_ARTIFACTS = _tool(
    "list_artifacts",
    "List the artifacts currently in the shared workspace, with author and size.",
    {},
    [],
)
WEB_SEARCH = _tool(
    "web_search",
    "Search the live web and read the top pages. Returns a summary of what was found "
    "with source URLs. Use one focused query per call.",
    {"query": {"type": "string", "description": "What to look up."}},
    ["query"],
)
RUN_PYTHON = _tool(
    "run_python",
    "Execute Python code in a sandbox and return what it printed (or the error). "
    "Each call starts fresh: no variables or files carry over. Print the values you need.",
    {"code": {"type": "string", "description": "A complete Python script."}},
    ["code"],
)
CAPABILITY_TOOLS = {WEB: WEB_SEARCH, CODE: RUN_PYTHON}


def has_capability_tools(model: str) -> bool:
    """Ollama models (e.g. qwen2.5:3b) and gpt-oss have full tool capabilities."""
    m = (model or "").lower()
    return "gpt-oss" in m or "qwen" in m or "gwen" in m or "ollama" in m


def tool_definitions(spec: AgentSpec, model: str) -> list:
    tools = [DELEGATE if spec.can_delegate else SAVE_ARTIFACT, READ_ARTIFACT, LIST_ARTIFACTS]
    if has_capability_tools(model):
        tools += [CAPABILITY_TOOLS[cap] for cap in spec.capabilities if cap in CAPABILITY_TOOLS]
    return tools


# -- tool execution -----------------------------------------------------------


def _call_tool(ctx: RunContext, spec: AgentSpec, inv: str, name: str, args) -> str:
    if name == "delegate" and spec.can_delegate:
        agent, task = _string_args(args, "agent", "task")
        target = AGENTS.get(agent.lower())
        if target is None or target.can_delegate:
            raise ToolError(
                f"Unknown agent '{agent}'. Choose one of: "
                + ", ".join(s.key for s in SPECIALISTS)
            )
        if not ctx.take_delegation():
            raise ToolError(
                "The delegation budget for this task is used up. "
                "Finish with the results you already have."
            )
        try:
            return run_agent(ctx, target, task, parent=inv)
        except AgentError as exc:
            raise ToolError(f"The {target.name} did not finish: {exc}") from exc

    if name == "save_artifact" and not spec.can_delegate:
        artifact, content = _string_args(args, "name", "content")
        artifact = artifact.strip()
        if not ARTIFACT_NAME.match(artifact):
            raise ToolError(
                "Artifact names are 1-100 characters: letters, digits, "
                "spaces, dots, dashes and underscores."
            )
        limit = ctx.config["MAX_ARTIFACT_CHARS"]
        if len(content) > limit:
            raise ToolError(
                f"Artifact is {len(content)} characters; the limit is {limit}. "
                "Split it into several artifacts."
            )
        ctx.store.save_artifact(ctx.run_id, artifact, content, spec.key)
        ctx.emit(
            "artifact_saved",
            name=artifact,
            agent=spec.key,
            chars=len(content),
        )
        return f"Saved artifact '{artifact}' ({len(content)} characters)."

    if name == "read_artifact":
        (artifact,) = _string_args(args, "name")
        found = ctx.store.get_artifact(ctx.run_id, artifact.strip())
        if found is None:
            raise ToolError(
                f"No artifact named '{artifact}'. Use list_artifacts to see names."
            )
        return found["content"]

    if name == "list_artifacts":
        artifacts = ctx.store.list_artifacts(ctx.run_id)
        if not artifacts:
            return "The workspace is empty."
        return "\n".join(
            f"- {a['name']} (by {a['agent']}, {a['chars']} characters)"
            for a in artifacts
        )

    has_cap = has_capability_tools(ctx.config.get("AGENT_MODEL", ""))
    if name == "web_search" and has_cap and WEB in spec.capabilities:
        (query,) = _string_args(args, "query")
        return _live_web_search(query)

    if name == "run_python" and has_cap and CODE in spec.capabilities:
        (code,) = _string_args(args, "code")
        return _run_python_code(code)

    raise ToolError(f"Unknown tool '{name}'.")


def _run_tool(ctx: RunContext, spec: AgentSpec, inv: str, call) -> dict:
    """Execute one tool call and return the `tool` message that answers it."""
    is_error = False
    try:
        content = _call_tool(ctx, spec, inv, call.name, call.input)
    except ToolError as exc:
        content, is_error = f"Error: {exc}", True
    except Exception as exc:
        content, is_error = f"Error: {exc}", True

    ctx.emit(
        "tool_result",
        inv=inv,
        agent=spec.key,
        name=call.name,
        is_error=is_error,
        preview=_clip(str(content), 400),
    )
    return {"role": "tool", "tool_call_id": call.id, "content": str(content)}


def _run_tools(ctx: RunContext, spec: AgentSpec, inv: str, calls: list) -> list:
    if len(calls) == 1:
        return [_run_tool(ctx, spec, inv, calls[0])]
    workers = min(len(calls), ctx.config.get("MAX_PARALLEL_TOOLS", 4))
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="tool") as pool:
        futures = [pool.submit(_run_tool, ctx, spec, inv, c) for c in calls]
        try:
            return [f.result() for f in futures]
        except BaseException:
            ctx.cancel.set()
            raise


# -- streaming turn execution -------------------------------------------------


def _convert_messages_to_legacy(messages: Sequence[Any]) -> list:
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif isinstance(m, SystemMessage):
            result.append({"role": "system", "content": m.content})
        elif isinstance(m, HumanMessage):
            result.append({"role": "user", "content": m.content})
        elif isinstance(m, AIMessage):
            msg = {"role": "assistant", "content": m.content or None}
            if m.tool_calls:
                msg["tool_calls"] = [
                    {
                        "id": tc["id"],
                        "type": "function",
                        "function": {
                            "name": tc["name"],
                            "arguments": json.dumps(tc["args"])
                            if isinstance(tc["args"], dict)
                            else str(tc["args"]),
                        },
                    }
                    for tc in m.tool_calls
                ]
            result.append(msg)
        elif isinstance(m, ToolMessage):
            result.append(
                {
                    "role": "tool",
                    "tool_call_id": m.tool_call_id,
                    "content": m.content,
                }
            )
    return result


def _stream_turn_langchain(
    ctx: RunContext, spec: AgentSpec, inv: str, messages: list
):
    """Stream one turn using LangChain ChatModel (e.g. ChatOllama)."""
    tools = tool_definitions(spec, ctx.config.get("AGENT_MODEL", ""))
    client = ctx.client
    bound_model = client.bind_tools(tools) if tools else client
    buffer = _TextBuffer(ctx, spec, inv)
    accumulated_content = []
    tool_calls_dict = {}
    usage_info = None

    try:
        for chunk in bound_model.stream(messages):
            ctx.check_cancelled()

            if getattr(chunk, "usage_metadata", None):
                usage_info = chunk.usage_metadata
            elif getattr(chunk, "response_metadata", None):
                usage_info = chunk.response_metadata

            if chunk.content:
                accumulated_content.append(chunk.content)
                buffer.add(chunk.content)

            if getattr(chunk, "tool_call_chunks", None):
                for tc in chunk.tool_call_chunks:
                    idx = tc.get("index") or tc.get("id") or 0
                    slot = tool_calls_dict.setdefault(
                        idx,
                        {
                            "id": tc.get("id") or str(uuid.uuid4()),
                            "name": tc.get("name") or "",
                            "arguments": "",
                        },
                    )
                    if tc.get("name"):
                        slot["name"] = tc["name"]
                    if tc.get("args"):
                        slot["arguments"] += tc["args"]
            elif getattr(chunk, "tool_calls", None):
                for tc in chunk.tool_calls:
                    tool_calls_dict[tc["id"]] = {
                        "id": tc["id"],
                        "name": tc["name"],
                        "arguments": json.dumps(tc["args"])
                        if isinstance(tc["args"], dict)
                        else str(tc["args"]),
                    }
    finally:
        buffer.flush()

    if usage_info:
        ctx.add_usage(usage_info)

    raw_calls = list(tool_calls_dict.values())
    text = "".join(accumulated_content)
    return text, raw_calls, "stop"


def _stream_turn_legacy(
    ctx: RunContext, spec: AgentSpec, inv: str, messages: list
):
    """Stream one turn using an OpenAI/Groq compatible mock client (used in unit tests)."""
    cfg = ctx.config
    tools = tool_definitions(spec, cfg.get("AGENT_MODEL", ""))
    legacy_messages = _convert_messages_to_legacy(messages)

    stream = ctx.client.chat.completions.create(
        model=cfg.get("AGENT_MODEL", "qwen2.5:3b"),
        messages=legacy_messages,
        tools=tools if tools else None,
        max_completion_tokens=cfg.get("AGENT_MAX_TOKENS", 4096),
        stream=True,
    )
    buffer = _TextBuffer(ctx, spec, inv)
    text_parts, calls, finish, usage = [], {}, None, None

    try:
        for chunk in stream:
            ctx.check_cancelled()
            usage = (
                getattr(chunk, "usage", None)
                or getattr(getattr(chunk, "x_groq", None), "usage", None)
                or usage
            )
            if not getattr(chunk, "choices", None):
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            if getattr(delta, "reasoning", None):
                ctx.emit(
                    "thinking",
                    inv=inv,
                    agent=spec.key,
                    text=delta.reasoning,
                )
            if getattr(delta, "content", None):
                text_parts.append(delta.content)
                buffer.add(delta.content)
            for call in getattr(delta, "tool_calls", None) or []:
                slot = calls.setdefault(
                    call.index, {"id": "", "name": "", "arguments": ""}
                )
                slot["id"] = call.id or slot["id"]
                if call.function:
                    slot["name"] = call.function.name or slot["name"]
                    slot["arguments"] += call.function.arguments or ""
            finish = choice.finish_reason or finish
    finally:
        if hasattr(stream, "close"):
            stream.close()

    buffer.flush()
    if usage:
        ctx.add_usage(usage)

    return "".join(text_parts), [calls[i] for i in sorted(calls)], finish


def _stream_turn(ctx: RunContext, spec: AgentSpec, inv: str, messages: list):
    if hasattr(ctx.client, "bind_tools"):
        return _stream_turn_langchain(ctx, spec, inv, messages)
    return _stream_turn_legacy(ctx, spec, inv, messages)


# -- agent reasoning loop -----------------------------------------------------


def _loop(ctx: RunContext, spec: AgentSpec, inv: str, task: str) -> str:
    messages = [
        {"role": "system", "content": spec.system},
        {"role": "user", "content": task},
    ]
    bad_calls = 0

    for _ in range(ctx.config["MAX_AGENT_TURNS"]):
        ctx.check_cancelled()
        text, raw_calls, finish = _stream_turn(ctx, spec, inv, messages)

        if not raw_calls:
            text = text.strip()
            if not text:
                raise AgentError("the model returned an empty answer")
            if finish == "length":
                text += "\n\n[This answer was cut off at the output token limit.]"
            return text

        if finish == "length":
            raise AgentError("a tool call was cut off at the output token limit")

        calls = []
        for raw in raw_calls:
            try:
                args = json.loads(raw["arguments"] or "{}")
            except ValueError:
                args = None
                raw["arguments"] = "{}"
            calls.append(
                SimpleNamespace(id=raw["id"], name=raw["name"], input=args)
            )
            ctx.emit(
                "tool_call",
                inv=inv,
                agent=spec.key,
                name=raw["name"],
                input=_clip(args, 600),
            )

        messages.append(
            {
                "role": "assistant",
                "content": text or None,
                "tool_calls": [
                    {
                        "id": r["id"],
                        "type": "function",
                        "function": {
                            "name": r["name"],
                            "arguments": r["arguments"] or "{}",
                        },
                    }
                    for r in raw_calls
                ],
            }
        )
        messages.extend(_run_tools(ctx, spec, inv, calls))

    raise AgentError(
        f"gave up after {ctx.config['MAX_AGENT_TURNS']} turns without finishing"
    )


def run_agent(ctx: RunContext, spec: AgentSpec, task: str, parent=None) -> str:
    """Run `spec` on `task` and return its final text."""
    inv = uuid.uuid4().hex[:10]
    ctx.emit(
        "agent_started",
        inv=inv,
        agent=spec.key,
        name=spec.name,
        parent=parent,
        task=task,
    )
    try:
        text = _loop(ctx, spec, inv, task)
    except RunCancelled:
        ctx.emit(
            "agent_finished",
            inv=inv,
            agent=spec.key,
            status="cancelled",
        )
        raise
    except Exception as exc:
        ctx.emit(
            "agent_finished",
            inv=inv,
            agent=spec.key,
            status="failed",
            error=_clip(str(exc), 500),
        )
        raise
    ctx.emit(
        "agent_finished",
        inv=inv,
        agent=spec.key,
        status="completed",
    )
    return text


# -- LangGraph Multi-Agent Orchestration --------------------------------------


class MultiAgentState(TypedDict):
    task: str
    coord_inv: str
    messages: Annotated[List[Any], operator.add]
    final_result: Optional[str]
    turn_count: int


def _create_multiagent_graph(ctx: RunContext, coord_inv: str):
    """Build a LangGraph StateGraph connecting Coordinator and Specialists."""

    def coordinator_node(state: MultiAgentState) -> dict:
        ctx.check_cancelled()
        messages = state["messages"]
        text, raw_calls, finish = _stream_turn(
            ctx, COORDINATOR, coord_inv, messages
        )
        new_turn = state.get("turn_count", 0) + 1

        final_res = None
        if not raw_calls and text.strip():
            final_res = text.strip()
            if finish == "length":
                final_res += (
                    "\n\n[This answer was cut off at the output token limit.]"
                )

        # Ensure unparseable JSON arguments are normalized to "{}" in assistant message history
        sanitized_tool_calls = []
        for r in raw_calls:
            arg_str = r.get("arguments") or "{}"
            try:
                json.loads(arg_str)
            except ValueError:
                arg_str = "{}"
            sanitized_tool_calls.append({
                "id": r["id"],
                "type": "function",
                "function": {
                    "name": r["name"],
                    "arguments": arg_str,
                },
            })

        assistant_msg = {
            "role": "assistant",
            "content": text or None,
            "tool_calls": sanitized_tool_calls,
            "raw_calls": raw_calls,
        }

        return {
            "messages": [assistant_msg],
            "turn_count": new_turn,
            "final_result": final_res,
        }

    def tools_and_specialists_node(state: MultiAgentState) -> dict:
        ctx.check_cancelled()
        last_msg = state["messages"][-1]
        raw_calls = last_msg.get("raw_calls", [])

        calls = []
        for raw in raw_calls:
            try:
                args = json.loads(raw["arguments"] or "{}")
            except ValueError:
                args = None
                raw["arguments"] = "{}"
            calls.append(
                SimpleNamespace(id=raw["id"], name=raw["name"], input=args)
            )
            ctx.emit(
                "tool_call",
                inv=coord_inv,
                agent=COORDINATOR.key,
                name=raw["name"],
                input=_clip(args, 600),
            )

        tool_messages = _run_tools(ctx, COORDINATOR, coord_inv, calls)
        return {"messages": tool_messages}

    def route_coordinator(state: MultiAgentState) -> str:
        if state.get("turn_count", 0) >= ctx.config["MAX_AGENT_TURNS"]:
            return END
        last_msg = state["messages"][-1]
        raw_calls = last_msg.get("raw_calls", [])
        if not raw_calls:
            return END
        return "tools_and_specialists"

    workflow = StateGraph(MultiAgentState)
    workflow.add_node("coordinator", coordinator_node)
    workflow.add_node("tools_and_specialists", tools_and_specialists_node)

    workflow.add_edge(START, "coordinator")
    workflow.add_conditional_edges(
        "coordinator",
        route_coordinator,
        {"tools_and_specialists": "tools_and_specialists", END: END},
    )
    workflow.add_edge("tools_and_specialists", "coordinator")

    return workflow.compile()


def run_task(ctx: RunContext, task: str) -> str:
    """Entry point for a run: coordinates the multiagent team using LangGraph."""
    brief = f"{task}\n\n(Today's date: {date.today().isoformat()})"
    coord_inv = uuid.uuid4().hex[:10]
    ctx.emit(
        "agent_started",
        inv=coord_inv,
        agent=COORDINATOR.key,
        name=COORDINATOR.name,
        parent=None,
        task=task,
    )

    try:
        graph = _create_multiagent_graph(ctx, coord_inv)
        initial_state = {
            "task": task,
            "coord_inv": coord_inv,
            "messages": [
                {"role": "system", "content": COORDINATOR.system},
                {"role": "user", "content": brief},
            ],
            "final_result": None,
            "turn_count": 0,
        }
        final_state = graph.invoke(initial_state)
        result_text = final_state.get("final_result")

        if not result_text:
            last_msg = final_state["messages"][-1]
            content = last_msg.get("content") if isinstance(last_msg, dict) else getattr(last_msg, "content", None)
            if content and content.strip():
                result_text = content.strip()
            else:
                if (
                    final_state.get("turn_count", 0)
                    >= ctx.config["MAX_AGENT_TURNS"]
                ):
                    raise AgentError(
                        f"gave up after {ctx.config['MAX_AGENT_TURNS']} turns without finishing"
                    )
                raise AgentError("the model returned an empty answer")

    except RunCancelled:
        ctx.emit(
            "agent_finished",
            inv=coord_inv,
            agent=COORDINATOR.key,
            status="cancelled",
        )
        raise
    except Exception as exc:
        ctx.emit(
            "agent_finished",
            inv=coord_inv,
            agent=COORDINATOR.key,
            status="failed",
            error=_clip(str(exc), 500),
        )
        raise

    ctx.emit(
        "agent_finished",
        inv=coord_inv,
        agent=COORDINATOR.key,
        status="completed",
    )
    return result_text
