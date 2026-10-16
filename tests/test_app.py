"""End-to-end tests against a scripted fake of the Groq streaming client."""
import json
import threading
import time
from types import SimpleNamespace as NS

from app import create_app

USAGE = NS(prompt_tokens=100, completion_tokens=20)


def chunk(content=None, tool_calls=None, finish=None, usage=None):
    delta = NS(content=content, tool_calls=tool_calls, reasoning=None)
    return NS(choices=[NS(delta=delta, finish_reason=finish)],
              x_groq=NS(usage=usage) if usage else None)


def say(text, finish="stop"):
    return [chunk(content=text), chunk(finish=finish, usage=USAGE)]


def call(*calls):
    """Tool calls, with each one's JSON arguments split across two chunks as on the wire."""
    chunks = []
    for index, (name, args) in enumerate(calls):
        raw = args if isinstance(args, str) else json.dumps(args)
        half = len(raw) // 2
        chunks.append(chunk(tool_calls=[NS(index=index, id=f"call_{index}",
                                           function=NS(name=name, arguments=raw[:half]))]))
        chunks.append(chunk(tool_calls=[NS(index=index, id=None,
                                           function=NS(name=None, arguments=raw[half:]))]))
    return [*chunks, chunk(finish="tool_calls", usage=USAGE)]


class FakeStream:
    def __init__(self, chunks, gate=None):
        self.chunks, self.gate = chunks, gate

    def __iter__(self):
        for item in self.chunks:
            if self.gate:
                self.gate.wait(5)
            yield item

    def close(self):
        pass


class FakeGroq:
    """Answers from `script(system, messages) -> chunks`; records every request."""

    def __init__(self, script, gate=None):
        self.requests = []
        self.chat = NS(completions=NS(create=self._create))
        self._script, self._gate = script, gate

    def _create(self, **request):
        self.requests.append(request)
        system, *messages = request["messages"]
        return FakeStream(self._script(system["content"], messages), self._gate)


def make_client(tmp_path, script, gate=None, **overrides):
    fake = FakeGroq(script, gate)
    app = create_app(
        {"DATABASE_PATH": str(tmp_path / "test.db"), "TESTING": True, **overrides},
        client_factory=lambda: fake,
    )
    return app.test_client(), fake


def wait_for(client, run_id, timeout=10):
    deadline = time.time() + timeout
    while time.time() < deadline:
        run = client.get(f"/api/runs/{run_id}").get_json()
        if run["status"] in ("completed", "failed", "cancelled"):
            return run
        time.sleep(0.02)
    raise AssertionError("run did not finish")


def start(client, task="t"):
    return client.post("/api/runs", json={"task": task}).get_json()["id"]


def is_coordinator(system):
    return "You are the Coordinator" in system


def tool_messages(request):
    return [m for m in request["messages"] if m["role"] == "tool"]


def delegating_script(system, messages):
    if is_coordinator(system):
        if len(messages) == 1:
            return call(("delegate", {"agent": "writer", "task": "Draft a haiku."}),
                        ("delegate", {"agent": "reviewer", "task": "Check the plan."}))
        return say("Final: all done.")
    if "You are the Writer" in system:
        if len(messages) == 1:
            return call(("save_artifact", {"name": "haiku.md", "content": "five seven five"}))
        return say("Saved the haiku as haiku.md.")
    return say("Looks fine.")


def test_coordinator_delegates_and_synthesizes(tmp_path):
    client, fake = make_client(tmp_path, delegating_script)
    response = client.post("/api/runs", json={"task": "Write a haiku"})
    assert response.status_code == 202
    run = wait_for(client, response.get_json()["id"])

    assert run["status"] == "completed"
    assert run["result"] == "Final: all done."
    assert [a["name"] for a in run["artifacts"]] == ["haiku.md"]
    # coordinator x2, writer x2, reviewer x1
    assert run["usage"] == {"input_tokens": 500, "output_tokens": 100}

    # Both specialist reports came back to the coordinator, matched to their calls.
    final_request = [r for r in fake.requests if is_coordinator(r["messages"][0]["content"])][-1]
    assert tool_messages(final_request) == [
        {"role": "tool", "tool_call_id": "call_0", "content": "Saved the haiku as haiku.md."},
        {"role": "tool", "tool_call_id": "call_1", "content": "Looks fine."},
    ]
    assistant = final_request["messages"][2]
    assert json.loads(assistant["tool_calls"][0]["function"]["arguments"])["agent"] == "writer"

    artifact = client.get(f"/api/runs/{run['id']}/artifacts/haiku.md").get_json()
    assert artifact["content"] == "five seven five" and artifact["agent"] == "writer"


def test_stream_replays_events_in_order(tmp_path):
    client, _ = make_client(tmp_path, delegating_script)
    run_id = start(client, "Write a haiku")
    wait_for(client, run_id)

    body = client.get(f"/api/runs/{run_id}/stream").get_data(as_text=True)
    events = [json.loads(line[6:]) for line in body.splitlines() if line.startswith("data: ")]
    types = [e["type"] for e in events]
    assert types[0] == "run_started" and types[-1] == "run_finished"
    assert types.count("agent_started") == 3 and types.count("agent_finished") == 3
    assert "artifact_saved" in types and "text_delta" in types
    coordinator = next(e for e in events if e["type"] == "agent_started")
    children = [e for e in events if e["type"] == "agent_started" and e["parent"]]
    assert all(c["parent"] == coordinator["inv"] for c in children)

    # Resuming from the last id yields nothing new.
    resumed = client.get(f"/api/runs/{run_id}/stream", headers={"Last-Event-ID": str(events[-1]["seq"])})
    assert "data:" not in resumed.get_data(as_text=True)


def test_bad_tool_input_is_reported_to_the_agent(tmp_path):
    def script(system, messages):
        if len(messages) == 1:
            return call(("delegate", {"agent": "nobody", "task": "x"}),
                        ("read_artifact", {"name": 7}),
                        ("read_artifact", '{"broken'))
        return say("Recovered.")

    client, fake = make_client(tmp_path, script)
    run = wait_for(client, start(client))
    assert run["status"] == "completed" and run["result"] == "Recovered."
    results = [m["content"] for m in tool_messages(fake.requests[-1])]
    assert "Unknown agent" in results[0]
    assert "INVALID_JSON" in results[1] and "INVALID_JSON" in results[2]
    echoed = fake.requests[-1]["messages"][2]["tool_calls"][2]["function"]["arguments"]
    assert echoed == "{}"


def test_delegation_budget_is_enforced(tmp_path):
    def script(system, messages):
        if is_coordinator(system):
            if len(messages) == 1:
                return call(*[("delegate", {"agent": "writer", "task": "x"})] * 3)
            return say("Done.")
        return say("ok")

    client, fake = make_client(tmp_path, script, MAX_DELEGATIONS=2)
    run = wait_for(client, start(client))
    results = [m["content"] for m in tool_messages(fake.requests[-1])]
    assert run["status"] == "completed"
    assert sum("delegation budget" in r for r in results) == 1


def test_empty_answer_truncation_and_turn_limit(tmp_path):
    client, _ = make_client(tmp_path / "a", lambda s, m: say(""))
    run = wait_for(client, start(client))
    assert run["status"] == "failed" and "empty answer" in run["error"]

    client, _ = make_client(tmp_path / "b", lambda s, m: say("Partial", finish="length"))
    run = wait_for(client, start(client))
    assert run["status"] == "completed" and "cut off" in run["result"]

    client, fake = make_client(tmp_path / "c", lambda s, m: call(("list_artifacts", "")),
                               MAX_AGENT_TURNS=3)
    run = wait_for(client, start(client))
    assert run["status"] == "failed" and "3 turns" in run["error"]
    assert len(fake.requests) == 3


def test_cancel_and_concurrency_limit(tmp_path):
    gate = threading.Event()
    client, _ = make_client(tmp_path, lambda s, m: say("hi"), gate=gate, MAX_CONCURRENT_RUNS=1)
    run_id = start(client)
    assert client.post("/api/runs", json={"task": "another"}).status_code == 429
    assert client.delete(f"/api/runs/{run_id}").status_code == 409

    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 202
    gate.set()
    assert wait_for(client, run_id)["status"] == "cancelled"
    assert client.post(f"/api/runs/{run_id}/cancel").status_code == 409
    assert client.delete(f"/api/runs/{run_id}").status_code == 204
    assert client.get(f"/api/runs/{run_id}").status_code == 404


def test_request_shape_and_validation(tmp_path):
    def script(system, messages):
        if is_coordinator(system) and len(messages) == 1:
            return call(("delegate", {"agent": "researcher", "task": "Look it up."}))
        return say("hi")

    client, fake = make_client(tmp_path, script, MAX_TASK_CHARS=10,
                               AGENT_MODEL="openai/gpt-oss-120b")
    assert client.post("/api/runs", json={"task": "   "}).status_code == 400
    assert client.post("/api/runs", data="nope").status_code == 400
    assert client.post("/api/runs", json={"task": "x" * 11}).status_code == 413
    assert client.get("/api/runs/missing").status_code == 404
    assert client.get("/").status_code == 200
    assert len(client.get("/api/agents").get_json()) == 6

    wait_for(client, start(client, "hello"))
    coordinator, researcher = fake.requests[0], fake.requests[1]
    assert coordinator["model"] == "openai/gpt-oss-120b" and coordinator["stream"] is True
    assert [t["function"]["name"] for t in coordinator["tools"]] == [
        "delegate", "read_artifact", "list_artifacts"]
    assert researcher["tools"][0]["function"]["name"] == "save_artifact"
    assert researcher["tools"][-1]["function"]["name"] == "web_search"
    assert researcher["messages"][1] == {"role": "user", "content": "Look it up."}


def test_hosted_tools_are_only_sent_to_models_that_have_them(tmp_path):
    def script(system, messages):
        if is_coordinator(system) and len(messages) == 1:
            return call(("delegate", {"agent": "analyst", "task": "Add."}))
        return say("hi")

    client, fake = make_client(tmp_path, script, AGENT_MODEL="llama-3.3-70b-versatile")
    wait_for(client, start(client))
    names = [t["function"]["name"] for t in fake.requests[1]["tools"]]
    assert names == ["save_artifact", "read_artifact", "list_artifacts"]


def test_interrupted_runs_are_failed_on_restart(tmp_path):
    client, _ = make_client(tmp_path, lambda s, m: say("hi"))
    client.application.extensions["store"].create_run("stale", "old task")
    client, _ = make_client(tmp_path, lambda s, m: say("hi"))
    run = client.get("/api/runs/stale").get_json()
    assert run["status"] == "failed" and "restart" in run["error"]
