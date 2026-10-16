import json
import os

from flask import Blueprint, Response, current_app, jsonify, render_template, request

from .agents import AGENTS
from .db import TERMINAL
from .runs import Busy

bp = Blueprint("main", __name__)


def _store():
    return current_app.extensions["store"]


def _runs():
    return current_app.extensions["runs"]


def _error(status: int, message: str):
    return jsonify({"error": message}), status


@bp.get("/")
def index():
    return render_template("index.html")


@bp.get("/api/health")
def health():
    import urllib.request
    ollama_ok = False
    base_url = current_app.config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        req = urllib.request.Request(f"{base_url}/api/tags")
        with urllib.request.urlopen(req, timeout=1.5) as resp:
            ollama_ok = (resp.status == 200)
    except Exception:
        ollama_ok = False

    return jsonify({
        "status": "ok" if ollama_ok else "warning",
        "model": current_app.config["AGENT_MODEL"],
        "credentials_in_env": ollama_ok,
        "ollama_connected": ollama_ok,
    })


@bp.get("/api/agents")
def list_agents():
    return jsonify([
        {
            "key": a.key,
            "name": a.name,
            "summary": a.summary,
            "tools": list(a.capabilities),
        }
        for a in AGENTS.values()
    ])


@bp.post("/api/runs")
def create_run():
    body = request.get_json(silent=True)
    task = body.get("task") if isinstance(body, dict) else None
    if not isinstance(task, str) or not task.strip():
        return _error(400, "Provide a non-empty 'task' string.")
    task = task.strip()
    limit = current_app.config["MAX_TASK_CHARS"]
    if len(task) > limit:
        return _error(413, f"Task is {len(task)} characters; the limit is {limit}.")
    try:
        run_id = _runs().submit(task)
    except Busy:
        return _error(429, "Too many tasks are running. Wait for one to finish and try again.")
    return jsonify(_store().get_run(run_id)), 202


@bp.get("/api/runs")
def list_runs():
    return jsonify(_store().list_runs())


@bp.get("/api/runs/<run_id>")
def get_run(run_id):
    run = _store().get_run(run_id)
    if run is None:
        return _error(404, "Run not found.")
    run["artifacts"] = _store().list_artifacts(run_id)
    return jsonify(run)


@bp.delete("/api/runs/<run_id>")
def delete_run(run_id):
    if _store().get_run(run_id) is None:
        return _error(404, "Run not found.")
    if _runs().is_active(run_id):
        return _error(409, "Cancel the run before deleting it.")
    _store().delete_run(run_id)
    return "", 204


@bp.post("/api/runs/<run_id>/cancel")
def cancel_run(run_id):
    if _store().get_run(run_id) is None:
        return _error(404, "Run not found.")
    if not _runs().cancel(run_id):
        return _error(409, "Run is not active.")
    return jsonify({"status": "cancelling"}), 202


@bp.get("/api/runs/<run_id>/artifacts/<name>")
def get_artifact(run_id, name):
    artifact = _store().get_artifact(run_id, name)
    if artifact is None:
        return _error(404, "Artifact not found.")
    return jsonify(artifact)


@bp.get("/api/runs/<run_id>/stream")
def stream_run(run_id):
    """Server-sent events: replays the run's log, then follows it live."""
    store, hub = _store(), current_app.extensions["hub"]
    if store.get_run(run_id) is None:
        return _error(404, "Run not found.")
    try:
        last = int(request.headers.get("Last-Event-ID") or request.args.get("after") or 0)
    except ValueError:
        return _error(400, "Invalid event id.")

    def generate():
        nonlocal last
        settled = False
        while True:
            version = hub.version
            events = store.events_after(run_id, last)
            for event in events:
                last = event["seq"]
                yield f"id: {last}\ndata: {json.dumps(event)}\n\n"
                if event["type"] == "run_finished":
                    return
            if events:
                continue
            run = store.get_run(run_id)
            if run is None:
                return
            if run["status"] in TERMINAL:
                # The status is written just before the closing run_finished
                # event, so give that event a moment to land before giving up.
                if settled:
                    return
                settled = True
                hub.wait(version, timeout=2)
            elif not hub.wait(version, timeout=15):
                yield ": keep-alive\n\n"

    return Response(
        generate(),
        mimetype="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
