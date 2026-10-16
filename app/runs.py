"""Background execution of runs: queueing, cancellation and failure reporting."""
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor

from .agents import AgentError, RunCancelled, RunContext, run_task

log = logging.getLogger(__name__)


class Busy(Exception):
    pass


def _describe(exc: Exception) -> str:
    if isinstance(exc, AgentError):
        return f"The coordinator could not finish: {exc}"
    err_str = str(exc)
    if "Connection refused" in err_str or "ConnectError" in err_str or "connection error" in err_str.lower():
        return "Could not connect to Ollama. Make sure Ollama is installed and running (`ollama serve`)."
    if "not found" in err_str.lower() and "model" in err_str.lower():
        return f"Model not found in Ollama: {exc}. Run 'ollama pull qwen2.5:3b' to install it."
    return f"Unexpected error: {exc}"


def _default_client(config):
    from langchain_ollama import ChatOllama
    model = config.get("AGENT_MODEL", "qwen2.5:3b")
    if "gwen" in model.lower():
        model = model.lower().replace("gwen", "qwen")
    base_url = config.get("OLLAMA_BASE_URL", "http://localhost:11434")
    temp = config.get("AGENT_TEMPERATURE", 0.1)
    return ChatOllama(model=model, base_url=base_url, temperature=temp)


class RunManager:
    """Executes runs on a bounded thread pool, one coordinator per run.

    State lives in this process, so serve the app from a single process
    (threads are fine).
    """

    def __init__(self, config, store, hub, client_factory=None):
        self.config = config
        self.store = store
        self.hub = hub
        self._client_factory = client_factory or (lambda: _default_client(config))
        self._client = None
        self._active = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(
            max_workers=config["MAX_CONCURRENT_RUNS"], thread_name_prefix="run"
        )
        # Runs left unfinished by a previous process can never complete.
        for run_id in store.unfinished_runs():
            self._finish(run_id, "failed", error="Interrupted by a server restart.")

    def _get_client(self):
        with self._lock:
            if self._client is None:
                self._client = self._client_factory()
            return self._client

    def submit(self, task: str) -> str:
        run_id = uuid.uuid4().hex[:12]
        cancel = threading.Event()
        with self._lock:
            if len(self._active) >= self.config["MAX_CONCURRENT_RUNS"]:
                raise Busy()
            self._active[run_id] = cancel
        self.store.create_run(run_id, task)
        self._pool.submit(self._execute, run_id, task, cancel)
        return run_id

    def cancel(self, run_id: str) -> bool:
        with self._lock:
            cancel = self._active.get(run_id)
        if cancel is None:
            return False
        cancel.set()
        return True

    def is_active(self, run_id: str) -> bool:
        with self._lock:
            return run_id in self._active

    def _finish(self, run_id, status, result=None, error=None, usage=None):
        usage = dict(usage or {})
        self.store.finish_run(run_id, status, result=result, error=error, usage=usage)
        self.hub.emit(run_id, "run_finished",
                      {"status": status, "result": result, "error": error, "usage": usage})

    def _execute(self, run_id: str, task: str, cancel: threading.Event) -> None:
        ctx = None
        outcome = {"status": "completed"}
        try:
            self.store.start_run(run_id)
            self.hub.emit(run_id, "run_started", {"task": task})
            ctx = RunContext(run_id, self._get_client(), self.config, self.store, self.hub, cancel)
            outcome["result"] = run_task(ctx, task)
        except RunCancelled:
            outcome = {"status": "cancelled", "error": "Cancelled."}
        except Exception as exc:  # the run must always reach a terminal state
            if not isinstance(exc, AgentError):
                log.exception("run %s failed", run_id)
            outcome = {"status": "failed", "error": _describe(exc)}
        # Free the slot first so a finished run is never still reported as active.
        with self._lock:
            self._active.pop(run_id, None)
        self._finish(run_id, usage=ctx.usage if ctx else None, **outcome)
        log.info("run %s %s", run_id, outcome["status"])
