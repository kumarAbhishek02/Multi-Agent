import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _int(name: str, default: int) -> int:
    return int(os.getenv(name, default))


def load_config() -> dict:
    """Read settings from the environment (after .env has been loaded)."""
    raw_model = os.getenv("AGENT_MODEL") or "qwen2.5:3b"
    # Normalize model name if user typed "gwen2.5:3b" or similar
    normalized_model = raw_model
    if "gwen" in raw_model.lower():
        normalized_model = raw_model.lower().replace("gwen", "qwen")

    return {
        "AGENT_MODEL": normalized_model,
        "OLLAMA_BASE_URL": os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
        "AGENT_TEMPERATURE": float(os.getenv("AGENT_TEMPERATURE", "0.1")),
        "AGENT_MAX_TOKENS": _int("AGENT_MAX_TOKENS", 4096),
        "DATABASE_PATH": os.getenv("DATABASE_PATH", str(BASE_DIR / "instance" / "multi_agent.db")),
        "MAX_CONCURRENT_RUNS": _int("MAX_CONCURRENT_RUNS", 3),
        "MAX_DELEGATIONS": _int("MAX_DELEGATIONS", 12),
        "MAX_AGENT_TURNS": _int("MAX_AGENT_TURNS", 16),
        "MAX_PARALLEL_TOOLS": _int("MAX_PARALLEL_TOOLS", 4),
        "MAX_TASK_CHARS": _int("MAX_TASK_CHARS", 20000),
        "MAX_ARTIFACT_CHARS": _int("MAX_ARTIFACT_CHARS", 200000),
        "MAX_CONTENT_LENGTH": 1024 * 1024,
    }
