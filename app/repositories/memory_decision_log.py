import json
from datetime import datetime
from pathlib import Path

from app.core.settings import MEMORY_DECISION_LOG_PATH


def append_memory_decision_log(
    user_id: str,
    result: dict,
    *,
    source: str = "structured_memory_storage",
    timestamp: str | None = None,
    log_path: str | None = None,
) -> dict:
    target_path = Path(log_path or MEMORY_DECISION_LOG_PATH)
    target_path.parent.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": timestamp or datetime.now().isoformat(),
        "user_id": user_id,
        "category": result.get("category"),
        "field": result.get("field"),
        "proposed_value": result.get("proposed_value"),
        "existing_value": result.get("existing_value"),
        "decision": result.get("decision"),
        "reason": result.get("reason"),
        "source": source,
    }

    with target_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry) + "\n")

    return entry
