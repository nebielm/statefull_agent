import json
import os
from typing import Any, Dict, List

from app.core.logging import logger
from app.core.settings import USER_INFO_PATH
from app.models.memory import IMMUTABLE_KEYS, MEMORY_SCHEMA


def load_user_data(file_path: str = USER_INFO_PATH) -> Dict[str, Any]:
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def lookup_user_value(user_data: Dict[str, Any], key: str):
    if key in user_data:
        return user_data[key]

    for category_values in user_data.values():
        if isinstance(category_values, dict) and key in category_values:
            return category_values[key]

    return None


def is_valid_key(key, category):
    if key in IMMUTABLE_KEYS:
        return False
    return key in MEMORY_SCHEMA.get(category, [])


def controlled_structured_data_storage(user_id: str, key: str, value: str, category: str):
    logger.info(
        f"[MEMORY STORAGE]: Started structured user data storage for: key: {key}, value: {value}, category: {category}."
    )

    if key in IMMUTABLE_KEYS:
        logger.info(
            f"[MEMORY STORAGE]: IGNORED: Key: {key} is immutable and cannot be stored by agent."
        )
        return f"{key} is immutable and cannot be stored by agent."

    if not is_valid_key(key, category):
        logger.info(f"[MEMORY STORAGE]: IGNORED: Key: {key} is not allowed in {category}.")
        return f"{key} is not allowed in {category}."

    try:
        os.makedirs(os.path.dirname(USER_INFO_PATH), exist_ok=True)
        data = load_user_data(USER_INFO_PATH)

        if user_id not in data:
            data[user_id] = {}
        if category not in data[user_id]:
            data[user_id][category] = {}

        user_data = data[user_id]
        if key not in user_data[category] or user_data[category][key] != value:
            user_data[category][key] = value
            with open(USER_INFO_PATH, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(f"[MEMORY STORAGE]: ✅ Structured User data storage success: Stored {key}: {value}")
            return f"Stored {key}: {value}"
        logger.info(f"[MEMORY STORAGE]: No change for {key}")
        return f"No change for {key}"
    except Exception as e:
        logger.error(f"[MEMORY STORAGE]: ❌ Error while storing structured User data: {str(e)}")
        return


def retrieve_structured_memory(
    user_id: str,
    relevant_categories: List[str] = None,
    relevant_keys: List[str] = None,
    k: int = 5,
) -> List[Dict[str, Any]]:
    logger.info("[RETRIEVAL SYSTEM]: Retrieving structured user data from user DB")
    data = load_user_data(USER_INFO_PATH)
    if not data:
        logger.error("[RETRIEVAL SYSTEM]: ❌ Error while opening user DB file.")
        return []
    try:
        user_data = data.get(user_id, {})
        results = []

        for category, items in user_data.items():
            if relevant_categories and category not in relevant_categories:
                continue

            for key, value in items.items():
                if relevant_keys and key not in relevant_keys:
                    continue

                score = 1.0 if relevant_categories and category in relevant_categories else 0.7

                results.append(
                    {
                        "key": key,
                        "value": value,
                        "category": category,
                        "score": score,
                    }
                )
        logger.info("[RETRIEVAL SYSTEM]: ✅ Successfully retrieved structured user data.")
        return results[:k]
    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving structured user data: {str(e)}")
        return []
