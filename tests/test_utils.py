from pathlib import Path

import pytest
from langchain_core.messages import AIMessage

from app.core import settings
from app.utils.dates import calculate_age_from_birthdate
from app.utils.formatting import format_prompt
from app.utils.math_tools import safe_eval
from app.utils.validation import validate_agent_output, validate_user_input


def test_format_prompt_applies_standard_string_formatting():
    assert format_prompt("Hello {name}", name="Alice") == "Hello Alice"


def test_safe_eval_supports_allowed_arithmetic():
    assert safe_eval("2 + 3 * 4") == 14
    assert safe_eval("-5 + 2") == -3


def test_safe_eval_rejects_invalid_expressions():
    with pytest.raises(ValueError):
        safe_eval("__import__('os').system('echo hi')")


def test_calculate_age_from_birthdate_uses_calendar_aware_logic():
    assert calculate_age_from_birthdate("1995-04-12", current_date="2026-05-04") == 31
    assert calculate_age_from_birthdate("1995-12-12", current_date="2026-05-04") == 30


def test_validate_user_input_strips_and_validates_length():
    assert validate_user_input("  hello  ") == "hello"

    with pytest.raises(ValueError):
        validate_user_input("   ")

    with pytest.raises(ValueError):
        validate_user_input("x" * 2001)


def test_validate_agent_output_handles_none_empty_and_tool_calls():
    fallback = validate_agent_output(None)
    assert fallback.content == "Something went wrong."

    empty = validate_agent_output(AIMessage(content=""))
    assert empty.content == "I couldn't generate a proper response."

    tool_call_message = AIMessage(
        content="",
        tool_calls=[{"name": "get_current_time", "args": {}, "id": "call_1", "type": "tool_call"}],
    )
    assert validate_agent_output(tool_call_message) is tool_call_message


def test_settings_expose_expected_project_root_paths_and_names():
    expected_root = Path(__file__).resolve().parents[1]

    assert settings.PROJECT_ROOT == expected_root
    assert settings.DATA_DIR == expected_root / "data"
    assert settings.CHAT_MODEL_NAME == "openai/gpt-3.5-turbo"
    assert settings.EMBEDDING_MODEL_NAME == "BAAI/bge-large-en"
    assert Path(settings.CORE_KNOWLEDGE_DIR) == settings.DATA_DIR / "core_knowledge"
    assert Path(settings.USER_MEMORY_DIR) == settings.DATA_DIR / "user_memory"
    assert Path(settings.USER_INFO_PATH) == settings.DATA_DIR / "user_info.json"
