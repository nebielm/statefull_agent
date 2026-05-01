import pytest
from langchain_core.messages import AIMessage

from app.core import settings
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


def test_settings_expose_expected_default_paths_and_names():
    assert settings.CHAT_MODEL_NAME == "openai/gpt-3.5-turbo"
    assert settings.EMBEDDING_MODEL_NAME == "BAAI/bge-large-en"
    assert settings.CORE_KNOWLEDGE_DIR == "data/core_knowledge"
    assert settings.USER_MEMORY_DIR == "data/user_memory"
    assert settings.USER_INFO_PATH == "data/user_info.json"
