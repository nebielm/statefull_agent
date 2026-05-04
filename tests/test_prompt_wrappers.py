from app.llm.prompts.memory import DATA_SELECTION_PROMPT as APP_MEMORY_PROMPT
from app.llm.prompts.retrieval import DATA_RELEVANCE_PROMPT as APP_RETRIEVAL_PROMPT
from app.llm.prompts.tools import TOOL_SELECTION_PROMPT as APP_TOOLS_PROMPT


def test_memory_prompt_contains_expected_schema_instructions():
    assert "MEMORY SCHEMA" in APP_MEMORY_PROMPT


def test_retrieval_prompt_contains_expected_output_contract():
    assert "structured_to_retrieve" in APP_RETRIEVAL_PROMPT


def test_tools_prompt_contains_expected_output_contract():
    assert "Return ONLY a JSON list of tool names." in APP_TOOLS_PROMPT
