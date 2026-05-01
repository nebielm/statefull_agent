from app.llm.prompts.memory import DATA_SELECTION_PROMPT as APP_MEMORY_PROMPT
from app.llm.prompts.retrieval import DATA_RELEVANCE_PROMPT as APP_RETRIEVAL_PROMPT
from app.llm.prompts.tools import TOOL_SELECTION_PROMPT as APP_TOOLS_PROMPT
from prompts.memory import DATA_SELECTION_PROMPT
from prompts.retrieval import DATA_RELEVANCE_PROMPT
from prompts.tools import TOOL_SELECTION_PROMPT


def test_memory_prompt_wrapper_reexports_application_prompt():
    assert DATA_SELECTION_PROMPT is APP_MEMORY_PROMPT
    assert "MEMORY SCHEMA" in DATA_SELECTION_PROMPT


def test_retrieval_prompt_wrapper_reexports_application_prompt():
    assert DATA_RELEVANCE_PROMPT is APP_RETRIEVAL_PROMPT
    assert "structured_to_retrieve" in DATA_RELEVANCE_PROMPT


def test_tools_prompt_wrapper_reexports_application_prompt():
    assert TOOL_SELECTION_PROMPT is APP_TOOLS_PROMPT
    assert "Return ONLY a JSON list of tool names." in TOOL_SELECTION_PROMPT
