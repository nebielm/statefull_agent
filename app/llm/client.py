import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openrouter import ChatOpenRouter

from app.core.logging import logger
from app.core.settings import CHAT_MODEL_NAME, EMBEDDING_MODEL_NAME, OPENROUTER_API_KEY


if not OPENROUTER_API_KEY:
    raise ValueError("Missing OPENROUTER_API_KEY or OPENAI_API_KEY in environment.")


llm = ChatOpenRouter(
    model=CHAT_MODEL_NAME,
    api_key=OPENROUTER_API_KEY,
    temperature=0,
)

embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL_NAME)


def call_llm_json(prompt: str, default) -> dict:
    response = llm.invoke(prompt)

    try:
        content = response.content.strip()

        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(content)

    except Exception:
        logger.warning("Invalid JSON, retrying once...")
        retry_prompt = f"Fix this JSON:\n{content}"
        retry = llm.invoke(retry_prompt)

        try:
            return json.loads(retry.content)
        except Exception:
            logger.error("Failed twice → returning default")
            return default
