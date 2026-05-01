import os

from dotenv import find_dotenv, load_dotenv


load_dotenv(find_dotenv())

OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY")
CHAT_MODEL_NAME = "openai/gpt-3.5-turbo"
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en"

CORE_KNOWLEDGE_DIR = "data/core_knowledge"
USER_MEMORY_DIR = "data/user_memory"
USER_INFO_PATH = "data/user_info.json"

CORE_KNOWLEDGE_COLLECTION = "core_knowledge"
USER_MEMORY_COLLECTION = "user_memory"
