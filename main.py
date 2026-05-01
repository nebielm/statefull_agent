import os
import ast
import operator as op
import time
import logging
import sys
import shutil
import json
from typing import Dict, Any
from prompts.memory import DATA_SELECTION_PROMPT
from prompts.retrieval import DATA_RELEVANCE_PROMPT
from prompts.tools import TOOL_SELECTION_PROMPT
import uuid
import requests
from datetime import datetime
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, List, Union, Annotated, Sequence, Literal, Dict, Any
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langchain_core.tools import tool, Tool
from langchain_openai import OpenAI
from langchain_openrouter import ChatOpenRouter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import TextLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

logger = logging.getLogger("mica")
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
formatter = logging.Formatter(
    '{"time":"%(asctime)s","level":"%(levelname)s","message":"%(message)s"}'
)
handler.setFormatter(formatter)

if not logger.handlers:
    logger.addHandler(handler)

load_dotenv(find_dotenv())


############################
# 0. HELPER FUNCTIONS
############################
def format_prompt(template: str, **kwargs):
    return template.format(**kwargs)

def call_llm_json(prompt: str, default) -> dict:
    response = llm.invoke(prompt)

    try:
        content = response.content.strip()

        # hard safety cleanup
        if "```" in content:
            content = content.replace("```json", "").replace("```", "").strip()

        return json.loads(content)

    except Exception:
        logger.warning("Invalid JSON, retrying once...")
        # retry once
        retry_prompt = f"Fix this JSON:\n{content}"
        retry = llm.invoke(retry_prompt)

        try:
            return json.loads(retry.content)
        except:
            logger.error("Failed twice → returning default")
            return default


# ✅ Define allowed schema (you can expand this over time)
ALLOWED_KEYS = {
    "weight",
    "target_weight",
    "location",
    "goal",
    "balance",
    "calories",
    "height",
    "age"
}

def extract_ephemeral_updates(
    user_text: str,
    agent_text: str
) -> Dict[str, Any]:
    """
    Extracts CURRENT state updates from user + agent messages.

    Returns:
        dict of updated state values (safe, filtered, validated)
    """

    # 🔧 1. Combine signals (VERY important)
    combined_text = f"""
    USER MESSAGE:
    {user_text}

    AGENT RESPONSE:
    {agent_text}
    """

    # 🔧 2. Strong extraction prompt
    prompt = f"""
    You are a state extraction engine.

    Extract ONLY the CURRENT user state from the conversation below.

    RULES:
    - Only extract FINAL / CURRENT values (not past values)
    - Do NOT infer or guess
    - Do NOT include historical values
    - Keep keys short and generic (e.g. weight, location, balance)
    - Output MUST be valid JSON (no text, no explanation)

    Example:
    Input:
    "User: I lost 10kg. Assistant: Your new weight is 120kg"
    Output:
    {{"weight": 120}}

    Conversation:
    {combined_text}
    """

    # 🔧 3. Call LLM
    try:
        response = llm.invoke(prompt)
        raw_output = response.content.strip()
    except Exception:
        return {}

    # 🔧 4. Safe JSON parsing
    try:
        parsed = json.loads(raw_output)
    except Exception:
        return {}

    # 🔧 5. Validate structure
    if not isinstance(parsed, dict):
        return {}

    # 🔧 6. Filter allowed keys (ANTI-HALLUCINATION)
    filtered = {}
    for key, value in parsed.items():
        if key in ALLOWED_KEYS:
            filtered[key] = value

    return filtered


# allowed operators
ALLOWED_OPERATORS = {
    ast.Add: op.add,
    ast.Sub: op.sub,
    ast.Mult: op.mul,
    ast.Div: op.truediv,
    ast.Pow: op.pow,
    ast.USub: op.neg
}


def safe_eval(expr: str):
    def _eval(node):
        if isinstance(node, ast.Num):  # numbers
            return node.n

        elif isinstance(node, ast.BinOp):  # binary ops
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return ALLOWED_OPERATORS[type(node.op)](
                _eval(node.left),
                _eval(node.right)
            )

        elif isinstance(node, ast.UnaryOp):  # -1
            if type(node.op) not in ALLOWED_OPERATORS:
                raise ValueError("Operator not allowed")
            return ALLOWED_OPERATORS[type(node.op)](
                _eval(node.operand)
            )

        else:
            raise ValueError("Invalid expression")

    node = ast.parse(expr, mode='eval').body
    return _eval(node)

def validate_user_input(text: str) -> str:
    text = text.strip()
    if not text:
        raise ValueError("Empty input")
    if len(text) > 2000:
        raise ValueError("Input too long")
    return text

def validate_agent_output(response: AIMessage) -> AIMessage:
    if response is None:
        return AIMessage(content="Something went wrong.")
    if not response.content or not response.content.strip():
        return AIMessage(content="I couldn't generate a proper response.")

    return response

############################
# 1. CONFIG + LLM SETUP
############################

API_KEY = os.environ["OPENAI_API_KEY"]

llm = OpenAI(
    model="openai/gpt-3.5-turbo",
    openrouter_api_key=API_KEY,
    temperature=0
)

embeddings = HuggingFaceEmbeddings(model_name="BAAI/bge-large-en")

# Cache for tool combinations
bound_models_cache = {}  # key: tuple of tool names, value: bound LLM

############################
# 2. STATE (AgentState)
############################

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_memory: Dict[str, Any]
    user_id: uuid.UUID
    request_id: uuid.UUID
    context: Dict[
        Literal["structured", "unstructured", "knowledge"],
        Any
    ]


############################
# 3. VECTOR DATABASE
############################

def ingest_knowledge(
        persist_dir: str,  # knowledge db="data/core_knowledge", user db= "data/user_memory"
        collection_name: str,  # knowledge db="core_knowledge", user db= "user_memory"
        doc_folder: str | None = None,  # knowledge db="knowledge_base/", user db= None
        force_rebuild: bool | None = False,  # knowledge db=, , user db= ...
):
    logger.info(f"[System]: Start {collection_name} Knowledge base creation")
    try:
        # 1️⃣ Rebuild if needed
        if force_rebuild and os.path.exists(persist_dir):
            shutil.rmtree(persist_dir)

        # 2️⃣ Always initialize empty DB first
        vectorstore = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_dir
        )

        # 3️⃣ OPTIONAL: ingest documents if they exist
        if doc_folder and os.path.exists(doc_folder) and collection_name == "core_knowledge":
            loader = DirectoryLoader(doc_folder, glob="**/*.txt")
            documents = loader.load()

            if documents:
                text_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500,
                    chunk_overlap=50
                )
                chunks = text_splitter.split_documents(documents)

                # Assign category metadata
                for c in chunks:
                    source = c.metadata.get("source", "").lower()
                    if "recipes" in source:
                        category = "recipe"
                    elif "nutrition" in source:
                        category = "nutrition"
                    elif "safety" in source:
                        category = "safety"
                    else:
                        category = "general"
                    c.metadata = {
                        "category": category,
                        "tags": [],
                        "timestamp": datetime.now().isoformat(),
                        "source": "core"
                    }

                vectorstore.add_documents(chunks)

        logger.info(f"[System]:✅ {collection_name} Knowledge base created successfully")
        return vectorstore
    except Exception as e:
        logger.error(f"[System]: ❌ Error while creating {collection_name} Knowledge base:{str(e)}")

############################
# 4. MEMORY (STRUCTURED + UNSTRUCTURED)
############################


# 4.1 Memory extraction (LLM → memory decisions)

def extract_memory_updates(text: str) -> dict:
    logger.info(f"[MEMORY EXTRACTION]: Start extracting updates on user data from query: {text}")
    try:
        prompt = format_prompt(
            DATA_SELECTION_PROMPT,
            text=text,
            memory_schema=MEMORY_SCHEMA,
            immutable_keys=IMMUTABLE_KEYS,
            allowed_types=ALLOWED_TYPES
        )

        output = call_llm_json(prompt, {
                "structured": [],
                "unstructured": []
            })

        # ✅ deterministic validation
        if not isinstance(output, dict):
            return {"structured": [], "unstructured": []}

        structured = output.get("structured", [])
        unstructured = output.get("unstructured", [])

        if not isinstance(structured, list):
            structured = []

        if not isinstance(unstructured, list):
            unstructured = []
        logger.info(f"[MEMORY EXTRACTION]:✅Extracted updates on user data successfully: structured: {str(structured)}, unstructured: {str(unstructured)}")
        return {
            "structured": structured,
            "unstructured": unstructured
        }

    except Exception as e:
        logger.error(f"[MEMORY EXTRACTION]: ❌ Error while extracting updates on user data: {str(e)}")
        return {"structured": [], "unstructured": []}


def extract_knowledge(raw_text: str) -> dict:
    """
    Extract structured knowledge for storage.
    Returns: summary, category, tags
    """
    logger.info(f"[MEMORY EXTRACTION]: Start extracting knowledge updates from findings: {raw_text}")

    if not raw_text or len(raw_text) < 20:
        logger.info(f"[MEMORY EXTRACTION]: IGNORED: Text must be at least 20 characters long.")
        return {
            "summary": raw_text,
            "category": "general",
            "tags": []
        }

    prompt = f"""
    Extract structured knowledge from the text.

    Return a JSON object with the following fields:
    - summary: a concise factual summary of the text, between 100 and 250 words, capturing all important points.
    - category: one word representing the main topic (e.g., tech, health, nutrition, fitness, general).
    - tags: a list of 3-6 keywords representing the key concepts or entities.

    Rules:
    - Be consistent in style and formatting.
    - Do NOT include explanations, commentary, or extra text.
    - Output ONLY valid JSON.
    - Focus on factual and verifiable information.
    - If the text is very short (<100 words), summarize it fully but keep it concise.

    Text:
    {raw_text}
    """
    try:
        data = call_llm_json(prompt, default={
            "summary": raw_text[:200],
            "category": "general",
            "tags": []
        })

        if not isinstance(data, dict):
            logger.error("[MEMORY EXTRACTION]: Invalid JSON format")
            return {
                "summary": raw_text[:200],
                "category": "general",
                "tags": []
            }
        result = {
            "summary": data.get("summary", raw_text[:200]),
            "category": data.get("category", "general"),
            "tags": data.get("tags", [])
        }
        logger.info("[MEMORY EXTRACTION]:✅Extracted updates on knowledge base successfully.")
        return result
    except Exception as e:
        logger.error(f"[MEMORY EXTRACTION]: ❌ Error while extracting updates on user data: {str(e)}")
        return {
            "summary": raw_text[:200],
            "category": "general",
            "tags": []
        }


def extract_retrieval_plan(text: str) -> dict:
    logger.info(f"[MEMORY EXTRACTION]: Start extracting retrieving plan for user data and knowledge base data for query: {text}")
    try:
        prompt = format_prompt(
            DATA_RELEVANCE_PROMPT,
            user_input=text,
            memory_schema=MEMORY_SCHEMA,
            allowed_types=ALLOWED_TYPES
        )

        output = call_llm_json(prompt, {
            "structured_to_retrieve": [],
            "unstructured_to_retrieve": []
        })

        if not isinstance(output, dict):
            return {
                "structured_to_retrieve": [],
                "unstructured_to_retrieve": []
            }
        logger.info(
            "[MEMORY EXTRACTION]: ✅ Extracted retrieving plan for user data and knowledge base data.")
        return {
            "structured_to_retrieve": output.get("structured_to_retrieve", []),
            "unstructured_to_retrieve": output.get("unstructured_to_retrieve", [])
        }
    except Exception as e:
        logger.error(f"[MEMORY EXTRACTION]: ❌ Error while extracting retrieving plan for user data and knowledge base data:{str(e)}")
        return {
            "structured_to_retrieve": [],
            "unstructured_to_retrieve": []
        }


# 4.2 Structured memory storage

IMMUTABLE_KEYS = ["name", "birthdate", "country of origin", "skin type"]

MEMORY_SCHEMA = {
    "profile": ["city", "job", "education"],
    "preferences": ["favorite_food", "hobbies", "diet"],
    "health": ["allergies"],
    "household": ["household_size"],
    "kitchen": ["appliances"],
    "dynamic": ["current_goal", "mood", "weight"]
}

ALLOWED_TYPES = ["habit", "preference", "diet", "dislike", "behavior", "context"]

def is_valid_key(key, category):
    if key in IMMUTABLE_KEYS:
        return False
    return key in MEMORY_SCHEMA.get(category, [])

def controlled_structured_data_storage(user_id: str, key: str, value: str, category: str):
    logger.info(f"[MEMORY STORAGE]: Started structured user data storage for: key: {key}, value: {value}, category: {category}.")

    if key in IMMUTABLE_KEYS:
        logger.info(
            f"[MEMORY STORAGE]: IGNORED: Key: {key} is immutable and cannot be stored by agent.")
        return f"{key} is immutable and cannot be stored by agent."

    if not is_valid_key(key, category):
        logger.info(
            f"[MEMORY STORAGE]: IGNORED: Key: {key} is not allowed in {category}.")
        return f"{key} is not allowed in {category}."

    try:
        # load file
        file_path = "data/user_info.json"
        # ✅ Create folder if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        try:
            with open(file_path, "r") as f:
                content = f.read().strip()
                data = json.loads(content) if content else {}
        except (FileNotFoundError, json.JSONDecodeError):
            data = {}

        # ✅ Update logic
        if user_id not in data:
            data[user_id] = {}
        if category not in data[user_id]:
            data[user_id][category] = {}

        user_data = data[user_id]
        if key not in user_data[category] or user_data[category][key] != value:
            user_data[category][key] = value
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2)
            logger.info(
                f"[MEMORY STORAGE]: ✅ Structured User data storage success: Stored {key}: {value}")
            return f"Stored {key}: {value}"
        logger.info(
            f"[MEMORY STORAGE]: No change for {key}")
        return f"No change for {key}"
    except Exception as e:
        logger.error(
            f"[MEMORY STORAGE]: ❌ Error while storing structured User data: {str(e)}")
        return



# 4.3 Unstructured memory storage

def normalize_type(t: str) -> str:
    t = t.lower().strip()
    mapping = {
        "likes": "preference",
        "like": "preference",
        "dislikes": "dislike",
        "hate": "dislike",
        "routine": "habit"
    }
    if t in mapping:
        return mapping[t]
    if t not in ALLOWED_TYPES:
        return "context"
    return t

def similar_memory_exists(
        vectorstore,
        text: str,
        threshold: float = 0.85,
        k: int = 3,
        metadata_filter: dict | None = None
):
    """
    Checks if a similar entry already exists in vector DB.
    Works with any metadata schema.
    """
    results = vectorstore.similarity_search_with_score(
        query=text,
        k=k,
        filter=metadata_filter if metadata_filter else None
    )

    for doc, score in results:
        # ⚠️ Chroma returns distance → smaller = more similar
        similarity = max(0, 1 - score)  # convert to similarity

        if similarity >= threshold:
            return {
                "exists": True,
                "similarity": similarity,
                "doc": doc.page_content
            }

    return {
        "exists": False,
    }

def controlled_unstructured_data_storage(user_vectorstore, text: str, type: str, user_id: str = "default_user"):
    """Stores unstructured memory into Chroma vector DB"""
    logger.info(
        f"[MEMORY STORAGE]: Started unstructured user data storage for: text: {text}, type: {type}.")

    if not text or len(text) < 5:
        logger.info("[MEMORY STORAGE]: IGNORED: Text must be at least 5 characters long.")
        return "ignored: too short"

    # 🔒 Deduplication check
    similarity = similar_memory_exists(user_vectorstore, text, metadata_filter={"user_id": user_id})
    if similarity.get("exists"):
        logger.info("[MEMORY STORAGE]: IGNORED: Similar memory already exists.")
        return "ignored: similar memory already exists"

    try:

        clean_type = normalize_type(type)
        metadata = {
            "user_id": user_id,
            "type": clean_type,
            "timestamp": datetime.now().isoformat(),
        }

        user_vectorstore.add_texts(
            texts=[text],
            metadatas=[metadata],
            ids=[str(uuid.uuid4())]
        )
        logger.info(f"[MEMORY STORAGE]: ✅ Stored {text} successfully.")
        return f"stored {text}, (type: {clean_type}) "

    except Exception as e:
        logger.error(f"[MEMORY STORAGE]: ❌ Error while storing unstructured memory: {str(e)}")



# 4.4 Memory enrichment (knowledge DB writing)

def enrich_knowledge(vectorstore, source: str, raw_text: str, chunk_size: int, chunk_overlap: int):
    """
    Store important new knowledge for future use.
    Only store concise, useful, and reusable information.
    """
    logger.info(f"[MEMORY STORAGE]: Started enriching knowledge based on source: {source}")
    if not raw_text or len(raw_text) < 5:
        logger.info("[MEMORY STORAGE]: IGNORED: Text must be at least 5 characters long.")
        return "ignored: too short"

    try:

        # 1️⃣ Extract structured knowledge using LLM
        data = extract_knowledge(raw_text, llm)

        summary_text = data.get("summary", raw_text)
        category = data.get("category", "general").lower()
        tags = [t.strip() for t in data.get("tags", [])] if data.get("tags") else []

        # 2️⃣ Split summary into chunks if too long
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap
        )
        chunks = splitter.split_text(summary_text)

        stored_count = 0
        for chunk in chunks:
            # 3️⃣ Skip if similar chunk exists
            similarity = similar_memory_exists(
                vectorstore=vectorstore,
                text=chunk
            )
            if similarity.get("exists"):
                continue

            # 4️⃣ Prepare metadata
            metadata = {
                "category": category,
                "tags": tags,
                "timestamp": datetime.now().isoformat(),
                "source": source
            }

            # 5️⃣ Add chunk to vector DB
            vectorstore.add_texts(
                texts=[chunk],
                metadatas=[metadata],
                ids=[str(uuid.uuid4())]
            )
            stored_count += 1
        logger.info(f"[MEMORY STORAGE]: ✅ Enriching knowledge base successfully: Stored {str(stored_count)} new chunk(s) from source '{source}'")
        return f"Stored {stored_count} new chunk(s) from source '{source}'."

    except Exception as e:
        logger.error(f"[MEMORY STORAGE]: ❌ Error while Enriching knowledge base: {str(e)}")


############################
# 5. RETRIEVAL SYSTEM
############################

# 5.1 Structured retrieval

def retrieve_structured_memory(user_id: str, relevant_categories: List[str] = None, relevant_keys: List[str] = None,
                               k: int = 5) -> List[Dict[str, Any]]:
    """
    Retrieve the most relevant structured memories for a user.

    Args:
        user_id: ID of the user.
        relevant_keys: Optional list of keys to filter by (from LLM relevance output).
        relevant_categories: Optional list of categories to filter by (from LLM relevance output).
        k: Number of results to return.

    Returns:
        List of dicts with keys: key, value, category, score.
    """
    logger.info("[RETRIEVAL SYSTEM]: Retrieving structured user data from user DB")
    file_path = "data/user_info.json"
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        logger.error("[RETRIEVAL SYSTEM]: ❌ Error while opening user DB file.")
        return []
    try:
        user_data = data.get(user_id, {})
        results = []

        # Flatten user data into list of key-value-category
        for category, items in user_data.items():

            # 🔒 Filter by category
            if relevant_categories and category not in relevant_categories:
                continue

            for key, value in items.items():

                # 🔒 Filter by key
                if relevant_keys and key not in relevant_keys:
                    continue

                # 🧠 Light priority (NOT real ranking)
                score = 1.0 if relevant_categories and category in relevant_categories else 0.7

                results.append({
                    "key": key,
                    "value": value,
                    "category": category,
                    "score": score
                })
        logger.info(f"[RETRIEVAL SYSTEM]: ✅ Successfully retrieved structured user data.")
        return results[:k]
    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving structured user data: {str(e)}")


# 5.2 Unstructured retrieval

#### is it in this case good to give retrieval system user message or should llm also create a message for retrieval ??
def retrieve_unstructured_memory(user_vectorstore, message: str, user_id: str, relevant_types: List[str], k: int = 5) -> \
List[Dict]:
    """
    Retrieve top-k relevant unstructured memories for a user, combining the user message
    with LLM-identified relevant keys/types.

    Args:
        message: The user's current message.
        user_id: The ID of the user.
        relevant_types: List of relevant keywords, keys, or types extracted by LLM.
        k: Number of results to return.

    Returns:
        A list of dicts with 'text', 'type', and 'score' for the top-k memories.
    """
    logger.info(f"[RETRIEVAL SYSTEM]: Start retrieving unstructured user data.")

    # Combine the user message with LLM findings to create a richer query
    search_query = " ".join([message] + relevant_types)
    try:
        results = user_vectorstore.similarity_search_with_score(
            query=search_query,
            k=k * 2,
            filter={"user_id": user_id}
        )
        scored = []

        for doc, distance in results:
            similarity = 1 - distance  # convert distance → similarity

            timestamp = doc.metadata.get("timestamp")

            # default old date if missing
            if timestamp:
                dt = datetime.fromisoformat(timestamp)
                age_days = (datetime.now() - dt).days
            else:
                age_days = 999

            # 🧠 score = relevance + recency
            recency_score = max(0, 1 - age_days / 30)  # decay over 30 days

            # combine similarity + recency
            combined_score = similarity * 0.7 + recency_score * 0.3  # weighting can be tuned

            scored.append({
                "text": doc.page_content,
                "type": doc.metadata.get("type"),
                "score": combined_score,
            })

        # Sort by combined score
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)

        logger.info(f"[RETRIEVAL SYSTEM]: ✅ Successfully retrieved unstructured user data.")
        return scored[:k]

    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving unstructured user data: {str(e)}")
        return []


# 5.3 Knowledge retrieval

def retrieve_knowledge_docs(knowledge_vectorstore, message: str, k: int = 5) -> List[Dict]:
    """
    Retrieve top-k knowledge chunks relevant to the user's message.
    """
    logger.info(f"[RETRIEVAL SYSTEM]: Start retrieving knowledge base data.")

    try:
        results = knowledge_vectorstore.similarity_search_with_score(
            query=message,
            k=k
        )

        scored = []
        for doc, distance in results:
            scored.append({
                "text": doc.page_content,
                "category": doc.metadata.get("category"),
                "tags": doc.metadata.get("tags"),
                "score": 1 - distance
            })

        # Sort by similarity
        scored = sorted(scored, key=lambda x: x["score"], reverse=True)
        logger.info(f"[RETRIEVAL SYSTEM]: ✅ Successfully retrieved knowledge base data.")
        return scored[:k]

    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving knowledge base data: {str(e)}")
        return []


# 5.4 Ranking system

def retrieve_relevant_context_for_user(all_context: dict, message: str, k: int = 5):
    """
    Takes combined structured, unstructured, and knowledge contexts and ranks the top-k
    relevant entries for the given user message using an LLM.

    all_context = {
        "structured": [...],       # list of dicts
        "unstructured": [...],     # list of strings or dicts
        "knowledge": [...],        # list of strings
    }
    """

    logger.info("[RETRIEVAL SYSTEM]: Start retrieving and ranking relevant context data.")

    # Prepare prompt for LLM
    ranking_prompt = f"""
    You are an AI assistant that ranks memory and knowledge context for another AI assistant.

    User message:
    "{message}"

    Your job:
    Select the MOST relevant items from the provided context.

    RULES:
    - Do NOT invent new data
    - ONLY return items from the lists
    - Keep original structure
    - Be strict and selective

    Return TOP {k} per category.

    OUTPUT (JSON ONLY):
    {{
      "structured": [...],
      "unstructured": [...],
      "knowledge": [...]
    }}

    Structured:
    {all_context.get("structured", [])}

    Unstructured:
    {all_context.get("unstructured", [])}

    Knowledge:
    {all_context.get("knowledge", [])}
    """
    try:
        output = call_llm_json(
                ranking_prompt,
                default={
                    "structured": all_context.get("structured", [])[:k],
                    "unstructured": all_context.get("unstructured", [])[:k],
                    "knowledge": all_context.get("knowledge", [])[:k],
                }
            )

        if not isinstance(output, dict):
            return {
                "structured": all_context.get("structured", [])[:k],
                "unstructured": all_context.get("unstructured", [])[:k],
                "knowledge": all_context.get("knowledge", [])[:k],
            }
        logger.info(f"[RETRIEVAL SYSTEM]: ✅ Successfully retrieved relevant context data.")
        return output
    except Exception as e:
        logger.error(f"[RETRIEVAL SYSTEM]: ❌ Error while retrieving relevant context data: {str(e)}")


############################
# 6. TOOLS
############################

# 6.1 Basic tools

@tool
def calculator(expression: str) -> float:
    """Evaluate a mathematical expression like '40 + 12 * 2' safely."""
    logger.info(f"[TOOL]: Calculator: Evaluating expression: {expression}")
    if len(expression) > 50:
        logger.info("[TOOL]: Calculator: IGNORD: Expression must be under 50 characters.")
        return "Expression too long"
    try:
        result = safe_eval(expression)
        logger.info(f"[TOOL]: ✅ Calculator: Successfully evaluated expression: {expression} result: {str(result)}")
        return result
    except Exception as e:
        logger.error(f"[TOOL]: ❌ Calculator: Failed to evaluate expression: {str(e)}")
        return "Invalid mathematical expression"

@tool
def get_current_time() -> str:
    """Returns the current date and time"""
    logger.info("[TOOL]: CURRENT TIME: Getting current time")
    time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    logger.info(f"[TOOL]: ✅ CURRENT TIME: {str(time)}")
    return time

@tool
def get_current_age(user_id: str) -> float:
    """Returns the current age"""
    logger.info(f"[TOOL] CURRENT USER AGE: Getting current age.")
    file_path = "data/user_info.json"
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(f"[TOOL] CURRENT USER AGE: No file {file_path} found.")
        return
    try:
        user_data = data.get(user_id, {})
        birthday = user_data.get("birthday", "No data found")
        birthdate = datetime.fromisoformat(birthday)
        logger.info(f"[TOOL] CURRENT USER AGE: birthday: {birthday}")
        age = (datetime.now() - birthdate).days // 365
        logger.info(f"[TOOL] ✅ CURRENT USER AGE: {str(age)}")
        return age
    except Exception as e:
        logger.info(f"[TOOL] ❌ CURRENT USER AGE: Error while getting user info {e}.")
        return

@tool
def get_user_info(user_id: str, key: str) -> str:
    """Retrieve stored information about a user"""
    logger.info(f"[TOOL] USER INFO: Getting user info key: {key}.")
    # load file
    file_path = "data/user_info.json"
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    except FileNotFoundError:
        logger.info(f"[TOOL] USER INFO: No file {file_path} found.")
        return ""
    try:
        user_data = data.get(user_id, {})
        value = user_data.get(key, "No data found")
        logger.info(f"[TOOL] ✅ USER INFO: {key}: {value}")
        return f"{key}: {value}"
    except Exception as e:
        logger.info(f"[TOOL] ❌ USER INFO: Error while getting user info {e}.")
        return ""


# 6.2 External tools

@tool
def semantic_scholar_search(query: str, store: bool = False) -> List[str]:
    """
    Search Semantic Scholar for scientific papers.

    Use this for:
    - health, nutrition, medical, or scientific questions

    The query should be:
    - short and keyword-based (not a full sentence)
    - include key concepts (e.g., "vitamin D immunity effects dosage")

    Avoid:
    - conversational language
    - unnecessary words
    """
    # https://api.semanticscholar.org/api-docs/graph#tag/Paper-Data/operation/get_graph_paper_relevance_search
    logger.info(f"[TOOL] SEMANTIC SCHOLAR: Called with query: {query}")
    url = "https://api.semanticscholar.org/graph/v1/paper/search"

    params = {
        "query": query,
        "limit": 5,
        "fields": "title,abstract,year",
        "minCitationCount": 5
    }
    for attempt in range(2):  # max 2 tries
        logger.info(f"[TOOL] SEMANTIC SCHOLAR: Request Attempt: {str(attempt)}")
        try:
            response = requests.get(url, params=params)

            if response.status_code != 200:
                logger.info(f"[TOOL] ❌ SEMANTIC SCHOLAR: Bad status: {str(response.status_code)}")
                raise Exception(f"Bad status: {str(response.status_code)}")

            data = response.json()

            paper_texts = []
            for paper in data.get("data", []):
                abstract = paper.get('abstract', '')
                if not abstract:
                    continue
                text = f"Year of publication:{paper.get('year', '')} Title:{paper.get('title', '')}\n{paper.get('abstract', '')}"
                paper_texts.append(text)

            # ✅ success → return immediately
            if store:
                logger.info("[TOOL] SEMANTIC SCHOLAR: Saving findings in knowledge db")
                for text in paper_texts:
                    enrich_knowledge(
                        knowledge_vectorstore,
                        source="SemanticScholar",
                        raw_text=text,
                        chunk_size=300,
                        chunk_overlap=0,
                    )
            logger.info(f"[TOOL] ✅ SEMANTIC SCHOLAR: Findings: {len(paper_texts)}")

            return paper_texts

        except Exception as e:
            if attempt == 1:
                logger.error(f"[TOOL] ❌ SEMANTIC SCHOLAR: Error while request: {str(e)}")
                return []
            time.sleep(1)




############################
# 7. TOOL SELECTION / ROUTING
############################

tools = [calculator, get_current_time, get_current_age, get_user_info, semantic_scholar_search]

tool_metadata = [
    {
        "name": t.name,
        "description": t.description
    }
    for t in tools
]

def select_tools_via_llm(query: str):
    logger.info(f"[TOOL SELECTOR]: Called with query: {query}")
    try:
        prompt = format_prompt(
            TOOL_SELECTION_PROMPT,
            query=query,
            tool_metadata=tool_metadata
        )

        raw_output = call_llm_json(prompt, default=[])

        # ✅ deterministic validation ONLY
        if not isinstance(raw_output, list):
            logger.info(f"[TOOL SELECTOR]: Output not a list: {str(raw_output)} --> returning empty list")
            return []

        valid_names = {t["name"] for t in tool_metadata}

        cleaned = list({name for name in raw_output if name in valid_names})
        if not cleaned:
            logger.info(f"[TOOL SELECTOR]: Not cleaned: {str(cleaned)} --> returning empty list")
            return []

        tool_map = {t.name: t for t in tools}
        tools_list = [tool_map[name] for name in cleaned]
        logger.info(f"[TOOL SELECTOR]: ✅ Tools selected successfully: returning {str(tools_list)} tools")
        return tools_list

    except Exception as e:
        logger.error(f"[TOOL SELECTOR]: Error while selecting tool: {str(e)}")
        return []

def get_bound_model(selected_tools):
    """Return an LLM bound to the given list of tools, using cache if available."""
    # Use a tuple of tool names as the cache key
    logger.info(f"[TOOL SELECTOR]: Selecting Bound model.")
    tool_key = tuple(sorted([t.name for t in selected_tools]))

    if tool_key in bound_models_cache:
        return bound_models_cache[tool_key]

    # Not in cache → bind and store
    model = llm.bind_tools(selected_tools)
    bound_models_cache[tool_key] = model
    logger.info(f"[TOOL SELECTOR]: Tool Bound LLM model selected.")
    return model

# Meta-tool definition
meta_tool_selector = Tool(
    name="tool_selector",
    func=lambda query: select_tools_via_llm(query),
    description="Decides which tools are needed for a given user query"
)



############################
# 8. GRAPH NODES
############################

# 8.1 Agent brain

def agent_node(state: AgentState) -> AgentState:
    """This node will sole the request you input."""
    logger.info(f"[AGENT NODE]: START")
    try:
        logger.info(f"[AGENT NODE]: New request | messages={len(state['messages'])}")
        if "working_memory" not in state:
            state["working_memory"] = {}
        last_user_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        validate_user_input(text=last_user_msg)
        state["working_memory"]["last_user_msg"] = last_user_msg
        base_context = state.get("context", {})
        working_memory = state.get("working_memory", {})
        def merge_context(base, working_memory):
            merged = base.copy()
            structured = merged.get("structured", [])
            for key, value in working_memory.items():
                structured.append({
                    "key": key,
                    "value": value,
                    "category": "working_memory",
                    "score": 1.0
                })
            merged["structured"] = structured
            return merged
        effective_context = merge_context(base_context, working_memory)
        system_prompt = SystemMessage(
            content="You are my AI assistant, please answer my query to the best of your ability."
                    "Only use a tool if the information is not already in conversation history")
        context_message = SystemMessage(
            content=f"Relevant context (latest state):\n{effective_context}"
        )
        selected_tools = meta_tool_selector.func(last_user_msg)
        model = get_bound_model(selected_tools)
        logger.info(f"[AGENT NODE]: Invoking agent")
        response = model.invoke([system_prompt, context_message] + list(state["messages"]))
        validated_response = validate_agent_output(response=response)
        ephemeral_updates = extract_ephemeral_updates(
            user_text=last_user_msg,
            agent_text=validated_response.content
        )
        logger.info(f"[AGENT NODE]: Agent response: {validated_response.content}")
        state["messages"].append(validated_response)
        for key, value in ephemeral_updates.items():
            state["working_memory"][key] = value
        logger.info(f"[AGENT NODE]: New state prepared | messages={len(state['messages'])}")
        logger.info(f"[AGENT NODE]: New state prepared | working_memory={str(state['working_memory'])}")
        logger.info(f"[AGENT NODE]: END")
        return state
    except Exception as e:
        logger.error(f"[AGENT NODE] Failed: {e}")
        return state


# 8.2 Memory updater

def memory_updater_node(state: AgentState, runtime: Runtime) -> AgentState:
    """This node stores necessary information about user and about enriches knowledge from agent dynamically."""
    logger.info(f"[MEMORY UPDATER NODE]: START")
    user_vectorstore = runtime.context["user_vectorstore"]
    user_id = str(state.get("user_id"))
    try:
        last_user_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )
        last_ai_msg = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, AIMessage)),
            ""
        )
        memory_updates = extract_memory_updates(
            text=f"{last_user_msg}\n{last_ai_msg}"
        )
        for item in memory_updates.get("structured", []):
            controlled_structured_data_storage(
                key=item["key"],
                value=item["value"],
                category=item.get("category"),
                user_id=user_id
            )

        for item in memory_updates.get("unstructured", []):
            controlled_unstructured_data_storage(
                user_vectorstore=user_vectorstore,
                text=item["text"],
                type=item.get("type", "general"),
                user_id=user_id
            )

        logger.info("[MEMORY UPDATER NODE]: END")
        return state
    except Exception as e:
        logger.error(f"[MEMORY UPDATER NODE] Failed: {e}")
        return state

# 8.3 Context retrieval

def context_retrieval_node(state: AgentState, runtime: Runtime) -> AgentState:
    logger.info(f"[CONTEXT NODE]: START")
    try:
        knowledge_vectorstore = runtime.context["knowledge_vectorstore"]
        user_vectorstore = runtime.context["user_vectorstore"]
        user_id = str(state["user_id"])
        message = next(
            (m.content for m in reversed(state["messages"]) if isinstance(m, HumanMessage)),
            ""
        )

        logger.info("[CONTEXT NODE]: Extracting retrieval plan")
        # 0.1 LLM decides what to retrieve
        data_to_retrieve = extract_retrieval_plan(text=message)
        logger.info(f"[CONTEXT NODE]: Retrieval plan: {str(data_to_retrieve)}")


        # 0.2 Parse LLM output
        structured_items = data_to_retrieve.get("structured_to_retrieve", [])
        unstructured_items = data_to_retrieve.get("unstructured_to_retrieve", [])

        relevant_categories = list({item["table"] for item in structured_items})
        relevant_keys = list({item["key"] for item in structured_items})
        relevant_types = list({item["type"] for item in unstructured_items})

        logger.info("[CONTEXT NODE]: Retrieving Structured User Data")
        # 1. Structured memory
        structured = retrieve_structured_memory(user_id, relevant_categories=relevant_categories,
                                                relevant_keys=relevant_keys, k=5)
        logger.info(f"[CONTEXT NODE]: Structured Data retrieved count: {len(structured)}")

        logger.info("[CONTEXT NODE]: Retrieving Unstructured User Data")
        # 2. Unstructured memory
        unstructured_docs = retrieve_unstructured_memory(user_vectorstore=user_vectorstore, message=message,
                                                         relevant_types=relevant_types, user_id=user_id, k=5)
        logger.info(f"[CONTEXT NODE]: Unstructured Data retrieved count: {len(unstructured_docs)}")

        logger.info("[CONTEXT NODE]: Retrieving Knowledge Base Data")
        # 3. Knowledge base
        knowledge_docs = retrieve_knowledge_docs(knowledge_vectorstore=knowledge_vectorstore, message=message, k=5)

        logger.info(f"[CONTEXT NODE]: Knowledge Data retrieved count: {len(knowledge_docs)}")

        # 4. Combine
        retrieved_context = {
            "structured": structured,
            "unstructured": unstructured_docs,
            "knowledge": knowledge_docs
        }

        logger.info("[CONTEXT NODE]: Retrieving ranked relevant context")
        # 5. Final ranking
        retrieved_relevant_context = retrieve_relevant_context_for_user(all_context=retrieved_context, message=message, k=5)

        state["context"] = retrieved_relevant_context
        logger.info(f"[CONTEXT NODE]: New state prepared | context: {str(state.get('context'))}")
        logger.info(f"[CONTEXT NODE]: END")
        return state
    except Exception as e:
        logger.error(f"[CONTEXT NODE] Failed: {e}")
        return state

# 8.4 Router

def agent_router(state: AgentState):
    """Decides how the agent should continue"""
    # Tool logic
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", []):
        logger.info("[ROUTER NODE]: TOOLS NEEDED: Routing to --> TOOLS NODE")
        return "tools"

    # End
    logger.info("[ROUTER NODE]: REASONING ENDED: Routing to --> MEMORY UPDATOR")
    return "memory_updater"


############################
# 9. GRAPH BUILD
############################

knowledge_vectorstore = ingest_knowledge(persist_dir="data/core_knowledge", collection_name="core_knowledge")
user_vectorstore = ingest_knowledge(persist_dir="data/user_memory", collection_name="user_memory")

graph = StateGraph(AgentState)

graph.add_node("memory_updater", memory_updater_node)
graph.add_node("context_retrieval", context_retrieval_node)
graph.add_node("our_agent", agent_node)
graph.add_node("tools", ToolNode(tools=tools))

graph.set_entry_point("context_retrieval")
graph.add_edge("context_retrieval", "our_agent")

graph.add_conditional_edges(
    "our_agent",
    agent_router,
    {
        "memory_updater": "memory_updater",
        "tools": "tools"
    }

)

graph.add_edge("tools", "our_agent")
graph.add_edge("memory_updater", END)

app = graph.compile()


############################
# 10. CLI LOOP (TEMPORARY)
############################


user_id = str(uuid.uuid4())

def chat():
    state = {
        "request_id": uuid.uuid4(),
        "user_id": user_id,
        "messages": [],
        "memory_updates": {
            "structured": [],
            "unstructured": []
        },
        "context": {},
    }

    while True:
        request_id = str(uuid.uuid4())
        state["request_id"] = request_id
        user_input = input("\nYou: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        # Add user message
        state["messages"].append(HumanMessage(content=user_input))

        # Run graph
        result = app.invoke(
            input=state,
            context={"knowledge_vectorstore": knowledge_vectorstore, "user_vectorstore": user_vectorstore}
        )

        # Update full state
        state = result

        human_msg = state["messages"][-2]

        # Get last AI message
        ai_msg = state["messages"][-1]

        logger.info(f"[{str(state['request_id'])}] User: {human_msg.content}")

        logger.info(f"[{str(state['request_id'])}] AI AGENT: {ai_msg.content}")


if __name__ == "__main__":
    chat()
