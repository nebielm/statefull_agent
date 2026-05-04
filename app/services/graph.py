from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

from app.core.logging import logger
from app.llm.extractors import extract_ephemeral_updates, extract_memory_updates, extract_retrieval_plan
from app.repositories.user_memory import controlled_structured_data_storage, retrieve_structured_memory
from app.schemas.state import AgentState
from app.services.memory import controlled_unstructured_data_storage
from app.services.retrieval import (
    retrieve_knowledge_docs,
    retrieve_relevant_context_for_user,
    retrieve_unstructured_memory,
)
from app.services.tools import get_bound_model, select_tools_via_llm, tools
from app.utils.validation import validate_agent_output, validate_user_input


def agent_node(state: AgentState) -> AgentState:
    """This node will sole the request you input."""
    logger.info("[AGENT NODE]: START")
    try:
        logger.info(f"[AGENT NODE]: New request | messages={len(state['messages'])}")
        if "working_memory" not in state:
            state["working_memory"] = {}
        last_user_msg = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "",
        )

        validate_user_input(text=last_user_msg)
        state["working_memory"]["last_user_msg"] = last_user_msg
        base_context = state.get("context") or {}
        working_memory = state.get("working_memory", {})

        def merge_context(base, working_memory):
            merged = base.copy()
            structured = list(merged.get("structured", []))
            for key, value in working_memory.items():
                structured.append(
                    {
                        "key": key,
                        "value": value,
                        "category": "working_memory",
                        "score": 1.0,
                    }
                )
            merged["structured"] = structured
            return merged

        effective_context = merge_context(base_context, working_memory)
        system_prompt = SystemMessage(
            content="You are my AI assistant, please answer my query to the best of your ability."
            "Only use a tool if the information is not already in conversation history"
        )
        context_message = SystemMessage(content=f"Relevant context (latest state):\n{effective_context}")
        selected_tools = select_tools_via_llm(last_user_msg)
        model = get_bound_model(selected_tools)
        logger.info("[AGENT NODE]: Invoking agent")
        response = model.invoke([system_prompt, context_message] + list(state["messages"]))
        validated_response = validate_agent_output(response=response)
        ephemeral_updates = extract_ephemeral_updates(
            user_text=last_user_msg,
            agent_text=validated_response.content,
        )
        logger.info(f"[AGENT NODE]: Agent response: {validated_response.content}")
        state["messages"].append(validated_response)
        for key, value in ephemeral_updates.items():
            state["working_memory"][key] = value
        logger.info(f"[AGENT NODE]: New state prepared | messages={len(state['messages'])}")
        logger.info(f"[AGENT NODE]: New state prepared | working_memory={str(state['working_memory'])}")
        logger.info("[AGENT NODE]: END")
        return state
    except Exception as e:
        logger.exception(f"[AGENT NODE] Failed: {e}")
        return state


def memory_updater_node(state: AgentState, runtime: Runtime) -> AgentState:
    """This node stores necessary information about user and about enriches knowledge from agent dynamically."""
    logger.info("[MEMORY UPDATER NODE]: START")
    user_vectorstore = runtime.context["user_vectorstore"]
    user_id = str(state.get("user_id"))
    try:
        last_user_msg = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "",
        )
        last_ai_msg = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, AIMessage)),
            "",
        )
        memory_updates = extract_memory_updates(text=f"{last_user_msg}\n{last_ai_msg}")
        state.setdefault("memory_updates", {})
        state["memory_updates"]["structured_results"] = []
        for item in memory_updates.get("structured", []):
            result = controlled_structured_data_storage(
                key=item["key"],
                value=item["value"],
                category=item.get("category"),
                user_id=user_id,
            )
            state["memory_updates"]["structured_results"].append(result)

        for item in memory_updates.get("unstructured", []):
            controlled_unstructured_data_storage(
                user_vectorstore=user_vectorstore,
                text=item["text"],
                type=item.get("type", "general"),
                user_id=user_id,
            )

        logger.info("[MEMORY UPDATER NODE]: END")
        return state
    except Exception as e:
        logger.exception(f"[MEMORY UPDATER NODE] Failed: {e}")
        return state


def context_retrieval_node(state: AgentState, runtime: Runtime) -> AgentState:
    logger.info("[CONTEXT NODE]: START")
    try:
        knowledge_vectorstore = runtime.context["knowledge_vectorstore"]
        user_vectorstore = runtime.context["user_vectorstore"]
        user_id = str(state["user_id"])
        message = next(
            (message.content for message in reversed(state["messages"]) if isinstance(message, HumanMessage)),
            "",
        )

        logger.info("[CONTEXT NODE]: Extracting retrieval plan")
        data_to_retrieve = extract_retrieval_plan(text=message)
        logger.info(f"[CONTEXT NODE]: Retrieval plan: {str(data_to_retrieve)}")

        structured_items = data_to_retrieve.get("structured_to_retrieve", [])
        unstructured_items = data_to_retrieve.get("unstructured_to_retrieve", [])

        relevant_categories = list(
            {
                item.get("table") or item.get("category")
                for item in structured_items
                if item.get("table") or item.get("category")
            }
        )
        relevant_keys = list({item.get("key") for item in structured_items if item.get("key")})
        relevant_types = list({item.get("type") for item in unstructured_items if item.get("type")})

        logger.info("[CONTEXT NODE]: Retrieving Structured User Data")
        structured = retrieve_structured_memory(
            user_id,
            relevant_categories=relevant_categories,
            relevant_keys=relevant_keys,
            k=5,
        )
        logger.info(f"[CONTEXT NODE]: Structured Data retrieved count: {len(structured)}")

        logger.info("[CONTEXT NODE]: Retrieving Unstructured User Data")
        unstructured_docs = retrieve_unstructured_memory(
            user_vectorstore=user_vectorstore,
            message=message,
            relevant_types=relevant_types,
            user_id=user_id,
            k=5,
        )
        logger.info(f"[CONTEXT NODE]: Unstructured Data retrieved count: {len(unstructured_docs)}")

        logger.info("[CONTEXT NODE]: Retrieving Knowledge Base Data")
        knowledge_docs = retrieve_knowledge_docs(
            knowledge_vectorstore=knowledge_vectorstore,
            message=message,
            k=5,
        )

        logger.info(f"[CONTEXT NODE]: Knowledge Data retrieved count: {len(knowledge_docs)}")

        retrieved_context = {
            "structured": structured,
            "unstructured": unstructured_docs,
            "knowledge": knowledge_docs,
        }

        logger.info("[CONTEXT NODE]: Retrieving ranked relevant context")
        retrieved_relevant_context = retrieve_relevant_context_for_user(
            all_context=retrieved_context,
            message=message,
            k=5,
        )

        state["context"] = retrieved_relevant_context
        logger.info(f"[CONTEXT NODE]: New state prepared | context: {str(state.get('context'))}")
        logger.info("[CONTEXT NODE]: END")
        return state
    except Exception as e:
        logger.exception(f"[CONTEXT NODE] Failed: {e}")
        return state


def agent_router(state: AgentState):
    """Decides how the agent should continue"""
    last_msg = state["messages"][-1]
    if getattr(last_msg, "tool_calls", []):
        logger.info("[ROUTER NODE]: TOOLS NEEDED: Routing to --> TOOLS NODE")
        return "tools"

    logger.info("[ROUTER NODE]: REASONING ENDED: Routing to --> MEMORY UPDATOR")
    return "memory_updater"


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
        "tools": "tools",
    },
)

graph.add_edge("tools", "our_agent")
graph.add_edge("memory_updater", END)

app = graph.compile()
