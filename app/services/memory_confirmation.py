from langchain_core.messages import AIMessage


def build_pending_memory_confirmation(result: dict):
    if not result or result.get("decision") != "needs_confirmation":
        return None

    return {
        "field": result.get("field"),
        "category": result.get("category"),
        "existing_value": result.get("existing_value"),
        "proposed_value": result.get("proposed_value"),
        "reason": result.get("reason"),
    }


def build_confirmation_message(pending_confirmation: dict) -> str:
    return (
        f"I currently have {pending_confirmation['existing_value']} saved as your "
        f"{pending_confirmation['field']}. Do you want me to replace it with "
        f"{pending_confirmation['proposed_value']}?"
    )


def apply_confirmation_prompt_to_state(state: dict, pending_confirmation: dict):
    confirmation_message = build_confirmation_message(pending_confirmation)

    if state.get("messages") and isinstance(state["messages"][-1], AIMessage):
        last_ai_message = state["messages"][-1]
        if isinstance(last_ai_message.content, str) and last_ai_message.content.strip():
            state["messages"][-1] = AIMessage(
                content=f"{last_ai_message.content}\n\n{confirmation_message}"
            )
        else:
            state["messages"][-1] = AIMessage(content=confirmation_message)
    else:
        state.setdefault("messages", []).append(AIMessage(content=confirmation_message))

    return state
