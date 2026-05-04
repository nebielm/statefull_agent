import re

from langchain_core.messages import AIMessage

from app.repositories.memory_decision_log import append_memory_decision_log
from app.repositories.user_memory import apply_confirmed_structured_correction, build_structured_storage_result


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


def classify_confirmation_reply(text: str) -> str:
    normalized = re.sub(r"\s+", " ", text.strip().lower())
    normalized = normalized.strip(".,!?;:'\"")

    confirm_replies = {
        "yes",
        "y",
        "correct",
        "exactly",
        "stimmt",
        "ja",
    }
    reject_replies = {
        "no",
        "n",
        "nope",
        "keep old",
        "nein",
    }

    if normalized in confirm_replies:
        return "confirm"
    if normalized in reject_replies:
        return "reject"
    return "unclear"


def resolve_pending_confirmation(user_id: str, reply_text: str, pending_confirmation: dict):
    classification = classify_confirmation_reply(reply_text)

    if classification == "confirm":
        result = apply_confirmed_structured_correction(
            user_id=user_id,
            key=pending_confirmation["field"],
            value=pending_confirmation["proposed_value"],
            category=pending_confirmation["category"],
            expected_existing_value=pending_confirmation["existing_value"],
        )
        append_memory_decision_log(
            user_id=user_id,
            result=result,
            source="confirmation_resolver",
        )
        if result.get("decision") != "confirmed_update_applied":
            return {
                "status": "failed",
                "message": (
                    f"I couldn't apply that update safely, so I kept your "
                    f"{pending_confirmation['field']} as {pending_confirmation['existing_value']}."
                ),
                "result": result,
            }
        return {
            "status": "confirmed",
            "message": (
                f"Got it - I updated your {pending_confirmation['field']} "
                f"to {pending_confirmation['proposed_value']}."
            ),
            "result": result,
        }

    if classification == "reject":
        result = build_structured_storage_result(
            decision="confirmation_rejected",
            field=pending_confirmation["field"],
            category=pending_confirmation["category"],
            existing_value=pending_confirmation["existing_value"],
            proposed_value=pending_confirmation["proposed_value"],
            reason="user rejected immutable update",
        )
        append_memory_decision_log(
            user_id=user_id,
            result=result,
            source="confirmation_resolver",
        )
        return {
            "status": "rejected",
            "message": (
                f"Okay, I kept your {pending_confirmation['field']} "
                f"as {pending_confirmation['existing_value']}."
            ),
            "result": result,
        }

    return {
        "status": "unclear",
        "message": (
            f"Please answer yes or no so I know whether to update your "
            f"{pending_confirmation['field']}."
        ),
        "result": None,
    }
