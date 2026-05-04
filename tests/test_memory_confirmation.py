from langchain_core.messages import AIMessage

from app.services.memory_confirmation import (
    apply_confirmation_prompt_to_state,
    build_confirmation_message,
    build_pending_memory_confirmation,
)


def test_build_pending_memory_confirmation_returns_none_for_non_conflicts():
    assert build_pending_memory_confirmation({"decision": "stored"}) is None
    assert build_pending_memory_confirmation({"decision": "no_change"}) is None
    assert build_pending_memory_confirmation({"decision": "ignored"}) is None


def test_build_pending_memory_confirmation_returns_pending_structure_for_conflict():
    pending = build_pending_memory_confirmation(
        {
            "decision": "needs_confirmation",
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        }
    )

    assert pending == {
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }


def test_build_confirmation_message_includes_field_existing_and_proposed_values():
    message = build_confirmation_message(
        {
            "field": "birthdate",
            "category": "profile",
            "existing_value": "1995-04-12",
            "proposed_value": "1996-04-12",
            "reason": "immutable field conflict",
        }
    )

    assert message == (
        "I currently have 1995-04-12 saved as your birthdate. "
        "Do you want me to replace it with 1996-04-12?"
    )


def test_apply_confirmation_prompt_to_state_updates_last_ai_message():
    state = {
        "messages": [AIMessage(content="Thanks for the correction.")],
    }
    pending_confirmation = {
        "field": "birthdate",
        "category": "profile",
        "existing_value": "1995-04-12",
        "proposed_value": "1996-04-12",
        "reason": "immutable field conflict",
    }

    apply_confirmation_prompt_to_state(state, pending_confirmation)

    assert state["messages"][-1].content == (
        "Thanks for the correction.\n\n"
        "I currently have 1995-04-12 saved as your birthdate. "
        "Do you want me to replace it with 1996-04-12?"
    )
