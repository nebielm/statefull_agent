from langchain_core.messages import AIMessage


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
    if getattr(response, "tool_calls", []):
        return response
    if not response.content or not response.content.strip():
        return AIMessage(content="I couldn't generate a proper response.")

    return response
