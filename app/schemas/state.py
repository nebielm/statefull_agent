import uuid
from typing import Annotated, Any, Dict, Literal, Sequence, TypedDict

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    working_memory: Dict[str, Any]
    user_id: uuid.UUID
    request_id: uuid.UUID
    context: Dict[Literal["structured", "unstructured", "knowledge"], Any]
