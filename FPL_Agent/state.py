from typing import Annotated, Sequence
from typing_extensions import TypedDict
import operator
from langchain_core.messages import BaseMessage

# The state passed between agents and tools
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]
    sender: str