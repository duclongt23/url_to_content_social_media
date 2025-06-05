from typing_extensions import TypedDict, List, Literal
from pydantic import BaseModel
from langgraph.graph.message import MessagesState
import operator
from typing import Annotated

class ResearchQuestion(TypedDict):
    questions: list[str]

class InputState(TypedDict):
    topic: str
    research: str
    platforms: list[str]
    contents: Annotated[list, operator.add]
    generated_content: str

