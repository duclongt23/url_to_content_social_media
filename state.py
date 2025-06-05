
from typing_extensions import TypedDict, List, Literal
from pydantic import BaseModel
from langgraph.graph.message import MessagesState
import operator
from typing import Annotated


Platform = Literal['Twitter', 'Linkedin', 'Instagram', 'Blog']


class InputState(TypedDict):
    url: str
    platforms: list[Platform]

class SummaryInputState(TypedDict):
    text: str
    platforms: list[Platform]

class SummaryOutputState(TypedDict):
    text: str
    text_summary: str
    platforms: list[Platform]

class ResearchOutputState(TypedDict):
    text: str
    research: str
    platforms: list[Platform]

class IntentMatchingInputState(TypedDict):
    text: str
    research: str
    platforms: list[Platform]

class FinalState(TypedDict):
    contents: Annotated[list, operator.add]

class GeneratedContent(TypedDict):
    generated_content: str

