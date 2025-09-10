from typing_extensions import NotRequired, TypedDict, List, Annotated
from langgraph.graph.message import AnyMessage, add_messages


class TravelState(TypedDict):
    messages: Annotated[
        List[AnyMessage], add_messages
    ]  # history of messages for agent to think
    extracted_info: NotRequired[
        dict
    ]  # extracted info from input: destination, duration, preferences, budget, constraints
    search_results: NotRequired[List[dict]]  # results from search queries
    itinerary: NotRequired[str]  # final itinerary in Markdown format
