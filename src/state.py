from typing_extensions import TypedDict, List, Annotated
from langgraph.graph.message import AnyMessage, add_messages


class TravelState(TypedDict):
    messages: Annotated[
        List[AnyMessage], add_messages
    ]  # Lịch sử messages để agent suy nghĩ
    extracted_info: dict  # Thông tin trích xuất từ input: destination, duration, preferences, budget, constraints
    search_results: List[dict]  # Kết quả từ các search queries
    itinerary: str  # Lịch trình cuối cùng dưới dạng Markdown
