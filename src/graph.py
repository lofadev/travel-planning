from langgraph.graph import StateGraph, END
from src.state import TravelState
from src.node import parse_input, search_info, generate_itinerary

# Xây dựng graph
workflow = StateGraph(TravelState)

# Thêm nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("search_info", search_info)
workflow.add_node("generate_itinerary", generate_itinerary)

# Thiết lập edges (tuyến tính cho đơn giản, có thể thêm conditional nếu cần loop)
workflow.add_edge("parse_input", "search_info")
workflow.add_edge("search_info", "generate_itinerary")
workflow.add_edge("generate_itinerary", END)

# Set entry point
workflow.set_entry_point("parse_input")

# Compile graph
graph = workflow.compile()
