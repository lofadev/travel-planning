from langgraph.graph import StateGraph, END
from src.state import TravelState
from src.node import (
    ask_for_info,
    check_info,
    parse_input,
    search_info,
    generate_itinerary,
)

# Xây dựng graph
workflow = StateGraph(TravelState)

# Thêm nodes
workflow.add_node("parse_input", parse_input)
workflow.add_node("ask_for_info", ask_for_info)
workflow.add_node("search_info", search_info)
workflow.add_node("generate_itinerary", generate_itinerary)

# Sử dụng conditional edges trực tiếp sau parse_input
workflow.add_conditional_edges(
    "parse_input",
    check_info,
    {"search_info": "search_info", "ask_for_info": "ask_for_info"},
)
workflow.add_edge("ask_for_info", END)  # End với message hỏi
workflow.add_edge("search_info", "generate_itinerary")
workflow.add_edge("generate_itinerary", END)

# Set entry point
workflow.set_entry_point("parse_input")


# Compile graph
graph = workflow.compile()
