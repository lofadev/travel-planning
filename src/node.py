import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from langchain_core.messages import AIMessage
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from tavily import TavilyClient

from src.prompt import ask_prompt, extract_prompt, generate_prompt
from src.state import TravelState

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

llm = ChatOpenAI(model=MODEL, temperature=0.1)

tavily_tool = TavilyClient(api_key=TAVILY_API_KEY)


# Node: Parse input to extract info from Vietnamese request
def parse_input(state: TravelState) -> dict:
    # get the last message which is the input from user
    input_text = state["messages"][-1].content

    # get the extracted info before (if any)
    previous_info = state.get("extracted_info", {})

    # create context from old info to help AI understand the context
    context_info = ""
    if previous_info:
        context_info = f"Thông tin đã biết trước đó: {json.dumps(previous_info, ensure_ascii=False)}\n"

    extracted_prompt = extract_prompt(context_info, input_text)

    parser = JsonOutputParser()
    chain = extracted_prompt | llm | parser
    new_extracted = chain.invoke({})

    # Merge old info with new info (prioritize new info if not null)
    final_extracted = previous_info.copy() if previous_info else {}
    for key, value in new_extracted.items():
        if value is not None:
            final_extracted[key] = value
    #
    # add message about the extracted info to the history
    return {
        "extracted_info": final_extracted,
        "messages": [
            AIMessage(
                content=f"Đã trích xuất và cập nhật thông tin: {json.dumps(final_extracted, ensure_ascii=False)}"
            )
        ],
    }


# Conditional edge: (destination, departure_location, duration, people_count)
def check_info(state: TravelState) -> str:
    extracted = state["extracted_info"]
    required_fields = ["destination", "departure_location", "duration", "people_count"]

    # check if all required fields are present
    has_all_required = all(extracted.get(field) for field in required_fields)

    if has_all_required:
        return "search_info"
    else:
        return "ask_for_info"


# Node: ask again if missing info
def ask_for_info(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    missing = []
    if not extracted.get("destination"):
        missing.append("điểm đến du lịch")
    if not extracted.get("departure_location"):
        missing.append("địa điểm xuất phát")
    if not extracted.get("duration"):
        missing.append("số ngày đi (ví dụ: 3 ngày 2 đêm)")
    if not extracted.get("people_count"):
        missing.append("số người đi")

    asked_prompt = ask_prompt(missing, extracted)

    parser = StrOutputParser()
    chain = asked_prompt | llm | parser
    question = chain.invoke({})

    return {"itinerary": question}


# Helper function to perform a single search
def perform_single_search(query_info):
    query, query_type = query_info
    try:
        result = tavily_tool.search(
            query=query,
            max_results=5,  # increase to 5 for more comprehensive results
            include_answer=True,
            # include_raw_content=True,  # include full content for detailed information
            country="vietnam",
            time_range="year",
            # search_depth="advanced",  # use advanced search for more thorough results
        )
        return {"query": query, "query_type": query_type, "results": result}
    except Exception as e:
        return {"query": query, "query_type": query_type, "results": {"error": str(e)}}


# Node 2: search info using Tavily tool with parallel queries
def search_info(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    destination = extracted.get("destination", "")
    departure_location = extracted.get("departure_location", "")
    people_count = extracted.get("people_count", "")

    # create preferences string from user's preferences
    preferences_str = ""
    if extracted.get("preferences"):
        preferences_str = " ".join(extracted["preferences"])

    transports_str = ""
    if extracted.get("transports"):
        transports_str = " ".join(extracted["transports"])

    # make queries for each type of information
    queries = [
        # Query 1: Accommodation with detailed information
        (
            f"Khách sạn homestay resort chỗ nghỉ tốt ở {destination} cho {people_count} người",
            "accommodation",
        ),
        # Query 2: Dining with specific restaurant details
        (
            f"Nhà hàng quán ăn ngon ở {destination} cho {people_count} người",
            "dining",
        ),
        # Query 3: Attractions with complete address information
        (
            f"Địa điểm tham quan du lịch vui chơi giải trí ở {destination}, {preferences_str}",
            "attractions",
        ),
        # Query 4: Transportation with detailed cost and schedule
        (
            f"Phương tiện di chuyển bằng {transports_str} từ {departure_location} đến {destination} cho "
            f"{people_count} người",
            "transportation",
        ),
    ]

    # perform all queries in parallel
    search_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        # submit all searches
        future_to_query = {
            executor.submit(perform_single_search, query_info): query_info
            for query_info in queries
        }

        # collect results as they complete
        for future in as_completed(future_to_query):
            try:
                result = future.result()
                search_results.append(result)
            except Exception as e:
                query_info = future_to_query[future]
                print(f"Exception occurred for {query_info[1]}: {e}")
                search_results.append(
                    {
                        "query": query_info[0],
                        "query_type": query_info[1],
                        "results": {"error": str(e)},
                    }
                )

    return {
        "search_results": search_results,
    }


# Node 3: Generate itinerary based on all information
def generate_itinerary(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    searches = state["search_results"]

    # prepare context from searches with improved formatting for different query types
    search_summary_parts = []
    for s in searches:
        query_type = s.get("query_type", "general")
        query_type_vietnamese = {
            "accommodation": "Chỗ nghỉ",
            "dining": "Ăn uống",
            "attractions": "Tham quan/Vui chơi",
            "transportation": "Di chuyển/Chi phí",
            "general": "Tổng quát",
        }.get(query_type, query_type)

        search_summary_parts.append(
            f"=== {query_type_vietnamese.upper()} ===\n"
            f"Results: {json.dumps(s['results'], ensure_ascii=False)}\n"
        )

    search_summary = "\n".join(search_summary_parts)
    print("search_summary", search_summary)

    generated_prompt = generate_prompt(extracted, search_summary)

    parser = StrOutputParser()
    chain = generated_prompt | llm | parser

    itinerary_md = chain.invoke({"messages": state["messages"][-5:][::-1]})

    return {"itinerary": itinerary_md}
