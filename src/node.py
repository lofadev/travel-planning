import json
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch

from src.state import TravelState

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")
# Khởi tạo LLM (sử dụng OpenAI, thay đổi nếu cần)
llm = ChatOpenAI(model=MODEL, temperature=0.1)  # Giảm temperature để tăng tốc độ

# LLM với streaming cho generate_itinerary
streaming_llm = ChatOpenAI(model=MODEL, temperature=0.1, streaming=True)

# Tool Tavily Search (tối đa 10 kết quả mỗi query)
tavily_tool = TavilySearch(max_results=10)


# Node 1: Parse input để trích xuất thông tin từ yêu cầu tiếng Việt
def parse_input(state: TravelState) -> dict:
    input_text = state["messages"][-1].content  # Lấy message cuối cùng là input từ user
    
    # Lấy thông tin đã trích xuất trước đó (nếu có)
    previous_info = state.get("extracted_info", {})
    
    # Tạo context từ thông tin cũ để AI hiểu ngữ cảnh
    context_info = ""
    if previous_info:
        context_info = f"Thông tin đã biết trước đó: {json.dumps(previous_info, ensure_ascii=False)}\n"

    # Prompt để trích xuất thông tin bằng tiếng Việt
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Bạn là chuyên gia phân tích yêu cầu du lịch. "
                "Trích xuất các thông tin sau từ văn bản tiếng Việt, trả về dưới dạng JSON: "
                "destination (địa điểm), duration (số ngày, ví dụ: '3 ngày 2 đêm'), "
                "people_count (số người đi, ví dụ: 2 hoặc 4), "
                "preferences (danh sách sở thích, ví dụ: ['cà phê chill', 'chụp ảnh thiên nhiên']), "
                "budget (ngân sách, ví dụ: 'tầm trung'), constraints (ràng buộc, ví dụ: ['không đi bộ nhiều']). "
                "Nếu không có thông tin trong tin nhắn hiện tại, để null cho field đó. "
                "LƯU Ý: Hãy xem xét thông tin đã biết trước đó để hiểu ngữ cảnh."
            ),
            HumanMessage(content=f"{context_info}Tin nhắn hiện tại: {input_text}"),
        ]
    )

    parser = JsonOutputParser()
    chain = extract_prompt | llm | parser
    new_extracted = chain.invoke({})
    
    # Merge thông tin cũ với thông tin mới (ưu tiên thông tin mới nếu không null)
    final_extracted = previous_info.copy() if previous_info else {}
    for key, value in new_extracted.items():
        if value is not None:
            final_extracted[key] = value

    # Thêm message về kết quả trích xuất vào lịch sử
    return {
        "extracted_info": final_extracted,
        "messages": [
            AIMessage(
                content=f"Đã trích xuất và cập nhật thông tin: {json.dumps(final_extracted, ensure_ascii=False)}"
            )
        ],
    }


# Conditional edge: Kiểm tra nếu đủ info (destination, duration, people_count không null)
def check_info(state: TravelState) -> str:
    extracted = state["extracted_info"]
    required_fields = ["destination", "duration", "people_count"]
    
    # Kiểm tra xem có đủ thông tin cần thiết không
    has_all_required = all(extracted.get(field) for field in required_fields)
    
    if has_all_required:
        return "search_info"
    else:
        return "ask_for_info"

# Node: Hỏi lại nếu thiếu info
def ask_for_info(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    missing = []
    if not extracted.get("destination"):
        missing.append("điểm đến du lịch")
    if not extracted.get("duration"):
        missing.append("số ngày đi (ví dụ: 3 ngày 2 đêm)")
    if not extracted.get("people_count"):
        missing.append("số người đi")
    
    # Prompt để generate câu hỏi thông minh, lịch sự bằng tiếng Việt
    ask_prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content="Bạn là trợ lý du lịch thân thiện. Dựa trên thông tin đã có, hãy hỏi lại người dùng về các thông tin thiếu một cách thông minh và lịch sự. "
                              "Chỉ hỏi về những gì thiếu, và gợi ý nếu cần. Trả về chỉ câu hỏi dưới dạng text đơn giản."),
        HumanMessage(content=f"Thông tin thiếu: {', '.join(missing)}. Thông tin hiện có: {json.dumps(extracted, ensure_ascii=False)}")
    ])
    
    parser = StrOutputParser()
    chain = ask_prompt | llm | parser
    question = chain.invoke({})
    
    return {
        "itinerary": question,  # Sử dụng itinerary để lưu message hỏi lại làm output
        "messages": [AIMessage(content=question)]
    }

# Node 2: Tìm kiếm thông tin sử dụng Tavily tool
def search_info(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    destination = extracted.get("destination", "unknown")
    budget = extracted.get("budget", "tầm trung")
    duration = extracted.get("duration", "unknown")
    people_count = extracted.get("people_count", "unknown")

    # Tạo preferences string từ sở thích người dùng
    preferences_str = ""
    if extracted.get("preferences"):
        preferences_str = " ".join(extracted["preferences"])
    
    # Tạo 1 query tổng quát nhưng bao quát - tập trung vào 3 yếu tố chính
    comprehensive_query = (
        f"Du lịch {destination} {duration} cho {people_count} người ngân sách {budget}: "
        f"nhà hàng quán ăn địa phương giá cả, khách sạn homestay chỗ nghỉ giá tốt, "
        f"chi phí ước tính chi tiết ăn ở di chuyển {preferences_str}"
    )

    # Thực hiện search với query tổng hợp
    result = tavily_tool.invoke({
        "query": comprehensive_query,
    })
    
    search_results = [{"query": comprehensive_query, "results": result}]

    # Thêm message về kết quả search
    summary = f"Đã tìm kiếm thông tin tổng hợp cho {destination}, thu được {len(result)} kết quả về ăn uống, chỗ nghỉ và giá cả."
    return {"search_results": search_results, "messages": [AIMessage(content=summary)]}


# Node 3: Generate itinerary dựa trên tất cả thông tin
def generate_itinerary(state: TravelState) -> dict:
    extracted = state["extracted_info"]
    searches = state["search_results"]

    # Chuẩn bị context từ searches
    search_summary = "\n".join(
        [
            f"Query: {s['query']}\nResults: {json.dumps(s['results'], ensure_ascii=False)}"
            for s in searches
        ]
    )

    # Prompt để generate lịch trình Markdown tập trung vào 3 yếu tố chính
    generate_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Tạo lịch trình du lịch Markdown. Mỗi ngày gồm 4 phần: Schedule, Transportation, Dining, Cost. "
                "Ví dụ:\n## Ngày 1: Khám phá trung tâm\n### 📅 Schedule\n- 8:00 - Ăn sáng\n- 9:00 - Tham quan\n"
                "### 🚗 Transportation\n- Taxi: 50k\n### 🍽️ Dining\n- Sáng: Phở - 50k\n"
                "### 💰 Cost\n- Tổng: 100k VNĐ\n\nKết thúc bằng tổng hợp chi phí toàn bộ."
            ),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(
                content=f"Địa điểm: {extracted.get('destination')}, Thời gian: {extracted.get('duration')}, "
                f"Số người: {extracted.get('people_count')}, Ngân sách: {extracted.get('budget', 'tầm trung')}\n\n"
                f"{search_summary.strip()}"
            ),
        ]
    )

    parser = StrOutputParser()
    chain = generate_prompt | streaming_llm | parser
    
    # Sử dụng streaming để tăng tốc độ perceived
    itinerary_md = ""
    for chunk in chain.stream({"messages": state["messages"]}):
        itinerary_md += chunk

    return {
        "itinerary": itinerary_md,
        "messages": [AIMessage(content="Đã tạo lịch trình hoàn chỉnh.")],
    }
