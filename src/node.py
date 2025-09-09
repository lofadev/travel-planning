import json
import os
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import JsonOutputParser, StrOutputParser
from langchain_openai import ChatOpenAI
from langchain_tavily import TavilySearch
from langchain_core.runnables import RunnableConfig

from src.state import TravelState

from dotenv import load_dotenv

load_dotenv()

MODEL = os.getenv("OPENAI_MODEL")
# Khởi tạo LLM (sử dụng OpenAI, thay đổi nếu cần)
llm = ChatOpenAI(model=MODEL, temperature=0.7)

# Tool Tavily Search (tối đa 5 kết quả mỗi query để tránh overload)
tavily_tool = TavilySearch(max_results=5)


# Node 1: Parse input để trích xuất thông tin từ yêu cầu tiếng Việt
def parse_input(state: TravelState, config: RunnableConfig) -> dict:
    input_text = state["messages"][-1].content  # Lấy message cuối cùng là input từ user

    # Prompt để trích xuất thông tin bằng tiếng Việt
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Bạn là chuyên gia phân tích yêu cầu du lịch. "
                "Trích xuất các thông tin sau từ văn bản tiếng Việt, trả về dưới dạng JSON: "
                "destination (địa điểm), duration (số ngày, ví dụ: '3 ngày 2 đêm'), "
                "preferences (danh sách sở thích, ví dụ: ['cà phê chill', 'chụp ảnh thiên nhiên']), "
                "budget (ngân sách, ví dụ: 'tầm trung'), constraints (ràng buộc, ví dụ: ['không đi bộ nhiều']). "
                "Nếu không có thông tin, để null."
            ),
            HumanMessage(content=input_text),
        ]
    )

    parser = JsonOutputParser()
    chain = extract_prompt | llm | parser
    extracted = chain.invoke({})

    # Thêm message về kết quả trích xuất vào lịch sử
    return {
        "extracted_info": extracted,
        "messages": [
            AIMessage(
                content=f"Đã trích xuất thông tin: {json.dumps(extracted, ensure_ascii=False)}"
            )
        ],
    }


# Node 2: Tìm kiếm thông tin sử dụng Tavily tool
def search_info(state: TravelState, config: RunnableConfig) -> dict:
    extracted = state["extracted_info"]
    destination = extracted.get("destination", "unknown")
    budget = extracted.get("budget", "tầm trung")

    # Tạo các queries động dựa trên extracted info (bằng tiếng Việt để search tốt hơn)
    queries = [
        f"Địa điểm du lịch hot ở {destination} năm 2025",
        f"Quán cà phê chill và địa điểm chụp ảnh thiên nhiên ở {destination}",
        f"Giá vé, giờ mở cửa các địa điểm du lịch ở {destination} ngân sách {budget}",
        f"Nhà hàng ăn uống được đánh giá cao ở {destination} gần đây",
        f"Phương tiện di chuyển và lịch trình gợi ý cho chuyến đi {extracted.get('duration', 'unknown')} ở {destination}",
    ]

    # Thực hiện search song song
    search_results = []
    for query in queries:
        result = tavily_tool.invoke({"query": query})
        search_results.append({"query": query, "results": result})

    # Thêm message về kết quả search
    summary = f"Đã tìm kiếm {len(queries)} queries, thu được {sum(len(r['results']) for r in search_results)} kết quả."
    return {"search_results": search_results, "messages": [AIMessage(content=summary)]}


# Node 3: Generate itinerary dựa trên tất cả thông tin
def generate_itinerary(state: TravelState, config: RunnableConfig) -> dict:
    extracted = state["extracted_info"]
    searches = state["search_results"]

    # Chuẩn bị context từ searches
    search_summary = "\n".join(
        [
            f"Query: {s['query']}\nResults: {json.dumps(s['results'], ensure_ascii=False)}"
            for s in searches
        ]
    )

    # Prompt để generate lịch trình Markdown bằng tiếng Việt
    generate_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Bạn là chuyên gia lập kế hoạch du lịch. Dựa trên thông tin trích xuất và kết quả search, "
                "tạo lịch trình chi tiết cho từng ngày dưới dạng Markdown. Bao gồm: "
                "- Schedule (lịch trình hàng ngày)\n"
                "- Transportation (phương tiện di chuyển)\n"
                "- Dining suggestions (gợi ý ăn uống)\n"
                "- Estimated cost (ước tính chi phí tổng và chi tiết).\n"
                "Làm cho nó hợp lý, cá nhân hóa theo sở thích và ràng buộc. Sử dụng dữ liệu từ search để cập nhật."
            ),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(
                content=f"Thông tin trích xuất: {json.dumps(extracted, ensure_ascii=False)}\n"
                f"Kết quả search: {search_summary}"
            ),
        ]
    )

    parser = StrOutputParser()
    chain = generate_prompt | llm | parser
    itinerary_md = chain.invoke({"messages": state["messages"]})

    return {
        "itinerary": itinerary_md,
        "messages": [AIMessage(content="Đã tạo lịch trình hoàn chỉnh.")],
    }
