from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


# Prompt to extract info in Vietnamese
def extract_prompt(context_info: str, input_text: str):
    extract_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Bạn là chuyên gia phân tích yêu cầu du lịch. "
                    "Trích xuất các thông tin sau từ văn bản tiếng Việt, "
                    "trả về dưới dạng JSON: destination (địa điểm đến), "
                    "departure_location (địa điểm xuất phát), "
                    "duration (số ngày, ví dụ: '3 ngày 2 đêm'), "
                    "people_count (số người đi, ví dụ: 2 hoặc 4), "
                    "preferences (danh sách sở thích, ví dụ: ['cà phê chill', 'chụp ảnh thiên nhiên']), "
                    "budget (ngân sách, ví dụ: 'tầm trung'), "
                    "constraints (ràng buộc, ví dụ: ['không đi bộ nhiều']). "
                    "Nếu không có thông tin trong tin nhắn hiện tại, để null cho field đó. "
                    "LƯU Ý: Hãy xem xét thông tin đã biết trước đó để hiểu ngữ cảnh."
                )
            ),
            HumanMessage(content=f"{context_info}Tin nhắn hiện tại: {input_text}"),
        ]
    )
    return extract_prompt


# Prompt to generate a smart, polite question in Vietnamese
def ask_prompt(missing: list, extracted: dict):
    ask_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content="Bạn là trợ lý du lịch thân thiện. "
                "Dựa trên thông tin đã có, "
                "hãy hỏi lại người dùng về các thông tin thiếu một cách thông minh và lịch sự. "
                "Chỉ hỏi về những gì thiếu, và gợi ý nếu cần. Trả về câu hỏi dạng text đơn giản"
            ),
            HumanMessage(
                content=f"Thông tin thiếu: {', '.join(missing)}. "
                f"Thông tin hiện có: "
                f"{', '.join([str(v) for v in extracted.values() if v is not None])}"
            ),
        ]
    )
    return ask_prompt


# Prompt to generate itinerary Markdown focusing on the main 4 factors
def generate_prompt(extracted: dict, search_summary: str):
    generate_prompt = ChatPromptTemplate.from_messages(
        [
            SystemMessage(
                content=(
                    "Bạn là chuyên gia lập kế hoạch du lịch. Tạo lịch trình du lịch chi tiết dưới dạng Markdown "
                    "dựa trên thông tin người dùng và kết quả tìm kiếm từ Tavily. "
                    "Cấu trúc mỗi ngày với 4 phần chính: Schedule (lịch trình hoạt động, bao gồm vui chơi giải trí), "
                    "Transportation (phương tiện di chuyển), Dining Suggestions (gợi ý ăn uống), "
                    "Estimated Cost (chi phí ước tính). "
                    "Ví dụ cấu trúc:\n## Ngày 1: Tiêu đề ngày\n### 📅 Schedule\n- 8:00: Hoạt động 1 "
                    "(vui chơi)\n- 12:00: Hoạt động 2\n"
                    "### 🚗 Transportation\n- Từ xuất phát: Máy bay (chi tiết từ search)\n- Trong ngày: Taxi 50k\n"
                    "### 🍽️ Dining Suggestions\n- Sáng: Quán ABC (địa chỉ) - 50k\n- Trưa: ...\n"
                    "### 💰 Estimated Cost\n- Tổng ngày: 1.000k VNĐ (chi tiết phân loại)\n\n"
                    "Kết thúc lịch trình bằng phần tổng hợp: Tổng chi phí toàn bộ "
                    "(bao gồm vé di chuyển khứ hồi từ điểm xuất phát), "
                    "LƯU Ý QUAN TRỌNG:\n- CHỈ SỬ DỤNG THÔNG TIN TỪ KẾT QUẢ TÌM KIẾM TRONG HumanMessage. "
                    "Không thêm, suy đoán hoặc bịa đặt bất kỳ thông tin nào không có trong search_summary, "
                    "đặc biệt là tên địa điểm, địa chỉ, giá cả, hoặc chi tiết cụ thể. "
                    "- Đối với chỗ nghỉ, ăn uống, vui chơi: Trích xuất chính xác tên, địa chỉ, giá từ search nếu có; "
                    "- Tích hợp preferences và constraints vào lịch trình một cách hợp lý, chỉ dựa trên search. "
                    "- Ưu tiên tính chính xác, khả thi, và phù hợp với budget, duration, people_count. "
                    "- Nếu thông tin từ search không đủ, giữ lịch trình đơn giản và chỉ ra phần nào cần thêm dữ liệu."
                )
            ),
            MessagesPlaceholder(variable_name="messages"),
            HumanMessage(
                content=(
                    f"Thông tin người dùng:\n- Xuất phát từ: {extracted.get('departure_location')}\n"
                    f"- Điểm đến: {extracted.get('destination')}\n"
                    f"- Thời gian: {extracted.get('duration')}\n"
                    f"- Số người: {extracted.get('people_count')}\n"
                    f"- Sở thích: {', '.join(extracted.get('preferences'))}\n"
                    f"- Ngân sách: {extracted.get('budget', 'tầm trung')}\n"
                    f"- Ràng buộc: {', '.join(extracted.get('constraints'))}\n\n"
                    f"Kết quả tìm kiếm từ Tavily: {search_summary.strip()}"
                )
            ),
        ]
    )
    return generate_prompt
