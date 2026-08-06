from ollama import chat
import ollama
import json
from backend.services.chroma_service import search_document
from backend.services.history_service import (get_history, add_message)

SYSTEM_PROMPT = """
Bạn là Company AI Assistant của VNPT EPAY.
Luôn trả lời bằng tiếng Việt.
Ưu tiên tuyệt đối sử dụng thông tin trong DANH SÁCH TÀI LIỆU.
Nếu tài liệu đã có câu trả lời thì hãy trả lời trực tiếp.
Có thể tổng hợp thông tin từ nhiều đoạn tài liệu.
Không được sử dụng kiến thức bên ngoài.
Chỉ khi toàn bộ tài liệu không chứa thông tin thì mới trả lời:
"Tôi không tìm thấy thông tin trong tài liệu."
"""

def get_top_k(question):
    q = question.lower()
    if any(keyword in q for keyword in ["liệt kê", "toàn bộ", "tất cả", "bao gồm", "danh sách"]):
        return 10
    if any(keyword in q for keyword in ["khác nhau", "so sánh", "giống nhau"]):
        return 8
    if any(keyword in q for keyword in ["bảng", "cột", "hàng", "tham số", "request"]):
        return 8
    if any(keyword in q for keyword in ["là gì", "là ai", "định nghĩa"]):
        return 3
    return 6

def split_into_subquestions(question):
    """
    Tách câu hỏi gốc thành các câu hỏi con ĐỘC LẬP (mỗi câu tự đủ nghĩa,
    không cần đọc câu khác mới hiểu). Mục đích: mỗi ý trong câu hỏi được
    tìm kiếm (retrieval) RIÊNG BIỆT, thay vì gộp chung vào 1 vector duy nhất
    rồi để ý này lấn át ý kia khi tìm top_k.

    Nếu câu hỏi chỉ có 1 ý, hoặc bước phân tích lỗi/rỗng, trả về đúng câu hỏi gốc
    (an toàn, không làm mất thông tin).
    """
    prompt = f"""
Nhiệm vụ: tách câu hỏi sau thành các câu hỏi con độc lập.

Quy tắc bắt buộc:
- Mỗi câu hỏi con phải tự đầy đủ ý nghĩa, không phụ thuộc câu khác.
- Giữ nguyên tên API, tên bảng, tên trường dữ liệu, số liệu có trong câu gốc.
- Nếu câu hỏi gốc chỉ có 1 ý duy nhất, trả về chính xác câu hỏi đó, không thêm bớt.
- Chỉ trả về danh sách câu hỏi con, mỗi câu 1 dòng.
- Không đánh số, không giải thích, không thêm bất kỳ chữ nào khác ngoài các câu hỏi.

CÂU HỎI GỐC:
{question}
"""
    try:
        response = chat(
            model="qwen2.5:7b",
            messages=[{"role": "user", "content": prompt}]
        )
        raw = response["message"]["content"].strip()
        sub_questions = [
            line.strip("-•* ").strip()
            for line in raw.split("\n")
            if line.strip()
        ]
    except Exception:
        sub_questions = []

    if not sub_questions:
        return [question]
    return sub_questions

def rewrite_query(question):
    prompt = f"""
Bạn là AI chuyên tạo truy vấn tìm kiếm.
Không được trả lời câu hỏi.
Chỉ sinh ra tối đa 5 câu truy vấn khác nhau.

Quy tắc:
- giữ nguyên ý nghĩa
- giữ nguyên tên API
- giữ nguyên tên bảng
- giữ nguyên tên trường
- được phép dùng từ đồng nghĩa
- mỗi truy vấn trên 1 dòng
- không đánh số
- không giải thích
Câu hỏi:
{question}
"""
    try:
        response = chat(
            model="qwen2.5:7b",
            messages=[
                {
                    "role":"user",
                    "content":prompt
                }
            ]
        )
        text = response["message"]["content"]
    except Exception as e:
        print("Lỗi rewrite_query:", e)
        return [question]

    queries = []
    for line in text.split("\n"):
        line=line.strip()
        if line:
            queries.append(line)
    if not queries:
        return [question]
    return queries

def _search_and_collect(queries, top_k, seen, documents, metadatas, keyword_scores):
    """
    Search theo danh sách queries, gộp thêm kết quả vào documents/metadatas/
    keyword_scores hiện có, dùng chung 1 set "seen" để loại trùng xuyên suốt
    nhiều lượt gọi (kể cả giữa lần tìm đầu và lần fallback sau này).
    """
    per_query_k = max(3, top_k // len(queries))
    for q in queries:
        result = search_document(q, top_k=per_query_k)
        for doc, meta, score in zip(
            result["documents"][0], result["metadatas"][0], result["keyword_scores"][0]
        ):
            key = (meta.get("source"), meta.get("chunk_index"))
            if key in seen:
                continue
            seen.add(key)
            documents.append(doc)
            metadatas.append(meta)
            keyword_scores.append(score)


def _retrieve_documents(question, sub_questions, top_k):
    """
    Lần 1: tìm bằng câu hỏi con + câu hỏi gốc (CHƯA rewrite) -> nhanh, đủ dùng
    với đa số câu hỏi nhờ retrieval đã có hybrid rank (vector + từ khoá).

    Chỉ khi kết quả lần 1 có QUÁ ÍT chunk "khớp từ khoá thật" (keyword_score > 0),
    mới kích hoạt rewrite_query() làm phương án DỰ PHÒNG — tìm thêm bằng các
    cách diễn đạt khác (đề phòng câu hỏi dùng từ khác hẳn tài liệu). Nhờ vậy,
    phần lớn câu hỏi thông thường không phải trả thêm 5+ lượt gọi model chỉ
    để viết lại truy vấn một cách không cần thiết.
    """
    MIN_GOOD_MATCHES = 3

    documents, metadatas, keyword_scores = [], [], []
    seen = set()

    first_pass_queries = list(sub_questions)
    if question not in first_pass_queries:
        first_pass_queries.append(question)

    _search_and_collect(first_pass_queries, top_k, seen, documents, metadatas, keyword_scores)

    good_matches = sum(1 for s in keyword_scores if s > 0)
    print(f"[debug] Lần 1 (chưa rewrite): {len(documents)} chunk, {good_matches} chunk khớp từ khoá tốt")

    if good_matches < MIN_GOOD_MATCHES:
        print("[debug] Kết quả lần 1 chưa đủ tốt -> kích hoạt rewrite_query() (fallback)...")
        expanded_queries = []
        for sq in sub_questions:
            expanded_queries.extend(rewrite_query(sq))

        _search_and_collect(expanded_queries, top_k, seen, documents, metadatas, keyword_scores)
        print(f"[debug] Sau fallback: tổng {len(documents)} chunk")
    else:
        print("[debug] Kết quả lần 1 đã đủ tốt -> bỏ qua rewrite_query() để tiết kiệm thời gian.")

    return documents, metadatas

def need_history(question):
    prompt = f"""
Chỉ trả lời đúng 1 từ: YES hoặc NO.
YES: nếu câu hỏi cần lịch sử hội thoại mới hiểu.
NO: nếu câu hỏi đã đầy đủ ý nghĩa.
Câu hỏi:
{question}
"""
    try:
        
        response = chat(
            model="qwen2.5:7b",
            messages=[{
                    "role": "user",
                    "content": prompt}]
        )
        answer = response["message"]["content"].strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print("Lỗi need_history:", e)
        return True 

def ask_ai(question):
    top_k = get_top_k(question)
    sub_questions = split_into_subquestions(question)

    print("=" * 80)
    print("Câu hỏi gốc:", question)
    print("Câu hỏi con:", sub_questions)
    print("=" * 80)

    documents, metadatas = _retrieve_documents(question, sub_questions, top_k)

    print("=" * 50)
    print("Documents: ")
    for doc in documents:
        print(doc[:300])
        print("=" * 30)
    print("=" * 50)

    history_text = ""
    if need_history(question):
        history = get_history()
        for msg in history:
            history_text += f"{msg['role']}:\n{msg['content']}\n\n"

    # Không tìm thấy document
    if len(documents) == 0:
        answer = "Tôi không tìm thấy thông tin trong tài liệu."
        add_message("user", question)
        add_message("assistant", answer)
        return answer

    content = ""
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        is_table = "|" in doc
        content += f"""
    ====================
    Tài liệu {i}

    Tên file:
    {meta["source"]}

    Tiêu đề:
    {meta.get("title")}

    Loại:
    {"BẢNG DỮ LIỆU" if is_table else "VĂN BẢN"}

    Nội dung:
    {doc}
    ====================
    """

    sub_questions_text = "\n".join(f"- {sq}" for sq in sub_questions)

    prompt = f"""
    Bạn là trợ lý AI nội bộ.
    ====================
    CÁC Ý CẦN TRẢ LỜI (đã tách từ câu hỏi gốc)
    {sub_questions_text}
    ====================
    QUY TẮC:
    1. Chỉ sử dụng thông tin trong DANH SÁCH TÀI LIỆU.
    2. Không được tự suy diễn, không bổ sung kiến thức bên ngoài.
    3. Nếu nhiều tài liệu cùng chứa thông tin thì hãy tổng hợp chúng thành một câu trả lời hoàn chỉnh.
        Không cần mỗi tài liệu đều phải chứa đầy đủ đáp án.
    4. Nếu tài liệu không ghi thì phải trả lời: "Tài liệu không đề cập."
    5. Không được bịa thêm thông tin.
    6. Nếu trong tài liệu có BẢNG DỮ LIỆU thì:
       - Luôn ưu tiên sử dụng bảng trước.
       - Mỗi hàng là một bản ghi, dòng đầu tiên là tên cột.
       - Khi trả lời phải đối chiếu đúng hàng và đúng cột.
    7. Phải trả lời ĐẦY ĐỦ TỪNG Ý trong mục "CÁC Ý CẦN TRẢ LỜI" ở trên,
       không được chỉ trả lời ý đầu tiên rồi dừng lại.
    8. Nếu một ý chưa có thông tin trong tài liệu, ghi rõ: "Tài liệu không đề cập"
       cho riêng ý đó, không bỏ qua hay gộp chung vào các ý khác.
    9. Nếu câu hỏi yêu cầu liệt kê thì phải liệt kê đầy đủ.
    10. Nếu tài liệu chứa sơ đồ, bảng có nhiều vai trò thì phải xác định đúng thông tin của từng vai trò
        không được gộp hay trộn lẫn các thông tin, công việc giữa các vai trò 
    ====================
    Lịch sử hội thoại
    {history_text}
    ====================
    - hãy sử dụng lịch sử hội thoại để hiểu ngữ cảnh của câu hỏi hiện tại
    - nếu lịch sử và tài liệu mâu thuẫn thì luôn ưu tiên tài liệu
    ====================
    DANH SÁCH TÀI LIỆU
    {content}
    ====================
    CÂU HỎI GỐC
    {question}
    ====================
    QUY TẮC TRẢ LỜI:
        - Chỉ sử dụng thông tin trong DANH SÁCH TÀI LIỆU.
        - Nếu tài liệu có thông tin thì hãy trả lời đầy đủ.
        - Có thể tổng hợp nhiều tài liệu nếu cần.
        - Không được bổ sung kiến thức bên ngoài.
        - Chỉ khi toàn bộ tài liệu đều không có thông tin thì mới trả lời:
        "Tôi không tìm thấy thông tin trong tài liệu."
    ====================
    TRẢ LỜI:
    """
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    print("===== QWEN3B ĐANG SUY LUẬN =====")
    response = chat(
        model="qwen2.5:7b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    print("===== QWEN3B XONG =====")
    answer = response["message"]["content"]
    add_message("user", question)
    add_message("assistant", answer)

    return answer