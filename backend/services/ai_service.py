from ollama import chat
import ollama
import json
import re
import unicodedata
from difflib import SequenceMatcher
from backend.services.chroma_service import search_document
from backend.services.history_service import (get_history, add_message)

SYSTEM_PROMPT = """
Bạn là Company AI Assistant của VNPT EPAY.
Luôn trả lời bằng tiếng Việt.
Nguồn kiến thức duy nhất để trả lời là
DANH SÁCH TÀI LIỆU được cung cấp trong prompt.
Bạn được phép:
- Trích xuất thông tin trực tiếp từ tài liệu.
- Tổng hợp thông tin từ nhiều đoạn tài liệu.
- Liên kết các dữ kiện có liên quan.
- Suy luận logic từ các dữ kiện đã xuất hiện trong tài liệu.
- Diễn giải lại nội dung tài liệu để trả lời đúng ý câu hỏi.
Không yêu cầu câu trả lời phải xuất hiện nguyên văn trong tài liệu.
QUAN TRỌNG:
- Mọi kết luận phải có căn cứ từ tài liệu.
- Không sử dụng kiến thức bên ngoài tài liệu.
- Không tự tạo dữ kiện, số liệu, tên, quy trình hoặc điều kiện
  không xuất hiện trong tài liệu.
- Nếu một kết luận là suy ra từ nhiều dữ kiện,
  phải bảo đảm kết luận đó phù hợp trực tiếp với các dữ kiện đó.
- Nếu tài liệu không đủ thông tin để trả lời hoặc suy ra,
  hãy nói rõ:
  "Tài liệu không cung cấp đủ thông tin để kết luận."
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
            model="qwen2.5:14b",
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
            model="qwen2.5:14b",
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
            model="qwen2.5:14b",
            messages=[{
                    "role": "user",
                    "content": prompt}]
        )
        answer = response["message"]["content"].strip().upper()
        return answer.startswith("YES")
    except Exception as e:
        print("Lỗi need_history:", e)
        return True 


def _normalize_text(value):
    """
    Chuẩn hóa để so sánh dữ liệu bảng:
    - lowercase
    - gộp khoảng trắng
    - giữ nguyên nội dung hiển thị ở dữ liệu gốc
    """
    return re.sub(r"\s+", " ", str(value or "").strip().lower())

def _search_normalize(value):
    """
    Chuẩn hóa mạnh hơn cho việc MATCH:
    bỏ dấu tiếng Việt + lowercase + chuẩn hóa ký tự.
    Chỉ dùng để so sánh, KHÔNG dùng để thay đổi dữ liệu trả về.
    """
    value = _normalize_text(value)
    value = unicodedata.normalize("NFD", value)
    value = "".join(
        ch for ch in value
        if unicodedata.category(ch) != "Mn"
    )
    value = value.replace("đ", "d")
    value = re.sub(r"[^a-z0-9]+", " ", value)
    return re.sub(r"\s+", " ", value).strip()

def _split_markdown_row(line):
    """
    Tách một dòng Markdown table:
    | A | B | C |
    -> ["A", "B", "C"]
    """
    line = line.strip()

    if not line.startswith("|"):
        return None
    body = line.strip("|")
    return [cell.strip() for cell in body.split("|")]

def _is_markdown_separator(cells):
    if not cells:
        return False

    for cell in cells:
        compact = re.sub(r"\s+", "", cell)
        if not re.fullmatch(r":?-{3,}:?", compact):
            return False
    return True

def _clean_structured_cell(value):
    """
    Chuẩn hóa cell dùng trong Table Engine.
    Không làm thay đổi ý nghĩa dữ liệu.
    """
    return re.sub(
        r"\s+",
        " ",
        str(value or "").strip()
    )

def _join_sparse_cell_parts(parts):
    """
    Ghép các mảnh của cùng một logical cell.

    PDF đôi khi tách một ô thành nhiều cột vật lý:
        NNHCMWAC | O
    phải trở thành:
        NNHCMWACO
    Còn văn bản thông thường:
        Công ty TNHH | ABC
    phải ghép bằng khoảng trắng.
    """
    cleaned = [
        _clean_structured_cell(part)
        for part in parts
        if _clean_structured_cell(part)
    ]
    if not cleaned:
        return ""
    if len(cleaned) == 1:
        return cleaned[0]
    if all(
        re.fullmatch(
            r"[A-Z0-9_.\-/]+",
            part
        )
        for part in cleaned
    ):
        return "".join(cleaned)
    # Văn bản bình thường
    return " ".join(cleaned)

def _compact_sparse_table(headers, data_rows):
    """
    Chuyển bảng có nhiều physical column rỗng
    thành bảng logical column.
    Ví dụ PyMuPDF có thể trả:
        ['', 'Mã', '', '',
         'Tên công ty', '', '',
         'Loại Bill', '']
    trong khi dữ liệu row lại nằm:
        ['HOMEPHONE', '', '',
         'Dịch vụ...', '', '',
         'Viễn thông', '', '']
    Hàm này biến nó thành:
        headers:
        ['Mã', 'Tên công ty', 'Loại Bill']
        row:
        ['HOMEPHONE', 'Dịch vụ...', 'Viễn thông']
    """
    if not headers:
        return [], []
    all_rows = [
        list(headers or [])
    ] + [
        list(row or [])
        for row in data_rows
    ]
    width = max(
        len(row)
        for row in all_rows
    )

    normalized_rows = []
    for row in all_rows:
        padded = (
            list(row)
            + [""] * (width - len(row))
        )
        padded = [
            _clean_structured_cell(cell)
            for cell in padded[:width]
        ]
        normalized_rows.append(
            padded
        )
    raw_headers = normalized_rows[0]
    body_rows = normalized_rows[1:]

    # Vị trí các header thật
    header_positions = [
        index
        for index, value in enumerate(
            raw_headers
        )
        if value
    ]
    if not header_positions:
        return raw_headers, body_rows
    # Nếu bảng vốn đã sạch:
    # A | B | C
    if (
        len(header_positions) == width
        and header_positions == list(
            range(width)
        )
    ):
        return raw_headers, body_rows
    segments = []
    for i, position in enumerate(
        header_positions
    ):
        if i == 0:
            left = 0
        else:
            previous = header_positions[i - 1]
            left = (
                previous + position
            ) // 2 + 1
        if i == len(header_positions) - 1:
            right = width - 1
        else:
            following = header_positions[i + 1]
            right = (
                position + following
            ) // 2
        segments.append(
            (left, right)
        )
    compact_headers = [
        raw_headers[position]
        for position in header_positions
    ]
    compact_rows = []
    for row in body_rows:
        compact_row = []
        for left, right in segments:
            parts = row[
                left:right + 1
            ]
            compact_row.append(
                _join_sparse_cell_parts(
                    parts
                )
            )
        compact_rows.append(
            compact_row
        )
    return (
        compact_headers,
        compact_rows
    )

def _extract_markdown_tables(doc, meta):
    """
    Đọc các bảng Markdown có trong một chunk Chroma.
    Trả về:
    [
        {
            "source": ...,
            "page": ...,
            "chunk_index": ...,
            "headers": [...],
            "rows": [
                {"Cột A": "...", "Cột B": "..."}
            ]
        }
    ]
    """
    lines = str(doc or "").splitlines()
    blocks = []
    current = []

    for line in lines:
        if line.strip().startswith("|"):
            current.append(line)
        else:
            if len(current) >= 2:
                blocks.append(current)
            current = []
    if len(current) >= 2:
        blocks.append(current)

    tables = []
    for block in blocks:
        parsed = [_split_markdown_row(line) for line in block]
        parsed = [row for row in parsed if row is not None]
        if len(parsed) < 2:
            continue

        # Tìm header + separator Markdown.
        header_index = None
        for i in range(len(parsed) - 1):
            if _is_markdown_separator(parsed[i + 1]):
                header_index = i
                break
        if header_index is None:
            continue
        raw_headers = [
            _clean_structured_cell(h)
            for h in parsed[header_index]
        ]
        if not any(raw_headers):
            continue
        raw_data_rows = []
        for cells in parsed[header_index + 2:]:
            if _is_markdown_separator(
                cells
            ):
                continue
            if cells and all(
                (not c)
                or re.fullmatch(
                    r"=+",
                    c.replace(" ", "")
                )
                for c in cells
            ):
                continue
            raw_data_rows.append(
                cells
            )
        headers, compact_rows = (
            _compact_sparse_table(
                raw_headers,
                raw_data_rows
            )
        )
        if not headers:
            continue
        rows = []
        for cells in compact_rows:
            if len(cells) < len(headers):
                cells = (
                    cells
                    + [""] * (
                        len(headers)
                        - len(cells)
                    )
                )
            elif len(cells) > len(headers):
                cells = cells[
                    :len(headers)
                ]
            row = {
                headers[i]:
                    _clean_structured_cell(
                        cells[i]
                    )
                for i in range(
                    len(headers)
                )
            }
            if any(row.values()):
                rows.append(
                    row
                )
            
        if rows:
            tables.append({
                "source": meta.get("source"),
                "page": meta.get("page"),
                "chunk_index": meta.get("chunk_index"),
                "headers": headers,
                "rows": rows
            })
    return tables

def _header_signature(headers):
    return tuple(
        _search_normalize(h)
        for h in headers
        if _search_normalize(h)
    )

def _safe_int(value, default=None):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _table_context_signature(meta):
    """
    Lấy phần ngữ cảnh/tên logic của bảng.
    Ví dụ:
    "Danh sách mã khu vực (Bảng 2 - Phần 1/3 - Trang 40)"
        ->
    "danh sach ma khu vuc"
    Không phụ thuộc số Bảng / Phần / Trang.
    """
    title = str(meta.get("title", "") or "").strip()
    title = re.sub(
        r"\s*\(bảng[^)]*\)\s*$",
        "",
        title,
        flags=re.IGNORECASE
    )
    return _search_normalize(title)

def _collect_table_groups(documents, metadatas):
    """
    Gom các table chunk thành logical table.

    Nguyên tắc:
    - Cùng source
    - Cùng header
    - PDF: các trang liên tiếp được xem là cùng một bảng
    - DOCX: các chunk liên tiếp được xem là cùng một bảng

    Context/title KHÔNG được dùng làm khóa cứng giữa các trang,
    vì bảng kéo dài nhiều trang có thể có title/context khác nhau.

    Context chỉ dùng để tránh gộp nhầm hai bảng khác nhau
    nằm trên CÙNG một trang.

    Hỗ trợ:
    - type = table
    - type = vision nếu Vision OCR ra Markdown table
    """

    fragments = []
    for doc, meta in zip(documents, metadatas):
        if meta.get("type") not in {"table", "vision"}:
            continue
        tables = _extract_markdown_tables(
            doc,
            meta
        )
        for table in tables:
            header_signature = _header_signature(
                table["headers"]
            )
            if not header_signature:
                continue
            page = _safe_int(
                meta.get("page"),
                default=-1
            )
            chunk_index = _safe_int(
                meta.get("chunk_index"),
                default=-1
            )
            context_signature = _table_context_signature(
                meta
            )
            fragments.append({
                "source": table["source"],
                "headers": table["headers"],
                "header_signature": header_signature,
                "context_signature": context_signature,
                "page": page,
                "chunk_index": chunk_index,
                "rows": table["rows"]
            })
    if not fragments:
        return []

    buckets = {}
    for fragment in fragments:
        key = (
            fragment["source"],
            fragment["header_signature"]
        )
        buckets.setdefault(
            key,
            []
        ).append(fragment)
    result = []

    def contexts_compatible(a, b):
        a = str(a or "").strip()
        b = str(b or "").strip()
        if not a or not b:
            return True
        if a == b:
            return True
        if a in b or b in a:
            return True
        similarity = SequenceMatcher(
            None,
            a,
            b
        ).ratio()
        return similarity >= 0.65

    for bucket_fragments in buckets.values():
        bucket_fragments.sort(
            key=lambda item: (
                (
                    item["page"]
                    if item["page"] >= 1
                    else 10**9
                ),
                (
                    item["chunk_index"]
                    if item["chunk_index"] >= 0
                    else 10**9
                )
            )
        )
        runs = []
        current_run = []
        for fragment in bucket_fragments:
            if not current_run:
                current_run = [fragment]
                continue
            previous = current_run[-1]
            previous_page = previous["page"]
            current_page = fragment["page"]
            previous_index = previous["chunk_index"]
            current_index = fragment["chunk_index"]

            same_run = False
            if (
                previous_page >= 1
                and current_page >= 1
            ):
                page_gap = (
                    current_page
                    - previous_page
                )
                if page_gap == 0:
                    if (
                        previous_index >= 0
                        and current_index >= 0
                    ):
                        index_gap = (
                            current_index
                            - previous_index
                        )
                        # Chunk phải nằm gần nhau
                        if index_gap <= 1:
                            # Nếu context tương thích
                            # -> cùng bảng
                            if contexts_compatible(
                                previous["context_signature"],
                                fragment["context_signature"]
                            ):
                                same_run = True
                elif page_gap == 1:

                    same_run = True
                # Cách từ 2 trang trở lên
                else:
                    same_run = False
            else:
                if (
                    previous_index >= 0
                    and current_index >= 0
                ):
                    index_gap = (
                        current_index
                        - previous_index
                    )
                    if (
                        index_gap <= 1
                        and contexts_compatible(
                            previous["context_signature"],
                            fragment["context_signature"]
                        )
                    ):
                        same_run = True
            if same_run:
                current_run.append(
                    fragment
                )
            else:
                runs.append(
                    current_run
                )
                current_run = [
                    fragment
                ]
        if current_run:
            runs.append(
                current_run
            )
        for run in runs:
            first = run[0]
            group = {
                "source": first["source"],
                "headers": first["headers"],
                "rows": [],
                "pages": []
            }
            seen_rows = set()
            seen_pages = set()
            for fragment in run:
                if fragment["page"] >= 1:
                    seen_pages.add(
                        fragment["page"]
                    )
                for row in fragment["rows"]:
                    row_key = tuple(
                        _normalize_text(
                            row.get(
                                header,
                                ""
                            )
                        )
                        for header in group["headers"]
                    )
                    if row_key in seen_rows:
                        continue
                    seen_rows.add(
                        row_key
                    )
                    group["rows"].append(
                        row
                    )
            group["pages"] = sorted(
                seen_pages
            )
            if group["rows"]:
                result.append(
                    group
                )
    return result

def _build_table_catalog(table_groups):
    """
    Chỉ đưa cho Query Planner:
    - tên file
    - tên cột THỰC TẾ
    - vài dòng mẫu
    Planner không được tự nghĩ ra tên cột.
    """
    catalog = []

    for i, group in enumerate(table_groups):
        samples = group["rows"][:3]
        catalog.append({
            "table_id": i,
            "source": group["source"],
            "headers": group["headers"],
            "sample_rows": samples
        })
    return catalog

def _extract_json_object(text):
    """
    Lấy JSON object kể cả khi model vô tình bọc ```json ... ```.
    """
    raw = str(text or "").strip()
    raw = re.sub(
        r"^```(?:json)?\s*",
        "",
        raw,
        flags=re.IGNORECASE
    )
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(0))
    except Exception:
        return None

def _plan_structured_table_operation(question, table_groups):
    """
    Query Planner dành riêng cho thao tác DỮ LIỆU CÓ CẤU TRÚC.

    Chỉ nhận các thao tác có thể thực hiện deterministic:
    - LIST
    - FILTER_LIST
    - COUNT
    - FILTER_COUNT
    LOOKUP / EXPLAIN / COMPARE / ANALYZE trả NONE
    để pipeline Qwen hiện tại xử lý.
    """
    if not table_groups:
        return None

    catalog = _build_table_catalog(table_groups)
    prompt = f"""
Bạn là Query Planner cho dữ liệu bảng.

NHIỆM VỤ:
Phân tích câu hỏi và quyết định có thể xử lý bằng thao tác bảng deterministic hay không.

CHỈ ĐƯỢC chọn một trong:
- LIST:
  Người dùng muốn liệt kê dữ liệu của một hoặc nhiều cột, không có điều kiện lọc.
- FILTER_LIST:
  Người dùng muốn lọc các hàng theo điều kiện rồi liệt kê một hoặc nhiều cột.
- COUNT:
  Người dùng muốn đếm số hàng / số giá trị, không có điều kiện lọc.
- FILTER_COUNT:
  Người dùng muốn lọc theo điều kiện rồi đếm.
- NONE:
  Câu hỏi cần giải thích, suy luận, so sánh, mô tả chi tiết,
  lookup một giá trị cụ thể, hoặc không phù hợp với 4 thao tác trên.

QUY TẮC:
1. CHỈ sử dụng table_id và tên cột xuất hiện trong DANH SÁCH BẢNG.
2. Không tự nghĩ ra tên cột.
3. Nếu cách gọi của người dùng gần nghĩa với một tên cột thật,
   hãy ánh xạ sang đúng tên cột thật.
4. FILTER_LIST/FILTER_COUNT phải xác định filter_column và filter_value.
5. Với điều kiện dạng "là / bằng / thuộc loại", ưu tiên operator = "equals".
6. Với điều kiện dạng "có chứa / chứa", dùng operator = "contains".
7. LIST/FILTER_LIST phải xác định return_columns.
8. Nếu không chắc chắn bảng/cột nào thì trả operation = "NONE".
9. Chỉ trả về JSON hợp lệ. Không Markdown. Không giải thích.

JSON schema:
{{
  "operation": "LIST|FILTER_LIST|COUNT|FILTER_COUNT|NONE",
  "table_id": 0,
  "filter_column": null,
  "operator": null,
  "filter_value": null,
  "return_columns": []
}}

DANH SÁCH BẢNG:
{json.dumps(catalog, ensure_ascii=False)}
CÂU HỎI:
{question}
"""
    try:
        response = chat(
            model="qwen2.5:14b",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ]
        )
        plan = _extract_json_object(
            response["message"]["content"]
        )
        print("[structured-plan]", plan)
        return plan
    except Exception as e:
        print("Lỗi structured table planner:", e)
        return None

def _resolve_column(requested, headers):
    """
    Ánh xạ tên cột planner trả về vào header thật.
    Không cho phép planner dùng một cột hoàn toàn không tồn tại.
    """
    if not requested:
        return None
    requested_norm = _search_normalize(requested)
    if not requested_norm:
        return None
    # 1. Exact normalized match
    for header in headers:
        if _search_normalize(header) == requested_norm:
            return header
    # 2. Một chuỗi chứa chuỗi kia
    contains_candidates = []
    for header in headers:
        header_norm = _search_normalize(header)
        if (
            requested_norm in header_norm
            or header_norm in requested_norm
        ):
            contains_candidates.append(header)
    if len(contains_candidates) == 1:
        return contains_candidates[0]
    # 3. Fuzzy nhẹ
    best_header = None
    best_score = 0.0
    for header in headers:
        score = SequenceMatcher(
            None,
            requested_norm,
            _search_normalize(header)
        ).ratio()
        if score > best_score:
            best_score = score
            best_header = header
    if best_score >= 0.72:
        return best_header
    return None

def _compare_cell(cell_value, operator, target_value):
    cell = _search_normalize(cell_value)
    target = _search_normalize(target_value)
    if not target:
        return False
    operator = _normalize_text(operator or "equals")
    if operator == "contains":
        return target in cell
    if operator == "not_equals":
        return cell != target
    # mặc định equals
    return cell == target

def _execute_structured_plan(plan, table_groups):
    """
    Thực thi deterministic.
    Trả None nếu plan không hợp lệ để fallback sang Qwen.
    """
    if not isinstance(plan, dict):
        return None

    operation = str(
        plan.get("operation", "NONE")
    ).upper()

    allowed = {
        "LIST",
        "FILTER_LIST",
        "COUNT",
        "FILTER_COUNT"
    }

    if operation not in allowed:
        return None
    try:
        table_id = int(plan.get("table_id"))
    except (TypeError, ValueError):
        return None
    if table_id < 0 or table_id >= len(table_groups):
        return None

    group = table_groups[table_id]
    headers = group["headers"]
    rows = list(group["rows"])

    print("\n" + "=" * 80)
    print("DEBUG STRUCTURED TABLE")
    print("=" * 80)

    print("TABLE ID:", table_id)
    print("HEADERS:", headers)
    print("TOTAL ROWS:", len(rows))

    print("\nFIRST 10 ROWS:")
    for i, row in enumerate(rows[:10], start=1):
        print(i, row)

    print("\nLAST 10 ROWS:")
    for i, row in enumerate(rows[-10:], start=1):
        print(i, row)

    print("=" * 80)


    if not rows:
        return None

    # =========================== FILTER ============================
    if operation in {
        "FILTER_LIST",
        "FILTER_COUNT"
    }:
        filter_column = _resolve_column(
            plan.get("filter_column"),
            headers
        )
        if not filter_column:
            print(
                "[structured] Không resolve được filter_column:",
                plan.get("filter_column")
            )
            return None
        filter_value = plan.get("filter_value")
        operator = plan.get("operator") or "equals"

        print("FILTER COLUMN:", filter_column)
        print("FILTER VALUE :", filter_value)
        print("OPERATOR     :", operator)
        print("\nCÁC GIÁ TRỊ CÓ TRONG CỘT FILTER:")
        unique_values = []
        for row in rows:
            value = row.get(filter_column, "")
            if value not in unique_values:
                unique_values.append(value)
        for value in unique_values:
            print(
                repr(value),
                "=> NORMALIZED:",
                repr(_search_normalize(value))
            )
        print(
            "TARGET NORMALIZED:",
            repr(_search_normalize(filter_value))
        )

        rows = [
            row
            for row in rows
            if _compare_cell(
                row.get(filter_column, ""),
                operator,
                filter_value
            )
        ]

    # ===================== COUNT ==============================
    if operation in {
        "COUNT",
        "FILTER_COUNT"
    }:
        if operation == "FILTER_COUNT":
            return f"Có {len(rows)} kết quả phù hợp."

        return f"Có {len(rows)} bản ghi."

    # =================== LIST ==================================
    requested_columns = plan.get(
        "return_columns",
        []
    )
    if not isinstance(requested_columns, list):
        return None
    resolved_columns = []
    for requested in requested_columns:
        resolved = _resolve_column(
            requested,
            headers
        )
        if (
            resolved
            and resolved not in resolved_columns
        ):
            resolved_columns.append(resolved)
    # Planner không chỉ ra được cột -> fallback Qwen.
    if not resolved_columns:
        return None
    if not rows:
        return (
            "Tôi không tìm thấy bản ghi nào trong bảng "
            "thỏa điều kiện được yêu cầu."
        )
    output_lines = []
    if len(resolved_columns) == 1:
        column = resolved_columns[0]
        for row in rows:
            value = row.get(column, "").strip()
            if value:
                output_lines.append(
                    f"- {value}"
                )
    else:
        for row in rows:
            parts = []
            for column in resolved_columns:
                value = row.get(
                    column,
                    ""
                ).strip()
                if value:
                    parts.append(
                        f"{column}: {value}"
                    )
            if parts:
                output_lines.append(
                    "- " + " | ".join(parts)
                )
    deduped = []
    seen = set()

    for line in output_lines:
        key = _normalize_text(line)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(line)

    if not deduped:
        return (
            "Tôi không tìm thấy bản ghi nào trong bảng "
            "thỏa điều kiện được yêu cầu."
        )
    return "\n".join(deduped)

def try_structured_table_answer(question, documents, metadatas):
    """
    Thử trả lời bằng table engine deterministic.
    Nếu không phù hợp -> None
    để ask_ai tiếp tục dùng Qwen như hiện tại.
    """
    table_groups = _collect_table_groups(
        documents,
        metadatas
    )
    if not table_groups:
        return None
    plan = _plan_structured_table_operation(
        question,
        table_groups
    )
    return _execute_structured_plan(
        plan,
        table_groups
    )

def ask_ai(question, session_id):
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

    # Không tìm thấy document
    if len(documents) == 0:
        answer = "Tôi không tìm thấy thông tin trong tài liệu."
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", answer)
        return answer

    structured_answer = try_structured_table_answer(
        question,
        documents,
        metadatas
    )
    if structured_answer is not None:
        print("===== STRUCTURED TABLE ENGINE =====")
        print(structured_answer)
        add_message(session_id, "user", question)
        add_message(session_id, "assistant", structured_answer)
        return structured_answer

    history_text = ""
    if need_history(question):
        history = get_history(session_id)
        for msg in history:
            history_text += f"{msg['role']}:\n{msg['content']}\n\n"

    content = ""
    for i, (doc, meta) in enumerate(zip(documents, metadatas), start=1):
        chunk_type = meta.get("type", "text")
        if chunk_type == "table":
            data_type = "BẢNG DỮ LIỆU"
        elif chunk_type == "vision":
            if "|" in doc:
                data_type = "BẢNG DỮ LIỆU OCR TỪ HÌNH ẢNH"
            else:
                data_type = "SƠ ĐỒ / HÌNH ẢNH / VISION"
        else:
            data_type = "VĂN BẢN"
        content += f"""
    ====================
    Tài liệu {i}

    Tên file:
    {meta["source"]}

    Tiêu đề:
    {meta.get("title")}

    Loại:
    {data_type}

    Trang:
    {meta.get("page")}

    Các cột:
    {meta.get("headers", "")}

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
    1. Chỉ sử dụng dữ kiện có trong DANH SÁCH TÀI LIỆU.
    2. Không yêu cầu đáp án phải được viết nguyên văn trong tài liệu.
    3. Được phép tổng hợp và suy luận từ một hoặc nhiều đoạn tài liệu
    nếu kết luận có thể rút ra hợp lý từ các dữ kiện đã cung cấp.
    4. Phân biệt rõ:
        - Thông tin trực tiếp: tài liệu ghi rõ.
        - Thông tin suy ra: kết luận từ các dữ kiện trong tài liệu.
        - Không đủ thông tin: tài liệu không có đủ dữ kiện để kết luận.
    5. Không được sử dụng kiến thức bên ngoài,
    không tự tạo dữ kiện, con số, tên, quy trình hoặc điều kiện
    mà tài liệu không cung cấp.
    6. Nếu trong tài liệu có BẢNG DỮ LIỆU thì:
        - Luôn ưu tiên sử dụng bảng trước.
        - Mỗi hàng là một bản ghi, dòng đầu tiên là tên cột.
        - Khi trả lời phải đối chiếu đúng hàng và đúng cột.
    7. Nếu tài liệu có Loại = SƠ ĐỒ / HÌNH ẢNH / VISION thì:
        - Đây là nội dung đã được model Vision đọc từ hình ảnh hoặc sơ đồ.
        - Phải giữ đúng các thành phần và vai trò.
        - Phải giữ đúng thứ tự luồng xử lý.
        - Nếu có các nhánh điều kiện thì phải phân biệt từng nhánh.
        - Không được tự tạo thêm bước không có trong nội dung Vision.
    8. Phải trả lời ĐẦY ĐỦ TỪNG Ý trong mục "CÁC Ý CẦN TRẢ LỜI" ở trên,
       không được chỉ trả lời ý đầu tiên rồi dừng lại.
    9. Với từng ý:
        - Nếu tài liệu trả lời trực tiếp → dùng thông tin trực tiếp.
        - Nếu không ghi trực tiếp nhưng có đủ dữ kiện → tổng hợp hoặc suy ra.
        - Chỉ khi không có đủ dữ kiện để trả lời hoặc suy ra mới ghi:
            "Tài liệu không cung cấp đủ thông tin để kết luận."
    10. Nếu câu hỏi yêu cầu liệt kê thì phải liệt kê đầy đủ.
    11. Nếu tài liệu chứa sơ đồ, bảng có nhiều vai trò thì phải xác định đúng thông tin của từng vai trò
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
        - Chỉ sử dụng DANH SÁCH TÀI LIỆU làm căn cứ.
        - Ưu tiên trả lời trực tiếp nếu tài liệu ghi rõ.
        - Nếu câu trả lời không xuất hiện nguyên văn nhưng có thể
        rút ra từ các dữ kiện trong tài liệu, hãy suy luận và trả lời.
        - Có thể kết hợp nhiều đoạn, nhiều bảng và nhiều trang
        để hình thành câu trả lời hoàn chỉnh.
        - Không được dùng kiến thức bên ngoài tài liệu.
        - Không được biến một khả năng thành một sự thật nếu tài liệu
        không đủ căn cứ.
        - Chỉ khi tài liệu không đủ dữ kiện để trả lời hoặc suy ra,
        mới trả lời:
        "Tài liệu không cung cấp đủ thông tin để kết luận."
    ====================
    TRẢ LỜI:
    """
    print("=" * 80)
    print(prompt)
    print("=" * 80)

    print("===== QWEN2.5:14B ĐANG SUY LUẬN =====")
    response = chat(
        model="qwen2.5:14b",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ]
    )
    print("===== /bye/2.5:14B XONG =====")
    answer = response["message"]["content"]
    add_message(session_id, "user", question)
    add_message(session_id, "assistant", answer)

    return answer