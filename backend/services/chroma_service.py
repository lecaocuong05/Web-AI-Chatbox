import chromadb
import uuid
import re
from collections import defaultdict

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(
    name = "company_documents"
)

# Các từ dừng (stopword) tiếng Việt phổ biến — không mang nhiều ý nghĩa phân
# biệt, nếu tính điểm ngang hàng với từ khoá thật (VD tên API, tên bảng) sẽ
# làm nhiễu kết quả rerank (chunk không liên quan vẫn lọt top nhờ trùng
# những từ rất phổ biến này). Danh sách không cần tuyệt đối đầy đủ, chỉ cần
# loại bỏ phần lớn từ nối/từ chức năng thường gặp.
VIETNAMESE_STOPWORDS = {
    "hãy", "nêu", "cho", "biết", "vui", "lòng", "làm", "ơn",
    "các", "những", "mọi", "toàn", "bộ", "cả",
    "là", "của", "và", "có", "được", "để", "trong", "với", "này", "đó", "kia",
    "khi", "nếu", "thì", "sẽ", "đã", "đang", "về", "như", "cũng", "hay", "hoặc",
    "một", "hai", "ba", "hai", "hoặc",
    "thông", "tin", "liên", "quan", "đến", "gì", "sao", "nào", "ai", "bao", "nhiêu",
    "rất", "vì", "do", "nên", "mà", "tại", "ra", "vào", "lên", "xuống",
    "theo", "từ", "trên", "dưới", "chỉ", "còn", "hết", "nữa", "lại", "đây",
    "tôi", "bạn", "mình", "ta", "chúng", "họ", "nó",
    "không", "chưa", "phải", "được", "cần", "muốn", "nhé", "ạ", "à", "nha",
}


def _extract_keywords(query: str):
    """Tách từ khoá từ câu hỏi, loại bỏ stopword để chỉ giữ lại từ mang nghĩa."""
    tokens = re.findall(r"\w+", query.lower())
    return [t for t in tokens if t not in VIETNAMESE_STOPWORDS and len(t) > 1]


def add_document(text,source,title,chunk_index=1,total_chunks=1,chunk_type="text",page=None,headers=None):
    clean_headers = [
        str(h).strip()
        for h in (headers or [])
        if h is not None and str(h).strip()
    ]
    metadata_page = (
        int(page)
        if page is not None
        else -1
    )

    print("=" * 60)
    print("ĐANG THÊM CHUNK")
    print(f"Source       : {source}")
    print(f"Title        : {title}")
    print(f"Chunk        : {chunk_index}/{total_chunks}")
    print(f"Type         : {chunk_type}")
    print(f"Page         : {metadata_page}")
    print(f"Headers      : {clean_headers}")
    print(f"Length       : {len(text)}")
    print("=" * 60)

    collection.add(
        ids=[str(uuid.uuid4())],
        documents=[text],
        metadatas=[{
            "source": source,
            "title": title,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks,
            "type": chunk_type,
            "page": metadata_page,
            "headers": ", ".join(clean_headers)
        }]
    )

def _normalize_text(text):
    return re.sub(
        r"\s+",
        " ",
        str(text or "").lower()
    ).strip()


def rerank_documents(query, documents, metadatas):
    """
    Rerank tổng quát:
    - Ưu tiên document chứa ĐỦ các ý trong câu hỏi.
    - Ưu tiên từ xuất hiện trong header.
    - Ưu tiên cụm từ liên tiếp.
    - Không để một từ lặp nhiều lần áp đảo toàn bộ ranking.
    """
    keywords = _extract_keywords(query)
    # bỏ từ trùng
    keywords = list(dict.fromkeys(keywords))
    scores = []
    for doc, meta in zip(documents, metadatas):
        text = _normalize_text(doc)
        title = _normalize_text(
            meta.get("title", ""))
        headers = _normalize_text(
            meta.get("headers", ""))
        combined = f"{title} {headers} {text}"
        if not keywords:
            scores.append(0)
            continue

# ============ ĐỘ PHỦ TỪ KHÓA =====================
        matched = [
            kw
            for kw in keywords
            if kw in combined
        ]
        coverage = len(matched) / len(keywords)
        score = coverage * 100

# ============== MATCH TRONG HEADER ===============
        for kw in keywords:
            if kw in headers:
                score += 8
            if kw in title:
                score += 4

# ============= CỤM TỪ LIÊN TIẾP ==================
        for i in range(len(keywords) - 1):
            phrase = (
                keywords[i]
                + " "
                + keywords[i + 1]
            )
            if phrase in headers:
                score += 15
            elif phrase in text:
                score += 8

# ================ TẦN SUẤT - nhưng GIỚI HẠN ===========
        for kw in keywords:
            score += min(
                text.count(kw),
                2
            )
        scores.append(score)
    return scores

# ====================================================
def _hybrid_rank(query, documents, metadatas, rrf_k=60):
    if not documents:
        return []
    keyword_scores = rerank_documents(
        query,
        documents,
        metadatas
    )
    keyword_rank = sorted(
        range(len(documents)),
        key=lambda i: keyword_scores[i],
        reverse=True
    )
    keyword_rank_of = {
        idx: rank
        for rank, idx in enumerate(keyword_rank)
    }
    # Điểm keyword lớn nhất để normalize về 0 → 1
    max_keyword_score = (
        max(keyword_scores)
        if keyword_scores
        else 0
    )
    fused = []
    for candidate_rank, (doc, meta) in enumerate(
        zip(documents, metadatas)
    ):
        kw_rank = keyword_rank_of[candidate_rank]
        # Reciprocal Rank Fusion
        rrf_score = (
            1 / (rrf_k + candidate_rank)
            +
            1 / (rrf_k + kw_rank)
        )

# ======== THÊM ĐỘ KHỚP KEYWORD THỰC TẾ ===========
        if max_keyword_score > 0:
            normalized_keyword_score = (
                keyword_scores[candidate_rank]
                / max_keyword_score
            )
            rrf_score += (
                normalized_keyword_score * 0.025
            )
        fused.append(
            (
                rrf_score,
                keyword_scores[candidate_rank],
                doc,
                meta
            )
        )
    fused.sort(
        key=lambda x: x[0],
        reverse=True
    )
    return fused

def _wants_full_table(query: str) -> bool:
    """
    Nhận diện câu hỏi có ý định lấy toàn bộ / danh sách dữ liệu
    Chỉ nhận diện Ý ĐỊNH của câu hỏi
    Không hard-code nội dung tài liệu như:
    bill, chung cư, mã khu vực, nhân viên ...
    """
    q = re.sub(
        r"\s+",
        " ",
        (query or "").lower()
    ).strip()

    direct_phrases = [
        "toàn bộ", "tất cả", "liệt kê", "danh sách", "đầy đủ"
    ]
    if any(
        phrase in q
        for phrase in direct_phrases
    ):
        return True

    patterns = [
        r"\bcó\s+(?:những|các)\b.+\bnào\b",
        r"\bnhững\b.+\bnào\b",
        r"\bcác\b.+\bnào\b",
        r"\bgồm\b.+\b(?:gì|nào)\b",
    ]
    return any(
        re.search(pattern, q)
        for pattern in patterns
    )

def _headers_signature(meta):
    """
    Chuẩn hóa header để xác định các table chunk
    có cùng cấu trúc bảng.

    Ví dụ:
    "Mã khu vực, Tên khu vực"
    sẽ giống nhau giữa trang 39, 40, 41.
    """
    raw_headers = meta.get("headers", "")
    if isinstance(raw_headers, (list, tuple)):
        values = []
        for value in raw_headers:
            value = re.sub(
                r"\s+",
                " ",
                str(value or "").strip().lower()
            )
            if value:
                values.append(value)
        return " | ".join(values)
    return re.sub(
        r"\s+",
        " ",
        str(raw_headers or "").strip().lower()
    )

def _expand_full_table_results(query, selected):
    """
    Nếu câu hỏi yêu cầu lấy toàn bộ/danh sách:

    1. Khi một table chunk lọt vào kết quả tìm kiếm
    2. Xác định cấu trúc bảng bằng headers
    3. Tìm tất cả table chunk:
       - cùng source
       - cùng headers
       - nằm trên các trang liên tiếp
    4. Lấy toàn bộ các chunk đó
    5. Sắp xếp theo page + chunk_index

    Không phụ thuộc title "Bảng 1", "Bảng 2"...
    nên hỗ trợ bảng kéo dài nhiều trang.
    """
    if not _wants_full_table(query):
        return selected
    expanded = []
    added = set()
    expanded_groups = set()
    source_cache = {}

    for item in selected:
        rrf, kw, doc, meta = item
        source = meta.get("source")
        chunk_index = meta.get("chunk_index")
        unique_key = (source, chunk_index)

        if meta.get("type") != "table":
            if unique_key not in added:
                expanded.append(item)
                added.add(unique_key)
            continue

        header_signature = _headers_signature(meta)
        try:
            seed_page = int(
                meta.get("page", -1)
            )
        except (TypeError, ValueError):
            seed_page = -1
        if (
            not source
            or not header_signature
            or seed_page < 1
        ):
            if unique_key not in added:
                expanded.append(item)
                added.add(unique_key)
            continue
        if source not in source_cache:
            source_cache[source] = collection.get(
                where={
                    "source": source
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )
        file_data = source_cache[source]
        matching_tables = []
        matching_pages = set()
        for sibling_doc, sibling_meta in zip(
            file_data["documents"],
            file_data["metadatas"]
        ):
            if sibling_meta.get("type") != "table":
                continue
            sibling_header_signature = (
                _headers_signature(
                    sibling_meta
                )
            )
            if (
                sibling_header_signature
                != header_signature
            ):
                continue

            try:
                sibling_page = int(
                    sibling_meta.get(
                        "page",
                        -1
                    )
                )
            except (TypeError, ValueError):
                continue

            if sibling_page < 1:
                continue

            try:
                sibling_index = int(
                    sibling_meta.get(
                        "chunk_index",
                        0
                    )
                )
            except (TypeError, ValueError):
                sibling_index = 0
            matching_tables.append(
                (
                    sibling_page,
                    sibling_index,
                    sibling_doc,
                    sibling_meta
                )
            )
            matching_pages.add(
                sibling_page
            )
        if seed_page not in matching_pages:

            if unique_key not in added:
                expanded.append(item)
                added.add(unique_key)
            continue
        connected_pages = {
            seed_page
        }
        page = seed_page - 1
        while page in matching_pages:
            connected_pages.add(page)
            page -= 1
        page = seed_page + 1
        while page in matching_pages:
            connected_pages.add(page)
            page += 1
        group_key = (
            source,
            header_signature,
            min(connected_pages),
            max(connected_pages)
        )

        # Nếu bảng này đã được expand ở một result trước
        if group_key in expanded_groups:
            continue
        expanded_groups.add(
            group_key
        )
        sibling_parts = [
            item
            for item in matching_tables
            if item[0] in connected_pages
        ]
        sibling_parts.sort(
            key=lambda x: (
                x[0],
                x[1]
            )
        )
        for (sibling_page, sibling_index, sibling_doc, sibling_meta) in sibling_parts:
            sibling_key = (
                sibling_meta.get("source"),
                sibling_meta.get("chunk_index")
            )
            if sibling_key in added:
                continue
            sibling_kw = rerank_documents(
                query,
                [sibling_doc],
                [sibling_meta]
            )[0]
            expanded.append(
                (
                    rrf,
                    sibling_kw,
                    sibling_doc,
                    sibling_meta
                )
            )
            added.add(
                sibling_key
            )

    return expanded

def _global_keyword_candidates(query, limit=20):
    """
    Tìm keyword trên TOÀN BỘ Chroma.
    Không phụ thuộc loại tài liệu hay lĩnh vực.
    """
    data = collection.get(
        include=[
            "documents",
            "metadatas"
        ]
    )
    documents = data.get("documents", [])
    metadatas = data.get("metadatas", [])
    if not documents:
        return []
    scores = rerank_documents(
        query,
        documents,
        metadatas
    )
    ranked_indexes = sorted(
        range(len(documents)),
        key=lambda i: scores[i],
        reverse=True
    )
    results = []

    for i in ranked_indexes:
        # Không có một từ khóa thực nào khớp thì dừng
        if scores[i] <= 0:
            continue
        results.append((
            documents[i],
            metadatas[i],
            scores[i]
        ))
        if len(results) >= limit:
            break
    return results


def _safe_int(value, default=-1):
    try:
        return int(value)
    except (TypeError, ValueError):
        return default

def _candidate_key(meta):
    return (
        meta.get("source"),
        meta.get("chunk_index")
    )

def _same_location(meta_a, meta_b):
    """
    Hai chunk được coi là gần nhau khi:
    - PDF: cùng source + cùng page.
    - DOCX/TXT không có page:
      cùng source + chunk_index cách nhau <= 2.
    """
    if meta_a.get("source") != meta_b.get("source"):
        return False

    page_a = _safe_int(
        meta_a.get("page"),
        -1
    )
    page_b = _safe_int(
        meta_b.get("page"),
        -1
    )

    if page_a >= 1 and page_b >= 1:
        return page_a == page_b

    idx_a = _safe_int(
        meta_a.get("chunk_index"),
        -10**9
    )
    idx_b = _safe_int(
        meta_b.get("chunk_index"),
        10**9
    )
    return abs(idx_a - idx_b) <= 2

def _add_cross_type_context_candidates(
    query,
    candidate_docs,
    candidate_metas,
    max_neighbors_per_anchor=5
):
    """
    Nối TEXT -> TABLE/VISION ở runtime.

    Nếu query match caption/tên section trong TEXT
    nhưng bảng không chứa tên đó, table/vision cùng vị trí
    vẫn được đưa vào candidate để rerank/Qwen xử lý.

    Không hard-code tên bảng hay lĩnh vực.
    """
    if not candidate_docs:
        return (
            candidate_docs,
            candidate_metas
        )
    docs = list(candidate_docs)
    metas = [
        dict(meta)
        for meta in candidate_metas
    ]
    key_to_pos = {
        _candidate_key(meta): i
        for i, meta in enumerate(metas)
    }
    source_cache = {}
    anchors = list(
        zip(
            candidate_docs,
            candidate_metas
        )
    )
    for anchor_doc, anchor_meta in anchors:
        anchor_type = anchor_meta.get(
            "type",
            "text"
        )
        if anchor_type != "text":
            continue
        anchor_score = rerank_documents(
            query,
            [anchor_doc],
            [anchor_meta]
        )[0]
        source = anchor_meta.get("source")
        if not source:
            continue
        if source not in source_cache:
            source_cache[source] = collection.get(
                where={
                    "source": source
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )
        file_data = source_cache[source]
        neighbors = []
        for neighbor_doc, neighbor_meta in zip(
            file_data.get("documents", []),
            file_data.get("metadatas", [])
        ):
            neighbor_type = neighbor_meta.get(
                "type",
                "text"
            )
            if neighbor_type not in {
                "table",
                "vision"
            }:
                continue
            if not _same_location(
                anchor_meta,
                neighbor_meta
            ):
                continue
            direct_score = rerank_documents(
                query,
                [neighbor_doc],
                [neighbor_meta]
            )[0]
            combined_score = (
                direct_score
                + min(anchor_score, 120) * 0.35
            )
            neighbors.append(
                (
                    combined_score,
                    direct_score,
                    neighbor_doc,
                    neighbor_meta
                )
            )
        neighbors.sort(
            key=lambda x: (
                x[0],
                x[1]
            ),
            reverse=True
        )
        for ( _, _, neighbor_doc, neighbor_meta
        ) in neighbors[:max_neighbors_per_anchor]:
            key = _candidate_key(
                neighbor_meta
            )
            contextual_doc = (
                "Ngữ cảnh liên quan gần bảng:\n"
                + str(anchor_doc).strip()
                + "\n========================\n"
                + str(neighbor_doc).strip()
            )
            if key in key_to_pos:
                pos = key_to_pos[key]
                if (
                    "Ngữ cảnh liên quan gần bảng:"
                    not in docs[pos]
                ):
                    docs[pos] = contextual_doc
            else:
                key_to_pos[key] = len(docs)
                docs.append(contextual_doc)
                metas.append(
                    dict(neighbor_meta)
                )
    return docs, metas

def search_document(query, top_k=6):
    SEARCH_MULTIPLIER = 4

    available = collection.count()
    if available == 0:
        return {
            "documents": [[]],
            "metadatas": [[]],
            "keyword_scores": [[]]
        }

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k * SEARCH_MULTIPLIER, 30, available),
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )
    hit_docs = result["documents"][0]
    hit_metas = result["metadatas"][0]
    hit_distances = result["distances"][0]

    keyword_candidates = _global_keyword_candidates(
        query,
        limit=min(top_k * 3, 30)
    )
    candidate_docs = list(hit_docs)
    candidate_metas = list(hit_metas)
    seen_candidates = set()
    for meta in candidate_metas:
        seen_candidates.add((
            meta.get("source"),
            meta.get("chunk_index")
        ))
    for doc, meta, kw_score in keyword_candidates:
        key = (
            meta.get("source"),
            meta.get("chunk_index")
        )
        if key in seen_candidates:
            continue
        candidate_docs.append(doc)
        candidate_metas.append(meta)
        seen_candidates.add(key)

    candidate_docs, candidate_metas = (
        _add_cross_type_context_candidates(
            query,
            candidate_docs,
            candidate_metas
        )
    )

    print("=" * 80)
    print("VECTOR SEARCH")
    print("=" * 80)

    for d, m in zip(hit_distances, hit_metas):
        print(
            f"{d:.4f} | "
            f"{m.get('type')} | "
            f"{m.get('title')} | "
            f"page={m.get('page')} | "
            f"chunk={m.get('chunk_index')}"
        )
    covered = defaultdict(set)
    file_cache = {}
    final_documents = []
    final_metadatas = []

    for doc, meta in zip(candidate_docs, candidate_metas):
        chunk_type = meta.get("type", "text")
        source = meta["source"]
        chunk_index = meta.get("chunk_index")
        # bảng và vision không merge
        if chunk_type in {"table", "vision"}:
            final_documents.append(doc)
            final_metadatas.append(meta)
            continue
        if chunk_index in covered[source]:
            continue
        if source not in file_cache:
            file_cache[source] = collection.get(
                where={
                    "source": source
                },
                include=[
                    "documents",
                    "metadatas"
                ]
            )

        file_data = file_cache[source]
        contexts = []
        window = set()
        for d, m in zip(
                file_data["documents"],
                file_data["metadatas"]
        ):
            if m.get("type") != chunk_type:
                continue
            idx = m.get("chunk_index")
            if idx is None:
                continue
            if abs(idx - chunk_index) <= 1:
                contexts.append((idx, d))
                window.add(idx)

        covered[source].update(window)
        contexts.sort(key=lambda x: x[0])
        merged = "\n\n".join(
            text
            for _, text in contexts
        )
        final_documents.append(merged)
        final_metadatas.append(meta)
    fused = _hybrid_rank(
        query,
        final_documents,
        final_metadatas
    )
    print("=" * 80)
    print("HYBRID RESULT")
    print("=" * 80)

    selected = fused[:top_k]
    selected = _expand_full_table_results(query, selected)

    for i, (rrf, kw, doc, meta) in enumerate(selected, 1):
        print(f"[{i}]")
        print("RRF :", round(rrf, 5))
        print("KW  :", kw)
        print("TYPE:", meta.get("type"))
        print("TITLE:", meta.get("title"))
        print("PAGE :", meta.get("page"))
        print("HEAD :", meta.get("headers"))
        print(doc[:400])
        print("-" * 80)

    return {
        "documents": [[
            d
            for _, _, d, _ in selected
        ]],
        "metadatas": [[
            m
            for _, _, _, m in selected
        ]],
        "keyword_scores": [[
            kw
            for _, kw, _, _ in selected
        ]]
    }

def show_all_documents():
    return collection.get()

def delete_document(source):
    result = collection.get(where={"source": source})
    ids = result["ids"]
    if ids:
        collection.delete(ids=ids)