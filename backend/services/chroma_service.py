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


def add_document(text, source, title, chunk_index=1, total_chunks=1):
    print("===== BẮT ĐẦU THÊM CHROMADB =====")
    print(f"Source: {source}")
    print(f"Title: {title}")
    print(f"Chunk index: {chunk_index}/{total_chunks}")
    print(f"Độ dài text: {len(text)} ký tự")

    collection.add(
        documents=[text],
        ids=[str(uuid.uuid4())],
        metadatas=[{
            "source": source,
            "title": title,
            "chunk_index": chunk_index,
            "total_chunks": total_chunks
        }]
    )
    print("===== ĐÃ LƯU CHROMADB =====")

def rerank_documents(query, documents, metadatas):
    """
    Chấm điểm các document theo mức độ khớp từ khoá THẬT SỰ mang nghĩa (đã
    loại bỏ stopword) — tránh đoạn không liên quan được điểm dương chỉ vì
    trùng từ phổ biến như "các", "thông tin".
    """
    keywords = _extract_keywords(query)
    scores = []
    for doc, meta in zip(documents, metadatas):
        score = 0
        text = doc.lower()
        title = meta.get("title", "").lower()
        for kw in keywords:
            if kw in title:
                score += 5
            score += text.count(kw)
        scores.append(score)
    return scores


def _hybrid_rank(query, documents, metadatas, rrf_k=60):
    """
    Kết hợp thứ hạng vector (Chroma, đã sắp theo distance -> index trong
    list = vector rank) với thứ hạng từ khoá (đã lọc stopword), bằng
    Reciprocal Rank Fusion (RRF).

    QUAN TRỌNG: đây là sắp xếp MỀM — không loại cứng chunk điểm từ khoá = 0.
    Lý do: nếu câu hỏi diễn đạt bằng từ đồng nghĩa khác với tài liệu, chunk
    ĐÚNG vẫn có thể bị điểm từ khoá = 0 dù vector similarity đã tìm đúng —
    loại cứng trong trường hợp đó làm ngữ cảnh gửi cho LLM bị thưa/rỗng oan,
    khiến model dễ "tự suy luận" theo kiến thức riêng thay vì dựa tài liệu.
    RRF vẫn đẩy chunk khớp từ khoá tốt lên đầu, nhưng không xoá sạch phần còn lại.
    """
    if not documents:
        return []

    keyword_scores = rerank_documents(query, documents, metadatas)
    keyword_ranked = sorted(
        range(len(documents)), key=lambda i: keyword_scores[i], reverse=True
    )
    keyword_rank_of = {idx: rank for rank, idx in enumerate(keyword_ranked)}

    fused = []
    for i, (doc, meta) in enumerate(zip(documents, metadatas)):
        vector_rank = i  # documents đã theo đúng thứ tự distance tăng dần
        kw_rank = keyword_rank_of[i]
        rrf_score = 1 / (rrf_k + vector_rank) + 1 / (rrf_k + kw_rank)
        fused.append((rrf_score, keyword_scores[i], doc, meta))

    fused.sort(key=lambda x: x[0], reverse=True)
    return fused


def search_document(query, top_k=6):
    SEARCH_MULTIPLIER = 4

    result = collection.query(
        query_texts=[query],
        n_results=min(top_k * SEARCH_MULTIPLIER, 30),
        include=["documents", "metadatas", "distances"]
    )

    hit_docs = result["documents"][0]
    hit_metas = result["metadatas"][0]
    hit_distances = result["distances"][0]

    print("=" * 80)
    print("QUERY: ", query)
    for d, m in zip(hit_distances, hit_metas):
        print(
            f"distance={d:.4f} | source={m['source']} | title={m.get('title')} | chunk={m.get('chunk_index')}"
        )
    print("=" * 80)

    final_documents = []
    final_metadatas = []

    covered = defaultdict(set)   
    file_cache = {}             

    MAX_CONTEXT = top_k
    for doc, meta in zip(hit_docs, hit_metas):
        source = meta["source"]
        chunk_index = meta.get("chunk_index")

        if chunk_index is None:
            final_documents.append(doc)
            final_metadatas.append(meta)
            continue

        if chunk_index in covered[source]:
            continue

        if source not in file_cache:
            file_cache[source] = collection.get(
                where={"source": source},
                include=["documents", "metadatas"]
            )
        file_data = file_cache[source]

        contexts = []
        window_indices = set()
        for f_doc, f_meta in zip(file_data["documents"], file_data["metadatas"]):
            f_index = f_meta.get("chunk_index")
            if f_index is None:
                continue
            if abs(f_index - chunk_index) <= 1:
                contexts.append((f_index, f_doc))
                window_indices.add(f_index)

        covered[source].update(window_indices)
        contexts.sort(key=lambda x: x[0])
        if not contexts:
            continue
        merged = "\n\n".join(text for _, text in contexts)

        final_documents.append(merged)
        final_metadatas.append(meta)

    # ---- Hybrid rank MỀM: vector rank + keyword rank qua RRF, không loại cứng ----
    fused = _hybrid_rank(query, final_documents, final_metadatas)
    
    print("="*60)
    print("CONTEXT SAU HYBRID")
    for i, (rrf_score, kw_score, doc, meta) in enumerate(fused[:MAX_CONTEXT], 1):
        print(f"[{i}]")
        print(f"RRF Score: {rrf_score:.5f}")
        print(f"Keyword Score: {kw_score}")
        print(f"Title: {meta['title']}")
        print(f"Scource: {meta['source']}")
        print("---- Nội Dung -----")
        print(doc[:500])
        print("="*80)
    print("="*80)

    print("=" * 60)
    print("[debug] Kết quả Hybrid (RRF) — rrf_score / keyword_score / title:")
    for rrf_score, kw_score, doc, meta in fused:
        print(f"  rrf={rrf_score:.5f}  keyword_score={kw_score}  title={meta['title']}")
    print("=" * 60)

    rank_documents = [doc for _, _, doc, meta in fused[:MAX_CONTEXT]]
    rank_metadatas = [meta for _, _, doc, meta in fused[:MAX_CONTEXT]]
    rank_keyword_scores = [kw for _, kw, doc, meta in fused[:MAX_CONTEXT]]

    return{
        "documents": [rank_documents],
        "metadatas": [rank_metadatas],
        "keyword_scores": [rank_keyword_scores]
    }

def show_all_documents():
    return collection.get()

def delete_document(source):
    result = collection.get(
        where = {"source": source}
    )
    ids = result["ids"]
    if ids:
        collection.delete(ids=ids)