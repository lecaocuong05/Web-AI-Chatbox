from pathlib import Path
from docx import Document
import fitz
import re
import io
import os
import tempfile
import hashlib
from docx.oxml.ns import qn
from docx.table import Table as DocxTable
from docx.text.paragraph import Paragraph as DocxParagraph
from backend.services.vision_service import describe_image

def split_into_chunks(text, chunk_size=1200, overlap=200):
    """
    Chia chunk nhưng ưu tiên cắt tại:
    1. xuống dòng
    2. dấu chấm
    3. dấu ;
    4. dấu :
    Nếu không có mới cắt cứng.
    """
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    length = len(text)
    while start < length:
        ideal_end = min(start + chunk_size, length)
        if ideal_end >= length:
            piece = text[start:].strip()
            if piece:
                chunks.append(piece)
            break

        split_pos = text.rfind("\n", start, ideal_end)
        if split_pos == -1:
            split_pos = text.rfind(".", start, ideal_end)
        if split_pos == -1:
            split_pos = text.rfind(";", start, ideal_end)
        if split_pos == -1:
            split_pos = text.rfind(":", start, ideal_end)
        if split_pos == -1:
            split_pos = ideal_end
        else:
            split_pos += 1

        chunk = text[start:split_pos].strip()
        if chunk:
            chunks.append(chunk)

        next_start = split_pos - overlap
        if next_start <= start:
            next_start = split_pos
        start = next_start
    return chunks

#===========================================================
TABLE_SEPARATOR_PATTERN = re.compile(r"^\|?[\s\-:|]+\|?$")

def split_markdown_table(text, chunk_size=1200):
    """
    Chia nội dung thành nhiều mảnh, LUÔN LẶP LẠI dòng tiêu đề bảng ở đầu mỗi
    mảnh nếu phát hiện đây là bảng Markdown -> mỗi chunk tự hiểu cột nào là
    cột gì, không phụ thuộc chunk đứng trước.

    Tự động nhận diện: tìm dòng phân cách kiểu "| --- | --- |" bất kỳ đâu
    trong text (không bắt buộc ở đầu, vì VLM đọc ảnh đôi khi thêm vài câu
    dẫn trước bảng). Nếu KHÔNG tìm thấy cấu trúc bảng -> coi là văn bản
    thường, dùng lại split_into_chunks() như bình thường.
    """
    if len(text) <= chunk_size:
        return [text]

    lines = text.split("\n")
    header_idx = None
    for i in range(1, len(lines)):
        if "|" in lines[i - 1] and TABLE_SEPARATOR_PATTERN.match(lines[i].strip()):
            header_idx = i - 1
            break

    if header_idx is None:
        # Không phải bảng Markdown -> chia như văn bản thường
        return split_into_chunks(text, chunk_size)

    preamble = "\n".join(lines[:header_idx]).strip()
    header = lines[header_idx]
    separator = lines[header_idx + 1]
    data_rows = lines[header_idx + 2:]
    header_block = (preamble + "\n" if preamble else "") + header + "\n" + separator

    chunks = []
    current_rows = []
    current_len = len(header_block)

    for row in data_rows:
        row_len = len(row) + 1
        if current_rows and current_len + row_len > chunk_size:
            chunks.append(header_block + "\n" + "\n".join(current_rows))
            current_rows = []
            current_len = len(header_block)
        current_rows.append(row)
        current_len += row_len

    if current_rows:
        chunks.append(header_block + "\n" + "\n".join(current_rows))

    return chunks if chunks else [text]
#===========================================================
def cluster_drawings(rects, distance = 40):
    clusters = []
    for rect in rects:
        merged = False
        for cluster in clusters:
            expand = fitz.Rect(
                cluster.x0 - distance,
                cluster.y0 - distance,
                cluster.x1 + distance,
                cluster.y1 + distance
            )
            if expand.intersects(rect):
                cluster |= rect
                merged = True
                break
        if not merged:
            clusters.append(fitz.Rect(rect))
    return clusters

#===========================================================
def table_to_markdown(rows):
    if not rows:
        return ""
    clean_rows = []
    for row in rows:
        clean_row = []
        for cell in row:
            if cell is None:
                clean_row.append("")
            else:
                clean_row.append(
                    str(cell).strip().replace("\n", " "))
        clean_rows.append(clean_row)

    clean_rows = [
        row for row in clean_rows
        if any(cell != "" for cell in row)
    ]
    if not clean_rows:
        return ""
    
    header = clean_rows[0]
    body = clean_rows[1:]
    col_count = len(header)
    lines = []

    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * col_count) + " |")
    for row in body:
        while len(row) < col_count:
            row.append("")

        row = row[:col_count]
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)

# ================= Hàm vision chung =========================
def vision_to_chunk(image_bytes, title):
    """
    image_bytes: bytes ảnh
    title      : tiêu đề chunk
    """
    chunks = []
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            delete=False
        ) as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        description = describe_image(temp_path)

        if description.strip() and "không chứa thông tin có giá trị tra cứu" not in description.lower():
            pieces = split_markdown_table(description)
            for piece in pieces:
                chunks.append({
                    "title": title,
                    "content": piece
                })
    except Exception as e:
        print("Vision:", e)
    finally:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
    return chunks 

#===========================================================
def iter_docx_blocks(document):
    body = document.element.body
    for child in body.iterchildren():
        if child.tag == qn("w:p"):
            yield DocxParagraph(child, document)
        elif child.tag == qn("w:tbl"):
            yield DocxTable(child, document)

#===================== TXT ===================================
def read_txt(file_path):
    with open(file_path, "r",encoding="utf-8") as f:
        lines = f.readlines()

    chunks = []
    current_title = "Không có tiêu đề"
    current_content = []
    global_chunk_index = 1

    def save_chunk():
        nonlocal current_content, global_chunk_index
        if current_content:
            text = "\n".join(current_content).strip()
            pieces = split_into_chunks(text)
            for piece in pieces:
                chunks.append({
                    "title": current_title,
                    "content": piece,
                    "chunk_index": global_chunk_index,
                    "total_chunks": 0
                })
                global_chunk_index += 1
            current_content = []

    for line in lines:
        text = line.strip()
        if not text:
            continue
        if(
            text.startswith("#")
            or re.match(r"^\d+\.", text)
            or text.upper().startswith("CHƯƠNG")
            or text.upper().startswith("PHẦN")
            or text.upper().startswith("MỤC")
            or re.match(r"^=+$", text)
            or re.match(r"^-+$", text)
        ):
            save_chunk()
            current_title = text.lstrip("#").strip()
        else:
            current_content.append(text)
    save_chunk()
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total
    return chunks

#===================== DOCX ===================================
def read_docx(file_path):
    doc = Document(file_path)
    relationship_images = {}
    total_images = len([
        rel for rel in doc.part.rels.values()
        if "image" in rel.target_ref
    ])
    current_image = 0
    print(f"Tổng số ảnh cần xử lý: {total_images}")
    vision_cache = {}
    for rel in doc.part.rels.values():
        if "image" in rel.target_ref:
            relationship_images[rel.rId] = rel.target_part.blob
    chunks = []
    h1 = ""
    h2 = ""
    h3 = ""
    current_content = []
    global_chunk_index = 1
    vision_chunk_index = VISION_CHUNK_INDEX_START
    MIN_IMAGE_SIZE_BYTES = 8_000  
    def current_title():
        title = " > ".join(
            x for x in [h1, h2, h3] if x
        )
        return title if title else "Không có tiêu đề"
    def save_chunk():
        nonlocal current_content
        nonlocal global_chunk_index
        if current_content:
            text = "\n".join(current_content).strip()
            pieces = split_into_chunks(text)
            for piece in pieces:
                chunks.append({
                    "title": current_title(),
                    "content": piece,
                    "chunk_index": global_chunk_index,
                    "total_chunks": 0
                })
                global_chunk_index += 1
            current_content = []

    for block in iter_docx_blocks(doc):
        if isinstance(block, DocxParagraph):
            text = block.text.strip()
            if text:
                style = block.style.name
                if style.startswith("Heading 1"):
                    save_chunk()
                    h1 = text
                    h2 = ""
                    h3 = ""
                elif style.startswith("Heading 2"):
                    save_chunk()
                    h2 = text
                    h3 = ""
                elif style.startswith("Heading 3"):
                    save_chunk()
                    h3 = text
                else:
                    current_content.append(text)
            try:
                for run in block.runs:
                    drawing = run._element.xpath(
                        ".//*[local-name()='blip']"
                    )
                    for item in drawing:
                        r_embed = item.get(
                            "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}embed"
                        )
                        if (
                            r_embed
                            and
                            r_embed in relationship_images
                        ):
                            image_bytes = relationship_images[r_embed]
                            image_hash = hashlib.md5(image_bytes).hexdigest()
                            if len(image_bytes) < MIN_IMAGE_SIZE_BYTES:
                                continue  
                            if image_hash in vision_cache:
                                current_image += 1
                                print(
                                    f"Ảnh {current_image}/{total_images}"
                                    "(đã cache)"
                                )
                                vision_chunks = vision_cache[image_hash]
                            else:
                                current_image += 1
                                print(
                                    f"Đang xử lý ảnh"
                                    f"{current_image}/{total_images}"
                                )
                                vision_chunks = vision_to_chunk(
                                    image_bytes,
                                    current_title() + " - Vision"
                                )
                                print(
                                    f"Xong ảnh"
                                    f"{current_image}/{total_images}"
                                )
                                vision_cache[image_hash] = vision_chunks
                            for vc in vision_chunks:
                                chunks.append({
                                    "title": vc["title"],
                                    "content": vc["content"],
                                    "chunk_index": vision_chunk_index,
                                    "total_chunks": 0
                                })
                                vision_chunk_index += 1
            except Exception as e:
                print("Vision DOCX:", e)
        elif isinstance(block, DocxTable):
            save_chunk()
            rows = []
            for row in block.rows:
                rows.append(
                    [cell.text.strip() for cell in row.cells]
                )
            md_table = table_to_markdown(rows)
            if md_table:
                table_pieces = split_markdown_table(md_table)
                for piece in table_pieces:
                    chunks.append({
                        "title": current_title() + " (Bảng)",
                        "content": piece,
                        "chunk_index": global_chunk_index,
                        "total_chunks": 0
                    })
                    global_chunk_index += 1
    save_chunk()
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total
    return chunks
#=============================================================
def _detect_boilerplate_lines(pdf, threshold_ratio=0.4):
    """
    Quét toàn bộ file để tìm các dòng lặp lại ở phần lớn số trang
    (thường là header/footer như "Phiên bản x.xx Tài liệu ..." hoặc số trang).
    Các dòng này sẽ bị loại khỏi nội dung để không làm nhiễu chunk và retrieval.
    """
    num_pages = len(pdf)
    line_page_count = {}
 
    for page in pdf:
        text = page.get_text()
        seen_this_page = set()
        for line in text.split("\n"):
            line = line.strip()
            if not line or line in seen_this_page:
                continue
            seen_this_page.add(line)
            line_page_count[line] = line_page_count.get(line, 0) + 1
 
    threshold = max(2, int(num_pages * threshold_ratio))
    return {line for line, count in line_page_count.items() if count >= threshold}

#===================== PDF ===================================
FIGURE_CAPTION_PATTERN = re.compile(r"(Hình|Sơ đồ|Biểu đồ)\s+\d+\s*[:.]", re.IGNORECASE)
VISION_CHUNK_INDEX_START = 100_000
DRAWING_COUNT_THRESHOLD = 5 
MIN_IMAGE_SIZE_BYTES_PDF = 8_000

def read_pdf(file_path):
    pdf = fitz.open(file_path)
    boilerplate_lines = _detect_boilerplate_lines(pdf)
    chunks = []
    global_chunk_index = 1
    vision_chunk_index = VISION_CHUNK_INDEX_START
    title = "Không có tiêu đề"
    content = []
    total_pages = len(pdf)
    processed_image_hashes = set()

    def save_chunk():
        nonlocal content, global_chunk_index
        if content:
            text = "\n".join(content).strip()
            pieces = split_into_chunks(text)
            for piece in pieces:
                chunks.append({
                    "title": title,
                    "content": piece,
                    "chunk_index": global_chunk_index,
                    "total_chunks": 0
                })
                global_chunk_index += 1
            content = []

    for page_num, page in enumerate(pdf, start=1):
        print(f"Đang xử lý trang {page_num}/{total_pages}...")
        table_bboxes = []
        page_text = page.get_text()
        has_real_text = len(page_text.strip()) >= 30  # ngưỡng nhỏ: gần như rỗng -> nghi là trang scan/ảnh

        try:
            found_tables = page.find_tables()
        except Exception:
            found_tables = None

        has_table = bool(found_tables and found_tables.tables)
        has_embedded_image = len(page.get_images(full=True)) > 0
        need_vision = False
        new_embedded_image_bytes = []  # giữ lại ảnh nhúng MỚI để gửi thẳng cho VLM (không cần render cả trang)
        if has_embedded_image:
            for img in page.get_images(full=True):
                xref = img[0]
                try:
                    image_dict = pdf.extract_image(xref)
                    image_bytes = image_dict["image"]
                    if len(image_bytes) < MIN_IMAGE_SIZE_BYTES_PDF:
                        continue
                    image_hash = hashlib.md5(image_bytes).hexdigest()
                    if image_hash not in processed_image_hashes:
                        processed_image_hashes.add(image_hash)
                        need_vision = True
                        new_embedded_image_bytes.append(image_bytes)
                except Exception as e:
                    print(f"Lỗi extract image: {e}")
                    continue

        has_figure_caption = bool(FIGURE_CAPTION_PATTERN.search(page_text))
        drawings = page.get_drawings()
        drawing_count = len(drawings)
        has_vector_diagram = (drawing_count >= DRAWING_COUNT_THRESHOLD) and not has_table

        print(f"   [debug] Trang {page_num}: drawing_count={drawing_count}, "
              f"has_table={has_table}, has_embedded_image={has_embedded_image}, "
              f"has_figure_caption={has_figure_caption}, has_real_text={has_real_text}")

        vision_chunks = []
        if(need_vision or has_figure_caption or has_vector_diagram):
            reason = []
            if need_vision:
                reason.append("có ảnh nhúng mới")
            if has_figure_caption:
                reason.append("có caption Hình/Sơ đồ")
            if has_vector_diagram:
                reason.append(f"nghi có sơ đồ vector ({drawing_count} đối tượng)")

            if not has_real_text:
                print(f"   -> Trang {page_num} {', '.join(reason)}, KHÔNG có text thật "
                      f"(nghi scan) -> gửi nguyên trang cho VLM...")
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes("png")
                vision_chunks = vision_to_chunk(img_bytes, f"Trang {page_num} - Vision")
                del pix
                del img_bytes

            else:
                print(f"   -> Trang {page_num} {', '.join(reason)}, có text thật "
                      f"-> chỉ gửi vùng ảnh/sơ đồ cho VLM (không gửi cả trang)...")

                # Ưu tiên 1: có ảnh nhúng thật MỚI -> gửi thẳng ảnh đó
                for image_bytes in new_embedded_image_bytes:
                    vc = vision_to_chunk(image_bytes, f"Trang {page_num} - Vision")
                    vision_chunks.extend(vc)

                # Ưu tiên 2: nghi có sơ đồ vector -> cắt đúng vùng chứa các
                # đối tượng vector đó rồi mới render (không render cả trang)
                if has_vector_diagram and drawings:

                    rects = [
                        fitz.Rect(d["rect"])
                        for d in drawings
                        if d.get("rect")
                    ]
                    if rects:
                        x0 = min(r.x0 for r in rects) - 10
                        y0 = min(r.y0 for r in rects) - 10
                        x1 = max(r.x1 for r in rects) + 10
                        y1 = max(r.y1 for r in rects) + 10
                        page_rect = page.rect
                        clip = fitz.Rect(
                            max(x0, 0),
                            max(y0, 0),
                            min(x1, page_rect.width),
                            min(y1, page_rect.height)
                        )
                        if clip.width >= 120 and clip.height >= 120:
                            ratio = clip.width / clip.height
                            if ratio <= 8:
                                if ratio >= 0.12:
                                    page_area = page.rect.get_area()
                                    if clip.get_area() <= page_area * 0.65:
                                        pix = page.get_pixmap(
                                            dpi=300,
                                            clip=clip
                                        )
                                        img_bytes = pix.tobytes("png")
                                        vc = vision_to_chunk(
                                            img_bytes,
                                            f"Trang {page_num} - Vision (sơ đồ)"
                                        )
                                        vision_chunks.extend(vc)
                                        del pix
                                        del img_bytes

        if has_table:
            for index, table in enumerate(found_tables.tables, start=1):
                rows = table.extract()
                md_table = table_to_markdown(rows)
                table_bboxes.append(table.bbox)
                if md_table:
                    table_chunks = split_markdown_table(md_table)
                    for piece in table_chunks:
                        chunks.append({
                            "title": f"{title} (Bảng {index} - Trang{page_num})",
                            "content": piece,
                            "chunk_index": global_chunk_index,
                            "total_chunks": 0
                        })
                        global_chunk_index +=1
                    
        blocks = page.get_text("blocks")
        for block in blocks:
            x0, y0, x1, y1, text, *_=block
            block_rect = fitz.Rect(x0, y0, x1, y1)
            inside_table = any(
                block_rect.intersects(fitz.Rect(bbox)) for bbox in table_bboxes
            )
            if inside_table:
                continue

            for line in text.split("\n"):
                line = line.strip()
                if not line or line in boilerplate_lines:
                    continue
                words = line.split()
                if (
                    len(words) <= 10
                    and "." not in line
                    and (
                        line.isupper()
                        or line.endswith(":")
                    )
                ):
                    save_chunk()
                    title = line
                else:
                    content.append(line)
        for vc in vision_chunks:
            chunks.append({
                "title": vc["title"],
                "content": vc["content"],
                "chunk_index": vision_chunk_index,
                "total_chunks": 0
            })
            vision_chunk_index += 1

    save_chunk()
    total = len(chunks)
    for c in chunks:
        c["total_chunks"] = total
    pdf.close()
    return chunks
