from fastapi import FastAPI, HTTPException
from os import listdir
from pydantic import BaseModel
from backend.services.ai_service import ask_ai
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import UploadFile, File
import shutil
from backend.services.document_service import (read_txt, read_docx, read_pdf)
from backend.services.chroma_service import add_document, search_document
from backend.services.chroma_service import show_all_documents
import os
from backend.services.chroma_service import delete_document as delete_chroma_document
from backend.services.chroma_service import collection

app = FastAPI()
BASE_DIR = Path(__file__).resolve().parent.parent
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok = True)
INDEX_HTML = BASE_DIR / "frontend" / "html" /"index.html"

class ChatRequest(BaseModel):
    question: str

app.mount(
    "/css",
    StaticFiles(directory=BASE_DIR / "frontend" / "css"),
    name="css"
)

app.mount(
    "/js",
    StaticFiles(directory=BASE_DIR / "frontend" / "js"),
    name="js"
)

@app.get("/")
def home():
    return FileResponse(INDEX_HTML)

@app.post("/chat")
def chat(request: ChatRequest):
    answer = ask_ai(request.question)

    print(show_all_documents())
    return {
        "answer": answer
    }
 
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    delete_chroma_document(file.filename)

    print("Đường dẫn:", save_path)
    print("Kích thước:", save_path.stat().st_size)

    suffix = save_path.suffix.lower()

    try:
        if suffix == ".txt":
            chunks = read_txt(save_path)
        elif suffix == ".docx":
            chunks = read_docx(save_path)
        elif suffix == ".pdf":
            chunks = read_pdf(save_path)
        else:
            raise HTTPException(400, f"Định dạng {suffix} chưa được hỗ trợ.")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f"Lỗi khi đọc tài liệu: {e}")

    if not chunks:
        raise HTTPException(
            400,
            "Không trích xuất được nội dung nào từ file này. "
            "File có thể là PDF dạng ảnh/scan (cần OCR), hoặc file rỗng."
        )

    for chunk in chunks:
        context = f"""
    Tên tài liệu:
    {file.filename}
    Tiêu đề:
    {chunk["title"]}
    Nội dung:
    {chunk["content"]}
    """.strip()
        add_document(
            text=context,
            source=file.filename,
            title=chunk["title"],
            chunk_index=chunk["chunk_index"],
            total_chunks=chunk["total_chunks"]
        )

    return {
        "message": "Upload thành công",
        "filename": file.filename,
        "chunks_indexed": len(chunks)
    }

@app.get("/documents")
def get_documents():
    files = []
    for file in listdir(UPLOAD_DIR):
        files.append(file)

    return {
        "files": files
    }

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)
    delete_chroma_document(filename)
    return{
        "message": "Đã xoá thành công"
    }