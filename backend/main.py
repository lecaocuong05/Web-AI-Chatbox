import traceback
from fastapi import FastAPI, HTTPException
from os import listdir
from pydantic import BaseModel
from backend.services.ai_service import ask_ai
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from fastapi import FastAPI, HTTPException, UploadFile, File
import shutil
from backend.services.document_service import (read_txt, read_docx, read_pdf)
from backend.services.chroma_service import add_document, search_document
from backend.services.chroma_service import show_all_documents
import os
from backend.services.chroma_service import delete_document as delete_chroma_document
from backend.services.chroma_service import collection
import uuid
from fastapi import Request, Response
from backend.services.history_service import clear_history

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
def chat(
    payload: ChatRequest,
    request: Request,
    response: Response
):
    session_id = request.cookies.get(
        "company_ai_session_id"
    )
    if not session_id:
        session_id = str(
            uuid.uuid4()
        )
        response.set_cookie(
            key="company_ai_session_id",
            value=session_id,
            max_age=60 * 60 * 24 * 30,
            httponly=True,
            samesite="lax",
            secure=False
        )
    print(
        "SESSION:",session_id
    )
    answer = ask_ai(payload.question, session_id)
    return{"answer": answer}

@app.post("/history/clear")
def clear_my_history(
    request: Request
):
    session_id = request.cookies.get(
        "company_ai_session_id"
    )
    if session_id:
        clear_history(
            session_id
        )
    return {
        "message": "Đã xóa lịch sử hội thoại"
    }
 
@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    save_path = UPLOAD_DIR / file.filename

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    print("Đường dẫn:", save_path)
    print("Kích thước:", save_path.stat().st_size)

    suffix = save_path.suffix.lower()
    
    # 1. ĐỌC VÀ CHIA CHUNK TRƯỚC
   
    if suffix == ".txt":
        chunks = read_txt(save_path)

    elif suffix == ".docx":
        chunks = read_docx(save_path)

    elif suffix == ".pdf":
        chunks = read_pdf(save_path)

    else:
        raise HTTPException(
            status_code=400,
            detail="Chỉ hỗ trợ PDF, DOCX và TXT"
        )

    print("=" * 60)
    print("TỔNG CHUNK:", len(chunks))

    text_count = sum(
        c.get("type") == "text"
        for c in chunks
    )

    table_count = sum(
        c.get("type") == "table"
        for c in chunks
    )

    vision_count = sum(
        c.get("type") == "vision"
        for c in chunks
    )

    print("TEXT   :", text_count)
    print("TABLE  :", table_count)
    print("VISION :", vision_count)
    print("=" * 60)

    # 2. XÓA VECTOR CŨ CỦA FILE NÀY
  
    delete_chroma_document(file.filename)

    # 3. LƯU TOÀN BỘ METADATA VÀO CHROMA
   
    for chunk in chunks:
        add_document(
            text=chunk["content"],
            source=file.filename,
            title=chunk["title"],
            chunk_index=chunk["chunk_index"],
            total_chunks=chunk["total_chunks"],

            # QUAN TRỌNG
            chunk_type=chunk.get("type", "text"),
            page=chunk.get("page"),
            headers=chunk.get("headers", [])
        )

    return {
        "message": "Upload thành công",
        "filename": file.filename,
        "total_chunks": len(chunks),
        "text_chunks": text_count,
        "table_chunks": table_count,
        "vision_chunks": vision_count
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