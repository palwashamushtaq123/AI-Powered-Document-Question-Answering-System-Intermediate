import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from mangum import Mangum

from app.rag_service import RAGService

app = FastAPI(title="AI Knowledge Assistant")

# Vercel serverless handler
handler = Mangum(app)

# Initialize RAG Service
rag_service = RAGService()


class QueryRequest(BaseModel):
    question: str
    prompt_type: Optional[str] = "role_based"
    top_k: Optional[int] = 3
    filter_category: Optional[str] = None
    category: Optional[str] = None
    source: Optional[str] = None


@app.get("/", response_class=HTMLResponse)
def read_index():
    base_dir = Path(__file__).resolve().parent

    possible_paths = [
        base_dir / "index.html",
        base_dir / "templates" / "index.html",
        base_dir / "static" / "index.html",
        Path("index.html")
    ]

    for html_path in possible_paths:
        if html_path.exists():
            with open(html_path, "r", encoding="utf-8") as f:
                return f.read()

    return f"<h1>index.html not found</h1><p>Searched in: {base_dir}</p>"


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    temp_file_path = None
    try:
        # File extension nikalna
        file_ext = Path(file.filename).suffix

        # Temporary file create karna (Separated from ./data folder)
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_file:
            shutil.copyfileobj(file.file, temp_file)
            temp_file_path = temp_file.name

        # Ingest into ChromaDB with metadata (source name original filename, category = user_upload)
        text = rag_service._extract_text(temp_file_path)
        if not text.strip():
            raise HTTPException(status_code=400, detail="Uploaded file is empty or text could not be extracted.")

        splitter = rag_service.store._embedding_function if hasattr(rag_service, 'embeddings') else None
        
        # Ingest file using original file name in metadata
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = splitter.split_text(text)
        
        metadatas = [
            {"source": file.filename, "category": "user_upload"} 
            for _ in chunks
        ]
        
        rag_service.store.add_texts(texts=chunks, metadatas=metadatas)

        return {"message": f"File '{file.filename}' uploaded and indexed successfully as User Document!"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # Clean up temporary file from disk so data folder stays untouched
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)


@app.post("/rag/query")
def query_rag(request: QueryRequest):
    try:
        res = rag_service.ask(
            question=request.question,
            prompt_type=request.prompt_type,
            top_k=request.top_k,
            category=request.category,
            filter_category=request.filter_category,
            source=request.source
        )
        return res
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))