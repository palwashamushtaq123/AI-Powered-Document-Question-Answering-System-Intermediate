import os
import shutil
from pathlib import Path
from typing import List, Dict, Any
from pypdf import PdfReader

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq


class RAGService:
    def __init__(self):
        # Embeddings model setup
        self.embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )
        self.persist_directory = "./chroma_db"
        
        # Fresh Database Start (Purana kachra clear karne ke liye)
        self._reset_db()

        # Initialize ChromaDB
        self.store = Chroma(
            collection_name="rag_documents",
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Groq Client setup
        groq_api_key = os.getenv("GROQ_API_KEY", "")
        self.client = Groq(api_key=groq_api_key) if groq_api_key else None

        # Automatically load default 5 files from 'data/' folder as system_data
        self.load_data_folder()

    def _reset_db(self):
        """Removes existing DB directory on boot to avoid stale data mixing"""
        if os.path.exists(self.persist_directory):
            try:
                shutil.rmtree(self.persist_directory)
                print("ChromaDB reset successfully for fresh session.")
            except Exception as e:
                print(f"Warning clearing ChromaDB folder: {e}")

    def _extract_text(self, file_path: str) -> str:
        """Extract text from PDF, TXT, or MD files"""
        path = Path(file_path)
        ext = path.suffix.lower()
        text = ""

        try:
            if ext == ".pdf":
                reader = PdfReader(file_path)
                for page in reader.pages:
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted + "\n"
            elif ext in [".txt", ".md"]:
                with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
        except Exception as e:
            print(f"Error reading {file_path}: {e}")

        return text

    def ingest_file(self, file_path: str, category: str = "user_upload"):
        """Ingest file with category: 'system_data' OR 'user_upload'"""
        text = self._extract_text(file_path)
        if not text.strip():
            print(f"Warning: No text extracted from {file_path}")
            return

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )
        chunks = splitter.split_text(text)
        filename = Path(file_path).name
        
        metadatas = [
            {"source": filename, "category": category} 
            for _ in chunks
        ]

        self.store.add_texts(texts=chunks, metadatas=metadatas)
        print(f"Successfully ingested {len(chunks)} chunks from {filename} as [{category}]")

    def load_data_folder(self, data_dir: str = "./data"):
        """Ingest 5 base documents as system_data"""
        folder = Path(data_dir)
        if not folder.exists():
            os.makedirs(folder, exist_ok=True)
            return

        supported_extensions = [".pdf", ".txt", ".md"]
        files = [f for f in folder.iterdir() if f.suffix.lower() in supported_extensions]

        if not files:
            print(f"No default documents found in {data_dir} directory.")
            return

        print(f"Auto-ingesting {len(files)} base file(s) from '{data_dir}' folder...")
        for file in files:
            self.ingest_file(str(file), category="system_data")

    def ask(self, question: str, top_k: int = 3, category: str = None, source: str = None, filter_category: str = None, prompt_type: str = "role_based") -> Dict[str, Any]:
        """
        Query database with strict category filtering.
        Accepts 'category', 'filter_category', or 'source' to prevent keyword errors.
        """
        # Determine actual filter category passed from any argument
        target_category = category or filter_category or source

        search_kwargs = {"k": top_k}
        
        # If user explicitly chose 'system_data' or 'user_upload'
        if target_category and target_category in ["system_data", "user_upload"]:
            search_kwargs["filter"] = {"category": target_category}

        results = self.store.similarity_search_with_score(question, **search_kwargs)

        context_chunks = []
        context_text = ""

        for doc, score in results:
            chunk_category = doc.metadata.get("category", "system_data")
            chunk_source = doc.metadata.get("source", "Unknown")
            
            chunk_data = {
                "text": doc.page_content,
                "source": f"[{chunk_category.upper()}] {chunk_source}",
                "distance": float(score)
            }
            context_chunks.append(chunk_data)
            context_text += f"\n- [{chunk_category.upper()} - {chunk_source}]: {doc.page_content}"

        if not context_chunks:
            return {
                "answer": "No relevant information found in the selected data category.",
                "context_chunks": []
            }

        # Prompts definition
        if prompt_type == "zero_shot":
            system_prompt = "Answer the question directly based only on the provided context."
        elif prompt_type == "few_shot":
            system_prompt = "You are a helpful assistant. Answer accurately based only on provided context."
        else: # role_based
            system_prompt = "You are an expert HR Specialist and Document Analyst. Answer accurately using only the given context."

        user_prompt = f"Context:\n{context_text}\n\nQuestion: {question}"

        if self.client:
            try:
                response = self.client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.2
                )
                answer = response.choices[0].message.content
            except Exception as e:
                answer = f"Error calling Groq API: {str(e)}"
        else:
            answer = f"Extracted Context:\n{context_text}"

        return {
            "answer": answer,
            "context_chunks": context_chunks
        }