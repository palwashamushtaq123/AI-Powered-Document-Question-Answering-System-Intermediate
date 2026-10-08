# 🧠 AI Knowledge Assistant — RAG & Semantic Search System

An intelligent **Retrieval-Augmented Generation (RAG)** web application built with **FastAPI, LangChain, ChromaDB, Sentence Transformers, and Groq LLM (Llama-3.3-70b)**.

The application allows users to query a pre-loaded knowledge base as well as upload their own **PDF, TXT, and Markdown documents**. It uses dense vector embeddings and semantic similarity search to retrieve relevant information before generating answers with the Groq LLM.

> **Note:** This project is configured and tested as a local application. No live deployment is currently provided.

---

## ✨ Key Features

### 📂 Dynamic Document Source Filtering

* **System Data Only** — Search strictly within the 5 pre-loaded knowledge documents.
* **User Uploaded Only** — Query only documents uploaded during the current session.
* **All Documents** — Search across both system and user-uploaded documents.

### 🤖 Prompt Engineering

Supports three prompting strategies:

* **Zero-Shot Prompting**
* **Few-Shot Prompting**
* **Role-Based Prompting**

### 🔎 Dense Vector Semantic Search

Uses:

* `all-MiniLM-L6-v2` embeddings
* ChromaDB vector database
* Cosine similarity-based retrieval

This allows the system to retrieve relevant content based on **meaning and context**, rather than relying only on exact keyword matches.

### 📄 Multi-Format Document Support

Users can upload:

* PDF
* TXT
* Markdown (`.md`)

### 🔐 Document Isolation

System documents and user-uploaded documents are categorized separately to prevent unintended cross-source retrieval.

### 🎨 Modern Web Interface

* Responsive dark-mode interface
* Document upload functionality
* Prompt strategy selector
* Data-source filtering
* Retrieved context display
* Similarity/distance score visualization
* Real-time status notifications

---

# 📚 Theory & Conceptual Understanding

## 1. Semantic Search vs. Keyword Search

### Keyword Search

Keyword or lexical search looks for direct matches between query terms and document text. Traditional approaches include **TF-IDF** and **BM25**.

Its limitation is that it may fail when the query uses different wording, synonyms, or contextual variations.

### Semantic Search

Semantic search converts text into numerical vector representations called **embeddings**.

The `all-MiniLM-L6-v2` model captures the semantic meaning and contextual relationships between text. ChromaDB then retrieves the most relevant document chunks based on vector similarity.

For example:

> Query: `How does vector storage work?`

A semantic search system can retrieve content discussing **embeddings, vector databases, similarity search, and document retrieval**, even when the exact query words are not present.

---

## 2. Prompting Techniques Implemented

### Zero-Shot Prompting

The model receives instructions without any example responses.

### Few-Shot Prompting

The model is guided using examples or contextual patterns to improve response consistency.

### Role-Based Prompting

The model is assigned a specific role, such as:

> **HR Specialist and Document Analyst**

This helps guide the model toward professional and domain-focused responses.

---

# 🏗️ RAG Pipeline

```text
User Question
      ↓
Query Processing
      ↓
Embedding Generation
      ↓
ChromaDB Similarity Search
      ↓
Relevant Document Chunks
      ↓
Context Construction
      ↓
Groq LLM
      ↓
Final Answer
```

### Document Ingestion Pipeline

```text
PDF / TXT / MD
      ↓
Text Extraction
      ↓
Text Chunking
      ↓
Sentence Transformer Embeddings
      ↓
ChromaDB
      ↓
Vector Retrieval
```

---

# 🛠️ Technology Stack

| Technology            | Purpose                          |
| --------------------- | -------------------------------- |
| Python                | Core programming language        |
| FastAPI               | Backend API framework            |
| LangChain             | Text splitting and RAG utilities |
| ChromaDB              | Vector database                  |
| Sentence Transformers | Text embeddings                  |
| Groq                  | LLM inference                    |
| Llama-3.3-70b         | Language model                   |
| Pypdf                 | PDF text extraction              |
| HTML/CSS/JavaScript   | Frontend                         |

---

# 📁 Project Structure

```text
AI-Powered-Document-Question-Answering-System-Intermediate/
│
├── app/
│   ├── __init__.py
│   ├── rag_service.py
│   ├── chunker.py
│   ├── config.py
│   ├── document_loader.py
│   ├── embedding_service.py
│   ├── groq_service.py
│   ├── ingest_service.py
│   └── vector_store.py
│
├── data/
│   ├── llm_governance.txt
│   ├── prompt_engineering.pdf
│   ├── rag_architecture.txt
│   ├── search_paradigms.md
│   └── vectordb_chroma.md
│
├── templates/
│   └── index.html
│
├── assets/
│   ├── Rag_ssystem_file_fewshot.png
│   ├── Rag_system_file_cot.png
│   ├── Rag_system_file_zeroshot.png
│   └── user_uploaded_file_output.png
│
├── api.py
├── cli.py
├── ingest.py
├── prompt_benchmark.py
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

---

# 🚀 Local Installation & Setup

## 1. Clone Repository

```bash
git clone https://github.com/palwashamushtaq123/AI-Powered-Document-Question-Answering-System-Intermediate.git
cd AI-Powered-Document-Question-Answering-System-Intermediate
```

## 2. Create Virtual Environment

```bash
python -m venv venv
```

### Windows

```bash
venv\Scripts\activate
```

### macOS / Linux

```bash
source venv/bin/activate
```

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## 4. Configure Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

## 5. Run the Application

```bash
uvicorn api:app --reload
```

Open the application in your browser:

```text
http://127.0.0.1:8000
```

---

# 🔌 API Endpoints

### Home

```text
GET /
```

Loads the web interface.

### Upload Document

```text
POST /upload
```

Uploads and indexes a PDF, TXT, or Markdown document.

### RAG Query

```text
POST /rag/query
```

Processes a question using semantic retrieval and the Groq LLM.

---

# 📸 Project Screenshots

The repository includes screenshots demonstrating:

* Zero-Shot prompting
* Few-Shot prompting
* Role-Based / CoT prompting
* User document upload and retrieval

Screenshots are available in the [`assets`](https://github.com/palwashamushtaq123/AI-Powered-Document-Question-Answering-System-Intermediate/tree/main/assets) folder.

---

# 🎯 Project Objective

The objective of this project is to demonstrate how **Retrieval-Augmented Generation** combines:

* Document processing
* Text chunking
* Dense embeddings
* Vector databases
* Semantic search
* Prompt engineering
* Large Language Models

to create a context-aware AI knowledge assistant that generates answers based on retrieved document information.

---

# 👤 Author

**Palwasha Sheikh**

AI & Machine Learning | Data Science

**AI & Machine Learning Assignment Submission**
