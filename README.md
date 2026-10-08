# 🧠 AI Knowledge Assistant (RAG & Semantic Search System)

An intelligent Retrieval-Augmented Generation (RAG) web application built using **FastAPI**, **LangChain**, **ChromaDB**, and **Groq LLM** (Llama-3.3-70b). 

This platform enables users to perform strict-isolated document querying on pre-loaded system knowledge bases as well as user-uploaded custom documents using advanced prompt engineering techniques and dense vector similarity search.

---

## 🔗 Live Demo
🌐 **Deployment Link:** [Click Here to View Live Project](https://your-vercel-app-url.vercel.app)

---

## ✨ Key Features & Highlights

- **Dynamic Data Source Filtering:**
  - **System Data Only:** Search strictly within the default 5 foundational documents in `./data/`.
  - **User Uploaded Only:** Query strictly against newly uploaded PDF/TXT/MD files without history leakage.
  - **All Documents:** Search seamlessly across both base knowledge and user-uploaded data sources.
- **Prompt Engineering Strategy Switcher (Part 1 Requirement):**
  - Dynamic toggling between **Zero-Shot**, **Few-Shot**, and **Role-Based Prompting**.
- **Dense Vector Semantic Search (Part 2 Requirement):**
  - High-precision similarity search powered by `sentence-transformers/all-MiniLM-L6-v2` embeddings and `ChromaDB`.
- **Automatic In-Memory Reset & Isolation:**
  - Temporary isolated storage for user uploads to guarantee zero leakage into system files.
- **Modern Responsive UI:**
  - Dark-mode HTML/JS client with real-time status notifications and retrieved chunk scoring visualization.

---

## 📚 Theory Requirements & Conceptual Answers

### 1. Semantic Search vs. Keyword Search
- **Keyword Search (Lexical Matching):** Matches literal query words directly against text tokens using traditional frequency algorithms (e.g., BM25, TF-IDF). It fails to extract relevant answers when a query uses synonyms, context shifts, or different phrasings without exact character matches.
- **Semantic Search (Dense Vector Retrieval):** Converts text into high-dimensional numerical embeddings using deep learning models (`all-MiniLM-L6-v2`). It captures the underlying **intent, meaning, and contextual relationships** of phrases, enabling accurate retrieval even when no common keywords are shared between the query and source documents.

### 2. Prompting Techniques Implemented
- **Zero-Shot Prompting:** Evaluates LLM capability using direct system instructions without providing prior example outputs.
- **Few-Shot Prompting:** Demonstrates target output structure through concise contextual patterns to guide answer generation.
- **Role-Based Prompting:** Instructs the LLM to adopt a domain persona (e.g., HR Specialist / Document Analyst) for precise and professional responses.

---

## 🛠️ Project Architecture & File Structure

```text
├── app/
│   ├── rag_service.py       # Core RAG engine, ChromaDB setup, & Groq LLM integration
│   └── ...                  # Helper modules (chunker, document loader, embedding service)
├── data/                    # Base System Documents (5 Default Files)
│   ├── llm_governance.txt
│   ├── prompt_engineering.pdf
│   ├── rag_architecture.txt
│   ├── search_paradigms.md
│   └── vectordb_chroma.md
├── templates/               # Web Interface
│   └── index.html           # Main UI with responsive layout
├── api.py                   # FastAPI server endpoints (/upload, /rag/query)
├── requirements.txt         # Project dependencies
├── .env.example             # Environment variables template
└── README.md                # Project documentation  
```  







---

# 🚀 Local Installation & Setup Guide

### 1. Clone Repository

git clone https://github.com/palwashamushtaq123/AI-Powered-Document-Question-Answering-System-Intermediate.git
cd AI-Powered-Document-Question-Answering-System-Intermediate

### 2. Create and Activate Virtual Environment

- **python -m venv venv**
# On Windows:
- **venv\Scripts\activate**
# On Mac/Linux:
- **source venv/bin/activate**


### 3. Install Dependencies

- **pip install -r requirements.txt**

### 4. Configure Environment Variables

Create a .env file in the root directory and add your Groq API key:

- **GROQ_API_KEY=your_groq_api_key_here**




### 5. Run Application

- **uvicorn api:app --reload**

**Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your web browser.**


---

# 👤 Author

**Palwasha Sheikh**

AI & Machine Learning Assignment Submission