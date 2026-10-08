import sys
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

try:
    from app.rag_service import RAGService
except ImportError:
    # Fallback import check
    import app.rag_service as rag_module
    print(f"Available items in app.rag_service: {dir(rag_module)}")
    raise

TEST_QUESTIONS = [
    "What is the main objective of this document?",
    "Summarize the key technical requirements mentioned.",
    "What are the limitations or constraints specified?",
    "Explain the architecture or setup process described.",
    "What are the recommended best practices?"
]

PROMPT_TYPES = ["zero_shot", "few_shot", "role_based"]

def run_benchmark():
    print("=== STARTING PROMPT ENGINEERING BENCHMARK ===")
    rag = RAGService()
    
    for idx, q in enumerate(TEST_QUESTIONS, 1):
        print(f"\n==========================================")
        print(f"QUESTION {idx}: {q}")
        print(f"==========================================")
        
        for p_type in PROMPT_TYPES:
            try:
                res = rag.ask(question=q, top_k=3, prompt_type=p_type)
                print(f"\n--- PROMPT TYPE: {p_type.upper()} ---")
                print(res.get("answer", "No response generated."))
            except Exception as err:
                print(f"\n--- PROMPT TYPE: {p_type.upper()} --- ERROR: {err}")

if __name__ == "__main__":
    run_benchmark()