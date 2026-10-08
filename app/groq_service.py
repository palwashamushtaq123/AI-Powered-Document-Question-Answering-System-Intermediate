from groq import Groq
from .config import settings

class GroqService:
    def __init__(self) -> None:
        settings.validate()
        self.client = Groq(api_key=settings.groq_api_key)

    def generate_answer(self, question: str, context: str, prompt_type: str = "role_based") -> str:
        if prompt_type == "zero_shot":
            # 1. Zero-shot Prompting: Direct question without context styling or examples
            messages = [
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        elif prompt_type == "few_shot":
            # 2. Few-shot Prompting: Provides examples of expected format
            messages = [
                {
                    "role": "system",
                    "content": "Answer the question based only on the provided context. Follow the example formats."
                },
                {
                    "role": "user",
                    "content": "Context: Python was created in 1991.\nQuestion: When was Python created?"
                },
                {
                    "role": "assistant",
                    "content": "Python was created in 1991."
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nQuestion: {question}"
                }
            ]
        else:
            # 3. Role-based Prompting (Default)
            messages = [
                {
                    "role": "system",
                    "content": (
                        "You are a Senior Technical Document Analyst. "
                        "Analyze the provided retrieved context thoroughly and answer the user question accurately. "
                        "Do not invent facts. If information is missing, state: 'I could not find enough information in the supplied documents.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"RETRIEVED CONTEXT:\n{context}\n\nUSER QUESTION:\n{question}",
                },
            ]

        response = self.client.chat.completions.create(
            model=settings.groq_model,
            messages=messages,
            temperature=0.1,
        )

        return response.choices[0].message.content or ""