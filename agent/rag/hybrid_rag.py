import os
from dotenv import load_dotenv
from google import genai
from .hybrid_retriever import hybrid_retriever
    
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
 
class HybridRAG:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash"
    ):

        self.client = genai.Client(api_key=api_key)
        self.model = model

    # Build Prompt

    def build_prompt(
        self,
        question: str,
        context: str
    ) -> str:

        return f"""
You are an AI assistant for Aurelia Hotels & Resorts.

Your job is to answer hotel staff questions using ONLY the retrieved context.

Rules:
- Answer only from the provided context.
- If the answer is not found, say:
  "I couldn't find this information in the hotel knowledge base."
- Do not make up policies.
- Keep answers clear and professional.

=========================
Retrieved Context
=========================

{context}

=========================
Question
=========================

{question}

=========================
Answer
=========================
"""

    # Ask

    def ask(
        self,
        question: str,
        k: int = 3,
        source=None
    ):

        context = hybrid_retriever.retrieve_context(
            query=question,
            k=k,
            source=source
        )

        if not context.strip():

            return {
                "answer": "No relevant documents were found.",
                "context": ""
            }

        prompt = self.build_prompt(
            question=question,
            context=context
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return {
            "answer": response.text,
            "context": context
        }

    # Interactive Chat

    def chat(self):

        print("=" * 60)
        print("Aurelia Hotels Hybrid RAG")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:

            question = input("\nQuestion: ")

            if question.lower() == "exit":
                break

            result = self.ask(question)

            print("\nAnswer\n")
            print(result["answer"])


if __name__ == "__main__":

    API_KEY = os.getenv("GEMINI_API_KEY")

    rag = HybridRAG(API_KEY)

    rag.chat()
