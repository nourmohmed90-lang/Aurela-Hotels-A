from google import genai

from .hybrid_retriever import hybrid_retriever
from .verifier import Verifier


class SelfRAG:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_attempts: int = 3
    ):

        self.client = genai.Client(api_key=api_key)

        self.model = model

        self.max_attempts = max_attempts

        self.verifier = Verifier(
            api_key=api_key,
            model=model
        )

    # Prompt

    def build_prompt(
        self,
        question: str,
        context: str
    ):

        return f"""
You are an AI assistant for Aurelia Hotels & Resorts.

Answer ONLY using the retrieved context.

If the answer is not found in the context,
say that you cannot find the information.

-------------------------
Retrieved Context
-------------------------

{context}

-------------------------
Question
-------------------------

{question}

-------------------------
Answer
-------------------------
"""

    # Ask

    def ask(
        self,
        question: str,
        k: int = 3
    ):

        retrieval_query = question

        context = ""

        verification_history = []

        for attempt in range(self.max_attempts):

            # Retrieve

            context = hybrid_retriever.retrieve_context(
                query=retrieval_query,
                k=k
            )

            if not context.strip():

                return {
                    "answer": "No relevant documents were found.",
                    "context": "",
                    "verification": verification_history
                }

            # Generate

            prompt = self.build_prompt(
                question,
                context
            )

            response = self.client.models.generate_content(
                model=self.model,
                contents=prompt
            )

            answer = response.text

            # Verify

            result = self.verifier.verify(
                question=question,
                context=context,
                answer=answer
            )

            verification_history.append(result)

            if result["verdict"] == "PASS":

                return {
                    "answer": answer,
                    "context": context,
                    "verification": verification_history
                }

            if result["verdict"] == "FAIL":

                return {
                    "answer": (
                        "I couldn't verify the generated answer "
                        "using the hotel knowledge base."
                    ),
                    "context": context,
                    "verification": verification_history
                }

            if result["verdict"] == "RETRIEVE":

                retrieval_query = (
                    question +
                    " " +
                    result["reason"]
                )

        return {
            "answer": (
                "I couldn't confidently answer the question "
                "after multiple retrieval attempts."
            ),
            "context": context,
            "verification": verification_history
        }

    # Interactive Chat

    def chat(self):

        print("=" * 60)
        print("Aurelia Hotels Self-RAG")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:

            question = input("\nQuestion: ")

            if question.lower() == "exit":
                break

            result = self.ask(question)

            print("\nAnswer\n")
            print(result["answer"])

            print("\nVerification\n")

            for step, item in enumerate(
                result["verification"],
                start=1
            ):

                print(f"Attempt {step}: {item}")


if __name__ == "__main__":

    from dotenv import load_dotenv
    import os

    load_dotenv()

    API_KEY = os.getenv("GEMINI_API_KEY")

    rag = SelfRAG(API_KEY)

    rag.chat()
