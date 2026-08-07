from google import genai

from .reasoning import ReasoningEngine
from .hybrid_retriever import hybrid_retriever
from dotenv import load_dotenv
import os

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")


class AgenticRAG:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash",
        max_iterations: int = 3
    ):

        self.client = genai.Client(api_key=api_key)

        self.reasoner = ReasoningEngine(
            api_key=api_key,
            model=model
        )

        self.model = model
        self.max_iterations = max_iterations


    # Final Answer Prompt

    def build_prompt(
        self,
        question: str,
        context: str
    ):

        return f"""
You are an AI assistant for Aurelia Hotels & Resorts.

Answer the user's question ONLY using the retrieved context.

If the answer cannot be found, say:

"I couldn't find this information in the hotel knowledge base."

--------------------------
Retrieved Context
--------------------------

{context}

--------------------------
Question
--------------------------

{question}

--------------------------
Answer
--------------------------
"""

    # Agentic Loop

    def ask(
        self,
        question: str,
        k: int = 3
    ):

        context = ""

        history = []

        retrieved_queries = set()

        for step in range(self.max_iterations):

            decision = self.reasoner.decide(
                question=question,
                current_context=context
            )

            history.append(
                {
                    "step": step + 1,
                    "decision": decision
                }
            )

            if decision["action"] == "ANSWER":
                break

            query = decision["query"]

            if query in retrieved_queries:
                break

            retrieved_queries.add(query)

            new_context = hybrid_retriever.retrieve_context(
                query=query,
                k=k
            )

            if not new_context.strip():
                break

            context += "\n\n" + new_context

        prompt = self.build_prompt(
            question,
            context
        )

        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        return {
            "answer": response.text,
            "context": context,
            "reasoning": history
        }

    # Interactive Chat

    def chat(self):

        print("=" * 60)
        print("Aurelia Hotels Agentic RAG")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:

            question = input("\nQuestion: ")

            if question.lower() == "exit":
                break

            result = self.ask(question)

            print("\nAnswer\n")
            print(result["answer"])

            print("\nReasoning\n")

            for item in result["reasoning"]:

                print(item)


if __name__ == "__main__":

    rag = AgenticRAG(API_KEY)

    rag.chat()
