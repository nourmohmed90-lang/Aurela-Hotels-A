import os
from google import genai
from dotenv import load_dotenv

load_dotenv()


class ReasoningEngine:

    def __init__(
        self,
        api_key: str = None,
        model: str = "gemini-2.5-flash"
    ):
        resolved_key = api_key or os.getenv("GEMINI_API_KEY")
        if not resolved_key:
            raise ValueError("GEMINI_API_KEY must be provided or set in environment variables.")

        self.client = genai.Client(api_key=resolved_key)
        self.model = model

    def decide(
        self,
        question: str,
        current_context: str
    ) -> dict:
        prompt = f"""
You are the reasoning module of an Agentic RAG system.
Your job is NOT to answer the user's question directly.

Decide whether:
1. More retrieval is needed to answer accurately.
2. There is already enough information in the context.

If more retrieval is needed, output:
RETRIEVE
SEARCH: <search query>

If enough information exists, output:
ANSWER

--------------------------
User Question
--------------------------
{question}

--------------------------
Retrieved Context
--------------------------
{current_context}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        text = response.text.strip()

        if text.startswith("ANSWER"):
            return {"action": "ANSWER", "query": None}

        if text.startswith("RETRIEVE"):
            search_query = question
            for line in text.splitlines():
                if line.upper().startswith("SEARCH:"):
                    search_query = line.split(":", 1)[1].strip()

            return {"action": "RETRIEVE", "query": search_query}

        return {"action": "ANSWER", "query": None}