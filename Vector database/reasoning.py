import os
from google import genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

class ReasoningEngine:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash"
    ):

        self.client = genai.Client(api_key=api_key)
        self.model = model

    # Decide the next action

    def decide(
        self,
        question: str,
        current_context: str
    ) -> dict:

        prompt = f"""
You are the reasoning module of an Agentic RAG system.

Your job is NOT to answer the user's question.

Instead, decide whether:

1. More retrieval is needed.
2. There is already enough information.

If more retrieval is needed:

Return

RETRIEVE
SEARCH: <search query>

If enough information exists:

Return

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

            return {
                "action": "ANSWER",
                "query": None
            }

        if text.startswith("RETRIEVE"):

            search_query = question

            for line in text.splitlines():

                if line.upper().startswith("SEARCH:"):

                    search_query = line.split(
                        ":", 1
                    )[1].strip()

            return {
                "action": "RETRIEVE",
                "query": search_query
            }

        # Default fallback

        return {
            "action": "ANSWER",
            "query": None
        }


reasoning_engine = None
