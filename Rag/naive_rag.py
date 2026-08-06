import os

from google import genai
from google.genai import types

from .retriever import retriever
from .prompts import SYSTEM_PROMPT


class NaiveRAG:

    def __init__(self):

        self.client = genai.Client(
            api_key=os.getenv("GEMINI_API_KEY")
        )

        self.model = "gemini-2.5-flash"

    def ask(
        self,
        question: str,
        source: str | None = None
    ):

        context = retriever.retrieve_context(
            query=question,
            source=source
        )

        if context == "":

            return {

                "answer":
                "I couldn't find this information in the Aurelia Hotels knowledge base.",

                "sources": []
            }

        prompt = f"""

Retrieved Context

{context}

----------------------

User Question

{question}

"""

        response = self.client.models.generate_content(

            model=self.model,

            contents=prompt,

            config=types.GenerateContentConfig(

                system_instruction=SYSTEM_PROMPT

            )

        )

        docs = retriever.retrieve(
            query=question,
            source=source
        )

        return {

            "answer": response.text,

            "sources": [
                d["metadata"]["source"]
                for d in docs
            ],

            "documents": docs

        }


rag = NaiveRAG()