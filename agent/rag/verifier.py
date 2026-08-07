import os
from google import genai


class Verifier:

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

    def verify(
        self,
        question: str,
        context: str,
        answer: str
    ) -> dict:
        prompt = f"""
You are the verification module of a Self-RAG system.
Verify whether the generated answer is fully supported by the retrieved context.

Evaluate the following:
1. Is the retrieved context relevant to the user's question?
2. Is every important claim in the answer supported by the context?
3. Does the answer invent facts that are not present in the context?

Output strictly in one of these formats:

VERDICT: PASS

VERDICT: RETRIEVE
REASON: <short reason>

VERDICT: FAIL
REASON: <short reason>

----------------------------------------
Question
----------------------------------------
{question}

----------------------------------------
Retrieved Context
----------------------------------------
{context}

----------------------------------------
Generated Answer
----------------------------------------
{answer}
"""
        response = self.client.models.generate_content(
            model=self.model,
            contents=prompt
        )

        text = response.text.strip()

        if "VERDICT: PASS" in text:
            return {"verdict": "PASS", "reason": ""}

        if "VERDICT: RETRIEVE" in text:
            reason = ""
            for line in text.splitlines():
                if line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            return {"verdict": "RETRIEVE", "reason": reason}

        if "VERDICT: FAIL" in text:
            reason = ""
            for line in text.splitlines():
                if line.startswith("REASON:"):
                    reason = line.replace("REASON:", "").strip()
            return {"verdict": "FAIL", "reason": reason}

        return {"verdict": "FAIL", "reason": "Unable to verify the answer."}