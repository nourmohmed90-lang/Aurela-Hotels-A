from google import genai


class Verifier:

    def __init__(
        self,
        api_key: str,
        model: str = "gemini-2.5-flash"
    ):

        self.client = genai.Client(api_key=api_key)
        self.model = model

    # Verify Answer

    def verify(
        self,
        question: str,
        context: str,
        answer: str
    ) -> dict:

        prompt = f"""
You are the verification module of a Self-RAG system.

Your task is NOT to answer the question.

Instead, verify whether the generated answer is fully supported
by the retrieved context.

Evaluate the following:

1. Is the retrieved context relevant to the user's question?
2. Is every important claim in the answer supported by the context?
3. Does the answer invent facts that are not present in the context?

Return ONLY one of these formats.

If the answer is completely supported:

VERDICT: PASS

If more retrieval may help:

VERDICT: RETRIEVE
REASON: <short reason>

If the answer is unsupported:

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

            return {
                "verdict": "PASS",
                "reason": ""
            }

        if "VERDICT: RETRIEVE" in text:

            reason = ""

            for line in text.splitlines():

                if line.startswith("REASON:"):

                    reason = line.replace(
                        "REASON:",
                        ""
                    ).strip()

            return {
                "verdict": "RETRIEVE",
                "reason": reason
            }

        if "VERDICT: FAIL" in text:

            reason = ""

            for line in text.splitlines():

                if line.startswith("REASON:"):

                    reason = line.replace(
                        "REASON:",
                        ""
                    ).strip()

            return {
                "verdict": "FAIL",
                "reason": reason
            }

        # Default fallback

        return {
            "verdict": "FAIL",
            "reason": "Unable to verify the answer."
        }


verifier = None
