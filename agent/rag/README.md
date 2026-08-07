# Aurelia Hotels & Resorts — RAG Architecture Evaluation

## 1. Project Overview

This project extends the **Aurelia Hotels & Resorts AI Assistant** by implementing and comparing three RAG architectures:

* **Hybrid Search RAG**
* **Agentic RAG**
* **Self-RAG**

The goal is to determine which architecture is most suitable for hotel staff by comparing:

* Answer accuracy
* Token usage per query
* Response latency
* Performance on different types of hotel-related questions

The systems use the Aurelia Hotels documents as the knowledge base.

---

# 2. RAG Architectures

## 2.1 Hybrid Search RAG

Hybrid Search combines:

* **Vector similarity search**
* **BM25 keyword search**

Vector search helps find documents based on semantic meaning, while BM25 is useful for exact identifiers such as:

* Room numbers
* Reservation IDs
* Policy IDs
* Names
* Specific keywords

### Architecture

```text
User Question
      |
      +------------------+
      |                  |
      v                  v
Vector Search          BM25
      |                  |
      +--------+---------+
               |
               v
        Combined Ranking
               |
               v
        Retrieved Context
               |
               v
              LLM
               |
               v
        Final Answer
```

---

# 3. Agentic RAG

Agentic RAG adds a reasoning loop around retrieval.

Instead of retrieving information only once, the agent decides whether it needs more information.

The agent can:

1. Analyze the question.
2. Decide what to retrieve.
3. Retrieve documents.
4. Observe the retrieved information.
5. Decide whether the information is sufficient.
6. Retrieve again if necessary.
7. Generate the final answer.

### Architecture

```text
User Question
      |
      v
    Reason
      |
      v
   Retrieve
      |
      v
   Observe
      |
      v
Enough Information?
    /       \
  No         Yes
  |           |
  v           v
Retrieve     Answer
 Again
```

Agentic RAG is particularly useful for questions that require information from several documents or policies.

---

# 4. Self-RAG

Self-RAG adds an explicit verification step before returning an answer.

After retrieving information and generating an answer, the system checks:

* Is the retrieved information relevant?
* Is the generated answer supported by the retrieved information?
* Did the model introduce unsupported information?

If the answer cannot be verified, the system can perform another retrieval or refuse to provide an unsupported answer.

### Architecture

```text
User Question
      |
      v
   Retrieve
      |
      v
 Generate Answer
      |
      v
    Verify
      |
   +--+--+
   |     |
 PASS   FAIL
   |     |
   v     v
Answer  Retrieve Again
```

---

# 5. Project Structure

```text
Aurelia-Hotels/
│
├── documents/
│   └── hotel documents
│
├── rag/
│   ├── __init__.py
│   │
│   ├── chunker.py
│   ├── embeddings.py
│   ├── vector_store.py
│   ├── bm25_store.py
│   ├── build_vector_db.py
│   ├── retrieve.py
│   │
│   ├── hybrid_retriever.py
│   ├── hybrid_rag.py
│   │
│   ├── reasoning.py
│   ├── agentic.py
│   │
│   ├── verifier.py
│   └── self_rag.py
│
├── vector_db/
│   └── generated indexes
│
├── requirements.txt
├── .env
└── README.md
```

---

# 6. Installation

Create a virtual environment:

```bash
python -m venv venv
```

Activate it on Windows:

```bash
venv\Scripts\activate
```

Install the required libraries:

```bash
pip install -r requirements.txt
```

---

# 7. Environment Variables

Create a `.env` file in the project root:

```text
GEMINI_API_KEY=your_api_key_here
```

The API key is used by Gemini for:

* Answer generation
* Agentic reasoning
* Self-RAG verification

Do not commit the `.env` file to Git.

---

# 8. Preparing the Knowledge Base

Place the Aurelia Hotels documents inside the `documents` folder.

Example:

```text
documents/
├── reservation_policy.pdf
├── cancellation_policy.pdf
├── vip_policy.pdf
├── room_maintenance.pdf
├── guest_services.pdf
└── branch_transfer_policy.pdf
```

The exact documents depend on the files provided for the project.

---

# 9. Building the Database

The documents must first be processed and indexed.

The system creates:

* Document chunks
* Vector embeddings
* Vector database entries
* BM25 keyword index

Because the project uses package imports such as:

```python
from .chunker import load_documents
```

run the database builder from the project root:

```bash
python -m rag.build_vector_db
```

This is important because running the file directly can cause:

```text
ImportError: attempted relative import with no known parent package
```

---

# 10. Running Hybrid Search RAG

Run:

```bash
python -m rag.hybrid_rag
```

Hybrid RAG performs:

```text
Question
   |
   +---- Vector Search
   |
   +---- BM25 Search
   |
   v
Combined Results
   |
   v
Gemini
   |
   v
Answer
```

---

# 11. Running Agentic RAG

Run:

```bash
python -m rag.agentic
```

Agentic RAG may perform multiple retrieval rounds.

Example:

```text
Question:
Can a VIP guest transfer a reservation to another branch after check-in?

Step 1:
RETRIEVE
Search: VIP reservation transfer

Step 2:
RETRIEVE
Search: branch transfer policy

Step 3:
ANSWER
```

The maximum number of iterations is limited to prevent unnecessary retrieval and API calls.

---

# 12. Running Self-RAG

Run:

```bash
python -m rag.self_rag
```

Self-RAG follows:

```text
Retrieve
   |
Generate
   |
Verify
   |
   +---- PASS ----> Return Answer
   |
   +---- RETRIEVE -> Search Again
   |
   +---- FAIL ----> Do Not Trust Answer
```

This prevents the system from blindly returning unsupported answers.

---

# 13. Evaluation Test Set

The test set contains questions designed to highlight the differences between the architectures.

Each architecture should be tested against **every question**.

This gives:

```text
8 questions × 4 architectures = 32 evaluations
```

---

## Q1 — Simple Lookup

**Question:**

> What is the hotel's check-in time?

### Expected strength

**Naive RAG**

### Reason

The information should be available directly in one relevant document. No complex reasoning or multiple retrieval rounds should be necessary.

---

## Q2 — Simple Policy Question

**Question:**

> What is the hotel's cancellation policy?

### Expected strength

**Naive RAG**

### Reason

This is a straightforward knowledge-base lookup.

---

## Q3 — Exact Policy Identifier

**Question:**

> What does policy HR-204 say about room maintenance?

### Expected strength

**Hybrid Search RAG**

### Reason

`HR-204` is an exact identifier. BM25 can match the exact identifier while vector search provides semantic matching.

---

## Q4 — Exact Room Identifier

**Question:**

> What is the status and policy associated with Room 1205?

### Expected strength

**Hybrid Search RAG**

### Reason

`1205` is an exact identifier. Keyword matching can be especially useful for this type of query.

---

## Q5 — Multi-Policy Question

**Question:**

> A VIP guest wants to transfer their reservation to another Aurelia branch after check-in. What should the hotel staff do?

### Expected strength

**Agentic RAG**

### Reason

The answer may require information from several areas:

* VIP policy
* Reservation policy
* Check-in policy
* Branch transfer policy

The agent can retrieve information in multiple rounds.

---

## Q6 — Multiple Retrieval Rounds

**Question:**

> A guest wants to cancel their reservation after checking in and also requests a refund. Which policies should staff consider?

### Expected strength

**Agentic RAG**

### Reason

The question combines cancellation and refund information and may require multiple retrieval operations before the system has enough evidence.

---

## Q7 — Verification Question

**Question:**

> Are all VIP guests guaranteed a free room upgrade regardless of availability?

### Expected strength

**Self-RAG**

### Reason

The system must verify whether the documents actually guarantee a room upgrade.

It should not assume that:

```text
VIP → Guaranteed Upgrade
```

unless the knowledge base explicitly supports that claim.

---

## Q8 — Unsupported Generalization

**Question:**

> Is breakfast completely free for every guest at every Aurelia branch?

### Expected strength

**Self-RAG**

### Reason

The system should verify whether the documents support such a broad statement rather than generating an unsupported answer.

---

# 14. Evaluation Method

Every architecture is tested using the same questions.

For each answer, record:

### Accuracy

```text
1.0 = Correct
0.5 = Partially correct
0.0 = Incorrect
```

### Latency

Measure the time from sending the query until the final answer is produced.

### Token Usage

Record the total number of input and output tokens used.

For architectures that perform multiple LLM calls, all calls should be included.

For example:

```text
Agentic RAG Token Usage =
Reasoning Tokens
+
Retrieval/Decision Tokens
+
Final Answer Tokens
```

---

# 15. Evaluation Results

The following numbers are **illustrative example results** showing the expected format. They are not claimed to be measurements from the current machine or documents.

| Architecture      | Accuracy | Avg. Tokens / Query | Avg. Latency / Query |
| ----------------- | -------: | ------------------: | -------------------: |
| Naive RAG         |      75% |               1,050 |                1.1 s |
| Hybrid Search RAG |    87.5% |               1,180 |                1.3 s |
| Agentic RAG       |   93.75% |               2,050 |                2.7 s |
| Self-RAG          |   96.25% |               2,350 |                3.1 s |

### Per-question illustrative results

| Question | Naive | Hybrid | Agentic | Self-RAG |
| -------- | ----: | -----: | ------: | -------: |
| Q1       |     ✓ |      ✓ |       ✓ |        ✓ |
| Q2       |     ✓ |      ✓ |       ✓ |        ✓ |
| Q3       |     ✗ |      ✓ |       ✓ |        ✓ |
| Q4       |     ✗ |      ✓ |       ✓ |        ✓ |
| Q5       |     ~ |      ~ |       ✓ |        ✓ |
| Q6       |     ~ |      ✓ |       ✓ |        ✓ |
| Q7       |     ✗ |      ✗ |       ✓ |        ✓ |
| Q8       |     ✗ |      ✗ |       ✓ |        ✓ |

`✓` = correct
`~` = partially correct
`✗` = incorrect

---

# 16. Interpreting the Results

The expected trade-off is:

### Naive RAG

Advantages:

* Lowest latency
* Lowest token usage
* Simple architecture
* Good for straightforward questions

Disadvantages:

* Less effective with exact identifiers
* Only performs a single retrieval
* Does not explicitly verify answers

---

### Hybrid Search RAG

Advantages:

* Better exact-match retrieval
* Combines semantic and keyword search
* Useful for room numbers and policy IDs

Disadvantages:

* More expensive than simple vector retrieval
* Still normally performs only one retrieval round
* Does not independently verify generated answers

---

### Agentic RAG

Advantages:

* Can break complex questions into retrieval steps
* Can search again when information is missing
* Better for multi-policy questions

Disadvantages:

* Higher latency
* Higher token usage
* More Gemini calls
* More complex architecture

---

### Self-RAG

Advantages:

* Checks whether retrieved information is relevant
* Checks whether the answer is supported
* Helps reduce unsupported answers
* Can retrieve again when verification fails

Disadvantages:

* Highest latency in many cases
* Higher token usage
* Requires an additional verification step
* More API calls

---

# 17. Choosing the Architecture for Aurelia Hotels

The architecture should not be selected simply because it is the most advanced.

The decision should consider Aurelia Hotels' actual query patterns.

Hotel staff may ask:

* Simple policy questions
* Questions involving room numbers
* Questions involving reservation IDs
* VIP-related questions
* Questions requiring several hotel policies
* Questions where an incorrect answer could cause a service problem

Based on the illustrative evaluation above:

```text
Naive RAG
75% accuracy
1.1 s latency
1,050 tokens

Hybrid RAG
87.5% accuracy
1.3 s latency
1,180 tokens

Agentic RAG
93.75% accuracy
2.7 s latency
2,050 tokens

Self-RAG
96.25% accuracy
3.1 s latency
2,350 tokens
```

Self-RAG gives the highest accuracy, but it also has the highest latency and token usage.

For Aurelia Hotels, a reasonable production choice would therefore depend on the importance of **answer reliability versus speed and cost**.

If hotel staff frequently ask complex questions and unsupported answers are costly, **Self-RAG** is a strong choice.

If the majority of queries involve exact room numbers, reservation IDs, or policy identifiers, while fast responses are important, **Hybrid Search RAG** may provide a better balance.

If the evaluation shows that complex multi-policy questions are common, **Agentic RAG** may be the best compromise between accuracy and cost.

The final deployment decision should be based on the measured evaluation results rather than the architecture's complexity.

---

# 18. Final Comparison

The three architectures demonstrate different improvements:

```text
Naive RAG
    ↓
Basic retrieval
    ↓
Good for simple questions

Hybrid RAG
    ↓
Vector + BM25
    ↓
Better exact and semantic retrieval

Agentic RAG
    ↓
Reason + Retrieve Again
    ↓
Better for complex questions

Self-RAG
    ↓
Retrieve + Generate + Verify
    ↓
Better protection against unsupported answers
```

The purpose of the evaluation is therefore to determine which trade-off best matches the real needs of **Aurelia Hotels & Resorts**.
