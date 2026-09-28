import os
import pickle
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# ENVIRONMENT
# ============================================================

# Loads GOOGLE_API_KEY (and anything else) from a local .env file.
# Without this call, os.getenv("GOOGLE_API_KEY") only ever sees
# variables that happen to already be exported in the shell.
load_dotenv()


# ============================================================
# CONFIGURATION
# ============================================================

# Anchor paths to this file's location instead of the current
# working directory, so the app works the same whether it's
# launched with `streamlit run app.py` from this folder or
# from somewhere else entirely.
BASE_DIR = Path(__file__).resolve().parent
INDEX_FILE = BASE_DIR / "grant_index.pkl"

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Google retires Gemini model names on a rolling basis (Gemini 2.0
# is already gone, Gemini 2.5 is scheduled to be shut down on
# 16 Oct 2026). "gemini-3.5-flash" is the current stable, generally
# available flash-tier model as of this writing. If calls start
# failing with a "model not found" error, check
# https://ai.google.dev/gemini-api/docs/models for the current
# name, or switch to the auto-updating alias "gemini-flash-latest".
GEMINI_MODEL = "gemini-3.5-flash"

TOP_K = 5


# ============================================================
# LOAD EMBEDDING MODEL (once per process)
# ============================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(EMBEDDING_MODEL)


# ============================================================
# LOAD SAVED INDEX
# ============================================================

@lru_cache(maxsize=1)
def load_index():
    """
    Load the prebuilt grant index from disk.

    Cached with lru_cache so the ~9-grant pickle only gets read
    and deserialized once per process, instead of on every single
    call to analyze_proposal() (i.e. every button click in the
    Streamlit app).
    """

    if not INDEX_FILE.exists():
        raise FileNotFoundError(
            f"{INDEX_FILE} not found.\n"
            "Run build_index.py first."
        )

    print("Loading saved grant index...")

    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

    grants = index["grants"]
    embeddings = np.asarray(index["embeddings"])

    print(f"Loaded {len(grants)} grant documents.")
    print(f"Embedding shape: {embeddings.shape}")

    return grants, embeddings


# ============================================================
# RETRIEVE GRANTS
# ============================================================

def retrieve_grants(
    proposal,
    grants,
    embeddings,
    top_k=TOP_K
):

    query_embedding = embedding_model.encode(
        proposal,
        convert_to_numpy=True,
        normalize_embeddings=True
    )

    similarities = np.dot(
        embeddings,
        query_embedding
    )

    top_indices = np.argsort(
        similarities
    )[::-1][:top_k]

    results = []

    for index in top_indices:

        results.append({
            "grant": grants[index],
            "score": float(similarities[index])
        })

    return results


# ============================================================
# FORMAT GRANT DATA
# ============================================================

def build_context(results):

    context_parts = []

    for i, result in enumerate(results, start=1):

        grant = result["grant"]

        context_parts.append(
            f"""
SOURCE {i}
==================================================

Source file:
{grant["source"]}

Grant information:
{grant["text"]}

Similarity score:
{result["score"]:.4f}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# CREATE GEMINI
# ============================================================

@lru_cache(maxsize=1)
def create_llm():

    api_key = os.getenv("GOOGLE_API_KEY")

    if not api_key:
        raise RuntimeError(
            "GOOGLE_API_KEY is not set. Create a .env file "
            "(see .env.example) with your Google AI Studio key."
        )

    return ChatGoogleGenerativeAI(
        model=GEMINI_MODEL,
        google_api_key=api_key,
        temperature=0.1,
        max_retries=0
    )


# ============================================================
# GEMINI REQUEST
# ============================================================

def ask_gemini(llm, prompt, max_attempts=3):

    for attempt in range(1, max_attempts + 1):

        print(
            f"Sending request to Gemini "
            f"(attempt {attempt}/{max_attempts})..."
        )

        try:

            response = llm.invoke(prompt)

            return response.content

        except Exception as e:

            error_text = str(e)

            # ------------------------------------------------
            # QUOTA ERROR
            # ------------------------------------------------

            if (
                "429" in error_text
                or "RESOURCE_EXHAUSTED" in error_text
            ):

                raise RuntimeError(
                    "Gemini API quota exceeded. "
                    "Please wait for the quota to reset "
                    "or use an available model/quota."
                )

            # ------------------------------------------------
            # TEMPORARY SERVER ERROR
            # ------------------------------------------------

            if (
                "503" in error_text
                or "UNAVAILABLE" in error_text
            ):

                if attempt < max_attempts:

                    wait_time = 5 * attempt

                    print(
                        f"Gemini temporarily unavailable. "
                        f"Retrying in {wait_time} seconds..."
                    )

                    time.sleep(wait_time)

                    continue

            raise


# ============================================================
# MAIN RAG FUNCTION
# ============================================================

def analyze_proposal(proposal, top_k=TOP_K):

    # --------------------------------------------------------
    # LOAD SAVED INDEX
    # --------------------------------------------------------

    grants, embeddings = load_index()

    # --------------------------------------------------------
    # RETRIEVE RELEVANT GRANTS
    # --------------------------------------------------------

    results = retrieve_grants(
        proposal,
        grants,
        embeddings,
        top_k
    )

    # --------------------------------------------------------
    # PRINT RETRIEVED SOURCES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, result in enumerate(results, start=1):

        grant = result["grant"]

        print()
        print(f"SOURCE {i}")
        print("-" * 70)

        print(
            f"File: {grant['source']}"
        )

        print(
            f"Similarity: {result['score']:.4f}"
        )

        print()

        print(grant["text"])

    # --------------------------------------------------------
    # BUILD GEMINI CONTEXT
    # --------------------------------------------------------

    context = build_context(results)

    # --------------------------------------------------------
    # GEMINI PROMPT
    # --------------------------------------------------------

    prompt = f"""
You are ResearchGrant AI, a research grant matching
and proposal analysis assistant.

Analyze the user's research proposal ONLY against the
retrieved grant documents below.

IMPORTANT RULES:

1. Use ONLY the retrieved grant documents for grant-specific
   information.

2. Do not invent grant names, funding amounts, eligibility
   requirements, focus areas, objectives, or required documents.

3. Preserve terminology exactly as it appears in the source.

4. Do not change terms such as:
   Institution
   Applicant
   Eligibility
   Budget
   Proposal

5. Separate SOURCE FACTS from YOUR ANALYSIS.

6. Never claim official eligibility.

Use wording such as:

"The proposal appears to match the stated applicant category."

7. If information is absent from the proposal, write:

"Not specified in the provided proposal."

8. If information is absent from the grant documents, write:

"Not specified in the provided grant documents."

9. Every grant-specific statement must identify its source file.

10. Do not use outside knowledge.

11. Do not treat semantic similarity as proof of eligibility,
funding qualification, or grant approval.

12. Do not invent missing proposal information.

13. When describing alignment, explain the connection between
the proposal and the exact grant objective/focus area.

14. Do not state that an inference is explicitly stated in
the source.

USER RESEARCH PROPOSAL
==================================================

{proposal}

RETRIEVED GRANT DOCUMENTS
==================================================

{context}


Generate the following report:

# Notice

Include exactly this notice:

"Notice: This system provides an informational analysis only.
It does not make an official funding or eligibility decision,
nor does it guarantee eligibility or funding approval."


# 1. Potentially Relevant Funding Opportunities

For every retrieved grant include:

- Grant name
- Source file
- Funding amount

Do not invent funding information.


# 2. Why They Match

For each grant:

### Grant requirement / focus area

State the relevant focus area exactly as given in the source.

### Proposal match

Explain how the proposal relates to that focus area.

Clearly distinguish source fact from analysis.


# 3. Eligibility Requirements

For each grant:

- List the eligible applicant categories exactly as provided.
- Explain whether the applicant information explicitly stated
  in the proposal appears to fit those categories.

Do NOT make an official eligibility decision.


# 4. Missing Requirements or Information

Identify requirements from the retrieved grants that are not
present in the proposal.

Use:

"Not specified in the provided proposal."

Do not assume missing information exists.


# 5. Required Documents

For each grant:

List the required documents exactly as they appear in the
source JSON.


# 6. Proposal-Funding Alignment

For each grant:

- Grant objective
- Proposal alignment

The grant objective must come from the source.

The proposal alignment must be clearly identified as analysis.


# 7. Suggested Proposal Improvements

Suggest improvements ONLY when they are directly connected
to requirements stated in the retrieved grant documents.

Do not introduce unrelated recommendations.


# 8. Evidence and Sources

List every source file used.

Use the exact source filenames.


Keep the report factual, structured and concise.
"""

    # --------------------------------------------------------
    # CALL GEMINI
    # --------------------------------------------------------

    llm = create_llm()

    answer = ask_gemini(
        llm,
        prompt
    )

    return answer, results


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    proposal = """
    We are developing an autonomous underwater robot for detecting
    and collecting plastic waste from water bodies.

    The system uses computer vision and AI-based plastic detection,
    underwater robotics, sensors, motors and an embedded controller.

    The project is being developed by undergraduate engineering
    students at an academic institution.
    """

    answer, sources = analyze_proposal(
        proposal
    )

    print()
    print("=" * 70)
    print("RESEARCHGRANT AI RESULT")
    print("=" * 70)
    print()

    print(answer)
