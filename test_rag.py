from rag import analyze_proposal


# ============================================================
# TEST PROPOSAL
# ============================================================

proposal = """
We are developing an autonomous underwater robot for detecting
and collecting plastic waste from water bodies.

The system uses computer vision and AI-based plastic detection,
underwater robotics, sensors, motors and an embedded controller.

The project is being developed by undergraduate engineering
students at an academic institution.
"""


# ============================================================
# START
# ============================================================

print("=" * 70)
print("STARTING RESEARCHGRANT AI")
print("=" * 70)


try:

    answer, sources = analyze_proposal(proposal)

    # --------------------------------------------------------
    # GEMINI RESULT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RESEARCHGRANT AI RESULT")
    print("=" * 70)
    print()

    print(answer)


    # --------------------------------------------------------
    # RETRIEVED SOURCES
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("RETRIEVED SOURCES")
    print("=" * 70)

    for i, result in enumerate(sources, start=1):

        # Our new rag.py returns:
        #
        # {
        #     "grant": {...},
        #     "score": 0.xxxx
        # }

        grant = result["grant"]
        score = result["score"]

        print()
        print(f"SOURCE {i}")
        print("-" * 70)

        print(f"File: {grant['source']}")
        print(f"Similarity: {score:.4f}")
        print()

        print(grant["text"])


except Exception as e:

    print()
    print("=" * 70)
    print("ERROR")
    print("=" * 70)

    print(type(e).__name__)
    print(str(e))