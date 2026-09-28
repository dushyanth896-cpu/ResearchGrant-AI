"""
Retrieval evaluation for ResearchGrant AI.

This script measures how well the semantic-retrieval step (the
embedding + cosine-similarity search in rag.py) surfaces the right
grant(s) for a proposal — independent of Gemini's downstream
analysis, which is not evaluated here.

It runs a small hand-labeled set of sample proposals through
`retrieve_grants()` and reports standard retrieval metrics:

- Hit@K   : was at least one relevant grant in the top K results?
- MRR     : mean reciprocal rank of the first relevant grant.

Run with:
    python evaluate.py
"""

from rag import load_index, retrieve_grants


# ============================================================
# LABELED EVALUATION SET
#
# Each case pairs a sample proposal with the grant ID(s) (see the
# "id" field in the grant_*.json files) that are genuinely relevant
# to it. Add more cases here as the grant catalog grows.
# ============================================================

EVAL_SET = [
    {
        "proposal": (
            "We are developing an autonomous underwater robot for "
            "detecting and collecting plastic waste from water "
            "bodies. The system uses computer vision, AI-based "
            "plastic detection, underwater robotics, sensors, "
            "motors and an embedded controller. The project is "
            "being developed by undergraduate engineering students "
            "at an academic institution."
        ),
        "relevant_ids": [6, 2, 3],
    },
    {
        "proposal": (
            "Our team is building an IoT-based smart irrigation "
            "system that uses soil-moisture sensors and a machine "
            "learning model to optimize water usage on small farms, "
            "improving crop yield while reducing waste."
        ),
        "relevant_ids": [5],
    },
    {
        "proposal": (
            "We propose a low-cost air-quality monitoring network "
            "using distributed IoT sensors and an AI model to "
            "predict pollution hotspots in urban areas, with "
            "real-time dashboards for city planners."
        ),
        "relevant_ids": [8],
    },
    {
        "proposal": (
            "This project designs a mobile app and community "
            "collection program that rewards households for "
            "recycling e-waste, paired with a small material-"
            "recovery facility that reclaims metals for reuse."
        ),
        "relevant_ids": [7],
    },
    {
        "proposal": (
            "A student team is prototyping a solar-powered weather "
            "station that combines renewable energy harvesting with "
            "environmental sensors to monitor microclimates on "
            "campus, as a step toward a functional field-ready "
            "device."
        ),
        "relevant_ids": [9, 4],
    },
]


K_VALUES = (1, 3, 5)


# ============================================================
# METRICS
# ============================================================

def evaluate_case(case, grants, embeddings, max_k):
    """
    Retrieve the top `max_k` grants for a single case and return
    the ranked list of grant IDs plus the rank (1-indexed) of the
    first relevant grant, or None if none of the top `max_k` were
    relevant.
    """

    results = retrieve_grants(
        case["proposal"],
        grants,
        embeddings,
        top_k=max_k
    )

    retrieved_ids = []

    for result in results:
        data = result["grant"].get("data", {})
        retrieved_ids.append(data.get("id"))

    relevant = set(case["relevant_ids"])

    first_hit_rank = None

    for rank, grant_id in enumerate(retrieved_ids, start=1):
        if grant_id in relevant:
            first_hit_rank = rank
            break

    return retrieved_ids, first_hit_rank


def main():

    print("=" * 70)
    print("RESEARCHGRANT AI — RETRIEVAL EVALUATION")
    print("=" * 70)
    print()

    grants, embeddings = load_index()

    max_k = max(K_VALUES)

    hits_at_k = {k: 0 for k in K_VALUES}
    reciprocal_ranks = []

    for i, case in enumerate(EVAL_SET, start=1):

        retrieved_ids, first_hit_rank = evaluate_case(
            case, grants, embeddings, max_k
        )

        print(f"Case {i}")
        print("-" * 70)
        print(f"Proposal   : {case['proposal'][:90].strip()}...")
        print(f"Expected   : {case['relevant_ids']}")
        print(f"Retrieved  : {retrieved_ids}")

        if first_hit_rank is not None:
            print(f"First hit  : rank {first_hit_rank}")
            reciprocal_ranks.append(1.0 / first_hit_rank)
        else:
            print("First hit  : none in top "
                  f"{max_k}")
            reciprocal_ranks.append(0.0)

        for k in K_VALUES:
            if first_hit_rank is not None and first_hit_rank <= k:
                hits_at_k[k] += 1

        print()

    n_cases = len(EVAL_SET)

    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)

    for k in K_VALUES:
        hit_rate = hits_at_k[k] / n_cases
        print(f"Hit@{k:<2}: {hit_rate:.2%} "
              f"({hits_at_k[k]}/{n_cases} cases)")

    mrr = sum(reciprocal_ranks) / n_cases
    print(f"MRR   : {mrr:.3f}")
    print()


if __name__ == "__main__":
    main()
