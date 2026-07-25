import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Calculate cosine similarity.

    All embeddings are normalized, so cosine similarity
    equals the dot product.
    """
    return float(np.dot(first, second))


def print_rankings(
    state: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> None:
    """
    Print candidate concepts ordered by similarity
    to the current latent state.
    """

    scores = []

    for name, embedding in candidates.items():
        similarity = cosine_similarity(
            state,
            embedding,
        )

        scores.append(
            (name, similarity)
        )

    # Sort highest similarity first.
    scores.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    for name, similarity in scores:
        print(
            f"{name:<12} "
            f"{similarity:.4f}"
        )


def main() -> None:
    encoder = TextEncoder()

    # -----------------------------
    # 1. Define the learned memory.
    # -----------------------------

    memory_center = encoder.encode(
        "Matu's cat's name"
    )

    target = encoder.encode(
        "Rufus"
    )

    # Direction pointing from the semantic trigger region
    # toward the target concept.
    direction = (
        target
        - memory_center
    )

    memory = FlowMemory(
        center=memory_center,
        direction=direction,
        strength=1.0,
        radius=1.0,
    )

    # -----------------------------
    # 2. Define the query.
    # -----------------------------

    query_text = (
        "What is Matu's cat's name?"
    )

    query = encoder.encode(
        query_text
    )

    # -----------------------------
    # 3. Define candidate concepts.
    # -----------------------------

    candidate_names = [
        "Rufus",
        "Helsinki",
        "Paris",
        "dog",
        "cat",
        "Finland",
        "France",
    ]

    candidates = {
        name: encoder.encode(name)
        for name in candidate_names
    }

    # -----------------------------
    # 4. Measure baseline geometry.
    # -----------------------------

    print()
    print("QUERY")
    print(query_text)

    print()
    print("BEFORE FLOW")
    print("-" * 30)

    print_rankings(
        query,
        candidates,
    )

    rufus_before = cosine_similarity(
        query,
        target,
    )

    # -----------------------------
    # 5. Run latent dynamics.
    # -----------------------------

    final_state, trajectory = flow(
        start=query,
        memories=[memory],
        steps=10,
        step_size=0.1,
    )

    # -----------------------------
    # 6. Measure final geometry.
    # -----------------------------

    print()
    print("AFTER FLOW")
    print("-" * 30)

    print_rankings(
        final_state,
        candidates,
    )

    rufus_after = cosine_similarity(
        final_state,
        target,
    )

    print()
    print("RUFUS SIMILARITY")
    print("-" * 30)

    print(
        f"Before: {rufus_before:.4f}"
    )

    print(
        f"After:  {rufus_after:.4f}"
    )

    print(
        f"Change: "
        f"{rufus_after - rufus_before:+.4f}"
    )

    print()
    print(
        f"Trajectory contained "
        f"{len(trajectory)} states."
    )


if __name__ == "__main__":
    main()