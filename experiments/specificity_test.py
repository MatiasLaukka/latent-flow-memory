import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Return cosine similarity between two normalized vectors.

    Because our encoder normalizes every embedding, cosine
    similarity is just the dot product.
    """
    return float(np.dot(first, second))


def main() -> None:
    # Load the frozen sentence encoder.
    encoder = TextEncoder()

    # ---------------------------------------------------------
    # 1. Recreate the same learned memory as before.
    # ---------------------------------------------------------

    memory_center = encoder.encode(
        "Matu's cat's name"
    )

    target = encoder.encode(
        "Rufus"
    )

    direction = target - memory_center

    memory = FlowMemory(
        center=memory_center,
        direction=direction,
        strength=1.0,
        radius=1.0,
    )

    # ---------------------------------------------------------
    # 2. Define queries with different relationships
    #    to the learned memory.
    # ---------------------------------------------------------

    queries = [
        # Directly related to the learned memory.
        "What is Matu's cat's name?",

        # A paraphrase that should ideally activate the memory.
        "What is the name of Matu's pet?",

        # Related to Rufus, but asks a different relationship.
        "Who is Rufus?",

        # Completely unrelated geography questions.
        "What is the capital of Finland?",
        "What is the capital of France?",

        # Related to cats in general, but not specifically
        # to Matu or Rufus.
        "What animal says meow?",
    ]

    print()
    print("LATENT FLOW SPECIFICITY TEST")
    print("=" * 78)

    print(
        f"{'Query':<40}"
        f"{'Before':>10}"
        f"{'After':>10}"
        f"{'Change':>10}"
    )

    print("-" * 78)

    # ---------------------------------------------------------
    # 3. Test every query against the same memory.
    # ---------------------------------------------------------

    for query_text in queries:
        query = encoder.encode(
            query_text
        )

        # Measure how close the original query is to Rufus.
        similarity_before = cosine_similarity(
            query,
            target,
        )

        # Let the query move through the learned field.
        final_state, trajectory = flow(
            start=query,
            memories=[memory],
            steps=10,
            step_size=0.1,
        )

        # Measure how close the final state is to Rufus.
        similarity_after = cosine_similarity(
            final_state,
            target,
        )

        change = (
            similarity_after
            - similarity_before
        )

        print(
            f"{query_text:<40}"
            f"{similarity_before:>10.4f}"
            f"{similarity_after:>10.4f}"
            f"{change:>+10.4f}"
        )


if __name__ == "__main__":
    main()