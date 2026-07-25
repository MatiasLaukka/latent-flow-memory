import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Return cosine similarity between normalized vectors.
    """
    return float(np.dot(first, second))


def main() -> None:
    encoder = TextEncoder()

    # The semantic location where this memory should activate.
    memory_center = encoder.encode(
        "Matu's cat's name"
    )

    # The target concept.
    target = encoder.encode(
        "Rufus"
    )

    # The stored flow direction.
    direction = target - memory_center

    # Test both relevant and unrelated queries.
    queries = [
        (
            "direct",
            "What is Matu's cat's name?",
        ),
        (
            "paraphrase",
            "What is the name of Matu's pet?",
        ),
        (
            "already-rufus",
            "Who is Rufus?",
        ),
        (
            "finland",
            "What is the capital of Finland?",
        ),
        (
            "france",
            "What is the capital of France?",
        ),
        (
            "generic-cat",
            "What animal says meow?",
        ),
    ]

    # Test increasingly broad memory fields.
    radii = [
        0.2,
        0.3,
        0.4,
        0.5,
        0.6,
        0.8,
        1.0,
    ]

    print()
    print("RADIUS SWEEP")
    print("=" * 96)

    print(
        f"{'Radius':>8}"
        f"{'Direct':>12}"
        f"{'Paraphrase':>12}"
        f"{'Rufus':>12}"
        f"{'Finland':>12}"
        f"{'France':>12}"
        f"{'Cat':>12}"
    )

    print("-" * 96)

    for radius in radii:
        memory = FlowMemory(
            center=memory_center,
            direction=direction,
            strength=1.0,
            radius=radius,
        )

        changes = {}

        for label, query_text in queries:
            query = encoder.encode(
                query_text
            )

            similarity_before = cosine_similarity(
                query,
                target,
            )

            final_state, _ = flow(
                start=query,
                memories=[memory],
                steps=10,
                step_size=0.1,
            )

            similarity_after = cosine_similarity(
                final_state,
                target,
            )

            changes[label] = (
                similarity_after
                - similarity_before
            )

        print(
            f"{radius:>8.2f}"
            f"{changes['direct']:>+12.4f}"
            f"{changes['paraphrase']:>+12.4f}"
            f"{changes['already-rufus']:>+12.4f}"
            f"{changes['finland']:>+12.4f}"
            f"{changes['france']:>+12.4f}"
            f"{changes['generic-cat']:>+12.4f}"
        )


if __name__ == "__main__":
    main()