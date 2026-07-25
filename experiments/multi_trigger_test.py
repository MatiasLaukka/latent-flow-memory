import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """Return cosine similarity between normalized vectors."""
    return float(np.dot(first, second))


def main() -> None:
    encoder = TextEncoder()

    target = encoder.encode("Rufus")

    # Several semantic descriptions of the same memory trigger.
    trigger_texts = [
        "Matu's cat's name",
        "name of Matu's cat",
        "Matu's pet's name",
        "what Matu calls his cat",
    ]

    memories = []

    for trigger_text in trigger_texts:
        center = encoder.encode(trigger_text)

        # Every trigger points toward the same target.
        direction = target - center

        memories.append(
            FlowMemory(
                center=center,
                direction=direction,
                strength=1.0,
                radius=0.35,
            )
        )

    queries = [
        (
            "direct",
            "What is Matu's cat's name?",
        ),
        (
            "paraphrase-1",
            "What is the name of Matu's pet?",
        ),
        (
            "paraphrase-2",
            "What does Matu call his cat?",
        ),
        (
            "paraphrase-3",
            "Tell me the name of Matu's feline.",
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

    print()
    print("MULTI-TRIGGER MEMORY TEST")
    print("=" * 84)

    print(
        f"{'Query':<42}"
        f"{'Before':>10}"
        f"{'After':>10}"
        f"{'Change':>10}"
    )

    print("-" * 84)

    for label, query_text in queries:
        query = encoder.encode(query_text)

        before = cosine_similarity(
            query,
            target,
        )

        final_state, _ = flow(
            start=query,
            memories=memories,
            steps=10,
            step_size=0.1,
        )

        after = cosine_similarity(
            final_state,
            target,
        )

        print(
            f"{label:<42}"
            f"{before:>10.4f}"
            f"{after:>10.4f}"
            f"{after - before:>+10.4f}"
        )


if __name__ == "__main__":
    main()