import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    return float(np.dot(first, second))


def create_memories(
    encoder: TextEncoder,
    trigger_texts: list[str],
    target_text: str,
    radius: float = 0.35,
) -> list[FlowMemory]:
    """
    Create several narrow trigger memories that all point
    toward the same target concept.
    """

    target = encoder.encode(target_text)
    memories = []

    for trigger_text in trigger_texts:
        center = encoder.encode(trigger_text)
        direction = target - center

        memories.append(
            FlowMemory(
                center=center,
                direction=direction,
                strength=1.0,
                radius=radius,
            )
        )

    return memories


def main() -> None:
    encoder = TextEncoder()

    rufus_target = encoder.encode("Rufus")
    helsinki_target = encoder.encode("Helsinki")

    rufus_memories = create_memories(
        encoder=encoder,
        trigger_texts=[
            "Matu's cat's name",
            "name of Matu's cat",
            "Matu's pet's name",
            "what Matu calls his cat",
        ],
        target_text="Rufus",
    )

    helsinki_memories = create_memories(
        encoder=encoder,
        trigger_texts=[
            "Finland's capital",
            "capital of Finland",
            "Finnish capital city",
            "the capital city of Finland",
        ],
        target_text="Helsinki",
    )

    all_memories = (
        rufus_memories
        + helsinki_memories
    )

    queries = [
        (
            "rufus-direct",
            "What is Matu's cat's name?",
            "Rufus",
        ),
        (
            "rufus-paraphrase",
            "What does Matu call his cat?",
            "Rufus",
        ),
        (
            "helsinki-direct",
            "What is the capital of Finland?",
            "Helsinki",
        ),
        (
            "helsinki-paraphrase",
            "Name Finland's capital city.",
            "Helsinki",
        ),
        (
            "france-control",
            "What is the capital of France?",
            "none",
        ),
    ]

    print()
    print("INTERFERENCE TEST")
    print("=" * 100)

    print(
        f"{'Query':<24}"
        f"{'Rufus before':>14}"
        f"{'Rufus after':>14}"
        f"{'Helsinki before':>17}"
        f"{'Helsinki after':>16}"
    )

    print("-" * 100)

    for label, query_text, expected in queries:
        query = encoder.encode(query_text)

        rufus_before = cosine_similarity(
            query,
            rufus_target,
        )

        helsinki_before = cosine_similarity(
            query,
            helsinki_target,
        )

        final_state, _ = flow(
            start=query,
            memories=all_memories,
            steps=10,
            step_size=0.1,
        )

        rufus_after = cosine_similarity(
            final_state,
            rufus_target,
        )

        helsinki_after = cosine_similarity(
            final_state,
            helsinki_target,
        )

        print(
            f"{label:<24}"
            f"{rufus_before:>14.4f}"
            f"{rufus_after:>14.4f}"
            f"{helsinki_before:>17.4f}"
            f"{helsinki_after:>16.4f}"
        )


if __name__ == "__main__":
    main()