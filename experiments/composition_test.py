from src.baselines import (
    kernel_weighted_target,
    nearest_target,
    one_step_target_flow,
)
from src.encoder import TextEncoder
from src.evaluation import (
    rank_candidates,
    trajectory_similarities,
)
from src.relations import (
    TargetMemory,
    target_flow,
)


def create_target_memories(
    encoder: TextEncoder,
    trigger_texts: list[str],
    target_text: str,
    radius: float = 0.35,
) -> list[TargetMemory]:
    """
    Create multiple narrow semantic triggers that all
    point toward one target.
    """

    target = encoder.encode(
        target_text
    )

    memories = []

    for trigger_text in trigger_texts:
        center = encoder.encode(
            trigger_text
        )

        memories.append(
            TargetMemory(
                center=center,
                target=target,
                strength=1.0,
                radius=radius,
            )
        )

    return memories


def print_final_ranking(
    method_name: str,
    state,
    candidates,
) -> None:
    print()
    print(method_name)
    print("-" * 50)

    for name, score in rank_candidates(
        state=state,
        candidates=candidates,
    ):
        print(
            f"{name:<12} "
            f"{score:.4f}"
        )


def print_trajectory(
    trajectory,
    candidates,
) -> None:
    """
    Print the similarities relevant to the two-hop chains.
    """

    similarities = trajectory_similarities(
        trajectory=trajectory,
        candidates=candidates,
    )

    print()
    print("ITERATIVE TRAJECTORY")
    print("=" * 78)

    print(
        f"{'Step':>6}"
        f"{'Acme':>12}"
        f"{'Globex':>12}"
        f"{'Helsinki':>14}"
        f"{'Paris':>12}"
    )

    print("-" * 78)

    for step, scores in enumerate(
        similarities
    ):
        print(
            f"{step:>6}"
            f"{scores['Acme']:>12.4f}"
            f"{scores['Globex']:>12.4f}"
            f"{scores['Helsinki']:>14.4f}"
            f"{scores['Paris']:>12.4f}"
        )


def run_query(
    encoder: TextEncoder,
    query_text: str,
    memories: list[TargetMemory],
    candidates: dict,
) -> None:
    query = encoder.encode(
        query_text
    )

    print()
    print("=" * 78)
    print(f"QUERY: {query_text}")
    print("=" * 78)

    nearest_result = nearest_target(
        query=query,
        memories=memories,
    )

    kernel_result = kernel_weighted_target(
        query=query,
        memories=memories,
    )

    one_step_result = one_step_target_flow(
        query=query,
        memories=memories,
        step_size=0.1,
    )

    iterative_result, trajectory = target_flow(
        start=query,
        memories=memories,
        steps=50,
        step_size=0.1,
    )

    print_final_ranking(
        method_name="NEAREST TRIGGER",
        state=nearest_result,
        candidates=candidates,
    )

    print_final_ranking(
        method_name="KERNEL-WEIGHTED TARGET",
        state=kernel_result,
        candidates=candidates,
    )

    print_final_ranking(
        method_name="ONE-STEP FLOW",
        state=one_step_result,
        candidates=candidates,
    )

    print_final_ranking(
        method_name="ITERATIVE FLOW",
        state=iterative_result,
        candidates=candidates,
    )

    print_trajectory(
        trajectory=trajectory,
        candidates=candidates,
    )


def main() -> None:
    encoder = TextEncoder()

    alice_to_acme = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Alice's employer",
            "company Alice works for",
            "where Alice works",
            "Alice's workplace",
        ],
        target_text="Acme",
    )

    acme_to_helsinki = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Acme's headquarters",
            "where Acme is headquartered",
            "location of Acme",
            "Acme headquarters city",
        ],
        target_text="Helsinki",
    )

    bob_to_globex = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Bob's employer",
            "company Bob works for",
            "where Bob works",
            "Bob's workplace",
        ],
        target_text="Globex",
    )

    globex_to_paris = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Globex's headquarters",
            "where Globex is headquartered",
            "location of Globex",
            "Globex headquarters city",
        ],
        target_text="Paris",
    )

    memories = (
        alice_to_acme
        + acme_to_helsinki
        + bob_to_globex
        + globex_to_paris
    )

    candidate_names = [
        "Alice",
        "Bob",
        "Acme",
        "Globex",
        "Helsinki",
        "Paris",
        "Finland",
        "France",
    ]

    candidates = {
        name: encoder.encode(name)
        for name in candidate_names
    }

    run_query(
        encoder=encoder,
        query_text=(
            "Where is Alice's workplace "
            "headquartered?"
        ),
        memories=memories,
        candidates=candidates,
    )

    run_query(
        encoder=encoder,
        query_text=(
            "Where is Bob's workplace "
            "headquartered?"
        ),
        memories=memories,
        candidates=candidates,
    )


if __name__ == "__main__":
    main()