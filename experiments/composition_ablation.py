from dataclasses import dataclass

import numpy as np

from src.encoder import TextEncoder
from src.evaluation import (
    rank_candidates,
    trajectory_similarities,
)
from src.relations import (
    TargetMemory,
    target_flow,
)


@dataclass
class AblationResult:
    """
    Stores the most important measurements from one
    ablation condition.
    """

    condition_name: str
    winner: str
    intermediate_peak_step: int
    destination_takeover_step: int | None
    intermediate_gain: float
    intermediate_final_similarity: float
    destination_final_similarity: float
    alternative_final_similarity: float


def create_target_memories(
    encoder: TextEncoder,
    trigger_texts: list[str],
    target_text: str,
    radius: float = 0.35,
    strength: float = 1.0,
) -> list[TargetMemory]:
    """
    Create several narrow semantic trigger regions that
    all move the latent state toward the same target.
    """

    target = encoder.encode(target_text)

    memories: list[TargetMemory] = []

    for trigger_text in trigger_texts:
        memories.append(
            TargetMemory(
                center=encoder.encode(trigger_text),
                target=target,
                strength=strength,
                radius=radius,
            )
        )

    return memories


def find_peak_step(
    similarities: list[dict[str, float]],
    candidate_name: str,
) -> int:
    """
    Return the trajectory step where one candidate reaches
    its highest similarity.
    """

    return max(
        range(len(similarities)),
        key=lambda step: similarities[step][candidate_name],
    )


def find_takeover_step(
    similarities: list[dict[str, float]],
    destination_name: str,
    intermediate_name: str,
    intermediate_peak_step: int,
) -> int | None:
    """
    Return the first step after the intermediate's peak
    where the destination becomes more similar than the
    intermediate.

    This ignores accidental destination dominance at
    step 0 before the first-hop concept has activated.
    """

    for step in range(
        intermediate_peak_step + 1,
        len(similarities),
    ):
        scores = similarities[step]

        if (
            scores[destination_name]
            > scores[intermediate_name]
        ):
            return step

    return None


def evaluate_condition(
    condition_name: str,
    query: np.ndarray,
    memories: list[TargetMemory],
    candidates: dict[str, np.ndarray],
    intermediate_name: str,
    destination_name: str,
    alternative_name: str,
    steps: int = 50,
    step_size: float = 0.1,
) -> AblationResult:
    """
    Run one memory condition and calculate its final result
    and trajectory-level handoff measurements.
    """

    final_state, trajectory = target_flow(
        start=query,
        memories=memories,
        steps=steps,
        step_size=step_size,
    )

    final_ranking = rank_candidates(
        state=final_state,
        candidates=candidates,
    )

    similarities = trajectory_similarities(
        trajectory=trajectory,
        candidates=candidates,
    )

    intermediate_peak_step = find_peak_step(
        similarities=similarities,
        candidate_name=intermediate_name,
    )

    intermediate_start_similarity = similarities[0][intermediate_name]

    intermediate_peak_similarity = similarities[intermediate_peak_step][intermediate_name]

    intermediate_gain = (
    intermediate_peak_similarity
    - intermediate_start_similarity
)

    destination_takeover_step = find_takeover_step(
        similarities=similarities,
        destination_name=destination_name,
        intermediate_name=intermediate_name,
        intermediate_peak_step=intermediate_peak_step,
    )

    final_scores = similarities[-1]

    return AblationResult(
    condition_name=condition_name,
    winner=final_ranking[0][0],
    intermediate_peak_step=intermediate_peak_step,
    destination_takeover_step=destination_takeover_step,
    intermediate_gain=intermediate_gain,
    intermediate_final_similarity=final_scores[
        intermediate_name
    ],
    destination_final_similarity=final_scores[
        destination_name
    ],
    alternative_final_similarity=final_scores[
        alternative_name
    ],
)


def print_result(
    result: AblationResult,
    intermediate_name: str,
    destination_name: str,
    alternative_name: str,
) -> None:
    """
    Print one compact result row.
    """

    takeover = (
        str(result.destination_takeover_step)
        if result.destination_takeover_step is not None
        else "never"
    )

    print(
        f"{result.condition_name:<24}"
        f"{result.winner:<12}"
        f"{result.intermediate_peak_step:>10}"
        f"{takeover:>12}"
        f"{result.intermediate_gain:>12.4f}"
        f"{result.intermediate_final_similarity:>12.4f}"
        f"{result.destination_final_similarity:>14.4f}"
        f"{result.alternative_final_similarity:>14.4f}"
)


def run_ablation_suite(
    query_text: str,
    encoder: TextEncoder,
    conditions: dict[str, list[TargetMemory]],
    candidates: dict[str, np.ndarray],
    intermediate_name: str,
    destination_name: str,
    alternative_name: str,
) -> None:
    """
    Run every causal ablation condition for one query.
    """

    query = encoder.encode(query_text)

    print()
    print("=" * 110)
    print(f"QUERY: {query_text}")
    print("=" * 110)

    print(
        f"{'Condition':<24}"
        f"{'Winner':<12}"
        f"{'Peak step':>10}"
        f"{'Takeover':>12}"
        f"{'Int. gain':>12}"
        f"{intermediate_name:>12}"
        f"{destination_name:>14}"
        f"{alternative_name:>14}"
)

    print("-" * 110)

    for condition_name, condition in conditions.items():
        expected_destination = condition[
            "expected_destination"
        ]

        memories = condition[
            "memories"
        ]

        expected_alternative = (
            alternative_name
            if expected_destination == destination_name
            else destination_name
        )

        result = evaluate_condition(
            condition_name=condition_name,
            query=query,
            memories=memories,
            candidates=candidates,
            intermediate_name=intermediate_name,
            destination_name=expected_destination,
            alternative_name=expected_alternative,
        )

        print_result(
            result=result,
            intermediate_name=intermediate_name,
            destination_name=expected_destination,
            alternative_name=expected_alternative,
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

    acme_to_paris = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Acme's headquarters",
            "where Acme is headquartered",
            "location of Acme",
            "Acme headquarters city",
        ],
        target_text="Paris",
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

    globex_to_helsinki = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Globex's headquarters",
            "where Globex is headquartered",
            "location of Globex",
            "Globex headquarters city",
        ],
        target_text="Helsinki",
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

    alice_conditions = {
        "full-chain": {
            "memories": (
                alice_to_acme
                + acme_to_helsinki
            ),
            "expected_destination": "Helsinki",
        },
        "first-hop-only": {
            "memories": alice_to_acme,
            "expected_destination": "Helsinki",
        },
        "second-hop-only": {
            "memories": acme_to_helsinki,
            "expected_destination": "Helsinki",
        },
        "shuffled-destination": {
            "memories": (
                alice_to_acme
                + acme_to_paris
            ),
            "expected_destination": "Paris",
        },
    }

    run_ablation_suite(
        query_text=(
            "Where is Alice's workplace headquartered?"
        ),
        encoder=encoder,
        conditions=alice_conditions,
        candidates=candidates,
        intermediate_name="Acme",
        destination_name="Helsinki",
        alternative_name="Paris",
    )

    bob_conditions = {
        "full-chain": {
            "memories": (
                bob_to_globex
                + globex_to_paris
            ),
            "expected_destination": "Paris",
        },
        "first-hop-only": {
            "memories": bob_to_globex,
            "expected_destination": "Paris",
        },
        "second-hop-only": {
            "memories": globex_to_paris,
            "expected_destination": "Paris",
        },
        "shuffled-destination": {
            "memories": (
                bob_to_globex
                + globex_to_helsinki
            ),
            "expected_destination": "Helsinki",
        },
    }

    run_ablation_suite(
        query_text=(
            "Where is Bob's workplace headquartered?"
        ),
        encoder=encoder,
        conditions=bob_conditions,
        candidates=candidates,
        intermediate_name="Globex",
        destination_name="Paris",
        alternative_name="Helsinki",
    )


if __name__ == "__main__":
    main()