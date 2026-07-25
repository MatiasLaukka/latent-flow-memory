from dataclasses import dataclass
from time import perf_counter

import numpy as np

from experiments.composition_robustness import (
    alice_curated_queries,
    bob_curated_queries,
    create_target_memories,
    generated_queries,
)
from src.encoder import TextEncoder
from src.relations import (
    TargetMemory,
    target_flow,
)
from src.robustness import (
    QueryCase,
    QueryResult,
    evaluate_trajectory,
)
from src.scaling import (
    DistractorChain,
    ScalingSummary,
    fixed_distractor_pool,
    summarize_load,
)


LOAD_LEVELS = [
    0,
    3,
    8,
    18,
    48,
]


@dataclass(frozen=True)
class LoadResult:
    """
    Complete output from one distractor load.
    """

    summary: ScalingSummary
    query_results: list[QueryResult]
    runtimes_seconds: list[float]
    wrong_city_similarities: list[float]

def create_distractor_memories(
    encoder: TextEncoder,
    chains: list[DistractorChain],
) -> list[TargetMemory]:
    """
    Convert complete distractor chains into the same
    four-trigger-per-relation memory format as the targets.
    """

    memories: list[TargetMemory] = []

    for chain in chains:
        person_to_company = create_target_memories(
            encoder=encoder,
            trigger_texts=[
                f"{chain.person}'s employer",
                (
                    f"company {chain.person} "
                    "works for"
                ),
                f"where {chain.person} works",
                f"{chain.person}'s workplace",
            ],
            target_text=chain.company,
        )

        company_to_city = create_target_memories(
            encoder=encoder,
            trigger_texts=[
                f"{chain.company}'s headquarters",
                (
                    f"where {chain.company} "
                    "is headquartered"
                ),
                f"location of {chain.company}",
                (
                    f"{chain.company} "
                    "headquarters city"
                ),
            ],
            target_text=chain.city,
        )

        memories.extend(
            person_to_company
        )

        memories.extend(
            company_to_city
        )

    return memories

def benchmark_cases() -> list[QueryCase]:
    """
    Reuse the complete 72-query paraphrase benchmark.
    """

    return (
        alice_curated_queries()
        + bob_curated_queries()
        + generated_queries(
            person="Alice",
            intermediate="Acme",
            destination="Helsinki",
            chain_name="alice",
        )
        + generated_queries(
            person="Bob",
            intermediate="Globex",
            destination="Paris",
            chain_name="bob",
        )
    )


def create_candidates(
    encoder: TextEncoder,
    distractor_chains: list[DistractorChain],
) -> dict[str, np.ndarray]:
    """
    Build candidates for target concepts plus every
    distractor person, company, and city.
    """

    names = {
        "Alice",
        "Bob",
        "Acme",
        "Globex",
        "Helsinki",
        "Paris",
        "Finland",
        "France",
    }

    for chain in distractor_chains:
        names.add(chain.person)
        names.add(chain.company)
        names.add(chain.city)

    return {
        name: encoder.encode(name)
        for name in sorted(names)
    }


def city_candidate_names(
    distractor_chains: list[DistractorChain],
) -> set[str]:
    """
    Return only city candidates for second-hop
    interference measurements.
    """

    return {
        "Helsinki",
        "Paris",
        *[
            chain.city
            for chain in distractor_chains
        ],
    }

def strongest_wrong_city_similarity(
    final_state: np.ndarray,
    expected_destination: str,
    city_names: set[str],
    candidates: dict[str, np.ndarray],
) -> float:
    """
    Return the highest final similarity to an incorrect
    city candidate.
    """

    wrong_city_scores = [
        float(
            np.dot(
                final_state,
                candidates[city_name],
            )
        )
        for city_name in city_names
        if city_name != expected_destination
    ]

    if not wrong_city_scores:
        raise ValueError(
            "At least one wrong city candidate is required."
        )

    return max(
        wrong_city_scores
    )

def run_load(
    *,
    encoder: TextEncoder,
    target_memories: list[TargetMemory],
    distractor_pool: list[DistractorChain],
    distractor_count: int,
    cases: list[QueryCase],
) -> LoadResult:
    """
    Run all benchmark queries at one cumulative
    distractor load.
    """

    active_distractors = distractor_pool[
        :distractor_count
    ]

    distractor_memories = (
        create_distractor_memories(
            encoder=encoder,
            chains=active_distractors,
        )
    )

    memories = (
        target_memories
        + distractor_memories
    )

    candidates = create_candidates(
        encoder=encoder,
        distractor_chains=active_distractors,
    )

    cities = city_candidate_names(
        distractor_chains=active_distractors,
    )

    query_results: list[QueryResult] = []
    runtimes_seconds: list[float] = []
    wrong_city_similarities: list[float] = []

    for case in cases:
        query = encoder.encode(
            case.text
        )

        start_time = perf_counter()

        final_state, trajectory = target_flow(
            start=query,
            memories=memories,
            steps=50,
            step_size=0.1,
        )

        runtime_seconds = (
            perf_counter()
            - start_time
        )

        result = evaluate_trajectory(
            case=case,
            trajectory=trajectory,
            candidates=candidates,
        )

        wrong_city_similarity = (
            strongest_wrong_city_similarity(
                final_state=final_state,
                expected_destination=(
                    case.expected_destination
                ),
                city_names=cities,
                candidates=candidates,
            )
        )

        query_results.append(result)
        runtimes_seconds.append(
            runtime_seconds
        )
        wrong_city_similarities.append(
            wrong_city_similarity
        )

    summary = summarize_load(
        distractor_count=distractor_count,
        results=query_results,
        runtimes_seconds=runtimes_seconds,
        wrong_city_similarities=(
            wrong_city_similarities
        ),
    )

    return LoadResult(
        summary=summary,
        query_results=query_results,
        runtimes_seconds=runtimes_seconds,
        wrong_city_similarities=(
            wrong_city_similarities
        ),
    )

def display_optional_number(
    value: float | int | None,
    decimals: int = 2,
) -> str:
    if value is None:
        return "never"

    if isinstance(value, int):
        return str(value)

    return f"{value:.{decimals}f}"


def print_summary_header() -> None:
    print()
    print("=" * 132)
    print("FIXED-POOL DISTRACTOR SCALING")
    print("=" * 132)

    print(
        f"{'Distractors':>12}"
        f"{'Correct':>10}"
        f"{'Total':>8}"
        f"{'Accuracy':>12}"
        f"{'Avg margin':>14}"
        f"{'Min margin':>14}"
        f"{'Avg gain':>12}"
        f"{'Avg takeover':>15}"
        f"{'Max takeover':>15}"
        f"{'No takeover':>14}"
        f"{'Wrong city':>13}"
        f"{'Runtime ms':>13}"
    )

    print("-" * 132)


def print_summary_row(
    summary: ScalingSummary,
) -> None:
    average_takeover = display_optional_number(
        summary.average_takeover_step,
    )

    maximum_takeover = display_optional_number(
        summary.maximum_takeover_step,
    )

    print(
        f"{summary.distractor_count:>12}"
        f"{summary.correct_count:>10}"
        f"{summary.total_count:>8}"
        f"{summary.accuracy:>12.3f}"
        f"{summary.average_destination_margin:>14.4f}"
        f"{summary.minimum_destination_margin:>14.4f}"
        f"{summary.average_intermediate_gain:>12.4f}"
        f"{average_takeover:>15}"
        f"{maximum_takeover:>15}"
        f"{summary.no_takeover_count:>14}"
        f"{summary.average_wrong_city_similarity:>13.4f}"
        f"{summary.average_runtime_seconds * 1000:>13.3f}"
    )


def print_weakest_queries(
    load_result: LoadResult,
    limit: int = 5,
) -> None:
    weakest = sorted(
        load_result.query_results,
        key=lambda result: (
            result.destination_margin
        ),
    )[:limit]

    print()
    print(
        "Weakest queries at "
        f"{load_result.summary.distractor_count} "
        "distractors"
    )

    print("-" * 110)

    for result in weakest:
        takeover = display_optional_number(
            result.destination_takeover_step,
        )

        print(
            f"{result.destination_margin:>8.4f}  "
            f"{result.winner:<18}"
            f"{takeover:>8}  "
            f"{result.case.text}"
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

    target_memories = (
        alice_to_acme
        + acme_to_helsinki
        + bob_to_globex
        + globex_to_paris
    )

    distractor_pool = fixed_distractor_pool()
    cases = benchmark_cases()

    load_results = []

    for distractor_count in LOAD_LEVELS:
        print(
            "Running load with "
            f"{distractor_count} distractors..."
        )

        load_result = run_load(
            encoder=encoder,
            target_memories=target_memories,
            distractor_pool=distractor_pool,
            distractor_count=distractor_count,
            cases=cases,
        )

        load_results.append(
            load_result
        )

    print_summary_header()

    for load_result in load_results:
        print_summary_row(
            load_result.summary
        )

    for load_result in load_results:
        print_weakest_queries(
            load_result=load_result,
            limit=5,
        )


if __name__ == "__main__":
    main()