from dataclasses import dataclass
from time import perf_counter
import re

import numpy as np

from experiments.composition_distractor_scaling import (
    LOAD_LEVELS,
    benchmark_cases,
)
from experiments.linked_flow_branching import (
    build_graph,
    create_linked_memory,
)
from src.encoder import TextEncoder
from src.evaluation import rank_candidates
from src.linked_flow import (
    RoutingStatus,
    run_linked_flow,
)
from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    rank_memories_by_triggers,
)
from src.robustness import QueryCase
from src.scaling import (
    DistractorChain,
    fixed_distractor_pool,
)


GLOBAL_BASELINE_ACCURACY = {
    0: 1.000,
    3: 1.000,
    8: 0.514,
    18: 0.000,
    48: 0.000,
}


@dataclass(frozen=True)
class LinkedScalingSummary:
    """
    Aggregate linked-flow results for one distractor load.
    """

    distractor_count: int
    correct_count: int
    total_count: int
    accuracy: float

    correct_root_count: int
    root_accuracy: float

    correct_successor_count: int
    successor_accuracy: float

    exact_route_count: int
    exact_route_accuracy: float

    average_root_margin: float
    average_runtime_seconds: float


def slug(text: str) -> str:
    """
    Convert a human-readable concept into a stable memory-ID
    fragment.
    """

    cleaned = re.sub(
        r"[^a-z0-9]+",
        "-",
        text.lower(),
    )

    return cleaned.strip("-")


def create_distractor_chain_memories(
    *,
    encoder: TextEncoder,
    chain: DistractorChain,
    works_at,
    headquartered_in,
) -> tuple[LinkedMemory, LinkedMemory]:
    """
    Convert one distractor fact chain into two linked memories:

        person --works-at--> company
        company --headquartered-in--> city
    """

    root_id = (
        f"{slug(chain.person)}-works-at-"
        f"{slug(chain.company)}"
    )

    headquarters_id = (
        f"{slug(chain.company)}-headquartered-in-"
        f"{slug(chain.city)}"
    )

    root = create_linked_memory(
        encoder=encoder,
        memory_id=root_id,
        source=chain.person,
        relation=works_at,
        target=chain.company,
        trigger_phrases=[
            f"{chain.person}'s employer",
            f"company {chain.person} works for",
            f"where {chain.person} works",
            f"{chain.person}'s workplace",
        ],
        successor_ids=(
            headquarters_id,
        ),
    )

    headquarters = create_linked_memory(
        encoder=encoder,
        memory_id=headquarters_id,
        source=chain.company,
        relation=headquartered_in,
        target=chain.city,
        trigger_phrases=[
            f"{chain.company}'s headquarters",
            (
                f"where {chain.company} "
                "is headquartered"
            ),
            f"location of {chain.company}",
            f"{chain.company} headquarters city",
        ],
    )

    return (
        root,
        headquarters,
    )


def build_scaling_graph(
    *,
    encoder: TextEncoder,
    distractors: list[DistractorChain],
) -> LinkedMemoryGraph:
    """
    Build the original Alice/Bob linked graph and add a
    requested number of distractor chains.

    Only person-to-company memories are graph roots.
    """

    target_graph = build_graph(
        encoder=encoder,
    )

    works_at = target_graph.get(
        "alice-works-at-acme"
    ).relation

    headquartered_in = target_graph.get(
        "acme-headquartered-in-helsinki"
    ).relation

    memories = list(
        target_graph.memories
    )

    root_ids = list(
        target_graph.root_ids
    )

    for chain in distractors:
        (
            distractor_root,
            distractor_headquarters,
        ) = create_distractor_chain_memories(
            encoder=encoder,
            chain=chain,
            works_at=works_at,
            headquartered_in=headquartered_in,
        )

        memories.append(
            distractor_root
        )

        memories.append(
            distractor_headquarters
        )

        root_ids.append(
            distractor_root.memory_id
        )

    return LinkedMemoryGraph(
        memories=tuple(memories),
        root_ids=tuple(root_ids),
    )


def create_candidates(
    *,
    encoder: TextEncoder,
    graph: LinkedMemoryGraph,
) -> dict[str, np.ndarray]:
    """
    Create decoding candidates from every source and target
    concept in the active graph.
    """

    names: set[str] = set()

    for memory in graph.memories:
        names.add(
            memory.source
        )
        names.add(
            memory.target
        )

    return {
        name: encoder.encode(name)
        for name in sorted(names)
    }


def expected_root_id(
    case: QueryCase,
) -> str:
    """
    Return the correct first memory for one benchmark case.
    """

    if case.chain_name == "alice":
        return "alice-works-at-acme"

    if case.chain_name == "bob":
        return "bob-works-at-globex"

    raise ValueError(
        "Unknown target chain: "
        f"{case.chain_name}"
    )


def expected_successor_id(
    case: QueryCase,
) -> str:
    """
    Return the correct headquarters memory for one case.
    """

    if case.chain_name == "alice":
        return (
            "acme-headquartered-in-helsinki"
        )

    if case.chain_name == "bob":
        return (
            "globex-headquartered-in-paris"
        )

    raise ValueError(
        "Unknown target chain: "
        f"{case.chain_name}"
    )


def root_selection_margin(
    *,
    graph: LinkedMemoryGraph,
    query: np.ndarray,
) -> float:
    """
    Measure how clearly the best root beats the second-best
    root.

    Larger is better.
    """

    rankings = rank_memories_by_triggers(
        memories=graph.root_memories(),
        query=query,
    )

    if len(rankings) < 2:
        return 0.0

    return (
        rankings[0].score
        - rankings[1].score
    )


def run_load(
    *,
    encoder: TextEncoder,
    distractor_pool: list[DistractorChain],
    distractor_count: int,
    cases: list[QueryCase],
) -> LinkedScalingSummary:
    """
    Run the complete benchmark at one cumulative distractor
    load.
    """

    active_distractors = (
        distractor_pool[
            :distractor_count
        ]
    )

    graph = build_scaling_graph(
        encoder=encoder,
        distractors=active_distractors,
    )

    candidates = create_candidates(
        encoder=encoder,
        graph=graph,
    )

    correct_count = 0
    correct_root_count = 0
    correct_successor_count = 0
    exact_route_count = 0

    root_margins: list[float] = []
    runtimes_seconds: list[float] = []

    for case in cases:
        query = encoder.encode(
            case.text
        )

        root_margins.append(
            root_selection_margin(
                graph=graph,
                query=query,
            )
        )

        start_time = perf_counter()

        result = run_linked_flow(
            query=query,
            graph=graph,
            handoff_threshold=0.95,
            max_steps_per_hop=50,
            step_size=0.1,
            max_hops=2,
        )

        runtime_seconds = (
            perf_counter()
            - start_time
        )

        runtimes_seconds.append(
            runtime_seconds
        )

        ranking = rank_candidates(
            state=result.final_state,
            candidates=candidates,
        )

        winner = ranking[0][0]

        wanted_root = expected_root_id(
            case
        )

        wanted_successor = (
            expected_successor_id(
                case
            )
        )

        actual_route = (
            result.traversed_memory_ids
        )

        root_correct = (
            len(actual_route) >= 1
            and actual_route[0]
            == wanted_root
        )

        successor_correct = (
            len(actual_route) >= 2
            and actual_route[1]
            == wanted_successor
        )

        exact_route_correct = (
            actual_route
            == (
                wanted_root,
                wanted_successor,
            )
        )

        final_correct = (
            result.status
            == RoutingStatus.COMPLETED
            and winner
            == case.expected_destination
        )

        correct_root_count += int(
            root_correct
        )

        correct_successor_count += int(
            successor_correct
        )

        exact_route_count += int(
            exact_route_correct
        )

        correct_count += int(
            final_correct
        )

    total_count = len(
        cases
    )

    return LinkedScalingSummary(
        distractor_count=(
            distractor_count
        ),
        correct_count=(
            correct_count
        ),
        total_count=(
            total_count
        ),
        accuracy=(
            correct_count
            / total_count
        ),
        correct_root_count=(
            correct_root_count
        ),
        root_accuracy=(
            correct_root_count
            / total_count
        ),
        correct_successor_count=(
            correct_successor_count
        ),
        successor_accuracy=(
            correct_successor_count
            / total_count
        ),
        exact_route_count=(
            exact_route_count
        ),
        exact_route_accuracy=(
            exact_route_count
            / total_count
        ),
        average_root_margin=float(
            np.mean(
                root_margins
            )
        ),
        average_runtime_seconds=float(
            np.mean(
                runtimes_seconds
            )
        ),
    )


def print_header() -> None:
    print()
    print("=" * 112)

    print(
        "LINKED-FLOW FIXED-POOL "
        "DISTRACTOR SCALING"
    )

    print("=" * 112)

    print(
        f"{'Distractors':>12}"
        f"{'Global':>10}"
        f"{'Linked':>10}"
        f"{'Root':>10}"
        f"{'Successor':>12}"
        f"{'Exact route':>14}"
        f"{'Root margin':>14}"
        f"{'Runtime ms':>14}"
    )

    print("-" * 112)


def print_summary(
    summary: LinkedScalingSummary,
) -> None:
    global_accuracy = (
        GLOBAL_BASELINE_ACCURACY[
            summary.distractor_count
        ]
    )

    print(
        f"{summary.distractor_count:>12}"
        f"{global_accuracy:>10.3f}"
        f"{summary.accuracy:>10.3f}"
        f"{summary.root_accuracy:>10.3f}"
        f"{summary.successor_accuracy:>12.3f}"
        f"{summary.exact_route_accuracy:>14.3f}"
        f"{summary.average_root_margin:>14.4f}"
        f"{summary.average_runtime_seconds * 1000:>14.3f}"
    )


def main() -> None:
    print(
        "Loading sentence encoder..."
    )

    encoder = TextEncoder()

    distractor_pool = (
        fixed_distractor_pool()
    )

    cases = benchmark_cases()

    print(
        f"Benchmark queries: {len(cases)}"
    )

    print(
        f"Available distractors: "
        f"{len(distractor_pool)}"
    )

    summaries: list[
        LinkedScalingSummary
    ] = []

    for distractor_count in (
        LOAD_LEVELS
    ):
        print()
        print(
            "Running linked flow with "
            f"{distractor_count} "
            "distractor chains..."
        )

        summary = run_load(
            encoder=encoder,
            distractor_pool=(
                distractor_pool
            ),
            distractor_count=(
                distractor_count
            ),
            cases=cases,
        )

        summaries.append(
            summary
        )

        print(
            f"  correct: "
            f"{summary.correct_count}/"
            f"{summary.total_count}"
        )

        print(
            f"  root correct: "
            f"{summary.correct_root_count}/"
            f"{summary.total_count}"
        )

        print(
            f"  exact route: "
            f"{summary.exact_route_count}/"
            f"{summary.total_count}"
        )

    print_header()

    for summary in summaries:
        print_summary(
            summary
        )

    print()
    print(
        "Global = previous summed-field baseline."
    )

    print(
        "Linked = final-answer accuracy from "
        "the new linked-flow architecture."
    )

    print(
        "Root = correct person-to-company "
        "memory selected."
    )

    print(
        "Successor = correct headquarters "
        "successor selected."
    )

    print(
        "Exact route = both linked memories "
        "were correct."
    )


if __name__ == "__main__":
    main()