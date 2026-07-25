import numpy as np

from src.encoder import TextEncoder
from src.evaluation import rank_candidates
from src.linked_flow import (
    RoutingStatus,
    run_linked_flow,
)
from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    RelationIntent,
)


def embed_phrases(
    encoder: TextEncoder,
    phrases: list[str],
) -> tuple[np.ndarray, ...]:
    """
    Embed several alternative phrases using the shared
    frozen sentence encoder.
    """

    return tuple(
        encoder.encode(phrase)
        for phrase in phrases
    )


def create_relation_intent(
    *,
    encoder: TextEncoder,
    name: str,
    phrases: list[str],
) -> RelationIntent:
    """
    Create one typed relation with several query-intent
    phrase embeddings.
    """

    return RelationIntent(
        name=name,
        phrase_embeddings=embed_phrases(
            encoder=encoder,
            phrases=phrases,
        ),
    )


def create_linked_memory(
    *,
    encoder: TextEncoder,
    memory_id: str,
    source: str,
    relation: RelationIntent,
    target: str,
    trigger_phrases: list[str],
    successor_ids: tuple[str, ...] = (),
    radius: float = 0.35,
    strength: float = 1.0,
) -> LinkedMemory:
    """
    Create one linked transition memory from natural-language
    source, relation, target, and trigger descriptions.
    """

    return LinkedMemory(
        memory_id=memory_id,
        source=source,
        relation=relation,
        target=target,
        trigger_centers=embed_phrases(
            encoder=encoder,
            phrases=trigger_phrases,
        ),
        target_vector=encoder.encode(target),
        successor_ids=successor_ids,
        radius=radius,
        strength=strength,
    )


def build_graph(
    encoder: TextEncoder,
) -> LinkedMemoryGraph:
    """
    Construct a branching graph for Alice and Bob.
    """

    works_at = create_relation_intent(
        encoder=encoder,
        name="works-at",
        phrases=[
            "works at",
            "employer",
            "company someone works for",
            "workplace",
        ],
    )

    headquartered_in = create_relation_intent(
        encoder=encoder,
        name="headquartered-in",
        phrases=[
            "headquartered in",
            "headquarters location",
            "where is it based",
            "main office city",
        ],
    )

    founded_by = create_relation_intent(
        encoder=encoder,
        name="founded-by",
        phrases=[
            "founded by",
            "who founded it",
            "company founder",
            "original creator",
        ],
    )

    operates_in = create_relation_intent(
        encoder=encoder,
        name="operates-in",
        phrases=[
            "operates in",
            "where does it operate",
            "operating region",
            "service area",
        ],
    )

    memories = (
        create_linked_memory(
            encoder=encoder,
            memory_id="alice-works-at-acme",
            source="Alice",
            relation=works_at,
            target="Acme",
            trigger_phrases=[
                "Alice's employer",
                "company Alice works for",
                "where Alice works",
                "Alice's workplace",
            ],
            successor_ids=(
                "acme-headquartered-in-helsinki",
                "acme-founded-by-maria",
                "acme-operates-in-finland",
            ),
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="acme-headquartered-in-helsinki",
            source="Acme",
            relation=headquartered_in,
            target="Helsinki",
            trigger_phrases=[
                "Acme headquarters",
                "where Acme is based",
                "Acme main office",
                "Acme headquarters city",
            ],
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="acme-founded-by-maria",
            source="Acme",
            relation=founded_by,
            target="Maria",
            trigger_phrases=[
                "Acme founder",
                "who founded Acme",
                "Acme was founded by",
                "creator of Acme",
            ],
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="acme-operates-in-finland",
            source="Acme",
            relation=operates_in,
            target="Finland",
            trigger_phrases=[
                "where Acme operates",
                "Acme operating region",
                "Acme service area",
                "Acme operates in",
            ],
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="bob-works-at-globex",
            source="Bob",
            relation=works_at,
            target="Globex",
            trigger_phrases=[
                "Bob's employer",
                "company Bob works for",
                "where Bob works",
                "Bob's workplace",
            ],
            successor_ids=(
                "globex-headquartered-in-paris",
                "globex-founded-by-susan",
                "globex-operates-in-europe",
            ),
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="globex-headquartered-in-paris",
            source="Globex",
            relation=headquartered_in,
            target="Paris",
            trigger_phrases=[
                "Globex headquarters",
                "where Globex is based",
                "Globex main office",
                "Globex headquarters city",
            ],
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="globex-founded-by-susan",
            source="Globex",
            relation=founded_by,
            target="Susan",
            trigger_phrases=[
                "Globex founder",
                "who founded Globex",
                "Globex was founded by",
                "creator of Globex",
            ],
        ),
        create_linked_memory(
            encoder=encoder,
            memory_id="globex-operates-in-europe",
            source="Globex",
            relation=operates_in,
            target="Europe",
            trigger_phrases=[
                "where Globex operates",
                "Globex operating region",
                "Globex service area",
                "Globex operates in",
            ],
        ),
    )

    return LinkedMemoryGraph(
        memories=memories,
        root_ids=(
            "alice-works-at-acme",
            "bob-works-at-globex",
        ),
    )


def build_candidates(
    encoder: TextEncoder,
) -> dict[str, np.ndarray]:
    """
    Build a fixed candidate set for final-state decoding.
    """

    names = [
        "Alice",
        "Bob",
        "Acme",
        "Globex",
        "Helsinki",
        "Paris",
        "Maria",
        "Susan",
        "Finland",
        "Europe",
    ]

    return {
        name: encoder.encode(name)
        for name in names
    }


def print_result(
    *,
    query_text: str,
    expected: str,
    result,
    candidates: dict[str, np.ndarray],
) -> bool:
    """
    Print one human-readable routed-query report.
    """

    ranking = rank_candidates(
        state=result.final_state,
        candidates=candidates,
    )

    winner = ranking[0][0]

    route = " -> ".join(
        result.traversed_memory_ids
    )

    relations = " -> ".join(
        result.traversed_relation_names
    )

    passed = (
        result.status == RoutingStatus.COMPLETED
        and winner == expected
    )

    print("=" * 80)
    print(f"Query:    {query_text}")
    print(f"Status:   {result.status.value}")
    print(f"Route:    {route or '-'}")
    print(f"Relations:{relations or '-'}")
    print(
        "Root score:"
        f" {result.root_selection_score}"
    )
    print(f"Total steps: {result.total_steps}")

    for index, hop in enumerate(
        result.hop_records,
        start=1,
    ):
        print()
        print(f"Hop {index}")
        print(
            f"  Memory:   {hop.memory_id}"
        )
        print(
            f"  Relation: {hop.relation_name}"
        )
        print(
            f"  Source:   {hop.source}"
        )
        print(
            f"  Target:   {hop.target}"
        )
        print(
            "  Selection score:"
            f" {hop.selection_score:.4f}"
        )
        print(
            f"  Steps:    {hop.steps_used}"
        )

        if hop.successor_scores:
            print("  Successor scores:")

            for memory_id, score in (
                hop.successor_scores
            ):
                print(
                    f"    {memory_id}: "
                    f"{score:.4f}"
                )

    print()
    print(f"Winner:   {winner}")
    print(f"Expected: {expected}")
    print(
        "Top candidates:"
    )

    for name, score in ranking[:5]:
        print(
            f"  {name}: {score:.4f}"
        )

    print(
        "Result:   "
        + (
            "PASS"
            if passed
            else "FAIL"
        )
    )

    if result.failure_reason is not None:
        print(
            "Failure reason: "
            f"{result.failure_reason}"
        )

    return passed


def main() -> None:
    encoder = TextEncoder()

    graph = build_graph(
        encoder=encoder,
    )

    candidates = build_candidates(
        encoder=encoder,
    )

    queries = [
        (
            "Where is Alice's employer headquartered?",
            "Helsinki",
        ),
        (
            "Who founded Alice's employer?",
            "Maria",
        ),
        (
            "Where does Alice's employer operate?",
            "Finland",
        ),
        (
            "Where is Bob's employer headquartered?",
            "Paris",
        ),
        (
            "Who founded Bob's employer?",
            "Susan",
        ),
        (
            "Where does Bob's employer operate?",
            "Europe",
        ),
    ]

    pass_count = 0

    for query_text, expected in queries:
        result = run_linked_flow(
            query=encoder.encode(
                query_text
            ),
            graph=graph,
            handoff_threshold=0.95,
            max_steps_per_hop=50,
            step_size=0.1,
            max_hops=2,
        )

        passed = print_result(
            query_text=query_text,
            expected=expected,
            result=result,
            candidates=candidates,
        )

        pass_count += int(passed)

    print()
    print("=" * 80)
    print(
        f"Passed: {pass_count}/{len(queries)}"
    )


if __name__ == "__main__":
    main()