from collections.abc import Iterable

from src.encoder import TextEncoder
from src.relations import (
    TargetMemory,
    target_flow,
)
from src.robustness import (
    QueryCase,
    QueryResult,
    evaluate_trajectory,
    summarize_results,
)


def create_target_memories(
    encoder: TextEncoder,
    trigger_texts: list[str],
    target_text: str,
    radius: float = 0.35,
) -> list[TargetMemory]:
    target = encoder.encode(target_text)

    return [
        TargetMemory(
            center=encoder.encode(trigger_text),
            target=target,
            strength=1.0,
            radius=radius,
        )
        for trigger_text in trigger_texts
    ]

def alice_curated_queries() -> list[QueryCase]:
    straightforward = [
        "Which city is Alice's employer based in?",
        "Where is the company Alice works for headquartered?",
        "What is the headquarters location of Alice's workplace?",
        "In what city is Alice's employer headquartered?",
        "Where are the headquarters of Alice's company?",
        "What city contains the headquarters of Alice's employer?",
        "Where is Alice's company based?",
        "Name the headquarters city of Alice's employer.",
    ]

    hard = [
        (
            "Alice is employed by a company. In which city "
            "is that company's main office?"
        ),
        (
            "Given Alice's employer, identify the city where "
            "its headquarters are located."
        ),
        (
            "The company employing Alice has its headquarters "
            "in what city?"
        ),
        (
            "Trace Alice to her employer and state where that "
            "organization is headquartered."
        ),
        (
            "Alice works for an organization whose central "
            "office is located in which city?"
        ),
        (
            "Determine the city serving as the headquarters "
            "of the business that employs Alice."
        ),
        (
            "Which city would you reach by following Alice's "
            "employment relationship to the company's headquarters?"
        ),
        (
            "Identify Alice's employer, then give the city of "
            "that employer's main headquarters."
        ),
    ]

    return [
        QueryCase(
            text=text,
            chain_name="alice",
            category="curated-straightforward",
            expected_intermediate="Acme",
            expected_destination="Helsinki",
        )
        for text in straightforward
    ] + [
        QueryCase(
            text=text,
            chain_name="alice",
            category="curated-hard",
            expected_intermediate="Acme",
            expected_destination="Helsinki",
        )
        for text in hard
    ]

def bob_curated_queries() -> list[QueryCase]:
    straightforward = [
        "Which city is Bob's employer based in?",
        "Where is the company Bob works for headquartered?",
        "What is the headquarters location of Bob's workplace?",
        "In what city is Bob's employer headquartered?",
        "Where are the headquarters of Bob's company?",
        "What city contains the headquarters of Bob's employer?",
        "Where is Bob's company based?",
        "Name the headquarters city of Bob's employer.",
    ]

    hard = [
        (
            "Bob is employed by a company. In which city "
            "is that company's main office?"
        ),
        (
            "Given Bob's employer, identify the city where "
            "its headquarters are located."
        ),
        (
            "The company employing Bob has its headquarters "
            "in what city?"
        ),
        (
            "Trace Bob to his employer and state where that "
            "organization is headquartered."
        ),
        (
            "Bob works for an organization whose central "
            "office is located in which city?"
        ),
        (
            "Determine the city serving as the headquarters "
            "of the business that employs Bob."
        ),
        (
            "Which city would you reach by following Bob's "
            "employment relationship to the company's headquarters?"
        ),
        (
            "Identify Bob's employer, then give the city of "
            "that employer's main headquarters."
        ),
    ]

    return [
        QueryCase(
            text=text,
            chain_name="bob",
            category="curated-straightforward",
            expected_intermediate="Globex",
            expected_destination="Paris",
        )
        for text in straightforward
    ] + [
        QueryCase(
            text=text,
            chain_name="bob",
            category="curated-hard",
            expected_intermediate="Globex",
            expected_destination="Paris",
        )
        for text in hard
    ]

def generated_queries(
    person: str,
    intermediate: str,
    destination: str,
    chain_name: str,
) -> list[QueryCase]:
    templates = [
        "Where is {person}'s employer headquartered?",
        "Which city is home to {person}'s employer?",
        "Where is the company employing {person} based?",
        "What is the base city of {person}'s workplace?",
        "In which city does {person}'s employer have its headquarters?",
        "Name the city where {person}'s company is headquartered.",
        "What city is associated with the headquarters of {person}'s employer?",
        "Where can the headquarters of {person}'s workplace be found?",
        "Which city contains the main office of {person}'s company?",
        "State the headquarters city for the organization employing {person}.",
        "Follow {person}'s employment to the employer's headquarters city.",
        "What city do you get by tracing {person} to their employer's headquarters?",
        "Identify the city in which {person}'s employer maintains its central office.",
        "Where is the main corporate office of the company that employs {person}?",
        "What is the headquarters city of the business for which {person} works?",
        "The employer of {person} is headquartered in which city?",
        "Where does the organization employing {person} have its main office?",
        "Which city serves as the home base of {person}'s employer?",
        "Give the city location of the headquarters of {person}'s company.",
        "Find the employer of {person}, then report its headquarters city.",
    ]

    return [
        QueryCase(
            text=template.format(
                person=person,
            ),
            chain_name=chain_name,
            category="generated",
            expected_intermediate=intermediate,
            expected_destination=destination,
        )
        for template in templates
    ]

def run_case(
    case: QueryCase,
    encoder: TextEncoder,
    memories: list[TargetMemory],
    candidates: dict,
) -> QueryResult:
    query = encoder.encode(
        case.text
    )

    _, trajectory = target_flow(
        start=query,
        memories=memories,
        steps=50,
        step_size=0.1,
    )

    return evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )


def print_result(
    result: QueryResult,
) -> None:
    takeover = (
        str(result.destination_takeover_step)
        if result.destination_takeover_step is not None
        else "never"
    )

    status = (
        "PASS"
        if result.correct
        else "FAIL"
    )

    print(
        f"{status:<6}"
        f"{result.case.chain_name:<8}"
        f"{result.case.category:<24}"
        f"{result.winner:<12}"
        f"{result.destination_margin:>10.4f}"
        f"{result.intermediate_gain:>12.4f}"
        f"{takeover:>10}  "
        f"{result.case.text}"
    )


def print_summary(
    results: Iterable[QueryResult],
) -> None:
    summaries = summarize_results(
        list(results)
    )

    print()
    print("=" * 90)
    print("SUMMARY")
    print("=" * 90)

    print(
        f"{'Category':<26}"
        f"{'Correct':>10}"
        f"{'Total':>8}"
        f"{'Accuracy':>12}"
        f"{'Avg margin':>14}"
        f"{'Avg gain':>12}"
        f"{'Takeovers':>12}"
    )

    print("-" * 90)

    for category, summary in summaries.items():
        print(
            f"{category:<26}"
            f"{summary.correct_count:>10}"
            f"{summary.total_count:>8}"
            f"{summary.accuracy:>12.3f}"
            f"{summary.average_destination_margin:>14.4f}"
            f"{summary.average_intermediate_gain:>12.4f}"
            f"{summary.takeover_count:>12}"
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

    cases = (
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

    print()
    print("=" * 140)
    print("COMPOSITION ROBUSTNESS BENCHMARK")
    print("=" * 140)

    print(
        f"{'Result':<6}"
        f"{'Chain':<8}"
        f"{'Category':<24}"
        f"{'Winner':<12}"
        f"{'Margin':>10}"
        f"{'Int. gain':>12}"
        f"{'Takeover':>10}  "
        f"Query"
    )

    print("-" * 140)

    results = []

    for case in cases:
        result = run_case(
            case=case,
            encoder=encoder,
            memories=memories,
            candidates=candidates,
        )

        results.append(result)

        print_result(result)

    print_summary(results)


if __name__ == "__main__":
    main()