from dataclasses import dataclass

import numpy as np

from src.robustness import QueryResult


@dataclass(frozen=True)
class DistractorChain:
    """
    One deterministic person-to-company-to-city chain.
    """

    person: str
    company: str
    city: str


@dataclass(frozen=True)
class ScalingSummary:
    """
    Aggregate benchmark measurements for one memory load.
    """

    distractor_count: int
    correct_count: int
    total_count: int
    accuracy: float
    average_destination_margin: float
    minimum_destination_margin: float
    average_intermediate_gain: float
    average_takeover_step: float | None
    maximum_takeover_step: int | None
    no_takeover_count: int
    average_wrong_city_similarity: float
    average_runtime_seconds: float


def fixed_distractor_pool() -> list[DistractorChain]:
    """
    Return the fixed ordered pool used by every scaling run.

    Do not reorder this list. Load levels use prefixes of
    the pool, so order is part of experiment determinism.
    """

    return [
        DistractorChain(
            person="Carol",
            company="Initech",
            city="Berlin",
        ),
        DistractorChain(
            person="David",
            company="Umbrella Corporation",
            city="Madrid",
        ),
        DistractorChain(
            person="Emma",
            company="Hooli",
            city="Dublin",
        ),
        DistractorChain(
            person="Frank",
            company="Vehement Capital",
            city="Oslo",
        ),
        DistractorChain(
            person="Grace",
            company="Wonka Industries",
            city="Vienna",
        ),
        DistractorChain(
            person="Henry",
            company="Stark Industries",
            city="Rome",
        ),
        DistractorChain(
            person="Isabel",
            company="Cyberdyne Systems",
            city="Lisbon",
        ),
        DistractorChain(
            person="Jack",
            company="Massive Dynamic",
            city="Prague",
        ),
        DistractorChain(
            person="Karen",
            company="Soylent Corporation",
            city="Warsaw",
        ),
        DistractorChain(
            person="Leo",
            company="Tyrell Corporation",
            city="Brussels",
        ),
        DistractorChain(
            person="Maria",
            company="Waystar Royco",
            city="Athens",
        ),
        DistractorChain(
            person="Nathan",
            company="Pied Piper",
            city="Amsterdam",
        ),
        DistractorChain(
            person="Olivia",
            company="Aperture Science",
            city="Copenhagen",
        ),
        DistractorChain(
            person="Peter",
            company="Oscorp",
            city="Stockholm",
        ),
        DistractorChain(
            person="Rachel",
            company="Prestige Worldwide",
            city="Budapest",
        ),
        DistractorChain(
            person="Samuel",
            company="Vandelay Industries",
            city="Zagreb",
        ),
        DistractorChain(
            person="Tina",
            company="Nakatomi Corporation",
            city="Sofia",
        ),
        DistractorChain(
            person="Victor",
            company="Oceanic Airlines",
            city="Bucharest",
        ),
        DistractorChain(
            person="Wendy",
            company="Initrode",
            city="Tallinn",
        ),
        DistractorChain(
            person="Aaron",
            company="Globomantics",
            city="Riga",
        ),
        DistractorChain(
            person="Bianca",
            company="Monarch Solutions",
            city="Vilnius",
        ),
        DistractorChain(
            person="Caleb",
            company="Summit Dynamics",
            city="Reykjavik",
        ),
        DistractorChain(
            person="Diana",
            company="Blue Horizon Labs",
            city="Luxembourg",
        ),
        DistractorChain(
            person="Ethan",
            company="Evergreen Systems",
            city="Bern",
        ),
        DistractorChain(
            person="Fiona",
            company="Northstar Analytics",
            city="Helsinki",
        ),
        DistractorChain(
            person="George",
            company="Silverline Media",
            city="Paris",
        ),
        DistractorChain(
            person="Hannah",
            company="Redwood Robotics",
            city="Berlin",
        ),
        DistractorChain(
            person="Ian",
            company="Vertex Consulting",
            city="Madrid",
        ),
        DistractorChain(
            person="Julia",
            company="Aurora Networks",
            city="Dublin",
        ),
        DistractorChain(
            person="Kevin",
            company="Pioneer Logistics",
            city="Oslo",
        ),
        DistractorChain(
            person="Laura",
            company="Crescent Technologies",
            city="Vienna",
        ),
        DistractorChain(
            person="Martin",
            company="Atlas Manufacturing",
            city="Rome",
        ),
        DistractorChain(
            person="Nora",
            company="Beacon Software",
            city="Lisbon",
        ),
        DistractorChain(
            person="Oscar",
            company="Cobalt Energy",
            city="Prague",
        ),
        DistractorChain(
            person="Paula",
            company="Delta Research",
            city="Warsaw",
        ),
        DistractorChain(
            person="Quentin",
            company="Ember Studios",
            city="Brussels",
        ),
        DistractorChain(
            person="Rebecca",
            company="Falcon Aerospace",
            city="Athens",
        ),
        DistractorChain(
            person="Simon",
            company="Granite Finance",
            city="Amsterdam",
        ),
        DistractorChain(
            person="Teresa",
            company="Harbor Health",
            city="Copenhagen",
        ),
        DistractorChain(
            person="Ursula",
            company="Ironwood Security",
            city="Stockholm",
        ),
        DistractorChain(
            person="Vincent",
            company="Juniper Foods",
            city="Budapest",
        ),
        DistractorChain(
            person="Willow",
            company="Keystone Design",
            city="Zagreb",
        ),
        DistractorChain(
            person="Xavier",
            company="Lighthouse Telecom",
            city="Sofia",
        ),
        DistractorChain(
            person="Yasmin",
            company="Meridian Biotech",
            city="Bucharest",
        ),
        DistractorChain(
            person="Zachary",
            company="Nimbus Ventures",
            city="Tallinn",
        ),
        DistractorChain(
            person="Amelia",
            company="Oakridge Engineering",
            city="Riga",
        ),
        DistractorChain(
            person="Benjamin",
            company="Palisade Entertainment",
            city="Vilnius",
        ),
        DistractorChain(
            person="Clara",
            company="Quantum Retail",
            city="Reykjavik",
        ),
    ]


def summarize_load(
    distractor_count: int,
    results: list[QueryResult],
    runtimes_seconds: list[float],
    wrong_city_similarities: list[float],
) -> ScalingSummary:
    """
    Aggregate all query results at one distractor load.
    """

    if not results:
        raise ValueError(
            "Cannot summarize an empty load."
        )

    if not (
        len(results)
        == len(runtimes_seconds)
        == len(wrong_city_similarities)
    ):
        raise ValueError(
            "Results, runtimes, and wrong-city scores "
            "must have equal lengths."
        )

    correct_count = sum(
        result.correct
        for result in results
    )

    takeover_steps = [
        result.destination_takeover_step
        for result in results
        if result.destination_takeover_step
        is not None
    ]

    average_takeover_step = (
        float(np.mean(takeover_steps))
        if takeover_steps
        else None
    )

    maximum_takeover_step = (
        max(takeover_steps)
        if takeover_steps
        else None
    )

    return ScalingSummary(
        distractor_count=distractor_count,
        correct_count=correct_count,
        total_count=len(results),
        accuracy=(
            correct_count
            / len(results)
        ),
        average_destination_margin=float(
            np.mean(
                [
                    result.destination_margin
                    for result in results
                ]
            )
        ),
        minimum_destination_margin=min(
            result.destination_margin
            for result in results
        ),
        average_intermediate_gain=float(
            np.mean(
                [
                    result.intermediate_gain
                    for result in results
                ]
            )
        ),
        average_takeover_step=average_takeover_step,
        maximum_takeover_step=maximum_takeover_step,
        no_takeover_count=(
            len(results)
            - len(takeover_steps)
        ),
        average_wrong_city_similarity=float(
            np.mean(
                wrong_city_similarities
            )
        ),
        average_runtime_seconds=float(
            np.mean(
                runtimes_seconds
            )
        ),
    )