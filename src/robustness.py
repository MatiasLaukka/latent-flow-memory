from dataclasses import dataclass

import numpy as np

from src.evaluation import (
    rank_candidates,
    trajectory_similarities,
)


@dataclass(frozen=True)
class QueryCase:
    """
    One labeled natural-language composition query.
    """

    text: str
    chain_name: str
    category: str
    expected_intermediate: str
    expected_destination: str


@dataclass(frozen=True)
class QueryResult:
    """
    Measurements produced by one composition query.
    """

    case: QueryCase
    winner: str
    correct: bool
    intermediate_peak_step: int
    intermediate_gain: float
    destination_takeover_step: int | None
    destination_similarity: float
    strongest_wrong_candidate: str
    strongest_wrong_similarity: float
    destination_margin: float


@dataclass(frozen=True)
class CategorySummary:
    """
    Aggregate measurements for one query category.
    """

    category: str
    correct_count: int
    total_count: int
    accuracy: float
    average_destination_margin: float
    average_intermediate_gain: float
    takeover_count: int


def find_peak_step(
    similarities: list[dict[str, float]],
    candidate_name: str,
) -> int:
    """
    Return the trajectory step with maximum similarity
    to a candidate.
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
    Find the first post-peak step where the destination
    overtakes the intermediate.
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


def evaluate_trajectory(
    case: QueryCase,
    trajectory: list[np.ndarray],
    candidates: dict[str, np.ndarray],
) -> QueryResult:
    """
    Evaluate one complete latent trajectory.
    """

    if not trajectory:
        raise ValueError(
            "Trajectory must contain at least one state."
        )

    final_state = trajectory[-1]

    final_ranking = rank_candidates(
        state=final_state,
        candidates=candidates,
    )

    winner = final_ranking[0][0]

    similarities = trajectory_similarities(
        trajectory=trajectory,
        candidates=candidates,
    )

    intermediate_peak_step = find_peak_step(
        similarities=similarities,
        candidate_name=case.expected_intermediate,
    )

    intermediate_start_similarity = similarities[0][
        case.expected_intermediate
    ]

    intermediate_peak_similarity = similarities[
        intermediate_peak_step
    ][case.expected_intermediate]

    intermediate_gain = (
        intermediate_peak_similarity
        - intermediate_start_similarity
    )

    destination_takeover_step = find_takeover_step(
        similarities=similarities,
        destination_name=case.expected_destination,
        intermediate_name=case.expected_intermediate,
        intermediate_peak_step=intermediate_peak_step,
    )

    destination_similarity = similarities[-1][
        case.expected_destination
    ]

    wrong_rankings = [
        (name, score)
        for name, score in final_ranking
        if name != case.expected_destination
    ]

    strongest_wrong_candidate = wrong_rankings[0][0]
    strongest_wrong_similarity = wrong_rankings[0][1]

    destination_margin = (
        destination_similarity
        - strongest_wrong_similarity
    )

    return QueryResult(
        case=case,
        winner=winner,
        correct=(
            winner
            == case.expected_destination
        ),
        intermediate_peak_step=intermediate_peak_step,
        intermediate_gain=intermediate_gain,
        destination_takeover_step=destination_takeover_step,
        destination_similarity=destination_similarity,
        strongest_wrong_candidate=strongest_wrong_candidate,
        strongest_wrong_similarity=strongest_wrong_similarity,
        destination_margin=destination_margin,
    )


def summarize_category(
    category: str,
    results: list[QueryResult],
) -> CategorySummary:
    """
    Aggregate a non-empty set of query results.
    """

    if not results:
        raise ValueError(
            "Cannot summarize an empty result list."
        )

    correct_count = sum(
        result.correct
        for result in results
    )

    total_count = len(results)

    average_destination_margin = float(
        np.mean(
            [
                result.destination_margin
                for result in results
            ]
        )
    )

    average_intermediate_gain = float(
        np.mean(
            [
                result.intermediate_gain
                for result in results
            ]
        )
    )

    takeover_count = sum(
        result.destination_takeover_step
        is not None
        for result in results
    )

    return CategorySummary(
        category=category,
        correct_count=correct_count,
        total_count=total_count,
        accuracy=correct_count / total_count,
        average_destination_margin=(
            average_destination_margin
        ),
        average_intermediate_gain=(
            average_intermediate_gain
        ),
        takeover_count=takeover_count,
    )


def summarize_results(
    results: list[QueryResult],
) -> dict[str, CategorySummary]:
    """
    Summarize results by category and overall.
    """

    if not results:
        raise ValueError(
            "Cannot summarize an empty result list."
        )

    categories = sorted(
        {
            result.case.category
            for result in results
        }
    )

    summaries = {}

    for category in categories:
        category_results = [
            result
            for result in results
            if result.case.category == category
        ]

        summaries[category] = summarize_category(
            category=category,
            results=category_results,
        )

    summaries["overall"] = summarize_category(
        category="overall",
        results=results,
    )

    return summaries