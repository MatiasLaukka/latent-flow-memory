import numpy as np

from src.robustness import (
    QueryCase,
    QueryResult,
)
from src.scaling import (
    ScalingSummary,
    summarize_load,
)


def create_result(
    *,
    text: str,
    correct: bool,
    margin: float,
    intermediate_gain: float,
    takeover_step: int | None,
    strongest_wrong_candidate: str,
    strongest_wrong_similarity: float,
) -> QueryResult:
    """
    Create a compact synthetic QueryResult for testing
    aggregate scaling metrics.
    """

    case = QueryCase(
        text=text,
        chain_name="alice",
        category="generated",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    return QueryResult(
        case=case,
        winner=(
            "Helsinki"
            if correct
            else strongest_wrong_candidate
        ),
        correct=correct,
        intermediate_peak_step=10,
        intermediate_gain=intermediate_gain,
        destination_takeover_step=takeover_step,
        destination_similarity=(
            strongest_wrong_similarity
            + margin
        ),
        strongest_wrong_candidate=(
            strongest_wrong_candidate
        ),
        strongest_wrong_similarity=(
            strongest_wrong_similarity
        ),
        destination_margin=margin,
    )


def test_summarize_load_calculates_accuracy():
    results = [
        create_result(
            text="query one",
            correct=True,
            margin=0.2,
            intermediate_gain=0.6,
            takeover_step=20,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.5,
        ),
        create_result(
            text="query two",
            correct=False,
            margin=-0.1,
            intermediate_gain=0.4,
            takeover_step=None,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.7,
        ),
    ]

    summary = summarize_load(
        distractor_count=8,
        results=results,
        runtimes_seconds=[
            0.1,
            0.2,
        ],
        wrong_city_similarities=[
            0.5,
            0.7,
        ],
    )

    assert summary.correct_count == 1
    assert summary.total_count == 2
    assert np.isclose(
        summary.accuracy,
        0.5,
    )


def test_summarize_load_calculates_margin_statistics():
    results = [
        create_result(
            text="query one",
            correct=True,
            margin=0.2,
            intermediate_gain=0.6,
            takeover_step=20,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.5,
        ),
        create_result(
            text="query two",
            correct=True,
            margin=0.1,
            intermediate_gain=0.4,
            takeover_step=30,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.6,
        ),
    ]

    summary = summarize_load(
        distractor_count=3,
        results=results,
        runtimes_seconds=[
            0.1,
            0.2,
        ],
        wrong_city_similarities=[
            0.5,
            0.6,
        ],
    )

    assert np.isclose(
        summary.average_destination_margin,
        0.15,
    )

    assert np.isclose(
        summary.minimum_destination_margin,
        0.1,
    )


def test_summarize_load_ignores_missing_takeovers_in_average():
    results = [
        create_result(
            text="query one",
            correct=True,
            margin=0.2,
            intermediate_gain=0.6,
            takeover_step=20,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.5,
        ),
        create_result(
            text="query two",
            correct=False,
            margin=-0.1,
            intermediate_gain=0.4,
            takeover_step=None,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.7,
        ),
        create_result(
            text="query three",
            correct=True,
            margin=0.1,
            intermediate_gain=0.5,
            takeover_step=40,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.6,
        ),
    ]

    summary = summarize_load(
        distractor_count=18,
        results=results,
        runtimes_seconds=[
            0.1,
            0.2,
            0.3,
        ],
        wrong_city_similarities=[
            0.5,
            0.7,
            0.6,
        ],
    )

    assert np.isclose(
        summary.average_takeover_step,
        30.0,
    )

    assert summary.maximum_takeover_step == 40
    assert summary.no_takeover_count == 1


def test_summarize_load_calculates_runtime_and_wrong_city_mean():
    results = [
        create_result(
            text="query one",
            correct=True,
            margin=0.2,
            intermediate_gain=0.6,
            takeover_step=20,
            strongest_wrong_candidate="Paris",
            strongest_wrong_similarity=0.5,
        ),
        create_result(
            text="query two",
            correct=True,
            margin=0.1,
            intermediate_gain=0.4,
            takeover_step=30,
            strongest_wrong_candidate="Madrid",
            strongest_wrong_similarity=0.7,
        ),
    ]

    summary = summarize_load(
        distractor_count=48,
        results=results,
        runtimes_seconds=[
            0.1,
            0.3,
        ],
        wrong_city_similarities=[
            0.4,
            0.8,
        ],
    )

    assert np.isclose(
        summary.average_runtime_seconds,
        0.2,
    )

    assert np.isclose(
        summary.average_wrong_city_similarity,
        0.6,
    )


def test_summarize_load_returns_scaling_summary():
    result = create_result(
        text="query",
        correct=True,
        margin=0.2,
        intermediate_gain=0.6,
        takeover_step=20,
        strongest_wrong_candidate="Paris",
        strongest_wrong_similarity=0.5,
    )

    summary = summarize_load(
        distractor_count=0,
        results=[result],
        runtimes_seconds=[0.1],
        wrong_city_similarities=[0.5],
    )

    assert isinstance(
        summary,
        ScalingSummary,
    )

    assert summary.distractor_count == 0


def test_summarize_load_rejects_empty_results():
    try:
        summarize_load(
            distractor_count=0,
            results=[],
            runtimes_seconds=[],
            wrong_city_similarities=[],
        )
    except ValueError as error:
        assert str(error) == (
            "Cannot summarize an empty load."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty results."
        )


def test_summarize_load_rejects_mismatched_lengths():
    result = create_result(
        text="query",
        correct=True,
        margin=0.2,
        intermediate_gain=0.6,
        takeover_step=20,
        strongest_wrong_candidate="Paris",
        strongest_wrong_similarity=0.5,
    )

    try:
        summarize_load(
            distractor_count=0,
            results=[result],
            runtimes_seconds=[],
            wrong_city_similarities=[0.5],
        )
    except ValueError as error:
        assert str(error) == (
            "Results, runtimes, and wrong-city scores "
            "must have equal lengths."
        )
    else:
        raise AssertionError(
            "Expected ValueError for mismatched lengths."
        )