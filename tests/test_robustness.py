import numpy as np

from src.robustness import (
    QueryCase,
    evaluate_trajectory,
    summarize_results,
)


def normalized(values: list[float]) -> np.ndarray:
    vector = np.array(
        values,
        dtype=np.float32,
    )

    return vector / np.linalg.norm(vector)


def test_evaluate_trajectory_detects_correct_winner():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([-1.0, 0.0]),
    }

    trajectory = [
        normalized([1.0, 0.0]),
        normalized([0.7, 0.7]),
        normalized([0.0, 1.0]),
    ]

    case = QueryCase(
        text="Where is Alice's employer headquartered?",
        chain_name="alice",
        category="curated-straightforward",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    result = evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )

    assert result.winner == "Helsinki"
    assert result.correct is True


def test_evaluate_trajectory_finds_intermediate_peak():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([-1.0, 0.0]),
    }

    trajectory = [
        normalized([0.7, 0.7]),
        normalized([1.0, 0.0]),
        normalized([0.7, 0.7]),
        normalized([0.0, 1.0]),
    ]

    case = QueryCase(
        text="Where is Alice's employer headquartered?",
        chain_name="alice",
        category="curated-straightforward",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    result = evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )

    assert result.intermediate_peak_step == 1


def test_evaluate_trajectory_finds_takeover_after_peak():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([-1.0, 0.0]),
    }

    trajectory = [
        normalized([0.7, 0.7]),
        normalized([1.0, 0.0]),
        normalized([0.8, 0.6]),
        normalized([0.4, 0.9]),
    ]

    case = QueryCase(
        text="Where is Alice's employer headquartered?",
        chain_name="alice",
        category="curated-hard",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    result = evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )

    assert result.destination_takeover_step == 3


def test_evaluate_trajectory_reports_never_when_no_takeover():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([-1.0, 0.0]),
    }

    trajectory = [
        normalized([0.9, 0.1]),
        normalized([1.0, 0.0]),
        normalized([0.9, 0.1]),
    ]

    case = QueryCase(
        text="Where is Alice's employer headquartered?",
        chain_name="alice",
        category="curated-hard",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    result = evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )

    assert result.destination_takeover_step is None


def test_evaluate_trajectory_calculates_destination_margin():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([0.6, 0.8]),
    }

    trajectory = [
        normalized([1.0, 0.0]),
        normalized([0.0, 1.0]),
    ]

    case = QueryCase(
        text="Where is Alice's employer headquartered?",
        chain_name="alice",
        category="generated",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    result = evaluate_trajectory(
        case=case,
        trajectory=trajectory,
        candidates=candidates,
    )

    assert np.isclose(
        result.destination_similarity,
        1.0,
    )

    assert np.isclose(
        result.strongest_wrong_similarity,
        0.8,
    )

    assert np.isclose(
        result.destination_margin,
        0.2,
    )


def test_summarize_results_groups_by_category():
    candidates = {
        "Acme": normalized([1.0, 0.0]),
        "Helsinki": normalized([0.0, 1.0]),
        "Paris": normalized([-1.0, 0.0]),
    }

    correct_case = QueryCase(
        text="Correct query",
        chain_name="alice",
        category="curated-straightforward",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    incorrect_case = QueryCase(
        text="Incorrect query",
        chain_name="alice",
        category="curated-hard",
        expected_intermediate="Acme",
        expected_destination="Helsinki",
    )

    correct_result = evaluate_trajectory(
        case=correct_case,
        trajectory=[
            normalized([1.0, 0.0]),
            normalized([0.0, 1.0]),
        ],
        candidates=candidates,
    )

    incorrect_result = evaluate_trajectory(
        case=incorrect_case,
        trajectory=[
            normalized([0.0, 1.0]),
            normalized([-1.0, 0.0]),
        ],
        candidates=candidates,
    )

    summaries = summarize_results(
        [
            correct_result,
            incorrect_result,
        ]
    )

    assert summaries[
        "curated-straightforward"
    ].correct_count == 1

    assert summaries[
        "curated-straightforward"
    ].total_count == 1

    assert summaries[
        "curated-hard"
    ].correct_count == 0

    assert summaries["overall"].total_count == 2