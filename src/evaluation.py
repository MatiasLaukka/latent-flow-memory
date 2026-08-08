import numpy as np


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Return cosine similarity for normalized vectors.
    """

    return float(
        np.dot(
            first,
            second,
        )
    )


def rank_candidates(
    state: np.ndarray,
    candidates: dict[str, np.ndarray],
) -> list[tuple[str, float]]:
    """
    Rank candidate embeddings by similarity to a state.
    """

    rankings = [
        (
            name,
            cosine_similarity(
                state,
                embedding,
            ),
        )
        for name, embedding in candidates.items()
    ]

    rankings.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return rankings


def trajectory_similarities(
    trajectory: list[np.ndarray],
    candidates: dict[str, np.ndarray],
) -> list[dict[str, float]]:
    """
    Calculate every candidate's similarity at each
    trajectory step.
    """

    return [
        {
            name: cosine_similarity(
                state,
                embedding,
            )
            for name, embedding in candidates.items()
        }
        for state in trajectory
    ]