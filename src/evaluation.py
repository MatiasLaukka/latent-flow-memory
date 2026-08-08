import numpy as np


def cosine_similarity(
    a: np.ndarray,
    b: np.ndarray,
) -> float:
    """
    Return cosine similarity between two vectors.

    Project embeddings and latent states are normally
    already normalized, but computing the norms here keeps
    this helper safe for general use.
    """

    a_norm = float(
        np.linalg.norm(a)
    )

    b_norm = float(
        np.linalg.norm(b)
    )

    if (
        a_norm == 0.0
        or b_norm == 0.0
    ):
        raise ValueError(
            "Cosine similarity is undefined "
            "for zero-length vectors."
        )

    return float(
        np.dot(a, b)
        / (a_norm * b_norm)
    )


def rank_candidates(
    state: np.ndarray,
    candidates: dict[
        str,
        np.ndarray,
    ],
) -> list[
    tuple[str, float]
]:
    """
    Rank candidate concept vectors by cosine similarity
    to the supplied latent state.

    Highest similarity is returned first.
    """

    if not candidates:
        raise ValueError(
            "At least one candidate is required."
        )

    rankings = [
        (
            name,
            cosine_similarity(
                state,
                candidate,
            ),
        )
        for name, candidate
        in candidates.items()
    ]

    rankings.sort(
        key=lambda item: item[1],
        reverse=True,
    )

    return rankings


def trajectory_similarities(
    trajectory: list[np.ndarray],
    candidates: dict[
        str,
        np.ndarray,
    ],
) -> list[
    dict[str, float]
]:
    """
    Calculate candidate similarities at every trajectory
    step.

    Each returned dictionary maps candidate name to its
    cosine similarity with that trajectory state.
    """

    if not trajectory:
        raise ValueError(
            "Trajectory must contain at least one state."
        )

    if not candidates:
        raise ValueError(
            "At least one candidate is required."
        )

    return [
        {
            name: cosine_similarity(
                state,
                candidate,
            )
            for name, candidate
            in candidates.items()
        }
        for state in trajectory
    ]