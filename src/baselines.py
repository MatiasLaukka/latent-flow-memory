import numpy as np

from src.dynamics import normalize
from src.relations import (
    TargetMemory,
    gaussian_weight,
    target_flow,
)


def nearest_target(
    query: np.ndarray,
    memories: list[TargetMemory],
) -> np.ndarray:
    """
    Return the target attached to the closest memory centre.
    """

    if not memories:
        raise ValueError(
            "At least one memory is required."
        )

    nearest_memory = min(
        memories,
        key=lambda memory: np.linalg.norm(
            query
            - memory.center
        ),
    )

    return nearest_memory.target.copy()


def kernel_weighted_target(
    query: np.ndarray,
    memories: list[TargetMemory],
) -> np.ndarray:
    """
    Blend memory targets using Gaussian similarity weights.
    """

    if not memories:
        raise ValueError(
            "At least one memory is required."
        )

    weighted_sum = np.zeros_like(query)
    total_weight = 0.0

    for memory in memories:
        weight = gaussian_weight(
            x=query,
            center=memory.center,
            radius=memory.radius,
        )

        weighted_sum = (
            weighted_sum
            + weight
            * memory.target
        )

        total_weight += weight

    if total_weight == 0.0:
        return query.copy()

    averaged_target = (
        weighted_sum
        / total_weight
    )

    return normalize(
        averaged_target
    )


def one_step_target_flow(
    query: np.ndarray,
    memories: list[TargetMemory],
    step_size: float = 0.1,
) -> np.ndarray:
    """
    Apply exactly one target-flow update.
    """

    final_state, _ = target_flow(
        start=query,
        memories=memories,
        steps=1,
        step_size=step_size,
    )

    return final_state