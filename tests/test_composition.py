import numpy as np

from src.baselines import (
    kernel_weighted_target,
    nearest_target,
    one_step_target_flow,
)
from src.relations import TargetMemory


def normalized(
    values: list[float],
) -> np.ndarray:
    vector = np.array(
        values,
        dtype=np.float32,
    )

    return (
        vector
        / np.linalg.norm(vector)
    )


def test_nearest_target_returns_target_of_closest_center():
    query = normalized(
        [1.0, 0.0]
    )

    close_memory = TargetMemory(
        center=normalized(
            [0.9, 0.1]
        ),
        target=normalized(
            [0.0, 1.0]
        ),
        radius=1.0,
    )

    far_memory = TargetMemory(
        center=normalized(
            [-1.0, 0.0]
        ),
        target=normalized(
            [0.0, -1.0]
        ),
        radius=1.0,
    )

    result = nearest_target(
        query=query,
        memories=[
            close_memory,
            far_memory,
        ],
    )

    assert np.allclose(
        result,
        close_memory.target,
    )


def test_nearest_target_rejects_empty_memory_list():
    query = normalized(
        [1.0, 0.0]
    )

    try:
        nearest_target(
            query=query,
            memories=[],
        )
    except ValueError as error:
        assert str(error) == (
            "At least one memory is required."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty memories."
        )


def test_kernel_weighted_target_favors_nearby_memory():
    query = normalized(
        [1.0, 0.0]
    )

    close_memory = TargetMemory(
        center=normalized(
            [0.9, 0.1]
        ),
        target=normalized(
            [0.0, 1.0]
        ),
        radius=0.5,
    )

    far_memory = TargetMemory(
        center=normalized(
            [-1.0, 0.0]
        ),
        target=normalized(
            [0.0, -1.0]
        ),
        radius=0.5,
    )

    result = kernel_weighted_target(
        query=query,
        memories=[
            close_memory,
            far_memory,
        ],
    )

    close_similarity = float(
        np.dot(
            result,
            close_memory.target,
        )
    )

    far_similarity = float(
        np.dot(
            result,
            far_memory.target,
        )
    )

    assert close_similarity > far_similarity


def test_one_step_target_flow_moves_toward_target():
    query = normalized(
        [1.0, 0.0]
    )

    target = normalized(
        [0.0, 1.0]
    )

    memory = TargetMemory(
        center=query,
        target=target,
        radius=1.0,
    )

    result = one_step_target_flow(
        query=query,
        memories=[memory],
        step_size=0.1,
    )

    assert (
        float(np.dot(result, target))
        > float(np.dot(query, target))
    )