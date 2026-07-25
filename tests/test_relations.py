import numpy as np

from src.relations import (
    TargetMemory,
    target_memory_influence,
)


def test_target_memory_influence_points_toward_target():
    # Start at the origin.
    center = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    # The target lies directly to the right.
    target = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    memory = TargetMemory(
        center=center,
        target=target,
        strength=1.0,
        radius=1.0,
    )

    influence = target_memory_influence(
        memory=memory,
        x=center,
    )

    # At the centre, the memory should point toward:
    #
    # target - current_state
    #
    # [1, 0] - [0, 0] = [1, 0]
    assert np.allclose(
        influence,
        np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
    )


def test_target_memory_influence_is_strongest_at_center():
    center = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    target = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    memory = TargetMemory(
        center=center,
        target=target,
        strength=1.0,
        radius=1.0,
    )

    at_center = target_memory_influence(
        memory=memory,
        x=center,
    )

    farther_away = target_memory_influence(
        memory=memory,
        x=np.array(
            [0.0, 2.0],
            dtype=np.float32,
        ),
    )

    assert (
        np.linalg.norm(at_center)
        > np.linalg.norm(farther_away)
    )


def test_target_memory_has_almost_no_influence_far_away():
    center = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    target = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    memory = TargetMemory(
        center=center,
        target=target,
        strength=1.0,
        radius=0.5,
    )

    influence = target_memory_influence(
        memory=memory,
        x=np.array(
            [10.0, 10.0],
            dtype=np.float32,
        ),
    )

    assert np.allclose(
        influence,
        np.zeros_like(influence),
        atol=1e-6,
    )


def test_target_memory_produces_zero_movement_at_target():
    center = np.array(
        [0.0, 0.0],
        dtype=np.float32,
    )

    target = np.array(
        [1.0, 0.0],
        dtype=np.float32,
    )

    memory = TargetMemory(
        center=center,
        target=target,
        strength=1.0,
        radius=2.0,
    )

    influence = target_memory_influence(
        memory=memory,
        x=target,
    )

    # Because target - x equals zero at the target.
    assert np.allclose(
        influence,
        np.zeros_like(target),
        atol=1e-6,
    )


def test_target_memory_rejects_non_positive_radius():
    memory = TargetMemory(
        center=np.array(
            [0.0, 0.0],
            dtype=np.float32,
        ),
        target=np.array(
            [1.0, 0.0],
            dtype=np.float32,
        ),
        radius=0.0,
    )

    try:
        target_memory_influence(
            memory=memory,
            x=memory.center,
        )
    except ValueError as error:
        assert str(error) == (
            "Memory radius must be greater than zero."
        )
    else:
        raise AssertionError(
            "Expected ValueError for radius <= 0."
        )