import numpy as np

from src.memory import FlowMemory
from src.field import memory_influence, vector_field


def test_no_memories_produce_zero_vector():
    # Create a simple 3-dimensional state.
    x = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    # With no memories, the field should not push the state anywhere.
    result = vector_field(x, memories=[])

    assert np.allclose(
        result,
        np.zeros_like(x),
    )


def test_memory_influence_is_strongest_at_center():
    # The memory is centered at the origin.
    center = np.array([0.0, 0.0], dtype=np.float32)

    # The learned direction points to the right.
    direction = np.array([1.0, 0.0], dtype=np.float32)

    memory = FlowMemory(
        center=center,
        direction=direction,
        strength=1.0,
        radius=1.0,
    )

    # Evaluate the memory exactly at its center.
    result = memory_influence(
        memory,
        x=center,
    )

    # At the center, the Gaussian weight equals 1.
    # Therefore the influence should equal:
    #
    # strength × direction
    expected = direction

    assert np.allclose(result, expected)


def test_memory_influence_decreases_with_distance():
    center = np.array([0.0, 0.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)

    memory = FlowMemory(
        center=center,
        direction=direction,
        strength=1.0,
        radius=1.0,
    )

    near_point = np.array([0.1, 0.0], dtype=np.float32)
    far_point = np.array([3.0, 0.0], dtype=np.float32)

    near_influence = memory_influence(
        memory,
        x=near_point,
    )

    far_influence = memory_influence(
        memory,
        x=far_point,
    )

    # Compare the lengths of the resulting vectors.
    near_magnitude = np.linalg.norm(near_influence)
    far_magnitude = np.linalg.norm(far_influence)

    assert near_magnitude > far_magnitude


def test_increasing_strength_increases_influence():
    center = np.array([0.0, 0.0], dtype=np.float32)
    direction = np.array([1.0, 0.0], dtype=np.float32)

    weak_memory = FlowMemory(
        center=center,
        direction=direction,
        strength=1.0,
        radius=1.0,
    )

    strong_memory = FlowMemory(
        center=center,
        direction=direction,
        strength=2.0,
        radius=1.0,
    )

    weak_influence = memory_influence(
        weak_memory,
        x=center,
    )

    strong_influence = memory_influence(
        strong_memory,
        x=center,
    )

    assert np.linalg.norm(strong_influence) > np.linalg.norm(
        weak_influence
    )


def test_vector_field_sums_multiple_memories():
    x = np.array([0.0, 0.0], dtype=np.float32)

    right_memory = FlowMemory(
        center=x,
        direction=np.array([1.0, 0.0], dtype=np.float32),
        strength=1.0,
        radius=1.0,
    )

    upward_memory = FlowMemory(
        center=x,
        direction=np.array([0.0, 1.0], dtype=np.float32),
        strength=1.0,
        radius=1.0,
    )

    result = vector_field(
        x,
        memories=[
            right_memory,
            upward_memory,
        ],
    )

    # At the shared center, both Gaussian weights are 1.
    # Their directions should therefore add together.
    expected = np.array([1.0, 1.0], dtype=np.float32)

    assert np.allclose(result, expected)