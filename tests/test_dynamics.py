import numpy as np

from src.dynamics import flow
from src.memory import FlowMemory


def test_flow_returns_requested_number_of_steps():
    # Start with a simple normalized 2D vector.
    start = np.array([1.0, 0.0], dtype=np.float32)

    # Create one memory that pushes upward.
    memory = FlowMemory(
        center=start,
        direction=np.array([0.0, 1.0], dtype=np.float32),
        strength=1.0,
        radius=1.0,
    )

    final_state, trajectory = flow(
        start=start,
        memories=[memory],
        steps=5,
        step_size=0.1,
    )

    # The trajectory includes the original starting point,
    # followed by one state for each update step.
    #
    # Therefore:
    #
    # 1 initial state + 5 updates = 6 states
    assert len(trajectory) == 6


def test_flow_keeps_states_normalized():
    start = np.array([1.0, 0.0], dtype=np.float32)

    memory = FlowMemory(
        center=start,
        direction=np.array([0.0, 1.0], dtype=np.float32),
        strength=1.0,
        radius=1.0,
    )

    final_state, trajectory = flow(
        start=start,
        memories=[memory],
        steps=5,
        step_size=0.1,
    )

    # Every point in the trajectory should remain on the
    # unit hypersphere:
    #
    # ||x|| = 1
    for state in trajectory:
        assert np.isclose(
            np.linalg.norm(state),
            1.0,
            atol=1e-6,
        )


def test_zero_field_leaves_state_unchanged():
    start = np.array([1.0, 0.0], dtype=np.float32)

    final_state, trajectory = flow(
        start=start,
        memories=[],
        steps=5,
        step_size=0.1,
    )

    # No memories means:
    #
    # V(x) = 0
    #
    # Therefore:
    #
    # x(t+1) = x(t)
    assert np.allclose(
        final_state,
        start,
        atol=1e-6,
    )


def test_flow_moves_state_in_memory_direction():
    start = np.array([1.0, 0.0], dtype=np.float32)

    # This memory pushes the state upward in the second dimension.
    memory = FlowMemory(
        center=start,
        direction=np.array([0.0, 1.0], dtype=np.float32),
        strength=1.0,
        radius=1.0,
    )

    final_state, trajectory = flow(
        start=start,
        memories=[memory],
        steps=1,
        step_size=0.1,
    )

    # Initially the second coordinate is 0.
    #
    # After one upward push, it should become positive.
    assert final_state[1] > start[1]