import numpy as np

from src.linked_flow import (
    RoutingStatus,
    flow_linked_edge,
)
from src.linked_memory import (
    LinkedMemory,
    RelationIntent,
)


def normalized(*values: float) -> np.ndarray:
    """
    Create a normalized float32 vector for synthetic tests.
    """

    vector = np.array(
        values,
        dtype=np.float32,
    )

    return vector / np.linalg.norm(vector)


def create_memory(
    *,
    memory_id: str = "edge",
    center: np.ndarray | None = None,
    target: np.ndarray | None = None,
    radius: float = 2.0,
) -> LinkedMemory:
    """
    Create one synthetic linked-memory edge.
    """

    actual_center = (
        center
        if center is not None
        else normalized(
            1.0,
            0.0,
        )
    )

    actual_target = (
        target
        if target is not None
        else normalized(
            0.0,
            1.0,
        )
    )

    return LinkedMemory(
        memory_id=memory_id,
        source="source",
        relation=RelationIntent(
            name="relation",
            phrase_embeddings=(
                actual_center,
            ),
        ),
        target="target",
        trigger_centers=(
            actual_center,
        ),
        target_vector=actual_target,
        radius=radius,
    )


def test_edge_flow_reaches_target_threshold():
    edge = create_memory()

    result = flow_linked_edge(
        start=normalized(
            1.0,
            0.0,
        ),
        memory=edge,
        handoff_threshold=0.95,
        max_steps=100,
        step_size=0.1,
    )

    assert (
        result.status
        == RoutingStatus.COMPLETED
    )
    assert result.target_reached is True
    assert result.steps_used > 0

    final_similarity = float(
        np.dot(
            result.final_state,
            edge.target_vector,
        )
    )

    assert final_similarity >= 0.95


def test_edge_flow_reports_target_not_reached():
    edge = create_memory(
        radius=0.01,
    )

    result = flow_linked_edge(
        start=normalized(
            -1.0,
            0.0,
        ),
        memory=edge,
        handoff_threshold=0.99,
        max_steps=3,
        step_size=0.1,
    )

    assert (
        result.status
        == RoutingStatus.TARGET_NOT_REACHED
    )
    assert result.target_reached is False
    assert result.steps_used == 3


def test_edge_flow_trajectory_contains_initial_state():
    start = normalized(
        1.0,
        0.0,
    )

    result = flow_linked_edge(
        start=start,
        memory=create_memory(),
        handoff_threshold=0.8,
        max_steps=10,
        step_size=0.1,
    )

    assert np.allclose(
        result.trajectory[0],
        start,
    )

    assert len(result.trajectory) == (
        result.steps_used + 1
    )


def test_edge_flow_completes_immediately_at_target():
    target = normalized(
        0.0,
        1.0,
    )

    result = flow_linked_edge(
        start=target,
        memory=create_memory(
            target=target,
        ),
        handoff_threshold=0.95,
        max_steps=10,
        step_size=0.1,
    )

    assert (
        result.status
        == RoutingStatus.COMPLETED
    )
    assert result.target_reached is True
    assert result.steps_used == 0
    assert len(result.trajectory) == 1


def test_edge_flow_uses_strongest_trigger_center():
    start = normalized(
        1.0,
        0.0,
    )

    target = normalized(
        0.0,
        1.0,
    )

    edge = LinkedMemory(
        memory_id="multi-trigger-edge",
        source="source",
        relation=RelationIntent(
            name="relation",
            phrase_embeddings=(
                start,
            ),
        ),
        target="target",
        trigger_centers=(
            normalized(
                -1.0,
                0.0,
            ),
            start,
        ),
        target_vector=target,
        radius=0.5,
    )

    result = flow_linked_edge(
        start=start,
        memory=edge,
        handoff_threshold=0.8,
        max_steps=50,
        step_size=0.1,
    )

    assert (
        result.status
        == RoutingStatus.COMPLETED
    )