import numpy as np

from src.linked_flow import (
    RoutingStatus,
    flow_linked_edge,
)
from src.linked_memory import (
    LinkedMemory,
    RelationIntent,
)
from src.linked_flow import run_linked_flow
from src.linked_memory import LinkedMemoryGraph
from src.thought_routes import ThoughtRouteStore


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

def test_edge_flow_reports_target_not_reached_with_zero_strength():
    edge = LinkedMemory(
        memory_id="inactive-edge",
        source="source",
        relation=RelationIntent(
            name="relation",
            phrase_embeddings=(
                normalized(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="target",
        trigger_centers=(
            normalized(
                1.0,
                0.0,
            ),
        ),
        target_vector=normalized(
            0.0,
            1.0,
        ),
        strength=0.0,
        radius=0.01,
    )

    result = flow_linked_edge(
        start=normalized(
            1.0,
            0.0,
        ),
        memory=edge,
        handoff_threshold=0.95,
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

def branching_graph() -> LinkedMemoryGraph:
    works_at = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            normalized(
                0.7,
                0.7,
            ),
        ),
    )

    headquarters = RelationIntent(
        name="headquartered-in",
        phrase_embeddings=(
            normalized(
                1.0,
                0.0,
            ),
        ),
    )

    founded_by = RelationIntent(
        name="founded-by",
        phrase_embeddings=(
            normalized(
                0.0,
                1.0,
            ),
        ),
    )

    root = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=works_at,
        target="Globex",
        trigger_centers=(
            normalized(
                1.0,
                0.0,
            ),
        ),
        target_vector=normalized(
            0.0,
            1.0,
        ),
        successor_ids=(
            "globex-headquartered-in-paris",
            "globex-founded-by-susan",
        ),
        radius=2.0,
    )

    headquarters_memory = LinkedMemory(
        memory_id="globex-headquartered-in-paris",
        source="Globex",
        relation=headquarters,
        target="Paris",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            -1.0,
            0.0,
        ),
        radius=2.0,
    )

    founder_memory = LinkedMemory(
        memory_id="globex-founded-by-susan",
        source="Globex",
        relation=founded_by,
        target="Susan",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            1.0,
            0.0,
        ),
        radius=2.0,
    )

    return LinkedMemoryGraph(
        memories=(
            root,
            headquarters_memory,
            founder_memory,
        ),
        root_ids=(
            root.memory_id,
        ),
    )


def test_linked_flow_selects_headquarters_successor():
    result = run_linked_flow(
        query=normalized(
            1.0,
            0.0,
        ),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert (
        result.status
        == RoutingStatus.COMPLETED
    )

    assert result.traversed_memory_ids == (
        "bob-works-at-globex",
        "globex-headquartered-in-paris",
    )


def test_linked_flow_selects_founder_successor():
    result = run_linked_flow(
        query=normalized(
            0.0,
            1.0,
        ),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert (
        result.status
        == RoutingStatus.COMPLETED
    )

    assert result.traversed_memory_ids == (
        "bob-works-at-globex",
        "globex-founded-by-susan",
    )


def test_linked_flow_respects_successor_threshold():
    result = run_linked_flow(
        query=normalized(
            0.7,
            0.7,
        ),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
        successor_minimum_score=0.95,
    )

    assert (
        result.status
        == RoutingStatus.NO_MATCHING_SUCCESSOR
    )


def test_linked_flow_reports_maximum_hops_reached():
    result = run_linked_flow(
        query=normalized(
            1.0,
            0.0,
        ),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=1,
    )

    assert (
        result.status
        == RoutingStatus.MAXIMUM_HOPS_REACHED
    )


def test_linked_flow_reports_no_root_memory():
    graph = LinkedMemoryGraph(
        memories=(),
        root_ids=(),
    )

    result = run_linked_flow(
        query=normalized(
            1.0,
            0.0,
        ),
        graph=graph,
    )

    assert (
        result.status
        == RoutingStatus.NO_ROOT_MEMORY
    )


def test_linked_flow_reports_root_score_below_threshold():
    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                normalized(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            1.0,
            0.0,
        ),
        radius=2.0,
    )

    graph = LinkedMemoryGraph(
        memories=(root,),
        root_ids=(root.memory_id,),
    )

    result = run_linked_flow(
        query=normalized(
            1.0,
            0.0,
        ),
        graph=graph,
        root_minimum_score=0.5,
    )

    assert (
        result.status
        == RoutingStatus.ROOT_SCORE_BELOW_THRESHOLD
    )


def test_selected_edge_remains_active_after_leaving_trigger_region():
    start = normalized(
        1.0,
        0.0,
    )

    target = normalized(
        0.0,
        1.0,
    )

    edge = create_memory(
        center=start,
        target=target,
        radius=0.01,
    )

    result = flow_linked_edge(
        start=start,
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

def test_reinforced_route_can_win_close_root_selection():
    query = normalized(
        1.0,
        0.0,
    )

    works_at = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            query,
        ),
    )

    next_relation = RelationIntent(
        name="next",
        phrase_embeddings=(
            query,
        ),
    )

    root_a = LinkedMemory(
        memory_id="root-a",
        source="A",
        relation=works_at,
        target="A-company",
        trigger_centers=(
            normalized(
                0.99,
                np.sqrt(
                    1.0 - 0.99**2
                ),
            ),
        ),
        target_vector=normalized(
            0.0,
            1.0,
        ),
        successor_ids=(
            "a-next",
        ),
        radius=2.0,
    )

    root_b = LinkedMemory(
        memory_id="root-b",
        source="B",
        relation=works_at,
        target="B-company",
        trigger_centers=(
            normalized(
                0.995,
                np.sqrt(
                    1.0 - 0.995**2
                ),
            ),
        ),
        target_vector=normalized(
            0.0,
            -1.0,
        ),
        successor_ids=(
            "b-next",
        ),
        radius=2.0,
    )

    a_next = LinkedMemory(
        memory_id="a-next",
        source="A-company",
        relation=next_relation,
        target="A-answer",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            -1.0,
            0.0,
        ),
        radius=2.0,
    )

    b_next = LinkedMemory(
        memory_id="b-next",
        source="B-company",
        relation=next_relation,
        target="B-answer",
        trigger_centers=(
            normalized(
                0.0,
                -1.0,
            ),
        ),
        target_vector=normalized(
            -1.0,
            0.0,
        ),
        radius=2.0,
    )

    graph = LinkedMemoryGraph(
        memories=(
            root_a,
            root_b,
            a_next,
            b_next,
        ),
        root_ids=(
            "root-a",
            "root-b",
        ),
    )

    baseline = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert baseline.traversed_memory_ids[0] == (
        "root-b"
    )

    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=(
            "root-a",
            "a-next",
        ),
        context_embedding=query,
    )

    reinforced = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
        thought_routes=store,
    )

    assert reinforced.traversed_memory_ids == (
        "root-a",
        "a-next",
    )


def test_reinforced_route_can_win_close_successor_selection():
    query = normalized(
        1.0,
        0.0,
    )

    root_relation = RelationIntent(
        name="root",
        phrase_embeddings=(
            query,
        ),
    )

    relation_a = RelationIntent(
        name="relation-a",
        phrase_embeddings=(
            normalized(
                0.99,
                np.sqrt(
                    1.0 - 0.99**2
                ),
            ),
        ),
    )

    relation_b = RelationIntent(
        name="relation-b",
        phrase_embeddings=(
            normalized(
                0.995,
                np.sqrt(
                    1.0 - 0.995**2
                ),
            ),
        ),
    )

    root = LinkedMemory(
        memory_id="root",
        source="source",
        relation=root_relation,
        target="middle",
        trigger_centers=(
            query,
        ),
        target_vector=normalized(
            0.0,
            1.0,
        ),
        successor_ids=(
            "successor-a",
            "successor-b",
        ),
        radius=2.0,
    )

    successor_a = LinkedMemory(
        memory_id="successor-a",
        source="middle",
        relation=relation_a,
        target="answer-a",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            -1.0,
            0.0,
        ),
        radius=2.0,
    )

    successor_b = LinkedMemory(
        memory_id="successor-b",
        source="middle",
        relation=relation_b,
        target="answer-b",
        trigger_centers=(
            normalized(
                0.0,
                1.0,
            ),
        ),
        target_vector=normalized(
            1.0,
            0.0,
        ),
        radius=2.0,
    )

    graph = LinkedMemoryGraph(
        memories=(
            root,
            successor_a,
            successor_b,
        ),
        root_ids=(
            "root",
        ),
    )

    baseline = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert baseline.traversed_memory_ids == (
        "root",
        "successor-b",
    )

    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=(
            "root",
            "successor-a",
        ),
        context_embedding=query,
    )

    reinforced = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
        thought_routes=store,
    )

    assert reinforced.traversed_memory_ids == (
        "root",
        "successor-a",
    )