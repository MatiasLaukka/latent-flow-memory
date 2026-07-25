from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.dynamics import normalize
from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    score_successors,
    select_root_memory,
)
from src.relations import gaussian_weight

class RoutingStatus(str, Enum):
    """
    Possible outcomes from linked-memory routing.
    """

    COMPLETED = "completed"

    NO_ROOT_MEMORY = "no-root-memory"

    ROOT_SCORE_BELOW_THRESHOLD = (
        "root-score-below-threshold"
    )

    TARGET_NOT_REACHED = "target-not-reached"

    NO_SUCCESSORS = "no-successors"

    NO_MATCHING_SUCCESSOR = (
        "no-matching-successor"
    )

    CYCLE_DETECTED = "cycle-detected"

    MAXIMUM_HOPS_REACHED = (
        "maximum-hops-reached"
    )


@dataclass(frozen=True)
class EdgeFlowResult:
    """
    Result of moving through one selected linked-memory
    transition.
    """

    status: RoutingStatus
    final_state: np.ndarray
    trajectory: list[np.ndarray]
    steps_used: int
    target_reached: bool

@dataclass(frozen=True)
class HopRecord:
    memory_id: str
    relation_name: str
    source: str
    target: str
    selection_score: float
    successor_scores: tuple[
        tuple[str, float],
        ...
    ]
    steps_used: int


@dataclass(frozen=True)
class LinkedFlowResult:
    status: RoutingStatus
    final_state: np.ndarray
    traversed_memory_ids: tuple[str, ...]
    traversed_relation_names: tuple[str, ...]
    hop_records: tuple[HopRecord, ...]
    total_steps: int
    root_selection_score: float | None
    failure_reason: str | None

def run_linked_flow(
    query: np.ndarray,
    graph: LinkedMemoryGraph,
    *,
    handoff_threshold: float = 0.95,
    max_steps_per_hop: int = 50,
    step_size: float = 0.1,
    max_hops: int = 4,
    root_minimum_score: float | None = None,
    successor_minimum_score: float | None = None,
) -> LinkedFlowResult:
    """
    Select a root memory and traverse query-conditioned
    successor memories.
    """

    original_query = normalize(
        np.array(
            query,
            dtype=np.float32,
            copy=True,
        )
    )

    current = original_query.copy()

    if not graph.root_ids:
        return LinkedFlowResult(
            status=RoutingStatus.NO_ROOT_MEMORY,
            final_state=current,
            traversed_memory_ids=(),
            traversed_relation_names=(),
            hop_records=(),
            total_steps=0,
            root_selection_score=None,
            failure_reason=(
                "The graph contains no root memories."
            ),
        )

    root_without_threshold = select_root_memory(
        graph=graph,
        query=original_query,
    )

    if root_without_threshold is None:
        return LinkedFlowResult(
            status=RoutingStatus.NO_ROOT_MEMORY,
            final_state=current,
            traversed_memory_ids=(),
            traversed_relation_names=(),
            hop_records=(),
            total_steps=0,
            root_selection_score=None,
            failure_reason=(
                "No root memory could be selected."
            ),
        )

    if (
        root_minimum_score is not None
        and root_without_threshold.score
        < root_minimum_score
    ):
        return LinkedFlowResult(
            status=(
                RoutingStatus
                .ROOT_SCORE_BELOW_THRESHOLD
            ),
            final_state=current,
            traversed_memory_ids=(),
            traversed_relation_names=(),
            hop_records=(),
            total_steps=0,
            root_selection_score=(
                root_without_threshold.score
            ),
            failure_reason=(
                "Best root-memory score was below "
                "the configured threshold."
            ),
        )

    active_score = root_without_threshold

    traversed_ids: list[str] = []
    traversed_relations: list[str] = []
    hop_records: list[HopRecord] = []
    total_steps = 0

    for hop_index in range(max_hops):
        memory = active_score.memory

        if memory.memory_id in traversed_ids:
            return LinkedFlowResult(
                status=RoutingStatus.CYCLE_DETECTED,
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=(
                    "Routing attempted to revisit "
                    f"{memory.memory_id}."
                ),
            )

        edge_result = flow_linked_edge(
            start=current,
            memory=memory,
            handoff_threshold=(
                handoff_threshold
            ),
            max_steps=max_steps_per_hop,
            step_size=step_size,
        )

        current = edge_result.final_state
        total_steps += edge_result.steps_used

        if (
            edge_result.status
            == RoutingStatus.TARGET_NOT_REACHED
        ):
            return LinkedFlowResult(
                status=(
                    RoutingStatus.TARGET_NOT_REACHED
                ),
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=(
                    "The active memory did not reach "
                    "its target threshold."
                ),
            )

        traversed_ids.append(
            memory.memory_id
        )

        traversed_relations.append(
            memory.relation.name
        )

        successor_scores = score_successors(
            graph=graph,
            memory_id=memory.memory_id,
            query=original_query,
        )

        hop_records.append(
            HopRecord(
                memory_id=memory.memory_id,
                relation_name=(
                    memory.relation.name
                ),
                source=memory.source,
                target=memory.target,
                selection_score=(
                    active_score.score
                ),
                successor_scores=tuple(
                    (
                        score.memory.memory_id,
                        score.score,
                    )
                    for score in successor_scores
                ),
                steps_used=edge_result.steps_used,
            )
        )

        if not successor_scores:
            return LinkedFlowResult(
                status=RoutingStatus.COMPLETED,
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=None,
            )

        if hop_index + 1 >= max_hops:
            return LinkedFlowResult(
                status=(
                    RoutingStatus
                    .MAXIMUM_HOPS_REACHED
                ),
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=(
                    "The route still had successors "
                    "after reaching the hop limit."
                ),
            )

        eligible_scores = [
            score
            for score in successor_scores
            if score.memory.memory_id
            not in traversed_ids
        ]

        if not eligible_scores:
            return LinkedFlowResult(
                status=RoutingStatus.CYCLE_DETECTED,
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=(
                    "Every successor would revisit "
                    "an already traversed memory."
                ),
            )

        selected_successor = eligible_scores[0]

        if (
            successor_minimum_score is not None
            and selected_successor.score
            < successor_minimum_score
        ):
            return LinkedFlowResult(
                status=(
                    RoutingStatus
                    .NO_MATCHING_SUCCESSOR
                ),
                final_state=current,
                traversed_memory_ids=tuple(
                    traversed_ids
                ),
                traversed_relation_names=tuple(
                    traversed_relations
                ),
                hop_records=tuple(hop_records),
                total_steps=total_steps,
                root_selection_score=(
                    root_without_threshold.score
                ),
                failure_reason=(
                    "Best successor score was below "
                    "the configured threshold."
                ),
            )

        active_score = selected_successor

    return LinkedFlowResult(
        status=(
            RoutingStatus.MAXIMUM_HOPS_REACHED
        ),
        final_state=current,
        traversed_memory_ids=tuple(
            traversed_ids
        ),
        traversed_relation_names=tuple(
            traversed_relations
        ),
        hop_records=tuple(hop_records),
        total_steps=total_steps,
        root_selection_score=(
            root_without_threshold.score
        ),
        failure_reason=(
            "The maximum hop count was reached."
        ),
    )

def linked_memory_influence(
    memory: LinkedMemory,
    x: np.ndarray,
) -> np.ndarray:
    """
    Calculate the target-seeking influence of one linked
    memory.

    A linked memory may have several trigger centres. The
    strongest local activation controls the edge strength.
    """

    if not memory.trigger_centers:
        raise ValueError(
            "Linked memory must have at least one "
            "trigger centre."
        )

    weight = max(
        gaussian_weight(
            x=x,
            center=center,
            radius=memory.radius,
        )
        for center in memory.trigger_centers
    )

    direction_to_target = (
        memory.target_vector
        - x
    )

    return (
        memory.strength
        * weight
        * direction_to_target
    )


def flow_linked_edge(
    start: np.ndarray,
    memory: LinkedMemory,
    handoff_threshold: float = 0.95,
    max_steps: int = 50,
    step_size: float = 0.1,
) -> EdgeFlowResult:
    """
    Move through one selected linked-memory edge.

    Only the selected memory contributes to the field.
    The edge completes once cosine similarity to the target
    reaches the handoff threshold.
    """

    current = normalize(
        np.array(
            start,
            dtype=np.float32,
            copy=True,
        )
    )

    trajectory = [
        current.copy()
    ]

    initial_similarity = float(
        np.dot(
            current,
            memory.target_vector,
        )
    )

    if initial_similarity >= handoff_threshold:
        return EdgeFlowResult(
            status=RoutingStatus.COMPLETED,
            final_state=current,
            trajectory=trajectory,
            steps_used=0,
            target_reached=True,
        )

    for step in range(
        1,
        max_steps + 1,
    ):
        direction = linked_memory_influence(
            memory=memory,
            x=current,
        )

        current = normalize(
            current
            + step_size * direction
        )

        trajectory.append(
            current.copy()
        )

        target_similarity = float(
            np.dot(
                current,
                memory.target_vector,
            )
        )

        if (
            target_similarity
            >= handoff_threshold
        ):
            return EdgeFlowResult(
                status=RoutingStatus.COMPLETED,
                final_state=current,
                trajectory=trajectory,
                steps_used=step,
                target_reached=True,
            )

    return EdgeFlowResult(
        status=RoutingStatus.TARGET_NOT_REACHED,
        final_state=current,
        trajectory=trajectory,
        steps_used=max_steps,
        target_reached=False,
    )