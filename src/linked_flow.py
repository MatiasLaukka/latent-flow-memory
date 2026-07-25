from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.dynamics import normalize
from src.linked_memory import LinkedMemory
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