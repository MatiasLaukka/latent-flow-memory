from dataclasses import dataclass

import numpy as np


@dataclass
class TargetMemory:
    """
    Represents a local memory basin in latent space.

    center:
        The semantic region where the memory activates.

    target:
        The latent point toward which nearby states move.

    strength:
        Controls the strength of the movement.

    radius:
        Controls how wide the activation region is.
    """

    center: np.ndarray
    target: np.ndarray
    strength: float = 1.0
    radius: float = 0.35


def gaussian_weight(
    x: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> float:
    """
    Calculate Gaussian activation around a centre.

    The result is 1 at the centre and approaches 0
    as distance increases.
    """

    if radius <= 0:
        raise ValueError(
            "Memory radius must be greater than zero."
        )

    difference = x - center

    distance_squared = float(
        np.dot(
            difference,
            difference,
        )
    )

    return float(
        np.exp(
            -distance_squared
            / (2.0 * radius**2)
        )
    )


def target_memory_influence(
    memory: TargetMemory,
    x: np.ndarray,
) -> np.ndarray:
    """
    Calculate the direction in which one target memory
    moves the current latent state.

    Unlike the original FlowMemory, the direction depends
    on the current state:

        direction = target - x

    This means movement becomes smaller as x approaches
    the target.
    """

    weight = gaussian_weight(
        x=x,
        center=memory.center,
        radius=memory.radius,
    )

    direction_to_target = (
        memory.target
        - x
    )

    return (
        memory.strength
        * weight
        * direction_to_target
    )