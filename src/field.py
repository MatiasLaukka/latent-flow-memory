import numpy as np

from src.memory import FlowMemory


def memory_influence(
    memory: FlowMemory,
    x: np.ndarray,
) -> np.ndarray:
    """
    Calculate how strongly one memory influences one point.

    The memory uses a Gaussian radial weighting function:

        weight = exp(
            -distance² / (2 × radius²)
        )

    At the memory center:

        distance = 0
        weight = exp(0) = 1

    Farther away, the weight approaches zero.
    """

    # Calculate the difference between the current state
    # and the memory center.
    difference = x - memory.center

    # Squared Euclidean distance:
    #
    # ||x - c||²
    #
    # We do not need the square root because the Gaussian
    # formula uses squared distance directly.
    distance_squared = float(
        np.dot(difference, difference)
    )

    # A radius of zero would cause division by zero.
    if memory.radius <= 0:
        raise ValueError(
            "Memory radius must be greater than zero."
        )

    # Calculate how active the memory is at this location.
    weight = np.exp(
        -distance_squared
        / (2.0 * memory.radius**2)
    )

    # The final influence preserves the stored direction,
    # scaled by both the memory strength and local weight.
    return (
        memory.strength
        * weight
        * memory.direction
    )


def vector_field(
    x: np.ndarray,
    memories: list[FlowMemory],
) -> np.ndarray:
    """
    Calculate the total directional field at one latent point.

    Every active memory contributes a direction vector.

    The total field is:

        V(x) = V₁(x) + V₂(x) + ... + Vₙ(x)
    """

    # Start with a zero vector having the same shape and
    # numerical type as the current state.
    total = np.zeros_like(x)

    for memory in memories:
        total = total + memory_influence(
            memory,
            x,
        )

    return total