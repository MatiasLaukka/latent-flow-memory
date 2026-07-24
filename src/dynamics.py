import numpy as np

from src.field import vector_field
from src.memory import FlowMemory


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    Normalize a vector so that its Euclidean length is 1.

    For a vector:

        x = [x1, x2, ..., xn]

    its Euclidean norm is:

        ||x|| = sqrt(x1² + x2² + ... + xn²)

    We divide every component by this norm:

        x_normalized = x / ||x||

    This keeps our latent states on the same unit hypersphere
    as the normalized SentenceTransformer embeddings.
    """

    norm = np.linalg.norm(vector)

    # Avoid division by zero.
    #
    # A zero vector has no direction, so normalization
    # would be mathematically undefined.
    if norm == 0:
        return vector.copy()

    return vector / norm


def flow(
    start: np.ndarray,
    memories: list[FlowMemory],
    steps: int = 10,
    step_size: float = 0.1,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Move a latent state through the learned vector field.

    The basic update rule is:

        x(t+1) = x(t) + alpha * V(x(t))

    where:

        x(t)  = current latent state
        V(x)  = total vector-field influence
        alpha = step_size

    After each movement, the state is normalized again.

    Returns
    -------
    final_state:
        The position after all flow steps.

    trajectory:
        Every visited state, including the starting state.
    """

    # Copy the input so that we do not accidentally modify
    # the caller's original NumPy array.
    current = np.array(
        start,
        dtype=np.float32,
        copy=True,
    )

    # Ensure we begin with a normalized state.
    current = normalize(current)

    # Store the full path.
    #
    # This will later let us visualize:
    #
    # x0 -> x1 -> x2 -> ... -> xN
    trajectory = [
        current.copy()
    ]

    for _ in range(steps):
        # Ask the vector field:
        #
        # "At this exact position in latent space,
        # which direction should I move?"
        direction = vector_field(
            current,
            memories,
        )

        # Euler integration step:
        #
        # new_position =
        # current_position + step_size * direction
        #
        # This is the simplest numerical approximation of:
        #
        # dx/dt = V(x)
        current = (
            current
            + step_size * direction
        )

        # Keep the state normalized.
        #
        # SentenceTransformer embeddings live on a normalized
        # hypersphere in our experiment, so we want the moving
        # state to remain in the same geometry.
        current = normalize(current)

        # Save this point in the trajectory.
        trajectory.append(
            current.copy()
        )

    return current, trajectory