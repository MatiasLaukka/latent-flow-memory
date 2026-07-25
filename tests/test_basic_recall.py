import numpy as np

from src.dynamics import flow
from src.encoder import TextEncoder
from src.memory import FlowMemory


def cosine_similarity(
    first: np.ndarray,
    second: np.ndarray,
) -> float:
    """
    Because our embeddings are normalized, cosine similarity
    is simply their dot product.
    """
    return float(np.dot(first, second))


def test_flow_memory_moves_query_closer_to_rufus():
    # Load the frozen semantic encoder.
    encoder = TextEncoder()

    # This is the semantic region where we want the memory
    # to become relevant.
    memory_center = encoder.encode(
        "Matu's cat's name"
    )

    # This is the target concept the learned flow should
    # steer the activation toward.
    target = encoder.encode(
        "Rufus"
    )

    # The simplest possible learned direction:
    #
    # target - center
    #
    # Geometrically, this points from the memory center
    # toward Rufus in the 384-dimensional embedding space.
    direction = target - memory_center

    memory = FlowMemory(
        center=memory_center,
        direction=direction,

        # Start with fairly strong influence so the first
        # controlled experiment clearly demonstrates movement.
        strength=1.0,

        # Radius determines how broad the local influence is.
        # We may tune this later experimentally.
        radius=1.0,
    )

    # Importantly, this query is NOT identical to the memory center.
    #
    # We want to see whether a semantically related query can
    # still enter the memory's region of influence.
    query = encoder.encode(
        "What is Matu's cat's name?"
    )

    similarity_before = cosine_similarity(
        query,
        target,
    )

    final_state, trajectory = flow(
        start=query,
        memories=[memory],
        steps=10,
        step_size=0.1,
    )

    similarity_after = cosine_similarity(
        final_state,
        target,
    )

    # This is our core Milestone 1 hypothesis:
    #
    # applying the learned latent flow should move the query
    # closer to the representation of Rufus.
    assert similarity_after > similarity_before