import numpy as np

from src.thought_routes import (
    ReinforcementConfig,
    ThoughtRouteStore,
)


def unit_vector(
    index: int,
    size: int = 4,
) -> np.ndarray:
    vector = np.zeros(
        size,
        dtype=np.float32,
    )
    vector[index] = 1.0
    return vector


def test_validated_success_creates_route() -> None:
    store = ThoughtRouteStore()

    route = store.record_validated_success(
        memory_ids=(
            "bob-works-at-globex",
            "globex-headquartered-in-paris",
        ),
        context_embedding=unit_vector(0),
    )

    assert route.memory_ids == (
        "bob-works-at-globex",
        "globex-headquartered-in-paris",
    )
    assert route.strength == 0.10
    assert route.successful_uses == 1
    assert route.explicit_confirmations == 0
    assert len(route.context_embeddings) == 1


def test_same_ordered_path_reuses_existing_route() -> None:
    store = ThoughtRouteStore()

    first = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    second = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(1),
    )

    assert first.route_id == second.route_id
    assert len(store.routes()) == 1
    assert second.successful_uses == 2


def test_different_ordered_path_creates_different_route() -> None:
    store = ThoughtRouteStore()

    first = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    second = store.record_validated_success(
        memory_ids=("a", "c"),
        context_embedding=unit_vector(0),
    )

    assert first.route_id != second.route_id
    assert len(store.routes()) == 2