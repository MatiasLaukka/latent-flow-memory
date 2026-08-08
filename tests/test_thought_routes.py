import numpy as np
import pytest

from src.thought_routes import (
    ReinforcementConfig,
    ThoughtRouteStore,
)
from src.linked_flow import RoutingStatus


class FakeFlowResult:
    """
    Minimal stand-in for LinkedFlowResult.

    The thought-route store should only care about the
    routing status and the memories that were traversed.
    Keeping this fake small also prevents the unit test
    from depending on every LinkedFlowResult field.
    """

    def __init__(
        self,
        *,
        status: RoutingStatus,
        traversed_memory_ids: tuple[str, ...],
    ) -> None:
        self.status = status
        self.traversed_memory_ids = (
            traversed_memory_ids
        )


def test_incorrect_outcome_does_not_create_route() -> None:
    store = ThoughtRouteStore()

    result = FakeFlowResult(
        status=RoutingStatus.COMPLETED,
        traversed_memory_ids=("a", "b"),
    )

    learned = store.record_validated_result(
        result=result,
        context_embedding=unit_vector(0),
        outcome_valid=False,
    )

    assert learned is None
    assert store.routes() == ()


def test_incomplete_result_does_not_create_route() -> None:
    store = ThoughtRouteStore()

    result = FakeFlowResult(
        status=RoutingStatus.TARGET_NOT_REACHED,
        traversed_memory_ids=("a",),
    )

    learned = store.record_validated_result(
        result=result,
        context_embedding=unit_vector(0),
        outcome_valid=True,
    )

    assert learned is None
    assert store.routes() == ()


def test_single_hop_result_does_not_create_thought_route() -> None:
    store = ThoughtRouteStore()

    result = FakeFlowResult(
        status=RoutingStatus.COMPLETED,
        traversed_memory_ids=("a",),
    )

    learned = store.record_validated_result(
        result=result,
        context_embedding=unit_vector(0),
        outcome_valid=True,
    )

    assert learned is None
    assert store.routes() == ()


def test_valid_completed_multi_hop_result_creates_route() -> None:
    store = ThoughtRouteStore()

    result = FakeFlowResult(
        status=RoutingStatus.COMPLETED,
        traversed_memory_ids=("a", "b"),
    )

    learned = store.record_validated_result(
        result=result,
        context_embedding=unit_vector(0),
        outcome_valid=True,
    )

    assert learned is not None
    assert learned.memory_ids == (
        "a",
        "b",
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


from src.thought_routes import (
    context_similarity,
    route_bonus,
)


def test_successful_reuse_adds_small_increment() -> None:
    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    route = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(1),
    )

    assert route.strength == 0.11
    assert route.successful_uses == 2


def test_explicit_confirmation_adds_large_increment() -> None:
    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    route = store.record_explicit_confirmation(
        memory_ids=("a", "b"),
    )

    assert route.strength == 0.25
    assert route.explicit_confirmations == 1


def test_strength_is_capped() -> None:
    config = ReinforcementConfig(
        initial_strength=0.90,
        successful_reuse_increment=0.20,
        explicit_confirmation_increment=0.30,
        maximum_strength=1.00,
        bonus_scale=0.10,
    )

    store = ThoughtRouteStore(
        config=config,
    )

    store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    reused = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(1),
    )

    confirmed = store.record_explicit_confirmation(
        memory_ids=("a", "b"),
    )

    assert reused.strength == 1.00
    assert confirmed.strength == 1.00


def test_identical_context_is_not_duplicated() -> None:
    store = ThoughtRouteStore()
    context = unit_vector(0)

    store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=context,
    )

    route = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=context.copy(),
    )

    assert len(route.context_embeddings) == 1
    assert route.successful_uses == 2


def test_context_similarity_uses_best_context() -> None:
    store = ThoughtRouteStore()

    route = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    route = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(1),
    )

    similarity = context_similarity(
        route=route,
        query=unit_vector(1),
    )

    assert similarity == 1.0


def test_route_bonus_is_bounded_product() -> None:
    store = ThoughtRouteStore()

    route = store.record_validated_success(
        memory_ids=("a", "b"),
        context_embedding=unit_vector(0),
    )

    bonus = route_bonus(
        route=route,
        query=unit_vector(0),
        bonus_scale=0.10,
    )

    assert bonus == pytest.approx(0.01)


def test_compatible_routes_require_matching_prefix() -> None:
    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=(
            "root-a",
            "next-a",
        ),
        context_embedding=unit_vector(0),
    )

    store.record_validated_success(
        memory_ids=(
            "root-b",
            "next-b",
        ),
        context_embedding=unit_vector(1),
    )

    matches = store.compatible_routes(
        prefix=("root-a",),
    )

    assert len(matches) == 1
    assert matches[0].memory_ids == (
        "root-a",
        "next-a",
    )