from dataclasses import dataclass, replace
from typing import Protocol

import numpy as np


@dataclass(frozen=True)
class ReinforcementConfig:
    initial_strength: float = 0.10
    successful_reuse_increment: float = 0.01
    explicit_confirmation_increment: float = 0.15
    maximum_strength: float = 1.00
    bonus_scale: float = 0.10


@dataclass(frozen=True)
class ThoughtRoute:
    route_id: str
    memory_ids: tuple[str, ...]
    context_embeddings: tuple[np.ndarray, ...]
    strength: float
    successful_uses: int
    explicit_confirmations: int

class CompletedRouteResult(Protocol):
    """
    Structural interface for a routing result.

    We deliberately do not import LinkedFlowResult here.

    Why?
    linked_flow.py already imports ThoughtRouteStore.
    If thought_routes.py imported linked_flow.py back,
    the two modules would depend on each other and create
    a circular import.

    A Protocol lets us say:
    "I accept anything with these fields."
    """

    status: object
    traversed_memory_ids: tuple[str, ...]

def context_similarity(
    *,
    route: ThoughtRoute,
    query: np.ndarray,
) -> float:
    if not route.context_embeddings:
        return 0.0

    return max(
        float(
            np.dot(
                query,
                context,
            )
        )
        for context
        in route.context_embeddings
    )


def route_bonus(
    *,
    route: ThoughtRoute,
    query: np.ndarray,
    bonus_scale: float,
) -> float:
    similarity = max(
        0.0,
        context_similarity(
            route=route,
            query=query,
        ),
    )

    return (
        bonus_scale
        * similarity
        * route.strength
    )


class ThoughtRouteStore:
    def __init__(
        self,
        config: ReinforcementConfig | None = None,
    ) -> None:
        self.config = (
            config
            or ReinforcementConfig()
        )

        self._routes: dict[
            tuple[str, ...],
            ThoughtRoute,
        ] = {}

    def routes(
        self,
    ) -> tuple[ThoughtRoute, ...]:
        return tuple(
            self._routes.values()
        )

    def get(
        self,
        memory_ids: tuple[str, ...],
    ) -> ThoughtRoute | None:
        return self._routes.get(
            tuple(memory_ids)
        )

    def compatible_routes(
        self,
        *,
        prefix: tuple[str, ...],
    ) -> tuple[ThoughtRoute, ...]:
        wanted = tuple(prefix)

        return tuple(
            route
            for route
            in self._routes.values()
            if route.memory_ids[
                :len(wanted)
            ] == wanted
        )

    def record_validated_success(
        self,
        *,
        memory_ids: tuple[str, ...],
        context_embedding: np.ndarray,
    ) -> ThoughtRoute:
        key = tuple(memory_ids)

        if len(key) < 2:
            raise ValueError(
                "Thought routes must contain "
                "at least two memories."
            )

        context = np.array(
            context_embedding,
            dtype=np.float32,
            copy=True,
        )

        existing = self._routes.get(
            key
        )

        if existing is None:
            route = ThoughtRoute(
                route_id=" -> ".join(
                    key
                ),
                memory_ids=key,
                context_embeddings=(
                    context,
                ),
                strength=(
                    self.config
                    .initial_strength
                ),
                successful_uses=1,
                explicit_confirmations=0,
            )

        else:
            already_known = any(
                float(
                    np.dot(
                        context,
                        known_context,
                    )
                ) >= 0.999999
                for known_context
                in existing.context_embeddings
            )

            if already_known:
                contexts = (
                    existing
                    .context_embeddings
                )
            else:
                contexts = (
                    existing
                    .context_embeddings
                    + (context,)
                )

            route = replace(
                existing,
                context_embeddings=contexts,
                strength=min(
                    (
                        self.config
                        .maximum_strength
                    ),
                    (
                        existing.strength
                        + self.config
                        .successful_reuse_increment
                    ),
                ),
                successful_uses=(
                    existing
                    .successful_uses
                    + 1
                ),
            )

        self._routes[key] = route

        return route

    def record_validated_result(
        self,
        *,
        result: CompletedRouteResult,
        context_embedding: np.ndarray,
        outcome_valid: bool,
    ) -> ThoughtRoute | None:
        """
        Learn from a routing result only when there is
        positive evidence that the complete route worked.

        The router selecting a route is not itself proof
        that the route was correct. That distinction is
        important: otherwise routing mistakes would train
        themselves into stronger future habits.
        """

        # RoutingStatus inherits from str + Enum, but using
        # `.value` when available also keeps this method
        # compatible with lightweight test doubles.
        status_value = getattr(
            result.status,
            "value",
            result.status,
        )

        # Incomplete routing is not a successful experience.
        if status_value != "completed":
            return None

        # Completion alone is insufficient. The caller must
        # independently validate that the result was useful
        # or correct.
        if not outcome_valid:
            return None

        # ThoughtRoute represents a learned multi-step
        # pattern. A single memory is not a route for this
        # milestone.
        if len(
            result.traversed_memory_ids
        ) < 2:
            return None

        # Once all evidence gates pass, delegate to the
        # existing reinforcement primitive instead of
        # duplicating its creation/update logic.
        return self.record_validated_success(
            memory_ids=(
                result.traversed_memory_ids
            ),
            context_embedding=(
                context_embedding
            ),
        )

    def record_explicit_confirmation(
        self,
        *,
        memory_ids: tuple[str, ...],
    ) -> ThoughtRoute:
        key = tuple(memory_ids)

        existing = self._routes.get(
            key
        )

        if existing is None:
            raise KeyError(
                "Cannot confirm a thought "
                "route that has not been learned."
            )

        route = replace(
            existing,
            strength=min(
                (
                    self.config
                    .maximum_strength
                ),
                (
                    existing.strength
                    + self.config
                    .explicit_confirmation_increment
                ),
            ),
            explicit_confirmations=(
                existing
                .explicit_confirmations
                + 1
            ),
        )

        self._routes[key] = route

        return route