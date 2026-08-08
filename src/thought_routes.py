from dataclasses import dataclass, replace

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
            route = replace(
                existing,
                context_embeddings=(
                    existing.context_embeddings
                    + (context,)
                ),
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
                    existing.successful_uses
                    + 1
                ),
            )

        self._routes[key] = route

        return route

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
                existing.explicit_confirmations
                + 1
            ),
        )

        self._routes[key] = route

        return route