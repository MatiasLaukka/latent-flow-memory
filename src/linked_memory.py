from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RelationIntent:
    """
    Describes one semantic relation type.

    phrase_embeddings contains alternative natural-language
    descriptions of the same relation, such as:

        headquartered in
        headquarters location
        main office city
    """

    name: str
    phrase_embeddings: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LinkedMemory:
    """
    Represents one typed transition in the memory graph.

    Example:

        Bob --works-at--> Globex
    """

    memory_id: str
    source: str
    relation: RelationIntent
    target: str

    # Semantic entry points for this specific memory.
    trigger_centers: tuple[np.ndarray, ...]

    # Latent destination of this transition.
    target_vector: np.ndarray

    # Memories that may become active after this transition.
    successor_ids: tuple[str, ...] = ()

    strength: float = 1.0
    radius: float = 0.35


@dataclass(frozen=True)
class MemoryScore:
    """
    Associates a memory with a semantic routing score.
    """

    memory: LinkedMemory
    score: float


@dataclass
class LinkedMemoryGraph:
    """
    Stores linked memories and validates their references.
    """

    memories: tuple[LinkedMemory, ...]
    root_ids: tuple[str, ...]

    _by_id: dict[str, LinkedMemory] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        by_id = {
            memory.memory_id: memory
            for memory in self.memories
        }

        if len(by_id) != len(self.memories):
            raise ValueError(
                "Memory IDs must be unique."
            )

        for memory in self.memories:
            for successor_id in memory.successor_ids:
                if successor_id not in by_id:
                    raise ValueError(
                        "Unknown successor memory ID: "
                        f"{successor_id}"
                    )

        for root_id in self.root_ids:
            if root_id not in by_id:
                raise ValueError(
                    "Unknown root memory ID: "
                    f"{root_id}"
                )

        self._by_id = by_id

    def get(
        self,
        memory_id: str,
    ) -> LinkedMemory:
        """
        Return one memory by its unique identifier.
        """

        return self._by_id[memory_id]

    def root_memories(
        self,
    ) -> tuple[LinkedMemory, ...]:
        """
        Return memories eligible for initial global routing.
        """

        return tuple(
            self._by_id[root_id]
            for root_id in self.root_ids
        )

    def successors(
        self,
        memory_id: str,
    ) -> tuple[LinkedMemory, ...]:
        """
        Return only the memories directly linked from the
        specified memory.
        """

        memory = self.get(memory_id)

        return tuple(
            self._by_id[successor_id]
            for successor_id
            in memory.successor_ids
        )

def score_phrase_set(
    query: np.ndarray,
    phrase_embeddings: tuple[np.ndarray, ...],
) -> float:
    """
    Return the strongest semantic match between a query
    and a set of alternative phrase embeddings.

    The vectors are expected to be normalized.
    """

    if not phrase_embeddings:
        raise ValueError(
            "At least one phrase embedding is required."
        )

    return max(
        float(
            np.dot(
                query,
                phrase_embedding,
            )
        )
        for phrase_embedding
        in phrase_embeddings
    )


def rank_memories_by_triggers(
    memories: tuple[LinkedMemory, ...],
    query: np.ndarray,
) -> list[MemoryScore]:
    """
    Rank memories using their specific trigger centres.
    """

    ranked = [
        MemoryScore(
            memory=memory,
            score=score_phrase_set(
                query=query,
                phrase_embeddings=(
                    memory.trigger_centers
                ),
            ),
        )
        for memory in memories
    ]

    ranked.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return ranked


def select_root_memory(
    graph: LinkedMemoryGraph,
    query: np.ndarray,
    minimum_score: float | None = None,
) -> MemoryScore | None:
    """
    Select the root memory whose trigger phrases best
    match the original query.
    """

    roots = graph.root_memories()

    if not roots:
        return None

    selected = rank_memories_by_triggers(
        memories=roots,
        query=query,
    )[0]

    if (
        minimum_score is not None
        and selected.score < minimum_score
    ):
        return None

    return selected


def score_successors(
    graph: LinkedMemoryGraph,
    memory_id: str,
    query: np.ndarray,
) -> list[MemoryScore]:
    """
    Rank direct successor memories using their relation
    intent phrases.

    This deliberately does not use successor trigger
    centres. Trigger centres control local flow geometry;
    relation phrases control route selection.
    """

    successors = graph.successors(
        memory_id
    )

    ranked = [
        MemoryScore(
            memory=memory,
            score=score_phrase_set(
                query=query,
                phrase_embeddings=(
                    memory.relation.phrase_embeddings
                ),
            ),
        )
        for memory in successors
    ]

    ranked.sort(
        key=lambda item: item.score,
        reverse=True,
    )

    return ranked


def select_successor(
    graph: LinkedMemoryGraph,
    memory_id: str,
    query: np.ndarray,
    minimum_score: float | None = None,
) -> MemoryScore | None:
    """
    Select the direct successor whose relation intent best
    matches the original query.
    """

    scores = score_successors(
        graph=graph,
        memory_id=memory_id,
        query=query,
    )

    if not scores:
        return None

    selected = scores[0]

    if (
        minimum_score is not None
        and selected.score < minimum_score
    ):
        return None

    return selected