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