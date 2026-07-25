import numpy as np
import pytest

from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    RelationIntent,
)


def vector(*values: float) -> np.ndarray:
    """
    Create a small float32 vector for synthetic tests.
    """

    return np.array(
        values,
        dtype=np.float32,
    )


def test_graph_returns_root_memories():
    works_at = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            vector(1.0, 0.0),
        ),
    )

    root = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=works_at,
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
        successor_ids=(
            "globex-headquartered-in-paris",
        ),
    )

    successor = LinkedMemory(
        memory_id="globex-headquartered-in-paris",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(0.0, 1.0),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(0.0, 1.0),
        ),
        target_vector=vector(-1.0, 0.0),
    )

    graph = LinkedMemoryGraph(
        memories=(
            root,
            successor,
        ),
        root_ids=(
            root.memory_id,
        ),
    )

    assert graph.root_memories() == (root,)


def test_graph_returns_successor_memories():
    successor = LinkedMemory(
        memory_id="globex-headquartered-in-paris",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(0.0, 1.0),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(0.0, 1.0),
        ),
        target_vector=vector(-1.0, 0.0),
    )

    root = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(1.0, 0.0),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
        successor_ids=(
            successor.memory_id,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(
            root,
            successor,
        ),
        root_ids=(
            root.memory_id,
        ),
    )

    assert graph.successors(
        root.memory_id
    ) == (successor,)


def test_graph_gets_memory_by_id():
    memory = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(1.0, 0.0),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
    )

    graph = LinkedMemoryGraph(
        memories=(memory,),
        root_ids=(memory.memory_id,),
    )

    assert graph.get(
        memory.memory_id
    ) is memory


def test_graph_rejects_duplicate_memory_ids():
    relation = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            vector(1.0, 0.0),
        ),
    )

    first = LinkedMemory(
        memory_id="duplicate",
        source="Bob",
        relation=relation,
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
    )

    second = LinkedMemory(
        memory_id="duplicate",
        source="Alice",
        relation=relation,
        target="Acme",
        trigger_centers=(
            vector(0.9, 0.1),
        ),
        target_vector=vector(0.0, -1.0),
    )

    with pytest.raises(
        ValueError,
        match="Memory IDs must be unique.",
    ):
        LinkedMemoryGraph(
            memories=(
                first,
                second,
            ),
            root_ids=(
                first.memory_id,
            ),
        )


def test_graph_rejects_unknown_successor_id():
    memory = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(1.0, 0.0),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
        successor_ids=(
            "missing-memory",
        ),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unknown successor memory ID: "
            "missing-memory"
        ),
    ):
        LinkedMemoryGraph(
            memories=(memory,),
            root_ids=(memory.memory_id,),
        )


def test_graph_rejects_unknown_root_id():
    memory = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(1.0, 0.0),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, 1.0),
    )

    with pytest.raises(
        ValueError,
        match=(
            "Unknown root memory ID: "
            "missing-root"
        ),
    ):
        LinkedMemoryGraph(
            memories=(memory,),
            root_ids=("missing-root",),
        )