import numpy as np
import pytest

from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    RelationIntent,
    score_phrase_set,
    score_successors,
    select_root_memory,
    select_successor,
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


def test_score_phrase_set_uses_best_phrase():
    query = vector(
        1.0,
        0.0,
    )

    score = score_phrase_set(
        query=query,
        phrase_embeddings=(
            vector(
                0.0,
                1.0,
            ),
            vector(
                0.9,
                0.1,
            ),
        ),
    )

    expected = float(
        np.dot(
            query,
            vector(
                0.9,
                0.1,
            ),
        )
    )

    assert np.isclose(
        score,
        expected,
    )


def test_score_phrase_set_rejects_empty_phrases():
    with pytest.raises(
        ValueError,
        match=(
            "At least one phrase embedding "
            "is required."
        ),
    ):
        score_phrase_set(
            query=vector(
                1.0,
                0.0,
            ),
            phrase_embeddings=(),
        )


def test_select_root_memory_uses_best_trigger_match():
    works_at = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            vector(
                1.0,
                0.0,
            ),
        ),
    )

    bob = LinkedMemory(
        memory_id="bob-root",
        source="Bob",
        relation=works_at,
        target="Globex",
        trigger_centers=(
            vector(
                1.0,
                0.0,
            ),
            vector(
                0.8,
                0.2,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
    )

    alice = LinkedMemory(
        memory_id="alice-root",
        source="Alice",
        relation=works_at,
        target="Acme",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            -1.0,
            0.0,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(
            bob,
            alice,
        ),
        root_ids=(
            bob.memory_id,
            alice.memory_id,
        ),
    )

    selected = select_root_memory(
        graph=graph,
        query=vector(
            1.0,
            0.0,
        ),
    )

    assert selected is not None
    assert (
        selected.memory.memory_id
        == "bob-root"
    )
    assert np.isclose(
        selected.score,
        1.0,
    )


def test_select_root_memory_returns_none_without_roots():
    memory = LinkedMemory(
        memory_id="not-a-root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                1.0,
                0.0,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(memory,),
        root_ids=(),
    )

    selected = select_root_memory(
        graph=graph,
        query=vector(
            1.0,
            0.0,
        ),
    )

    assert selected is None


def test_select_root_memory_respects_minimum_score():
    memory = LinkedMemory(
        memory_id="bob-root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            1.0,
            0.0,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(memory,),
        root_ids=(
            memory.memory_id,
        ),
    )

    selected = select_root_memory(
        graph=graph,
        query=vector(
            1.0,
            0.0,
        ),
        minimum_score=0.5,
    )

    assert selected is None


def test_score_successors_uses_relation_intent():
    headquarters = LinkedMemory(
        memory_id="hq",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
                vector(
                    0.9,
                    0.1,
                ),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            -1.0,
            0.0,
        ),
    )

    founder = LinkedMemory(
        memory_id="founder",
        source="Globex",
        relation=RelationIntent(
            name="founded-by",
            phrase_embeddings=(
                vector(
                    0.0,
                    1.0,
                ),
            ),
        ),
        target="Susan",
        trigger_centers=(
            vector(
                1.0,
                0.0,
            ),
        ),
        target_vector=vector(
            0.0,
            -1.0,
        ),
    )

    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    0.5,
                    0.5,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                0.5,
                0.5,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
        successor_ids=(
            headquarters.memory_id,
            founder.memory_id,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(
            root,
            headquarters,
            founder,
        ),
        root_ids=(
            root.memory_id,
        ),
    )

    scores = score_successors(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(
            1.0,
            0.0,
        ),
    )

    assert len(scores) == 2
    assert (
        scores[0].memory.memory_id
        == "hq"
    )
    assert (
        scores[1].memory.memory_id
        == "founder"
    )


def test_select_successor_returns_best_relation():
    headquarters = LinkedMemory(
        memory_id="hq",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            -1.0,
            0.0,
        ),
    )

    founder = LinkedMemory(
        memory_id="founder",
        source="Globex",
        relation=RelationIntent(
            name="founded-by",
            phrase_embeddings=(
                vector(
                    0.0,
                    1.0,
                ),
            ),
        ),
        target="Susan",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            1.0,
            0.0,
        ),
    )

    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    0.5,
                    0.5,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                0.5,
                0.5,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
        successor_ids=(
            headquarters.memory_id,
            founder.memory_id,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(
            root,
            headquarters,
            founder,
        ),
        root_ids=(
            root.memory_id,
        ),
    )

    selected = select_successor(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(
            1.0,
            0.0,
        ),
    )

    assert selected is not None
    assert (
        selected.memory.memory_id
        == "hq"
    )


def test_select_successor_returns_none_without_successors():
    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                1.0,
                0.0,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
    )

    graph = LinkedMemoryGraph(
        memories=(root,),
        root_ids=(
            root.memory_id,
        ),
    )

    selected = select_successor(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(
            1.0,
            0.0,
        ),
    )

    assert selected is None


def test_select_successor_respects_minimum_score():
    successor = LinkedMemory(
        memory_id="hq",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(
                    0.0,
                    1.0,
                ),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(
                0.0,
                1.0,
            ),
        ),
        target_vector=vector(
            -1.0,
            0.0,
        ),
    )

    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(
                    1.0,
                    0.0,
                ),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(
                1.0,
                0.0,
            ),
        ),
        target_vector=vector(
            0.0,
            1.0,
        ),
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

    selected = select_successor(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(
            1.0,
            0.0,
        ),
        minimum_score=0.5,
    )

    assert selected is None