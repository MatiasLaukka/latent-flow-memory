import numpy as np
import pytest

from src.baselines import (
    kernel_weighted_target,
    nearest_target,
    one_step_target_flow,
)
from src.encoder import TextEncoder
from src.evaluation import rank_candidates
from src.relations import (
    TargetMemory,
    target_flow,
)


def normalized(
    values: list[float],
) -> np.ndarray:
    vector = np.array(
        values,
        dtype=np.float32,
    )

    return (
        vector
        / np.linalg.norm(vector)
    )


def test_nearest_target_returns_target_of_closest_center():
    query = normalized(
        [1.0, 0.0]
    )

    close_memory = TargetMemory(
        center=normalized(
            [0.9, 0.1]
        ),
        target=normalized(
            [0.0, 1.0]
        ),
        radius=1.0,
    )

    far_memory = TargetMemory(
        center=normalized(
            [-1.0, 0.0]
        ),
        target=normalized(
            [0.0, -1.0]
        ),
        radius=1.0,
    )

    result = nearest_target(
        query=query,
        memories=[
            close_memory,
            far_memory,
        ],
    )

    assert np.allclose(
        result,
        close_memory.target,
    )


def test_nearest_target_rejects_empty_memory_list():
    query = normalized(
        [1.0, 0.0]
    )

    try:
        nearest_target(
            query=query,
            memories=[],
        )
    except ValueError as error:
        assert str(error) == (
            "At least one memory is required."
        )
    else:
        raise AssertionError(
            "Expected ValueError for empty memories."
        )


def test_kernel_weighted_target_favors_nearby_memory():
    query = normalized(
        [1.0, 0.0]
    )

    close_memory = TargetMemory(
        center=normalized(
            [0.9, 0.1]
        ),
        target=normalized(
            [0.0, 1.0]
        ),
        radius=0.5,
    )

    far_memory = TargetMemory(
        center=normalized(
            [-1.0, 0.0]
        ),
        target=normalized(
            [0.0, -1.0]
        ),
        radius=0.5,
    )

    result = kernel_weighted_target(
        query=query,
        memories=[
            close_memory,
            far_memory,
        ],
    )

    close_similarity = float(
        np.dot(
            result,
            close_memory.target,
        )
    )

    far_similarity = float(
        np.dot(
            result,
            far_memory.target,
        )
    )

    assert close_similarity > far_similarity


def test_one_step_target_flow_moves_toward_target():
    query = normalized(
        [1.0, 0.0]
    )

    target = normalized(
        [0.0, 1.0]
    )

    memory = TargetMemory(
        center=query,
        target=target,
        radius=1.0,
    )

    result = one_step_target_flow(
        query=query,
        memories=[memory],
        step_size=0.1,
    )

    assert (
        float(np.dot(result, target))
        > float(np.dot(query, target))
    )


def create_target_memories(
    encoder: TextEncoder,
    trigger_texts: list[str],
    target_text: str,
    radius: float = 0.35,
) -> list[TargetMemory]:
    """
    Create several narrow trigger regions pointing toward
    the same target concept.
    """

    target = encoder.encode(target_text)

    return [
        TargetMemory(
            center=encoder.encode(trigger_text),
            target=target,
            strength=1.0,
            radius=radius,
        )
        for trigger_text in trigger_texts
    ]


@pytest.fixture(scope="module")
def composition_setup():
    """
    Load the sentence encoder only once for this test module.
    """

    encoder = TextEncoder()

    alice_to_acme = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Alice's employer",
            "company Alice works for",
            "where Alice works",
            "Alice's workplace",
        ],
        target_text="Acme",
    )

    acme_to_helsinki = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Acme's headquarters",
            "where Acme is headquartered",
            "location of Acme",
            "Acme headquarters city",
        ],
        target_text="Helsinki",
    )

    acme_to_paris = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Acme's headquarters",
            "where Acme is headquartered",
            "location of Acme",
            "Acme headquarters city",
        ],
        target_text="Paris",
    )

    bob_to_globex = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Bob's employer",
            "company Bob works for",
            "where Bob works",
            "Bob's workplace",
        ],
        target_text="Globex",
    )

    globex_to_paris = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Globex's headquarters",
            "where Globex is headquartered",
            "location of Globex",
            "Globex headquarters city",
        ],
        target_text="Paris",
    )

    globex_to_helsinki = create_target_memories(
        encoder=encoder,
        trigger_texts=[
            "Globex's headquarters",
            "where Globex is headquartered",
            "location of Globex",
            "Globex headquarters city",
        ],
        target_text="Helsinki",
    )

    candidate_names = [
        "Alice",
        "Bob",
        "Acme",
        "Globex",
        "Helsinki",
        "Paris",
        "Finland",
        "France",
    ]

    candidates = {
        name: encoder.encode(name)
        for name in candidate_names
    }

    return {
        "encoder": encoder,
        "candidates": candidates,
        "alice_to_acme": alice_to_acme,
        "acme_to_helsinki": acme_to_helsinki,
        "acme_to_paris": acme_to_paris,
        "bob_to_globex": bob_to_globex,
        "globex_to_paris": globex_to_paris,
        "globex_to_helsinki": globex_to_helsinki,
    }

def flow_winner(
    query: np.ndarray,
    memories: list[TargetMemory],
    candidates: dict[str, np.ndarray],
) -> str:
    """
    Run a 50-step flow and return the highest-ranked
    candidate at the final state.
    """

    final_state, _ = target_flow(
        start=query,
        memories=memories,
        steps=50,
        step_size=0.1,
    )

    ranking = rank_candidates(
        state=final_state,
        candidates=candidates,
    )

    return ranking[0][0]

def test_alice_full_chain_reaches_helsinki(
    composition_setup,
):
    setup = composition_setup
    encoder = setup["encoder"]

    query = encoder.encode(
        "Where is Alice's workplace headquartered?"
    )

    winner = flow_winner(
        query=query,
        memories=(
            setup["alice_to_acme"]
            + setup["acme_to_helsinki"]
        ),
        candidates=setup["candidates"],
    )

    assert winner == "Helsinki"


def test_alice_first_hop_only_stops_at_acme(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Alice's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=setup["alice_to_acme"],
            candidates=setup["candidates"],
        )

        assert winner == "Acme"


def test_alice_second_hop_only_does_not_reach_helsinki(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Alice's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=setup["acme_to_helsinki"],
            candidates=setup["candidates"],
        )

        assert winner != "Helsinki"


def test_alice_shuffled_second_hop_reaches_paris(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Alice's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=(
                setup["alice_to_acme"]
                + setup["acme_to_paris"]
            ),
            candidates=setup["candidates"],
        )

        assert winner == "Paris"

def test_bob_full_chain_reaches_paris(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Bob's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=(
                setup["bob_to_globex"]
                + setup["globex_to_paris"]
            ),
            candidates=setup["candidates"],
        )

        assert winner == "Paris"


def test_bob_first_hop_only_stops_at_globex(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Bob's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=setup["bob_to_globex"],
            candidates=setup["candidates"],
        )

        assert winner == "Globex"


def test_bob_second_hop_only_does_not_reach_paris(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Bob's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=setup["globex_to_paris"],
            candidates=setup["candidates"],
        )

        assert winner != "Paris"


def test_bob_shuffled_second_hop_reaches_helsinki(
    composition_setup,
):
        setup = composition_setup
        encoder = setup["encoder"]

        query = encoder.encode(
            "Where is Bob's workplace headquartered?"
        )

        winner = flow_winner(
            query=query,
            memories=(
                setup["bob_to_globex"]
                + setup["globex_to_helsinki"]
            ),
            candidates=setup["candidates"],
        )

        assert winner == "Helsinki"