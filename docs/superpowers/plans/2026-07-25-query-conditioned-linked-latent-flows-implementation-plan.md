# Query-Conditioned Linked Latent Flows Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a query-conditioned linked-memory graph that performs continuous latent movement within each selected relation edge while routing only through typed successor memories.

**Architecture:** `src/linked_memory.py` owns immutable graph structures, validation, and semantic scoring. `src/linked_flow.py` owns staged latent traversal, target handoff, successor routing, failure reporting, and hop/cycle limits. The existing global `TargetMemory` and `target_flow` implementation remains unchanged as a baseline.

**Tech Stack:** Python 3.12, NumPy, pytest, existing `TextEncoder`, existing `normalize`, existing candidate-ranking helpers.

## Global Constraints

- Use `python -m pytest`, not bare `pytest`.
- Run experiment modules from the repository root with `python -m experiments.<module>`.
- Preserve the existing global summed-field implementation for baseline comparisons.
- Write tests before production code and verify the intended failure.
- Use deterministic fixed memory graphs and distractor ordering.
- Maximum linked-flow hops default to `4`.
- No LLM router, learned router, automatic extraction, or automatic memory creation in this milestone.
- No free-text answer generation; use fixed candidate decoding.
- Root selection is global; all routing after the first hop is restricted to explicit successors.

---

## File structure

### Create

- `src/linked_memory.py`
  - Relation intent definitions
  - Linked transition-memory structure
  - Graph construction and validation
  - Root selection
  - Successor relation scoring and selection

- `src/linked_flow.py`
  - Routing statuses
  - Hop records and complete routing result
  - One-edge latent movement
  - Target handoff
  - Linked multi-hop execution
  - Cycle, threshold, and hop-limit handling

- `tests/test_linked_memory.py`
  - Fast synthetic-vector graph and routing tests

- `tests/test_linked_flow.py`
  - Fast synthetic-vector traversal and failure tests

- `experiments/linked_flow_branching.py`
  - Alice/Bob branching graph with headquarters, founder, and operating-region successors

- `experiments/linked_flow_distractor_scaling.py`
  - Fixed-pool comparison of global summed flow and linked flow

### Reuse without modification

- `src/relations.py`
  - `TargetMemory`
  - `target_memory_influence`
  - existing global-field baseline

- `src/dynamics.py`
  - `normalize`

- `src/encoder.py`
  - `TextEncoder`

- `src/evaluation.py`
  - `rank_candidates`

- `src/robustness.py`
  - existing per-query global-flow metrics where applicable

- `src/scaling.py`
  - fixed distractor pool and scaling summaries

---

# Task 1: Linked-memory graph structures and validation

**Files:**
- Create: `src/linked_memory.py`
- Create: `tests/test_linked_memory.py`

**Interfaces:**
- Produces:
  - `RelationIntent`
  - `LinkedMemory`
  - `LinkedMemoryGraph`
  - `MemoryScore`
  - `score_phrase_set(query, phrases)`
  - `select_root_memory(graph, query, minimum_score=None)`
  - `score_successors(graph, memory_id, query)`
  - `select_successor(graph, memory_id, query, minimum_score=None)`

## Step 1: Create the files

- [ ] Run:

```powershell
New-Item -ItemType File -Force `
    src\linked_memory.py, `
    tests\test_linked_memory.py
```

## Step 2: Write the initial failing structure tests

- [ ] Put this in `tests/test_linked_memory.py`:

```python
import numpy as np
import pytest

from src.linked_memory import (
    LinkedMemory,
    LinkedMemoryGraph,
    RelationIntent,
)


def vector(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


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
        match="Unknown successor memory ID: missing-memory",
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
        match="Unknown root memory ID: missing-root",
    ):
        LinkedMemoryGraph(
            memories=(memory,),
            root_ids=("missing-root",),
        )
```

## Step 3: Verify RED

- [ ] Run:

```powershell
python -m pytest tests\test_linked_memory.py -v
```

Expected: import failure because `src.linked_memory` has no structures yet.

## Step 4: Implement the minimal structures

- [ ] Put this initial implementation in `src/linked_memory.py`:

```python
from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class RelationIntent:
    name: str
    phrase_embeddings: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class LinkedMemory:
    memory_id: str
    source: str
    relation: RelationIntent
    target: str
    trigger_centers: tuple[np.ndarray, ...]
    target_vector: np.ndarray
    successor_ids: tuple[str, ...] = ()
    strength: float = 1.0
    radius: float = 0.35


@dataclass(frozen=True)
class MemoryScore:
    memory: LinkedMemory
    score: float


@dataclass
class LinkedMemoryGraph:
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

    def get(self, memory_id: str) -> LinkedMemory:
        return self._by_id[memory_id]

    def root_memories(self) -> tuple[LinkedMemory, ...]:
        return tuple(
            self._by_id[root_id]
            for root_id in self.root_ids
        )

    def successors(
        self,
        memory_id: str,
    ) -> tuple[LinkedMemory, ...]:
        memory = self.get(memory_id)

        return tuple(
            self._by_id[successor_id]
            for successor_id in memory.successor_ids
        )
```

## Step 5: Verify GREEN

- [ ] Run:

```powershell
python -m pytest tests\test_linked_memory.py -v
python -m pytest -v
```

## Step 6: Commit

- [ ] Run:

```powershell
git add src\linked_memory.py tests\test_linked_memory.py
git commit -m "feat: add linked memory graph structures"
git push
```

---

# Task 2: Multi-phrase root and successor scoring

**Files:**
- Modify: `src/linked_memory.py`
- Modify: `tests/test_linked_memory.py`

**Interfaces:**
- Consumes the structures from Task 1.
- Produces deterministic semantic entry and successor selection.

## Step 1: Add failing scoring tests

- [ ] Append to `tests/test_linked_memory.py`:

```python
from src.linked_memory import (
    score_phrase_set,
    score_successors,
    select_root_memory,
    select_successor,
)


def test_score_phrase_set_uses_best_phrase():
    query = vector(1.0, 0.0)

    score = score_phrase_set(
        query=query,
        phrase_embeddings=(
            vector(0.0, 1.0),
            vector(0.9, 0.1),
        ),
    )

    expected = float(
        np.dot(
            query,
            vector(0.9, 0.1),
        )
    )

    assert np.isclose(score, expected)


def test_select_root_memory_uses_best_trigger_match():
    relation = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            vector(1.0, 0.0),
        ),
    )

    bob = LinkedMemory(
        memory_id="bob-root",
        source="Bob",
        relation=relation,
        target="Globex",
        trigger_centers=(
            vector(1.0, 0.0),
            vector(0.8, 0.2),
        ),
        target_vector=vector(0.0, 1.0),
    )

    alice = LinkedMemory(
        memory_id="alice-root",
        source="Alice",
        relation=relation,
        target="Acme",
        trigger_centers=(
            vector(0.0, 1.0),
        ),
        target_vector=vector(-1.0, 0.0),
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
        query=vector(1.0, 0.0),
    )

    assert selected is not None
    assert selected.memory.memory_id == "bob-root"


def test_select_root_memory_respects_minimum_score():
    memory = LinkedMemory(
        memory_id="bob-root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(1.0, 0.0),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(0.0, 1.0),
        ),
        target_vector=vector(1.0, 0.0),
    )

    graph = LinkedMemoryGraph(
        memories=(memory,),
        root_ids=(memory.memory_id,),
    )

    selected = select_root_memory(
        graph=graph,
        query=vector(1.0, 0.0),
        minimum_score=0.5,
    )

    assert selected is None


def test_select_successor_uses_relation_intent_not_trigger():
    headquarters = LinkedMemory(
        memory_id="hq",
        source="Globex",
        relation=RelationIntent(
            name="headquartered-in",
            phrase_embeddings=(
                vector(1.0, 0.0),
                vector(0.9, 0.1),
            ),
        ),
        target="Paris",
        trigger_centers=(
            vector(0.0, 1.0),
        ),
        target_vector=vector(-1.0, 0.0),
    )

    founder = LinkedMemory(
        memory_id="founder",
        source="Globex",
        relation=RelationIntent(
            name="founded-by",
            phrase_embeddings=(
                vector(0.0, 1.0),
            ),
        ),
        target="Susan",
        trigger_centers=(
            vector(1.0, 0.0),
        ),
        target_vector=vector(0.0, -1.0),
    )

    root = LinkedMemory(
        memory_id="root",
        source="Bob",
        relation=RelationIntent(
            name="works-at",
            phrase_embeddings=(
                vector(0.5, 0.5),
            ),
        ),
        target="Globex",
        trigger_centers=(
            vector(0.5, 0.5),
        ),
        target_vector=vector(0.0, 1.0),
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
        root_ids=(root.memory_id,),
    )

    scores = score_successors(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(1.0, 0.0),
    )

    assert scores[0].memory.memory_id == "hq"

    selected = select_successor(
        graph=graph,
        memory_id=root.memory_id,
        query=vector(1.0, 0.0),
    )

    assert selected is not None
    assert selected.memory.memory_id == "hq"
```

## Step 2: Verify RED

- [ ] Run:

```powershell
python -m pytest tests\test_linked_memory.py -v
```

Expected: import errors for the new functions.

## Step 3: Implement scoring

- [ ] Append to `src/linked_memory.py`:

```python
def score_phrase_set(
    query: np.ndarray,
    phrase_embeddings: tuple[np.ndarray, ...],
) -> float:
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
        for phrase_embedding in phrase_embeddings
    )


def rank_memories_by_triggers(
    memories: tuple[LinkedMemory, ...],
    query: np.ndarray,
) -> list[MemoryScore]:
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
    successors = graph.successors(memory_id)

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
```

## Step 4: Verify GREEN

- [ ] Run:

```powershell
python -m pytest tests\test_linked_memory.py -v
python -m pytest -v
```

## Step 5: Commit

- [ ] Run:

```powershell
git add src\linked_memory.py tests\test_linked_memory.py
git commit -m "feat: add query-conditioned memory routing"
git push
```

---

# Task 3: One-edge linked latent movement

**Files:**
- Create: `src/linked_flow.py`
- Create: `tests/test_linked_flow.py`

**Interfaces:**
- Produces:
  - `RoutingStatus`
  - `EdgeFlowResult`
  - `flow_linked_edge(...)`

## Step 1: Write failing edge-flow tests

- [ ] Put this in `tests/test_linked_flow.py`:

```python
import numpy as np

from src.linked_flow import (
    RoutingStatus,
    flow_linked_edge,
)
from src.linked_memory import (
    LinkedMemory,
    RelationIntent,
)


def normalized(*values: float) -> np.ndarray:
    vector = np.array(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def memory(
    *,
    memory_id: str = "edge",
    center: np.ndarray | None = None,
    target: np.ndarray | None = None,
    radius: float = 2.0,
) -> LinkedMemory:
    actual_center = (
        center
        if center is not None
        else normalized(1.0, 0.0)
    )

    actual_target = (
        target
        if target is not None
        else normalized(0.0, 1.0)
    )

    return LinkedMemory(
        memory_id=memory_id,
        source="source",
        relation=RelationIntent(
            name="relation",
            phrase_embeddings=(
                actual_center,
            ),
        ),
        target="target",
        trigger_centers=(
            actual_center,
        ),
        target_vector=actual_target,
        radius=radius,
    )


def test_edge_flow_reaches_target_threshold():
    edge = memory()

    result = flow_linked_edge(
        start=normalized(1.0, 0.0),
        memory=edge,
        handoff_threshold=0.95,
        max_steps=100,
        step_size=0.1,
    )

    assert result.status == RoutingStatus.COMPLETED
    assert result.target_reached is True
    assert result.steps_used > 0
    assert float(
        np.dot(
            result.final_state,
            edge.target_vector,
        )
    ) >= 0.95


def test_edge_flow_reports_target_not_reached():
    edge = memory(
        radius=0.01,
    )

    result = flow_linked_edge(
        start=normalized(-1.0, 0.0),
        memory=edge,
        handoff_threshold=0.99,
        max_steps=3,
        step_size=0.1,
    )

    assert (
        result.status
        == RoutingStatus.TARGET_NOT_REACHED
    )
    assert result.target_reached is False
    assert result.steps_used == 3


def test_edge_flow_trajectory_contains_initial_state():
    edge = memory()

    result = flow_linked_edge(
        start=normalized(1.0, 0.0),
        memory=edge,
        handoff_threshold=0.8,
        max_steps=10,
        step_size=0.1,
    )

    assert np.allclose(
        result.trajectory[0],
        normalized(1.0, 0.0),
    )
    assert len(result.trajectory) == (
        result.steps_used + 1
    )
```

## Step 2: Verify RED

- [ ] Run:

```powershell
python -m pytest tests\test_linked_flow.py -v
```

## Step 3: Implement one-edge flow

- [ ] Put this in `src/linked_flow.py`:

```python
from dataclasses import dataclass
from enum import Enum

import numpy as np

from src.dynamics import normalize
from src.linked_memory import LinkedMemory
from src.relations import gaussian_weight


class RoutingStatus(str, Enum):
    COMPLETED = "completed"
    NO_ROOT_MEMORY = "no-root-memory"
    ROOT_SCORE_BELOW_THRESHOLD = (
        "root-score-below-threshold"
    )
    TARGET_NOT_REACHED = "target-not-reached"
    NO_SUCCESSORS = "no-successors"
    NO_MATCHING_SUCCESSOR = (
        "no-matching-successor"
    )
    CYCLE_DETECTED = "cycle-detected"
    MAXIMUM_HOPS_REACHED = (
        "maximum-hops-reached"
    )


@dataclass(frozen=True)
class EdgeFlowResult:
    status: RoutingStatus
    final_state: np.ndarray
    trajectory: list[np.ndarray]
    steps_used: int
    target_reached: bool


def linked_memory_influence(
    memory: LinkedMemory,
    x: np.ndarray,
) -> np.ndarray:
    weight = max(
        gaussian_weight(
            x=x,
            center=center,
            radius=memory.radius,
        )
        for center in memory.trigger_centers
    )

    return (
        memory.strength
        * weight
        * (
            memory.target_vector
            - x
        )
    )


def flow_linked_edge(
    start: np.ndarray,
    memory: LinkedMemory,
    handoff_threshold: float = 0.95,
    max_steps: int = 50,
    step_size: float = 0.1,
) -> EdgeFlowResult:
    current = normalize(
        np.array(
            start,
            dtype=np.float32,
            copy=True,
        )
    )

    trajectory = [
        current.copy()
    ]

    initial_similarity = float(
        np.dot(
            current,
            memory.target_vector,
        )
    )

    if initial_similarity >= handoff_threshold:
        return EdgeFlowResult(
            status=RoutingStatus.COMPLETED,
            final_state=current,
            trajectory=trajectory,
            steps_used=0,
            target_reached=True,
        )

    for step in range(1, max_steps + 1):
        direction = linked_memory_influence(
            memory=memory,
            x=current,
        )

        current = normalize(
            current
            + step_size * direction
        )

        trajectory.append(
            current.copy()
        )

        similarity = float(
            np.dot(
                current,
                memory.target_vector,
            )
        )

        if similarity >= handoff_threshold:
            return EdgeFlowResult(
                status=RoutingStatus.COMPLETED,
                final_state=current,
                trajectory=trajectory,
                steps_used=step,
                target_reached=True,
            )

    return EdgeFlowResult(
        status=RoutingStatus.TARGET_NOT_REACHED,
        final_state=current,
        trajectory=trajectory,
        steps_used=max_steps,
        target_reached=False,
    )
```

## Step 4: Verify GREEN

- [ ] Run:

```powershell
python -m pytest tests\test_linked_flow.py -v
python -m pytest -v
```

## Step 5: Commit

- [ ] Run:

```powershell
git add src\linked_flow.py tests\test_linked_flow.py
git commit -m "feat: add linked edge flow"
git push
```

---

# Task 4: Complete linked routing and failure reporting

**Files:**
- Modify: `src/linked_flow.py`
- Modify: `tests/test_linked_flow.py`

**Interfaces:**
- Produces:
  - `HopRecord`
  - `LinkedFlowResult`
  - `run_linked_flow(...)`

## Step 1: Add failing two-hop and failure tests

- [ ] Append tests covering these exact behaviors to `tests/test_linked_flow.py`:

```python
from src.linked_flow import run_linked_flow
from src.linked_memory import LinkedMemoryGraph


def branching_graph() -> LinkedMemoryGraph:
    works_at = RelationIntent(
        name="works-at",
        phrase_embeddings=(
            normalized(0.7, 0.7),
        ),
    )

    headquarters = RelationIntent(
        name="headquartered-in",
        phrase_embeddings=(
            normalized(1.0, 0.0),
        ),
    )

    founded_by = RelationIntent(
        name="founded-by",
        phrase_embeddings=(
            normalized(0.0, 1.0),
        ),
    )

    root = LinkedMemory(
        memory_id="bob-works-at-globex",
        source="Bob",
        relation=works_at,
        target="Globex",
        trigger_centers=(
            normalized(1.0, 0.0),
        ),
        target_vector=normalized(0.0, 1.0),
        successor_ids=(
            "globex-headquartered-in-paris",
            "globex-founded-by-susan",
        ),
        radius=2.0,
    )

    headquarters_memory = LinkedMemory(
        memory_id="globex-headquartered-in-paris",
        source="Globex",
        relation=headquarters,
        target="Paris",
        trigger_centers=(
            normalized(0.0, 1.0),
        ),
        target_vector=normalized(-1.0, 0.0),
        radius=2.0,
    )

    founder_memory = LinkedMemory(
        memory_id="globex-founded-by-susan",
        source="Globex",
        relation=founded_by,
        target="Susan",
        trigger_centers=(
            normalized(0.0, 1.0),
        ),
        target_vector=normalized(1.0, 0.0),
        radius=2.0,
    )

    return LinkedMemoryGraph(
        memories=(
            root,
            headquarters_memory,
            founder_memory,
        ),
        root_ids=(root.memory_id,),
    )


def test_linked_flow_selects_headquarters_successor():
    result = run_linked_flow(
        query=normalized(1.0, 0.0),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert result.status == RoutingStatus.COMPLETED
    assert result.traversed_memory_ids == (
        "bob-works-at-globex",
        "globex-headquartered-in-paris",
    )


def test_linked_flow_selects_founder_successor():
    result = run_linked_flow(
        query=normalized(0.0, 1.0),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
    )

    assert result.status == RoutingStatus.COMPLETED
    assert result.traversed_memory_ids == (
        "bob-works-at-globex",
        "globex-founded-by-susan",
    )


def test_linked_flow_reports_no_successors_when_more_hops_required():
    graph = branching_graph()

    result = run_linked_flow(
        query=normalized(1.0, 0.0),
        graph=graph,
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=3,
    )

    assert result.status == RoutingStatus.NO_SUCCESSORS


def test_linked_flow_respects_successor_threshold():
    result = run_linked_flow(
        query=normalized(0.7, 0.7),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=2,
        successor_minimum_score=0.95,
    )

    assert (
        result.status
        == RoutingStatus.NO_MATCHING_SUCCESSOR
    )


def test_linked_flow_reports_maximum_hops_reached():
    result = run_linked_flow(
        query=normalized(1.0, 0.0),
        graph=branching_graph(),
        handoff_threshold=0.90,
        max_steps_per_hop=100,
        step_size=0.1,
        max_hops=1,
    )

    assert (
        result.status
        == RoutingStatus.MAXIMUM_HOPS_REACHED
    )
```

## Step 2: Verify RED

- [ ] Run:

```powershell
python -m pytest tests\test_linked_flow.py -v
```

## Step 3: Implement route result records and traversal

- [ ] Add imports in `src/linked_flow.py`:

```python
from src.linked_memory import (
    LinkedMemoryGraph,
    MemoryScore,
    score_successors,
    select_root_memory,
)
```

- [ ] Add:

```python
@dataclass(frozen=True)
class HopRecord:
    memory_id: str
    relation_name: str
    source: str
    target: str
    selection_score: float
    successor_scores: tuple[
        tuple[str, float],
        ...
    ]
    steps_used: int


@dataclass(frozen=True)
class LinkedFlowResult:
    status: RoutingStatus
    final_state: np.ndarray
    traversed_memory_ids: tuple[str, ...]
    traversed_relation_names: tuple[str, ...]
    hop_records: tuple[HopRecord, ...]
    total_steps: int
    root_selection_score: float | None
    failure_reason: str | None
```

- [ ] Implement `run_linked_flow(...)` with this exact contract:

```python
def run_linked_flow(
    query: np.ndarray,
    graph: LinkedMemoryGraph,
    *,
    handoff_threshold: float = 0.95,
    max_steps_per_hop: int = 50,
    step_size: float = 0.1,
    max_hops: int = 4,
    root_minimum_score: float | None = None,
    successor_minimum_score: float | None = None,
) -> LinkedFlowResult:
```

Implementation rules:

1. Normalize a copy of `query` as both the original query and initial state.
2. Return `NO_ROOT_MEMORY` when the graph has no roots.
3. Rank/select the root using `select_root_memory`.
4. Distinguish no roots from score below threshold.
5. For every hop:
   - reject a revisited memory ID with `CYCLE_DETECTED`
   - call `flow_linked_edge`
   - stop immediately on `TARGET_NOT_REACHED`
   - record the hop
6. After a completed edge:
   - if `hop_count == max_hops`, return `MAXIMUM_HOPS_REACHED` when successors exist, otherwise `COMPLETED`
   - if there are no successors, return `COMPLETED` for a terminal answer edge
   - score only direct successors
   - filter already-traversed successor IDs
   - if all successors are traversed, return `CYCLE_DETECTED`
   - if the best remaining score is below threshold, return `NO_MATCHING_SUCCESSOR`
   - activate the best successor
7. `total_steps` is the sum of edge steps.
8. `failure_reason` is `None` only for `COMPLETED`.

Important correction to the provisional tests: a terminal edge with no successors is a successful `COMPLETED` route. `NO_SUCCESSORS` is reserved for configurations that explicitly require continuation. To avoid ambiguity, do not expose a `required_hops` option yet. Remove `test_linked_flow_reports_no_successors_when_more_hops_required`; keep terminal edges successful.

## Step 4: Verify GREEN

- [ ] Run:

```powershell
python -m pytest tests\test_linked_flow.py -v
python -m pytest -v
```

## Step 5: Commit

- [ ] Run:

```powershell
git add src\linked_flow.py tests\test_linked_flow.py
git commit -m "feat: add query-conditioned linked traversal"
git push
```

---

# Task 5: Real-encoder branching experiment

**Files:**
- Create: `experiments/linked_flow_branching.py`

**Interfaces:**
- Uses the linked-memory and linked-flow APIs.
- Produces an inspectable report for six routes:
  - Alice headquarters, founder, operating region
  - Bob headquarters, founder, operating region

## Step 1: Create the file

- [ ] Run:

```powershell
New-Item -ItemType File -Force `
    experiments\linked_flow_branching.py
```

## Step 2: Build embedding helpers

- [ ] In `experiments/linked_flow_branching.py`, define:

```python
def embed_phrases(
    encoder: TextEncoder,
    phrases: list[str],
) -> tuple[np.ndarray, ...]:
    return tuple(
        encoder.encode(phrase)
        for phrase in phrases
    )


def relation_intent(
    encoder: TextEncoder,
    name: str,
    phrases: list[str],
) -> RelationIntent:
    return RelationIntent(
        name=name,
        phrase_embeddings=embed_phrases(
            encoder,
            phrases,
        ),
    )


def linked_memory(
    *,
    encoder: TextEncoder,
    memory_id: str,
    source: str,
    relation: RelationIntent,
    target: str,
    trigger_phrases: list[str],
    successor_ids: tuple[str, ...] = (),
    radius: float = 0.35,
) -> LinkedMemory:
    return LinkedMemory(
        memory_id=memory_id,
        source=source,
        relation=relation,
        target=target,
        trigger_centers=embed_phrases(
            encoder,
            trigger_phrases,
        ),
        target_vector=encoder.encode(target),
        successor_ids=successor_ids,
        radius=radius,
    )
```

## Step 3: Construct relation intents

Use these phrase sets:

```python
works_at_phrases = [
    "works at",
    "employer",
    "company someone works for",
    "workplace",
]

headquarters_phrases = [
    "headquartered in",
    "headquarters location",
    "where is it based",
    "main office city",
]

founder_phrases = [
    "founded by",
    "who founded it",
    "company founder",
    "original creator",
]

operates_phrases = [
    "operates in",
    "where does it operate",
    "operating region",
    "service area",
]
```

## Step 4: Construct the Alice and Bob graph

Use:

```text
Alice --works-at→ Acme
Acme --headquartered-in→ Helsinki
Acme --founded-by→ Maria
Acme --operates-in→ Finland

Bob --works-at→ Globex
Globex --headquartered-in→ Paris
Globex --founded-by→ Susan
Globex --operates-in→ Europe
```

Root trigger phrases should match the existing multi-trigger format:

```python
[
    "Alice's employer",
    "company Alice works for",
    "where Alice works",
    "Alice's workplace",
]
```

Every successor should use source-specific trigger phrases, even though query routing uses relation-intent phrases. These trigger centres define the local flow basin.

## Step 5: Run six queries

Use:

```python
queries = [
    (
        "Where is Alice's employer headquartered?",
        "Helsinki",
    ),
    (
        "Who founded Alice's employer?",
        "Maria",
    ),
    (
        "Where does Alice's employer operate?",
        "Finland",
    ),
    (
        "Where is Bob's employer headquartered?",
        "Paris",
    ),
    (
        "Who founded Bob's employer?",
        "Susan",
    ),
    (
        "Where does Bob's employer operate?",
        "Europe",
    ),
]
```

For each query:

1. Run `run_linked_flow`.
2. Rank fixed candidates:
   - Alice, Bob, Acme, Globex
   - Helsinki, Paris
   - Maria, Susan
   - Finland, Europe
3. Print:
   - query
   - status
   - selected root ID
   - route
   - relation names
   - selection scores
   - steps per hop
   - final winner
   - expected answer
   - PASS/FAIL

Use `max_hops=2`.

## Step 6: Run and inspect

- [ ] Run:

```powershell
python -m experiments.linked_flow_branching
```

Do not create regression tests from the observed thresholds yet. First inspect:

- whether all six root selections are correct
- whether relation-intent routing separates headquarters/founder/operates
- whether `handoff_threshold=0.95` is reachable
- whether the final vectors decode to their intended candidates

## Step 7: Commit the experiment

- [ ] Run:

```powershell
git add experiments\linked_flow_branching.py
git commit -m "test: evaluate branching linked flows"
git push
```

---

# Task 6: Branching regression tests with the real encoder

**Files:**
- Modify: `tests/test_linked_flow.py`

Use a module-scoped `TextEncoder` fixture, following the existing `tests/test_composition.py` pattern.

Add six regression tests only after the branching experiment succeeds. Each test should assert:

- `status == RoutingStatus.COMPLETED`
- exact two-memory route
- exact final winner

Do not assert exact floating-point scores or exact step counts. Those are too brittle across model/library versions.

Run:

```powershell
python -m pytest tests\test_linked_flow.py -v
python -m pytest -v
```

Commit:

```powershell
git add tests\test_linked_flow.py
git commit -m "test: lock in linked branching behavior"
git push
```

---

# Task 7: Fixed-pool linked-flow distractor scaling

**Files:**
- Create: `experiments/linked_flow_distractor_scaling.py`

**Architecture:**

- Reuse `fixed_distractor_pool()` from `src/scaling.py`.
- Build one linked graph for each load:
  - 0
  - 3
  - 8
  - 18
  - 48 distractors
- Every distractor branch contains:
  - person `works-at` company
  - company `headquartered-in` city
- Target Alice and Bob branches retain all three typed successors.
- The 72-query benchmark remains headquarters-only for direct comparison with the previous scaling result.
- Only root selection is global.
- Later flow uses only explicit successors.

## Required measurements per load

Record:

- correct final answers
- total queries
- final accuracy
- correct root selection count/rate
- correct successor relation count/rate
- complete exact-route count/rate
- average root score
- average root selection margin
- average successor score
- average successor selection margin
- average steps per hop
- average total steps
- average runtime per query
- failure counts grouped by `RoutingStatus`

## Root and successor margins

Add helper functions inside the experiment first; only move them into source code if reused later.

Root margin:

```text
best root score - second-best root score
```

Successor margin:

```text
selected successor score - second-best successor score
```

For a single eligible successor, record the margin as `None`, not infinity.

## Comparison table

Print the prior global summed-field accuracy alongside linked-flow accuracy:

```text
Distractors | Global accuracy | Linked accuracy | Root accuracy | Successor accuracy | Exact route
```

Hardcode the previously observed global values in the experiment report with a clear label:

```python
GLOBAL_BASELINE_ACCURACY = {
    0: 1.000,
    3: 1.000,
    8: 0.514,
    18: 0.000,
    48: 0.000,
}
```

Do not rerun the expensive global baseline inside the first linked scaling script.

## Expected diagnostic pattern

The linked hypothesis predicts:

- successor accuracy stays near 100% once the correct root is chosen
- root selection becomes the primary scaling bottleneck
- wrong distractor cities no longer accumulate during the second hop
- runtime still grows at root selection, but per-hop flow cost remains local

## Run

```powershell
python -m experiments.linked_flow_distractor_scaling `
    | Tee-Object linked-flow-scaling-results.txt
```

Then:

```powershell
python -m pytest -v
```

Commit:

```powershell
git add experiments\linked_flow_distractor_scaling.py
git commit -m "test: measure linked-flow distractor scaling"
git push
```

---

# Task 8: Final milestone evaluation

After the scaling experiment, classify the result:

## Outcome A: Linked flow succeeds at 48 distractors

Proceed to:

- indexed/ANN root lookup
- automatic linked-memory formation design
- confidence and provenance

## Outcome B: Successor routing succeeds but root selection fails

Proceed to:

- entity-aware root indexing
- sparse root retrieval
- query decomposition for entity and relation intent

This would still validate the linked architecture.

## Outcome C: Correct root but wrong typed successor

Proceed to:

- relation-intent phrase improvements
- query relation extraction
- learned router investigation

## Outcome D: Correct routing but latent edge flow fails

Proceed to:

- adaptive handoff thresholds
- target-centred edge dynamics
- manifold-validity diagnostics

## Final verification commands

```powershell
python -m pytest -v
python -m experiments.linked_flow_branching
python -m experiments.linked_flow_distractor_scaling
git status
```

The working tree should contain only deliberately saved result files, or be clean.
