# Reinforced Thought Routes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add learned, route-level reinforcement so validated successful multi-hop traversals become persistent `ThoughtRoute` objects that bias future routing for semantically equivalent queries without changing latent edge dynamics.

**Architecture:** Add a focused `src/thought_routes.py` module containing the route data model, store, reinforcement rules, context similarity, and bounded bonus calculation. Integrate those bonuses into root and successor selection inside `run_linked_flow()` through optional parameters, while leaving `flow_linked_edge()` unchanged. Keep validation outside the router: callers decide whether an outcome was correct, then explicitly record or reinforce the completed route.

**Tech Stack:** Python 3.12, NumPy, pytest, existing `TextEncoder`, `LinkedMemoryGraph`, `run_linked_flow`, and normalized sentence embeddings.

## Global Constraints

- A `ThoughtRoute` is created only after a multi-hop traversal has completed and the caller positively validates the outcome.
- Route identity is the exact ordered tuple of linked-memory IDs.
- Reinforcement applies to the whole route, not individual edges.
- Initial route strength is `0.10`.
- Validated successful reuse adds `0.01`.
- Explicit confirmation adds `0.15`.
- Maximum route strength is `1.00`.
- Reinforcement changes route selection only; it must not change latent edge-flow dynamics.
- The route-selection bonus is additive and bounded: `semantic_score + lambda * context_similarity * route_strength`.
- `lambda` is configurable and deliberately small in the first experiment.
- Later successful paraphrases add their query embeddings to the same route context set.
- Negative valence, decay, clarification-driven creation, schema consolidation, and LLM integration remain out of scope.
- Existing linked-flow behavior must remain unchanged when no route store is supplied.

---

## File Structure

- Create `src/thought_routes.py`
  - Defines reinforcement constants/configuration.
  - Defines `ThoughtRoute`.
  - Defines `ThoughtRouteStore`.
  - Handles create/reuse/confirmation updates.
  - Calculates context similarity and bounded route bonuses.
  - Finds compatible learned routes for a candidate root or successor prefix.

- Create `tests/test_thought_routes.py`
  - Unit tests for route identity, creation, reinforcement, caps, context accumulation, and bonus calculation.

- Modify `src/linked_flow.py`
  - Accepts an optional route store and bonus scale/config.
  - Adds route bonuses during root selection and successor selection.
  - Does not modify `flow_linked_edge()` or `linked_memory_influence()`.

- Modify `tests/test_linked_flow.py`
  - Integration tests proving zero-learning compatibility and learned route selection effects.

- Create `experiments/reinforced_thought_routes.py`
  - Runs the Milestone 1.7 before/after experiment.
  - Trains one Bob headquarters route.
  - Evaluates same-route paraphrases, wrong-intent Bob queries, and same-intent Alice queries.
  - Prints semantic scores, route bonuses, margins, exact-route accuracy, and final-answer accuracy.

---

### Task 1: Route Data Model and Store

**Files:**
- Create: `src/thought_routes.py`
- Create: `tests/test_thought_routes.py`

**Interfaces:**
- Produces:
  - `ReinforcementConfig`
  - `ThoughtRoute`
  - `ThoughtRouteStore`
  - `ThoughtRouteStore.get(memory_ids)`
  - `ThoughtRouteStore.record_validated_success(memory_ids, context_embedding)`
  - `ThoughtRouteStore.record_explicit_confirmation(memory_ids)`
  - `ThoughtRouteStore.routes()`

- [ ] **Step 1: Write failing tests for route creation and identity**

Create `tests/test_thought_routes.py` with:

```python
import numpy as np

from src.thought_routes import (
    ReinforcementConfig,
    ThoughtRouteStore,
)


def unit_vector(index: int, size: int = 4) -> np.ndarray:
    vector = np.zeros(size, dtype=np.float32)
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
```

- [ ] **Step 2: Run tests and verify failure**

Run:

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: collection/import failure because `src.thought_routes` does not exist.

- [ ] **Step 3: Implement minimal route model and store**

Create `src/thought_routes.py`:

```python
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
        self.config = config or ReinforcementConfig()
        self._routes: dict[
            tuple[str, ...],
            ThoughtRoute,
        ] = {}

    def routes(self) -> tuple[ThoughtRoute, ...]:
        return tuple(self._routes.values())

    def get(
        self,
        memory_ids: tuple[str, ...],
    ) -> ThoughtRoute | None:
        return self._routes.get(tuple(memory_ids))

    def record_validated_success(
        self,
        *,
        memory_ids: tuple[str, ...],
        context_embedding: np.ndarray,
    ) -> ThoughtRoute:
        key = tuple(memory_ids)

        if len(key) < 2:
            raise ValueError(
                "Thought routes must contain at least two memories."
            )

        context = np.array(
            context_embedding,
            dtype=np.float32,
            copy=True,
        )

        existing = self._routes.get(key)

        if existing is None:
            route = ThoughtRoute(
                route_id=" -> ".join(key),
                memory_ids=key,
                context_embeddings=(context,),
                strength=self.config.initial_strength,
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
                    self.config.maximum_strength,
                    existing.strength
                    + self.config.successful_reuse_increment,
                ),
                successful_uses=(
                    existing.successful_uses + 1
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
        existing = self._routes.get(key)

        if existing is None:
            raise KeyError(
                "Cannot confirm a thought route that has not been learned."
            )

        route = replace(
            existing,
            strength=min(
                self.config.maximum_strength,
                existing.strength
                + self.config.explicit_confirmation_increment,
            ),
            explicit_confirmations=(
                existing.explicit_confirmations + 1
            ),
        )

        self._routes[key] = route
        return route
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: all Task 1 tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/thought_routes.py tests/test_thought_routes.py
git commit -m "feat: add learned thought route store"
```

---

### Task 2: Reinforcement Rules, Context Deduplication, and Bonus Calculation

**Files:**
- Modify: `src/thought_routes.py`
- Modify: `tests/test_thought_routes.py`

**Interfaces:**
- Produces:
  - `context_similarity(route, query)`
  - `route_bonus(route, query, bonus_scale)`
  - `ThoughtRouteStore.compatible_routes(prefix)`
  - duplicate-context suppression using cosine-equivalent embeddings

- [ ] **Step 1: Add failing reinforcement and bonus tests**

Append to `tests/test_thought_routes.py`:

```python
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
    store = ThoughtRouteStore(config=config)

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

    assert bonus == 0.01


def test_compatible_routes_require_matching_prefix() -> None:
    store = ThoughtRouteStore()

    store.record_validated_success(
        memory_ids=("root-a", "next-a"),
        context_embedding=unit_vector(0),
    )
    store.record_validated_success(
        memory_ids=("root-b", "next-b"),
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
```

- [ ] **Step 2: Run the new tests and verify failure**

Run:

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: failures for missing `context_similarity`, `route_bonus`, `compatible_routes`, and duplicate-context handling.

- [ ] **Step 3: Implement context and bonus helpers**

Update `src/thought_routes.py` with:

```python
def context_similarity(
    *,
    route: ThoughtRoute,
    query: np.ndarray,
) -> float:
    if not route.context_embeddings:
        return 0.0

    return max(
        float(np.dot(query, context))
        for context in route.context_embeddings
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
```

Add to `ThoughtRouteStore`:

```python
    def compatible_routes(
        self,
        *,
        prefix: tuple[str, ...],
    ) -> tuple[ThoughtRoute, ...]:
        wanted = tuple(prefix)

        return tuple(
            route
            for route in self._routes.values()
            if route.memory_ids[:len(wanted)]
            == wanted
        )
```

Replace the unconditional context append in `record_validated_success()` with:

```python
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

            contexts = (
                existing.context_embeddings
                if already_known
                else (
                    existing.context_embeddings
                    + (context,)
                )
            )

            route = replace(
                existing,
                context_embeddings=contexts,
                strength=min(
                    self.config.maximum_strength,
                    existing.strength
                    + self.config.successful_reuse_increment,
                ),
                successful_uses=(
                    existing.successful_uses + 1
                ),
            )
```

- [ ] **Step 4: Run the unit tests**

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: all thought-route unit tests pass.

- [ ] **Step 5: Commit**

```powershell
git add src/thought_routes.py tests/test_thought_routes.py
git commit -m "feat: score and reinforce thought routes"
```

---

### Task 3: Add Optional Route Bonuses to Linked Routing

**Files:**
- Modify: `src/linked_flow.py`
- Modify: `tests/test_linked_flow.py`

**Interfaces:**
- Consumes:
  - `ThoughtRouteStore`
  - `route_bonus(...)`
- Modifies:
  - `run_linked_flow(..., thought_routes=None, route_bonus_scale=None)`
- Produces:
  - root selection using semantic score plus relevant route bonus,
  - successor selection using semantic score plus relevant route bonus.
- Preserves:
  - `flow_linked_edge()` unchanged,
  - behavior identical to current routing when `thought_routes is None`.

- [ ] **Step 1: Add a regression test proving no-store behavior is unchanged**

Append to `tests/test_linked_flow.py` a test using the same graph fixture already used by the file:

```python
def test_no_thought_route_store_preserves_existing_route(
    branching_graph,
    bob_headquarters_query,
) -> None:
    result = run_linked_flow(
        query=bob_headquarters_query,
        graph=branching_graph,
        handoff_threshold=0.95,
        max_steps_per_hop=50,
        step_size=0.1,
        max_hops=2,
    )

    assert result.status == RoutingStatus.COMPLETED
    assert result.traversed_memory_ids == (
        "bob-works-at-globex",
        "globex-headquartered-in-paris",
    )
```

If the existing test file does not expose those fixture names, use its existing graph/query construction helpers rather than introducing a second graph shape.

- [ ] **Step 2: Add a failing test where reinforcement breaks a close semantic tie**

Construct a tiny deterministic graph with normalized vectors so two roots are semantically close but one has a learned route bonus:

```python
def test_reinforced_route_can_win_close_root_selection() -> None:
    # Query favors root-b by only 0.005 semantically.
    query = normalized([1.0, 0.0])

    # root-a semantic score = 0.990
    # root-b semantic score = 0.995
    # Learned route-a context similarity = 1.0
    # route strength = 0.10
    # bonus scale = 0.10
    # route-a gets +0.010 and should win.
    ...
```

The assertion must verify the first traversed memory is the reinforced root.

- [ ] **Step 3: Run the focused tests and verify the reinforcement test fails**

```powershell
python -m pytest tests/test_linked_flow.py -v
```

Expected: existing tests pass; the new reinforcement-selection test fails because `run_linked_flow()` does not yet accept/use a route store.

- [ ] **Step 4: Add helper functions inside `src/linked_flow.py` for adjusted scores**

Import:

```python
from src.thought_routes import (
    ThoughtRouteStore,
    route_bonus,
)
```

Add an internal helper:

```python
def _route_adjusted_score(
    *,
    semantic_score: float,
    candidate_memory_id: str,
    traversed_prefix: tuple[str, ...],
    query: np.ndarray,
    thought_routes: ThoughtRouteStore | None,
    bonus_scale: float,
) -> float:
    if thought_routes is None:
        return semantic_score

    candidate_prefix = (
        traversed_prefix
        + (candidate_memory_id,)
    )

    compatible = (
        thought_routes.compatible_routes(
            prefix=candidate_prefix,
        )
    )

    if not compatible:
        return semantic_score

    bonus = max(
        route_bonus(
            route=route,
            query=query,
            bonus_scale=bonus_scale,
        )
        for route in compatible
    )

    return semantic_score + bonus
```

- [ ] **Step 5: Extend `run_linked_flow()` signature**

Change:

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
    thought_routes: ThoughtRouteStore | None = None,
    route_bonus_scale: float | None = None,
) -> LinkedFlowResult:
```

Resolve scale once near the beginning:

```python
    effective_bonus_scale = (
        route_bonus_scale
        if route_bonus_scale is not None
        else (
            thought_routes.config.bonus_scale
            if thought_routes is not None
            else 0.0
        )
    )
```

- [ ] **Step 6: Apply adjusted scoring to root selection**

Instead of relying only on `select_root_memory()`, obtain the ranked roots with the existing trigger-ranking helper, convert each semantic score to an adjusted score with `traversed_prefix=()`, then select the maximum adjusted score.

Preserve the existing `MemoryScore` type so the rest of `run_linked_flow()` does not need to change shape.

The selected root's `MemoryScore.score` should be the adjusted score because hop records and thresholds should reflect the actual selection score.

- [ ] **Step 7: Apply adjusted scoring to successor selection**

After `score_successors(...)` returns semantic successor scores, rebuild the list so each score becomes:

```python
adjusted = _route_adjusted_score(
    semantic_score=score.score,
    candidate_memory_id=score.memory.memory_id,
    traversed_prefix=tuple(traversed_ids),
    query=original_query,
    thought_routes=thought_routes,
    bonus_scale=effective_bonus_scale,
)
```

Sort descending by adjusted score before choosing `eligible_scores[0]`.

- [ ] **Step 8: Do not touch latent edge dynamics**

Verify there are no changes to:

```python
def linked_memory_influence(...):
    ...

def flow_linked_edge(...):
    ...
```

- [ ] **Step 9: Run linked-flow tests**

```powershell
python -m pytest tests/test_linked_flow.py -v
```

Expected: all linked-flow tests pass, including the reinforced close-tie case.

- [ ] **Step 10: Run the entire suite**

```powershell
python -m pytest -v
```

Expected: all existing tests pass.

- [ ] **Step 11: Commit**

```powershell
git add src/linked_flow.py tests/test_linked_flow.py
git commit -m "feat: bias linked routing with thought routes"
```

---

### Task 4: Record Only Positively Validated Completed Routes

**Files:**
- Modify: `src/thought_routes.py`
- Modify: `tests/test_thought_routes.py`

**Interfaces:**
- Consumes:
  - `LinkedFlowResult`
  - `RoutingStatus`
- Produces:
  - `ThoughtRouteStore.record_validated_result(result, context_embedding, outcome_valid)`
- Behavior:
  - creates/reinforces only when `result.status == COMPLETED`,
  - requires `outcome_valid is True`,
  - requires at least two traversed memories,
  - otherwise returns `None` and does not mutate the store.

- [ ] **Step 1: Add failing validation-gate tests**

Append:

```python
from src.linked_flow import (
    LinkedFlowResult,
    RoutingStatus,
)


def make_result(
    *,
    status: RoutingStatus,
    memory_ids: tuple[str, ...],
) -> LinkedFlowResult:
    return LinkedFlowResult(
        status=status,
        final_state=unit_vector(0),
        traversed_memory_ids=memory_ids,
        traversed_relation_names=(),
        hop_records=(),
        total_steps=0,
        root_selection_score=None,
        failure_reason=None,
    )


def test_incorrect_outcome_does_not_create_route() -> None:
    store = ThoughtRouteStore()

    learned = store.record_validated_result(
        result=make_result(
            status=RoutingStatus.COMPLETED,
            memory_ids=("a", "b"),
        ),
        context_embedding=unit_vector(0),
        outcome_valid=False,
    )

    assert learned is None
    assert store.routes() == ()


def test_incomplete_route_does_not_create_route() -> None:
    store = ThoughtRouteStore()

    learned = store.record_validated_result(
        result=make_result(
            status=RoutingStatus.TARGET_NOT_REACHED,
            memory_ids=("a",),
        ),
        context_embedding=unit_vector(0),
        outcome_valid=True,
    )

    assert learned is None
    assert store.routes() == ()


def test_valid_completed_result_creates_route() -> None:
    store = ThoughtRouteStore()

    learned = store.record_validated_result(
        result=make_result(
            status=RoutingStatus.COMPLETED,
            memory_ids=("a", "b"),
        ),
        context_embedding=unit_vector(0),
        outcome_valid=True,
    )

    assert learned is not None
    assert learned.memory_ids == ("a", "b")
```

- [ ] **Step 2: Run tests and verify failure**

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: failure because `record_validated_result` does not exist.

- [ ] **Step 3: Implement the validation gate without introducing a circular import**

Do **not** import `src.linked_flow` at module import time inside `src/thought_routes.py`.

Use a structural protocol:

```python
from typing import Protocol


class CompletedRouteResult(Protocol):
    status: object
    traversed_memory_ids: tuple[str, ...]
```

Implement:

```python
    def record_validated_result(
        self,
        *,
        result: CompletedRouteResult,
        context_embedding: np.ndarray,
        outcome_valid: bool,
    ) -> ThoughtRoute | None:
        status_value = getattr(
            result.status,
            "value",
            result.status,
        )

        if status_value != "completed":
            return None

        if not outcome_valid:
            return None

        if len(result.traversed_memory_ids) < 2:
            return None

        return self.record_validated_success(
            memory_ids=result.traversed_memory_ids,
            context_embedding=context_embedding,
        )
```

- [ ] **Step 4: Run unit tests**

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: all pass.

- [ ] **Step 5: Run entire suite**

```powershell
python -m pytest -v
```

Expected: all pass.

- [ ] **Step 6: Commit**

```powershell
git add src/thought_routes.py tests/test_thought_routes.py
git commit -m "feat: learn routes only from validated outcomes"
```

---

### Task 5: Build the Reinforced Thought Route Experiment

**Files:**
- Create: `experiments/reinforced_thought_routes.py`

**Interfaces:**
- Reuses:
  - `TextEncoder`
  - `build_graph()` from `experiments.linked_flow_branching`
  - `build_candidates()` from `experiments.linked_flow_branching`
  - `rank_candidates`
  - `run_linked_flow`
  - `ThoughtRouteStore`
- Produces:
  - deterministic before/after benchmark output for three query groups.

- [ ] **Step 1: Define experiment query groups**

Create `experiments/reinforced_thought_routes.py` with:

```python
from dataclasses import dataclass

import numpy as np

from experiments.linked_flow_branching import (
    build_candidates,
    build_graph,
)
from src.encoder import TextEncoder
from src.evaluation import rank_candidates
from src.linked_flow import (
    RoutingStatus,
    run_linked_flow,
)
from src.thought_routes import (
    ThoughtRouteStore,
)


@dataclass(frozen=True)
class ExperimentCase:
    text: str
    expected: str
    expected_route: tuple[str, ...]
    group: str


BOB_HQ_ROUTE = (
    "bob-works-at-globex",
    "globex-headquartered-in-paris",
)


CASES = (
    ExperimentCase(
        text="Where is the company Bob works for based?",
        expected="Paris",
        expected_route=BOB_HQ_ROUTE,
        group="same-route-paraphrase",
    ),
    ExperimentCase(
        text="What city is Bob's workplace headquartered in?",
        expected="Paris",
        expected_route=BOB_HQ_ROUTE,
        group="same-route-paraphrase",
    ),
    ExperimentCase(
        text="Where does Bob's employer operate?",
        expected="Europe",
        expected_route=(
            "bob-works-at-globex",
            "globex-operates-in-europe",
        ),
        group="different-intent",
    ),
    ExperimentCase(
        text="Who founded Bob's employer?",
        expected="Susan",
        expected_route=(
            "bob-works-at-globex",
            "globex-founded-by-susan",
        ),
        group="different-intent",
    ),
    ExperimentCase(
        text="Where is Alice's employer headquartered?",
        expected="Helsinki",
        expected_route=(
            "alice-works-at-acme",
            "acme-headquartered-in-helsinki",
        ),
        group="different-entity",
    ),
)
```

- [ ] **Step 2: Add a reusable evaluator**

Implement:

```python
def evaluate_case(
    *,
    encoder: TextEncoder,
    graph,
    candidates: dict[str, np.ndarray],
    case: ExperimentCase,
    store: ThoughtRouteStore | None,
):
    query = encoder.encode(case.text)

    result = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.95,
        max_steps_per_hop=50,
        step_size=0.1,
        max_hops=2,
        thought_routes=store,
    )

    ranking = rank_candidates(
        state=result.final_state,
        candidates=candidates,
    )

    winner = ranking[0][0]

    return {
        "query": query,
        "result": result,
        "winner": winner,
        "route_correct": (
            result.traversed_memory_ids
            == case.expected_route
        ),
        "answer_correct": (
            result.status == RoutingStatus.COMPLETED
            and winner == case.expected
        ),
    }
```

- [ ] **Step 3: Learn the seed route from one validated experience**

In `main()`:

```python
encoder = TextEncoder()
graph = build_graph(encoder=encoder)
candidates = build_candidates(encoder=encoder)
store = ThoughtRouteStore()

seed_text = "Where is Bob's employer headquartered?"
seed_query = encoder.encode(seed_text)

seed_result = run_linked_flow(
    query=seed_query,
    graph=graph,
    handoff_threshold=0.95,
    max_steps_per_hop=50,
    step_size=0.1,
    max_hops=2,
)

seed_ranking = rank_candidates(
    state=seed_result.final_state,
    candidates=candidates,
)

seed_valid = (
    seed_result.status == RoutingStatus.COMPLETED
    and seed_ranking[0][0] == "Paris"
)

learned = store.record_validated_result(
    result=seed_result,
    context_embedding=seed_query,
    outcome_valid=seed_valid,
)

assert learned is not None
assert learned.memory_ids == BOB_HQ_ROUTE
```

- [ ] **Step 4: Reinforce the learned route deliberately**

Apply several successful-reuse updates and one explicit confirmation so the before/after effect is measurable:

```python
for text in (
    "Where is Bob's employer headquartered?",
    "Where is the company Bob works for based?",
    "What city is Bob's workplace headquartered in?",
):
    query = encoder.encode(text)

    result = run_linked_flow(
        query=query,
        graph=graph,
        handoff_threshold=0.95,
        max_steps_per_hop=50,
        step_size=0.1,
        max_hops=2,
        thought_routes=store,
    )

    ranking = rank_candidates(
        state=result.final_state,
        candidates=candidates,
    )

    store.record_validated_result(
        result=result,
        context_embedding=query,
        outcome_valid=(
            result.status == RoutingStatus.COMPLETED
            and ranking[0][0] == "Paris"
        ),
    )

store.record_explicit_confirmation(
    memory_ids=BOB_HQ_ROUTE,
)
```

- [ ] **Step 5: Capture before and after results**

Evaluate `CASES` once with `store=None` and once with the reinforced `store`.

For each case print:

```text
group
query
before route
after route
before root score
after root score
before answer
after answer
```

Also print the learned route strength and number of stored context embeddings.

- [ ] **Step 6: Add group summaries**

For each group print:

```text
same-route-paraphrase:
    before exact-route accuracy
    after exact-route accuracy

different-intent:
    before exact-route accuracy
    after exact-route accuracy

different-entity:
    before exact-route accuracy
    after exact-route accuracy
```

Also report whether any protected group regressed.

- [ ] **Step 7: Run the experiment**

```powershell
python -m experiments.reinforced_thought_routes
```

Expected:
- seed route is learned only after validation,
- route strength is greater than `0.10` after reuse/confirmation,
- same-route paraphrases receive a measurable route-selection benefit,
- different-intent and different-entity cases remain on their own routes,
- all final answers remain correct.

- [ ] **Step 8: Save experiment output**

```powershell
python -m experiments.reinforced_thought_routes `
    | Tee-Object reinforced-thought-routes-results.txt
```

The existing `*-results.txt` ignore rule should keep the output untracked.

- [ ] **Step 9: Commit**

```powershell
git add experiments/reinforced_thought_routes.py
git commit -m "test: measure reinforced thought route transfer"
```

---

### Task 6: Regression and Milestone Verification

**Files:**
- No new production files required.
- Modify tests only if the experiment exposes a real bug.

**Interfaces:**
- Verifies the complete Milestone 1.7 contract.

- [ ] **Step 1: Run thought-route unit tests**

```powershell
python -m pytest tests/test_thought_routes.py -v
```

Expected: PASS.

- [ ] **Step 2: Run linked-flow integration tests**

```powershell
python -m pytest tests/test_linked_flow.py -v
```

Expected: PASS.

- [ ] **Step 3: Run the full suite**

```powershell
python -m pytest -v
```

Expected: PASS.

- [ ] **Step 4: Re-run the previous linked-flow branching experiment**

```powershell
python -m experiments.linked_flow_branching
```

Expected: the existing six branching cases still pass.

- [ ] **Step 5: Re-run distractor scaling**

```powershell
python -m experiments.linked_flow_distractor_scaling
```

Expected: no regression from the previous 72/72 results when no thought-route store is supplied.

- [ ] **Step 6: Run reinforced-thought-route experiment**

```powershell
python -m experiments.reinforced_thought_routes
```

Expected:
- reinforced Bob headquarters route retains correct behavior,
- same-route paraphrases gain selection support,
- Bob wrong-intent routes remain distinct,
- Alice headquarters route remains distinct,
- no latent-flow regression appears.

- [ ] **Step 7: Inspect Git status**

```powershell
git status --short
```

Expected: only intended tracked source/test changes; generated `*-results.txt` files do not appear.

- [ ] **Step 8: Final milestone commit if verification produced any tracked fixes**

If no fixes were needed, skip this commit.

If fixes were needed:

```powershell
git add src tests experiments
git commit -m "test: verify reinforced thought routes"
```

- [ ] **Step 9: Push branch**

```powershell
git push
```
