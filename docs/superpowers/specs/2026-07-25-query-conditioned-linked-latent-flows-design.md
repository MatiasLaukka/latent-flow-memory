# Query-Conditioned Linked Latent Flows Design

## Goal

Replace the flat global memory field with a sparse linked-memory architecture in which memories can activate only semantically relevant successor memories.

The milestone tests whether query-conditioned linked flows preserve correct two-hop composition under the full fixed pool of 48 same-domain distractor chains.

## Research hypothesis

A memory should represent both:

- a local latent flow from one concept toward another
- links to typed successor memories that may continue the trajectory

Instead of summing every memory at every step, the system performs:

1. one global semantic entry decision
2. local traversal through linked successor memories

This should prevent many weak unrelated memories from accumulating into global distractor attractors.

## Milestone

**Milestone 1.6 — Query-Conditioned Linked Latent Flows**

Primary success criterion:

- Alice and Bob queries select the correct first-hop memory
- queries select different typed successors from the same intermediate company
- all tested routes remain correct with the full 48-chain distractor pool active

## Memory model

Each linked transition memory represents one typed relation:

- memory ID
- source concept
- relation type
- target concept
- trigger phrases
- latent trigger centres
- latent target vector
- successor memory IDs

Conceptual example:

```text
memory_id: bob-works-at-globex
source: Bob
relation: works-at
target: Globex

trigger phrases:
- Bob's employer
- where Bob works
- company Bob works for
- Bob's workplace

successors:
- globex-headquartered-in-paris
- globex-founded-by-susan
- globex-operates-in-europe
```

The graph links constrain eligible future flows. They do not directly supply the answer.

## Runtime state

The linked-flow runtime carries:

```text
x_t
```

The current normalized latent state.

```text
q
```

The original query embedding, retained throughout routing.

```text
active_memory
```

The single transition memory currently controlling the flow.

```text
traversed_memory_ids
```

The transitions already taken, used for cycle prevention and reporting.

```text
hop_count
```

The number of completed relation transitions.

## Initial memory selection

The first transition is selected globally from root memories.

Each root memory has several natural-language trigger phrases. The query embedding is compared with all trigger-centre embeddings.

For memory `m`:

```text
entry_score(m, q) =
    max cosine_similarity(q, trigger_center)
```

The root memory with the highest entry score becomes active.

This is the only stage that searches globally in the first prototype.

After initial selection, unrelated root memories and unrelated branches no longer contribute to the field.

## Latent movement within one edge

A selected memory moves the current latent state toward its target using the existing target-seeking field:

```text
V_m(x) = strength_m * w_m(x) * (target_m - x)
```

where `w_m(x)` is the local Gaussian activation.

The state update remains:

```text
x_(t+1) = normalize(x_t + step_size * V_m(x_t))
```

Only the active memory contributes during this edge traversal.

## Edge handoff

A transition is considered complete when the current state reaches a target-similarity threshold:

```text
cosine_similarity(x_t, target_m) >= handoff_threshold
```

The first prototype uses a fixed threshold shared by all memories.

If the threshold is not reached within a fixed maximum number of steps per hop, routing fails with a target-not-reached result.

At a successful handoff:

1. add the current memory ID to the traversed set
2. inspect only the current memory's successor IDs
3. choose the best successor relation using query-intent matching
4. activate the chosen successor
5. continue latent flow from the current state

## Query-intent relation routing

Each relation type stores several intent phrases.

Examples:

### `headquartered-in`

- headquartered in
- headquarters location
- where is it based
- main office city

### `founded-by`

- founded by
- who founded it
- company founder
- original creator

### `operates-in`

- operates in
- where does it operate
- operating region
- service area

### `works-at`

- works at
- employer
- company someone works for
- workplace

Every intent phrase is embedded using the existing frozen sentence encoder.

For successor memory `s`:

```text
relation_score(s, q) =
    max cosine_similarity(q, intent_phrase_embedding)
```

The eligible successor with the highest relation score becomes active.

The full original query remains available at every hop.

## Successor threshold

The first prototype should support a minimum relation-score threshold.

If no successor exceeds the threshold, routing returns a no-matching-successor failure rather than taking an arbitrary edge.

The initial experiment should report raw successor scores so the threshold can be selected empirically rather than tuned invisibly.

## Multiple typed successors

The target graph must contain genuine branching.

### Bob branch

```text
Bob --works-at→ Globex

Globex --headquartered-in→ Paris
Globex --founded-by→ Susan
Globex --operates-in→ Europe
```

### Alice branch

```text
Alice --works-at→ Acme

Acme --headquartered-in→ Helsinki
Acme --founded-by→ Maria
Acme --operates-in→ Finland
```

Queries must route differently after the same first hop:

```text
Where is Bob's employer headquartered?
Who founded Bob's employer?
Where does Bob's employer operate?
```

Expected routes:

```text
Bob → Globex → Paris
Bob → Globex → Susan
Bob → Globex → Europe
```

Equivalent Alice queries must route through Acme to Helsinki, Maria, or Finland.

## Maximum hops and cycle prevention

The first prototype uses:

```text
maximum hops: 4
```

A successor memory whose ID already appears in `traversed_memory_ids` must not be selected.

Routing stops with a cycle-detected result if all eligible successors would revisit traversed memories.

The hop limit prevents malformed graphs from creating infinite traversal.

## Result model

Each routed query should report:

- query text
- final state
- final decoded candidate
- selected root memory ID
- traversed memory IDs
- traversed relation types
- root selection score
- successor relation scores at every handoff
- steps used per hop
- total steps
- success or failure status
- failure reason when applicable

## Failure statuses

The implementation must distinguish:

- `no-root-memory`
- `root-score-below-threshold`
- `target-not-reached`
- `no-successors`
- `no-matching-successor`
- `cycle-detected`
- `maximum-hops-reached`
- `completed`

This makes routing failures diagnosable rather than presenting every failure as a wrong final answer.

## Baselines

The linked-flow benchmark should compare:

1. existing global summed field
2. global top-k field
3. linked flow with full-query matching against one relation phrase
4. linked flow with multi-phrase query-intent matching

The main linked-flow implementation is option 4.

Top-k remains a useful baseline but is not the main architectural direction.

## Distractor scaling evaluation

Reuse the fixed ordered pool of 48 realistic same-domain distractor chains.

The linked graph contains the target Alice and Bob branches plus all distractor branches.

Only the initial root-memory selection is global. After the first hop, traversal is restricted to linked successors.

Run at these cumulative loads:

- 0 distractor chains
- 3 distractor chains
- 8 distractor chains
- 18 distractor chains
- 48 distractor chains

Reuse the existing 72-query paraphrase benchmark for headquarters queries.

Add branching queries for:

- headquarters
- founder
- operating region

The first linked prototype should evaluate at least the Alice and Bob branches under all three relation intents.

## Metrics

For each load and routing variant, record:

- final-answer accuracy
- correct-root-selection rate
- correct-successor-selection rate
- complete-route accuracy
- average root-selection margin
- average successor-selection margin
- average steps per hop
- average total steps
- routing failure counts by status
- average runtime per query

For headquarters queries, continue recording:

- destination margin
- intermediate gain
- destination takeover step

## Expected scaling behavior

The linked-flow hypothesis predicts:

- root selection may gradually become harder as distractors increase
- once the correct root is selected, successor routing should remain stable
- unrelated city memories should not collectively distort later hops
- runtime after root selection should depend on local branching factor rather than total memory count

The experiment should therefore separate root-routing failures from successor-routing failures.

## Initial interfaces

Planned implementation boundaries:

### `src/linked_memory.py`

Defines:

- relation-intent structures
- linked transition memory structures
- linked-memory graph validation
- root and successor scoring helpers

### `src/linked_flow.py`

Defines:

- linked-flow runtime state
- per-edge latent movement
- handoff detection
- query-conditioned successor selection
- cycle and hop-limit handling
- routed result structures

### `tests/test_linked_memory.py`

Uses small synthetic vectors to test:

- root selection
- multi-phrase relation scoring
- graph successor lookup
- graph validation
- threshold behavior

### `tests/test_linked_flow.py`

Uses synthetic vectors to test:

- one-edge traversal
- two-hop traversal
- different typed successors from one intermediate
- no-matching-successor behavior
- target-not-reached behavior
- cycle prevention
- maximum-hop handling

### `experiments/linked_flow_branching.py`

Runs the manually constructed Alice and Bob branching graph without distractors.

### `experiments/linked_flow_distractor_scaling.py`

Runs the fixed-pool scaling comparison against the existing global-field baseline.

## Scope exclusions

This milestone does not include:

- automatic memory creation from conversations
- entity and relation extraction
- learned successor routing
- LLM-based routing
- memory confidence
- provenance
- contradiction resolution
- temporal validity
- memory updates
- forgetting
- consolidation
- approximate-nearest-neighbour indexing
- free-text answer generation

These remain future milestones.

## Long-term direction

A future conversational memory-formation pipeline may perform:

```text
conversation
    ↓
entity and relation extraction
    ↓
create or update linked transition memories
    ↓
embed trigger and relation-intent phrases
    ↓
connect compatible successor memories
    ↓
future queries traverse the resulting flow structure
```

The current milestone manually constructs this graph to test whether the routing and flow architecture is sound before automating memory formation.
