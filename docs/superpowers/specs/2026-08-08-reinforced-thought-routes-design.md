# Reinforced Thought Routes Design

## Goal

Milestone 1.7 tests whether successful multi-hop latent-memory traversals can become persistent learned thought patterns that improve future routing for semantically equivalent queries without contaminating nearby but different queries.

The milestone introduces learned route-level reinforcement while deliberately leaving latent edge-flow dynamics unchanged.

## Scope

This milestone includes:

- learned `ThoughtRoute` objects created from successful multi-hop traversals,
- storage of the traversed memory IDs and the successful query embedding,
- small reinforcement after later successful reuse,
- large reinforcement after explicit confirmation,
- bounded additive route-selection bonuses,
- controlled paraphrase generalization through accumulated successful query embeddings,
- tests that separate valid transfer from wrong-entity and wrong-intent leakage.

This milestone does not include:

- negative or repulsive memories,
- decay or forgetting,
- clarification-triggered memory creation,
- automatic schema consolidation,
- modifications to latent edge forces,
- LLM integration.

## Learning Lifecycle

A `ThoughtRoute` does not exist before it has been experienced successfully.

Lifecycle:

```text
graph contains linked memories
        ↓
query triggers multi-hop traversal
        ↓
route completes
        ↓
outcome is positively validated
        ↓
create ThoughtRoute
        ↓
future successful reuse gives small reinforcement
        ↓
explicit confirmation gives large reinforcement

A completed traversal alone is not enough to create a route. The result must also have positive evidence that it was correct or useful.

For the initial experimental implementation, validation is deterministic: the benchmark evaluator already knows the expected answer, so a route is created only when the final answer is correct.

ThoughtRoute Data Model

A route-level learned object should contain at least:

ThoughtRoute
├─ route_id
├─ memory_ids
├─ context_embeddings
├─ strength
├─ successful_uses
└─ explicit_confirmations
route_id

Stable identifier for the learned thought pattern.

memory_ids

Exact ordered tuple of traversed linked-memory IDs.

Example:

(
  "bob-works-at-globex",
  "globex-headquartered-in-paris",
)

Reinforcement belongs to the whole route, not to the individual edges.

context_embeddings

One or more embeddings for successful queries that activated and validated this route.

The first successful query creates the first context embedding. Later successful paraphrases that use the same route add their own embeddings.

This allows the route to develop a broader semantic basin through successful experience rather than through manually enumerated paraphrases.

strength

Bounded route reinforcement value used only during selection.

Recommended initial values:

initial route strength:   0.10
successful reuse:        +0.01
explicit confirmation:  +0.15
maximum strength:         1.00

These values must be configurable constants rather than hard-coded inside routing logic.

successful_uses

Count of validated successful uses of the route.

explicit_confirmations

Count of explicit confirmations. This is separate from successful_uses because confirmation is a much stronger learning signal.

Reinforcement Semantics

Reinforcement changes which route is preferred.

It does not change how the latent state moves once a memory edge has been selected.

The existing flow_linked_edge() behavior remains unchanged.

This separation is intentional so route learning and latent dynamics can be evaluated independently.

Route-Selection Bonus

The selected approach is an additive bounded bonus.

Conceptually:

final_score
  = semantic_score
  + route_bonus

with:

route_bonus
  = lambda
  × context_similarity
  × route_strength

where:

semantic_score is the existing root or successor semantic score,
context_similarity is the best similarity between the current query and the route's stored successful context embeddings,
route_strength is the learned bounded reinforcement value,
lambda caps how much route learning can override semantic evidence.

lambda must be configurable and deliberately small in the first experiment.

The reinforcement mechanism must not rescue an unrelated query whose underlying semantic match is poor.

Controlled Generalization

A reinforced route should generalize to semantically equivalent paraphrases but not to nearby wrong intents or wrong entities.

Example learned route:

Bob → works-at → Globex → headquartered-in → Paris
Should benefit
Where is the company Bob works for based?
What city is Bob's workplace headquartered in?
Should not benefit: different intent
Where does Bob's employer operate?
Who founded Bob's employer?
Should not benefit: different entity
Where is Alice's employer headquartered?

Later successful paraphrases using the same route add their query embeddings to that route's context set.

Route Identity

For this milestone, route identity is defined by the exact ordered tuple of linked-memory IDs.

Two traversals belong to the same ThoughtRoute only if they contain the same ordered memory path.

Different terminal relations from the same root are separate routes.

Example:

Bob → Globex → Paris    ≠    Bob → Globex → Europe

They may share edges, but they do not share reinforcement.

Positive Evidence

Route creation and small reinforcement require positive evidence.

For tests and experiments:

RoutingStatus.COMPLETED
AND
final decoded answer == expected answer

Only then may the system create or reinforce a route.

A route must not be strengthened merely because the router selected it.

Explicit confirmation is a separate event that applies the larger configured reinforcement increment.

Proposed Components
src/thought_routes.py

Owns the route-level data model and reinforcement operations.

Responsibilities:

define ThoughtRoute,
store and retrieve learned routes,
create a route from a validated traversal,
reinforce successful reuse,
reinforce explicit confirmation,
add successful context embeddings,
calculate route/context similarity,
calculate bounded route bonuses.

This module must not implement latent edge dynamics.

Linked-flow integration

The current linked-routing layer should receive route-bonus information during selection without embedding learning logic inside flow_linked_edge().

The implementation should keep the interface narrow enough that the existing unreinforced behavior remains available and testable.

Data Flow
query
  ↓
semantic root/successor scoring
  ↓
look up compatible learned ThoughtRoutes
  ↓
score query against route context embeddings
  ↓
add bounded reinforcement bonus
  ↓
select root/successor
  ↓
run unchanged latent edge flow
  ↓
complete route
  ↓
validate final outcome
  ↓
create or reinforce ThoughtRoute
Testing Strategy
Unit tests

Test route-store behavior independently:

no route exists before validated use,
validated use creates route,
failed or incorrect traversal does not create route,
repeated successful use increases strength by the small increment,
explicit confirmation increases strength by the large increment,
strength never exceeds the configured maximum,
identical memory path reuses the same route,
different ordered paths create separate routes,
successful paraphrase context can be added without duplicating an identical context unnecessarily.
Routing integration tests

Verify that route bonuses influence selection but do not alter edge dynamics.

Important checks:

zero reinforcement reproduces existing linked-flow results,
a relevant reinforced route receives a higher route-selection score,
unrelated routes receive negligible or zero effective bonus,
reinforced route selection still completes through unchanged latent dynamics.
Milestone experiment

Train/reinforce one route, for example:

Bob → Globex → Paris

Then evaluate three query groups:

semantic paraphrases of the same entity and intent,
same entity with a different terminal intent,
same intent with a different entity.

Measure before and after reinforcement:

root selection score,
successor selection score,
relevant route bonus,
competing route bonus,
selection margin,
exact-route accuracy,
final-answer accuracy.

The experiment should demonstrate useful transfer without cross-intent or cross-entity contamination.

Success Criteria

Milestone 1.7 succeeds if:

routes are learned only from validated successful experience,
route reinforcement persists across later queries,
successful reuse strengthens the entire multi-hop route,
explicit confirmation produces a substantially larger reinforcement update than ordinary successful reuse,
relevant paraphrases receive a measurable useful routing benefit,
different-intent and different-entity queries do not receive a comparable benefit,
existing latent edge dynamics remain unchanged,
existing linked-flow tests continue to pass when reinforcement is disabled or absent.
Future Extensions

The following are intentionally deferred:

confidence separate from reinforcement strength,
negative valence or inhibitory routes,
temporal decay,
correction-driven weakening,
clarification-driven memory formation,
route-schema consolidation,
automatic abstraction from repeated routes,
direct coupling of learned memory dynamics into LLM hidden-state computation.

These should be considered only after the reinforcement mechanism demonstrates controlled useful generalization in isolation.