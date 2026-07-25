# Fixed-Pool Distractor Scaling Design

## Goal

Measure how two-hop latent composition degrades as a fixed, deterministic pool of semantically similar person-to-company-to-city chains is added to the memory field.

## Research question

Does the correct Alice or Bob chain remain dominant when many structurally similar employer and headquarters memories contribute to the same summed latent vector field?

## Target chains

The benchmark keeps these chains unchanged:

- Alice → Acme → Helsinki
- Bob → Globex → Paris

The existing paraphrase benchmark remains the evaluation query set.

## Distractor pool

Use a fixed ordered pool of realistic names, companies, and cities. Each distractor is a complete two-hop chain:

- person → company
- company → headquarters city

The pool ordering must never change between runs. Every load level uses a prefix of the same pool, making results reproducible and comparable across architecture changes.

Initial load levels:

- 0 distractor chains
- 3 distractor chains
- 8 distractor chains
- 18 distractor chains
- 48 distractor chains

The two target chains are not counted as distractors.

## Memory construction

Each relation uses the existing narrow multi-trigger strategy and `TargetMemory` implementation. Distractor chains must use the same number and style of trigger paraphrases as the target chains so the comparison is fair.

All memories are active simultaneously. The experiment must not add top-k gating, relation labels, or other architectural changes; it measures the current summed-field implementation as-is.

## Evaluation queries

Reuse the 72-query composition robustness benchmark:

- 16 curated straightforward queries
- 16 curated hard queries
- 40 generated queries

Only Alice and Bob queries are evaluated. Distractor chains exist solely to create interference.

## Measurements

For each load level, record:

- correct query count
- total query count
- accuracy
- average destination margin
- minimum destination margin
- average intermediate gain
- average destination takeover step among successful takeovers
- maximum destination takeover step
- number of queries with no takeover
- average strongest wrong-city similarity
- average runtime per query

Also print the weakest individual queries at each load level, ordered by destination margin.

## Wrong-city measurement

Wrong-city similarity must consider city candidates only, excluding:

- the expected destination
- people
- companies
- country distractors

This isolates second-hop city interference from general candidate ranking noise.

## Determinism

- Use a fixed ordered distractor pool stored in source code.
- Do not sample or shuffle distractors.
- Use identical encoder, radius, strength, step count, and step size at every load.
- Run load levels cumulatively using prefixes of the pool.

## Success interpretation

This experiment is exploratory and sets no pass threshold yet.

Possible outcomes:

1. Stable accuracy and margins: summed fields scale better than expected.
2. Stable accuracy but falling margins or later takeovers: the architecture is becoming fragile before outright failure.
3. Rapid accuracy collapse: weak activations from similar memories accumulate strongly, motivating competitive gating.
4. Chain-specific degradation: frozen encoder geometry creates uneven robustness across entities.

## Scope exclusions

This milestone does not include:

- random distractor sampling
- city-only distractor controls
- shared-entity conflicts
- contradictory facts
- top-k or softmax gating
- three-hop composition
- learned memory consolidation

Those are follow-up experiments selected from the observed degradation pattern.

## Files

Planned implementation boundaries:

- `src/scaling.py`: fixed distractor data structures and aggregate scaling metrics
- `tests/test_scaling.py`: fast synthetic-vector unit tests for scaling summaries
- `experiments/composition_distractor_scaling.py`: sentence-transformer benchmark runner and reporting

Existing `src/robustness.py` remains responsible for per-query trajectory evaluation.
