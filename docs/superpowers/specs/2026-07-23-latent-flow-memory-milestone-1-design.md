# Latent Flow Memory — Milestone 1 Design

## Goal

Build a minimal proof of concept for persistent memory represented as a learned directional influence over a pretrained semantic activation space.

The experiment will test whether a query embedding can be moved through a local vector field toward a learned target concept without:

* modifying the pretrained encoder,
* placing the learned fact in the query context,
* using retrieval-augmented generation,
* or retraining a language model.

The first test fact is:

> Matu's cat is named Rufus.

The corresponding query is:

> What is Matu's cat's name?

## Hypothesis

A learned association can be represented as a local directional field in latent space.

Given a latent state (x), the memory field produces a direction:

[
V(x)
]

and updates the state iteratively:

[
x_{t+1}=x_t+\alpha V(x_t)
]

A successful memory should cause semantically relevant query embeddings to move closer to the target representation while having little effect on unrelated regions.

## Architecture

### 1. Frozen Encoder

Use SentenceTransformers `all-MiniLM-L6-v2`.

The encoder maps text to normalized 384-dimensional embeddings:

[
E(text)\rightarrow x\in\mathbb{R}^{384}
]

The encoder remains completely frozen.

File:

`src/encoder.py`

Responsibilities:

* load the embedding model,
* encode individual strings,
* encode lists of strings,
* return normalized NumPy vectors.

### 2. Flow Memory

A memory represents a local deformation of latent dynamics.

Each memory contains:

* `center`: semantic region where the memory becomes relevant,
* `direction`: direction toward the learned target,
* `strength`: magnitude of its influence,
* `radius`: spatial range of influence.

File:

`src/memory.py`

For the first memory:

[
c=E(\text{"Matu's cat's name"})
]

[
y=E(\text{"Rufus"})
]

[
v=y-c
]

The memory stores (c) and (v).

### 3. Local Vector Field

The influence of a memory decreases with distance from its center.

Use a Gaussian radial function:

[
w(x)=
\exp\left(
-\frac{|x-c|^2}{2\sigma^2}
\right)
]

The memory's contribution is:

[
V_i(x)=s_iw_i(x)v_i
]

For multiple memories:

[
V(x)=\sum_i V_i(x)
]

File:

`src/field.py`

The implementation must return zero influence when no memories exist.

### 4. Latent Dynamics

A query begins at:

[
x_0=E(query)
]

It is iteratively updated:

[
x_{t+1}=x_t+\alpha V(x_t)
]

After each update, normalize the vector to maintain comparable cosine geometry.

File:

`src/dynamics.py`

The function must return:

* the final state,
* every intermediate state in the trajectory.

This trajectory will later be used for visualization and analysis.

## Experiment

File:

`experiments/basic_recall.py`

Candidate concepts should include at minimum:

* Rufus
* Helsinki
* Paris
* dog
* cat
* Finland
* France

### Baseline

Encode:

> What is Matu's cat's name?

Calculate cosine similarity between the query and every candidate.

Record rankings before applying any memory field.

### Memory Condition

Create the memory:

[
E(\text{"Matu's cat's name"})
\rightarrow
E(\text{"Rufus"})
]

Run the query through the vector field for multiple steps.

Calculate candidate similarities again.

### Primary Success Criterion

After applying the flow field:

[
similarity(x_T,E(\text{"Rufus"}))

>

similarity(x_0,E(\text{"Rufus"}))
]

The experiment should report the magnitude of this change.

A stronger result is obtained if Rufus also rises meaningfully in candidate ranking.

## Important Non-Goals

Milestone 1 will not include:

* LLM hidden-state modification,
* automatic memory extraction,
* sleep or consolidation,
* fast versus slow memory,
* forgetting,
* tagging,
* neural networks that learn the vector field,
* RAG,
* conversational UI,
* arbitrary question answering.

These features belong to later milestones.

## Testing

Unit tests should verify:

### Encoder

* output dimensionality is 384,
* returned embeddings are normalized,
* identical text produces identical embeddings.

### Flow Field

* no memories produce a zero vector,
* influence is strongest at the memory center,
* influence decreases with distance,
* increasing strength increases influence magnitude.

### Dynamics

* requested number of trajectory steps is returned,
* output vectors remain normalized,
* zero field leaves the state unchanged apart from negligible numerical error.

### Experiment

The integration test should verify that the Rufus similarity increases after applying the learned flow.

## Main Scientific Question

Milestone 1 is not intended to prove that vector-field memory is superior to ordinary retrieval.

It tests only the foundational mechanism:

> Can persistent knowledge be represented as a localized change in latent dynamics that predictably steers semantically relevant states?

If this fails even in a controlled setting, the architecture must be reconsidered before adding sleep, consolidation, or LLM integration.

If it succeeds, Milestone 2 will introduce runtime learning and persistent fast memory.
