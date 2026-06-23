Yes — **it can become misleading if “associative memory” is presented as the whole biological scope**.

But I would not remove it completely. I would treat **associative memory linking as the first executable test case**, not as the full theory.

The better framing is:

> **The repository studies branch-level dendritic accessibility as a general neural access-and-stabilization mechanism. The current article tests one narrow consequence: whether slow branch-level accessibility improves a simulator of associative-memory linking.**

That distinction is important.

## Why “associative memory” is too narrow as the umbrella

Your model is really about:

* dendritic branch availability,
* fast vs slow access states,
* local eligibility,
* replay/consolidation,
* perturbation-sensitive stabilization,
* branch reuse,
* hippocampal linking,
* and potentially prefrontal control / contextual gating.

The manuscript already says the framework separates fast dendritic/spine access from slower structural variables that bias which branches remain easiest to reopen, stabilize, and recruit across later events. That is broader than associative memory alone. 

It also explicitly treats structural accessibility as not carrying mnemonic content directly, but as biasing fast access through local geometry, transport readiness, and metabolic support.  That is a **general dendritic access mechanism**, not only a memory mechanism.

## Why “associative memory” is still useful

For publication, “associative memory linking” gives the model a **bounded test surface**.

Reviewers can understand:

* two traces,
* an overlap branch,
* replay,
* linking,
* focal damage,
* rescue,
* comparator baselines.

That is much easier to review than:

> “a general theory of dendritic accessibility across hippocampus and prefrontal cortex.”

So the best strategy is:

**Umbrella concept:** branch-level dendritic accessibility.
**Article testbed:** associative-memory linking.
**Biological emphasis:** hippocampal and prefrontal dendritic systems.
**Claim boundary:** the simulator tests one consequence, not the whole dendritic theory.

## Suggested wording

I would change the framing from:

> “A branch-resolved cytoskeletal-dendritic accessibility model of associative memory”

to something like:

> **“Branch-level dendritic accessibility as a slow structural constraint: a minimal simulator test in associative-memory linking”**

or:

> **“A minimal branch-accessibility simulator for testing slow dendritic constraints on memory linking”**

or, if you want the broader biology visible:

> **“Slow dendritic accessibility and memory linking: a branch-resolved simulator framework”**

## How to phrase the scope in the article

Use a paragraph like this:

> This paper does not propose that branch-level structural accessibility is limited to associative memory. The broader hypothesis is that dendritic branches can carry slow access biases that affect stabilization, reuse, and recruitment across neural systems, including hippocampal and prefrontal circuits. Associative-memory linking is used here as a deliberately narrow testbed because it offers clear operational signatures: trace overlap, replay-dependent consolidation, selective perturbation, and targeted rescue. The simulator therefore tests one falsifiable consequence of a broader dendritic-accessibility framework, rather than the full biological scope of dendritic computation.

That would protect you from both sides: it avoids being too narrow biologically, but it also avoids overclaiming.

## My recommendation

Do **not** title the whole project around associative memory.
Do title the **first article** around associative-memory linking.

So:

* Repository / program: **cytodendritic accessibility / branch-level dendritic access**
* Article v2: **minimal simulator of associative-memory linking**
* Future articles: hippocampus–PFC access gating, replay/stabilization, prefrontal context control, structural vulnerability, sleep/replay, etc.

That is the cleanest hierarchy.
