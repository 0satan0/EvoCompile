# casememory

One file per usage example. Frontmatter `scheme:` is the retrieval link: when the tree (or explore) picks that scheme, these cards are attached to the SchemeHit and shown to the Optimizer.

- `role: canonical` — curated correct usage (seeded from eval/recipes + scheme_index).
- `role: collected` — written by `agent/loop.py` when `--no-memory-write` is off and a round produced a usable recipe.

Do not treat these as a model-name lookup table. Copy the **how** + recipe shape, then substitute the current case name.
