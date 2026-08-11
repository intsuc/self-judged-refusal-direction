# Self-judged refusal direction

`self-judged-refusal-direction` extracts, evaluates, and permanently removes a rank-1 trajectory-level refusal
direction from instruction models through architecture-specific adapters. The direction represents either refusal
consideration in generated thinking or actual refusal in the final answer; it is not limited to refusal wording in the
visible answer.

For every prompt, the pipeline generates a complete thinking-enabled target trajectory. The same immutable base
checkpoint then acts as a thinking-disabled judge that reads the original prompt, thinking, final answer, and parser
metadata. An enum trie restricts its output to exactly `REFUSAL`, `NON_REFUSAL`, or `UNCERTAIN` followed by EOS.

## Pipeline

1. Normalize, deduplicate, group, and split raw prompts before labeling.
2. Generate baseline thinking and final-answer trajectories.
3. Classify trajectories with the unchanged base checkpoint.
4. Collect online activation statistics at adapter-defined boundaries.
5. Build mean-difference directions and evaluate them through staged causal interventions.
6. Apply the selected `WeightEditPlan` permanently and save a standard Transformers checkpoint.
7. Reload the checkpoint in a fresh offline process and evaluate the independent test split once.

Temporary intervention and permanent editing are derived from the same plan. Exported checkpoints require no hooks,
monkey patches, custom decoder layers, remote code, or this package at inference time.

## Usage

Install the locked environment:

```bash
uv sync --locked
```

Create or copy a YAML configuration and set at least:

- `model.id`, a 40-character commit `model.revision`, and a registered `model.adapter`
- `data.raw_prompt_files`
- `run.output_dir`
- dataset sizes and evaluation thresholds appropriate for the run

Prompt inputs may be text, JSON, JSONL, NDJSON, or CSV. Optional `data.quality_text_files` provide a separate corpus for
CE-loss evaluation.

Inspect compatibility before starting the full run:

```bash
uv run self-judged-refusal-direction inspect-model --config path/to/config.yaml
uv run self-judged-refusal-direction run --config path/to/config.yaml
```

The stages can also be run separately:

```text
generate-baseline-trajectories
judge-baseline-trajectories
collect-activations
build-candidates
evaluate-candidates
export-model
evaluate-export
```

Each stage validates artifact content and provenance, including model revision, generation and judge profiles, and chat
template hashes. A failed parser, context overflow, incompatible topology, or infrastructure error is kept separate
from semantic `UNCERTAIN` results and fails closed according to the configuration.

## Architecture adapters

Model-specific chat rendering, response parsing, activation boundaries, topology discovery, and edit targets are
contained in `ArchitectureAdapter` implementations registered by name. A new model family can be supported without
changing the pipeline, judging, evaluation, or export orchestration.

The included `gemma4` adapter and [reference configuration](configs/gemma4_31b_it.yaml) support
`google/gemma-4-31B-it` at pinned revision `842da3794eaa0b77d5f08bae87a17459d91ff475`. The implementation's
mean-activation difference and weight-orthogonalization approach builds on
[`andyrdt/refusal_direction`](https://github.com/andyrdt/refusal_direction/tree/9d852fae1a9121c78b29142de733cb1340770cc3).

## Artifacts and privacy

Run artifacts and exported model files are written to separate directories. Raw thinking trajectories are
owner-readable private artifacts and are never included in exported checkpoints. Hub pushes, public endpoints, and
automatic publication are disabled. Exported weights can be loaded with the model family's standard Transformers auto
class and `trust_remote_code=False`.
