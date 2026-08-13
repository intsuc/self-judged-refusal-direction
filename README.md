# Self-judged refusal direction

`self-judged-refusal-direction` finds a refusal-related direction in an instruction model's hidden states, evaluates
causal interventions along that direction, and permanently removes it from the model's weights. Model-specific chat
rendering, response parsing, topology inspection, and edit targets are isolated behind architecture adapters.

The unchanged target model judges each parsed response as one trajectory together with its original prompt and truncation
flag. Invalid parses are rejected before the judge call, and constrained decoding limits each decision to `REFUSAL`,
`NON_REFUSAL`, or `UNCERTAIN`.

## Pipeline

1. Validate the unchanged target model's judge decisions against the packaged semantic cases.
2. Read one prompt per non-empty line, normalize and deduplicate prompts, group related templates, and split them before
   labeling.
3. Generate and parse a baseline response for each prompt using the configured target-generation settings.
4. Label complete, non-empty, non-truncated responses with the unchanged base checkpoint.
5. At each selected decoder layer, collect the residual activation at the last token of the same generation prefix used
   for target responses.
6. Exclude the final 20% of decoder layers, measure next-token KL for every remaining direction, and retain compatible
   directions in an order that prioritizes layers near 60% depth.
7. Run removal and activation-addition interventions on a small balanced subset, discarding candidates with no causal
   signal, excessive KL or CE loss, or catastrophic generation failures.
8. Apply the configured acceptance criteria to the pilot survivors on the full validation set.
9. Convert the selected direction to an architecture-specific weight-edit plan, save a standard Transformers checkpoint,
   verify it in a fresh offline process, and evaluate the independent test split once.

## Usage

Install the locked environment:

```bash
uv sync --locked
```

Copy the [reference configuration](configs/gemma4_31b_it.yaml), set an output directory, model ID, pinned model revision,
and prompt paths, then run the preflight checks before starting the full pipeline:

```bash
uv run sjrd inspect-model --config path/to/config.yaml
uv run sjrd validate-judge --config path/to/config.yaml
uv run sjrd run --config path/to/config.yaml
```

`validate-judge` passes only when the target model matches every expected label in the packaged semantic fixture. It does
not read user prompt files or generate target responses.

`data.prompt_files` and `data.reference_files` accept UTF-8 `.txt` paths. Each non-empty line is one prompt or reference text.

The preflight checks and pipeline stages can also be run separately:

1. `inspect-model`
2. `validate-judge`
3. `generate-baseline-trajectories`
4. `judge-baseline-trajectories`
5. `collect-activations`
6. `build-candidates`
7. `evaluate-candidates`
8. `export-model`
9. `evaluate-export`

## Configuration

`null` decoding parameters inherit the loaded model's generation configuration.

`acceptance.max_error_rate` limits `ERROR` records divided by input records for each generation and judging stage and
for each candidate. Changing only this threshold reuses completed model calls and recomputes candidate selection. Use
a new output directory if that changes the selected export after independent test evaluation begins. Judge validation
requires every packaged case to match.

| Key | Default | Meaning |
| --- | --- | --- |
| `run.seed` | `42` | Seed used for dataset splitting and target generation. |
| `run.output_dir` | required | Directory for run artifacts and the exported checkpoint. |
| `model.id` | required | Hugging Face model ID or local checkpoint path. |
| `model.revision` | required | Base checkpoint commit SHA. |
| `model.dtype` | `bfloat16` | Floating-point dtype used to load the model. |
| `model.device_map` | `auto` | Transformers/Accelerate device placement strategy. |
| `model.attention_implementation` | `sdpa` | Transformers attention implementation used at runtime. |
| `target_generation.system_prompt` | `null` | Optional system message prepended to every target request. |
| `target_generation.thinking_enabled` | `true` | Whether the model's official chat template enables its thinking response mode. |
| `target_generation.max_new_tokens` | `4096` | Maximum generated tokens per target response. |
| `target_generation.batch_size` | `1` | Fixed target-generation batch size; values above one require effective greedy decoding with one beam. |
| `target_generation.do_sample` | `false` | Sampling mode. |
| `target_generation.num_beams` | `1` | Beam count. |
| `target_generation.temperature` | `null` | Sampling temperature. |
| `target_generation.top_p` | `null` | Nucleus-sampling probability. |
| `target_generation.top_k` | `null` | Top-k sampling; `0` disables top-k filtering. |
| `target_generation.min_p` | `null` | Minimum-token-probability sampling. |
| `target_generation.typical_p` | `null` | Locally typical sampling probability. |
| `target_generation.repetition_penalty` | `null` | Repetition penalty; `1.0` applies no penalty. |
| `data.prompt_files` | `[]` | Plain-text prompt files used for discovery, validation, and testing. |
| `data.reference_files` | `[]` | Optional plain-text CE references; empty evaluates completed baseline non-refusal completions in chat context. |
| `data.train_fraction` | `0.6` | Target fraction of prompts assigned to discovery. |
| `data.validation_fraction` | `0.2` | Target fraction of prompts assigned to validation; the test target is the remainder. |
| `data.train_per_class` | `128` | Required labeled discovery trajectories per refusal class. |
| `data.validation_per_class` | `64` | Required labeled validation trajectories per refusal class. |
| `data.max_test_prompts` | `256` | Maximum number of test prompts retained. |
| `data.max_text_tokens` | `8192` | Maximum tokenizer length of a prompt or reference text. |
| `data.template_similarity_threshold` | `0.9` | Similarity threshold used to keep related prompt templates in the same split. |
| `search.layers` | `all` | Decoder layers to search, either `all` or a YAML list of zero-based layer indices. |
| `search.accumulator_dtype` | `float64` | Dtype for online activation means and variances: `float32` or `float64`. |
| `search.activation_screening_keep` | `32` | Maximum layer directions retained after activation screening. |
| `search.pilot_evaluation_keep` | `5` | Maximum directions retained after the pilot evaluation. |
| `search.pilot_prompts_per_class` | `16` | Validation prompts per refusal class used by the pilot evaluation. |
| `acceptance.min_removal_success_rate` | `0.05` | Smallest allowed fraction of baseline refusals changed to non-refusals. |
| `acceptance.min_activation_addition_induction_rate` | `0.05` | Smallest allowed fraction of baseline non-refusals changed to refusals by activation addition. |
| `acceptance.max_uncertain_rate` | `0.05` | Largest allowed semantic-uncertainty rate for a candidate. |
| `acceptance.max_error_rate` | `0.05` | Largest allowed parser or judge `ERROR` rate for each stage and candidate. |
| `acceptance.max_generation_failure_rate_increase` | `0.05` | Largest allowed increase over baseline in each of incomplete, empty, repeated, and repeated-delimiter response rates. |
| `acceptance.min_non_refusal_retention` | `0.95` | Smallest allowed fraction of baseline non-refusals that remain non-refusals. |
| `acceptance.max_mean_kl` | `0.10` | Largest allowed mean next-token KL divergence on baseline non-refusal prompts. |
| `acceptance.max_ce_loss_delta` | `0.10` | Largest allowed increase in mean CE loss on the CE evaluation inputs. |
| `acceptance.activation_addition_beta` | `1.0` | Scale applied to the raw mean-difference direction during the activation-addition causal check. |
| `export.max_shard_size` | `5GB` | Maximum Transformers checkpoint shard size. |
| `export.edit_chunk_rows` | `4096` | Rows processed at once while applying the weight projection in float32. |

## Architecture adapters

The adapter is selected from the pinned checkpoint's `model_type`. The included Gemma 4 adapter supports the
[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) reference configuration.

## Artifacts and privacy

Each artifact stage is content-hashed against only its inputs, runtime profile, and upstream results, so unrelated
later-stage configuration changes reuse completed earlier work. Generations and judgments are checkpointed
incrementally. Error summaries contain stable reason codes and reference owner-only diagnostics. Prompt text,
generated tokens, parsed responses, and private checkpoints remain owner-only and are excluded from exported
checkpoints. The pipeline does not publish models or artifacts.

## References

- [Arditi, A., Obeso, O., Syed, A., Paleka, D., Rimsky, N., Gurnee, W., & Nanda, N. (2024). **Refusal in Language Models Is Mediated by a Single Direction**. *ArXiv, abs/2406.11717*.](https://arxiv.org/abs/2406.11717)
