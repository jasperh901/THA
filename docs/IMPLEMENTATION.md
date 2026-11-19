# Implementation Notes

## Scope

This repository contains the THA main path and its five matched ablation switches. It deliberately excludes unrelated baselines, exploratory experiment variants, generated result caches, copied backbone source, and feature-extraction environments.

## Frozen-Backbone Contract

`FrozenBackbone.from_pretrained` resolves one of the model IDs in `src/tha/configs/backbones.yaml` through `AutoModelForCausalLM` and `AutoTokenizer`. A local compatible model directory may be supplied at runtime. Immediately after loading:

- every backbone parameter has `requires_grad=False`;
- the backbone remains in evaluation mode when THA enters training mode;
- the optimizer receives only parameters whose gradient flag is enabled;
- `save_adapter` serializes `adapter.state_dict()`, never `backbone.state_dict()`.

No file from Qwen, ChatGLM3, or LLaMA2 is vendored in this repository.

## Forward Path

The implementation follows one stable sequence:

1. Tokenize the raw utterance or conversation and read its frozen token embeddings.
2. Build a mask-aware text summary for top-down context.
3. Encode variable-length audio and visual streams into fixed evidence banks and scalar reliability estimates.
4. Reliability-weight and resample both memories into eight affective states.
5. Reconcile local, memory-grounded, and global expert views.
6. Project each state toward frozen semantic anchors with a low-rank residual.
7. Insert the eight projected states between the text and task prompt.
8. At selected frozen Transformer layers, let only this affective span re-read the compact memory.
9. Train the causal LM response tokens while gradients flow through the frozen computation graph into THA only.

The backbone input mask preserves the fixed affective span across a batch. This makes one hook state valid for every sample even when text lengths differ.

## Runtime Layer Injection

`LayerInjectionManager` registers forward hooks on the configured block container:

| Backbone | Layer container | Selected indices |
| --- | --- | --- |
| Qwen-1.8B | `transformer.h` | 8, 16, 23 |
| ChatGLM3-6B | `transformer.encoder.layers` | 8, 16, 24 |
| LLaMA2-7B | `model.layers` | 8, 16, 24 |

The hook accepts both batch-first and sequence-first hidden states. It replaces only the affective-token interval, leaves text states untouched, and preserves tuple/list return structures used by remote model implementations. The manager is active only inside a THA backbone call and restores its previous state on exit.

## Checkpoint Format

An adapter checkpoint is a PyTorch mapping with:

```text
format: tha-adapter-v1
config: complete composed run configuration
disabled_modules: matched ablation switches
state_dict: THA parameters and semantic-anchor buffers
extra: best epoch and validation metrics
```

The checkpoint must be paired with the same backbone family and dataset configuration. A local backbone path can differ from the configured Hugging Face ID if its architecture and tokenizer are compatible.

The composed configuration records every resolved dataset, adapter, and training value. Dataset/backbone defaults therefore remain traceable after training, while an explicit `--seed` override is preserved in the same checkpoint metadata.

## Generation And Metrics

Evaluation uses deterministic greedy decoding with a short task-specific token budget. Numeric output is parsed, clipped to the dataset range, and reported with MAE, Pearson correlation, binary accuracy/F1, and rounded seven-class accuracy. Classification output is matched against the configured label inventory and reported with accuracy, weighted F1, and parse coverage.

Predictions are written as JSON Lines with sample ID, target, parsed prediction, and raw decoded response.

## CPU Verification

The test suite builds a small synthetic causal model that exposes the same embedding, block-list, tuple-output, and causal-loss contracts used by the supported backbones. It verifies:

- configuration composition and both dataset formats;
- THA tensor shapes and gradient flow;
- batch-first runtime layer injection;
- teacher-forced loss and greedy generation;
- adapter-only checkpoint save/load;
- metric parsing and computation.

These tests neither download model weights nor initialize CUDA.
