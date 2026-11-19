<div align="center">

# THA

### A Thalamocortical-Inspired Affective Computing Adapter for Large Language Models

[Project page](https://jasperh901.github.io/THA/) | [Data guide](docs/DATA.md) | [Training profiles](docs/CONFIGS.md) | [Implementation](docs/IMPLEMENTATION.md)

</div>

THA (Thalamus Adapter) is a lightweight adapter for multimodal sentiment analysis and emotion recognition with frozen large language models. It instantiates Bionic Affective Computing (BAC), a functional biomimetic paradigm inspired by higher-order thalamocortical communication. THA selects context-relevant acoustic and visual evidence, regulates each modality by estimated reliability, compresses the result into eight affective tokens, and refreshes those tokens at selected Transformer depths.

The reference implementation supports Qwen-1.8B, ChatGLM3-6B, and LLaMA2-7B through their original Hugging Face implementations. No backbone source code or baseline implementation is copied into this repository.

## Highlights

- Frozen backbone: the embedding table, Transformer blocks, and language-model head receive no gradients.
- Bounded communication: six evidence slots per nontextual modality and eight affective tokens.
- Shared implementation: one adapter, data, training, evaluation, and checkpoint path for all supported settings.
- Adapter-only checkpoints: saved files contain THA state, configuration, and metrics, never base-model weights.

## Results

Headline test results with the ChatGLM3-6B backbone are shown below. Accuracy and F1 values are percentages.

| Dataset | Task | Main results |
| --- | --- | --- |
| MOSEI | sentiment regression | Acc-2 87.31, F1 87.13, MAE 0.500, Corr 0.793 |
| SIMS-V2 | sentiment regression | Acc-2 83.75, F1 83.77, MAE 0.274, Corr 0.727 |
| CHERMA | emotion recognition | Acc 74.79, WF1 74.77 |

Complete backbone and parameter tables are available in [docs/RESULTS.md](docs/RESULTS.md).

## Installation

Python 3.10 and the pinned package versions are recommended.

```bash
git clone https://github.com/jasperh901/THA.git
cd THA
conda env create -f environment.yml
conda activate tha
pip install -e .
```

Alternatively, install into an existing Python 3.10 environment:

```bash
pip install -r requirements.txt
pip install -e . --no-deps
```

The environment contains only the packages used by THA. Dataset feature extraction packages and unrelated experiment frameworks are intentionally excluded because the released processed features are consumed directly.

## Data

Download the processed datasets from the original MSE-Adapter release links:

| Dataset | Download | Expected local path |
| --- | --- | --- |
| MOSEI | [MOSEI_processed](https://huggingface.co/datasets/AZYoung/MOSEI_processed) | `data/MOSEI/Processed/unaligned_50.pkl` |
| SIMS-V2 | [SIMSV2_processed](https://huggingface.co/datasets/AZYoung/SIMSV2_processed) | `data/SIMS_V2/ch-simsv2s.pkl` |
| CHERMA | [CHERMA0723_processed](https://huggingface.co/datasets/AZYoung/CHERMA0723_processed) | `data/CHERMA0723/cherma_{split}.pkl` |

All three datasets are read directly. See [the data guide](docs/DATA.md) for their layouts and schemas.

Validate any configured dataset without loading an LLM or using a GPU:

```bash
tha inspect-data \
  --backbone chatglm3 \
  --dataset mosei \
  --data-root ./data
```

## Backbones

THA loads the original model implementation and weights from the configured model ID:

| Key | Model ID |
| --- | --- |
| `qwen18` | [Qwen/Qwen-1_8B](https://huggingface.co/Qwen/Qwen-1_8B) |
| `chatglm3` | [THUDM/chatglm3-6b-base](https://huggingface.co/THUDM/chatglm3-6b-base) |
| `llama2` | [meta-llama/Llama-2-7b-hf](https://huggingface.co/meta-llama/Llama-2-7b-hf) |

For a downloaded model, pass `--backbone-path /path/to/model`. LLaMA2 access requires accepting Meta's model license on Hugging Face.

## Training

```bash
tha train \
  --backbone chatglm3 \
  --dataset mosei \
  --data-root ./data \
  --output-dir outputs/mosei_chatglm3
```

The command uses the default profile for the selected dataset and backbone. Pass `--seed` to override its seed. See [docs/CONFIGS.md](docs/CONFIGS.md) for the profile settings. Outputs include:

```text
outputs/mosei_chatglm3/
|-- best_adapter.pt
|-- metrics.json
|-- run_config.json
`-- test_predictions.jsonl
```

Evaluate a saved adapter:

```bash
tha evaluate \
  --backbone chatglm3 \
  --dataset mosei \
  --data-root ./data \
  --checkpoint outputs/mosei_chatglm3/best_adapter.pt \
  --output-dir outputs/mosei_chatglm3_eval
```

Matched module ablations use the same path and disable exactly one operation:

```bash
tha train \
  --backbone chatglm3 \
  --dataset mosei \
  --data-root ./data \
  --output-dir outputs/ablation_no_semantic_projector \
  --disable-module semantic_projector
```

Available switches are `text_guidance`, `query_resampler`, `expert_gate`, `semantic_projector`, and `layerwise_fusion`.

## Repository Layout

```text
THA/
|-- assets/                 # Project-page assets
|-- docs/                   # Data, configuration, and implementation
|-- scripts/                # Thin command-line examples
|-- src/tha/
|   |-- configs/            # Packaged backbones, datasets, and adapter profiles
|   |-- backbone.py         # Original-model loader and runtime layer hooks
|   |-- modules.py          # Five THA operations
|   |-- model.py            # Prompt assembly, training, and generation
|   |-- data.py             # Unified multimodal datasets and collation
|   |-- metrics.py          # MSA and ERC metrics
|   `-- trainer.py          # Adapter-only optimization and evaluation
|-- tests/                  # CPU-only unit and integration tests
`-- index.html              # GitHub Pages project site
```

## Testing

Tests use a small synthetic causal model and do not download a backbone or reserve a GPU.

```bash
pip install -r requirements-dev.txt
pytest
ruff check src tests
```

## Acknowledgments

The processed dataset links are provided by the [MSE-Adapter](https://github.com/AZYoung233/MSE-Adapter) project. Backbone weights and model code remain subject to their respective licenses.

## License

The THA reference implementation is released under the [MIT License](LICENSE). Dataset and backbone licenses are not changed by this repository.
