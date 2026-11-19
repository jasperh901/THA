# Data Preparation

THA consumes language text and pre-extracted acoustic and visual sequences. Feature extraction is outside the released training environment: use the processed benchmark files linked in the main README.

## Expected Layout

```text
data/
|-- MOSEI/Processed/unaligned_50.pkl
|-- SIMS_V2/ch-simsv2s.pkl
`-- CHERMA0723/
    |-- cherma_train.pkl
    |-- cherma_valid.pkl
    `-- cherma_test.pkl
```

Paths can be changed in `src/tha/configs/datasets.yaml`. The command line takes the directory above these dataset folders as `--data-root`.

## MOSEI And SIMS-V2

Both datasets use one pickle containing `train`, `valid`, and `test` mappings. For each split THA reads:

| Field | Meaning |
| --- | --- |
| `raw_text` | Untokenized utterance text |
| `audio` | Acoustic sequence, shaped `[samples, time, feature]` |
| `vision` | Visual sequence, shaped `[samples, time, feature]` |
| `regression_labels` | Multimodal sentiment target |
| `id` | Optional stable sample identifier |
| `audio_lengths`, `vision_lengths` | Optional valid sequence lengths |

When explicit lengths are absent, the loader finds the last nonzero frame. NaN and infinite padding values are converted to zero. Tokenization is performed lazily with the selected backbone tokenizer.

## CHERMA

CHERMA is read directly from its three record pickles. Each record provides `features.text`, `features.audio`, `features.video`, and a string `label`. The configured Chinese emotion inventory is:

```text
anger, disgust, fear, happiness, neutral, sadness, surprise
```

The repository stores these labels in their original Chinese form in `src/tha/configs/datasets.yaml`.

## Validation

Run the model-independent inspection command before training:

```bash
tha inspect-data --backbone chatglm3 --dataset cherma --data-root ./data
```

This opens all splits, checks their schema and configured feature widths, and prints sample counts and target ranges. It does not load the LLM and does not use CUDA.
