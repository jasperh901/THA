# Benchmark Results

Accuracy and F1 values are percentages. MAE is minimized; all other metrics are maximized.

## Multimodal Sentiment Analysis

| Backbone | MOSEI Acc-2 | F1 | Acc-7 | MAE | Corr | SIMS-V2 Acc-2 | F1 | Acc2-weak | MAE | Corr |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.8B | 85.15 | 85.03 | 54.75 | 0.517 | 0.791 | 79.30 | 79.34 | 73.29 | 0.322 | 0.668 |
| LLaMA2-7B | 85.94 | 85.71 | 54.11 | 0.535 | 0.758 | 79.40 | 79.12 | 73.29 | 0.329 | 0.631 |
| ChatGLM3-6B | **87.31** | **87.13** | 55.48 | **0.500** | **0.793** | **83.75** | **83.77** | **76.19** | **0.274** | 0.727 |

## Emotion Recognition In Conversations

| Backbone | CHERMA Acc | CHERMA WF1 |
| --- | ---: | ---: |
| Qwen-1.8B | 73.19 | 73.56 |
| LLaMA2-7B | 73.19 | 73.01 |
| ChatGLM3-6B | **74.79** | **74.77** |

## Trainable Core Parameters

Backbone parameters are excluded because they remain frozen.

| Backbone | MOSEI | SIMS-V2 | CHERMA |
| --- | ---: | ---: | ---: |
| Qwen-1.8B | 1.18M | 1.42M | 1.68M |
| LLaMA2-7B | 1.77M | 2.02M | 2.28M |
| ChatGLM3-6B | 1.78M | 2.02M | 2.28M |

## Fixed Main-Path Configuration

| Quantity | Value |
| --- | ---: |
| Adapter hidden width | 128 |
| Projection rank | 16 |
| Evidence slots per nontextual modality | 6 |
| Affective tokens | 8 |
| Dropout | 0.1 |
