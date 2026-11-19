# Training Profiles

Each dataset and backbone has a default training configuration.

| Backbone | Dataset | Seed | Batch | Learning rate | Epochs | Early stop | Output tokens |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Qwen-1.8B | MOSEI | 7777 | 4 | 5e-5 | 30 | 10 | 4 |
| Qwen-1.8B | SIMS-V2 | 3333 | 4 | 4e-5 | 20 | 10 | 4 |
| Qwen-1.8B | CHERMA | 7777 | 4 | 5e-5 | 30 | 8 | 4 |
| ChatGLM3-6B | MOSEI | 3333 | 1 | 5e-5 | 30 | 10 | 4 |
| ChatGLM3-6B | SIMS-V2 | 2222 | 1 | 5e-5 | 30 | 10 | 4 |
| ChatGLM3-6B | CHERMA | 3333 | 1 | 5e-5 | 30 | 8 | 2 |
| LLaMA2-7B | MOSEI | 2022 | 4 | 3e-5 | 24 | 10 | 5 |
| LLaMA2-7B | SIMS-V2 | 2022 | 4 | 4e-5 | 20 | 10 | 4 |
| LLaMA2-7B | CHERMA | 5055 | 1 | 3.5e-5 | 24 | 8 | 4 |

The exact prompt, data view, layer-gate setting, expert temperature, residual bias, and training parameters are stored with each profile in `src/tha/configs/datasets.yaml`. `compose_config` places the resolved values in `run_config.json` and adapter checkpoints.

Running without `--seed` uses the listed seed. An explicit `--seed` overrides only the seed and leaves the rest of the selected profile unchanged:

```bash
tha train \
  --backbone chatglm3 \
  --dataset simsv2 \
  --data-root ./data \
  --output-dir outputs/chatglm3_simsv2

tha train \
  --backbone chatglm3 \
  --dataset simsv2 \
  --data-root ./data \
  --output-dir outputs/chatglm3_simsv2_seed3333 \
  --seed 3333
```
