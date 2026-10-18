# Reinforcement Learning from Human Feedback: Qwen2.5 LoRA Fine-Tuning on UltraFeedback with TRL

Fine-tuning of **Qwen2.5-0.5B-Instruct** with **LoRA** using Hugging Face **TRL** on **UltraFeedback**, a human-preference dataset built for RLHF. The model is trained with supervised fine-tuning on the preferred (`chosen`) responses. Runs on a free Colab T4 GPU.

## What the notebook does

1. Loads `Qwen/Qwen2.5-0.5B-Instruct` and its tokenizer.
2. Loads `HuggingFaceH4/ultrafeedback_binarized` (pinned revision) and converts the `chosen` conversations into prompt/completion chat format.
3. Fine-tunes on the first 5,000 examples of `train_sft` with `SFTTrainer` and a LoRA adapter.
4. Saves the adapter and tokenizer to `/content/sft` and zips them.

## Setup

| Setting | Value |
|---|---|
| Base model | Qwen2.5-0.5B-Instruct |
| Dataset | UltraFeedback binarized, 5,000 SFT samples |
| LoRA | r=16, alpha=16, dropout=0.05, on `q_proj`, `k_proj`, `v_proj`, `o_proj` |
| Learning rate | 2e-4 |
| Epochs | 1 |
| Max length | 512 tokens |
| Batch | 1 per device × 8 gradient accumulation |
| Precision | bf16, gradient checkpointing |

## Results

625 steps in about 54 minutes on a T4. Final training loss **1.32**, mean token accuracy **0.63**.

## Run it

Open `rlhf.ipynb` in Google Colab, set the runtime to GPU, and run all cells.

```bash
pip install transformers trl peft datasets accelerate
```

## Stack

PyTorch · Transformers · TRL · PEFT · Datasets · Accelerate
