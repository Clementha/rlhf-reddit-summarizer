# 🧠 GPT-Qwen: Reinforcement Learning from Human Feedback (RLHF) with Qwen

This project implements the ideas from the 2020 paper **[“Learning to summarize from human feedback”](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)** using the [Qwen](https://huggingface.co/Qwen) language model family. The goal is to fine-tune a model to produce higher-quality summaries, guided by a reward model trained on human preferences.

The pipeline uses:
- A **LoRA-based reward model** trained to predict summary preference
- A **PPO (Proximal Policy Optimization)** training loop (without `trl`) to fine-tune a language model
- **WandB logging**, gradient accumulation, and VRAM-friendly settings for efficient training on <10 GB GPUs

---

## 🧪 Paper Background

The method was first introduced in [Stiennon et al. (2020), NeurIPS](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf). It proposes a way to:
1. Collect human preferences over pairs of model-generated summaries
2. Train a reward model to mimic those preferences
3. Use reinforcement learning (specifically PPO) to fine-tune a language model using the learned reward

---

## 🛠️ Usage

### 1️⃣ Train and save the reward model
```bash
python train_qwen_LoRA.py
```

This script:
- Loads `openai/summarize_from_feedback` (comparisons split)
- Applies LoRA to Qwen
- Trains a reward model using pairwise preference loss
- Saves both LoRA adapter weights and the reward head

---

### 2️⃣ Train and save the PPO model
```bash
python train_rlhf_no_trl.py
```

This script:
- Loads the reward model and Qwen policy model
- Samples responses using generation
- Computes reward and value functions
- Optimizes the policy with PPO
- Logs rewards, losses, and advantage values to [Weights & Biases](https://wandb.ai/)

---

### 3️⃣ Predict summary using PPO-trained policy
```bash
usage: predict_ppo.py [-h] --title TITLE --post POST

# Example:
python predict_ppo.py \
  --title "Amazon Orders 350,000 Employees To Relocate Or Resign Without Severance" \
  --post "This directive goes beyond simple logistics to represent Amazon's overarching strategic goal of preserving its competitive advantage in a market that is extremely dynamic. Being physically close encourages impromptu conversations, speedier decision-making, and a greater sense of purpose, elements that are challenging to replicate virtually. With decades of operational experience, Amazon's leadership understands that real-time idea collisions foster innovation."
```

✅ Output:
```text
🔄 Loading tokenizer and model...

📘 Title: Amazon Orders 350,000 Employees To Relocate Or Resign Without Severance
📝 Post: This directive goes beyond simple logistics...

🤖 Generating summary...

✅ Summary:
Amazon's strategy of relocating or resigning employees without severance aims to enhance operational efficiency, foster innovation, and maintain its competitive edge in a rapidly evolving market. The move reflects Amazon's commitment to physical proximity, which encourages impromptu conversations, speedier decision-making, and a stronger sense of purpose.
```

---

## 📦 Project Structure

```
gpt-qwen/
├── train_qwen_LoRA.py        # Reward model training (LoRA + pairwise loss)
├── train_rlhf_no_trl.py      # PPO training loop (manual PPO logic)
├── predict_ppo.py            # Inference script for summarization
├── artifacts/                # Saved LoRA weights, PPO policy checkpoints, reward heads
```

---

## ✅ Dependencies

- `transformers`
- `datasets`
- `peft`
- `bitsandbytes`
- `torch`
- `wandb`

Install them via:

```bash
pip install -r requirements.txt
```

---

## 🙌 Acknowledgements

- Based on [OpenAI's 2020 summarization paper](https://proceedings.neurips.cc/paper/2020/file/1f89885d556929e98d3ef9b86448f951-Paper.pdf)
- Built with [Qwen](https://huggingface.co/Qwen) from Alibaba Cloud
- Uses [LoRA](https://arxiv.org/abs/2106.09685) from PEFT
