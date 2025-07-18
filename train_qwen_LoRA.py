# train_qwen.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    get_linear_schedule_with_warmup
)
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes.optim import Adam8bit
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
BATCH_SIZE = 5
LR = 1e-5
EPOCHS = 3
CHECKPOINT_PATH = "./artifacts/qwen_loRA"

# -------------------------------
# Dataset
# -------------------------------

dataset = load_dataset(
    "openai/summarize_from_feedback",
    "comparisons",
    split="train"
)

# dataset = dataset.select(range(5000))  # For testing, use a small subset

print(f"Dataset length: {len(dataset)}")

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token  # ✅ for decoder LM

def collate_fn(batch):
    chosen = []
    rejected = []
    for ex in batch:
        summaries = ex["summaries"]
        choice = ex["choice"]
        chosen_summary = summaries[choice]["text"]
        rejected_summary = summaries[1 - choice]["text"]
        chosen.append(chosen_summary)
        rejected.append(rejected_summary)
    chosen_enc = tokenizer(
        chosen,
        padding=True,
        truncation=True,
        max_length=128,  # ✅ shorter for VRAM
        return_tensors="pt"
    )
    rejected_enc = tokenizer(
        rejected,
        padding=True,
        truncation=True,
        max_length=128,
        return_tensors="pt"
    )
    return {
        "chosen_input_ids": chosen_enc["input_ids"],
        "chosen_attention_mask": chosen_enc["attention_mask"],
        "rejected_input_ids": rejected_enc["input_ids"],
        "rejected_attention_mask": rejected_enc["attention_mask"],
    }

dataloader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    collate_fn=collate_fn
)

# -------------------------------
# Reward Model with LoRA + Qwen
# -------------------------------

class RewardModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()

        # ✅ LoRA config
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=8,
            lora_alpha=16,
            lora_dropout=0.05,
            target_modules=["q_proj", "v_proj"]
        )

        base_model = AutoModelForCausalLM.from_pretrained(encoder_name)
        self.encoder = get_peft_model(base_model, lora_config)
        self.encoder.gradient_checkpointing_enable()

        # print(self.encoder) # Debug: print model structure

        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids, 
            attention_mask=attention_mask,
            output_hidden_states=True
            )
        hidden_states = outputs.hidden_states[-1]  # [B, T, H]
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
        rewards = self.reward_head(pooled)
        return rewards.squeeze(-1)

model = RewardModel(MODEL_NAME).to(DEVICE)

# print("\n🔍 Trainable params:")
# for name, param in model.named_parameters():
#     if param.requires_grad:
#         print(f"{name:60} {param.shape}")

# Load checkpoint if exists
# if os.path.exists(CHECKPOINT_PATH):
#     model.load_state_dict(torch.load(CHECKPOINT_PATH))
#     print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")

# -------------------------------
# Optimizer: bitsandbytes Adam8bit
# -------------------------------

optimizer = Adam8bit(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

total_steps = len(dataloader) * EPOCHS
warmup_steps = int(0.1 * total_steps)
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

def pairwise_loss(chosen_rewards, rejected_rewards):
    return -torch.mean(torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)))

# -------------------------------
# Training loop
# -------------------------------

model.train()
for epoch in range(EPOCHS):
    print(f"\n🚀 Starting epoch {epoch+1}/{EPOCHS}...")
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}", ncols=0)):
        chosen_ids = batch["chosen_input_ids"].to(DEVICE)
        chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
        rejected_ids = batch["rejected_input_ids"].to(DEVICE)
        rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)

        if batch_idx % 500 == 0:
            diff_mean = (chosen_rewards - rejected_rewards).mean().item()
            print(f"[Batch {batch_idx}] Chosen mean: {chosen_rewards.mean().item():.2f} | "
                  f"Rejected mean: {rejected_rewards.mean().item():.2f} | "
                  f"Diff mean: {diff_mean:.2f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.8f}")

        loss = pairwise_loss(chosen_rewards, rejected_rewards)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()  # optional

    avg_loss = epoch_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

    # torch.save(model.state_dict(), CHECKPOINT_PATH)
    os.makedirs(CHECKPOINT_PATH, exist_ok=True)
    model.encoder.save_pretrained(CHECKPOINT_PATH)
    torch.save(model.reward_head.state_dict(), os.path.join(CHECKPOINT_PATH, "reward_head.pt"))
    print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

print("🎉 Training complete!")