# train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm
from bitsandbytes.optim import Adam8bit

# -------------------------------
# Config
# -------------------------------

# Setup device
if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
    print("Using NVIDIA CUDA GPU acceleration.")
elif torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
    print("Using Apple MPS GPU acceleration.")
else:
    DEVICE = torch.device("cpu")
    print("No GPU found, falling back to CPU.")


MODEL_NAME = "Qwen/Qwen3-0.6B-Base"  # ✅ Qwen model
BATCH_SIZE = 1
LR = 1e-5
EPOCHS = 3
CHECKPOINT_PATH = "./artifacts/reward_model_qwen3-0.6B-Base-last.pt"

# -------------------------------
# Dataset
# -------------------------------

dataset = load_dataset(
    "openai/summarize_from_feedback",
    "comparisons",
    split="train"
)

print(f"Dataset length: {len(dataset)}")  # e.g. ~92k

# -------------------------------
# Tokenizer (Qwen-specific)
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token  # ✅ Needed for decoder LMs

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
        max_length=128,  # was 512 or default, reduce to save memory
        return_tensors="pt")
    rejected_enc = tokenizer(rejected, padding=True, truncation=True, return_tensors="pt")
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
# Reward Model for decoder-only LM (Qwen)
# -------------------------------

class RewardModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.encoder.gradient_checkpointing_enable()  # Big save for RAM
        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.last_hidden_state  # [B, T, H]
        # Pool last non-padded token for each example
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
        rewards = self.reward_head(pooled)
        return rewards.squeeze(-1)

model = RewardModel(MODEL_NAME).to(DEVICE)

# Unfreeze encoder
for param in model.encoder.parameters():
    param.requires_grad = True

# Load checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")

# optimizer = optim.AdamW(
#     filter(lambda p: p.requires_grad, model.parameters()),
#     lr=LR
# )

optimizer = Adam8bit(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

# Scheduler with linear warmup + decay
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
# Training loop with tqdm & save
# -------------------------------

model.train()
for epoch in range(EPOCHS):
    print(f"\n🚀 Starting epoch {epoch+1}/{EPOCHS}...")
    epoch_loss = 0.0

    for batch_idx, batch in enumerate(tqdm(dataloader, desc=f"Epoch {epoch+1}")):
        chosen_ids = batch["chosen_input_ids"].to(DEVICE)
        chosen_mask = batch["chosen_attention_mask"].to(DEVICE)
        rejected_ids = batch["rejected_input_ids"].to(DEVICE)
        rejected_mask = batch["rejected_attention_mask"].to(DEVICE)

        chosen_rewards = model(chosen_ids, chosen_mask)
        rejected_rewards = model(rejected_ids, rejected_mask)

        if batch_idx % 100 == 0:
            diff_mean = (chosen_rewards - rejected_rewards).mean().item()
            print(f"[Batch {batch_idx}] Chosen mean: {chosen_rewards.mean().item():.4f} | "
                  f"Rejected mean: {rejected_rewards.mean().item():.4f} | "
                  f"Diff mean: {diff_mean:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.8f}")

        loss = pairwise_loss(chosen_rewards, rejected_rewards)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        torch.cuda.empty_cache()

    avg_loss = epoch_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

print("🎉 Training complete!")