import os
import torch
import torch.nn as nn
import torch.optim as optim
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.utils.data import DataLoader
from tqdm import tqdm

# -------------------------------
# Config
# -------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "bert-base-uncased"
BATCH_SIZE = 32
LR = 1e-5
EPOCHS = 3  # change as you like
CHECKPOINT_PATH = "reward_model_last.pt"

# -------------------------------
# Dataset
# -------------------------------

dataset = load_dataset(
    "openai/summarize_from_feedback",
    "comparisons",
    split="train"
)

print(f"Dataset length: {len(dataset)}")  # e.g. ~92k

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

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
    chosen_enc = tokenizer(chosen, padding=True, truncation=True, return_tensors="pt")
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
# Reward Model
# -------------------------------

class RewardModel(nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(encoder_name)
        self.reward_head = nn.Linear(self.encoder.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embeds = outputs.last_hidden_state[:, 0, :]
        rewards = self.reward_head(cls_embeds)
        return rewards.squeeze(-1)

model = RewardModel(MODEL_NAME).to(DEVICE)

# Unfreeze encoder if needed
for param in model.encoder.parameters():
    param.requires_grad = True

# Load from checkpoint if exists
if os.path.exists(CHECKPOINT_PATH):
    model.load_state_dict(torch.load(CHECKPOINT_PATH))
    print(f"✅ Loaded checkpoint from {CHECKPOINT_PATH}")

optimizer = optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=LR
)

total_steps = len(dataloader) * EPOCHS
warmup_steps = int(0.1 * total_steps)  # 10% warmup, adjustable
print(f"Total steps: {total_steps} | Warmup steps: {warmup_steps}")

scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

def pairwise_loss(chosen_rewards, rejected_rewards):
    return -torch.mean(torch.log(torch.sigmoid(chosen_rewards - rejected_rewards)))

# -------------------------------
# Training Loop with tqdm & checkpoint
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
            print(f"[Batch {batch_idx}] Chosen mean: {chosen_rewards.mean().item():.4f} | "
                  f"Rejected mean: {rejected_rewards.mean().item():.4f} | "
                  f"Diff mean: {(chosen_rewards - rejected_rewards).mean().item():.4f}")
            
            current_lr = scheduler.get_last_lr()[0]
            print(f"LR: {current_lr:.8f}")

        loss = pairwise_loss(chosen_rewards, rejected_rewards)
        epoch_loss += loss.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

    avg_loss = epoch_loss / len(dataloader)
    print(f"✅ Epoch {epoch+1} done. Avg loss: {avg_loss:.4f}")

    # Save checkpoint
    torch.save(model.state_dict(), CHECKPOINT_PATH)
    print(f"💾 Saved checkpoint to {CHECKPOINT_PATH}")

print("🎉 Training complete!")