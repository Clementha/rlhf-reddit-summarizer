import os
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
)
from peft import PeftModel
from datasets import load_dataset
from tqdm import tqdm
import wandb
import warnings
warnings.filterwarnings("ignore", message="Setting `pad_token_id`")
warnings.filterwarnings("ignore", message="None of the inputs have requires_grad")
warnings.filterwarnings("ignore", message="`use_cache=True` is incompatible with gradient checkpointing")

# ⚙️ Config
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
CHECKPOINT_DIR = "./artifacts/qwen_loRA"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-5
BATCH_SIZE = 1
GRAD_ACCUM = 8
MAX_LEN = 256
MAX_NEW_TOKENS = 20
EPOCHS = 1

wandb.init(project="ppo-rlhf-no-trl", config={
    "lr": LR, "batch_size": BATCH_SIZE, "grad_accum": GRAD_ACCUM
})

# 🔤 Tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# 🤖 Load LoRA policy model
base_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
policy = PeftModel.from_pretrained(base_policy, CHECKPOINT_DIR)
policy.train()
policy.base_model.gradient_checkpointing_enable()
policy.base_model.enable_input_require_grads()
policy = policy.to(DEVICE)

# 🧠 Reward model
class RewardModel(torch.nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(encoder_name)
        self.reward_head = torch.nn.Linear(base.config.hidden_size, 1)

    def score(self, hidden_states):
        pooled = hidden_states[:, -1, :]
        return self.reward_head(pooled).squeeze(-1)

reward_model = RewardModel(MODEL_NAME).half().to(DEVICE)
reward_model.eval()

# 📄 Load and preprocess dataset
raw_dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train[:1000]")

def preprocess(example):
    return {"query": example["info"]["post"][:128]}  # safe truncation

processed_dataset = raw_dataset.map(preprocess)

def tokenize(example):
    tokens = tokenizer(
        example["query"],
        padding="max_length",
        truncation=True,
        max_length=MAX_LEN
    )
    return tokens

dataset = processed_dataset.map(tokenize, remove_columns=processed_dataset.column_names)
dataset.set_format(type="torch")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=data_collator)

# 🧮 PPO loss
def compute_ppo_loss(old_logprobs, new_logprobs, advantages, returns, values, clip_range=0.2, vf_coef=0.5, ent_coef=0.01):
    ratio = torch.exp(new_logprobs - old_logprobs)
    pg_loss1 = advantages * ratio
    pg_loss2 = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
    policy_loss = -torch.min(pg_loss1, pg_loss2).mean()
    value_loss = vf_coef * (returns - values).pow(2).mean()
    entropy_bonus = ent_coef * -new_logprobs.mean()
    return policy_loss + value_loss + entropy_bonus, policy_loss, value_loss, entropy_bonus

# 🚀 Optimizer
optimizer = torch.optim.Adam(policy.parameters(), lr=LR)

# 🌀 Training loop
step = 0
query_texts = [ex["info"]["post"][:128] for ex in raw_dataset]

for epoch in range(EPOCHS):
    for i, batch in enumerate(tqdm(loader, desc="PPO Batches", ncols=80)):
        input_ids = batch["input_ids"].to(DEVICE)
        attention_mask = batch["attention_mask"].to(DEVICE)
        input_texts = tokenizer.batch_decode(input_ids, skip_special_tokens=True)

        with torch.no_grad():
            response_ids = policy.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                top_k=50,
                use_cache=False,
                pad_token_id=tokenizer.pad_token_id,  # Explicitly set to avoid warning
            )
            responses = tokenizer.batch_decode(response_ids, skip_special_tokens=True)

        # Tokenize full prompt+response
        full_texts = [q + r for q, r in zip(input_texts, responses)]
        full_inputs = tokenizer(full_texts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_LEN).to(DEVICE)

        logits = policy(**full_inputs).logits
        logprobs = F.log_softmax(logits, dim=-1)
        new_logprobs = logprobs[:, -1, :].gather(1, full_inputs["input_ids"][:, -1].unsqueeze(-1)).squeeze()

        with torch.no_grad():
            values = logits[:, -1, :].mean(dim=-1)
            with torch.no_grad():
                output = policy.base_model(
                    input_ids=full_inputs["input_ids"],
                    attention_mask=full_inputs["attention_mask"],
                    output_hidden_states=True,
                    return_dict=True,
                )
                hidden = output.hidden_states[-1]  # (batch, seq_len, hidden_dim)
            rewards = reward_model.score(hidden)
            advantages = rewards - values
            returns = rewards
            old_logprobs = new_logprobs.detach()

        loss, pl, vl, ent = compute_ppo_loss(old_logprobs, new_logprobs, advantages, returns, values)
        loss = loss / GRAD_ACCUM
        loss.backward()

        if (step + 1) % GRAD_ACCUM == 0:
            optimizer.step()
            optimizer.zero_grad()

        wandb.log({
            "total_loss": loss.item(),
            "policy_loss": pl.item(),
            "value_loss": vl.item(),
            "entropy": ent.item(),
            "reward": rewards.mean().item()
        })

        step += 1

# 💾 Save model
torch.save(policy.state_dict(), "./artifacts/qwen_ppo_policy_manual.pt")
wandb.finish()