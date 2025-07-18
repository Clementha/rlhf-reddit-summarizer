import os
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling
)
from peft import PeftModel, get_peft_model
from trl import PPOTrainer, PPOConfig
from datasets import load_dataset
from tqdm import tqdm

# -------------------------------
# ⚙️ Config
# -------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
CHECKPOINT_DIR = "./artifacts/qwen_loRA"
REWARD_WEIGHTS_PATH = "./artifacts/reward_model_last_Qwen_loRA.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LR = 1e-5
BATCH_SIZE = 1

# -------------------------------
# Tokenizer
# -------------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# Load LoRA policy model
# -------------------------------
base_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
policy = PeftModel.from_pretrained(base_policy, CHECKPOINT_DIR)
policy.base_model.to(DEVICE)
policy.to(DEVICE)
policy.train()

# -------------------------------
# Reward model
# -------------------------------
class RewardModel(torch.nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(encoder_name)
        self.encoder = PeftModel.from_pretrained(base, CHECKPOINT_DIR)
        self.reward_head = torch.nn.Linear(base.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True
        )
        hidden_states = outputs.hidden_states[-1]
        lengths = attention_mask.sum(dim=1) - 1
        pooled = hidden_states[torch.arange(hidden_states.size(0)), lengths]
        rewards = self.reward_head(pooled)
        return rewards.squeeze(-1)

reward_model = RewardModel(MODEL_NAME).to(DEVICE)
reward_model.load_state_dict(torch.load(REWARD_WEIGHTS_PATH))
reward_model.eval()

# -------------------------------
# Dataset and Collator
# -------------------------------
dataset = load_dataset("openai/summarize_from_feedback", "comparisons", split="train[:1000]")
# print(dataset[0])
def custom_dataset_format(example):
    return {"query": example["info"]["post"][:128]}

dataset = dataset.map(custom_dataset_format)
def tokenize(example):
    encoding = tokenizer(
        example["query"],
        padding="max_length",
        truncation=True,
        max_length=512,
    )
    # Convert tensors to list so HuggingFace datasets can serialize them
    encoding = {k: v for k, v in encoding.items()}  # No squeeze or .tolist()
    encoding["query_text"] = example["query"]  # ✅ renamed to avoid tokenizer conflict
    return encoding

dataset = dataset.map(tokenize, batched=True, remove_columns=dataset.column_names)
query_texts = [example["query_text"] for example in dataset]

# ✅ Remove text fields before passing to PPOTrainer
dataset = dataset.remove_columns(["query_text"])
dataset.set_format(type="torch")
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# -------------------------------
# PPO Config and Trainer
# -------------------------------


# ✅ Define PPO config
ppo_config = PPOConfig(
    learning_rate=1e-5,
    batch_size=1,
    mini_batch_size=1,
    # optional: log_with="wandb" or similar
)

# ✅ Create PPOTrainer with the correct positional config


# Minimal processing class as required by TRL v0.19.1
class DummyProcessing:
    pad_token_id = tokenizer.pad_token_id  # ✅ Class-level attribute

    def __init__(self, tokenizer, config):
        self.tokenizer = tokenizer
        self.config = config

    def __call__(self, samples):
        return samples  # No processing; assumes dataset is already preprocessed

value_model = AutoModelForCausalLM.from_pretrained(MODEL_NAME).to(DEVICE)

# Strip fields that will break collation
existing_cols = set(dataset.column_names)
cols_to_remove = [col for col in ["query_text", "query"] if col in existing_cols]

# dataset = dataset.remove_columns(cols_to_remove)
dataset.set_format(type="torch")
#print(dataset[0])   #debugging: check dataset format

ppo_trainer = PPOTrainer(
    ppo_config,
    model=policy,
    ref_model=None,
    reward_model=reward_model,
    train_dataset=dataset,
    data_collator=data_collator,
    value_model=value_model,
    processing_class=DummyProcessing,
)

# -------------------------------
# Training loop
# -------------------------------
for epoch in range(1):
    for i in tqdm(range(len(dataset)), desc="🌀 PPO Loop"):
        example = dataset[i]
        query = query_texts[i]
        inputs = tokenizer(query, return_tensors="pt", padding=True).to(DEVICE)

        # print("📦 input_ids:", inputs["input_ids"].device)
        # print("🧠 policy.device:", next(policy.parameters()).device)
        # print("📊 embed_tokens.device:", policy.base_model.model.embed_tokens.weight.device)

        response_ids = policy.generate(
            **inputs,
            max_new_tokens=50,
            do_sample=True,
            top_k=50
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        with torch.no_grad():
            reward = reward_model(
                input_ids=response_ids,
                attention_mask=(response_ids != tokenizer.pad_token_id).long()
            ).cpu().item()

        ppo_trainer.train()

        torch.save(policy.state_dict(), "./artifacts/qwen_ppo_policy.pt")

print("✅ PPO training loop complete.")