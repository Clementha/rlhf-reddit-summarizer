import os
import torch
import wandb
from torch.utils.data import DataLoader
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    get_scheduler
)
from datasets import Dataset, load_dataset
import peft
from tqdm import tqdm

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# -- Data Preprocessing Class
class SummarizationDataProcessor:
    def __init__(self, tokenizer, max_length=512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def format_prompt(self, post, title):
        return f"Title: {title}\n\nPost: {post}\n\nSummary:"

    def format_prompt_with_subreddit(self, post, title, subreddit):
        return f"Subreddit: r/{subreddit}\nTitle: {title}\n\nPost: {post}\n\nSummary:"

    def process_summarize_from_feedback(self, data):
        processed = []
        for item in data:
            info = item["info"]
            post = info["post"]
            title = info["title"]
            subreddit = info.get("subreddit", "")
            summaries = item["summaries"]
            choice = item["choice"]
            chosen_summary = summaries[choice]["text"]
            prompt = self.format_prompt_with_subreddit(post, title, subreddit) if subreddit else self.format_prompt(post, title)
            if chosen_summary.strip():
                processed.append({
                    "input_text": prompt,
                    "target_text": chosen_summary,
                    "full_text": prompt + " " + chosen_summary,
                    "post_id": info["id"],
                    "worker_id": item.get("worker", "unknown")
                })
        return processed

    def tokenize_function(self, examples):
        tokenized = self.tokenizer(
            examples["full_text"],
            truncation=True,
            max_length=self.max_length,
            padding="max_length"
        )
        labels = []
        for input_text, full_text, input_ids in zip(examples["input_text"], examples["full_text"], tokenized["input_ids"]):
            input_tokens = self.tokenizer(input_text, add_special_tokens=False)["input_ids"]
            input_length = len(input_tokens)
            label = [-100] * input_length + input_ids[input_length:]
            label = label[:self.max_length] + [-100] * (self.max_length - len(label))
            labels.append(label)
        tokenized["labels"] = labels
        return tokenized

def load_and_prepare_data(dataset, tokenizer):
    processor = SummarizationDataProcessor(tokenizer)
    raw_data = [item for item in dataset]
    processed_data = processor.process_summarize_from_feedback(raw_data)
    dataset = Dataset.from_list(processed_data)
    return dataset.map(processor.tokenize_function, batched=True, remove_columns=dataset.column_names)

# -- Load Dataset
ds_train = load_dataset("openai/summarize_from_feedback", "comparisons")
train_dataset = ds_train["train"]
val_dataset = ds_train["validation"]
num_epochs = 5

# -- Model and Tokenizer
model_name = "distilbert/distilgpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
model.resize_token_embeddings(len(tokenizer))

# -- Apply LoRA
lora_config = peft.LoraConfig(
    r=4,
    lora_alpha=8,
    lora_dropout=0.1,
    bias="none",
    task_type="CAUSAL_LM"
)
model = peft.get_peft_model(model, lora_config)

model.to(device)

# -- Prepare Data
processed_train_dataset = load_and_prepare_data(train_dataset, tokenizer)
processed_val_dataset = load_and_prepare_data(val_dataset, tokenizer)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,
    pad_to_multiple_of=8,
    return_tensors="pt"
)

train_dataloader = DataLoader(processed_train_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)
val_dataloader = DataLoader(processed_val_dataset, shuffle=True, batch_size=4, collate_fn=data_collator)

# -- Optimizer and LR Scheduler
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
num_training_steps = num_epochs * len(train_dataloader)
lr_scheduler = get_scheduler("linear", optimizer=optimizer, num_warmup_steps=500, num_training_steps=num_training_steps)

scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

wandb.init(
    project="summarization-distilgpt2",
    config={
        "model_name": model_name,
        "epochs": num_epochs,
        "batch_size": 4,
        "learning_rate": 5e-5,
        "lr_scheduler": "linear",
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.lora_alpha,
        "lora_dropout": lora_config.lora_dropout,
    }
)




# -- Training Loop
model.train()
for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/1")
    total_loss = 0.0
    progress_bar = tqdm(train_dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.autocast("cuda", enabled=(device.type == "cuda")):
            outputs = model(**batch)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        lr_scheduler.step()

        total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        progress_bar.set_description(f"Train Loss: {avg_loss:.4f}")
        wandb.log({"train/loss": avg_loss, "epoch": epoch, "step": step})  # ✨ wandb

# -- Save model for PPO
save_dir = "./trained_models/distilgpt2-sft-summarization"
os.makedirs(save_dir, exist_ok=True)
model.save_pretrained(save_dir)
tokenizer.save_pretrained(save_dir)

artifact = wandb.Artifact("distilgpt2-sft-model", type="model")
artifact.add_dir(save_dir)
wandb.log_artifact(artifact)
wandb.finish()

print(f"Model saved to {save_dir}")