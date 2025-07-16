# train_ppo_manual.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model, TaskType
from bitsandbytes.optim import Adam8bit

# -------------------------------
# Config
# -------------------------------

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
LR = 1e-5

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

# -------------------------------
# LoRA policy model
# -------------------------------

lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    inference_mode=False,
    r=8,
    lora_alpha=16,
    lora_dropout=0.05
)

base_policy = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
policy = get_peft_model(base_policy, lora_config)
policy.gradient_checkpointing_enable()
policy.to(DEVICE)

optimizer = Adam8bit(
    filter(lambda p: p.requires_grad, policy.parameters()),
    lr=LR
)

# -------------------------------
# Frozen reward model
# -------------------------------

class RewardModel(torch.nn.Module):
    def __init__(self, encoder_name):
        super().__init__()
        base = AutoModelForCausalLM.from_pretrained(encoder_name)
        self.encoder = get_peft_model(base, lora_config)
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
reward_model.load_state_dict(torch.load("./artifacts/reward_model_last_Qwen_loRA.pt"))
reward_model.eval()

# -------------------------------
# Manual PPO loop (paper-style)
# -------------------------------

prompts = [
    # "What are the benefits of LoRA for large language models?",
    # "What is the capital of France?",
    "My boyfriend and I are long distance. We have a trip planned this summer which involves me going over to him in the USA. This will be the second time I have actually been with him in person. I am flying from the UK with my mum to the east coast. The original plan was for me to fly over to my boyfriend in the west coast (my parents are holidaying on the east coast) but because my mum was freaking out so much about me going to meet my boyfriend i said we can all road trip there together. I even invited her on the trip with us. I have given her all of our dates so that she can travel around with us.\n\nThe plan was for me to stay on the 4th July and fly back on the 5th. Mum knew this. I told her I had booked a flight back already from the west coast to east coast (where she would pick me up and we would fly back to the UK together). She has gone mad at me because she can't believe I would book a flight when she told me she didn't want me flying on my own. At the time I had booked it she told me she wasn't gonna road trip with us. She knew the trip was happening.......how else was I to get home if I don't fly? \n\nI am fine flying on my own it doesn't bother me at all. I feel like I have done everything I can to make her feel comfortable with this trip and she is just trying to sabotage it."
]

for epoch in range(2):
    for prompt in prompts:
        input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(DEVICE)

        response_ids = policy.generate(
            input_ids=input_ids,
            # max_new_tokens=50, //was 50
            max_length=20,
            do_sample=True
        )
        response_text = tokenizer.decode(response_ids[0], skip_special_tokens=True)

        # Compute reward
        enc = tokenizer(response_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
        with torch.no_grad():
            reward = reward_model(
                enc["input_ids"].to(DEVICE),
                enc["attention_mask"].to(DEVICE)
            ).cpu().item()

        print(f"\nPrompt: {prompt}\nResponse: {response_text}\nReward: {reward:.4f}")

        # Compute fake advantage (for example only)
        advantage = torch.tensor([reward], requires_grad=True, device=DEVICE)

        # Compute log probs
        outputs = policy(
            input_ids=input_ids,
            labels=input_ids
        )
        loss = -advantage * outputs.loss  # maximize reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

torch.save(policy.state_dict(), "./artifacts/qwen_policy_lora_ppo_manual.pt")
print("✅ Manual PPO training complete!")