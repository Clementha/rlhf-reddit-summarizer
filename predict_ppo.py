# predict.py

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import argparse
import os

# -------------------------------
# Config
# -------------------------------
MODEL_NAME = "Qwen/Qwen3-0.6B-Base"
CHECKPOINT_DIR = "./artifacts/qwen_loRA"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MAX_NEW_TOKENS = 128

# -------------------------------
# Load tokenizer and model
# -------------------------------
print("🔄 Loading tokenizer and model...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

model = PeftModel.from_pretrained(base_model, CHECKPOINT_DIR)
model.eval()
model = model.to(DEVICE)

# -------------------------------
# Generate Summary
# -------------------------------
def generate_summary(post: str) -> str:
    #prompt = f"Instruction: Summarize the post in one sentence.\n\nPost:\n{post}\n\nSummary:"
    # prompt = f"Please summarize the following Reddit post in 1–2 sentences:\n\n{post}\n\nSummary:"
    prompt = f"Instruction: Summarize the post in 1-3 sentences.\n\nPost:\n{post}\n\nSummary:"

    inputs = tokenizer(prompt, return_tensors="pt", padding=True).to(DEVICE)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            # top_k=50,
            # top_p=0.95,
            temperature=1.0,
            pad_token_id=tokenizer.pad_token_id,
            use_cache=True
        )

    full_output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    summary = full_output.split("Summary:")[-1].strip()
    return summary

# -------------------------------
# CLI
# -------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate summary with trained Qwen PPO model")
    parser.add_argument("--post", type=str, required=True, help="Content of the post")
    args = parser.parse_args()

    print("📝 Post:", args.post[:100] + ("..." if len(args.post) > 100 else ""))
    print("\n🤖 Generating summary...\n")

    summary = generate_summary(args.post)
    print("✅ Summary:\n", summary)