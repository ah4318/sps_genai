# rl_train_gpt2_format.py
"""
Simple REINFORCE-style post-training for GPT-2 QA model to enforce a response format.

Pipeline:
1. Load tokenizer (base GPT-2) + fine-tuned QA model checkpoint.
2. Sample questions from a tiny QA set (you can swap in SQuAD or your own data).
3. Generate an answer with model.generate().
4. Compute a reward based on whether the answer follows the desired format.
5. Compute REINFORCE loss = reward * NLL(generated tokens) and backprop.
6. Save RL-tuned model to ./gpt2-qa-rl.
"""

import random
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from torch.optim import AdamW

# --------------------
# 1. Config
# --------------------
BASE_MODEL_NAME = "openai-community/gpt2"
FINETUNED_CKPT_PATH = "./gpt2-qa/checkpoint-6000"   # your fine-tuned QA checkpoint
OUTPUT_RL_PATH = "./gpt2-qa-rl"

NUM_STEPS = 500           # RL steps (you can reduce if it’s slow)
MAX_GEN_TOKENS = 64       # max new tokens to generate
LR = 1e-5                 # small LR for RL post-training
FORMAT_PREFIX = "Answer:" # required prefix for answers – change if your assignment uses a different format

# Tiny toy QA set so script is self-contained.
# For the assignment you can replace this with SQuAD samples if you want.
QA_DATA = [
    {
        "context": "Reinforcement learning is a branch of machine learning focused on agents "
                   "that learn to act by interacting with an environment and receiving rewards.",
        "question": "What is reinforcement learning about?"
    },
    {
        "context": "GPT-2 is a transformer-based language model that can generate coherent text.",
        "question": "What type of model is GPT-2?"
    },
    {
        "context": "FastAPI is a modern Python web framework for building APIs quickly.",
        "question": "What is FastAPI used for?"
    }
]


# --------------------
# 2. Device (MPS for your M4 Mac)
# --------------------
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

print("Using device:", device)


# --------------------
# 3. Load tokenizer + model
# --------------------
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained("./gpt2-qa/checkpoint-6000")
model.resize_token_embeddings(len(tokenizer))
model.to(device)

optimizer = AdamW(model.parameters(), lr=LR)


# --------------------
# 4. Reward function
# --------------------
def compute_reward(prompt, generated):
    reward = 0.0

    # ----------------------------------------
    # 1. Format Reward: Must begin with "Answer:"
    # ----------------------------------------
    if generated.strip().startswith("Answer:"):
        reward += 1.0
    else:
        reward -= 1.0

    # ----------------------------------------
    # 2. Length Penalty: Keep answer short (< 8 tokens)
    # ----------------------------------------
    num_tokens = len(generated.split())
    if 1 < num_tokens <= 8:
        reward += 0.5
    else:
        reward -= 0.5

    # ----------------------------------------
    # 3. Repetition Penalty: DO NOT repeat prompt/context
    # ----------------------------------------
    prompt_lower = prompt.lower()
    generated_lower = generated.lower()

    if any(chunk in generated_lower for chunk in prompt_lower.split()[:5]):
        reward -= 1.0   # strong penalty for copying prompt

    # ----------------------------------------
    # 4. Pure Answer Penalty: Must not contain other labels
    # ----------------------------------------
    forbidden = ["Context:", "Question:", "Generated full text:", "Answer:\nAnswer"]
    if any(bad in generated for bad in forbidden):
        reward -= 1.0

    # ----------------------------------------
    # 5. Avoid empty or trivial answers
    # ----------------------------------------
    if generated.strip() in ["Answer:", "", "Answer: "]:
        reward -= 1.0

    return reward



# --------------------
# 5. Helper: build prompt & generate answer
# --------------------
def build_prompt(example):
    """
    Turn a QA example into a prompt for the model.
    Adjust wording to match what you used during supervised fine-tuning.
    """
    context = example["context"]
    question = example["question"]
    prompt = (
        f"Context: {context}\n"
        f"Question: {question}\n"
        f"{FORMAT_PREFIX} "  # we already give the prefix to the model
    )
    return prompt


def sample_answer(prompt: str):
    """
    Generate an answer given a prompt, returning:
    - full generated text (prompt + completion)
    - only the newly generated answer text (after the prompt)
    """
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        padding=False
    ).to(device)

    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        gen_ids = model.generate(
    **inputs,
    max_new_tokens=20,
    do_sample=True,
    top_p=0.9,
    temperature=0.7,
    repetition_penalty=1.8,
    eos_token_id=tokenizer.eos_token_id,
)


    full_text = tokenizer.decode(gen_ids[0], skip_special_tokens=True)
    completion_ids = gen_ids[0][input_len:]
    completion_text = tokenizer.decode(completion_ids, skip_special_tokens=True)

    return full_text, completion_text, gen_ids


# --------------------
# 6. RL training step (REINFORCE-style)
# --------------------
def rl_step(example, step: int):
    model.train()

    # 1) Build prompt & sample answer
    prompt = build_prompt(example)
    full_text, answer_text, gen_ids = sample_answer(prompt)

    # 2) Compute reward correctly
    reward = compute_reward(prompt, answer_text)

    # 3) Compute NLL loss over generated tokens only
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids_prompt = inputs["input_ids"][0]

    full_ids = gen_ids.to(device)

    labels = full_ids.clone()
    labels[:, : len(input_ids_prompt)] = -100  # ignore prompt tokens

    outputs = model(input_ids=full_ids, labels=labels)
    nll_loss = outputs.loss

    # 4) REINFORCE update
    rl_loss = reward * nll_loss

    optimizer.zero_grad()
    rl_loss.backward()
    optimizer.step()

    # 5) Logging
    print(f"Step {step:04d} | reward={reward:.3f} | nll_loss={nll_loss.item():.4f} | rl_loss={rl_loss.item():.4f}")
    print("Prompt:")
    print(prompt)
    print("Generated full text:")
    print(full_text)
    print("-" * 60)

    return reward, nll_loss.item(), rl_loss.item()



# --------------------
# 7. Main training loop
# --------------------
def main():
    print("Starting RL fine-tuning...")
    reward_history = []

    for step in range(1, NUM_STEPS + 1):
        example = random.choice(QA_DATA)
        reward, nll, rl_loss = rl_step(example, step)
        reward_history.append(reward)

        # Print average reward every 50 steps
        if step % 50 == 0:
            avg_r = sum(reward_history[-50:]) / 50.0
            print(f"\n>>> Step {step}: average reward over last 50 steps = {avg_r:.3f}\n")

    # Save RL-tuned model
    print("Saving RL-tuned model to:", OUTPUT_RL_PATH)
    model.save_pretrained(OUTPUT_RL_PATH)
    tokenizer.save_pretrained(OUTPUT_RL_PATH)
    print("Done.")


if __name__ == "__main__":
    main()

# ============================
# TEST THE TRAINED MODEL
# ============================
print("\n=== TESTING TRAINED MODEL ===")

test_prompt = build_prompt({
    "context": "GPT-2 is a transformer model developed by OpenAI.",
    "question": "What type of model is GPT-2?"
})

full_text, answer_text, _ = sample_answer(test_prompt)
print("OUTPUT:", answer_text)
