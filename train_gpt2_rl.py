from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

tokenizer = AutoTokenizer.from_pretrained("./gpt2-qa")
model = AutoModelForCausalLM.from_pretrained("./gpt2-qa").to(device)

tokenizer.pad_token = tokenizer.eos_token

def reward_fn(text):
    text = text.strip()
    return 1.0 if text.startswith("Answer:") else 0.0

def rl_train_step(model, tokenizer, prompt, optimizer, device="cpu"):
    model.train()

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate model output
    generated_ids = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.8,
    )
    text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    # Compute reward
    reward = reward_fn(text)

    # Compute loss for REINFORCE
    outputs = model(**inputs, labels=inputs["input_ids"])
    loss = outputs.loss
    rl_loss = -reward * loss

    optimizer.zero_grad()
    rl_loss.backward()
    optimizer.step()

    return text, reward, rl_loss.item()

optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for step in range(200):
    text, reward, loss = rl_train_step(
        model,
        tokenizer,
        prompt="Question: What is artificial intelligence?",
        optimizer=optimizer,
        device=device
    )

    print(f"Step {step} | reward={reward} | loss={loss}")
    print("Generated:", text)
    print("-" * 40)

tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
model = AutoModelForCausalLM.from_pretrained("./gpt2-qa/checkpoint-6000")


