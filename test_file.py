from rl_train_gpt2_format import build_prompt, sample_answer

# 1. Build a test prompt
test_prompt = build_prompt({
    "context": "GPT-2 is a transformer model developed by OpenAI.",
    "question": "What type of model is GPT-2?"
})

# 2. Generate answer using RL-trained model
full_text, answer_text, _ = sample_answer(test_prompt)

print("\n=== MODEL ANSWER ===")
print(answer_text)
