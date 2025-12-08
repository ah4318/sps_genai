# train_gpt2_qa.py
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer

device = "mps" if torch.backends.mps.is_available() else "cpu"
print("Using device:", device)

# 1. Load tokenizer
tokenizer = AutoTokenizer.from_pretrained("openai-community/gpt2")
tokenizer.pad_token = tokenizer.eos_token

# 2. Format SQuAD into QA text
def format_squad(example):
    question = example["question"]
    answer = example["answers"]["text"][0] if example["answers"]["text"] else ""
    text = f"Question: {question}\nAnswer: {answer}"
    return {"text": text}

# 3. Load SQuAD + convert to text format
dataset = load_dataset("squad")["train"]
dataset = dataset.map(format_squad)

# 4. Tokenize text into GPT-2 inputs
def tokenize_function(example):
    encodings = tokenizer(
        example["text"],
        truncation=True,
        padding="max_length",
        max_length=128,
    )
    encodings["labels"] = encodings["input_ids"].copy()
    return encodings

tokenized_dataset = dataset.map(tokenize_function)

# 5. Load GPT-2 model
model = AutoModelForCausalLM.from_pretrained("openai-community/gpt2")
model.resize_token_embeddings(len(tokenizer))

# 6. Training configuration
training_args = TrainingArguments(
    output_dir="./gpt2-qa",
    per_device_train_batch_size=2,
    num_train_epochs=1,
    logging_steps=50,
    save_steps=500,
    use_mps_device=True,   # ★ ENABLE APPLE GPU ★
)

# 7. Trainer object
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

# 8. Train!
trainer.train()

# 9. Save model
trainer.save_model("./gpt2_qa")
tokenizer.save_pretrained("./gpt2_qa")
