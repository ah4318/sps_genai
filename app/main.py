from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# Load trained GPT-2 model
MODEL_PATH = "/code/gpt2-qa/checkpoint-6000"

model = AutoModelForCausalLM.from_pretrained("myusername/my-model")
tokenizer = AutoTokenizer.from_pretrained("myusername/my-model")

device = "mps" if torch.backends.mps.is_available() else "cpu"
model = model.to(device)

class QARequest(BaseModel):
    context: str
    question: str


def build_prompt(context, question):
    return f"Context: {context}\nQuestion: {question}\nAnswer:"


def generate_answer(prompt):
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    output = model.generate(
        **enc,
        max_new_tokens=40,
        pad_token_id=tokenizer.eos_token_id
    )
    full = tokenizer.decode(output[0], skip_special_tokens=True)
    answer = full.split("Answer:")[-1].strip()
    return answer


@app.post("/gpt2/answer")
def gpt2_answer(req: QARequest):
    prompt = build_prompt(req.context, req.question)
    answer = generate_answer(prompt)
    return {"answer": answer}

