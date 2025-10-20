from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.classifier import predict_image_bytes  # <-- new import

app = FastAPI(title="Bigram + CNN API")

# --- Bigram part ---
corpus = [
    "The Count of Monte Cristo is a novel written by Alexandre Dumas.",
    "this is another example sentence",
    "we are generating text based on bigram probabilities",
    "bigram models are simple but effective"
]
bigram_model = BigramModel(corpus)

class TextGenerationRequest(BaseModel):
    start_word: str
    length: int

@app.post("/generate")
def generate_text(req: TextGenerationRequest):
    return {"generated_text": bigram_model.generate_text(req.start_word, req.length)}

# --- CNN Classifier part ---
@app.post("/classify")
async def classify_image(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        result = predict_image_bytes(image_bytes)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/")
def home():
    return {"message": "API with Bigram + CNN Classifier"}
