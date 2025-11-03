from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from app.bigram_model import BigramModel
from app.classifier import predict_image_bytes  # <-- new import
from helper_lib.model import get_model
from helper_lib.trainer import train_gan
from helper_lib.generator import generate_samples
import io
import base64
import matplotlib.pyplot as plt

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
@app.post("/gan/train")
async def train_gan_endpoint(epochs: int = 3):
    """
    Train GAN model on MNIST dataset.
    """
    model = get_model("gan")
    train_gan(model, epochs=epochs, device="cpu")
    return {"message": f"GAN trained for {epochs} epochs successfully."}


@app.get("/gan/generate")
async def generate_gan_samples(num_samples: int = 16):
    """
    Generate MNIST-like digits using the trained GAN.
    Returns a base64-encoded image.
    """
    model = get_model("gan")
    # Skip training for generation-only use (load pretrained if you want)
    fig, ax = plt.subplots()
    generate_samples(model, num_samples=num_samples)
    buf = io.BytesIO()
    plt.savefig(buf, format="png")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    buf.close()
    return {"image_base64": img_base64}

@app.get("/")
def home():
    return {"message": "API with Bigram + CNN Classifier"}
