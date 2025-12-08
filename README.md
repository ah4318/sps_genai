🚀 SPS GenAI Multi-Model API

A unified FastAPI service integrating all models developed across Modules 4–8.

This project combines multiple deep learning models into a single, unified API service.
All models can be accessed through FastAPI endpoints, allowing flexible experimentation and interaction.

📌 Implemented Models
🔤 1. Bigram Text Generator

A character-level Bigram language model that predicts the next token based on frequency statistics.

Endpoint: POST /bigram/next

🧠 2. Word Embeddings & Similarity (spaCy)

Includes two features:

Convert any word into a vector representation

Compute cosine similarity between two words

Endpoints:

POST /embedding

POST /similarity

🖼️ 3. CIFAR-10 CNN Classifier

A convolutional neural network trained on CIFAR-10 to classify uploaded images.

Endpoint: POST /classify

🎨 4. GAN (Training + Sampling)

Implements a basic Generative Adversarial Network with:

A standalone training loop

Sampling using the trained generator

Endpoints:

POST /gan/train

GET /gan/generate

🌫️ 5. Diffusion Model (DDPM-style Sampling)

A simplified diffusion sampling process to generate images from noise.

Endpoint: GET /diffusion/generate

⚡ 6. Energy-Based Model (EBM)

Generates samples using Langevin dynamics based on an energy function.

Endpoint: GET /ebm/generate

📡 API Summary Table
Category	Method	Endpoint	Description
Bigram	POST	/bigram/next	Predict next token
Embedding	POST	/embedding	Get word vector
Similarity	POST	/similarity	Compute cosine similarity
Classification	POST	/classify	CIFAR-10 image classification
GAN	POST	/gan/train	Train GAN
GAN	GET	/gan/generate	Generate sample
Diffusion	GET	/diffusion/generate	DDPM sampling
EBM	GET	/ebm/generate	Sample via EBM
📁 Project Structure
sps_genai/
│
├── app/
│   └── main.py              # FastAPI application and API endpoints
│
├── bigram_model.py          # Bigram text generation model
├── classifier.py            # CIFAR-10 image classifier
│
├── helper_lib/              # Advanced modules from Modules 6–8
│   ├── model.py             # CNN, GAN, Diffusion, EBM models
│   ├── trainer.py           # Training loops
│   ├── generator.py         # Sampling utilities
│   ├── diffusion.py         # Diffusion sampling implementation
│   ├── ebm.py               # Energy-based model sampling
│   ├── utils.py             # Helper functions
│   └── data_loader.py       # Dataset loaders
│
├── models/                  # Saved model weights
│
├── requirements.txt
└── README.md                # (this file)

▶️ How to Run the API
1️⃣ Install Dependencies
pip install -r requirements.txt

2️⃣ Start the FastAPI Server
uvicorn app.main:app --reload

3️⃣ Open the Interactive API Documentation

Once the server is running, open:

👉 http://127.0.0.1:8000/docs

This automatically generates a full API interface via Swagger UI.

🧪 Example Outputs (Optional)

You may include sample output images from your models:

## Diffusion Sample
![Diffusion Sample](diffusion_output.png)

## EBM Sample
![EBM Sample](ebm_output.png)

🎓 Assignment Coverage

This project fully implements the requirements across Modules 4–8:

Module 4: CNN classifier

Module 5: VAE (integrated into helper_lib)

Module 6: GAN training + sampling

Module 7: FastAPI service exposing ML endpoints

Module 8: Diffusion and EBM samplers

🙌 Acknowledgements

Course: Columbia University — SPS Applied Machine Learning / Deep Learning
Instructor: Maryam Fazel-Zarandi
