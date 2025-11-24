🌐 SPS GenAI Multi-Model API

A modular FastAPI application integrating multiple deep learning and generative models developed across SPS modules (4–8).
This project unifies text generation, embeddings, classification, GAN, VAE, Diffusion, and EBM sampling under a single API, fully containerized with Docker.

🚀 Features Overview
🔤 1. Bigram Text Generator

Generates text based on a simple character-level bigram language model.
Endpoint: POST /generate

🧠 2. Word Embedding / Similarity (spaCy)

Uses en_core_web_lg to compute word embeddings and cosine similarity.
Endpoints:

POST /embedding

POST /similarity

🖼️ 3. CNN Image Classifier (CIFAR10)

A Convolutional Neural Network trained on CIFAR-10 for image recognition.
Endpoint: POST /classify

🎨 4. GAN (Generative Adversarial Network)

Implements a generator + discriminator for generating synthetic images.
Endpoints:

POST /gan/train

GET /gan/generate

🌀 5. VAE (Variational Autoencoder)

Supports latent-space sampling + image reconstruction.
Endpoint: Available via helper library

🌫️ 6. Diffusion Model (Simplified DDPM Sampler)

A lightweight implementation of denoising diffusion sampling.
Endpoint:

GET /diffusion/generate

⚡ 7. Energy-Based Model (EBM)

Implements Langevin dynamics to iteratively reduce energy of images.
Endpoint:

GET /ebm/generate

🧩 API Endpoints Summary
Category	Method	Endpoint	Description
Text	POST	/generate	Generate text with Bigram model
Embedding	POST	/embedding	Get embedding for a word
Similarity	POST	/similarity	Compute similarity between two words
Classification	POST	/classify	Classify uploaded image
GAN	POST	/gan/train	Train GAN
GAN	GET	/gan/generate	Generate GAN samples
Diffusion	GET	/diffusion/generate	Generate images via diffusion sampling
EBM	GET	/ebm/generate	Generate images via EBM sampling
📁 Project Structure
sps_genai/
│
├── app/
│   ├── main.py                # FastAPI routes
│   ├── bigram_model.py
│   ├── classifier.py
│
├── helper_lib/
│   ├── model.py               # All model definitions (CNN, VAE, GAN, Diffusion, EBM)
│   ├── trainer.py             # Training loops
│   ├── generator.py           # Sampling utilities
│   ├── diffusion.py           # Diffusion sampler
│   ├── ebm.py                 # EBM sampler
│   ├── utils.py               # Helpers
│   ├── data_loader.py
│
├── models/                    # Saved checkpoints
├── requirements.txt
├── pyproject.toml
└── README.md

🛠 Installation
1️⃣ Clone the repository
git clone https://github.com/ah4318/sps_genai.git
cd sps_genai

2️⃣ Install dependencies

Using pip:

pip install -r requirements.txt


Or using uv:

uv sync

▶️ Run the API

Using uv:

uv run fastapi dev app/main.py


Or uvicorn:

uvicorn app.main:app --reload


Then open Swagger UI:
👉 http://127.0.0.1:8000/docs

🎨 Sample Outputs (Recommended)

You may drop two images into your project root:
diffusion_output.png, ebm_output.png
and they will automatically display in the README.

## 🖼 Diffusion Model Output
![Diffusion Output](diffusion_output.png)

## ⚡ EBM Sampling Output
![EBM Output](ebm_output.png)

🎓 Assignment Notes

This repository contains all components required for the SPS Generative AI assignments:

Module 4 – CNN

Module 5 – VAE

Module 6 – GAN + API integration

Module 7 – Deployment & multi-model API

Module 8 – Diffusion + EBM + API endpoints

Your implementation fulfills all required functionalities:
✔ FastAPI endpoints
✔ Generator + trainer integrations
✔ Multi-model support
✔ Clean project structure
✔ Docker deployment-ready

🙌 Acknowledgements

Developed for Columbia University SPS – Applied Machine Learning / Deep Learning Modules.