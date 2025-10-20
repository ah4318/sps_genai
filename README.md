# SPS GenAI Assignment

## Project Overview
This project implements a **FastAPI application** with two models:
- **Bigram Model** for text generation
- **Embedding Model** (spaCy `en_core_web_lg`) for word embeddings and similarity

The API is containerized using **Docker** for easy deployment and reproducibility.

---

## Features
- `/generate` → Generates text using the bigram model.
- `/embedding` → Returns embeddings for a given word.
- `/similarity` → Calculates similarity between two words.
- `/` → Root endpoint with a welcome message.

---

## Setup & Installation

### Local Setup
1. Clone this repository:
   ```bash
   git clone git@github.com:ah4318/sps_genai.git
   cd sps_genai

# FastAPI + CNN (CIFAR10) + Bigram

## Endpoints
- `GET /` — health message  
- `POST /generate` — text generation (bigram)  
- `POST /classify` — image file upload → CIFAR10 class prediction

## Setup (local)
```bash
uv sync
uv run python train_cifar10.py   # saves models/cnn_cifar10.pt
uv run fastapi dev app/main.py   # http://127.0.0.1:8000/docs
