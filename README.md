# SPS GenAI Assignment

## Project Overview
This project implements a **FastAPI application** with two models:
- **Bigram Model** for text generation  
- **Embedding Model** (spaCy `en_core_web_lg`) for word embeddings and similarity  
- **CNN Classifier (CIFAR10)** for image recognition  

The API is containerized using **Docker** for easy deployment and reproducibility.

---

## Features
- `/generate` → Generates text using the bigram model  
- `/embedding` → Returns embeddings for a given word  
- `/similarity` → Calculates similarity between two words  
- `/classify` → Classifies an uploaded image using a trained CNN model  
- `/` → Root endpoint with a welcome message  

---

## Setup & Installation

### Local Setup
1. Clone this repository:
   ```bash
   git clone git@github.com:ah4318/sps_genai.git
   cd sps_genai
