# === Base image ===
FROM python:3.12-slim-bookworm

# === Install dependencies ===
WORKDIR /code

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# Copy dependency files
COPY requirements.txt /code/

# Install Python dependencies via pip (simpler than uv)
RUN pip install --no-cache-dir -r requirements.txt

# === Copy app files ===
COPY ./app /code/app
COPY ./models /code/models

# Expose the port
EXPOSE 80

# === Run FastAPI ===
CMD ["fastapi", "run", "app/main.py", "--port", "80"]
