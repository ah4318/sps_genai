# === Base image ===
FROM python:3.12-slim-bookworm

# === Working directory ===
WORKDIR /code

# === Install system dependencies ===
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl && rm -rf /var/lib/apt/lists/*

# === Copy everything into the container ===
COPY . /code

# === Install Python dependencies ===
RUN pip install --no-cache-dir -r requirements.txt

# === Ensure Python can import helper_lib & app ===
ENV PYTHONPATH="/code"

# === Expose port ===
EXPOSE 80

# === Run FastAPI using uvicorn ===
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
