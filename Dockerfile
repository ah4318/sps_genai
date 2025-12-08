# ---------------------------------------------------------
# 1. Base Python image
# ---------------------------------------------------------
FROM python:3.10-slim

# ---------------------------------------------------------
# 2. Set working directory
# ---------------------------------------------------------
WORKDIR /code

# ---------------------------------------------------------
# 3. Install dependencies
# ---------------------------------------------------------
COPY requirements.txt /code/
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------
# 4. Copy your application code
# (all files small enough for GitHub)
# ---------------------------------------------------------
COPY app /code/app
COPY helper_lib /code/helper_lib
COPY main.py /code/main.py

# ---------------------------------------------------------
# 5. IMPORTANT:
# Do NOT COPY local model folders (too large for GitHub)
# Instructor will manually mount or download model if needed.
# ---------------------------------------------------------
# COPY gpt2-qa-rl /code/gpt2-qa-rl    <-- REMOVE THIS
# COPY checkpoint-6000 /code/checkpoint-6000 <-- REMOVE THIS

# ---------------------------------------------------------
# 6. Start FastAPI server
# ---------------------------------------------------------
CMD ["fastapi", "run", "main.py", "--port", "80", "--host", "0.0.0.0"]
