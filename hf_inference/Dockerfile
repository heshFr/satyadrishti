FROM python:3.10-slim

# HF Spaces expects port 7860
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV HF_HUB_OFFLINE=0
ENV TRANSFORMERS_VERBOSITY=error

WORKDIR /app

# System deps for opencv, audio processing
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Install Python deps
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy project code
COPY engine/ engine/
COPY server/config.py server/config.py
COPY server/inference_engine.py server/inference_engine.py
COPY server/__init__.py server/__init__.py
COPY hf_inference/app.py .

# Copy model weights (these are baked into the Docker image)
# For HF Spaces, models go in /app/models/
# They can also be downloaded at startup via HF Hub
COPY models/ models/

EXPOSE 7860

CMD ["python", "app.py"]
