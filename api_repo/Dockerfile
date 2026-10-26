# Hugging Face Spaces (Docker SDK) / any Docker host. CPU only.
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Spaces runs the container as user 1000
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface PYTHONUNBUFFERED=1
WORKDIR $HOME/app

# CPU-only torch first (much smaller than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the pretrained age model into the image so the first request is not slow
RUN python -c "from transformers import AutoImageProcessor, ViTForImageClassification as M; \
AutoImageProcessor.from_pretrained('nateraw/vit-age-classifier'); M.from_pretrained('nateraw/vit-age-classifier')"

COPY --chown=user . .

EXPOSE 7860
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
