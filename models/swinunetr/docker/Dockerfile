# Base image matches the training stack: PyTorch 2.5.1 + CUDA 12.1 + cuDNN 9.
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies first (cached unless requirements.txt changes).
# The evaluation runs with no network, so everything must be installed now.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the model weights into the image (no download at eval time).
RUN mkdir -p /opt/model
COPY model/model.pth /opt/model/model.pth

# Inference code last -- changes most often, keeps the layers above cached.
COPY predict.py .

# Synapse mounts /input (ro) and /output (rw); predict.py defaults to those.
ENTRYPOINT ["python", "predict.py"]
