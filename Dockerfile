FROM nvidia/cuda:12.8.0-cudnn-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    ffmpeg \
    libsndfile1 \
    git \
    build-essential \
    cmake \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv

RUN pip3 install --upgrade pip && \
    pip3 install --index-url https://download.pytorch.org/whl/cu128 torch torchaudio

COPY requirements.txt /srv/requirements.txt
RUN pip3 install -r /srv/requirements.txt

COPY app /srv/app

EXPOSE 8000

ENV MODEL_ID="nvidia/parakeet-tdt-0.6b-v2"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
