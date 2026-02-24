# Parakeet v2 EN ASR API (Blackwell RTX 5090 Ready)

## Features
- NVIDIA Parakeet v2 English model
- MP3 + WAV upload
- CUDA 12.8 (Blackwell compatible)
- FastAPI
- Docker + Docker Compose
- GPU enabled

## Requirements
- NVIDIA Driver R570+
- NVIDIA Container Toolkit
- Docker + Docker Compose v2

## Build & Run

docker compose up -d --build

## Health Check
curl http://localhost:8000/health

## Transcribe
curl -s http://localhost:8000/transcribe -F "file=@audio.mp3"

## Notes
Model cache is persisted in ./cache so downloads happen once.
