import os
import shutil
import subprocess
import tempfile
from typing import Optional

import torch
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from nemo.collections.asr.models import ASRModel

APP_TITLE = "Parakeet v2 EN ASR API"
MODEL_ID_DEFAULT = "nvidia/parakeet-tdt-0.6b-v2"

app = FastAPI(title=APP_TITLE)

MODEL_ID = os.getenv("MODEL_ID", MODEL_ID_DEFAULT)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

asr_model: Optional[ASRModel] = None

def _run(cmd):
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr)

def to_wav_16k_mono(src_path, dst_path):
    _run([
        "ffmpeg", "-y",
        "-i", src_path,
        "-ac", "1",
        "-ar", "16000",
        "-vn",
        dst_path
    ])

@app.on_event("startup")
def load_model():
    global asr_model
    asr_model = ASRModel.from_pretrained(model_name=MODEL_ID)
    asr_model.to(DEVICE)
    asr_model.eval()

@app.get("/health")
def health():
    return {
        "status": "ok",
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available(),
        "model_id": MODEL_ID,
    }

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if asr_model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    filename = (file.filename or "").lower()
    if not (filename.endswith(".wav") or filename.endswith(".mp3")):
        raise HTTPException(status_code=400, detail="Only .wav or .mp3 files supported")

    with tempfile.TemporaryDirectory() as tmpdir:
        src_path = os.path.join(tmpdir, file.filename or "audio")
        wav_path = os.path.join(tmpdir, "audio.wav")

        with open(src_path, "wb") as f:
            shutil.copyfileobj(file.file, f)

        to_wav_16k_mono(src_path, wav_path)

        result = asr_model.transcribe([wav_path])
        text = result[0] if isinstance(result, list) else str(result)

        return JSONResponse({"text": text})
