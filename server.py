"""
API REST — Pipeline de Escoliose
Endpoint: POST /analyze

Uso:
    uvicorn server:app --host 0.0.0.0 --port 8000
"""

import time
import sys
from pathlib import Path

import numpy as np
import torch
import cv2
import requests
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

sys.path.insert(0, str(Path(__file__).parent))
from models.hrnet import build_model
from data.dataset import IMAGE_SIZE

# ── Configuração ──────────────────────────────────────────────────────────────

CKPT_PATH = "results/scoliosis_hrnet/best.pth"

TRANSFORM = A.Compose([
    A.CLAHE(clip_limit=3.0, p=1.0),
    A.Resize(IMAGE_SIZE, IMAGE_SIZE),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2(),
])

# ── Carregar modelo uma vez no arranque ───────────────────────────────────────

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model  = build_model(arch="hrnet", num_outputs=3)
ckpt   = torch.load(CKPT_PATH, map_location=device, weights_only=False)
model.load_state_dict(ckpt["model"])
model  = model.to(device).eval()
print(f"[server] Modelo carregado em {device}")

# ── App ───────────────────────────────────────────────────────────────────────

app = FastAPI(title="Scoliosis Analysis API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["POST", "GET"],
    allow_headers=["*"],
)

# ── Schemas ───────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    estudoId: str
    imageUrl: str

class CobbAngles(BaseModel):
    thoracic_proximal: float
    thoracic_main: float
    lumbar: float

class AnalyzeResponse(BaseModel):
    estudoId: str
    cobb_angles: CobbAngles
    max_angle: float
    classification: str
    processing_time_ms: float

# ── Helpers ───────────────────────────────────────────────────────────────────

def classify(angle: float) -> str:
    if angle < 10:  return "NORMAL"
    if angle < 25:  return "LEVE"
    if angle < 40:  return "MODERADA"
    return "GRAVE"


def download_image(url: str) -> np.ndarray:
    try:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=400, detail=f"Erro ao descarregar imagem: {e}")

    arr = np.frombuffer(resp.content, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise HTTPException(status_code=400, detail="Não foi possível descodificar a imagem.")
    return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)


@torch.no_grad()
def run_inference(img: np.ndarray) -> list[float]:
    tensor = TRANSFORM(image=img)["image"].unsqueeze(0).to(device)
    return model(tensor)[0].cpu().tolist()

# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "device": str(device)}


@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(body: AnalyzeRequest):
    t0 = time.perf_counter()

    img    = download_image(body.imageUrl)
    angles = run_inference(img)

    t1         = time.perf_counter()
    max_angle  = max(angles)

    return AnalyzeResponse(
        estudoId   = body.estudoId,
        cobb_angles = CobbAngles(
            thoracic_proximal = round(angles[0], 2),
            thoracic_main     = round(angles[1], 2),
            lumbar            = round(angles[2], 2),
        ),
        max_angle          = round(max_angle, 2),
        classification     = classify(max_angle),
        processing_time_ms = round((t1 - t0) * 1000, 1),
    )
