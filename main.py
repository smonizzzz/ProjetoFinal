"""
Pipeline Deep Learning — Análise de Escoliose (regressão de ângulos de Cobb)

Uso (treino):
    python main.py --mode train --data_root ./data --epochs 100

Uso (teste):
    python main.py --mode test --data_root ./data --ckpt results/scoliosis_hrnet/best.pth

Uso (inferência numa imagem):
    python main.py --mode infer --image rx.jpg --ckpt results/scoliosis_hrnet/best.pth
"""

import argparse
import json
import sys
from pathlib import Path

import torch
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

sys.path.insert(0, str(Path(__file__).parent))

from data.dataset  import build_dataloaders, IMAGE_SIZE
from models.hrnet  import build_model
from train         import train, evaluate_test, DEFAULT_CONFIG


def severity(angle):
    if angle < 10:   return "Normal"
    if angle < 25:   return "Leve"
    if angle < 40:   return "Moderada"
    return "Grave"


@torch.no_grad()
def infer_single_image(image_path, ckpt_path, arch="hrnet", show=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = build_model(arch=arch, num_outputs=3)
    ckpt  = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    assert img is not None, f"Não foi possível ler: {image_path}"
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    transform = A.Compose([
        A.CLAHE(clip_limit=3.0, p=1.0),
        A.Resize(IMAGE_SIZE, IMAGE_SIZE),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ])
    tensor = transform(image=img)["image"].unsqueeze(0).to(device)

    angles = model(tensor)[0].cpu().tolist()
    labels = ["Torácico proximal", "Torácico principal", "Lombar"]

    print("\n── Ângulos de Cobb ──")
    for label, angle in zip(labels, angles):
        print(f"  {label}: {angle:.1f}° — {severity(angle)}")

    return angles


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mode",        choices=["train", "test", "infer"], default="train")
    p.add_argument("--data_root",   type=str,   default="./data")
    p.add_argument("--arch",        type=str,   default="hrnet")
    p.add_argument("--epochs",      type=int,   default=100)
    p.add_argument("--batch",       type=int,   default=8)
    p.add_argument("--lr",          type=float, default=1e-3)
    p.add_argument("--max_samples", type=int,   default=600)
    p.add_argument("--image",       type=str,   default=None)
    p.add_argument("--ckpt",        type=str,   default=None)
    p.add_argument("--exp_name",    type=str,   default="scoliosis_hrnet")
    p.add_argument("--resume",      type=str,   default=None)
    p.add_argument("--show",        action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    cfg  = {**DEFAULT_CONFIG,
            "arch": args.arch, "epochs": args.epochs,
            "batch_size": args.batch, "lr": args.lr,
            "exp_name": args.exp_name, "max_samples": args.max_samples}

    if args.mode == "train":
        print(f"\n== Modo: TREINO | {args.max_samples} imagens | {args.epochs} epochs ==")
        train_loader, val_loader, test_loader = build_dataloaders(
            args.data_root, cfg["batch_size"], cfg["num_workers"])
        model   = build_model(arch=cfg["arch"], num_outputs=cfg["num_outputs"])
        history = train(model, train_loader, val_loader, cfg, resume_ckpt=args.resume)
        best    = str(Path(cfg["output_dir"]) / cfg["exp_name"] / "best.pth")
        evaluate_test(model, test_loader, cfg, ckpt_path=best)

    elif args.mode == "test":
        assert args.ckpt, "Forneça --ckpt"
        _, _, test_loader = build_dataloaders(
            args.data_root, cfg["batch_size"], cfg["num_workers"])
        model   = build_model(arch=cfg["arch"], num_outputs=cfg["num_outputs"])
        results = evaluate_test(model, test_loader, cfg, ckpt_path=args.ckpt)
        out     = Path(cfg["output_dir"]) / cfg["exp_name"] / "test_results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            json.dump(results, f, indent=2)

    elif args.mode == "infer":
        assert args.image and args.ckpt, "Forneça --image e --ckpt"
        infer_single_image(args.image, args.ckpt, args.arch, args.show)


if __name__ == "__main__":
    main()
