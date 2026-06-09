import os
import json
import time
import copy
from pathlib import Path
from typing import Optional, Dict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts


DEFAULT_CONFIG = {
    "arch":         "hrnet",
    "num_outputs":  3,
    "image_size":   512,
    "batch_size":   8,
    "num_workers":  4,
    "max_samples":  600,
    "epochs":       100,
    "lr":           1e-3,
    "weight_decay": 1e-4,
    "patience":     15,
    "t0":           10,
    "t_mult":       2,
    "eta_min":      1e-6,
    "output_dir":   "results/",
    "exp_name":     "scoliosis_hrnet",
}


def cmae(pred, target):
    return float(torch.abs(pred - target).mean().item())


def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    losses, maes = [], []
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["angles"].to(device, non_blocking=True)

        preds = model(images)
        loss  = criterion(preds, targets)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()

        losses.append(loss.item())
        maes.append(torch.abs(preds.detach() - targets).mean().item())

    return {"loss": np.mean(losses), "cmae": np.mean(maes)}


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    losses, maes = [], []
    all_pred, all_target = [], []
    for batch in loader:
        images  = batch["image"].to(device, non_blocking=True)
        targets = batch["angles"].to(device, non_blocking=True)

        preds = model(images)
        loss  = criterion(preds, targets)

        losses.append(loss.item())
        maes.append(torch.abs(preds - targets).mean().item())
        all_pred.append(preds.cpu())
        all_target.append(targets.cpu())

    all_pred   = torch.cat(all_pred)
    all_target = torch.cat(all_target)
    per_angle  = torch.abs(all_pred - all_target).mean(dim=0)

    return {
        "loss":            np.mean(losses),
        "cmae":            np.mean(maes),
        "cmae_thoracic1":  per_angle[0].item(),
        "cmae_thoracic2":  per_angle[1].item(),
        "cmae_lumbar":     per_angle[2].item(),
    }


def train(model, train_loader, val_loader, cfg, resume_ckpt=None):
    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n[Treino] Dispositivo: {device}")
    model   = model.to(device)

    out_dir = Path(cfg["output_dir"]) / cfg["exp_name"]
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    optimizer = AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=cfg["t0"], T_mult=cfg["t_mult"], eta_min=cfg["eta_min"])
    criterion = nn.SmoothL1Loss()

    best_cmae    = float("inf")
    best_epoch   = 0
    best_weights = None
    patience_ctr = 0
    history      = {"train": [], "val": []}
    start_epoch  = 0

    if resume_ckpt and os.path.isfile(resume_ckpt):
        ckpt        = torch.load(resume_ckpt, map_location=device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        start_epoch = ckpt["epoch"] + 1
        best_cmae   = ckpt.get("best_cmae", float("inf"))
        history     = ckpt.get("history", history)
        print(f"[Treino] Retomando do epoch {start_epoch} | best CMAE: {best_cmae:.2f}°")

    for epoch in range(start_epoch, cfg["epochs"]):
        t0 = time.time()
        print(f"\nEpoch [{epoch+1}/{cfg['epochs']}] — lr={scheduler.get_last_lr()[0]:.2e}")

        train_m = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_m   = validate(model, val_loader, criterion, device)
        scheduler.step()

        history["train"].append(train_m)
        history["val"].append(val_m)

        print(f"  TREINO — loss={train_m['loss']:.4f} | CMAE={train_m['cmae']:.2f}deg")
        print(f"  VAL    — loss={val_m['loss']:.4f}   | CMAE={val_m['cmae']:.2f}deg "
              f"[T1={val_m['cmae_thoracic1']:.1f} T2={val_m['cmae_thoracic2']:.1f} L={val_m['cmae_lumbar']:.1f}] | {time.time()-t0:.1f}s")

        with open(out_dir / "history.json", "w") as f:
            json.dump(history, f, indent=2)

        torch.save({
            "epoch": epoch, "model": model.state_dict(),
            "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
            "best_cmae": best_cmae, "history": history, "cfg": cfg,
        }, out_dir / "last.pth")

        if val_m["cmae"] < best_cmae:
            best_cmae    = val_m["cmae"]
            best_epoch   = epoch + 1
            best_weights = copy.deepcopy(model.state_dict())
            patience_ctr = 0
            torch.save({"epoch": epoch, "model": best_weights, "cmae": best_cmae, "cfg": cfg},
                       out_dir / "best.pth")
            print(f"  ✓ Novo melhor modelo — CMAE={best_cmae:.2f}° (epoch {best_epoch})")
        else:
            patience_ctr += 1
            if patience_ctr >= cfg["patience"]:
                print(f"\n[Early stopping] Sem melhoria há {cfg['patience']} epochs.")
                break

    print(f"\n[Treino concluído] Melhor CMAE: {best_cmae:.2f}° (epoch {best_epoch})")
    if best_weights:
        model.load_state_dict(best_weights)
    return history


@torch.no_grad()
def evaluate_test(model, test_loader, cfg, ckpt_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model  = model.to(device)

    if ckpt_path:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model"])
        print(f"[Teste] Checkpoint: {ckpt_path}")

    criterion = nn.SmoothL1Loss()
    results   = validate(model, test_loader, criterion, device)

    print("\n── Resultados no conjunto de teste ──")
    print(f"  CMAE medio:          {results['cmae']:.2f} graus")
    print(f"  CMAE toracico (T1):  {results['cmae_thoracic1']:.2f} graus")
    print(f"  CMAE toracico (T2):  {results['cmae_thoracic2']:.2f} graus")
    print(f"  CMAE lombar:         {results['cmae_lumbar']:.2f} graus")
    return results
