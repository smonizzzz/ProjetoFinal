"""
Avaliacao completa do modelo com metricas de classificacao.

Converte os angulos de Cobb preditos em categorias clinicas:
  0 = Normal   (< 10 graus)
  1 = Leve     (10-25 graus)
  2 = Moderada (25-40 graus)
  3 = Grave    (> 40 graus)

Calcula: Accuracy, Precision, Recall, F1-Score, ROC-AUC
Gera curvas de treino e curva ROC.
"""

import sys
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)
from sklearn.preprocessing import label_binarize

sys.path.insert(0, str(Path(__file__).parent))
from data.dataset import build_dataloaders, IMAGE_SIZE
from models.hrnet import build_model

CKPT     = "results/scoliosis_hrnet/best.pth"
DATA_ROOT = "./data"
OUT_DIR  = Path("results/scoliosis_hrnet")
CLASSES  = ["Normal", "Leve", "Moderada", "Grave"]


def angle_to_class(angle):
    if angle < 10:  return 0
    if angle < 25:  return 1
    if angle < 40:  return 2
    return 3


def max_angle(angles):
    return float(np.max(angles))


@torch.no_grad()
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Carregar modelo
    model = build_model(arch="hrnet", num_outputs=3)
    ckpt  = torch.load(CKPT, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model"])
    model = model.to(device).eval()
    print(f"Modelo carregado: {CKPT}")

    # Carregar dados
    _, _, test_loader = build_dataloaders(DATA_ROOT, batch_size=8, num_workers=0, max_samples=1200)

    all_pred_angles  = []
    all_target_angles = []

    for batch in test_loader:
        images  = batch["image"].to(device)
        targets = batch["angles"].cpu().numpy()
        preds   = model(images).cpu().numpy()
        all_pred_angles.append(preds)
        all_target_angles.append(targets)

    pred_angles   = np.concatenate(all_pred_angles)   # (N, 3)
    target_angles = np.concatenate(all_target_angles) # (N, 3)

    # CMAE por regiao
    cmae_per = np.abs(pred_angles - target_angles).mean(axis=0)
    cmae_mean = cmae_per.mean()
    print(f"\nCMAE T1={cmae_per[0]:.2f} T2={cmae_per[1]:.2f} L={cmae_per[2]:.2f} | Media={cmae_mean:.2f}")

    # Converter para classes (usando o angulo maximo de cada amostra)
    pred_max   = pred_angles.max(axis=1)
    target_max = target_angles.max(axis=1)

    y_true = np.array([angle_to_class(a) for a in target_max])
    y_pred = np.array([angle_to_class(a) for a in pred_max])

    # Metricas de classificacao
    acc  = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    rec  = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1   = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n== Metricas de Classificacao ==")
    print(f"  Accuracy:  {acc*100:.1f}%")
    print(f"  Precision: {prec*100:.1f}%")
    print(f"  Recall:    {rec*100:.1f}%")
    print(f"  F1-Score:  {f1*100:.1f}%")
    print(f"\n{classification_report(y_true, y_pred, target_names=CLASSES, zero_division=0)}")

    # ROC-AUC (One-vs-Rest)
    classes_present = np.unique(y_true)
    y_true_bin = label_binarize(y_true, classes=[0, 1, 2, 3])

    # Score continuo: angulo predito normalizado por classe
    scores = np.zeros((len(pred_max), 4))
    for i, angle in enumerate(pred_max):
        scores[i, 0] = max(0, 10 - angle) / 10
        scores[i, 1] = max(0, 25 - abs(angle - 17.5)) / 17.5
        scores[i, 2] = max(0, 40 - abs(angle - 32.5)) / 32.5
        scores[i, 3] = max(0, angle - 40) / 40

    # Guardar metricas
    metrics = {
        "cmae_t1": float(cmae_per[0]),
        "cmae_t2": float(cmae_per[1]),
        "cmae_lumbar": float(cmae_per[2]),
        "cmae_mean": float(cmae_mean),
        "accuracy": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1_score": float(f1),
    }
    with open(OUT_DIR / "full_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    # ── GRAFICOS ──────────────────────────────────────────────────────────────

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Avaliacao do Modelo — Pipeline de Escoliose", fontsize=14, fontweight='bold')

    # 1. Curvas de treino
    history_path = OUT_DIR / "history.json"
    if history_path.exists():
        with open(history_path) as f:
            history = json.load(f)

        train_cmae = [e.get("cmae", 0) for e in history["train"]]
        val_cmae   = [e.get("cmae", 0) for e in history["val"]]
        epochs     = range(1, len(train_cmae) + 1)

        axes[0].plot(epochs, train_cmae, 'b-o', markersize=3, label="Treino")
        axes[0].plot(epochs, val_cmae,   'r-o', markersize=3, label="Validacao")
        axes[0].axhline(y=5, color='g', linestyle='--', alpha=0.7, label="Referencia 5 graus")
        axes[0].set_title("Curvas de Treino (CMAE por Epoch)", fontweight='bold')
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("CMAE (graus)")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_ylim(bottom=0)

    # 2. Curva ROC (binario: scoliose vs normal, angulo > 10)
    y_bin_true  = (target_max > 10).astype(int)
    y_bin_score = pred_max

    fpr, tpr, _ = roc_curve(y_bin_true, y_bin_score)
    try:
        auc = roc_auc_score(y_bin_true, y_bin_score)
    except:
        auc = 0.0

    axes[1].plot(fpr, tpr, 'b-', linewidth=2, label=f"ROC (AUC = {auc:.3f})")
    axes[1].plot([0, 1], [0, 1], 'k--', alpha=0.5, label="Aleatorio")
    axes[1].fill_between(fpr, tpr, alpha=0.1, color='blue')
    axes[1].set_title("Curva ROC — Scoliose vs Normal", fontweight='bold')
    axes[1].set_xlabel("Taxa de Falsos Positivos")
    axes[1].set_ylabel("Taxa de Verdadeiros Positivos")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    axes[1].set_xlim([0, 1])
    axes[1].set_ylim([0, 1])

    # 3. Metricas de classificacao (barras)
    metric_names  = ["Accuracy", "Precision", "Recall", "F1-Score"]
    metric_values = [acc * 100, prec * 100, rec * 100, f1 * 100]
    colors = ['#1a3a6b', '#2e6da4', '#4a9fd4', '#6bbfea']

    bars = axes[2].bar(metric_names, metric_values, color=colors, edgecolor='white', linewidth=1.5)
    axes[2].set_title("Metricas de Classificacao (%)", fontweight='bold')
    axes[2].set_ylabel("Valor (%)")
    axes[2].set_ylim([0, 110])
    axes[2].axhline(y=100, color='green', linestyle='--', alpha=0.3)
    for bar, val in zip(bars, metric_values):
        axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                    f"{val:.1f}%", ha='center', va='bottom', fontweight='bold', fontsize=11)
    axes[2].grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    out_path = OUT_DIR / "evaluation_plots.png"
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nGraficos guardados: {out_path}")

    # Guardar ROC AUC nas metricas
    metrics["roc_auc"] = float(auc)
    with open(OUT_DIR / "full_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nMetricas completas guardadas: {OUT_DIR / 'full_metrics.json'}")
    return metrics


if __name__ == "__main__":
    metrics = run_evaluation()
