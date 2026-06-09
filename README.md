# Pipeline Deep Learning — Análise de Exames de Escoliose

Projeto de Laboratório em Engenharia Informática — UTAD  
Aluno: Sérgio Miguel Pereira Moniz  
Orientador: António Cunha | Coorientador: António Gouveia

---

## Descrição

Pipeline completo de Deep Learning para análise automática de radiografias de escoliose:

- **Deteção automática de keypoints** vertebrais (68 pontos: 17 vértebras × 4 pontos)
- **Cálculo automático do ângulo de Cobb** (torácico e lombar)
- **Modelo principal:** HRNet-W32 (High-Resolution Network)
- **Dataset:** AASCE Challenge (609 radiografias anotadas)

---

## Estrutura do projeto

```
scoliosis_pipeline/
├── data/
│   └── dataset.py          ← Dataset PyTorch + augmentação + geração de heatmaps
├── models/
│   ├── hrnet.py            ← Arquiteturas HRNet-W32 e U-Net
│   └── losses.py           ← MSE heatmap + Wing Loss combinados
├── utils/
│   ├── metrics.py          ← SMAPE, PCK, CMAE, MRE + cálculo de Cobb
│   └── visualization.py    ← Visualização de KP, Cobb, curvas de treino
├── configs/
│   └── hrnet_w32.json      ← Configuração padrão (editável)
├── results/                ← Checkpoints e resultados (gerado automaticamente)
├── main.py                 ← Entry point (treino / teste / inferência)
├── train.py                ← Loop de treino com early stopping
└── requirements.txt
```

---

## Instalação

```bash
# 1. Criar ambiente virtual
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 2. Instalar dependências
pip install -r requirements.txt

# 3. (GPU) Instalar PyTorch com CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset AASCE

1. Registar em [MICCAI 2019 AASCE Challenge](http://spineweb.digitalimaginggroup.ca/)
2. Descarregar o dataset **boostnet_labeldata**
3. Colocar em:

```
data/
└── boostnet_labeldata/
    ├── data/       ← 001.jpg ... 609.jpg
    └── labels/     ← 001.npy ... 609.npy  (shape: 68×2)
```

---

## Utilização

### Treino completo

```bash
python main.py --mode train \
               --data_root ./data \
               --arch hrnet \
               --epochs 100 \
               --batch 8 \
               --lr 0.001 \
               --exp_name scoliosis_hrnet
```

Durante o treino são guardados automaticamente:
- `results/scoliosis_hrnet/best.pth`     ← melhor modelo (menor CMAE)
- `results/scoliosis_hrnet/last.pth`     ← último epoch (para retomar)
- `results/scoliosis_hrnet/history.json` ← histórico de métricas
- `results/scoliosis_hrnet/training_curves.png`

### Retomar treino interrompido

```bash
python main.py --mode train \
               --data_root ./data \
               --resume results/scoliosis_hrnet/last.pth
```

### Avaliação no conjunto de teste

```bash
python main.py --mode test \
               --data_root ./data \
               --ckpt results/scoliosis_hrnet/best.pth
```

### Inferência numa radiografia

```bash
python main.py --mode infer \
               --image caminho/para/radiografia.jpg \
               --ckpt results/scoliosis_hrnet/best.pth \
               --show
```

---

## Métricas

| Métrica | Descrição | Referência AASCE SOTA |
|---|---|---|
| **CMAE** | Cobb angle MAE (°) — métrica principal | ~5–7° |
| **MRE** | Mean Radial Error dos keypoints (px) | ~3–6 px |
| **SMAPE** | Symmetric MAPE dos keypoints (%) | <5% |
| **PCK@0.05** | % KP corretos (threshold=5% diagonal) | >90% |

### Interpretação clínica do ângulo de Cobb

| Ângulo | Severidade |
|---|---|
| < 10° | Normal |
| 10°–25° | Leve — observação periódica |
| 25°–40° | Moderada — colete ortopédico |
| > 40° | Grave — avaliação cirúrgica |

---

## Arquiteturas disponíveis

### HRNet-W32 (recomendado)
- Mantém representações de alta resolução em paralelo
- 4 branches: [32, 64, 128, 256] canais
- ~29M parâmetros
- Melhor precisão para keypoints vertebrais

### U-Net (prototipagem rápida)
- Encoder-decoder com skip connections
- ~31M parâmetros (com base_ch=64)
- Mais rápida de treinar, ligeiramente inferior em precisão

---

## Colab / GPU

Para treinar no Google Colab com GPU gratuita:

```python
# Clonar / montar o projeto
import sys
sys.path.insert(0, '/content/scoliosis_pipeline')

# Verificar GPU
import torch
print(torch.cuda.is_available())          # True se GPU disponível
print(torch.cuda.get_device_name(0))      # Ex: Tesla T4

# Treinar
!python main.py --mode train --data_root /content/data --epochs 50 --batch 4
```

---

## Referências

1. Wang, J., et al. "Deep High-Resolution Representation Learning for Visual Recognition." TPAMI, 2020.
2. Feng, Z-H., et al. "Wing Loss for Robust Facial Landmark Localisation with Convolutional Neural Networks." CVPR, 2018.
3. Wu, H., et al. "Automatic Landmark Estimation for Adolescent Idiopathic Scoliosis Assessment Using BoostNet." Medical Image Analysis, 2017.
4. MICCAI 2019 AASCE Challenge: Accurate Automated Spinal Curvature Estimation.
