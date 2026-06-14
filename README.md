# Pipeline Deep Learning — Análise de Exames de Escoliose

Projeto de Laboratório em Engenharia Informática — UTAD  
Aluno: Sérgio Miguel Pereira Moniz  
Orientador: António Cunha | Coorientador: António Gouveia

---

## Descrição

Pipeline completo de Deep Learning para análise automática de radiografias de escoliose:

- **Regressão direta dos ângulos de Cobb** (torácico proximal, torácico principal e lombar)
- **Classificação clínica automática** (Normal / Leve / Moderada / Grave)
- **Modelo principal:** HRNet-W32 (High-Resolution Network)
- **Dataset:** Spinal AI 2024 (20 000 radiografias anotadas)
- **API REST** para integração com aplicações web

---

## Estrutura do projeto

```
scoliosis_pipeline/
├── data/
│   └── dataset.py          ← Dataset PyTorch + augmentação de dados
├── models/
│   └── hrnet.py            ← Arquitetura HRNet-W32
├── configs/
│   └── hrnet_w32.json      ← Configuração do modelo
├── results/
│   └── scoliosis_hrnet/    ← Checkpoints e métricas
├── main.py                 ← Entry point (treino / teste / inferência)
├── train.py                ← Loop de treino com early stopping
├── evaluate.py             ← Avaliação completa com gráficos
├── server.py               ← API REST (FastAPI)
└── requirements.txt
```

---

## Instalação

```bash
# 1. Criar ambiente virtual
python -m venv venv
venv\Scripts\activate   # Windows

# 2. Instalar dependências
pip install -r requirements.txt

# 3. (GPU) Instalar PyTorch com CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

---

## Dataset — Spinal AI 2024

1. Descarregar o dataset em [Spinal AI 2024](https://spinalmetrics.com/)
2. Colocar em:

```
data/
└── spinal_ai2024/
    ├── images/
    │   ├── Spinal-AI2024-subset1/   ← treino
    │   ├── Spinal-AI2024-subset2/   ← treino
    │   ├── Spinal-AI2024-subset3/   ← treino
    │   ├── Spinal-AI2024-subset4/   ← treino
    │   └── Spinal-AI2024-subset5/   ← teste
    ├── Cobb_train_gt.csv/
    └── Cobb_test_gt.csv/
```

---

## Utilização

### Treino completo

```bash
python main.py --mode train \
               --data_root ./data \
               --epochs 100 \
               --batch 8 \
               --lr 0.001
```

Durante o treino são guardados automaticamente:
- `results/scoliosis_hrnet/best.pth`     ← melhor modelo (menor CMAE)
- `results/scoliosis_hrnet/last.pth`     ← último epoch (para retomar)
- `results/scoliosis_hrnet/history.json` ← histórico de métricas

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
               --ckpt results/scoliosis_hrnet/best.pth
```

### Servidor API REST

```bash
uvicorn server:app --host 0.0.0.0 --port 8000
```

Endpoint disponível:

```
POST /analyze
{
  "estudoId": "uuid-do-estudo",
  "imageUrl": "https://...signed-url..."
}
```

---

## Resultados

| Métrica | Valor |
|---|---|
| **CMAE médio** | **2.83°** |
| CMAE Torácico proximal (T1) | 2.62° |
| CMAE Torácico principal (T2) | 2.68° |
| CMAE Lombar | 3.18° |

| Referência | CMAE |
|---|---|
| AASCE Challenge SOTA | ~5–7° |
| **Este modelo** | **2.83°** |

---

## Interpretação clínica do ângulo de Cobb

| Ângulo | Severidade | Indicação |
|---|---|---|
| < 10° | Normal | — |
| 10°–25° | Leve | Observação periódica |
| 25°–40° | Moderada | Colete ortopédico |
| > 40° | Grave | Avaliação cirúrgica |

---

## Arquitetura — HRNet-W32

- Mantém representações de alta resolução em paralelo durante todo o processamento
- 4 branches: [32, 64, 128, 256] canais
- ~29M parâmetros
- Cabeça de regressão: AdaptiveAvgPool → Linear(480, 256) → Linear(256, 3)

---

## Referências

1. Wang, J., et al. "Deep High-Resolution Representation Learning for Visual Recognition." TPAMI, 2020.
2. MICCAI 2019 AASCE Challenge: Accurate Automated Spinal Curvature Estimation.
3. Spinal AI 2024 Challenge Dataset.
