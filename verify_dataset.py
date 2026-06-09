"""
Verificacao do dataset — conta imagens, deteta duplicados e valida CSV.
"""
import csv
from pathlib import Path
from collections import defaultdict

BASE      = Path("data/spinal_ai2024")
IMG_BASE  = BASE / "images"
SUBSETS   = ["Spinal-AI2024-subset1", "Spinal-AI2024-subset2",
             "Spinal-AI2024-subset3", "Spinal-AI2024-subset4",
             "Spinal-AI2024-subset5"]
TRAIN_CSV = BASE / "Cobb_train_gt.csv" / "Cobb_spinal-AI2024-train_gt.txt"
TEST_CSV  = BASE / "Cobb_test_gt.csv"  / "Spinal-AI2024-test_gt.txt"

print("=" * 60)
print("VERIFICACAO DO DATASET")
print("=" * 60)

# 1. Contar imagens por subset
print("\n[1] Imagens por subset:")
total_imgs = 0
subset_counts = {}
for subset in SUBSETS:
    path   = IMG_BASE / subset
    if path.exists():
        count = len(list(path.glob("*.jpg")))
        subset_counts[subset] = count
        total_imgs += count
        print(f"  {subset}: {count} imagens")
    else:
        print(f"  {subset}: PASTA NAO ENCONTRADA")
print(f"  Total: {total_imgs} imagens")

# 2. Verificar duplicados entre subsets
print("\n[2] Verificacao de duplicados entre subsets:")
all_files = defaultdict(list)
for subset in SUBSETS:
    path = IMG_BASE / subset
    if path.exists():
        for f in path.glob("*.jpg"):
            all_files[f.name].append(subset)

duplicates = {k: v for k, v in all_files.items() if len(v) > 1}
if duplicates:
    print(f"  ATENCAO: {len(duplicates)} nomes repetidos entre subsets diferentes")
    for name, subsets in list(duplicates.items())[:5]:
        print(f"    {name} aparece em: {subsets}")
    print(f"  (Isto e normal — cada subset tem imagens 000001-004000)")
else:
    print("  Sem duplicados entre subsets diferentes.")

# 3. Validar CSV de treino
print("\n[3] Validacao do CSV de treino:")
if TRAIN_CSV.exists():
    train_entries = []
    with open(TRAIN_CSV, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                train_entries.append(row[0].strip())
    print(f"  Entradas no CSV: {len(train_entries)}")

    # Verificar cobertura por subset
    train_subsets = SUBSETS[:4]
    for i, subset in enumerate(train_subsets):
        block     = train_entries[i * 4000 : (i + 1) * 4000]
        img_base  = IMG_BASE / subset
        found     = sum(1 for name in block if (img_base / name).exists())
        print(f"  {subset}: {found}/{len(block)} imagens com anotacao encontradas")
else:
    print(f"  ERRO: CSV nao encontrado em {TRAIN_CSV}")

# 4. Validar CSV de teste
print("\n[4] Validacao do CSV de teste:")
test_csv_found = None
test_csv_dir = BASE / "Cobb_test_gt.csv"
if test_csv_dir.exists():
    txts = list(test_csv_dir.glob("*.txt"))
    if txts:
        test_csv_found = txts[0]

if test_csv_found:
    test_entries = []
    with open(test_csv_found, newline="") as f:
        for row in csv.reader(f):
            if len(row) >= 4:
                test_entries.append(row[0].strip())
    print(f"  Entradas no CSV: {len(test_entries)}")
    img_base = IMG_BASE / "Spinal-AI2024-subset5"
    found    = sum(1 for name in test_entries if (img_base / name).exists())
    print(f"  Subset5: {found}/{len(test_entries)} imagens com anotacao encontradas")
else:
    print(f"  ERRO: CSV de teste nao encontrado")

print("\n" + "=" * 60)
print("RESUMO")
print("=" * 60)
print(f"  Treino + Validacao (subsets 1-4): ~{sum(subset_counts.get(s,0) for s in SUBSETS[:4])} imagens")
print(f"  Teste (subset 5):                 ~{subset_counts.get(SUBSETS[4], 0)} imagens")
print(f"  Total geral:                      ~{total_imgs} imagens")
print("=" * 60)
