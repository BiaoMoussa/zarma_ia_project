"""
Étape 4 — Consolidation du corpus parallèle FR↔Zarma.

Fusionne toutes les sources nettoyées en un corpus parallèle unifié :
- fr_dje_aligned.csv (Feriji, 33k paires)
- bible_cleaned.csv (Bible/Peace Corps/Feri Team, 6.3k paires)
- en_dje_translated_fr_dje.csv (EN traduit en FR, 60k paires)

Élimine les doublons, filtre les entrées trop courtes/longues,
et produit les splits train/validation/test finaux.
"""

import csv
import sys
from pathlib import Path
import random

random.seed(42)

csv.field_size_limit(sys.maxsize)

PROJECT = Path(__file__).resolve().parent.parent
CLEANED = PROJECT / "zarma_corpus" / "cleaned"

SOURCES = [
    CLEANED / "parallel" / "fr_dje_aligned.csv",
    CLEANED / "parallel" / "bible_cleaned.csv",
    CLEANED / "parallel" / "en_dje_translated_fr_dje.csv",
]

# Paramètres de qualité
MIN_CHARS_FR = 3
MIN_CHARS_DJE = 3
MAX_CHARS = 1_000

# Splits
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05   # → test gets the remaining 0.05

print("=" * 60)
print("CONSOLIDATION DU CORPUS PARALLÈLE FR↔ZARMA")
print("=" * 60)

# 1. Charger toutes les sources
all_pairs = []
stats = {}

for path in SOURCES:
    if not path.exists():
        print(f"\n⚠ {path.name} : introuvable, ignoré")
        continue

    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = [(r["fr"].strip(), r["dje"].strip()) for r in reader
                if r["fr"].strip() and r["dje"].strip()]

    stats[path.name] = len(rows)
    all_pairs.extend(rows)
    print(f"\n{path.name}: {len(rows):,} paires")

print(f"\n--- Total brut : {len(all_pairs):,} paires ---")

# 2. Filtrage qualité
filtered = []
removed_short = 0
removed_long = 0
for fr, dje in all_pairs:
    if len(fr) < MIN_CHARS_FR or len(dje) < MIN_CHARS_DJE:
        removed_short += 1
        continue
    if len(fr) > MAX_CHARS or len(dje) > MAX_CHARS:
        removed_long += 1
        continue
    filtered.append((fr, dje))

print(f"\nFiltrage qualité :")
print(f"  {removed_short:,} paires trop courtes (<{MIN_CHARS_FR} chars) supprimées")
print(f"  {removed_long:,} paires trop longues (>{MAX_CHARS} chars) supprimées")
print(f"  {len(filtered):,} paires conservées")

# 3. Dédoublonnage par le côté zarma (le plus discriminant)
seen_dje = set()
unique = []
dupes = 0
for fr, dje in filtered:
    if dje not in seen_dje:
        seen_dje.add(dje)
        unique.append((fr, dje))
    else:
        dupes += 1

print(f"\nDédoublonnage :")
print(f"  {dupes:,} doublons supprimés (basé sur le zarma)")
print(f"  {len(unique):,} paires uniques")

# 4. Mélanger et splitter
random.shuffle(unique)
n = len(unique)
n_train = int(n * TRAIN_RATIO)
n_valid = int(n * VALID_RATIO)

splits = {
    "train": unique[:n_train],
    "validation": unique[n_train:n_train + n_valid],
    "test": unique[n_train + n_valid:],
}

print(f"\nSplits :")
for name, data in splits.items():
    print(f"  {name}: {len(data):,} paires ({len(data)/n*100:.1f}%)")

# 5. Sauvegarde
for split_name, data in splits.items():
    out_path = CLEANED / "parallel" / f"fr_dje_{split_name}.csv"
    with open(out_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["fr", "dje"])
        writer.writeheader()
        for fr, dje in data:
            writer.writerow({"fr": fr, "dje": dje})
    print(f"  Sauvegardé : {out_path}")

# 6. Mettre à jour les fichiers monolingues pour refléter le corpus parallèle
print(f"\nMise à jour du corpus monolingue français...")
fr_mono_path = CLEANED / "monolingual" / "clean.text.fr.txt"
existing_fr = set()
if fr_mono_path.exists():
    with open(fr_mono_path, "r", encoding="utf-8") as f:
        existing_fr = set(line.strip() for line in f if line.strip())

# Ajouter le français des paires parallèles
all_fr = list(dict.fromkeys([fr for fr, dje in unique]))  # unique, preserve order
new_fr = [fr for fr in all_fr if fr not in existing_fr]
merged_fr = list(dict.fromkeys(list(existing_fr) + all_fr))

with open(fr_mono_path, "w", encoding="utf-8") as f:
    for sent in merged_fr:
        f.write(sent + "\n")
print(f"  clean.text.fr.txt : {len(existing_fr):,} → {len(merged_fr):,} phrases (+{len(new_fr):,})")

# 7. Stats globales
print(f"\n{'=' * 60}")
print(f"RÉSUMÉ FINAL")
print(f"{'=' * 60}")
print(f"Corpus parallèle FR↔ZMA : {len(unique):,} paires uniques")
print(f"Corpus monolingue Zarma : via clean_noisy_zarma.py")
print(f"Corpus monolingue Français : {len(merged_fr):,} phrases")
print(f"\nSources :")
for name, count in stats.items():
    print(f"  - {name}: {count:,} paires")
