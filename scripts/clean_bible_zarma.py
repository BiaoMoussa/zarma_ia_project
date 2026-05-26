"""
Étape 3 — Nettoyage du dataset bible_zarma (FR↔DJE parallèle).

Ce que fait le script :
1. Supprime les paires mal alignées (ratio de longueur > 3x)
2. Supprime les doublons (par rapport au corpus parallèle existant)
3. Sauvegarde le résultat comme paires parallèles FR↔Zarma propres
"""

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "zarma_corpus" / "raw" / "text" / "bible_zarma_train.csv"
EXISTING = PROJECT / "zarma_corpus" / "cleaned" / "parallel" / "fr_dje_aligned.csv"
OUTPUT = PROJECT / "zarma_corpus" / "cleaned" / "parallel" / "bible_cleaned.csv"

MAX_LENGTH_RATIO = 3.0   # FR/DJE ou DJE/FR

print("Chargement...")
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

print(f"  {len(rows):,} paires brutes")

# Filtrer les paires mal alignées
clean = []
removed_misaligned = 0
for r in rows:
    fr = r["fr"].strip()
    dje = r["dje"].strip()
    if not fr or not dje:
        continue
    ratio = len(fr) / max(len(dje), 1)
    inv_ratio = len(dje) / max(len(fr), 1)
    if ratio > MAX_LENGTH_RATIO or inv_ratio > MAX_LENGTH_RATIO:
        removed_misaligned += 1
        continue
    clean.append((fr, dje))

print(f"  {removed_misaligned} paires mal alignées supprimées")
print(f"  {len(clean):,} paires conservées")

# Dédoublonner (en gardant l'ordre)
seen = set()
unique = []
for fr, dje in clean:
    key = dje  # dédoublonner par le côté zarma
    if key not in seen:
        seen.add(key)
        unique.append((fr, dje))

dupes = len(clean) - len(unique)
print(f"  {dupes:,} doublons internes supprimés")
print(f"  {len(unique):,} paires uniques")

# Vérifier contre le corpus parallèle existant
if EXISTING.exists():
    with open(EXISTING, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        existing_dje = set(r["dje"].strip() for r in reader)
    new_pairs = [(fr, dje) for fr, dje in unique if dje not in existing_dje]
    print(f"\n  {len(unique) - len(new_pairs):,} déjà dans le corpus parallèle existant")
    print(f"  {len(new_pairs):,} nouvelles paires")
else:
    new_pairs = unique

# Sauvegarde
with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fr", "dje"])
    writer.writeheader()
    for fr, dje in new_pairs:
        writer.writerow({"fr": fr, "dje": dje})

print(f"\nFichier sauvegardé : {OUTPUT}")

# Aperçu
print("\nAperçu :")
for fr, dje in new_pairs[:3]:
    print(f"  FR:  {fr[:100]}...")
    print(f"  DJE: {dje[:100]}...")
    print()
