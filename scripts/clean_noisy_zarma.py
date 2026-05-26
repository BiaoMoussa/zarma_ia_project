"""
Étape 1 — Extraction et nettoyage du zarma depuis noisy_zarma.

Ce que fait ce script :
1. Extrait la colonne 'cleaned' du CSV noisy_zarma
2. Supprime les outliers (phrases trop longues ou trop courtes)
3. Dédoublonne
4. Fusionne avec le corpus monolingue zarma existant
"""

import csv
import sys
from pathlib import Path

csv.field_size_limit(sys.maxsize)

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "zarma_corpus" / "raw" / "text" / "noisy_zarma_train.csv"
EXISTING = PROJECT / "zarma_corpus" / "cleaned" / "monolingual" / "clean.text.dje.txt"
OUTPUT = PROJECT / "zarma_corpus" / "cleaned" / "monolingual" / "clean.text.dje.txt"

# Seuils de filtrage (basés sur l'analyse des percentiles)
MIN_CHARS = 4        # élimine "To", "Ci", "Di", "»", etc.
MAX_CHARS = 1_000    # élimine la ligne corrompue de 37k caractères

# 1. Charger le zarma nettoyé depuis noisy_zarma
print("Chargement de noisy_zarma_train.csv...")
with open(RAW, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    raw_cleaned = [row["cleaned"].strip() for row in reader if row["cleaned"].strip()]

print(f"  {len(raw_cleaned):,} lignes brutes")

# 2. Filtrer par longueur
filtered = [s for s in raw_cleaned if MIN_CHARS <= len(s) <= MAX_CHARS]
removed_short = sum(1 for s in raw_cleaned if len(s) < MIN_CHARS)
removed_long = sum(1 for s in raw_cleaned if len(s) > MAX_CHARS)
print(f"  {removed_short:,} phrases trop courtes (<{MIN_CHARS} chars) supprimées")
print(f"  {removed_long:,} phrases trop longues (>{MAX_CHARS} chars) supprimées")
print(f"  {len(filtered):,} phrases conservées")

# 3. Dédoublonner
unique = list(dict.fromkeys(filtered))  # preserve order, remove duplicates
print(f"  {len(unique):,} phrases uniques ({(1 - len(unique)/len(filtered))*100:.1f}% de doublons)")

# 4. Charger l'existant et fusionner
if EXISTING.exists():
    with open(EXISTING, "r", encoding="utf-8") as f:
        existing = [line.strip() for line in f if line.strip()]
    print(f"\nCorpus existant : {len(existing):,} phrases")
else:
    existing = []
    print("\nAucun corpus existant trouvé, création d'un nouveau fichier.")

merged = list(dict.fromkeys(existing + unique))
new_added = len(merged) - len(existing)
print(f"Fusionné : {len(merged):,} phrases ({new_added:,} nouvelles)")

# 5. Sauvegarder
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8") as f:
    for sentence in merged:
        f.write(sentence + "\n")

print(f"\nFichier sauvegardé : {OUTPUT}")
