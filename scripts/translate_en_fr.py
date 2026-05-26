"""
Étape 2 — Traduction EN→FR du dataset english-zarma.

Utilise Helsinki-NLP/opus-mt-en-fr, un modèle MarianMT spécialisé dans la
traduction anglais→français. Il tourne en local (CPU ou GPU selon disponibilité),
pas de limite de rate, pas de clé API.

Le modèle pèse ~300 Mo et sera téléchargé au premier lancement (caché ensuite).
"""

import csv
import sys
from pathlib import Path
import torch
from transformers import MarianMTModel, MarianTokenizer

PROJECT = Path(__file__).resolve().parent.parent
INPUT = PROJECT / "zarma_corpus" / "raw" / "text" / "english-zarma_sentence-pairs_mt560_train.csv"
OUTPUT = PROJECT / "zarma_corpus" / "cleaned" / "parallel" / "en_dje_translated_fr_dje.csv"

MODEL_NAME = "Helsinki-NLP/opus-mt-en-fr"
BATCH_SIZE = 32          # traduction par lots pour la vitesse
DEVICE = "cpu"  # CPU est plus stable pour la traduction par lots

csv.field_size_limit(sys.maxsize)

print(f"Device: {DEVICE}")
print(f"Chargement du modèle {MODEL_NAME}...")

tokenizer = MarianTokenizer.from_pretrained(MODEL_NAME)
model = MarianMTModel.from_pretrained(MODEL_NAME).to(DEVICE)

print("Chargement du dataset...")
with open(INPUT, "r", encoding="utf-8") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

english = [r["eng"].strip() for r in rows]
zarma = [r["dje"].strip() for r in rows]
total = len(english)
print(f"  {total:,} phrases à traduire")

french = []
errors = 0

print("Traduction en cours...", flush=True)
for i in range(0, total, BATCH_SIZE):
    batch = english[i : i + BATCH_SIZE]

    try:
        # Tokenize → générer → décoder
        inputs = tokenizer(batch, return_tensors="pt", padding=True, truncation=True, max_length=512).to(DEVICE)
        with torch.no_grad():
            translated = model.generate(**inputs, max_length=512)
        outputs = tokenizer.batch_decode(translated, skip_special_tokens=True)
        french.extend(outputs)
    except Exception as e:
        print(f"  Erreur lot {i//BATCH_SIZE}: {e}")
        french.extend([""] * len(batch))
        errors += len(batch)

    if (i + BATCH_SIZE) % 500 == 0 or i + BATCH_SIZE >= total:
        pct = min(i + BATCH_SIZE, total) / total * 100
        print(f"  {min(i + BATCH_SIZE, total):,}/{total:,} ({pct:.0f}%)")

print(f"\nTerminé !")
print(f"  {len(french):,} traductions")
print(f"  {errors} erreurs")

# Sauvegarde
OUTPUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT, "w", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["fr", "dje"])
    writer.writeheader()
    for fr, dje in zip(french, zarma):
        writer.writerow({"fr": fr, "dje": dje})

print(f"  Fichier : {OUTPUT}")

# Aperçu qualité
print("\nAperçu des traductions :")
for i in range(min(5, len(french))):
    print(f"  EN: {english[i][:100]}...")
    print(f"  FR: {french[i][:100]}...")
    print(f"  DJE: {zarma[i][:100]}...")
    print()
