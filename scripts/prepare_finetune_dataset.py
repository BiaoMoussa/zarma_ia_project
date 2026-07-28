"""
Étape 5 — Préparation du dataset de fine-tuning LoRA (format chat).

Convertit les paires FR↔Zarma consolidées (fr_dje_{train,validation,test}.csv)
en JSONL au format "messages" (system/user/assistant) attendu par TRL SFTTrainer
et le chat template de Qwen2.5-Instruct.

Chaque paire produit deux exemples : FR→Zarma et Zarma→FR (entraînement
bidirectionnel), avec des instructions variées pour éviter que le modèle
sur-apprenne une formulation unique.
"""

import csv
import json
import random
import sys
from pathlib import Path

random.seed(42)
csv.field_size_limit(sys.maxsize)

PROJECT = Path(__file__).resolve().parent.parent
PARALLEL = PROJECT / "zarma_corpus" / "cleaned" / "parallel"
OUTPUT = PROJECT / "zarma_corpus" / "finetune"

SYSTEM_PROMPT = (
    "Tu es un traducteur expert français ↔ zarma (Djerma), une langue parlée au Niger. "
    "Tu traduis fidèlement, en respectant le sens et le registre du texte source."
)

INSTRUCTIONS_FR_TO_DJE = [
    "Traduis en zarma : {text}",
    "Traduction en zarma : {text}",
    "Comment dit-on en zarma : « {text} » ?",
]

INSTRUCTIONS_DJE_TO_FR = [
    "Traduis en français : {text}",
    "Traduction en français : {text}",
    "Que signifie en français : « {text} » ?",
]

SPLITS = ["train", "validation", "test"]


def build_example(instruction_templates, source_text, target_text):
    instruction = random.choice(instruction_templates).format(text=source_text)
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
            {"role": "assistant", "content": target_text},
        ]
    }


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        csv_path = PARALLEL / f"fr_dje_{split}.csv"
        with open(csv_path, encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        examples = []
        for row in rows:
            fr, dje = row["fr"].strip(), row["dje"].strip()
            examples.append(build_example(INSTRUCTIONS_FR_TO_DJE, fr, dje))
            examples.append(build_example(INSTRUCTIONS_DJE_TO_FR, dje, fr))

        random.shuffle(examples)

        out_path = OUTPUT / f"{split}.jsonl"
        with open(out_path, "w", encoding="utf-8") as f:
            for ex in examples:
                f.write(json.dumps(ex, ensure_ascii=False) + "\n")

        print(f"{split}: {len(rows):,} paires -> {len(examples):,} exemples ({out_path})")


if __name__ == "__main__":
    main()
