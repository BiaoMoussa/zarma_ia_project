"""
Télécharge tous les datasets bruts depuis Hugging Face.

Usage:
    python scripts/download_datasets.py          # tout télécharger
    python scripts/download_datasets.py --list   # lister les datasets
    python scripts/download_datasets.py --only noisy_zarma  # un seul dataset

Les données sont sauvegardées dans zarma_corpus/raw/ (ignoré par git).
"""

import argparse
import csv
import sys
from pathlib import Path

try:
    from datasets import load_dataset
except ImportError:
    print("Installe huggingface/datasets : pip install datasets")
    sys.exit(1)

PROJECT = Path(__file__).resolve().parent.parent
RAW = PROJECT / "zarma_corpus" / "raw"

DATASETS = {
    "noisy_zarma": {
        "name": "27Group/noisy_zarma",
        "split": "train",
        "output": RAW / "text" / "noisy_zarma_train.csv",
        "description": "Zarma monolingue (~400k phrases)",
    },
    "english_zarma": {
        "name": "michsethowusu/english-zarma_sentence-pairs_mt560",
        "split": "train",
        "output": RAW / "text" / "english-zarma_sentence-pairs_mt560_train.csv",
        "description": "Paires parallèles EN↔Zarma (56k)",
    },
    "bible_zarma": {
        "name": "abdouaziz/bible_zarma",
        "split": "train",
        "output": RAW / "text" / "bible_zarma_train.csv",
        "description": "Bible FR↔Zarma (~7k paires)",
    },
    "zarma_tts": {
        "name": "birma091/zarma-tts-dataset",
        "split": "train",
        "output": RAW / "audio" / "zarma-tts-dataset_train",
        "description": "Audio TTS zarma (~270 samples)",
    },
    "zarma_pos": {
        "name": "27Group/Zarma_POS",
        "split": "train",
        "output": RAW / "text" / "ZarmaLanguageRules_train.csv",
        "description": "Règles linguistiques / POS tagging",
    },
}


def download_csv(ds_name: str, info: dict) -> None:
    print(f"\n[{ds_name}] {info['description']}")
    print(f"  Source : {info['name']}")

    try:
        dataset = load_dataset(info["name"], split=info["split"])
    except Exception as e:
        print(f"  Erreur : {e}")
        return

    print(f"  {len(dataset):,} lignes chargées")

    info["output"].parent.mkdir(parents=True, exist_ok=True)

    if ds_name == "zarma_tts":
        # Dataset audio : sauvegarder les métadonnées et les fichiers audio
        audio_dir = info["output"]
        audio_dir.mkdir(parents=True, exist_ok=True)

        for i, row in enumerate(dataset):
            audio = row.get("audio", {})
            if audio and "array" in audio:
                import numpy as np
                import soundfile as sf

                audio_path = audio_dir / f"audio_{i:03d}.wav"
                sf.write(str(audio_path), audio["array"], audio.get("sampling_rate", 16000))

        # Sauvegarder les métadonnées en CSV
        meta_path = RAW / "audio" / "zarma-tts-dataset_train_metadata.csv"
        meta_rows = [{k: v for k, v in row.items() if k != "audio"} for row in dataset]
        if meta_rows:
            with open(meta_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=meta_rows[0].keys())
                writer.writeheader()
                writer.writerows(meta_rows)

        print(f"  {len(dataset)} fichiers audio sauvegardés dans {audio_dir}")
        print(f"  Métadonnées : {meta_path}")

    else:
        # Dataset texte : sauvegarder en CSV
        with open(info["output"], "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=dataset.column_names)
            writer.writeheader()
            for row in dataset:
                # Convertir chaque valeur en string pour éviter les erreurs de sérialisation
                clean_row = {k: str(v) if v is not None else "" for k, v in row.items()}
                writer.writerow(clean_row)

        print(f"  Sauvegardé : {info['output']} ({info['output'].stat().st_size / 1024 / 1024:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Télécharger les datasets Zarma")
    parser.add_argument("--list", action="store_true", help="Lister les datasets disponibles")
    parser.add_argument("--only", type=str, help="Télécharger un seul dataset (clé)")
    args = parser.parse_args()

    if args.list:
        print("Datasets disponibles :\n")
        for key, info in DATASETS.items():
            print(f"  {key:20s} → {info['description']}")
            print(f"  {'':20s}   {info['name']}")
            print()
        return

    if args.only:
        if args.only not in DATASETS:
            print(f"Dataset inconnu : {args.only}")
            print(f"Choix : {', '.join(DATASETS.keys())}")
            sys.exit(1)
        download_csv(args.only, DATASETS[args.only])
        return

    print("Téléchargement des datasets Zarma...")
    print(f"Dossier cible : {RAW}\n")

    for key, info in DATASETS.items():
        download_csv(key, info)

    print("\nTéléchargement terminé !")
    print(f"Les données sont dans {RAW}/ (ignoré par git)")


if __name__ == "__main__":
    main()
