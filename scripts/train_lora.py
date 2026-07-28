"""
Étape 6 — Fine-tuning LoRA/QLoRA de Qwen2.5-7B-Instruct sur le corpus FR↔Zarma.

⚠️ Ce script nécessite un GPU CUDA (quantification 4 bits via bitsandbytes) :
il ne tourne PAS sur ce Mac (Apple M4, pas de CUDA). Prévu pour Google Colab,
RunPod ou Lambda Labs — voir courses/02-fine-tuning-lora.md pour les coûts
estimés par plateforme.

Installation sur l'environnement cloud :
    pip install -r requirements.txt -r requirements-train.txt

Lancement :
    python scripts/train_lora.py

Entrée : zarma_corpus/finetune/{train,validation}.jsonl
         (générés par scripts/prepare_finetune_dataset.py)
Sortie : lora_adapters/qwen2.5-7b-zarma-fr/ (poids LoRA, quelques dizaines de Mo)
"""

from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from trl import SFTConfig, SFTTrainer

PROJECT = Path(__file__).resolve().parent.parent
FINETUNE_DATA = PROJECT / "zarma_corpus" / "finetune"
OUTPUT_DIR = PROJECT / "lora_adapters" / "qwen2.5-7b-zarma-fr"

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
MAX_SEQ_LENGTH = 512  # cf. task #8 : à ajuster selon les stats de longueur réelles

# ── LoRA ──────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# ── Entraînement ──────────────────────────────────────────────────────────
NUM_EPOCHS = 3
LEARNING_RATE = 2e-4
PER_DEVICE_BATCH_SIZE = 4
GRAD_ACCUMULATION_STEPS = 4  # batch effectif = 4 × 4 = 16


def main():
    print(f"Chargement du tokenizer et du modèle {MODEL_NAME} (4-bit)...")

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        quantization_config=bnb_config,
        device_map="auto",
    )

    model = prepare_model_for_kbit_training(model)

    lora_config = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("\nChargement du dataset...")
    dataset = load_dataset(
        "json",
        data_files={
            "train": str(FINETUNE_DATA / "train.jsonl"),
            "validation": str(FINETUNE_DATA / "validation.jsonl"),
        },
    )
    print(f"  train: {len(dataset['train']):,} exemples")
    print(f"  validation: {len(dataset['validation']):,} exemples")

    training_args = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        per_device_eval_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=LEARNING_RATE,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        eval_strategy="epoch",
        save_strategy="epoch",
        bf16=True,
        gradient_checkpointing=True,
        max_length=MAX_SEQ_LENGTH,
        packing=False,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        processing_class=tokenizer,
    )

    print("\nEntraînement en cours...")
    trainer.train()

    print(f"\nSauvegarde de l'adaptateur LoRA dans {OUTPUT_DIR}...")
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    print("✓ Terminé !")


if __name__ == "__main__":
    main()
