"""
Étape 6 — Fine-tuning LoRA/QLoRA de Qwen2.5-7B-Instruct sur le corpus FR↔Zarma.

⚠️ La quantification 4 bits (bitsandbytes) nécessite un GPU CUDA : prévu pour
Google Colab, RunPod ou Lambda Labs — voir courses/02-fine-tuning-lora.md.
Sans GPU, le script charge le modèle en pleine précision (utile uniquement
pour un test de fumée avec un petit modèle, via --model et --no-4bit).

Installation sur l'environnement cloud :
    pip install -r requirements.txt -r requirements-train.txt

Exemples :
    # Run court pour valider le pipeline avant d'y mettre du budget GPU
    python scripts/train_lora.py --epochs 1 --train-fraction 0.1

    # Run complet, checkpoints sur Google Drive (reprise automatique si la
    # session Colab est coupée : relancer la même commande suffit)
    python scripts/train_lora.py --output-dir /content/drive/MyDrive/zarma_lora/qwen2.5-7b-zarma-fr

Entrée : zarma_corpus/finetune/{train,validation}.jsonl
         (générés par scripts/prepare_finetune_dataset.py, format prompt/completion)
Sortie : <output-dir>/ (adaptateur LoRA final, quelques dizaines de Mo)
         <output-dir>/checkpoint-*/ (checkpoints intermédiaires pour la reprise)
"""

import argparse
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers.trainer_utils import get_last_checkpoint
from trl import SFTConfig, SFTTrainer

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_DATA = PROJECT / "zarma_corpus" / "finetune"
DEFAULT_OUTPUT = PROJECT / "lora_adapters" / "qwen2.5-7b-zarma-fr"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

# ── LoRA ──────────────────────────────────────────────────────────────────
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    p.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT,
                   help="Mettre un chemin Google Drive sur Colab pour survivre aux déconnexions.")
    p.add_argument("--epochs", type=float, default=3)
    p.add_argument("--train-fraction", type=float, default=1.0,
                   help="Part du train utilisée (ex. 0.1 pour un premier run court).")
    p.add_argument("--max-eval-samples", type=int, default=500,
                   help="Limite la validation pour qu'elle reste rapide.")
    p.add_argument("--max-seq-length", type=int, default=512)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--grad-accum", type=int, default=4)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--no-4bit", action="store_true", help="Désactive QLoRA (test sans GPU).")
    p.add_argument("--no-resume", action="store_true", help="Ignore les checkpoints existants.")
    return p.parse_args()


def pick_precision():
    """bf16 sur Ampere et plus récent (A100, L4…), fp16 sur T4/V100, fp32 sur CPU.

    Le T4 de Colab gratuit ne supporte pas le bf16 : bf16=True y lève une erreur.
    """
    if not torch.cuda.is_available():
        return torch.float32, {"bf16": False, "fp16": False}
    if torch.cuda.is_bf16_supported():
        return torch.bfloat16, {"bf16": True, "fp16": False}
    return torch.float16, {"bf16": False, "fp16": True}


def main():
    args = parse_args()
    compute_dtype, precision_flags = pick_precision()
    use_4bit = torch.cuda.is_available() and not args.no_4bit
    if not torch.cuda.is_available() and not args.no_4bit:
        print("⚠ Pas de GPU CUDA : QLoRA 4 bits désactivé (test de fumée uniquement).")
    print(f"Précision de calcul : {compute_dtype} | QLoRA 4 bits : {use_4bit}")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model_kwargs = {"device_map": "auto"} if torch.cuda.is_available() else {}
    if use_4bit:
        model_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
        )
    else:
        model_kwargs["dtype"] = compute_dtype

    print(f"Chargement du modèle {args.model}...")
    model = AutoModelForCausalLM.from_pretrained(args.model, **model_kwargs)
    if use_4bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)

    model = get_peft_model(model, LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    ))
    model.print_trainable_parameters()

    print("\nChargement du dataset...")
    dataset = load_dataset("json", data_files={
        "train": str(args.data_dir / "train.jsonl"),
        "validation": str(args.data_dir / "validation.jsonl"),
    })
    train_ds = dataset["train"]
    if args.train_fraction < 1.0:
        train_ds = train_ds.shuffle(seed=42).select(range(int(len(train_ds) * args.train_fraction)))
    eval_ds = dataset["validation"]
    if args.max_eval_samples and len(eval_ds) > args.max_eval_samples:
        eval_ds = eval_ds.shuffle(seed=42).select(range(args.max_eval_samples))
    print(f"  train: {len(train_ds):,} exemples | validation: {len(eval_ds):,} exemples")

    training_args = SFTConfig(
        output_dir=str(args.output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        logging_steps=20,
        # Checkpoints réguliers (et pas seulement en fin d'époque) pour pouvoir
        # reprendre après une coupure Colab sans perdre des heures de calcul.
        eval_strategy="steps",
        eval_steps=args.save_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=2,
        gradient_checkpointing=True,
        max_length=args.max_seq_length,
        packing=False,
        # Perte calculée uniquement sur la traduction (format prompt/completion).
        completion_only_loss=True,
        report_to="none",
        **precision_flags,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
    )

    resume_from = None
    if not args.no_resume and args.output_dir.is_dir():
        resume_from = get_last_checkpoint(str(args.output_dir))
        if resume_from:
            print(f"\n↻ Reprise depuis {resume_from}")

    print("\nEntraînement en cours...")
    trainer.train(resume_from_checkpoint=resume_from)

    print(f"\nSauvegarde de l'adaptateur LoRA dans {args.output_dir}...")
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(str(args.output_dir))
    print("✓ Terminé !")


if __name__ == "__main__":
    main()
