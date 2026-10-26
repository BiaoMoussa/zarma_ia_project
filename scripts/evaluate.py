"""
Étape 7 — Évaluation de la traduction FR↔Zarma (chrF++ et BLEU).

Génère les traductions d'un échantillon du split de test, puis calcule
chrF++ (métrique principale, plus fiable que BLEU pour une langue peu dotée
à l'orthographe variable) et BLEU avec sacrebleu, dans chaque direction.

Toujours mesurer une BASELINE (modèle sans adaptateur) avant d'interpréter
les scores d'un modèle fine-tuné.

Exemples :
    # Baseline : Qwen2.5-7B-Instruct sans fine-tuning
    python scripts/evaluate.py --max-samples 300

    # Modèle fine-tuné
    python scripts/evaluate.py --adapter lora_adapters/qwen2.5-7b-zarma-fr --max-samples 300

    # Jeu de test « réel » relu par des locuteurs natifs (CSV avec colonnes fr,dje)
    python scripts/evaluate.py --adapter ... --test-csv zarma_corpus/eval/quotidien.csv

Sorties (dans --output-dir, par défaut eval_results/<nom>/) :
    predictions.jsonl : source, référence, prédiction pour chaque phrase
    metrics.json      : scores par direction
"""

import argparse
import csv
import json
import random
import sys
from pathlib import Path

import sacrebleu
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_finetune_dataset import (  # noqa: E402
    INSTRUCTIONS_DJE_TO_FR,
    INSTRUCTIONS_FR_TO_DJE,
    SYSTEM_PROMPT,
)

PROJECT = Path(__file__).resolve().parent.parent
DEFAULT_TEST = PROJECT / "zarma_corpus" / "cleaned" / "parallel" / "fr_dje_test.csv"
DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"

DIRECTIONS = {
    "fr2dje": ("fr", "dje", INSTRUCTIONS_FR_TO_DJE[0]),
    "dje2fr": ("dje", "fr", INSTRUCTIONS_DJE_TO_FR[0]),
}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--adapter", type=Path, help="Dossier de l'adaptateur LoRA (absent = baseline).")
    p.add_argument("--test-csv", type=Path, default=DEFAULT_TEST)
    p.add_argument("--direction", choices=["both", *DIRECTIONS], default="both")
    p.add_argument("--max-samples", type=int, default=300,
                   help="Nombre de paires évaluées (0 = toutes).")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--no-4bit", action="store_true")
    p.add_argument("--output-dir", type=Path)
    return p.parse_args()


def load_pairs(path, max_samples):
    csv.field_size_limit(sys.maxsize)
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["fr"].strip() and r["dje"].strip()]
    if max_samples and len(rows) > max_samples:
        rows = random.Random(42).sample(rows, max_samples)
    return rows


def load_model(args):
    tokenizer = AutoTokenizer.from_pretrained(args.adapter or args.model)
    tokenizer.padding_side = "left"  # indispensable pour la génération par lots
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs = {}
    if torch.cuda.is_available():
        kwargs["device_map"] = "auto"
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
        if not args.no_4bit:
            kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=dtype)
        else:
            kwargs["dtype"] = dtype
    model = AutoModelForCausalLM.from_pretrained(args.model, **kwargs)

    if args.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(args.adapter))
    model.eval()
    return model, tokenizer


@torch.no_grad()
def translate(model, tokenizer, sources, template, batch_size, max_new_tokens):
    outputs = []
    for i in range(0, len(sources), batch_size):
        batch = sources[i:i + batch_size]
        prompts = [
            tokenizer.apply_chat_template(
                [{"role": "system", "content": SYSTEM_PROMPT},
                 {"role": "user", "content": template.format(text=src)}],
                tokenize=False, add_generation_prompt=True)
            for src in batch
        ]
        enc = tokenizer(prompts, return_tensors="pt", padding=True,
                        add_special_tokens=False).to(model.device)
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False,
                             pad_token_id=tokenizer.pad_token_id)
        new_tokens = gen[:, enc["input_ids"].shape[1]:]
        outputs.extend(t.strip() for t in tokenizer.batch_decode(new_tokens, skip_special_tokens=True))
        print(f"  {min(i + batch_size, len(sources))}/{len(sources)}", flush=True)
    return outputs


def main():
    args = parse_args()
    name = args.adapter.name if args.adapter else "baseline-" + args.model.split("/")[-1]
    out_dir = args.output_dir or PROJECT / "eval_results" / name
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_pairs(args.test_csv, args.max_samples)
    print(f"{len(rows)} paires de {args.test_csv.name} | modèle : {name}")
    model, tokenizer = load_model(args)

    directions = list(DIRECTIONS) if args.direction == "both" else [args.direction]
    metrics, records = {}, []
    for direction in directions:
        src_lang, tgt_lang, template = DIRECTIONS[direction]
        print(f"\nDirection {direction}...")
        sources = [r[src_lang] for r in rows]
        refs = [r[tgt_lang] for r in rows]
        hyps = translate(model, tokenizer, sources, template, args.batch_size, args.max_new_tokens)

        chrf = sacrebleu.corpus_chrf(hyps, [refs], word_order=2)  # chrF++
        bleu = sacrebleu.corpus_bleu(hyps, [refs])
        metrics[direction] = {"chrF++": round(chrf.score, 2), "BLEU": round(bleu.score, 2), "n": len(hyps)}
        for r, hyp in zip(rows, hyps):
            records.append({"direction": direction, "source": r[src_lang], "reference": r[tgt_lang],
                            "prediction": hyp, "corpus": r.get("source", "")})

    with open(out_dir / "predictions.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec, ensure_ascii=False) + "\n" for rec in records)
    with open(out_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump({"model": name, "test_csv": str(args.test_csv), "metrics": metrics}, f, indent=2)

    print(f"\nRésultats ({out_dir}) :")
    for direction, m in metrics.items():
        print(f"  {direction}: chrF++ = {m['chrF++']:.2f} | BLEU = {m['BLEU']:.2f} (n={m['n']})")


if __name__ == "__main__":
    main()
