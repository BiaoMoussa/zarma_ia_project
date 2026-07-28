# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Building an AI assistant for **Zarma (Djerma)**, a Nigerien language spoken by several million people. The project aims to develop text understanding/generation, French↔Zarma translation, and eventually a voice assistant. This is both a learning project and a real initiative for linguistic inclusion in the Sahel.

## Technical Approach

- **Strategy**: Fine-tune existing open-source LLMs (Llama, Qwen, Mistral) rather than training from scratch
- **Method**: LoRA / QLoRA for parameter-efficient fine-tuning with limited GPU resources
- **Stack**: PyTorch, Hugging Face Transformers, PEFT, Tokenizers
- **GPU options**: Google Colab Pro, RunPod, Lambda Labs (budget-conscious cloud GPUs)

## Corpus Structure (`zarma_corpus/`)

| Directory | Content |
|---|---|
| `raw/text/` | Original CSVs from Hugging Face datasets (Feriji, Bible, english-zarma mt560, TTS metadata, etc.) |
| `raw/audio/` | WAV files for TTS (zarma-tts-dataset) |
| `cleaned/monolingual/` | 217,330 lines Zarma (`clean.text.dje.txt`) / 69,169 lines French (`clean.text.fr.txt`) |
| `cleaned/parallel/` | `fr_dje_aligned.csv` (Feriji, 33,059 pairs) + `bible_cleaned.csv` (6,300 pairs) + `en_dje_translated_fr_dje.csv` (EN→FR translated, 60,515 pairs) → consolidated into `fr_dje_{train,validation,test}.csv`, 69,642 unique pairs total |
| `tokenizer/` | JSON tokenizer configs for French and Zarma (BPE, retrained on enriched corpus) |
| `train/` | 62,677 lines Zarma + parallel French for training |
| `validation/` | 3,482 lines validation split |
| `test/` | 3,483 lines test split |

## Key External Resources

- **Feriji dataset (Hugging Face)**: `27Group/Feriji` — primary dataset with parallel sentences, monolingual text, glossary, pre-trained models
- **Feriji GitHub**: `27-GROUP/Feriji` — upstream repo with tooling
- **Other HF datasets**: `michsethowusu/english-zarma_sentence-pairs_mt560`, `27Group/noisy_zarma`, `27Group/Zarma_POS`, `abdouaziz/bible_zarma`, `birma091/zarma-tts-dataset`
- **Dictionaries**: bisharat.net/Zarma, denisnddo.free.fr, fr.glosbe.com/fr/dje

## Scripts

| Script | Purpose |
|---|---|
| `scripts/clean_noisy_zarma.py` | Extract clean Zarma from noisy_zarma dataset |
| `scripts/clean_bible_zarma.py` | Filter and deduplicate bible_zarma parallel pairs |
| `scripts/translate_en_fr.py` | Translate English→French using Helsinki-NLP/opus-mt-en-fr |
| `scripts/consolidate_corpus.py` | Merge all sources into final train/val/test splits |
| `scripts/train_tokenizer.py` | Train BPE tokenizer for Zarma |
| `scripts/train_tokenizer_fr.py` | Train BPE tokenizer for French |

## Learning Resources (`cours/`)

The `cours/` directory contains structured lessons written during the project:
- `01-tokenisation.md` — BPE, vocabulary, special tokens, specialized vs generic tokenizers
- `02-fine-tuning-lora.md` — Fine-tuning, LoRA, QLoRA, Unsloth, model selection
- `03-pipeline-complet.md` — Full pipeline from text to trained model, technical glossary

These are designed to be re-read independently. Future Claude instances should reference and enrich these courses.

## Project Phase

Currently in **Phase 1 — Research & Preparation**: data cleaned, EN→FR translation of english-zarma complete, parallel corpus consolidated (69,642 unique pairs, up from ~32.6k), and tokenizers retrained on the enriched corpus. Next: fine-tuning setup (LoRA/QLoRA).
