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
| `raw/text/` | Original CSVs from Hugging Face datasets (Feriji, Bible, TTS metadata, etc.) |
| `raw/audio/` | WAV files for TTS (zarma-tts-dataset) |
| `cleaned/monolingual/` | ~33k lines each of Zarma and French plain text |
| `cleaned/parallel/` | `fr_dje_aligned.csv` — 33,059 aligned French→Zarma sentence pairs |
| `tokenizer/` | JSON tokenizer configs for French and Zarma |
| `train/` | 26,120 lines Zarma + parallel French for training |
| `validation/` | Validation split |
| `test/` | Test split |

## Key External Resources

- **Feriji dataset (Hugging Face)**: `27Group/Feriji` — primary dataset with parallel sentences, monolingual text, glossary, pre-trained models
- **Feriji GitHub**: `27-GROUP/Feriji` — upstream repo with tooling
- **Other HF datasets**: `michsethowusu/english-zarma_sentence-pairs_mt560`, `27Group/noisy_zarma`, `27Group/Zarma_POS`, `abdouaziz/bible_zarma`, `birma091/zarma-tts-dataset`
- **Dictionaries**: bisharat.net/Zarma, denisnddo.free.fr, fr.glosbe.com/fr/dje

## Project Phase

Currently in **Phase 1 — Research & Preparation**: data cleaning, tokenizer building, dataset preparation. No training scripts or model code exist yet in this repo.
