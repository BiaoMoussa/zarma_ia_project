"""
Entraînement d'un tokenizer BPE pour le zarma (Djerma).

Utilise la bibliothèque tokenizers de Hugging Face pour créer un tokenizer
BPE (Byte-Pair Encoding) adapté au zarma.

Étapes :
1. Nettoyage : @-@ → -, normalisation Unicode NFC
2. Entraînement BPE sur le corpus monolingue
3. Sauvegarde au format Hugging Face compatible
4. Test sur des phrases exemples
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
import json

PROJECT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT / "zarma_corpus" / "cleaned" / "monolingual" / "clean.text.dje.txt"
OUTPUT = PROJECT / "zarma_corpus" / "tokenizer" / "bpe_zarma"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2  # un token doit apparaître au moins 2 fois

# ── 1. Nettoyer et charger le corpus ─────────────────────────────────────

print("Préparation du corpus...")
with open(CORPUS, "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

# Nettoyage : @-@ → -
# C'est un artefact de segmentation des corpus bibliques/parallèles
cleaned_lines = [line.replace("@-@", "-") for line in raw_lines]

print(f"  {len(cleaned_lines):,} phrases chargées et nettoyées")

# ── 2. Créer le tokenizer BPE ────────────────────────────────────────────

# BPE (Byte-Pair Encoding) : part des bytes, fusionne les plus fréquents
tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Normalisation Unicode : NFC (forme canonique composée)
# ──> 'ã' = U+00E3 (NFC) plutôt que 'a' + '◌̃' = U+0061+U+0303 (NFD)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),
    normalizers.Replace("@-@", "-"),  # double sécurité
    normalizers.StripAccents(),        # élimine les accents résiduels non-zarma
])

# Pré-tokenisation : sépare la ponctuation mais garde les lettres ensemble
# 'ŋ' et 'ɲ' sont traités comme des lettres normales grâce à \w
tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
    pre_tokenizers.Digits(individual_digits=False),  # "123" reste "123"
    pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True),
])

# Ajuster le pré-tokenizer pour ne PAS découper avec ByteLevel
# (ByteLevel encode en bytes, on veut rester en caractères lisibles)
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

tokenizer.decoder = decoders.ByteLevel()

# Entraîneur BPE
trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

print(f"\nEntraînement du tokenizer BPE...")
print(f"  Taille vocabulaire cible : {VOCAB_SIZE}")
print(f"  Fréquence minimale : {MIN_FREQUENCY}")

tokenizer.train_from_iterator(cleaned_lines, trainer, length=len(cleaned_lines))

vocab = tokenizer.get_vocab()
print(f"  Vocabulaire entraîné : {len(vocab):,} tokens")

# ── 3. Post-processing (templates pour CLS/SEP) ──────────────────────────

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

# ── 4. Sauvegarde ────────────────────────────────────────────────────────

OUTPUT.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(OUTPUT / "tokenizer.json"))
print(f"\nTokenizer sauvegardé dans {OUTPUT}/")

# ── 5. Test ──────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TESTS")
print("=" * 60)

test_sentences = [
    "A na zayo gana",
    "Irikoy na beena da ganda taka",
    "Zankey go koonu",
    "Ni mota go windo banda",
    "A ga koy wiciri kambu",
    "ŋwaari ga bori",  # avec 'ŋ'
    "Boro fo go no kaŋ ga ŋwaari ŋwaa",  # riche en 'ŋ'
    "Ay ma koy jeejay kwaara ra",  # phrase plus longue
    "Irikoy Biya mo goono ga yooje harey boŋ",
    "Woodin se no ay n'i bangandi ni se za doŋ",
]

for sent in test_sentences:
    output = tokenizer.encode(sent)
    print(f"\n  '{sent}'")
    print(f"  Tokens ({len(output.tokens)}): {output.tokens}")
    print(f"  IDs: {output.ids}")

# ── 6. Stats ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STATISTIQUES")
print("=" * 60)

vocab = tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

# Les premiers tokens (0-7 = special tokens)
print(f"\nTokens spéciaux :")
for token, tid in vocab_sorted[:7]:
    print(f"  [{tid}] {token!r}")

# Tokens les plus fréquents (IDs les plus bas après les spéciaux)
print(f"\nTokens les plus fréquents (IDs bas) :")
for token, tid in vocab_sorted[7:27]:
    print(f"  [{tid}] {token!r}")

# Distribution des longueurs de tokens
token_lens = [len(t) for t, _ in vocab_sorted]
avg_len = sum(token_lens) / len(token_lens)
print(f"\nLongueur moyenne des tokens : {avg_len:.1f} caractères")
print(f"Tokens d'1 caractère : {sum(1 for l in token_lens if l == 1)}")
print(f"Tokens de 2-3 caractères : {sum(1 for l in token_lens if 2 <= l <= 3)}")
print(f"Tokens de 4-6 caractères : {sum(1 for l in token_lens if 4 <= l <= 6)}")
print(f"Tokens de 7+ caractères : {sum(1 for l in token_lens if l >= 7)}")

print("\n✓ Terminé !")
