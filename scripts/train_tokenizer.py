"""
Entraînement d'un tokenizer BPE pour le zarma (Djerma) — V2.

Contrairement à la V1 (ByteLevel), cette version utilise un BPE au niveau
caractère, ce qui préserve les lettres spécifiques au zarma (ŋ, ɲ, voyelles
nasalisées) comme des caractères atomiques plutôt que de les éclater en bytes.

Approche : BPE classique (comme GPT-2 mais sans ByteLevel)
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from tokenizers.pre_tokenizers import Whitespace, Digits, Punctuation, Sequence

PROJECT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT / "zarma_corpus" / "cleaned" / "monolingual" / "clean.text.dje.txt"
OUTPUT = PROJECT / "zarma_corpus" / "tokenizer" / "bpe_zarma"

VOCAB_SIZE = 32_000
MIN_FREQUENCY = 2

# ── 1. Nettoyer et charger ───────────────────────────────────────────────

print("Préparation du corpus...")
with open(CORPUS, "r", encoding="utf-8") as f:
    raw_lines = [line.strip() for line in f if line.strip()]

# Nettoyer @-@ → -
cleaned_lines = [line.replace("@-@", "-") for line in raw_lines]
print(f"  {len(cleaned_lines):,} phrases")

# ── 2. Tokenizer BPE niveau caractère ────────────────────────────────────

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

# Normalisation : NFC (préserve ŋ, ɲ, ã, õ, ẽ comme caractères uniques)
tokenizer.normalizer = normalizers.Sequence([
    normalizers.NFC(),
    normalizers.Replace("@-@", "-"),
])

# Pré-tokenisation niveau caractère :
# - Whitespace : sépare les mots
# - Punctuation : isole la ponctuation
# - Digits : garde les nombres entiers
# On split par mot, puis on isole la ponctuation.
# L'ordre est important : Whitespace d'abord, puis Ponctuation dans chaque mot.
tokenizer.pre_tokenizer = Sequence([
    Whitespace(),
    Digits(individual_digits=False),
])

tokenizer.decoder = decoders.BPEDecoder(suffix="")

trainer = trainers.BpeTrainer(
    vocab_size=VOCAB_SIZE,
    min_frequency=MIN_FREQUENCY,
    special_tokens=["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "[BOS]", "[EOS]"],
    show_progress=True,
)

print(f"\nEntraînement du tokenizer BPE (niveau caractère)...")
print(f"  Vocab cible : {VOCAB_SIZE}")
print(f"  Fréquence min : {MIN_FREQUENCY}")

tokenizer.train_from_iterator(cleaned_lines, trainer, length=len(cleaned_lines))

vocab = tokenizer.get_vocab()
print(f"  Vocabulaire final : {len(vocab):,} tokens")

# ── 3. Post-processing ───────────────────────────────────────────────────

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
print(f"\nTokenizer sauvegardé : {OUTPUT}/tokenizer.json")

# ── 5. Tests ─────────────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("TESTS")
print("=" * 60)

test_sentences = [
    "A na zayo gana",
    "Irikoy na beena da ganda taka",
    "Zankey go koonu",
    "Ni mota go windo banda",
    "A ga koy wiciri kambu",
    "ŋwaari ga bori",
    "Boro fo go no kaŋ ga ŋwaari ŋwaa",
    "Ay ma koy jeejay kwaara ra",
    "Irikoy Biya mo goono ga yooje harey boŋ",
    # Test des nasalisées
    "A kãa hẽn",
    # Test de ɲ (n crochet gauche, son "gn")
    "A go ɲwaari ŋwaa",
]

for sent in test_sentences:
    output = tokenizer.encode(sent)
    print(f"\n  '{sent}'")
    print(f"  Tokens ({len(output.tokens)}): {output.tokens}")
    print(f"  IDs: {output.ids}")

# ── 6. Statistiques ─────────────────────────────────────────────────────

print("\n" + "=" * 60)
print("STATISTIQUES")
print("=" * 60)

vocab = tokenizer.get_vocab()
vocab_sorted = sorted(vocab.items(), key=lambda x: x[1])

# Tokens spéciaux
print(f"\nTokens spéciaux :")
for token, tid in vocab_sorted[:7]:
    print(f"  [{tid}] {token!r}")

# Tokens les plus fréquents (hors ponctuation)
print(f"\nTokens les plus fréquents :")
count = 0
for token, tid in vocab_sorted[7:]:
    if token.isalpha() and len(token) >= 2:
        print(f"  [{tid}] {token!r}")
        count += 1
        if count >= 15:
            break

# Distribution
token_lens = [len(t) for t, _ in vocab_sorted]
avg_len = sum(token_lens) / len(token_lens)
print(f"\nLongueur moyenne des tokens : {avg_len:.1f} caractères")
print(f"Tokens d'1 caractère : {sum(1 for l in token_lens if l == 1)}")
print(f"Tokens de 2-3 caractères : {sum(1 for l in token_lens if 2 <= l <= 3)}")
print(f"Tokens de 4-6 caractères : {sum(1 for l in token_lens if 4 <= l <= 6)}")
print(f"Tokens de 7-10 caractères : {sum(1 for l in token_lens if 7 <= l <= 10)}")
print(f"Tokens de 11+ caractères : {sum(1 for l in token_lens if l >= 11)}")

# Vérification : ŋ, ɲ, ã, õ, ẽ sont-ils des tokens ?
for char in ['ŋ', 'ɲ', 'ã', 'õ', 'ẽ']:
    if char in vocab:
        print(f"  '{char}' est un token unique (ID {vocab[char]})")
    else:
        print(f"  '{char}' n'est PAS un token unique — il est décomposé")

print("\n✓ Terminé !")
