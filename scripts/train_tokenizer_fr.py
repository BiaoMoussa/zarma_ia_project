"""
Entraînement d'un tokenizer BPE pour le français.

Même approche que le tokenizer zarma : BPE niveau caractère avec
normalisation NFC pour préserver les accents (é, à, è, ê, ë, etc.).
"""

from pathlib import Path
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors, normalizers
from tokenizers.pre_tokenizers import Whitespace, Digits, Sequence

PROJECT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT / "zarma_corpus" / "cleaned" / "monolingual" / "clean.text.fr.txt"
OUTPUT = PROJECT / "zarma_corpus" / "tokenizer" / "bpe_francais"

VOCAB_SIZE = 16_000  # corpus plus petit (33k lignes vs 217k), donc vocab réduit
MIN_FREQUENCY = 2

print("Chargement du corpus français...")
with open(CORPUS, "r", encoding="utf-8") as f:
    lines = [line.strip() for line in f if line.strip()]

print(f"  {len(lines):,} phrases")

tokenizer = Tokenizer(models.BPE(unk_token="[UNK]"))

tokenizer.normalizer = normalizers.Sequence([normalizers.NFC()])

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

print(f"\nEntraînement BPE français ({VOCAB_SIZE} tokens cible)...")
tokenizer.train_from_iterator(lines, trainer, length=len(lines))

vocab = tokenizer.get_vocab()
print(f"  Vocabulaire : {len(vocab):,} tokens")

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", tokenizer.token_to_id("[CLS]")),
        ("[SEP]", tokenizer.token_to_id("[SEP]")),
    ],
)

OUTPUT.mkdir(parents=True, exist_ok=True)
tokenizer.save(str(OUTPUT / "tokenizer.json"))
print(f"\nSauvegardé : {OUTPUT}/tokenizer.json")

# Tests
print("\n" + "=" * 60)
print("TESTS")
print("=" * 60)

tests = [
    "Au commencement, Dieu créa les cieux et la terre.",
    "La terre était informe et vide.",
    "Les enfants sont nus.",
    "Ta voiture est derrière la maison.",
    "Il a suivi le voleur.",
    "L'Éternel parla à Moïse.",
    "Je voudrais apprendre la langue zarma.",
]

for sent in tests:
    output = tokenizer.encode(sent)
    print(f"\n  '{sent}'")
    print(f"  Tokens ({len(output.tokens)}): {output.tokens}")

# Comparaison : combien de tokens pour les mêmes textes ?
print("\n" + "=" * 60)
print("COMPARAISON FRANÇAIS vs ZARMA")
print("=" * 60)

# Charger le tokenizer zarma
zarma_tok = Tokenizer.from_file(str(PROJECT / "zarma_corpus" / "tokenizer" / "bpe_zarma" / "tokenizer.json"))

comparisons = [
    ("Français", "Au commencement Dieu créa les cieux et la terre"),
    ("Zarma", "Sintina gaa Irikoy na beena da ganda taka"),
    ("Français", "Les enfants sont nus"),
    ("Zarma", "Zankey go koonu"),
    ("Français", "Ta voiture est derrière la maison"),
    ("Zarma", "Ni mota go windo banda"),
]

for lang, sent in comparisons:
    tok = tokenizer if lang == "Français" else zarma_tok
    output = tok.encode(sent)
    ratio = len(output.tokens) / len(sent.split())
    print(f"  [{lang}] {sent}")
    print(f"    {len(sent.split())} mots → {len(output.tokens)} tokens (ratio: {ratio:.1f}x)")
    print()

print("✓ Terminé !")
