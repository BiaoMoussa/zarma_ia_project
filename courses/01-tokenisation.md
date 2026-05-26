# Cours 1 — La Tokenisation

## Qu'est-ce que la tokenisation ?

Un modèle d'IA ne comprend pas les lettres ni les mots. Il ne comprend que des **nombres** (des vecteurs). La tokenisation est l'étape qui transforme du texte en nombres.

```
"Bonjour, comment ça va ?"
        ↓ tokenizer
[42, 168, 7, 391, 15, 8, 59]
```

Chaque nombre est un **token** — un morceau de texte. Le modèle reçoit ces nombres et apprend à prédire le token suivant.

---

## Les trois niveaux de découpage

Prenons la phrase zarma : **"A na zayo gana"** (Il a suivi le voleur)

| Niveau | Découpage | Tokens | Problème |
|---|---|---|---|
| **Caractère** | `A`, ` `, `n`, `a`, ` `, `z`, `a`, ... | 15 | Trop fin : perd le sens des mots, séquences très longues |
| **Mot** | `A`, `na`, `zayo`, `gana` | 4 | Si un mot est inconnu → `[UNK]`, perte d'information |
| **Sous-mot (BPE)** | `A`, `na`, `zay`, `o`, `gan`, `a` | 6 | Compromis idéal : mots fréquents entiers, mots rares décomposables |

---

## Pourquoi le sous-mot (BPE) est le standard

**BPE** = *Byte Pair Encoding*. L'algorithme part des caractères et **fusionne** progressivement les paires les plus fréquentes.

```
Étape 1 : 'g', 'a', 'n', 'a'  → chaque caractère est un token
Étape 2 : 'ga', 'n', 'a'      → 'g'+'a' apparaît souvent → 'ga'
Étape 3 : 'ga', 'na'          → 'n'+'a' aussi fréquent → 'na'
Étape finale : 'gana'         → 'ga'+'na' → un seul token !
```

### L'avantage clé

Si le modèle rencontre un mot nouveau comme *"zayokulu"* (tous les voleurs), il ne dira pas « je ne connais pas ce mot ». Il le découpera en `zay`, `o`, `kulu` — trois morceaux connus — et déduira le sens par composition.

---

## Tokenizer spécialisé vs tokenizer générique

| Tokenizer | Ratio tokens/mots | Efficacité | Exemple |
|---|---|---|---|
| **Spécialisé zarma** | ~1.3x | Excellente | `ŋwaari` → 1 token |
| **GPT-4 (multilingue)** | ~2-3x sur le zarma | Médiocre | `ŋwaari` → 2-3 tokens |
| **Mot par mot** | ~1x | Parfaite... | ...jusqu'au premier mot inconnu → `[UNK]` |

Un tokenizer spécialisé est **plus efficace** : moins de tokens pour dire la même chose = séquences plus courtes = entraînement plus rapide = meilleure qualité.

---

## Le format d'un tokenizer Hugging Face

```json
{
  "version": "1.0",
  "truncation": null,
  "padding": null,
  "added_tokens": [...],
  "normalizer": {...},
  "pre_tokenizer": {...},
  "post_processor": {...},
  "decoder": {...},
  "model": {
    "type": "BPE",
    "vocab": {
      "[PAD]": 0,
      "[UNK]": 1,
      "[CLS]": 2,
      "[SEP]": 3,
      ...
      "A": 35,
      "na": 140,
      "zayo": 7210,
      "gana": 365
    },
    "merges": [
      "g a",     → fusion 1 : ga
      "ga na",   → fusion 2 : gana
      ...
    ]
  }
}
```

### Les tokens spéciaux

| Token | Rôle |
|---|---|
| `[PAD]` | Remplissage (toutes les séquences d'un batch doivent avoir la même longueur) |
| `[UNK]` | Mot inconnu (ne devrait presque jamais apparaître avec BPE) |
| `[CLS]` | Début de phrase (classification) |
| `[SEP]` | Fin de phrase (séparation) |
| `[MASK]` | Token masqué (pour l'entraînement type BERT) |
| `[BOS]` / `[EOS]` | Beginning/End of Sequence |

---

## Ce qu'on a entraîné dans ce projet

| Propriété | Zarma | Français |
|---|---|---|
| Vocabulaire | 18 101 tokens | 16 000 tokens |
| Corpus | 217 330 phrases | 33 059 phrases |
| Ratio tokens/mots | ~1.4x | ~1.3x |
| Caractères spéciaux | `ŋ`, `ɲ`, `ã`, `õ`, `ẽ` | `é`, `à`, `è`, `ê`, `ë`... |
| Script | `scripts/train_tokenizer.py` | `scripts/train_tokenizer_fr.py` |
| Fichier | `zarma_corpus/tokenizer/bpe_zarma/tokenizer.json` | `zarma_corpus/tokenizer/bpe_francais/tokenizer.json` |

---

## Le BPE en action (exemple réel)

```
Phrase : "Boro fo go no kaŋ ga ŋwaari ŋwaa"

Tokens  : [CLS] [Boro] [fo] [go] [no] [kaŋ] [ga] [ŋwaari] [ŋwaa] [SEP]

→ 8 mots → 8 tokens de contenu + 2 tokens spéciaux = 10 tokens
→ Ratio : 1.0x sur les mots (parfait !)
→ 'ŋwaari' reste entier car il apparaît assez souvent
→ 'kaŋ' reste entier car c'est un mot grammatical très fréquent
```

```
Phrase : "A go ɲwaari ŋwaa"

Tokens  : [CLS] [A] [go] [ɲ] [waari] [ŋwaa] [SEP]

→ 'ɲwaari' est coupé en [ɲ] + [waari]
→ 'ɲ' est rare (850 occurrences dans 24M caractères)
→ La séquence 'ɲw' n'apparaît pas assez pour être fusionnée
→ Mais le modèle peut quand même comprendre le mot par composition
```

---

## Pour aller plus loin

- **WordPiece** (BERT) : variante de BPE qui choisit les fusions par vraisemblance plutôt que par fréquence
- **SentencePiece** (T5, Llama) : traite le texte comme un flux de bytes, gère toutes les langues sans pré-tokenisation
- **Unigram** (XLNet) : part d'un grand vocabulaire et élague les tokens les moins utiles
