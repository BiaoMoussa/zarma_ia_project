# Cours 3 — Le Pipeline Complet

## Vue d'ensemble

Voici le trajet complet d'une phrase française vers sa traduction en zarma, telle qu'elle est traitée par un modèle fine-tuné :

```
"Les enfants sont nus."
        │
        ▼
┌─ TOKENIZER FRANÇAIS ──────────────────────────┐
│ "Les enfants sont nus."                        │
│   ↓                                             │
│ [CLS] [Les] [enfants] [sont] [nus] [.] [SEP]  │
│   ↓                                             │
│ [2, 147, 89, 203, 1456, 19, 3]                │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─ EMBEDDING ────────────────────────────────────┐
│ Chaque token → un vecteur de 4096 dimensions    │
│                                                  │
│ [2]  → [0.12, -0.45, 0.78, ...] (4096 valeurs) │
│ [147]→ [0.34, 0.21, -0.09, ...]                │
│ [89] → [-0.56, 0.92, 0.11, ...]                │
│ ...                                              │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─ TRANSFORMER (32 couches) ─────────────────────┐
│                                                  │
│  Couche 1 : Self-Attention + Feed-Forward       │
│    • Chaque token regarde tous les autres       │
│    • "enfants" prête attention à "Les"          │
│                                                  │
│  Couche 2-31 : ...                              │
│                                                  │
│  Couche 32 :                                    │
│    • Les vecteurs encodent le sens complet      │
│    • vect[nus] contient l'info de toute la      │
│      phrase, pas juste le mot "nus"             │
│                                                  │
│  + LoRA : petites matrices adaptatrices          │
│    (ce sont elles qui contiennent le zarma)     │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─ TÊTE DE GÉNÉRATION (LM Head) ────────────────┐
│ Chaque position → distribution sur le vocab     │
│                                                  │
│ vect₁ → [P("Zankey")=0.8, P("Yee")=0.1, ...]  │
│          ↓ on prend le plus probable            │
│        "Zankey"                                 │
│                                                  │
│ Puis on recommence (autoregressive) :           │
│   [CLS] [Zankey] → prédit "go"                 │
│   [CLS] [Zankey] [go] → prédit "koonu"         │
│   ... jusqu'à prédire [SEP]                    │
└───────────────────┬─────────────────────────────┘
                    ▼
┌─ DETOKENIZER ZARMA ───────────────────────────┐
│ [CLS] [Zankey] [go] [koonu] [SEP]             │
│   ↓                                             │
│ "Zankey go koonu"                              │
└─────────────────────────────────────────────────┘
```

---

## Les composants clés

### 1. Le Transformer

C'est l'architecture inventée par Google en 2017 (*Attention Is All You Need*). Tous les LLMs modernes (GPT, Llama, Claude, Gemini) sont basés dessus.

Deux mécanismes principaux :

**Self-Attention** : Chaque token « regarde » tous les autres tokens de la phrase pour comprendre le contexte.

```
"Le chien a mangé la pomme parce qu'il avait faim"

Qui est "il" ? Le chien ou la pomme ?
→ L'attention permet au modèle de relier "il" à "chien"
   (même si 5 mots les séparent)
```

**Feed-Forward** : Une transformation non-linéaire appliquée à chaque token individuellement. C'est là que le « savoir » est stocké.

### 2. L'Embedding

Un embedding est un **vecteur dense** qui représente un token dans un espace sémantique.

```
Espace simplifié (2D au lieu de 4096D) :

          "femme" ●
                  │
        "reine" ● │  ● "roi"
                  │
                  │     ● "homme"
                  │
         "fille" ●│● "garçon"
                  │
   ───────────────┼─────────────────
                  │

Dans cet espace :
- "roi" - "homme" + "femme" ≈ "reine"
- Les mots similaires sont proches
- Les directions encodent des concepts
```

Au début de l'entraînement, ces vecteurs sont aléatoires. Le fine-tuning les ajuste pour que les paires de traduction (FR, Zarma) aient des représentations proches.

### 3. Le Tokenizer

Voir le [Cours 1 — Tokenisation](01-tokenisation.md).

### 4. LoRA/QLoRA

Voir le [Cours 2 — Fine-tuning et LoRA](02-fine-tuning-lora.md).

---

## Les formats de données

### Pour le fine-tuning d'un modèle de traduction

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un traducteur français → zarma."},
    {"role": "user", "content": "Traduis en zarma : Les enfants sont nus."},
    {"role": "assistant", "content": "Zankey go koonu"}
  ]
}
```

Dans ce projet, on utilise la variante **prompt / completion** de TRL
(`scripts/prepare_finetune_dataset.py`) : le prompt contient les messages
system + user, la completion le message assistant. SFTTrainer ne calcule alors
la perte que sur la traduction (`completion_only_loss`), et pas sur le prompt
système répété dans tous les exemples.

```json
{
  "prompt": [
    {"role": "system", "content": "Tu es un traducteur français → zarma."},
    {"role": "user", "content": "Traduis en zarma : Les enfants sont nus."}
  ],
  "completion": [{"role": "assistant", "content": "Zankey go koonu"}]
}
```

Ou en format plus simple :

```
<|im_start|>system
Tu es un traducteur français → zarma.<|im_end|>
<|im_start|>user
Traduis en zarma : Les enfants sont nus.<|im_end|>
<|im_start|>assistant
Zankey go koonu<|im_end|>
```

### Tokenisé en batch d'entraînement

```python
# Une batch de 4 exemples
input_ids = torch.tensor([
    [2, 147, 89, 203, 1456, 19, 3, 0, 0],    # pad à droite
    [2, 55, 782, 12, 3, 0, 0, 0, 0],
    [2, 334, 21, 921, 45, 557, 3, 0, 0],
    [2, 102, 67, 890, 12, 34, 78, 3, 0],
])  # shape: [4, 9]

attention_mask = torch.tensor([
    [1, 1, 1, 1, 1, 1, 1, 0, 0],    # 0 = padding, à ignorer
    [1, 1, 1, 1, 1, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 0, 0],
    [1, 1, 1, 1, 1, 1, 1, 1, 0],
])  # shape: [4, 9]

labels = torch.tensor([
    [2, 4985, 176, 1768, 19, 3, -100, -100, -100],  # -100 = ignorer (loss)
    [2, 1023, 445, 67, 3, -100, -100, -100, -100],
    ...
])
```

---

## Dépendances entre les étapes

```
Corpus brut (CSV, TXT)
    │
    ▼ scripts/clean_noisy_zarma.py, clean_bible_zarma.py, translate_en_fr.py
Corpus nettoyé (clean.text.dje.txt : 217,330 lignes, clean.text.fr.txt, fr_dje_aligned.csv,
                 bible_cleaned.csv, en_dje_translated_fr_dje.csv)
    │
    ▼ scripts/consolidate_corpus.py (+ scripts/normalize.py)
Splits train/validation/test normalisés, filtrés et sans fuite
(fr_dje_train.csv : 34,068 / fr_dje_validation.csv : 1,892 / fr_dje_test.csv : 1,892
 → 37,852 paires uniques ; voir « Leçons tirées du nettoyage » ci-dessous)
    │
    ▼ scripts/train_tokenizer.py, train_tokenizer_fr.py
Tokenizers (bpe_zarma/ : 32k vocab, bpe_francais/ : 16k vocab — pédagogiques,
            non utilisés par le fine-tuning de Qwen qui garde son propre tokenizer)
    │
    ▼ scripts/prepare_finetune_dataset.py → zarma_corpus/finetune/*.jsonl
    ▼ scripts/train_lora.py (QLoRA, reprise automatique depuis checkpoint)
Adaptateur LoRA FR ↔ Zarma
    │
    ▼ scripts/evaluate.py (chrF++ / BLEU, baseline vs fine-tuné)
Scores par direction
```

### Leçons tirées du nettoyage

- **Le corpus « 69 642 paires » était gonflé.** mt560 contient les mêmes versets
  que Feriji, mais tokenisés à la manière de Moses (`n 'ay`, `@-@`). Une fois
  l'orthographe normalisée, il ne reste qu'environ 38 000 phrases zarma distinctes.
- **Ces doublons créaient une fuite massive** : un verset pouvait être en train
  sous sa forme Moses et en test sous sa forme Feriji. Le split regroupe
  désormais les quasi-doublons (même français ou même zarma) dans un seul split.
- **Les versets à numérotation différente sont décalés** : quand le français de
  Feriji commence par « (55:16) », la phrase zarma correspond à un autre verset.
  Ces phrases zarma sont écartées dans toutes les sources.
- **Le corpus reste très majoritairement biblique** : les scores sur ce test ne
  disent rien de la qualité sur des phrases du quotidien. Il faut un jeu de test
  « réel » relu par des locuteurs natifs (`evaluate.py --test-csv`).

---

## Lexique technique

| Terme | Définition |
|---|---|
| **Token** | Unité de base du texte pour un modèle (mot, sous-mot, caractère) |
| **Vocabulaire** | Ensemble de tous les tokens connus par le tokenizer |
| **Embedding** | Vecteur dense représentant un token dans l'espace sémantique |
| **Transformer** | Architecture de réseau de neurones basée sur l'attention |
| **Self-Attention** | Mécanisme permettant à chaque token de « regarder » tous les autres |
| **Feed-Forward** | Couche de transformation appliquée indépendamment à chaque token |
| **Fine-tuning** | Spécialisation d'un modèle pré-entraîné sur une tâche spécifique |
| **LoRA** | Low-Rank Adaptation : fine-tuning économique via petites matrices |
| **QLoRA** | LoRA + quantification 4 bits du modèle de base |
| **PEFT** | Parameter-Efficient Fine-Tuning : famille de techniques dont LoRA |
| **Epoch** | Une passe complète sur l'ensemble des données d'entraînement |
| **Batch** | Groupe d'exemples traités en parallèle par le GPU |
| **Learning Rate** | Vitesse d'apprentissage : contrôle l'amplitude des mises à jour |
| **BLEU Score** | Métrique de qualité pour la traduction automatique |
| **Perplexité** | Mesure de « surprise » du modèle face à un texte (plus bas = meilleur) |
| **Inférence** | Utilisation du modèle entraîné pour générer du texte |
