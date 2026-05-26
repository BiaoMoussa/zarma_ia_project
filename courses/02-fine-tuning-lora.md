# Cours 2 — Fine-tuning et LoRA

## Le problème : partir de zéro est impossible

Entraîner un grand modèle de langue (LLM) from scratch nécessite :

- **Des milliards** de textes d'entraînement
- **Des centaines de GPU** pendant des semaines
- **Des millions de dollars** d'infrastructure

Notre projet a ~100 000 paires de phrases. C'est 10 000 fois trop peu.

---

## La solution : le fine-tuning

Au lieu de construire une maison, on **rénove une pièce**.

```
Modèle pré-entraîné (Llama, Qwen, Mistral...)
    → A déjà lu des milliards de textes
    → Connaît la grammaire, la syntaxe, le raisonnement
    → Connaît le français, l'anglais, des dizaines de langues
    → Ne connaît PAS le zarma
            ↓ fine-tuning sur nos 100k paires
Modèle spécialisé FR ↔ Zarma
    → A conservé toutes ses connaissances générales
    → A appris le vocabulaire et la grammaire zarma
    → Sait traduire entre les deux langues
```

### Analogie

C'est comme un chef cuisinier français qui apprend la cuisine nigérienne. Il connaît déjà les techniques de cuisine, les ustensiles, l'hygiène. Il lui suffit d'apprendre les ingrédients locaux et les recettes spécifiques. On ne lui réapprend pas à tenir un couteau.

---

## Pourquoi le fine-tuning complet est trop cher

Un modèle comme Llama-3-8B a **8 milliards de paramètres**. Chaque paramètre est un nombre à virgule flottante.

```
8 milliards de paramètres × 4 bytes (float32) = 32 Go
+ gradients (32 Go) + optimiseur Adam (32 Go) + activations (~16 Go)
= ~112 Go de VRAM nécessaires
```

Une RTX 4090 a 24 Go de VRAM. Un GPU A100 a 80 Go. Il faudrait un cluster de GPUs pour faire du fine-tuning complet.

---

## LoRA : l'astuce qui change tout

**LoRA** = *Low-Rank Adaptation* (Hu et al., 2021)

L'idée : au lieu de modifier les 8 milliards de paramètres, on **gèle** le modèle original et on ajoute de **petites matrices adaptatrices** à côté.

```
Poids original W (gelé, ne change pas) :
    [4096 × 4096] = 16 777 216 paramètres

Matrices LoRA A et B (entraînées) :
    A : [4096 × 32] = 131 072
    B : [32 × 4096] = 131 072
    Total : 262 144 paramètres

Ratio : 16M / 262k = 64x moins de paramètres à entraîner !
```

### Comment ça marche

```
Sortie normale :  h = W × x           (W est gelé)
Avec LoRA :       h = W × x + B × A × x
                         ↑            ↑
                    partie gelée   partie entraînée (LoRA)
```

Le modèle original reste intact. Les matrices A et B apprennent l'**adaptation** au zarma.

### Pourquoi "low-rank" ?

La matrice LoRA a un rang r = 32 (d'où le nom "low-rank"). C'est une décomposition qui capture l'essentiel de l'information avec très peu de paramètres. Comme une photo compressée en JPEG : vous ne voyez pas la différence, mais le fichier est 50x plus petit.

---

## QLoRA : encore plus économique

**QLoRA** = LoRA + Quantification 4 bits

| Aspect | LoRA | QLoRA |
|---|---|---|
| Précision du modèle de base | 16 bits (FP16) | 4 bits (NF4) |
| Mémoire pour Llama-8B | ~16 Go | ~6 Go |
| GPU nécessaire | RTX 4090 (24 Go) | RTX 3060 (12 Go) ou Colab gratuit |
| Qualité | Excellente | Quasi identique à LoRA |
| Vitesse | Rapide | ~30% plus lent |

La quantification 4 bits divise la mémoire par 4 avec une perte de qualité minime. C'est ce qui rend le fine-tuning accessible sur du matériel grand public.

---

## Le pipeline de fine-tuning

```
┌────────────────────────────────────────────┐
│  1. Préparer les données                    │
│  - Charger les paires FR ↔ Zarma           │
│  - Formater en prompts (chat template)      │
│  - Tokenizer avec les tokenizers spécialisés│
│  - Split train/validation/test              │
└─────────────────┬──────────────────────────┘
                  ▼
┌────────────────────────────────────────────┐
│  2. Configurer LoRA/QLoRA                   │
│  - Choisir le rang r (typiquement 8 à 64)  │
│  - Choisir les modules à adapter            │
│    (attention, feed-forward...)             │
│  - alpha = 2 × r (échelle d'apprentissage) │
└─────────────────┬──────────────────────────┘
                  ▼
┌────────────────────────────────────────────┐
│  3. Entraîner                               │
│  - 3 à 5 epochs (passes sur les données)   │
│  - Learning rate ~2e-4                     │
│  - Batch size adapté au GPU disponible      │
│  - Gradient checkpointing (économie mémoire)│
│  - Durée : 2-8h selon le GPU               │
└─────────────────┬──────────────────────────┘
                  ▼
┌────────────────────────────────────────────┐
│  4. Évaluer                                 │
│  - BLEU score sur le jeu de test           │
│  - Tests manuels (traductions réelles)      │
│  - Comparaison avant/après fine-tuning      │
└─────────────────┬──────────────────────────┘
                  ▼
┌────────────────────────────────────────────┐
│  5. Sauvegarder et déployer                 │
│  - Sauvegarder les poids LoRA (quelques Mo) │
│  - Fusionner avec le modèle de base (opt.)  │
│  - Inférence : charger modèle + adapter     │
└────────────────────────────────────────────┘
```

---

## Unsloth : l'accélérateur

[Unsloth](https://github.com/unslothai/unsloth) est une bibliothèque qui optimise l'entraînement QLoRA :

- **2x plus rapide** que l'implémentation standard Hugging Face
- **50% de mémoire en moins**
- Support natif de Llama, Mistral, Qwen
- Gratuit et open-source

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="Qwen/Qwen2.5-7B",
    max_seq_length=2048,
    load_in_4bit=True,  # QLoRA
)

model = FastLanguageModel.get_peft_model(
    model,
    r=16,               # rang LoRA
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_alpha=32,
    lora_dropout=0.05,
)
```

---

## Quel modèle de base choisir ?

| Modèle | Taille | Licence | Points forts | Points faibles |
|---|---|---|---|---|
| **Qwen-2.5-7B** | 7B | Apache 2.0 | Très multilingue, bon pour langues rares | Moins bon en français pur |
| **Llama-3.1-8B** | 8B | Llama | Excellent en français, mature | Licence restrictive |
| **Mistral-7B-v0.3** | 7B | Apache 2.0 | Européen, excellent français | Multilingue moyen |
| **Gemma-2-9B** | 9B | Gemma | Google, bonnes perfs | Licence restrictive |

**Recommandation pour ce projet : Qwen-2.5-7B**
- Licence permissive (pas de restrictions commerciales)
- Entraîné sur un corpus très diversifié linguistiquement
- S'adapte bien aux langues sous-représentées

---

## Coût estimé

| Plateforme | GPU | Coût/horaire | Temps estimé | Coût total |
|---|---|---|---|---|
| Google Colab Pro | T4/V100 | ~10 €/mois | 4-8h | ~10 € |
| RunPod | RTX 4090 | ~0.50 $/h | 2-4h | ~1-2 $ |
| Lambda Labs | A100 | ~1.10 $/h | 1-2h | ~2 $ |
| Kaggle | T4×2 | Gratuit | 3-6h | 0 € |

---

## Pour aller plus loin

- **Paper LoRA** : *LoRA: Low-Rank Adaptation of Large Language Models* (Hu et al., 2021)
- **Paper QLoRA** : *QLoRA: Efficient Finetuning of Quantized LLMs* (Dettmers et al., 2023)
- **Unsloth** : github.com/unslothai/unsloth
- **PEFT** (Hugging Face) : bibliothèque standard pour LoRA/QLoRA
