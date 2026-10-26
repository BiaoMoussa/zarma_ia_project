"""
Étape 4 — Consolidation du corpus parallèle FR↔Zarma.

Fusionne toutes les sources nettoyées en un corpus parallèle unifié :
- fr_dje_aligned.csv (Feriji, 33k paires)
- bible_cleaned.csv (Bible/Peace Corps/Feri Team, 6.3k paires)
- en_dje_translated_fr_dje.csv (EN traduit en FR, 60k paires)

Étapes :
1. Normalisation orthographique (voir scripts/normalize.py).
2. Filtres qualité : longueur, versets décalés (dans toutes les sources),
   ratio de longueur.
3. Dédoublonnage sur le zarma normalisé (priorité au français humain).
4. Regroupement des quasi-doublons (même phrase française OU même phrase
   zarma, par transitivité) : un groupe va entièrement dans un seul split,
   ce qui évite les fuites train → validation/test.
5. Splits train/validation/test + fichiers texte alignés.
6. Corpus monolingues : on en retire les phrases de validation/test.

Le script est idempotent : il ne lit que les trois sources ci-dessus et
réécrit toutes ses sorties.
"""

import csv
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

from normalize import (
    dedup_key,
    has_verse_ref,
    normalize_dje,
    normalize_fr,
    plausible_ratio,
)

SEED = 42
csv.field_size_limit(sys.maxsize)

PROJECT = Path(__file__).resolve().parent.parent
CORPUS = PROJECT / "zarma_corpus"
CLEANED = CORPUS / "cleaned"

SOURCES = {
    "feriji": CLEANED / "parallel" / "fr_dje_aligned.csv",
    "bible": CLEANED / "parallel" / "bible_cleaned.csv",
    "mt560_en2fr": CLEANED / "parallel" / "en_dje_translated_fr_dje.csv",
}

# Paramètres de qualité
MIN_CHARS = 3
MAX_CHARS = 1_000

# Splits
TRAIN_RATIO = 0.90
VALID_RATIO = 0.05   # → test reçoit les 0.05 restants


class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, i):
        while self.parent[i] != i:
            self.parent[i] = self.parent[self.parent[i]]
            i = self.parent[i]
        return i

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def load_sources():
    pairs, stats = [], {}
    for name, path in SOURCES.items():
        if not path.exists():
            print(f"⚠ {path.name} : introuvable, ignoré")
            continue
        with open(path, encoding="utf-8") as f:
            rows = [(r["fr"].strip(), r["dje"].strip(), name)
                    for r in csv.DictReader(f)
                    if r["fr"].strip() and r["dje"].strip()]
        stats[name] = len(rows)
        pairs.extend(rows)
        print(f"{path.name}: {len(rows):,} paires")
    return pairs, stats


def filter_pairs(pairs):
    # Les lignes Feriji dont le français commence par « (55:16) » marquent un
    # écart de numérotation des versets : la phrase zarma correspondante est
    # décalée. Les mêmes phrases zarma se retrouvent dans mt560 (Bible anglaise,
    # même décalage) : on les écarte donc dans TOUTES les sources.
    misaligned_dje = {dedup_key(normalize_dje(dje)) for fr, dje, _ in pairs if has_verse_ref(fr)}

    removed = Counter()
    kept = []
    for fr, dje, src in pairs:
        fr, dje = normalize_fr(fr), normalize_dje(dje)
        if dedup_key(dje) in misaligned_dje:
            removed["verset décalé (numérotation différente)"] += 1
        elif len(fr) < MIN_CHARS or len(dje) < MIN_CHARS:
            removed["trop courte"] += 1
        elif len(fr) > MAX_CHARS or len(dje) > MAX_CHARS:
            removed["trop longue"] += 1
        elif not plausible_ratio(fr, dje):
            removed["ratio de longueur suspect"] += 1
        else:
            kept.append((fr, dje, src))
    return kept, removed


# Priorité en cas de doublon zarma : traduction française humaine d'abord.
SOURCE_PRIORITY = {"feriji": 0, "bible": 1, "mt560_en2fr": 2}


def deduplicate(pairs):
    """Une seule paire par phrase zarma (normalisée), en gardant la source la
    plus fiable : le français de Feriji/Bible est humain, celui de mt560 est
    traduit automatiquement de l'anglais."""
    best = {}
    for order, (fr, dje, src) in enumerate(pairs):
        key = dedup_key(dje)
        rank = (SOURCE_PRIORITY.get(src, 99), order)
        if key not in best or rank < best[key][0]:
            best[key] = (rank, (fr, dje, src))
    return [pair for _, pair in sorted(best.values())]


def group_split(pairs):
    """Répartit les groupes de quasi-doublons entre les splits."""
    uf = UnionFind(len(pairs))
    first_by_key = {}
    for i, (fr, dje, _) in enumerate(pairs):
        for key in (("fr", dedup_key(fr)), ("dje", dedup_key(dje))):
            if key in first_by_key:
                uf.union(first_by_key[key], i)
            else:
                first_by_key[key] = i

    groups = defaultdict(list)
    for i in range(len(pairs)):
        groups[uf.find(i)].append(pairs[i])
    group_list = sorted(groups.values(), key=lambda g: (g[0][1], g[0][0]))
    random.Random(SEED).shuffle(group_list)

    n = len(pairs)
    targets = {"test": n * (1 - TRAIN_RATIO - VALID_RATIO), "validation": n * VALID_RATIO}
    splits = {"train": [], "validation": [], "test": []}
    for group in group_list:
        # On remplit test puis validation avec de petits groupes ; tout le
        # reste (dont les gros groupes de versets répétés) va dans train.
        for name in ("test", "validation"):
            if len(splits[name]) + len(group) <= targets[name] and len(group) <= 3:
                splits[name].extend(group)
                break
        else:
            splits["train"].extend(group)
    return splits, len(group_list)


def write_splits(splits):
    for name, data in splits.items():
        out_csv = CLEANED / "parallel" / f"fr_dje_{name}.csv"
        with open(out_csv, "w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["fr", "dje", "source"])
            writer.writeheader()
            for fr, dje, src in data:
                writer.writerow({"fr": fr, "dje": dje, "source": src})

        txt_dir = CORPUS / name
        txt_dir.mkdir(parents=True, exist_ok=True)
        for lang, idx in (("fr", 0), ("dje", 1)):
            with open(txt_dir / f"text.{lang}.txt", "w", encoding="utf-8") as f:
                for row in data:
                    f.write(row[idx].replace("\n", " ") + "\n")
        print(f"  {name}: {len(data):,} paires → {out_csv.name}, {name}/text.{{fr,dje}}.txt")


def update_monolingual(splits):
    """Retire des corpus monolingues les phrases de validation/test (anti-fuite)."""
    held_out = {
        "fr": {dedup_key(fr) for s in ("validation", "test") for fr, _, _ in splits[s]},
        "dje": {dedup_key(dje) for s in ("validation", "test") for _, dje, _ in splits[s]},
    }
    train_side = {
        "fr": [fr for fr, _, _ in splits["train"]],
        "dje": [dje for _, dje, _ in splits["train"]],
    }
    normalizer = {"fr": normalize_fr, "dje": normalize_dje}

    for lang in ("fr", "dje"):
        path = CLEANED / "monolingual" / f"clean.text.{lang}.txt"
        existing = []
        if path.exists():
            with open(path, encoding="utf-8") as f:
                existing = [normalizer[lang](l) for l in f if l.strip()]
        # Une variante par clé normalisée : on garde la plus complète
        # (ponctuation comprise), dans l'ordre de première apparition.
        best, dropped = {}, 0
        for sent in existing + train_side[lang]:
            key = dedup_key(sent)
            if key in held_out[lang]:
                dropped += 1
            elif key and (key not in best or len(sent) > len(best[key])):
                best[key] = sent
        merged = list(best.values())
        with open(path, "w", encoding="utf-8") as f:
            f.writelines(s + "\n" for s in merged)
        print(f"  clean.text.{lang}.txt : {len(existing):,} → {len(merged):,} phrases "
              f"({dropped:,} occurrences de validation/test retirées)")


def main():
    print("=" * 60)
    print("CONSOLIDATION DU CORPUS PARALLÈLE FR↔ZARMA")
    print("=" * 60)

    pairs, stats = load_sources()
    print(f"\nTotal brut : {len(pairs):,} paires")

    kept, removed = filter_pairs(pairs)
    print("\nFiltrage qualité :")
    for reason, count in removed.most_common():
        print(f"  - {count:,} paires supprimées : {reason}")
    print(f"  {len(kept):,} paires conservées")

    unique = deduplicate(kept)
    print(f"\nDédoublonnage (zarma normalisé, priorité aux traductions humaines) : {len(kept) - len(unique):,} doublons, "
          f"{len(unique):,} paires uniques")

    splits, n_groups = group_split(unique)
    print(f"\nSplits ({n_groups:,} groupes de quasi-doublons) :")
    write_splits(splits)

    print("\nCorpus monolingues :")
    update_monolingual(splits)

    print("\nRépartition par source :")
    for name, data in splits.items():
        counts = Counter(src for _, _, src in data)
        print(f"  {name}: " + ", ".join(f"{k}={v:,}" for k, v in sorted(counts.items())))


if __name__ == "__main__":
    main()
