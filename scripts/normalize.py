"""
Normalisation orthographique et filtres de qualité partagés par les scripts
de préparation du corpus.

Problèmes corrigés :
- Unicode : composition NFC (« ã » décomposé en « a » + U+0303 → « ã »).
- Artefacts de tokenisation Moses présents dans english-zarma mt560 :
  « n 'ay » → « n'ay », « @-@ » → « - », « caha ̃ » → « cahã »,
  espaces avant la ponctuation, guillemets droits espacés.
- Espaces multiples, espaces insécables.

Filtres :
- Références de versets en tête du français (« (7:26) ... ») : dans Feriji,
  ces lignes correspondent à des écarts de numérotation entre la Bible Segond
  et la Bible zarma ; la paire est alors presque toujours décalée.
- Ratio de longueur zarma/français hors bornes : signe de mauvais alignement.
"""

import re
import unicodedata

# ── Normalisation ─────────────────────────────────────────────────────────

_SPACES = re.compile(r"[ \t  ]+")
_MOSES_HYPHEN = re.compile(r"\s*@-@\s*")
_MOSES_APOS = re.compile(r"(\w) '(\w)")          # n 'ay  -> n'ay
_DETACHED_MARK = re.compile(r"(\S)\s+([̀-ͯ])")  # caha ̃ -> cahã
_SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:?!)\]»›])")
_SPACE_AFTER_OPEN = re.compile(r"([(\[«‹])\s+")
_VERSE_REF = re.compile(r"^\s*\(\d+[:.]\d+\)\s*")


def _fix_straight_quotes(text: str) -> str:
    """Recolle les guillemets droits laissés espacés par Moses (« " M 'iri . " »)."""
    if text.count('"') % 2:
        return text  # guillemets non appariés : on ne devine pas
    parts = text.split('"')
    # Les segments d'indice impair sont entre guillemets : on retire les
    # espaces collés à l'intérieur des guillemets.
    for i in range(1, len(parts), 2):
        parts[i] = parts[i].strip()
    return '"'.join(parts)


def _common(text: str) -> str:
    text = unicodedata.normalize("NFC", text)
    text = text.replace("’", "'").replace("ʼ", "'")
    text = _SPACES.sub(" ", text).strip()
    return text


def normalize_dje(text: str) -> str:
    """Normalise une phrase zarma."""
    text = unicodedata.normalize("NFD", text)
    text = _DETACHED_MARK.sub(r"\1\2", text)
    text = _common(text)
    text = _MOSES_HYPHEN.sub("-", text)
    text = _MOSES_APOS.sub(r"\1'\2", text)
    text = _SPACE_BEFORE_PUNCT.sub(r"\1", text)
    text = _SPACE_AFTER_OPEN.sub(r"\1", text)
    text = _fix_straight_quotes(text)
    return _SPACES.sub(" ", text).strip()


def normalize_fr(text: str) -> str:
    """Normalise une phrase française (sans toucher à la typographie française)."""
    return _common(text)


# ── Filtres ───────────────────────────────────────────────────────────────

MIN_RATIO = 0.5   # longueur zarma / longueur français (p1 ≈ 0,5 sur le corpus)
MAX_RATIO = 1.8   # p99 ≈ 1,6


def has_verse_ref(fr: str) -> bool:
    return bool(_VERSE_REF.match(fr))


def plausible_ratio(fr: str, dje: str) -> bool:
    ratio = len(dje) / max(1, len(fr))
    return MIN_RATIO <= ratio <= MAX_RATIO


def dedup_key(text: str) -> str:
    """Clé de dédoublonnage : minuscules, sans accents ni ponctuation."""
    text = unicodedata.normalize("NFKD", text.lower())
    text = "".join(c if c.isalnum() else " "
                   for c in text if not unicodedata.combining(c))
    return " ".join(text.split())
