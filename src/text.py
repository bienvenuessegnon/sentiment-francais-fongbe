"""Normalisation Unicode et squelettes lexicaux (pour éviter les fuites entre partitions)."""
from __future__ import annotations

import re
import unicodedata


_PUNCT = re.compile(r"[^\w]+", re.UNICODE)


def nfc(text: str) -> str:
    """Forme Unicode NFC, espaces normalisés : indispensable pour le fongbé (ɔ, ɛ, ɖ, tons)."""
    t = unicodedata.normalize("NFC", str(text).replace("\u00a0", " "))
    t = re.sub(r"\s+", " ", t).strip()
    return t


def n_words(text: str) -> int:
    return len(nfc(text).split())


def skeleton(text: str) -> str:
    """Squelette alphanumérique : casse, ponctuation et diacritiques (sauf lettres fon) retirés.

    Sert à regrouper les quasi-doublons avant un découpage train/test, comme le recommande
    la notice du corpus parallèle Fon–Français consolidé.
    """
    t = unicodedata.normalize("NFD", nfc(text).lower())
    kept = []
    for ch in t:
        if ch in "ɔɛɖŋǹḿẃ":
            kept.append(ch)
            continue
        cat = unicodedata.category(ch)
        if cat == "Mn":
            continue
        if ch.isalnum():
            kept.append(ch)
    return "".join(kept)


def group_key(fon: str, fr: str) -> str:
    return skeleton(fon) + "||" + skeleton(fr)
