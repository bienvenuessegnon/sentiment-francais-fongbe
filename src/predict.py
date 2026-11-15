"""Inférence légère pour l'application : langue puis sentiment."""
from __future__ import annotations

from functools import lru_cache

from src.config import LABEL_NAMES, LANG_NAMES
from src.lid import charger_lid, predire_lid
from src.model import charger, predire
from src.text import nfc


@lru_cache(maxsize=1)
def _modeles():
    sent = charger("sentiment_tfidf.joblib")
    lid = charger_lid("lid_fr_fon.joblib")
    return sent, lid


def analyser(texte: str) -> dict:
    t = nfc(texte)
    if len(t) < 2:
        return {
            "texte": t,
            "langue": None,
            "langue_libelle": None,
            "proba_fon": None,
            "label": None,
            "polarite": None,
            "proba_positif": None,
            "avertissement": "Texte trop court.",
        }
    sent, lid = _modeles()
    langues, p_fon = predire_lid(lid, [t])
    labels, p_pos = predire(sent, [t])
    langue = str(langues[0])
    out = {
        "texte": t,
        "langue": langue,
        "langue_libelle": LANG_NAMES.get(langue, langue),
        "proba_fon": float(p_fon[0]),
        "label": int(labels[0]),
        "polarite": LABEL_NAMES[int(labels[0])],
        "proba_positif": float(p_pos[0]),
        "avertissement": None,
    }
    if langue == "fon":
        out["avertissement"] = (
            "Pour le fongbé, l'étiquette a été apprise par transfert depuis le français "
            "(corpus parallèle), pas sur des annotations natives. À prendre avec prudence."
        )
    return out
