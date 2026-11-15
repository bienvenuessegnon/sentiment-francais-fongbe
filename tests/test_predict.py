from pathlib import Path

from src.config import MODELS_DIR
from src.predict import analyser


def test_modeles_sauvegardes():
    assert (MODELS_DIR / "sentiment_tfidf.joblib").exists()
    assert (MODELS_DIR / "lid_fr_fon.joblib").exists()


def test_analyser_francais_et_fongbe():
    pos = analyser("Je suis vraiment heureux de ta réussite.")
    assert pos["langue"] == "fr"
    assert pos["label"] == 1
    assert pos["avertissement"] is None

    neg = analyser("J'ai peur et je suis très en colère.")
    assert neg["langue"] == "fr"
    assert neg["label"] == 0

    fon = analyser("Vi lɛ ɖo awajijɛ mɛ.")
    assert fon["langue"] == "fon"
    assert fon["avertissement"]  # avertissement sur le transfert
    assert "locuteur" in fon["avertissement"] or "transfert" in fon["avertissement"] or "prudence" in fon["avertissement"]


def test_texte_trop_court():
    r = analyser(" ")
    assert r["label"] is None
    assert r["avertissement"]
