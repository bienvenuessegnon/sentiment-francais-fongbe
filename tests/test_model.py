import numpy as np

from src.model import metriques, pipeline_tfidf, predire
from src.lid import entrainer_lid, metriques_lid, predire_lid


def test_tfidf_apprend_un_signal_evident():
    textes = ["je suis heureux et ravi"] * 8 + ["c'est une horreur affreuse"] * 8
    labels = [1] * 8 + [0] * 8
    pipe = pipeline_tfidf(C=4.0)
    pipe.fit(textes, labels)
    y, proba = predire(pipe, ["je suis heureux", "quelle horreur affreuse"])
    assert list(y) == [1, 0]
    assert proba[0] > 0.6 and proba[1] < 0.4
    m = metriques([1, 0], y)
    assert m["f1_macro"] == 1.0
    assert m["matrice"] == [[1, 0], [0, 1]]


def test_lid_separe_francais_et_fongbe():
    fr = [
        "je vais à la maison demain matin",
        "nous sommes à l'école aujourd'hui",
        "il a cassé le vélo dans la cour",
        "merci pour votre visite à Cotonou",
    ] * 3
    fon = [
        "un ɖo xwe gbe",
        "mí ɖo azɔmɛ",
        "é nɔ́ y'àvĭ yoò",
        "vi lɛ ɖo awajijɛ mɛ",
    ] * 3
    textes = fr + fon
    langues = ["fr"] * len(fr) + ["fon"] * len(fon)
    pipe = entrainer_lid(textes, langues)
    y, p_fon = predire_lid(pipe, ["elle est à la maison", "é ɖò ayìjúnjɔ́n"])
    assert list(y) == ["fr", "fon"]
    assert p_fon[1] > 0.5 > p_fon[0]
    m = metriques_lid(["fr", "fon"], y)
    assert m["accuracy"] == 1.0
