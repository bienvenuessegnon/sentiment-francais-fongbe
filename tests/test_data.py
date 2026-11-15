from pathlib import Path

import pandas as pd

from src.config import PROCESSED_DIR
from src.text import skeleton


def _load(name):
    path = PROCESSED_DIR / name
    assert path.exists(), f"fichier manquant : {path}"
    return pd.read_csv(path)


def test_fichiers_traites_presents():
    for name in (
        "fr_train.csv",
        "fr_val.csv",
        "fr_test.csv",
        "fon_proj_train.csv",
        "fon_proj_val.csv",
        "fon_proj_test.csv",
        "fon_verifie.csv",
        "lid_train.csv",
        "lid_test.csv",
    ):
        df = _load(name)
        assert len(df) > 0


def test_allocine_decoupage_officiel_et_equilibre():
    train, val, test = _load("fr_train.csv"), _load("fr_val.csv"), _load("fr_test.csv")
    assert set(train.split_officiel) == {"train"}
    assert set(val.split_officiel) == {"validation"}
    assert set(test.split_officiel) == {"test"}
    for df, n in ((train, 10000), (val, 2000), (test, 2000)):
        assert len(df) == n
        assert set(df.label) == {0, 1}
        assert df.label.value_counts().min() == n // 2


def test_aucune_fuite_entre_partitions_fon():
    train, val, test = (
        _load("fon_proj_train.csv"),
        _load("fon_proj_val.csv"),
        _load("fon_proj_test.csv"),
    )
    verif = _load("fon_verifie.csv")
    blocs = {"train": train, "val": val, "test": test, "verif": verif}
    noms = list(blocs)
    for i, a in enumerate(noms):
        for b in noms[i + 1 :]:
            inter = set(blocs[a].groupe) & set(blocs[b].groupe)
            assert not inter, f"groupes partagés {a}/{b} : {len(inter)}"


def test_jeu_verifie_documentation_et_equilibre():
    verif = _load("fon_verifie.csv")
    assert len(verif) >= 50
    assert verif.label.value_counts().min() >= 25
    assert verif.source.eq("projection_ffr_verifiee").all()
    assert verif.verification.str.contains("non annotée par un locuteur natif").all()
    # le français du jeu vérifié a bien une polarité lexicale univoque
    from src.lexique import polarite_lexicale

    for texte, lab in zip(verif.texte_fr, verif.label):
        assert polarite_lexicale(texte) == int(lab)


def test_lid_textes_distincts():
    train, test = _load("lid_train.csv"), _load("lid_test.csv")
    assert set(train.langue) == {"fr", "fon"}
    assert set(test.langue) == {"fr", "fon"}
    assert set(train.texte).isdisjoint(set(test.texte))
