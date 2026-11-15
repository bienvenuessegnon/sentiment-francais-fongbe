from src.text import group_key, n_words, nfc, skeleton


def test_nfc_compacte_les_espaces_et_compose_les_caracteres():
    # é décomposé (e + accent) doit revenir à la forme composée
    decompose = "e\u0301 ɖo  azɔmɛ"
    assert nfc(decompose) == "é ɖo azɔmɛ"


def test_squelette_ignore_casse_ponctuation_et_accents_latins():
    assert skeleton("Elle est à la maison.") == skeleton("elle est a la maison")
    # les lettres propres au fongbé sont conservées
    assert "ɔ" in skeleton("azɔmɛ")
    assert "ɖ" in skeleton("é ɖò")


def test_group_key_regroupe_les_quasi_doublons():
    a = group_key("E ɖo xwe gbe.", "Elle est à la maison.")
    b = group_key("e ɖo xwe gbe", "elle est a la maison")
    assert a == b


def test_n_words():
    assert n_words("je suis heureux") == 3
    assert n_words("  ") == 0
