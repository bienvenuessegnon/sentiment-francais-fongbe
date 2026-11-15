from src.lexique import polarite_lexicale


def test_phrases_claires():
    assert polarite_lexicale("je suis heureux que tu sois venu") == 1
    assert polarite_lexicale("cette robe est jolie") == 1
    assert polarite_lexicale("j'ai un peu peur de lui") == 0
    assert polarite_lexicale("vous racontez un mensonge") == 0


def test_sous_chaines_et_locutions_exclues():
    # « beau » ne doit pas matcher « beaucoup »
    assert polarite_lexicale("il y a beaucoup de gens dans la rue") is None
    # « nul » dans « nulle part »
    assert polarite_lexicale("Kofi n'est allé nulle part") is None
    assert polarite_lexicale("ses beaux-parents exigèrent de lui bâtir une case") is None
    assert polarite_lexicale("L'oiseau aura beau crier, j'irai à la maison.") is None
    assert polarite_lexicale("je lui ai donné une carte de peur qu'elle ne se perde") is None


def test_negation_ecartee_et_pas_inversee():
    assert polarite_lexicale("les enfants ne sont pas trop heureux") is None
    assert polarite_lexicale("il n'est pas reconnaissant") is None
    assert polarite_lexicale("elle n'est pas du tout contente") is None
    assert polarite_lexicale("il a parlé sans peur") is None


def test_phrase_sans_indice():
    assert polarite_lexicale("elle est à l'école") is None
    assert polarite_lexicale("nous allons à la maison") is None
