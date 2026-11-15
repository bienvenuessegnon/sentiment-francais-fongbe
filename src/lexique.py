"""Lexique de polarité française, seul outil utilisé pour projeter les étiquettes vers le fongbé.

Liste courte et conservative, assemblée pour ce projet (ce n'est pas un lexique tiers).
Les correspondances se font sur des mots ou des locutions entières, pas sur des sous-chaînes
(« beau » ne doit pas matcher « beaucoup », ni « nul » dans « nulle part »).

L'étiquette d'une phrase n'est retenue que s'il existe au moins un indice d'un seul côté,
et qu'aucune exception connue ne s'applique.
"""
from __future__ import annotations

import re
import unicodedata

# Locutions (plus longues d'abord) puis mots.
POSITIFS = [
    "je vous remercie",
    "je te remercie",
    "mille mercis",
    "je suis heureux",
    "je suis heureuse",
    "je suis content",
    "je suis contente",
    "je suis ravi",
    "je suis ravie",
    "je suis enchanté",
    "je suis enchantée",
    "j'aime",
    "je t'aime",
    "je l'aime",
    "félicitations",
    "formidable",
    "excellent",
    "excellente",
    "magnifique",
    "merveilleux",
    "merveilleuse",
    "superbe",
    "génial",
    "géniale",
    "fantastique",
    "extraordinaire",
    "splendide",
    "délicieux",
    "délicieuse",
    "parfait",
    "parfaite",
    "bravo",
    "merci",
    "heureux",
    "heureuse",
    "content",
    "contente",
    "ravi",
    "ravie",
    "enchanté",
    "enchantée",
    "adoré",
    "adorée",
    "adore",
    "adorer",
    "amour",
    "joie",
    "plaisir",
    "sourire",
    "réussite",
    "succès",
    "victoire",
    "espoir",
    "paix",
    "courage",
    "fier",
    "fière",
    "fierté",
    "gentil",
    "gentille",
    "généreux",
    "généreuse",
    "reconnaissant",
    "reconnaissante",
    "reconnaissance",
    "chaleureux",
    "chaleureuse",
    "sympathique",
    "agréable",
    "joli",
    "jolie",
    "beau",
    "belle",
    "belles",
    "beaux",
    "bienvenue",
    "chanceux",
    "chanceuse",
    "bénédiction",
]

NEGATIFS = [
    "j'ai horreur",
    "je déteste",
    "je hais",
    "je pleure",
    "en colère",
    "horrible",
    "horreur",
    "affreux",
    "affreuse",
    "atroce",
    "terrible",
    "catastrophe",
    "désastre",
    "tristesse",
    "triste",
    "malheureux",
    "malheureuse",
    "malheur",
    "souffrance",
    "douleur",
    "angoisse",
    "colère",
    "furieux",
    "furieuse",
    "déçu",
    "déçue",
    "déception",
    "scandale",
    "inacceptable",
    "honteux",
    "honteuse",
    "honte",
    "violent",
    "violente",
    "cruel",
    "cruelle",
    "injuste",
    "menace",
    "menaces",
    "dangereux",
    "dangereuse",
    "danger",
    "haine",
    "peur",
    "déteste",
    "détesté",
    "détestée",
    "écœuré",
    "écœurée",
    "insupportable",
    "répugnant",
    "répugnante",
    "mensonge",
    "menteur",
    "menteuse",
    "abandonné",
    "abandonnée",
    "blessé",
    "blessée",
    "misère",
    "famine",
    "guerre",
    "jaloux",
    "jalouse",
    "méchant",
    "méchante",
    "angoissé",
    "angoissée",
    "désolé",
    "désolée",
]

# Locutions où un mot du lexique n'exprime pas une polarité de la phrase.
EXCEPTIONS = [
    r"\bnulle part\b",
    r"\bnul ne\b",
    r"\bnul n['’]est\b",
    r"\ba(voir)? beau\b",
    r"\bon a beau\b",
    r"\bde peur que\b",
    r"\bde peur qu",
    r"\bpeur que\b",
    r"\bbeaucoup\b",
    r"\bbeau-fils\b",
    r"\bbeau-frère\b",
    r"\bbeau-père\b",
    r"\bbelle-mère\b",
    r"\bbelle-sœur\b",
    r"\bbelle-fille\b",
    r"\bbeaux-parents\b",
    r"\bbeaux parents\b",
    r"\bbelle-famille\b",
    r"\bbelle famille\b",
    r"\baura beau\b",
    r"\bauras beau\b",
    r"\baurait beau\b",
    r"\baurai beau\b",
]


def _fold(text: str) -> str:
    t = unicodedata.normalize("NFC", str(text)).lower()
    t = t.replace("’", "'")
    t = re.sub(r"\s+", " ", t).strip()
    return t


def _word_re(expr: str) -> re.Pattern:
    # frontières : début/fin ou caractère non lettre (l'apostrophe fait partie de l'expression)
    parts = [re.escape(p) for p in expr.split(" ")]
    body = r"\s+".join(parts)
    return re.compile(rf"(?<![\w']){body}(?![\w'])", re.IGNORECASE)


_POS_RE = [(expr, _word_re(expr)) for expr in sorted(POSITIFS, key=len, reverse=True)]
_NEG_RE = [(expr, _word_re(expr)) for expr in sorted(NEGATIFS, key=len, reverse=True)]
_EXC_RE = [re.compile(p, re.IGNORECASE) for p in EXCEPTIONS]


def _masquer_exceptions(t: str) -> str:
    for rx in _EXC_RE:
        t = rx.sub(" ", t)
    return t


def indices_lexicaux(texte_fr: str) -> tuple[list[str], list[str]]:
    t = _masquer_exceptions(_fold(texte_fr))
    pos = [expr for expr, rx in _POS_RE if rx.search(t)]
    neg = [expr for expr, rx in _NEG_RE if rx.search(t)]
    return pos, neg


# Négation approximative : on écarte la phrase plutôt que d'inverser l'étiquette.
_NEGATION_RE = re.compile(
    r"\b(?:ne|n')\b.{0,40}\b(?:pas|plus|jamais|rien|aucun|aucune|guère)\b"
    r"|\bsans\s+(?:peur|joie|honneur|honte|succès|plaisir|espoir|paix)\b",
    re.IGNORECASE,
)


def contient_negation(texte_fr: str) -> bool:
    return _NEGATION_RE.search(_fold(texte_fr)) is not None


def polarite_lexicale(texte_fr: str) -> int | None:
    """1 si seulement des indices positifs, 0 si seulement des négatifs, sinon None."""
    if contient_negation(texte_fr):
        return None
    pos, neg = indices_lexicaux(texte_fr)
    if pos and not neg:
        return 1
    if neg and not pos:
        return 0
    return None
