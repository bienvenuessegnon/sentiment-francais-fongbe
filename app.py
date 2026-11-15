"""Démo : polarité d'une phrase en français ou en fongbé, avec identification de la langue.

Lancer : pip install -r requirements-app.txt && streamlit run app.py
"""
import pandas as pd
import streamlit as st

from src.predict import analyser

st.set_page_config(page_title="Sentiment français / fongbé", layout="centered")
INDIGO = "#4338CA"
TEAL = "#0F766E"
ROUGE = "#B91C1C"

EXEMPLES = [
    ("Français, positif", "Je suis vraiment heureux de ta réussite."),
    ("Français, négatif", "J'ai peur et je suis très en colère."),
    ("Fongbé, positif", "Vi lɛ ɖo awajijɛ mɛ."),  # « les enfants sont heureux »
    ("Fongbé, négatif", "Avun elɔ nyla hun."),  # « le chien est méchant »
]


@st.cache_resource
def _ok():
    # Charge les modèles une fois (via analyser)
    analyser("bonjour")
    return True


_ok()

st.markdown(
    """
    <style>
      .block-container {max-width: 720px; padding-top: 1.6rem;}
      textarea {font-size: 1.05rem !important;}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Sentiment français / fongbé")
st.caption(
    "Une phrase, deux langues. Le modèle repère d'abord si le texte est en français ou en fongbé, "
    "puis estime s'il est plutôt positif ou négatif. Pour le fongbé, les étiquettes d'apprentissage "
    "viennent d'un transfert depuis le français, pas d'une annotation native : à lire avec prudence."
)

exemple = st.selectbox("Exemples", ["(écrire une phrase)"] + [e[0] for e in EXEMPLES])
valeur = "" if exemple == "(écrire une phrase)" else dict(EXEMPLES)[exemple]
texte = st.text_area("Votre phrase", value=valeur, height=110, placeholder="Écrivez en français ou en fongbé…")

lancer = st.button("Analyser", type="primary")

if lancer or (texte and exemple != "(écrire une phrase)"):
    if not texte.strip():
        st.info("Saisissez une phrase.")
    else:
        r = analyser(texte)
        if r["avertissement"] and r["label"] is None:
            st.warning(r["avertissement"])
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.metric("Langue", r["langue_libelle"].capitalize())
                st.progress(min(max(r["proba_fon"], 0.0), 1.0), text=f"Score fongbé : {r['proba_fon']:.0%}")
            with c2:
                couleur = TEAL if r["label"] == 1 else ROUGE
                st.metric("Polarité", r["polarite"].capitalize())
                st.progress(min(max(r["proba_positif"], 0.0), 1.0), text=f"Score positif : {r['proba_positif']:.0%}")
            if r["avertissement"]:
                st.caption(r["avertissement"])

st.divider()
st.markdown(
    "Données : critiques Allociné (MIT) pour le français ; polarité du fongbé projetée depuis "
    "le français du corpus parallèle Dossou et al. (2021) et Degila et al. (2020), CC BY 4.0. "
    "Modèle de la démo : TF-IDF et régression logistique, assez léger pour un hébergement gratuit."
)
