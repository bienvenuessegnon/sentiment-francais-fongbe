"""Construit et exécute les deux carnets (exploration, modélisation)."""
from __future__ import annotations

from pathlib import Path

import nbformat
from nbclient import NotebookClient
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "notebooks"


def _nb(cells):
    return new_notebook(
        cells=cells,
        metadata={"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
    )


SETUP = """from pathlib import Path
import json
import pandas as pd
from IPython.display import Image, display

ROOT = Path.cwd() if (Path.cwd() / "data").exists() else Path.cwd().parent
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
FIG = ROOT / "reports" / "figures"
REP = ROOT / "reports"
"""


def exploration():
    return _nb(
        [
            new_markdown_cell(
                "# Exploration : français (Allociné) et fongbé (corpus parallèle)\n\n"
                "Volumes, longueurs, audit du corpus à étiquettes automatiques, "
                "et aperçu de la projection lexicale."
            ),
            new_code_cell(
                SETUP
                + """
fr_train = pd.read_csv(PROC / "fr_train.csv")
fr_test = pd.read_csv(PROC / "fr_test.csv")
fon_tr = pd.read_csv(PROC / "fon_proj_train.csv")
fon_te = pd.read_csv(PROC / "fon_proj_test.csv")
verif = pd.read_csv(PROC / "fon_verifie.csv")
audit = json.loads((RAW / "audit_fon_sentiments.json").read_text())
print("français train/test", len(fr_train), len(fr_test))
print("fongbé projeté train/test", len(fon_tr), len(fon_te))
print("fongbé vérifié", len(verif), verif.label.value_counts().to_dict())
"""
            ),
            new_markdown_cell("## Audit du corpus « Fon Sentiment » (écarté)"),
            new_code_cell(
                """print("n =", audit["n"])
part = 100 * audit["part_avec_lettre_fon"]
print("avec lettre fon ɔɛɖ :", audit["avec_lettre_fon_ɔɛɖ"], f"({part:.1f} %)")
print("latin sans lettre fon :", audit["latin_sans_lettre_fon"])
print("allure française sans lettre fon :", audit["allure_francaise_sans_lettre_fon"])
print("décision :", audit["decision"])
"""
            ),
            new_markdown_cell("## Longueur des textes"),
            new_code_cell(
                """display(Image(str(FIG / "02_longueurs.png")))
print("médiane mots FR (train)", fr_train.texte.str.split().str.len().median())
print("médiane mots FON (train projeté)", fon_tr.texte.str.split().str.len().median())
"""
            ),
            new_markdown_cell("## Composition des jeux d'évaluation"),
            new_code_cell('display(Image(str(FIG / "01_volumes.png")))'),
            new_markdown_cell(
                "## Exemples du jeu fongbé relu (polarité contrôlée sur le français)"
            ),
            new_code_cell(
                """for lab, nom in ((1, "positif"), (0, "négatif")):
    print("---", nom, "---")
    sub = verif[verif.label == lab].head(3)
    for _, r in sub.iterrows():
        print("FR :", r.texte_fr)
        print("FON:", r.texte)
        print()
"""
            ),
        ]
    )


def modelisation():
    return _nb(
        [
            new_markdown_cell(
                "# Modélisation : TF-IDF, transfert et identification de langue\n\n"
                "Les modèles sont déjà entraînés (`models/`). Ce carnet relit les métriques, "
                "les matrices et quelques erreurs."
            ),
            new_code_cell(
                SETUP
                + """
table = pd.read_csv(REP / "metrics_table.csv")
key = json.loads((REP / "key_figures.json").read_text())
display(table.round(3))
"""
            ),
            new_markdown_cell("## Comparaison des F1 macro"),
            new_code_cell('display(Image(str(FIG / "04_comparaison_f1.png")))'),
            new_markdown_cell(
                "## Matrices du modèle mixte (français + fongbé projeté)\n\n"
                "De gauche à droite : test Allociné, test fongbé projeté, jeu fongbé relu."
            ),
            new_code_cell('display(Image(str(FIG / "03_matrices.png")))'),
            new_markdown_cell("## N-grammes distinctifs"),
            new_code_cell(
                """display(Image(str(FIG / "05_coefficients_fr.png")))
display(Image(str(FIG / "06_coefficients_mix.png")))
"""
            ),
            new_markdown_cell("## Erreurs sur le jeu fongbé relu"),
            new_code_cell(
                """err = pd.read_csv(REP / "erreurs_TF-IDF_fr+fon_fon_verifie.csv")
print(len(err), "erreurs")
cols = [c for c in ["texte_fr", "texte", "label", "predit"] if c in err.columns]
display(err[cols].head(10))
"""
            ),
            new_markdown_cell("## Identification de langue"),
            new_code_cell(
                """lid = key["lid"]
print("exactitude", round(lid["accuracy"], 3), "F1 macro", round(lid["f1_macro"], 3), "n =", lid["n"])
print("Score élevé attendu : l'orthographe du fongbé est très distinctive.")
"""
            ),
        ]
    )


def ecrire_et_executer(nb, name: str):
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    nbformat.write(nb, path)
    client = NotebookClient(
        nb, timeout=180, kernel_name="python3", resources={"metadata": {"path": str(ROOT)}}
    )
    client.execute()
    nbformat.write(nb, path)
    print("écrit", path)


def main():
    ecrire_et_executer(exploration(), "01_exploration.ipynb")
    ecrire_et_executer(modelisation(), "02_modelisation.ipynb")


if __name__ == "__main__":
    main()
