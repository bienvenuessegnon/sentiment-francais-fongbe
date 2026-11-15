"""Figures du rapport (matplotlib, style simple)."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from src.config import COLOR_ACCENT, COLOR_FON, COLOR_FR, COLOR_NEG, COLOR_POS, FIGURES_DIR

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "normal",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "axes.grid": False,
    }
)


def _save(fig, name: str):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def volumes(fr: pd.DataFrame, fon: pd.DataFrame, verif: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    labels = ["Français\n(Allociné, test)", "Fongbé projeté\n(test)", "Fongbé vérifié\n(français relu)"]
    neg = [
        int((fr.label == 0).sum()),
        int((fon.label == 0).sum()),
        int((verif.label == 0).sum()) if len(verif) else 0,
    ]
    pos = [
        int((fr.label == 1).sum()),
        int((fon.label == 1).sum()),
        int((verif.label == 1).sum()) if len(verif) else 0,
    ]
    x = np.arange(len(labels))
    ax.bar(x, neg, color=COLOR_NEG, width=0.55, label="négatif")
    ax.bar(x, pos, bottom=neg, color=COLOR_POS, width=0.55, label="positif")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Nombre d'exemples")
    ax.set_title("Composition des jeux d'évaluation")
    ax.legend(frameon=False)
    for i, (a, b) in enumerate(zip(neg, pos)):
        ax.text(i, a + b + max(pos + neg) * 0.02, str(a + b), ha="center", va="bottom", fontsize=9)
    return _save(fig, "01_volumes.png")


def longueurs(fr: pd.DataFrame, fon: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    fr_len = fr.texte.str.split().str.len()
    fon_len = fon.texte.str.split().str.len()
    bins = range(0, min(81, int(max(fr_len.max(), fon_len.max()) + 2)))
    ax.hist(fr_len, bins=bins, color=COLOR_FR, alpha=0.55, label="français (Allociné)", density=True)
    ax.hist(fon_len, bins=range(0, 41), color=COLOR_FON, alpha=0.55, label="fongbé (corpus parallèle)", density=True)
    ax.set_xlabel("Nombre de mots")
    ax.set_ylabel("Densité")
    ax.set_title("Longueur des textes : le fongbé est beaucoup plus court")
    ax.legend(frameon=False)
    return _save(fig, "02_longueurs.png")


def matrices(matrices: dict[str, list]):
    """matrices : nom -> matrice 2x2."""
    n = len(matrices)
    fig, axes = plt.subplots(1, n, figsize=(4.1 * n, 3.8))
    if n == 1:
        axes = [axes]
    for ax, (titre, m) in zip(axes, matrices.items()):
        arr = np.asarray(m)
        im = ax.imshow(arr, cmap="Blues")
        ax.set_xticks([0, 1], ["négatif", "positif"])
        ax.set_yticks([0, 1], ["négatif", "positif"])
        ax.set_xlabel("Prédit")
        ax.set_ylabel("Réel")
        ax.set_title(titre)
        vmax = arr.max() if arr.max() else 1
        for i in range(2):
            for j in range(2):
                color = "white" if arr[i, j] > vmax * 0.65 else "black"
                ax.text(j, i, str(arr[i, j]), ha="center", va="center", color=color)
    fig.tight_layout()
    return _save(fig, "03_matrices.png")


def comparaison_f1(lignes: list[dict]):
    """lignes : {modele, jeu, f1_macro}."""
    df = pd.DataFrame(lignes)
    modeles = list(df.modele.unique())
    jeux = list(df.jeu.unique())
    fig, ax = plt.subplots(figsize=(8.2, 4.6))
    w = 0.8 / max(len(jeux), 1)
    x = np.arange(len(modeles))
    palette = [COLOR_FR, COLOR_FON, COLOR_ACCENT, "#78716C"]
    for i, jeu in enumerate(jeux):
        vals = [
            float(df[(df.modele == m) & (df.jeu == jeu)].f1_macro.iloc[0])
            if not df[(df.modele == m) & (df.jeu == jeu)].empty
            else 0
            for m in modeles
        ]
        bars = ax.bar(x + (i - (len(jeux) - 1) / 2) * w, vals, width=w, label=jeu, color=palette[i % len(palette)])
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.015, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(modeles, rotation=15, ha="right")
    ax.set_ylim(0, 1.12)
    ax.set_ylabel("F1 macro")
    ax.set_title("F1 macro selon le modèle et le jeu d'évaluation")
    ax.legend(frameon=False, fontsize=9)
    return _save(fig, "04_comparaison_f1.png")


def coefficients(df: pd.DataFrame, titre: str, name: str):
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 5.0))
    for ax, classe, color in (
        (axes[0], "négatif", COLOR_NEG),
        (axes[1], "positif", COLOR_POS),
    ):
        sub = df[df.classe == classe].copy()
        sub["feature"] = sub.feature.str.replace(r"^(word|char)__", "", regex=True)
        sub = sub.iloc[::-1]
        ax.barh(sub.feature, sub.coefficient.abs() if classe == "négatif" else sub.coefficient, color=color)
        ax.set_title(classe)
        ax.set_xlabel("|coefficient|" if classe == "négatif" else "coefficient")
    fig.suptitle(titre)
    fig.tight_layout()
    return _save(fig, name)
