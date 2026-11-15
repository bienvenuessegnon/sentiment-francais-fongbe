"""Enchaîne préparation, entraînement, figures et rapports.

Usage : python -m src.pipeline
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import data as D
from src import lid as L
from src import model as M
from src import plots
from src import report
from src.config import MODELS_DIR, PROCESSED_DIR, RAW_DIR, REPORTS, SEED
from src.text import nfc


def preparer():
    print("Téléchargement des sources (si besoin)…")
    D.download_sources()
    print("Audit du corpus à étiquettes automatiques…")
    audit = D.auditer_fon_sentiments()
    print(json.dumps(audit, ensure_ascii=False, indent=2))

    print("Échantillonnage Allociné…")
    fr = D.echantillon_allocine()
    D.enregistrer_table(fr["train"], "fr_train.csv")
    D.enregistrer_table(fr["validation"], "fr_val.csv")
    D.enregistrer_table(fr["test"], "fr_test.csv")
    for k, df in fr.items():
        print(f"  Allociné {k}: {len(df)} (pos={(df.label==1).sum()}, neg={(df.label==0).sum()})")

    print("Projection lexicale des étiquettes sur le corpus parallèle…")
    ffr = D.charger_ffr()
    cand = D.candidats_projection(ffr)
    print(f"  candidats (longueur) : {len(cand)}")
    proj = D.appliquer_projection(cand)
    print(
        f"  polarité lexicale univoque : {len(proj)} "
        f"(pos={(proj.label==1).sum()}, neg={(proj.label==0).sum()})"
    )
    train, val, test = D.decouper_groupes(proj)
    D.enregistrer_table(train, "fon_proj_train.csv")
    D.enregistrer_table(val, "fon_proj_val.csv")
    D.enregistrer_table(test, "fon_proj_test.csv")
    print(f"  partitions fon projeté : train={len(train)}, val={len(val)}, test={len(test)}")

    verif = D.jeu_verifie(test)  # uniquement à partir du test, pas de fuite
    # si trop peu d'exemples dans le test seul, compléter depuis val (toujours hors train)
    if len(verif) < 40:
        extra = D.jeu_verifie(pd.concat([val, test], ignore_index=True))
        # retirer tout groupe déjà vu en train
        extra = extra[~extra.groupe.isin(set(train.groupe))]
        verif = extra
    D.enregistrer_table(verif, "fon_verifie_candidats.csv")
    print(f"  candidats vérifiés (test/val, hors train) : {len(verif)}")

    lid_train, lid_test = D.construire_lid(ffr, fr["train"])
    D.enregistrer_table(lid_train, "lid_train.csv")
    D.enregistrer_table(lid_test, "lid_test.csv")
    print(f"  LID train={len(lid_train)}, test={len(lid_test)}")

    meta = {
        "seed": SEED,
        "allocine_train": int(len(fr["train"])),
        "allocine_val": int(len(fr["validation"])),
        "allocine_test": int(len(fr["test"])),
        "ffr_paires": int(len(ffr)),
        "projection_haute_confiance": int(len(proj)),
        "fon_train": int(len(train)),
        "fon_val": int(len(val)),
        "fon_test": int(len(test)),
        "fon_verifie_candidats": int(len(verif)),
        "audit_fon_sentiments": audit,
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "metadata.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    return meta


def _analyser_erreurs(df: pd.DataFrame, y_pred, nom: str, k: int = 12) -> pd.DataFrame:
    out = df.copy().reset_index(drop=True)
    out["predit"] = y_pred
    err = out[out.label != out.predit].copy()
    cols = [c for c in ["texte", "texte_fr", "fr", "label", "predit", "source"] if c in err.columns]
    err = err[cols]
    path = REPORTS / f"erreurs_{nom}.csv"
    err.head(200).to_csv(path, index=False)
    return err.head(k)


def entrainer_evaluer():
    fr_train = pd.read_csv(PROCESSED_DIR / "fr_train.csv")
    fr_val = pd.read_csv(PROCESSED_DIR / "fr_val.csv")
    fr_test = pd.read_csv(PROCESSED_DIR / "fr_test.csv")
    fon_train = pd.read_csv(PROCESSED_DIR / "fon_proj_train.csv")
    fon_val = pd.read_csv(PROCESSED_DIR / "fon_proj_val.csv")
    fon_test = pd.read_csv(PROCESSED_DIR / "fon_proj_test.csv")
    verif_path = PROCESSED_DIR / "fon_verifie.csv"
    if not verif_path.exists():
        # en attendant la relecture, on utilise les candidats
        verif = pd.read_csv(PROCESSED_DIR / "fon_verifie_candidats.csv")
    else:
        verif = pd.read_csv(verif_path)

    lid_train = pd.read_csv(PROCESSED_DIR / "lid_train.csv")
    lid_test = pd.read_csv(PROCESSED_DIR / "lid_test.csv")

    print("Modèle 1 : TF-IDF entraîné sur le français seulement…")
    m_fr = M.pipeline_tfidf()
    M.entrainer(m_fr, fr_train.texte, fr_train.label)

    print("Modèle 2 : TF-IDF entraîné sur français (Allociné + phrases courtes) + fongbé projeté…")
    # Le français du corpus parallèle (mêmes groupes que le train fongbé) apprend
    # les phrases courtes du quotidien, que les critiques de films ne couvrent pas.
    fr_court = fon_train[["texte_fr", "label"]].rename(columns={"texte_fr": "texte"})
    mix = pd.concat(
        [
            fr_train[["texte", "label"]],
            fr_court,
            fon_train[["texte", "label"]],
        ],
        ignore_index=True,
    )
    m_mix = M.pipeline_tfidf()
    M.entrainer(m_mix, mix.texte, mix.label)

    # Le modèle servi par l'application : le mixte, capable des deux langues
    M.sauvegarder(m_mix, "sentiment_tfidf.joblib")
    M.sauvegarder(m_fr, "sentiment_tfidf_fr.joblib")

    print("Identification de langue…")
    m_lid = L.entrainer_lid(lid_train.texte, lid_train.langue)
    L.sauvegarder_lid(m_lid)

    jeux = {
        "fr_test": fr_test,
        "fon_proj_test": fon_test,
        "fon_verifie": verif,
    }

    resultats = {}
    lignes_f1 = []
    matrices = {}
    rapports = {}

    def eval_modele(nom, pipe, jeu_nom, df):
        if df is None or len(df) == 0:
            return
        y, _p = M.predire(pipe, df.texte)
        met = M.metriques(df.label, y)
        resultats[f"{nom}::{jeu_nom}"] = {k: v for k, v in met.items() if k != "rapport"}
        rapports[f"{nom} / {jeu_nom}"] = met["rapport"]
        lignes_f1.append({"modele": nom, "jeu": jeu_nom, "f1_macro": met["f1_macro"], "accuracy": met["accuracy"]})
        if nom == "TF-IDF fr+fon":
            matrices[jeu_nom] = met["matrice"]
        print(f"  {nom:16s} {jeu_nom:16s} F1 macro={met['f1_macro']:.3f} acc={met['accuracy']:.3f} n={met['n']}")
        if jeu_nom in ("fr_test", "fon_proj_test", "fon_verifie"):
            _analyser_erreurs(df, y, f"{nom.replace(' ', '_')}_{jeu_nom}")

    for nom, pipe in (("TF-IDF fr", m_fr), ("TF-IDF fr+fon", m_mix)):
        for jeu_nom, df in jeux.items():
            eval_modele(nom, pipe, jeu_nom, df)
        # validation française, pour le réglage
        eval_modele(nom, pipe, "fr_val", fr_val)

    y_lid, _ = L.predire_lid(m_lid, lid_test.texte)
    lid_met = L.metriques_lid(lid_test.langue, y_lid)
    resultats["lid"] = lid_met
    print(f"  LID fr/fon        acc={lid_met['accuracy']:.3f} F1 macro={lid_met['f1_macro']:.3f}")

    coef_fr = M.coefficients_distinctifs(m_fr, k=15)
    coef_mix = M.coefficients_distinctifs(m_mix, k=15)
    coef_fr.to_csv(REPORTS / "coefficients_tfidf_fr.csv", index=False)
    coef_mix.to_csv(REPORTS / "coefficients_tfidf_mix.csv", index=False)

    print("Figures…")
    plots.volumes(fr_test, fon_test, verif)
    plots.longueurs(fr_train, fon_train)
    plots.matrices(matrices)
    plots.comparaison_f1(lignes_f1)
    plots.coefficients(coef_fr, "N-grammes les plus distinctifs (modèle français)", "05_coefficients_fr.png")
    plots.coefficients(coef_mix, "N-grammes les plus distinctifs (modèle français + fongbé)", "06_coefficients_mix.png")

    table = pd.DataFrame(lignes_f1)
    table.to_csv(REPORTS / "metrics_table.csv", index=False)

    key = {
        "resultats": resultats,
        "lid": lid_met,
        "rapports": rapports,
        "points": _points_cles(resultats, lid_met, fr_train, fon_train, verif),
    }
    report.ecrire_json(key, "key_figures.json")
    report.ecrire_metrics_md(key, table)
    print("Rapports écrits dans reports/")
    return key


def _points_cles(resultats, lid_met, fr_train, fon_train, verif) -> dict:
    def f1(cle):
        return resultats.get(cle, {}).get("f1_macro")

    return {
        "F1 macro TF-IDF français sur le test Allociné": _fmt(f1("TF-IDF fr::fr_test")),
        "F1 macro TF-IDF français appliqué au fongbé projeté (zéro-shot lexical)": _fmt(
            f1("TF-IDF fr::fon_proj_test")
        ),
        "F1 macro TF-IDF fr+fon sur le test fongbé projeté": _fmt(f1("TF-IDF fr+fon::fon_proj_test")),
        "F1 macro TF-IDF fr+fon sur le jeu fongbé vérifié": _fmt(f1("TF-IDF fr+fon::fon_verifie")),
        "Identification fr/fon (F1 macro)": _fmt(lid_met.get("f1_macro")),
        "Taille apprentissage français": str(len(fr_train)),
        "Taille apprentissage fongbé projeté": str(len(fon_train)),
        "Taille jeu fongbé vérifié": str(len(verif)),
    }


def _fmt(v):
    if v is None:
        return "n.d."
    return f"{v:.3f}"


def main():
    meta_path = RAW_DIR / "metadata.json"
    if not (PROCESSED_DIR / "fr_train.csv").exists():
        preparer()
    else:
        print("Jeux déjà préparés, passage à l'entraînement (supprimer data/processed pour relancer).")
    entrainer_evaluer()
    # Transformeur : optionnel, lancé seulement si les bibliothèques sont là
    try:
        from src.transformer_train import disponible, entrainer_et_evaluer

        if disponible():
            print("Fine-tuning MiniLM (1 époque, sous-ensemble)…")
            tr = entrainer_et_evaluer()
            if tr:
                print("  MiniLM fr_test F1:", _fmt(tr.get("fr_test", {}).get("f1_macro")))
                print("  MiniLM fon_proj F1:", _fmt(tr.get("fon_proj_test", {}).get("f1_macro")))
        else:
            print("torch/transformers absents : le fine-tuning transformeur est reporté.")
    except Exception as exc:  # le baseline TF-IDF reste valable
        print("Transformeur non abouti :", exc)
    print("Terminé.")


if __name__ == "__main__":
    main()
