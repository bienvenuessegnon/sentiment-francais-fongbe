"""Téléchargement, échantillonnage Allociné, projection d'étiquettes et partitions."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import GroupShuffleSplit, train_test_split

from src.config import (
    ALLOCINE_FILES,
    ALLOCINE_HF,
    EXTERNAL_DIR,
    FFR_HF,
    FON_SENT_HF,
    FR_MAX_CHARS,
    FR_MIN_CHARS,
    FR_TEST_N,
    FR_TRAIN_N,
    FR_VAL_N,
    PROCESSED_DIR,
    PROJ_MAX_CHARS_FR,
    PROJ_MIN_CHARS_FR,
    PROJ_MIN_WORDS_FON,
    PROJ_MIN_WORDS_FR,
    PROJ_PROBA_NEG,
    PROJ_PROBA_POS,
    RAW_DIR,
    SEED,
    VERIF_PROBA_NEG,
    VERIF_PROBA_POS,
    VERIF_TARGET_N,
)
from src.lexique import indices_lexicaux, polarite_lexicale
from src.text import group_key, n_words, nfc, skeleton


def _download(url: str, dest: Path, timeout: int = 120) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    r = requests.get(url, timeout=timeout, stream=True)
    r.raise_for_status()
    tmp = dest.with_suffix(dest.suffix + ".part")
    with tmp.open("wb") as f:
        for chunk in r.iter_content(1 << 20):
            if chunk:
                f.write(chunk)
    tmp.replace(dest)
    return dest


def download_sources() -> dict[str, Path]:
    """Télécharge les fichiers sources dans data/external/ s'ils sont absents."""
    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for split, name in ALLOCINE_FILES.items():
        paths[f"allocine_{split}"] = _download(
            f"{ALLOCINE_HF}/{name}", EXTERNAL_DIR / f"allocine_{split}.parquet"
        )
    paths["ffr"] = _download(FFR_HF, EXTERNAL_DIR / "fon_fr.csv")
    paths["fon_sent"] = _download(FON_SENT_HF, EXTERNAL_DIR / "fon_sentiments.parquet")
    return paths


def _load_allocine(split: str) -> pd.DataFrame:
    path = EXTERNAL_DIR / f"allocine_{split}.parquet"
    df = pd.read_parquet(path)
    df = df.rename(columns={"review": "texte"})
    df["texte"] = df["texte"].map(nfc)
    df["label"] = df["label"].astype(int)
    df["langue"] = "fr"
    df["source"] = "allocine"
    df["split_officiel"] = split
    df["n_car"] = df["texte"].str.len()
    return df


def echantillon_allocine() -> dict[str, pd.DataFrame]:
    """Sous-échantillons équilibrés, découpage officiel Allociné conservé."""
    rng = SEED
    out = {}
    quotas = {"train": FR_TRAIN_N, "validation": FR_VAL_N, "test": FR_TEST_N}
    for split, n in quotas.items():
        df = _load_allocine(split)
        df = df[(df.n_car >= FR_MIN_CHARS) & (df.n_car <= FR_MAX_CHARS)].copy()
        parts = []
        per = n // 2
        for lab in (0, 1):
            sub = df[df.label == lab]
            k = min(per, len(sub))
            parts.append(sub.sample(n=k, random_state=rng))
        out[split] = (
            pd.concat(parts, ignore_index=True)
            .sample(frac=1, random_state=rng)
            .reset_index(drop=True)
        )
    return out


def charger_ffr() -> pd.DataFrame:
    df = pd.read_csv(EXTERNAL_DIR / "fon_fr.csv")
    df["fon"] = df["fon"].map(nfc)
    df["fr"] = df["fr"].map(nfc)
    df = df[(df.fon.str.len() > 0) & (df.fr.str.len() > 0)].copy()
    df["groupe"] = [group_key(f, r) for f, r in zip(df.fon, df.fr)]
    df["n_mots_fr"] = df.fr.map(n_words)
    df["n_mots_fon"] = df.fon.map(n_words)
    df["n_car_fr"] = df.fr.str.len()
    return df


def auditer_fon_sentiments() -> dict:
    """Mesure la contamination visible du corpus à étiquettes automatiques."""
    df = pd.read_parquet(EXTERNAL_DIR / "fon_sentiments.parquet")
    col = "Fon" if "Fon" in df.columns else df.columns[0]
    textes = df[col].map(nfc)
    has_fon_letter = textes.map(lambda s: any(ch in "ɔɛɖƆƐƉ" for ch in s))
    latin_sans_fon = textes.map(
        lambda s: (not any(ch in "ɔɛɖƆƐƉ" for ch in s))
        and bool(__import__("re").fullmatch(r"[A-Za-zÀ-ÿ0-9 ,.'’!?;:\-«»()\"/]+", s) or False)
    )
    # pinyin / romanisation typique (lettres + tons numériques absents, mais syllabes accentuées type jié)
    cyrillic = textes.map(lambda s: any("\u0400" <= ch <= "\u04ff" for ch in s))
    # phrases clairement françaises
    fr_mots = textes.map(
        lambda s: bool(
            __import__("re").search(
                r"\b(le|la|les|des|une|est|dans|pour|dieu|livre|je|tu|il)\b",
                s.lower(),
            )
        )
        and not any(ch in "ɔɛɖƆƐƉ" for ch in s)
    )
    audit = {
        "n": int(len(df)),
        "colonnes": list(df.columns),
        "n_positif": int((df["sentiment"].astype(str).str.lower() == "positive").sum())
        if "sentiment" in df.columns
        else None,
        "n_negatif": int((df["sentiment"].astype(str).str.lower() == "negative").sum())
        if "sentiment" in df.columns
        else None,
        "avec_lettre_fon_ɔɛɖ": int(has_fon_letter.sum()),
        "part_avec_lettre_fon": round(float(has_fon_letter.mean()), 4),
        "latin_sans_lettre_fon": int(latin_sans_fon.sum()),
        "contient_cyrillique": int(cyrillic.sum()),
        "allure_francaise_sans_lettre_fon": int(fr_mots.sum()),
        "etiquettes": "automatiques (DistilBERT anglais sur une traduction, d'après la fiche HF)",
        "decision": "écarté de l'entraînement et de l'évaluation : étiquettes non humaines et mélange de langues visible",
    }
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_DIR / "audit_fon_sentiments.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return audit


def candidats_projection(ffr: pd.DataFrame) -> pd.DataFrame:
    m = (
        (ffr.n_mots_fr >= PROJ_MIN_WORDS_FR)
        & (ffr.n_mots_fon >= PROJ_MIN_WORDS_FON)
        & (ffr.n_car_fr >= PROJ_MIN_CHARS_FR)
        & (ffr.n_car_fr <= PROJ_MAX_CHARS_FR)
    )
    return ffr.loc[m].copy()


def appliquer_projection(candidats: pd.DataFrame, proba_fr: np.ndarray | None = None) -> pd.DataFrame:
    """Projette la polarité du français vers le fongbé, d'après le lexique uniquement.

    Le classifieur Allociné n'est pas utilisé pour étiqueter ces phrases : son domaine
    (critiques de films, textes longs) est trop éloigné des phrases courtes du corpus
    parallèle, et il classe presque tout le quotidien en « négatif ».
    """
    df = candidats.copy()
    polarites = df["fr"].map(polarite_lexicale)
    df = df.loc[polarites.notna()].copy()
    df["label"] = polarites.loc[df.index].astype(int)
    cues = df["fr"].map(lambda s: indices_lexicaux(s))
    df["indices_pos"] = cues.map(lambda x: ", ".join(x[0]))
    df["indices_neg"] = cues.map(lambda x: ", ".join(x[1]))
    if proba_fr is not None and len(proba_fr) == len(candidats):
        # conservé à titre indicatif, pas comme critère d'inclusion
        df["proba_positif_fr"] = pd.Series(proba_fr, index=candidats.index).loc[df.index].to_numpy()
    df["texte"] = df["fon"]
    df["texte_fr"] = df["fr"]
    df["langue"] = "fon"
    df["source"] = "projection_ffr"
    df["accord_lexique"] = True
    return df.reset_index(drop=True)


def decouper_groupes(
    df: pd.DataFrame, group_col: str = "groupe", test_size: float = 0.2, val_size: float = 0.15
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Découpage par groupes (squelettes) pour limiter la fuite de quasi-doublons."""
    # Un représentant par groupe, puis explosion.
    groupes = df[group_col].unique()
    # stratify on majority label per group if possible
    lab = df.groupby(group_col)["label"].mean().ge(0.5).astype(int)
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    # GroupShuffleSplit needs X, y, groups of rows
    dummy = np.arange(len(groupes))
    y = lab.reindex(groupes).fillna(0).astype(int).to_numpy()
    train_val_idx, test_idx = next(gss.split(dummy, y, groups=groupes))
    g_train_val, g_test = groupes[train_val_idx], groupes[test_idx]
    y_tv = y[train_val_idx]
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_size, random_state=SEED)
    tr_idx, va_idx = next(gss2.split(np.arange(len(g_train_val)), y_tv, groups=g_train_val))
    g_train, g_val = g_train_val[tr_idx], g_train_val[va_idx]
    train = df[df[group_col].isin(g_train)].copy()
    val = df[df[group_col].isin(g_val)].copy()
    test = df[df[group_col].isin(g_test)].copy()
    train["partition"] = "train"
    val["partition"] = "val"
    test["partition"] = "test"
    return train, val, test


def jeu_verifie(proj: pd.DataFrame) -> pd.DataFrame:
    """Candidats pour la relecture manuelle du français (hors train).

    Un exemple par groupe, autant que possible équilibré. Les étiquettes fongbé
    restent transférées : seule la polarité du français est contrôlée ensuite.
    """
    cand = proj.drop_duplicates("groupe").copy()
    parts = []
    for lab in (0, 1):
        sub = cand[cand.label == lab]
        k = min(max(VERIF_TARGET_N // 2, 1), len(sub))
        if k:
            parts.append(sub.sample(n=k, random_state=SEED))
    if not parts:
        return cand.iloc[0:0].copy()
    out = pd.concat(parts, ignore_index=True).sample(frac=1, random_state=SEED)
    out["source"] = "projection_ffr_verifiee"
    out["verification"] = (
        "polarité à contrôler sur le français ; étiquette transférée au fongbé, "
        "non annotée par un locuteur natif"
    )
    return out.reset_index(drop=True)


def construire_lid(ffr: pd.DataFrame, fr_extra: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Français vs fongbé à partir des paires FFR (plus un peu de français Allociné)."""
    # Échantillon de paires, une ligne fon et une ligne fr, groupes disjoints train/test
    sample = ffr.sample(n=min(12000, len(ffr)), random_state=SEED)
    groupes = sample.groupe.astype(str).unique().tolist()
    g_train, g_test = train_test_split(groupes, test_size=0.2, random_state=SEED)
    rows_train, rows_test = [], []
    for gset, bucket in ((g_train, rows_train), (g_test, rows_test)):
        sub = sample[sample.groupe.isin(gset)]
        bucket.append(pd.DataFrame({"texte": sub.fon.map(nfc), "langue": "fon"}))
        bucket.append(pd.DataFrame({"texte": sub.fr.map(nfc), "langue": "fr"}))
    extra_train, extra_test = train_test_split(
        fr_extra[["texte"]].assign(langue="fr"), test_size=0.2, random_state=SEED
    )
    rows_train.append(extra_train)
    rows_test.append(extra_test)
    train = pd.concat(rows_train, ignore_index=True).sample(frac=1, random_state=SEED)
    test = pd.concat(rows_test, ignore_index=True).sample(frac=1, random_state=SEED)
    # textes vides
    train = train[train.texte.str.len() >= 3].drop_duplicates("texte").reset_index(drop=True)
    test = test[test.texte.str.len() >= 3].drop_duplicates("texte").reset_index(drop=True)
    # aucun texte du test dans le train
    train = train[~train.texte.isin(set(test.texte))].reset_index(drop=True)
    return train, test


def enregistrer_table(df: pd.DataFrame, name: str) -> Path:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = PROCESSED_DIR / name
    df.to_csv(path, index=False)
    return path
