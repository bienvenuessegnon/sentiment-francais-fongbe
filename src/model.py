"""Baselines TF-IDF + régression logistique, et métriques."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from sklearn.pipeline import FeatureUnion, Pipeline

from src.config import CHAR_MAX_FEATURES, MODELS_DIR, SEED, WORD_MAX_FEATURES
from src.text import nfc


def pipeline_tfidf(C: float = 2.0) -> Pipeline:
    word = TfidfVectorizer(
        preprocessor=nfc,
        ngram_range=(1, 2),
        min_df=2,
        max_features=WORD_MAX_FEATURES,
        sublinear_tf=True,
    )
    char = TfidfVectorizer(
        preprocessor=nfc,
        analyzer="char",
        ngram_range=(3, 5),
        min_df=3,
        max_features=CHAR_MAX_FEATURES,
        sublinear_tf=True,
    )
    return Pipeline(
        [
            ("feats", FeatureUnion([("word", word), ("char", char)])),
            (
                "clf",
                LogisticRegression(
                    C=C,
                    max_iter=1000,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def entrainer(pipe: Pipeline, textes, labels) -> Pipeline:
    pipe.fit(list(textes), np.asarray(labels, dtype=int))
    return pipe


def predire(pipe: Pipeline, textes):
    X = list(textes)
    y = pipe.predict(X)
    proba = pipe.predict_proba(X)[:, 1]
    return y, proba


def metriques(y_vrai, y_pred, labels=(0, 1)) -> dict:
    y_vrai = np.asarray(y_vrai, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    p, r, f1, support = precision_recall_fscore_support(
        y_vrai, y_pred, labels=list(labels), zero_division=0
    )
    return {
        "n": int(len(y_vrai)),
        "accuracy": float(accuracy_score(y_vrai, y_pred)),
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro", zero_division=0)),
        "f1_negatif": float(f1[0]),
        "f1_positif": float(f1[1]),
        "precision_negatif": float(p[0]),
        "precision_positif": float(p[1]),
        "rappel_negatif": float(r[0]),
        "rappel_positif": float(r[1]),
        "support_negatif": int(support[0]),
        "support_positif": int(support[1]),
        "matrice": confusion_matrix(y_vrai, y_pred, labels=list(labels)).tolist(),
        "rapport": classification_report(
            y_vrai, y_pred, labels=list(labels), target_names=["négatif", "positif"], zero_division=0
        ),
    }


def sauvegarder(pipe: Pipeline, nom: str) -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / nom
    joblib.dump(pipe, path)
    return path


def charger(nom: str) -> Pipeline:
    return joblib.load(MODELS_DIR / nom)


def coefficients_distinctifs(pipe: Pipeline, k: int = 20) -> pd.DataFrame:
    """n-grammes les plus associés à chaque classe (côté mots, puis caractères)."""
    feats: FeatureUnion = pipe.named_steps["feats"]
    clf: LogisticRegression = pipe.named_steps["clf"]
    names = feats.get_feature_names_out()
    coef = clf.coef_.ravel()
    rows = []
    for signe, etiquette in ((1, "positif"), (-1, "négatif")):
        idx = np.argsort(signe * coef)[-k:][::-1]
        for i in idx:
            rows.append({"classe": etiquette, "feature": names[i], "coefficient": float(coef[i])})
    return pd.DataFrame(rows)
