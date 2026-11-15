"""Identification de langue français / fongbé (n-grammes de caractères)."""
from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline

from src.config import MODELS_DIR, SEED
from src.text import nfc


# ɔ ɛ ɖ sont des indices typiques, mais le classifieur les apprend tout seul.
def pipeline_lid() -> Pipeline:
    return Pipeline(
        [
            (
                "tfidf",
                TfidfVectorizer(
                    preprocessor=nfc,
                    analyzer="char",
                    ngram_range=(1, 4),
                    min_df=2,
                    max_features=40000,
                    sublinear_tf=True,
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    max_iter=500,
                    class_weight="balanced",
                    solver="liblinear",
                    random_state=SEED,
                ),
            ),
        ]
    )


def entrainer_lid(textes, langues) -> Pipeline:
    pipe = pipeline_lid()
    pipe.fit(list(textes), np.asarray(langues))
    return pipe


def predire_lid(pipe: Pipeline, textes):
    X = list(textes)
    y = pipe.predict(X)
    proba = pipe.predict_proba(X)
    classes = list(pipe.classes_)
    # proba de « fon » si présente
    if "fon" in classes:
        p_fon = proba[:, classes.index("fon")]
    else:
        p_fon = np.zeros(len(X))
    return y, p_fon


def metriques_lid(y_vrai, y_pred) -> dict:
    return {
        "n": int(len(y_vrai)),
        "accuracy": float(accuracy_score(y_vrai, y_pred)),
        "f1_macro": float(f1_score(y_vrai, y_pred, average="macro")),
    }


def sauvegarder_lid(pipe: Pipeline, nom: str = "lid_fr_fon.joblib") -> Path:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    path = MODELS_DIR / nom
    joblib.dump(pipe, path)
    return path


def charger_lid(nom: str = "lid_fr_fon.joblib") -> Pipeline:
    return joblib.load(MODELS_DIR / nom)
