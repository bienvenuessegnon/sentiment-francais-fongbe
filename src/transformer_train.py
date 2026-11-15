"""Fine-tuning d'un petit transformeur multilingue (MiniLM), prévu pour un CPU.

Le script est optionnel : s'il manque torch/transformers, le reste du projet
fonctionne avec le modèle TF-IDF. Entraînement volontairement court (1 époque,
sous-ensemble) pour rester réalisable sans GPU.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    MODELS_DIR,
    PROCESSED_DIR,
    REPORTS,
    SEED,
    TRANSFORMER_BATCH,
    TRANSFORMER_EPOCHS,
    TRANSFORMER_MAX_LEN,
    TRANSFORMER_NAME,
    TRANSFORMER_TRAIN_N,
)
from src.model import metriques


def disponible() -> bool:
    try:
        import torch  # noqa: F401
        import transformers  # noqa: F401

        return True
    except ImportError:
        return False


def _sous_ensemble(df: pd.DataFrame, n: int) -> pd.DataFrame:
    if len(df) <= n:
        return df
    parts = []
    per = n // 2
    for lab in (0, 1):
        sub = df[df.label == lab]
        k = min(per, len(sub))
        parts.append(sub.sample(n=k, random_state=SEED))
    return pd.concat(parts).sample(frac=1, random_state=SEED).reset_index(drop=True)


def entrainer_et_evaluer() -> dict | None:
    if not disponible():
        return None

    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_linear_schedule_with_warmup

    fr_train = pd.read_csv(PROCESSED_DIR / "fr_train.csv")
    fr_val = pd.read_csv(PROCESSED_DIR / "fr_val.csv")
    fr_test = pd.read_csv(PROCESSED_DIR / "fr_test.csv")
    fon_test = pd.read_csv(PROCESSED_DIR / "fon_proj_test.csv")
    verif_path = PROCESSED_DIR / "fon_verifie.csv"
    verif = pd.read_csv(verif_path) if verif_path.exists() else pd.DataFrame()

    train = _sous_ensemble(fr_train, TRANSFORMER_TRAIN_N)
    val = _sous_ensemble(fr_val, min(800, len(fr_val)))

    tok = AutoTokenizer.from_pretrained(TRANSFORMER_NAME)

    class TextDS(Dataset):
        def __init__(self, textes, labels):
            self.enc = tok(
                list(textes),
                truncation=True,
                padding=True,
                max_length=TRANSFORMER_MAX_LEN,
                return_tensors="pt",
            )
            self.y = torch.tensor(list(labels), dtype=torch.long)

        def __len__(self):
            return int(self.y.shape[0])

        def __getitem__(self, i):
            item = {k: v[i] for k, v in self.enc.items()}
            item["labels"] = self.y[i]
            return item

    model = AutoModelForSequenceClassification.from_pretrained(TRANSFORMER_NAME, num_labels=2)
    device = torch.device("cpu")
    model.to(device)
    train_loader = DataLoader(TextDS(train.texte, train.label), batch_size=TRANSFORMER_BATCH, shuffle=True)
    steps = TRANSFORMER_EPOCHS * len(train_loader)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-5)
    sched = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps)

    model.train()
    for _epoch in range(TRANSFORMER_EPOCHS):
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            out = model(**batch)
            out.loss.backward()
            opt.step()
            sched.step()
            opt.zero_grad()

    def ev(df: pd.DataFrame) -> dict:
        if df is None or len(df) == 0:
            return {}
        model.eval()
        preds, labels = [], []
        ds = TextDS(df.texte, df.label)
        loader = DataLoader(ds, batch_size=TRANSFORMER_BATCH * 2)
        with torch.no_grad():
            for batch in loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                logits = model(**{k: v for k, v in batch.items() if k != "labels"}).logits
                preds.extend(logits.argmax(-1).cpu().numpy().tolist())
                labels.extend(batch["labels"].cpu().numpy().tolist())
        return metriques(labels, preds)

    resultats = {
        "modele": TRANSFORMER_NAME,
        "n_train": int(len(train)),
        "epoques": TRANSFORMER_EPOCHS,
        "max_len": TRANSFORMER_MAX_LEN,
        "fr_val": ev(val),
        "fr_test": ev(fr_test),
        "fon_proj_test": ev(fon_test),
        "fon_verifie": ev(verif) if len(verif) else {},
    }

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dest = MODELS_DIR / "minilm_sentiment"
    dest.mkdir(exist_ok=True)
    model.save_pretrained(dest)
    tok.save_pretrained(dest)
    (REPORTS / "transformer_metrics.json").write_text(
        json.dumps(resultats, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return resultats
