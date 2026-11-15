"""Écriture des rapports chiffrés (JSON + Markdown)."""
from __future__ import annotations

import json
from datetime import date

import pandas as pd

from src.config import REPORTS


def ecrire_json(obj, nom: str):
    REPORTS.mkdir(parents=True, exist_ok=True)
    path = REPORTS / nom
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def ecrire_metrics_md(key: dict, table: pd.DataFrame) -> None:
    lines = [
        "# Métriques",
        "",
        f"Généré le {date.today().isoformat()}. Tous les chiffres viennent du code.",
        "",
        "## Tableau récapitulatif",
        "",
        _md_table(table),
        "",
        "## Points clés",
        "",
    ]
    for k, v in key.get("points", {}).items():
        lines.append(f"- **{k}** : {v}")
    lines += ["", "## Détail des rapports de classification", ""]
    for nom, bloc in key.get("rapports", {}).items():
        lines += [f"### {nom}", "", "```", bloc.strip(), "```", ""]
    (REPORTS / "metrics.md").write_text("\n".join(lines), encoding="utf-8")


def _md_table(df: pd.DataFrame) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            v = row[c]
            if isinstance(v, float):
                cells.append(f"{v:.3f}" if v <= 1.5 else f"{v:.1f}")
            else:
                cells.append(str(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
