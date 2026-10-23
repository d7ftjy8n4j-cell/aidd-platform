#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/ablation.py — P1-04 消融矩阵（legacy 协议，5 折同划分）
================================================================
维度：树数（25/50/100/200）、深度（5/10/15/20/None）、留一特征（12 组）、Top5 特征组。
产物：artifacts/ablation_metrics.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_rf_legacy_protocol import compute_features  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_CSV = REPO_ROOT / "data" / "raw" / "egfr_all_ic50_full.csv"
OUT_JSON = ARTIFACT_DIR / "ablation_metrics.json"
SEED = 42
BASE = {"n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
        "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True}


def main() -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import balanced_accuracy_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(DATA_CSV)
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    y = (pic50.fillna(-1).values >= float(pic50.median())).astype(int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X, _ = compute_features(df["SMILES"].astype(str).tolist(), names)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    folds = list(skf.split(X, y))

    def cv_eval(hp: dict, cols=None) -> dict:
        aucs, baccs = [], []
        Xc = X if cols is None else X[:, cols]
        for tr, te in folds:
            clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **hp)
            clf.fit(Xc[tr], y[tr])
            p = clf.predict_proba(Xc[te])[:, 1]
            aucs.append(roc_auc_score(y[te], p))
            baccs.append(balanced_accuracy_score(y[te], (p >= 0.5).astype(int)))
        return {"roc_auc": round(float(np.mean(aucs)), 4), "roc_auc_std": round(float(np.std(aucs, ddof=1)), 4),
                "bacc": round(float(np.mean(baccs)), 4)}

    results = {}

    print("== 树数消融 ==", flush=True)
    for n in (25, 50, 100, 200):
        hp = {**BASE, "n_estimators": n}
        results[f"n_estimators={n}"] = cv_eval(hp)
        print(f"  n={n}: {results[f'n_estimators={n}']}")

    print("== 深度消融 ==", flush=True)
    for d in (5, 10, 15, 20, None):
        hp = {**BASE, "max_depth": d}
        key = f"max_depth={d}"
        results[key] = cv_eval(hp)
        print(f"  depth={d}: {results[key]}")

    print("== 留一特征（12 组）==", flush=True)
    for j, n in enumerate(names):
        cols = [k for k in range(len(names)) if k != j]
        results[f"leave_out_{n}"] = cv_eval(BASE, cols)
        print(f"  -{n}: AUC={results[f'leave_out_{n}']['roc_auc']}")

    # Top5 特征（按部署 pkl 重要性：MolWt/TPSA/MolLogP/NumRotatableBonds/FractionCSP3）
    top5 = ["MolWt", "TPSA", "MolLogP", "NumRotatableBonds", "FractionCSP3"]
    cols5 = [names.index(n) for n in top5]
    results["top5_only"] = cv_eval(BASE, cols5)
    print(f"  Top5 only: {results['top5_only']}")

    results["_baseline_full"] = cv_eval(BASE)
    print(f"  full: {results['_baseline_full']}")

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-04",
        "protocol": "StratifiedKFold(5, shuffle, rs=42)，legacy 超参为基线",
        "results": results,
        "feature_names": names,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
