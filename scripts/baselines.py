#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/baselines.py — P1-05 基线对比（legacy 协议，与 RF/GNN 同划分）
======================================================================
基线：Dummy(多数类) / Logistic(标准化) / GaussianNB / HistGradientBoosting。
5 折 StratifiedKFold(42)，与 EXP-010b（RF 0.8680）、EXP-019（GNN 0.8517）可直接对比。
产物：artifacts/baselines_metrics.json
"""

from __future__ import annotations

import json
import sys
import time
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
OUT_JSON = ARTIFACT_DIR / "baselines_metrics.json"
SEED = 42

REFERENCES = {
    "RF (legacy 复现, EXP-010b)": {"roc_auc": 0.86798, "balanced_accuracy": 0.78449, "mcc": 0.57115},
    "GNN (legacy 复现, EXP-019)": {"roc_auc": 0.85168, "balanced_accuracy": 0.77141, "mcc": 0.54631},
}


def main() -> int:
    from sklearn.dummy import DummyClassifier
    from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score, matthews_corrcoef, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.naive_bayes import GaussianNB
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    df = pd.read_csv(DATA_CSV)
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    y = (pic50.fillna(-1).values >= float(pic50.median())).astype(int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X, _ = compute_features(df["SMILES"].astype(str).tolist(), names)

    models = {
        "Dummy (多数类)": lambda: DummyClassifier(strategy="most_frequent"),
        "Logistic (标准化)": lambda: make_pipeline(StandardScaler(),
                                                   LogisticRegression(max_iter=2000)),
        "GaussianNB": lambda: GaussianNB(),
        "HistGradientBoosting": lambda: HistGradientBoostingClassifier(random_state=SEED),
    }
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    results = {}
    for name, factory in models.items():
        aucs, baccs, mccs = [], [], []
        t0 = time.perf_counter()
        for tr, te in skf.split(X, y):
            m = factory()
            m.fit(X[tr], y[tr])
            if hasattr(m, "predict_proba"):
                p = m.predict_proba(X[te])[:, 1]
            else:
                p = m.decision_function(X[te]) if hasattr(m, "decision_function") else m.predict(X[te])
            aucs.append(roc_auc_score(y[te], p))
            baccs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
            mccs.append(matthews_corrcoef(y[te], m.predict(X[te])))
        results[name] = {
            "roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
            "balanced_accuracy": {"mean": float(np.mean(baccs)), "std": float(np.std(baccs, ddof=1))},
            "mcc": {"mean": float(np.mean(mccs)), "std": float(np.std(mccs, ddof=1))},
            "fit_seconds_total": round(time.perf_counter() - t0, 1),
        }
        r = results[name]
        print(f"{name:<24} AUC={r['roc_auc']['mean']:.4f}±{r['roc_auc']['std']:.4f}  "
              f"BAcc={r['balanced_accuracy']['mean']:.4f}  MCC={r['mcc']['mean']:.4f}")

    # RF 参照重跑一次同折（保证逐折可比，不只引用文档值）
    aucs, baccs, mccs = [], [], []
    for tr, te in skf.split(X, y):
        rf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **{
            "n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
            "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True})
        rf.fit(X[tr], y[tr])
        p = rf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], rf.predict(X[te])))
        mccs.append(matthews_corrcoef(y[te], rf.predict(X[te])))
    results["RF (同折重跑)"] = {
        "roc_auc": {"mean": float(np.mean(aucs)), "std": float(np.std(aucs, ddof=1))},
        "balanced_accuracy": {"mean": float(np.mean(baccs)), "std": float(np.std(baccs, ddof=1))},
        "mcc": {"mean": float(np.mean(mccs)), "std": float(np.std(mccs, ddof=1))},
    }
    r = results["RF (同折重跑)"]
    print(f"{'RF (同折重跑)':<24} AUC={r['roc_auc']['mean']:.4f}±{r['roc_auc']['std']:.4f}  "
          f"BAcc={r['balanced_accuracy']['mean']:.4f}  MCC={r['mcc']['mean']:.4f}")

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-05",
        "protocol": "StratifiedKFold(5, shuffle, rs=42)，与 RF/GNN 复现同划分",
        "n_rows": int(len(df)),
        "results": results,
        "references": REFERENCES,
        "note": "GNN 数值引自 EXP-019（同数据同划分，逐折可比见 EXP-023）",
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
