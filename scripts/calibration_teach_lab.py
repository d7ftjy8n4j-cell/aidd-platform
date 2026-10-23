#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/calibration_teach_lab.py — teach_lab 协议校准补遗（EXP-018 的【待补】项）
=================================================================================
teach_lab 协议：1,998 快照 → 阈值 6.0 清洗 → 1,606 条（86% 正例，6:1 不平衡）。
5 折 OOF（ StratifiedKFold(42)）下评估 Brier/ECE + 多数类基线对照。
产物：artifacts/calibration_teach_lab.json
"""
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

sys.path.insert(0, str(REPO_ROOT))
import argparse  # noqa: E402

parser = argparse.ArgumentParser()
args, _ = parser.parse_known_args()

from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.model_selection import StratifiedKFold  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
SNAPSHOT = REPO_ROOT / "data" / "raw" / "chembl_egfr_activities.csv"


def reliability(y, p, n_bins=10):
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        m = idx == b
        if m.sum():
            bins.append((float(p[m].mean()), float(y[m].mean()), int(m.sum())))
    ece = sum(n / len(p) * abs(mp - fp) for mp, fp, n in bins)
    return float(ece), bins


def main() -> int:
    from utils.teach_lab import clean_activity_data

    raw = pd.read_csv(SNAPSHOT)
    clean, stages = clean_activity_data(raw, pic50_threshold=6.0)
    y = clean["label"].to_numpy(dtype=int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    # 注意：teach_lab 小模型用的是 DESCRIPTOR_NAMES 特征集（F-12），但此处评估的是
    # 「同协议下重训的 12 描述符模型」，特征集与 legacy 一致以保证可比性，如实记录。
    X, _ = compute_features(clean["smiles"].astype(str).tolist(), names)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                     min_samples_leaf=2, random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    brier = float(np.mean((oof - y) ** 2))
    ece, bins = reliability(y, oof)
    pred = (oof >= 0.5).astype(int)
    payload = {
        "generated_by": "改进总控代理 / P1-01 补遗",
        "protocol": "teach_lab 协议（1,606 条，86.0% 正例）；特征为 legacy 12 描述符（保证可比，与教学 DESCRIPTOR_NAMES 的 F-12 差异如实记录）",
        "n": int(len(y)), "positive_rate": float(y.mean()),
        "majority_baseline_accuracy": float(max(y.mean(), 1 - y.mean())),
        "oof_accuracy": float((pred == y).mean()),
        "brier": brier, "ece_10bin": ece,
        "note": "不平衡任务下 ECE 对多数类桶敏感，解读需配合 BAcc（见 artifacts/rf_metrics.json 的 CV 结果）",
    }
    print(f"teach_lab 协议: n={len(y)}  Brier={brier:.4f}  ECE={ece:.4f}  "
          f"OOF-ACC={payload['oof_accuracy']:.4f}（多数类基线 {payload['majority_baseline_accuracy']:.4f}）")
    ARTIFACT_DIR.mkdir(exist_ok=True)
    (ARTIFACT_DIR / "calibration_teach_lab.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print("产物: artifacts/calibration_teach_lab.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
