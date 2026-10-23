#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/failure_cases.py — P1-03 失败案例库（OOF 错分提取与归因）
==================================================================

方法：与 EXP-010b/018 完全相同的 5 折 OOF 概率（每条样本都由「没见过它」的模型打分），
提取高置信度错分（FP：非活性被判活性；FN：反之），按置信度排序取 Top，
再从 pIC50 分带 / 骨架 / 描述符三个角度归因。

产物：artifacts/failure_cases.json + docs/FAILURE_CASES.md 由代理另行撰写。
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

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
OUT_JSON = ARTIFACT_DIR / "failure_cases.json"
SEED = 42
HP = {"n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
      "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True}
TOP_N = 15


def murcko(smiles: str) -> str:
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog("rdApp.*")
    try:
        s = MurckoScaffold.MurckoScaffoldSmiles(smiles=smiles, includeChirality=False)
        return s or "<none>"
    except Exception:
        return "<none>"


def main() -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    df = pd.read_csv(DATA_CSV)
    smiles_list = df["SMILES"].astype(str).tolist()
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    med = float(pic50.median())
    y = (pic50.fillna(-1).values >= med).astype(int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X, _ = compute_features(smiles_list, names)

    print(f"数据 {len(df)} 条，OOF 计算中…", flush=True)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **HP)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]

    pred = (oof >= 0.5).astype(int)
    fp_mask = (pred == 1) & (y == 0)   # 判活性，实际 median-非活性
    fn_mask = (pred == 0) & (y == 1)
    tn = int(((pred == 0) & (y == 0)).sum())
    tp = int(((pred == 1) & (y == 1)).sum())
    print(f"混淆（OOF n={len(y)}）: TN={tn} FP={int(fp_mask.sum())} FN={int(fn_mask.sum())} TP={tp}")

    scaffolds = [murcko(s) for s in smiles_list]
    pic50v = pic50.fillna(-1).values

    def top_cases(mask: np.ndarray) -> List[Dict[str, Any]]:
        idx = np.where(mask)[0]
        conf = np.abs(oof[idx] - 0.5)
        idx = idx[np.argsort(-conf)][:TOP_N]
        return [{
            "smiles": smiles_list[i], "pIC50": round(float(pic50v[i]), 3),
            "p_active": round(float(oof[i]), 4), "scaffold": scaffolds[i][:60],
        } for i in idx]

    fp_cases, fn_cases = top_cases(fp_mask), top_cases(fn_mask)

    # ---- 归因 1：错误是否集中在 median 边界带 ----
    bands = [(5.0, 6.0), (6.0, 6.5), (6.5, 7.0), (7.0, 7.5), (7.5, 8.0), (8.0, 12.0)]
    band_table = []
    for lo, hi in bands:
        m = (pic50v >= lo) & (pic50v < hi)
        if m.sum() == 0:
            continue
        band_table.append({
            "band": f"[{lo}, {hi})", "n": int(m.sum()),
            "fp_rate": float(fp_mask[m].mean()), "fn_rate": float(fn_mask[m].mean()),
            "acc": float((pred[m] == y[m]).mean()),
        })
        print(f"  band [{lo},{hi}): n={int(m.sum())}  FP率={band_table[-1]['fp_rate']:.3f}  "
              f"FN率={band_table[-1]['fn_rate']:.3f}  ACC={band_table[-1]['acc']:.3f}")

    # ---- 归因 2：FP 的骨架富集 ----
    fp_sc = Counter(scaffolds[i] for i in np.where(fp_mask)[0])
    all_sc = Counter(scaffolds)
    enrich = [{"scaffold": s[:60], "fp_count": c,
               "total": all_sc[s], "fp_share": c / all_sc[s]}
              for s, c in fp_sc.most_common(8)]

    # ---- 归因 3：描述符分布（错误 vs 正确） ----
    def desc_range(mask):
        return {n: [round(float(np.percentile(X[mask, j], 25)), 2),
                    round(float(np.percentile(X[mask, j], 75)), 2)] for j, n in enumerate(names)}

    attribution = {
        "fp_pIC50_median": round(float(np.median(pic50v[fp_mask])), 3),
        "fn_pIC50_median": round(float(np.median(pic50v[fn_mask])), 3),
        "fp_desc_IQR": desc_range(fp_mask),
        "fn_desc_IQR": desc_range(fn_mask),
        "correct_desc_IQR": desc_range(~(fp_mask | fn_mask)),
        "fp_scaffold_enrichment": enrich,
    }

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-03",
        "protocol": "5-fold OOF（同 EXP-010b 划分），median split",
        "confusion": {"tn": tn, "fp": int(fp_mask.sum()), "fn": int(fn_mask.sum()), "tp": tp},
        "error_rates": {"fp_rate": float(fp_mask.mean()), "fn_rate": float(fn_mask.mean())},
        "band_attribution": band_table,
        "top_fp_cases": fp_cases,
        "top_fn_cases": fn_cases,
        "attribution": attribution,
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
