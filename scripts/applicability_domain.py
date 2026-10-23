#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/applicability_domain.py — P1-02 适用域分析（legacy 协议）
==================================================================
定义模型「管多宽」：
1. 结构邻近度：测试分子对训练集的最近邻 Tanimoto 相似度（Morgan r=2, 2048 bit）；
2. 描述符包络：12 个描述符落在训练集 [P1, P99] 之外的形象化越界；
3. 分区性能：按 NN 相似度阈值（0.4 / 0.3）把测试集分区内/外，分别报 AUC/BAcc/MCC
   ——这是适用域分析的核心：越出适用域性能掉多少。

产物：artifacts/applicability_domain_metrics.json
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
OUT_JSON = ARTIFACT_DIR / "applicability_domain_metrics.json"
SEED = 42


def main() -> int:
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import AllChem
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 roc_auc_score)
    from sklearn.model_selection import train_test_split

    RDLogger.DisableLog("rdApp.*")
    df = pd.read_csv(DATA_CSV)
    smiles_list = df["SMILES"].astype(str).tolist()
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    y_all = (pic50.fillna(-1).values >= float(pic50.median())).astype(int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X, _ = compute_features(smiles_list, names)

    tr_i, te_i = train_test_split(np.arange(len(y_all)), test_size=0.2,
                                  random_state=SEED, stratify=y_all)
    print(f"划分: train={len(tr_i)} test={len(te_i)}", flush=True)

    # ---- RF（legacy 超参） ----
    clf = RandomForestClassifier(n_estimators=200, max_depth=15, max_features="sqrt",
                                 min_samples_leaf=2, random_state=SEED, n_jobs=-1)
    clf.fit(X[tr_i], y_all[tr_i])
    p = clf.predict_proba(X[te_i])[:, 1]
    y_te = y_all[te_i]

    # ---- 指纹与最近邻 Tanimoto ----
    t0 = time.perf_counter()
    fps_tr, fps_te = [], []
    for i in tr_i:
        m = Chem.MolFromSmiles(smiles_list[i])
        fps_tr.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)
    for i in te_i:
        m = Chem.MolFromSmiles(smiles_list[i])
        fps_te.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)
    valid_tr = [f for f in fps_tr if f is not None]
    print(f"指纹: train 有效 {len(valid_tr)}/{len(fps_tr)}，耗时 {time.perf_counter()-t0:.0f}s，计算 NN…", flush=True)

    nn_sim = np.zeros(len(te_i))
    for k, f in enumerate(fps_te):
        if f is None:
            nn_sim[k] = 0.0
            continue
        nn_sim[k] = max(DataStructs.BulkTanimotoSimilarity(f, valid_tr))
    print(f"NN 完成 {time.perf_counter()-t0:.0f}s", flush=True)

    qs = np.quantile(nn_sim, [0.05, 0.25, 0.5, 0.75, 0.95])
    print("test→train NN Tanimoto 分位数 p5/25/50/75/95:",
          ", ".join(f"{q:.3f}" for q in qs))

    # ---- 描述符包络越界 ----
    lo = np.percentile(X[tr_i], 1, axis=0)
    hi = np.percentile(X[tr_i], 99, axis=0)
    out_of_env = ((X[te_i] < lo) | (X[te_i] > hi)).any(axis=1)

    # ---- 分区性能 ----
    regions = {}
    for thr in (0.4, 0.3):
        m_in = nn_sim >= thr
        m_out = ~m_in
        reg = {}
        for label, mask in (("in_domain", m_in), ("out_domain", m_out)):
            n = int(mask.sum())
            if n and len(np.unique(y_te[mask])) > 1:
                reg[label] = {
                    "n": n, "share": float(mask.mean()),
                    "roc_auc": float(roc_auc_score(y_te[mask], p[mask])),
                    "balanced_accuracy": float(balanced_accuracy_score(y_te[mask], (p >= 0.5).astype(int)[mask])),
                    "mcc": float(matthews_corrcoef(y_te[mask], (p >= 0.5).astype(int)[mask])),
                }
            else:
                reg[label] = {"n": n, "share": float(mask.mean()), "note": "样本不足或单类别"}
        regions[f"nn_sim>={thr}"] = reg
        rin, rout = reg["in_domain"], reg["out_domain"]
        print(f"阈值 {thr}: 内域 n={rin['n']} AUC={rin.get('roc_auc', float('nan')):.4f} | "
              f"外域 n={rout['n']} AUC={rout.get('roc_auc', float('nan')):.4f}")

    overall_auc = float(roc_auc_score(y_te, p))
    print(f"全域 AUC={overall_auc:.4f}")

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-02",
        "protocol": {
            "split": "train_test_split(0.2, rs=42, stratify)",
            "fingerprint": "Morgan radius=2, 2048 bits",
            "ad_definition": "NN-Tanimoto(test→train) 阈值法 + 描述符 [P1,P99] 包络法",
        },
        "overall": {"n_test": int(len(te_i)), "roc_auc": overall_auc,
                    "bacc": float(balanced_accuracy_score(y_te, (p >= 0.5).astype(int)))},
        "nn_sim_quantiles": {"q05": float(qs[0]), "q25": float(qs[1]), "q50": float(qs[2]),
                             "q75": float(qs[3]), "q95": float(qs[4])},
        "regions_by_threshold": regions,
        "descriptor_envelope": {
            "out_of_envelope_rate": float(out_of_env.mean()),
            "out_of_envelope_auc": float(roc_auc_score(y_te[out_of_env], p[out_of_env]))
            if out_of_env.sum() > 1 and len(np.unique(y_te[out_of_env])) > 1 else None,
            "in_envelope_rate": float((~out_of_env).mean()),
        },
        "nn_sim_per_test_molecule": [round(float(v), 4) for v in nn_sim],
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
