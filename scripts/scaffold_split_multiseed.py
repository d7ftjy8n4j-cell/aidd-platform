#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/scaffold_split_multiseed.py — P0-05 多随机种子 scaffold split（legacy 协议）
====================================================================================

目的
----
在正式口径（数据集 A：13,286 条，median split）上回答审稿人 Q6/Q7：
「随机划分的 AUC 是否是骨架泄漏撑起来的？模型能泛化到全新骨架吗？」

方法
----
1. Bemis-Murcko 骨架提取（RDKit MurckoScaffoldSmiles，不含手性）。
2. 每个随机种子：按骨架分组贪心划分——洗牌骨架列表，把整骨架逐个放进测试集
   直到 ≥20% 分子，**train/test 骨架零重叠**（构造性保证 + 事后断言）。
3. 每个 seed 训练一个 RF（legacy 超参，random_state 与 split seed 配对），
   在测试集上报告 AUC / PR-AUC / Accuracy / BAcc / MCC / F1 / 混淆矩阵。
4. 汇总 ≥10 个 seed 的 mean±std（ddof=1），与随机划分参考值
   （legacy 5 折 CV AUC 0.8680±0.0050，EXP-010b）对照。

诚实性约定
----------
* 随机划分参考值引用自 artifacts/rf_legacy_protocol_metrics.json，不重跑、不口算。
* 单 seed 结果波动正常；结论只看多种子聚合，单种子不进任何材料。
* 全部指标落盘 artifacts/scaffold_multiseed_metrics.json，含数据 sha256。

用法
----
    python scripts/scaffold_split_multiseed.py               # 默认 10 seeds
    python scripts/scaffold_split_multiseed.py --n-seeds 20  # 加密
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

# 复用 legacy 复现脚本的同一特征代码路径（与部署 real_predictor 同源）
from train_rf_legacy_protocol import compute_features  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_CSV = REPO_ROOT / "data" / "raw" / "egfr_all_ic50_full.csv"
OUT_JSON = ARTIFACT_DIR / "scaffold_multiseed_metrics.json"
RANDOM_SPLIT_REF = {"auc": 0.8679837881666135, "std": 0.004950390952300648,
                    "source": "artifacts/rf_legacy_protocol_metrics.json (5-fold CV, EXP-010b)"}
HPARAMS = {"n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
           "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True}

SEP = "=" * 78


def _log(m: str) -> None:
    print(m, flush=True)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def murcko_scaffolds(smiles_list: List[str]) -> Tuple[List[str], int]:
    """逐分子 Bemis-Murcko 骨架；解析失败者记为独立单例骨架并计数。"""
    from rdkit import RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold

    RDLogger.DisableLog("rdApp.*")
    scaffolds: List[str] = []
    failed = 0
    for i, smi in enumerate(smiles_list):
        try:
            scf = MurckoScaffold.MurckoScaffoldSmiles(smiles=smi, includeChirality=False)
            if not scf:
                raise ValueError("empty scaffold")
        except Exception:
            failed += 1
            scf = f"<unparsable_{i}>"
        scaffolds.append(scf)
    return scaffolds, failed


def scaffold_split(smiles_list: List[str], scaffolds: List[str], seed: int,
                   test_size: float = 0.2) -> Tuple[np.ndarray, np.ndarray, int]:
    """按骨架分组贪心划分；返回 (train_idx, test_idx, 共享骨架数=0 的断言证据)。"""
    groups: Dict[str, List[int]] = {}
    for idx, scf in enumerate(scaffolds):
        groups.setdefault(scf, []).append(idx)

    order = list(groups.keys())
    random.Random(seed).shuffle(order)

    n_total = len(smiles_list)
    target = int(round(test_size * n_total))
    test_idx: List[int] = []
    test_scfs = set()
    for scf in order:
        if len(test_idx) >= target:
            break
        test_idx.extend(groups[scf])
        test_scfs.add(scf)

    test_set = set(test_idx)
    train_idx = [i for i in range(n_total) if i not in test_set]

    # 构造性零重叠断言
    train_scfs = {scaffolds[i] for i in train_idx}
    shared = train_scfs & test_scfs
    assert not shared, f"骨架重叠 {len(shared)} 个，划分实现错误"
    assert set(train_idx) & test_set == set(), "样本重叠"
    return np.array(sorted(train_idx)), np.array(sorted(test_idx)), 0


def train_eval(X: np.ndarray, y: np.ndarray, tr: np.ndarray, te: np.ndarray,
               seed: int) -> Dict[str, Any]:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score, average_precision_score

    clf = RandomForestClassifier(random_state=seed, n_jobs=-1, **HPARAMS)
    t0 = time.perf_counter()
    clf.fit(X[tr], y[tr])
    secs = time.perf_counter() - t0
    proba = clf.predict_proba(X[te])[:, 1]
    y_true = y[te]

    from sklearn.metrics import (accuracy_score, balanced_accuracy_score,
                                 confusion_matrix, f1_score, matthews_corrcoef)
    pred = (proba >= 0.5).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "seed": seed,
        "n_train": int(len(tr)), "n_test": int(len(te)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "pr_auc": float(average_precision_score(y_true, proba)),
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "train_seconds": round(secs, 2),
    }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="P0-05 多随机种子 scaffold split")
    ap.add_argument("--n-seeds", type=int, default=10)
    ap.add_argument("--seed-start", type=int, default=42)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--threshold", type=float, default=None,
                    help="默认 median split（与 legacy 协议一致）")
    args = ap.parse_args(argv)

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    _log(SEP)
    _log("药尘光 · P0-05 多随机种子 scaffold split（legacy 协议）")
    _log(SEP)

    if not DATA_CSV.exists():
        _log(f"[失败] 数据不存在: {DATA_CSV}")
        return 1

    df = pd.read_csv(DATA_CSV)
    smiles_list = df["SMILES"].astype(str).tolist()
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    med = float(pic50.median())
    thr = args.threshold if args.threshold is not None else med
    y = (pic50.fillna(-1).values >= thr).astype(int)
    _log(f"  数据: {len(df)} 条  sha256={_sha256_file(DATA_CSV)[:16]}…")
    _log(f"  标注: pIC50 >= {thr:.4f}（median split）  正例率={y.mean():.4f}")

    # ---- 特征（与 legacy 复现同一代码路径） ----
    from rdkit import Chem  # noqa: F401  确认 RDKit 可用
    feature_names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    _log(f"  特征: {len(feature_names)} 个 RDKit 描述符（compute_features 复用 legacy 脚本）")
    X, feat_diag = compute_features(smiles_list, feature_names)
    _log(f"  特征诊断: 不可解析 {feat_diag['unparsable_smiles']}，静默填 0 共 {feat_diag['zero_fill_total']}")

    # ---- 骨架 ----
    _log("\n  计算 Bemis-Murcko 骨架…")
    scaffolds, scf_failed = murcko_scaffolds(smiles_list)
    scf_counts = Counter(scaffolds)
    top10 = scf_counts.most_common(10)
    _log(f"  唯一骨架: {len(scf_counts)} / {len(smiles_list)}  解析失败: {scf_failed}")
    _log(f"  最大骨架占比: {top10[0][1] / len(smiles_list) * 100:.1f}%  Top5 合计: "
         f"{sum(c for _, c in top10[:5]) / len(smiles_list) * 100:.1f}%  Top10 合计: "
         f"{sum(c for _, c in top10) / len(smiles_list) * 100:.1f}%")

    # ---- 多种子划分 + 训练 ----
    seeds = list(range(args.seed_start, args.seed_start + args.n_seeds))
    _log(f"\n  {args.n_seeds} 个种子 × 骨架贪心划分（test≈{args.test_size:.0%}，零骨架重叠）")
    _log(f"  {'seed':>5} {'n_train':>8} {'n_test':>7} {'AUC':>8} {'BAcc':>8} {'MCC':>8} {'秒':>6}")
    results = []
    for seed in seeds:
        tr, te, _shared = scaffold_split(smiles_list, scaffolds, seed, args.test_size)
        r = train_eval(X, y, tr, te, seed)
        results.append(r)
        _log(f"  {seed:>5} {r['n_train']:>8} {r['n_test']:>7} "
             f"{r['roc_auc']:>8.4f} {r['balanced_accuracy']:>8.4f} {r['mcc']:>8.4f} "
             f"{r['train_seconds']:>6.1f}")

    agg: Dict[str, Dict[str, float]] = {}
    for key in ("roc_auc", "pr_auc", "accuracy", "balanced_accuracy", "mcc", "f1"):
        vals = np.array([r[key] for r in results], dtype=float)
        agg[key] = {"mean": float(vals.mean()), "std": float(vals.std(ddof=1)),
                    "min": float(vals.min()), "max": float(vals.max())}

    _log("")
    _log(f"  >>> Scaffold-split AUC  : {agg['roc_auc']['mean']:.4f} ± {agg['roc_auc']['std']:.4f}"
         f"   (range {agg['roc_auc']['min']:.4f} – {agg['roc_auc']['max']:.4f})")
    _log(f"  >>> 随机划分参考（5折CV）: {RANDOM_SPLIT_REF['auc']:.4f} ± {RANDOM_SPLIT_REF['std']:.4f}")
    delta = agg["roc_auc"]["mean"] - RANDOM_SPLIT_REF["auc"]
    _log(f"  >>> Δ(scaffold - random) : {delta:+.4f}")
    _log(f"  >>> BAcc {agg['balanced_accuracy']['mean']:.4f} ± {agg['balanced_accuracy']['std']:.4f}   "
         f"MCC {agg['mcc']['mean']:.4f} ± {agg['mcc']['std']:.4f}")

    payload = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/scaffold_split_multiseed.py",
        "generated_by": "改进总控代理 / P0-05",
        "purpose": "多随机种子 Bemis-Murcko scaffold split，确立 legacy 协议泛化结论（审稿 Q6/Q7）",
        "data": {
            "csv": "data/raw/egfr_all_ic50_full.csv",
            "csv_sha256": _sha256_file(DATA_CSV),
            "n_rows": int(len(df)),
            "labeling": f"median_split (pIC50 >= {thr:.4f})",
            "positive_rate": float(y.mean()),
        },
        "features": {"names": feature_names, "n": len(feature_names), "diagnostics": feat_diag},
        "hparams": {**HPARAMS, "random_state_note": "RF random_state 与 split seed 配对，seeds=42..%d" % seeds[-1]},
        "protocol": {
            "split": "scaffold_group_greedy（洗牌骨架列表，整骨架入测试集至 >=20%，构造性零重叠 + 断言）",
            "test_size": args.test_size,
            "n_seeds": args.n_seeds,
            "seeds": seeds,
            "reference_random_split": RANDOM_SPLIT_REF,
        },
        "scaffold_diversity": {
            "unique_scaffolds": len(scf_counts),
            "unparsable_treated_as_singleton": scf_failed,
            "largest_scaffold_count": top10[0][1],
            "largest_scaffold_share": top10[0][1] / len(smiles_list),
            "top5_share": sum(c for _, c in top10[:5]) / len(smiles_list),
            "top10_share": sum(c for _, c in top10) / len(smiles_list),
            "top10_scaffolds": [{"smiles": s, "count": c} for s, c in top10],
        },
        "per_seed": results,
        "aggregate": agg,
        "conclusion_hint": (
            "若 scaffold-split AUC 均值不低于随机划分参考（Δ>=0 或轻微为负但 std 内），"
            "则不存在骨架级泄漏，可主张泛化到新骨架；解释边界写入 EXP-021。"
        ),
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"\n  产物: {OUT_JSON}")
    _log(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
