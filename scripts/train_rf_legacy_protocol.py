#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_rf_legacy_protocol.py — 复现「原始训练协议」的 RF 指标
=====================================================================

背景（重要）
------------
P0-01 的 ``scripts/train_rf.py`` 用的是**当前** ``utils.teach_lab`` 的标注策略：
    pIC50 ≥ 6 活性 / pIC50 ≤ 5 非活性 / 5~6 丢弃

这条策略在 CHEMBL203 IC50 上得到 **86.0% 活性**，严重不平衡，导致
Accuracy 虚高、Balanced Accuracy 只有 0.66。

但项目 README 声称的「13,286 条 / 50.8% 活性 / RF AUC 0.867±0.005」
用的是**另一套标注策略** —— 经实测反推，是 **median split（中位数划分）**：

    pIC50 ≥ 7.0 → 活性(1)，否则 → 非活性(0)

median split 天然给出 50:50 平衡标签，因此 Accuracy 与 Balanced Accuracy
几乎相等（0.785 vs 0.785），**不存在不平衡虚高问题**。

本脚本用原始数据 ``egfr_all_ic50_full.csv``（13,286 条）走 median split，
验证 README 声称的 RF 指标能否复现。

实测结论（2026-10-24）
----------------------
    样本量      13,286      vs 声称 13,286      完全一致
    活性比例    50.81%      vs 声称 50.8%       完全一致
    CV AUC      0.8680±0.0050 vs 声称 0.867±0.005 一致
    CV ACC      0.7851±0.0073 vs 声称 0.782±0.005 一致

**README 的 RF 指标经复现，成立。**

用法
----
    python scripts/train_rf_legacy_protocol.py
    python scripts/train_rf_legacy_protocol.py --csv "C:/path/to/egfr_all_ic50_full.csv"
    python scripts/train_rf_legacy_protocol.py --median 7.0

作者：改进总控代理
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

warnings.filterwarnings("ignore")

if sys.platform == "win32":
    for _stream in (sys.stdout, sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "raw"
DEFAULT_CSV = DATA_DIR / "egfr_all_ic50_full.csv"
METRICS_JSON = ARTIFACT_DIR / "rf_legacy_protocol_metrics.json"
MODEL_OUT = ARTIFACT_DIR / "rf_legacy_protocol.pkl"
FEATURE_NAMES_JSON = REPO_ROOT / "feature_names.json"

DEFAULT_HPARAMS: Dict[str, Any] = {
    "n_estimators": 200,
    "max_depth": 15,
    "max_features": "sqrt",
    "min_samples_leaf": 2,
    "criterion": "gini",
    "bootstrap": True,
    "random_state": 42,
    "n_jobs": -1,
}

SEP = "=" * 78


def _log(m: str) -> None:
    print(m, flush=True)


def _section(t: str) -> None:
    _log("")
    _log(SEP)
    _log(t)
    _log(SEP)


def _rel(p: Path) -> str:
    try:
        return str(Path(p).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def compute_features(smiles_list: List[str], names: List[str]) -> tuple:
    """复刻 real_predictor.smiles_to_features：异常静默填 0。"""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")
    X = np.zeros((len(smiles_list), len(names)), dtype=float)
    fills = {n: 0 for n in names}
    unparsable = 0

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            unparsable += 1
            for n in names:
                fills[n] += 1
            continue
        for j, n in enumerate(names):
            try:
                v = float(getattr(Descriptors, n)(mol))
                if np.isnan(v) or np.isinf(v):
                    raise ValueError("nan/inf")
                X[i, j] = v
            except Exception:
                X[i, j] = 0.0
                fills[n] += 1

    return X, {
        "unparsable_smiles": unparsable,
        "zero_fill_counts": {k: v for k, v in fills.items() if v},
        "zero_fill_total": int(sum(fills.values())),
    }


def metrics_block(y_true: np.ndarray, y_prob: np.ndarray) -> Dict[str, Any]:
    from sklearn.metrics import (
        accuracy_score, average_precision_score, balanced_accuracy_score,
        confusion_matrix, f1_score, matthews_corrcoef, precision_score,
        recall_score, roc_auc_score,
    )

    pred = (y_prob >= 0.5).astype(int)
    out: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        out["pr_auc"] = float(average_precision_score(y_true, y_prob))
    except ValueError:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    except ValueError:
        out["confusion"] = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="原始协议（median split）RF 复现")
    ap.add_argument("--csv", default=str(DEFAULT_CSV))
    ap.add_argument("--median", type=float, default=None,
                    help="活性阈值，默认取数据中位数（median split）")
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    csv_path = Path(args.csv)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    _log(SEP)
    _log("药尘光 · 原始协议（median split）RF 指标复现")
    _log(SEP)
    _log(f"  时间   : {datetime.now().isoformat(timespec='seconds')}")
    _log(f"  Python : {platform.python_version()}")

    if not csv_path.exists():
        _log(f"[失败] 数据文件不存在: {csv_path}")
        _log("       请把 egfr_all_ic50_full.csv 放到 data/raw/ 下，或用 --csv 指定路径。")
        return 1

    _section("1. 读取原始数据")
    df = pd.read_csv(csv_path)
    _log(f"  文件  : {csv_path}")
    _log(f"  行数  : {len(df)}")
    _log(f"  列    : {list(df.columns)}")

    # 兼容不同列名
    smi_col = next((c for c in df.columns if c.strip().lower() in ("smiles", "canonical_smiles")), None)
    pic_col = next((c for c in df.columns if c.strip().lower() in ("pic50", "p_ic50")), None)
    if smi_col is None or pic_col is None:
        _log(f"[失败] 找不到 SMILES / pIC50 列。实际列: {list(df.columns)}")
        return 1
    _log(f"  SMILES 列: {smi_col}   pIC50 列: {pic_col}")

    _section("2. Median split 标注")
    threshold = args.median if args.median is not None else float(np.median(df[pic_col]))
    y = (df[pic_col].to_numpy(dtype=float) >= threshold).astype(int)
    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    _log(f"  阈值 median = {threshold:.4f}")
    _log(f"  活性(1) = {label_counts.get(1, 0)}  ({label_counts.get(1,0)/len(y)*100:.2f}%)")
    _log(f"  非活性(0) = {label_counts.get(0, 0)}  ({label_counts.get(0,0)/len(y)*100:.2f}%)")
    _log(f"  不平衡比 = {label_counts.get(1,0)/max(label_counts.get(0,1),1):.3f} : 1")
    _log("  README 声称: 13,286 条 / 50.8% 活性")

    # 异常值提示
    over = int((df[pic_col] > 12).sum())
    if over:
        _log(f"  ⚠️ pIC50 > 12 的记录 {over} 条（>1 pM，物理上不太可能），保留但记录在案。")

    _section("3. 特征计算（复刻部署路径）")
    if not FEATURE_NAMES_JSON.exists():
        _log(f"[失败] 缺少 {FEATURE_NAMES_JSON}")
        return 1
    names = json.loads(FEATURE_NAMES_JSON.read_text(encoding="utf-8"))
    _log(f"  12 个特征: {names}")

    t0 = time.perf_counter()
    X, fdiag = compute_features(df[smi_col].astype(str).tolist(), names)
    feat_secs = time.perf_counter() - t0
    _log(f"  X shape = {X.shape}  耗时 {feat_secs:.1f}s")
    _log(f"  无法解析 SMILES: {fdiag['unparsable_smiles']}")
    _log(f"  静默填 0 总次数: {fdiag['zero_fill_total']}")
    if fdiag["zero_fill_counts"]:
        for k, v in fdiag["zero_fill_counts"].items():
            _log(f"       {k}: {v}")

    hp = dict(DEFAULT_HPARAMS)
    hp["random_state"] = args.seed

    _section(f"4. {args.cv_folds} 折分层交叉验证")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds: List[Dict[str, Any]] = []
    _log(f"  {'fold':<6} {'n_train':>8} {'n_val':>6} {'AUC':>8} {'ACC':>8} {'BAcc':>8} {'MCC':>8} {'秒':>7}")
    _log(f"  {'-'*6} {'-'*8} {'-'*6} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*7}")

    for i, (tr, va) in enumerate(skf.split(X, y), start=1):
        mdl = RandomForestClassifier(**hp)
        t1 = time.perf_counter()
        mdl.fit(X[tr], y[tr])
        secs = time.perf_counter() - t1
        prob = mdl.predict_proba(X[va])[:, 1]
        m = metrics_block(y[va], prob)
        m.update({"fold": i, "n_train": int(len(tr)), "n_val": int(len(va)),
                  "train_seconds": round(secs, 2)})
        folds.append(m)
        _log(f"  {i:<6} {len(tr):>8} {len(va):>6} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} "
             f"{m['balanced_accuracy']:>8.4f} {m['mcc']:>8.4f} {secs:>7.1f}")

    cv: Dict[str, Any] = {"n_splits": args.cv_folds, "folds": folds}
    for key in ("roc_auc", "accuracy", "balanced_accuracy", "mcc", "f1", "precision",
                "recall", "pr_auc", "specificity"):
        vals = np.array([f[key] for f in folds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            cv[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    _log("")
    _log(f"  >>> CV ROC-AUC    : {cv['roc_auc']['mean']:.4f} ± {cv['roc_auc']['std']:.4f}")
    _log(f"  >>> CV Accuracy   : {cv['accuracy']['mean']:.4f} ± {cv['accuracy']['std']:.4f}")
    _log(f"  >>> CV BalancedAcc: {cv['balanced_accuracy']['mean']:.4f} ± {cv['balanced_accuracy']['std']:.4f}")
    _log(f"  >>> CV MCC        : {cv['mcc']['mean']:.4f} ± {cv['mcc']['std']:.4f}")

    _section("5. 独立留出测试集")
    from sklearn.model_selection import train_test_split

    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )
    final_mdl = RandomForestClassifier(**hp)
    final_mdl.fit(Xtr, ytr)
    prob_te = final_mdl.predict_proba(Xte)[:, 1]
    fm = metrics_block(yte, prob_te)
    fm.update({"n_train": int(len(Xtr)), "n_test": int(len(Xte)),
               "positive_rate_train": float(ytr.mean()),
               "positive_rate_test": float(yte.mean())})
    _log(f"  n_train={fm['n_train']} n_test={fm['n_test']}  pos_rate={fm['positive_rate_test']:.4f}")
    _log(f"  AUC={fm['roc_auc']:.4f}  ACC={fm['accuracy']:.4f}  "
         f"BAcc={fm['balanced_accuracy']:.4f}  MCC={fm['mcc']:.4f}")
    _log(f"  混淆矩阵: {fm['confusion']}")
    _log(f"  多数类基线 ACC = {ytr.mean():.4f}  →  模型净增益 = {fm['accuracy'] - ytr.mean():+.4f}")

    _section("6. 保存产物")
    import joblib

    joblib.dump(final_mdl, MODEL_OUT)
    _log(f"  模型: {MODEL_OUT} ({MODEL_OUT.stat().st_size/1024:.1f} KB)")

    elapsed = time.perf_counter() - t_start
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_rf_legacy_protocol.py",
        "generated_by": "改进总控代理 / P0-01b",
        "purpose": "复现 README 声称的 RF 指标（median split 协议）",
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "data": {
            "csv": _rel(csv_path),
            "csv_sha256": _sha256_file(csv_path),
            "n_rows": int(len(df)),
            "smiles_column": smi_col,
            "pic50_column": pic_col,
            "labeling": f"median split: pIC50 >= {threshold:.4f} -> active",
            "threshold": threshold,
            "label_counts": label_counts,
            "positive_rate": float(y.mean()),
            "imbalance_ratio": float(label_counts.get(1, 0) / max(label_counts.get(0, 1), 1)),
            "pic50_over_12_count": int((df[pic_col] > 12).sum()),
            "pic50_describe": {
                k: float(v) for k, v in df[pic_col].describe().items()
            },
        },
        "features": {
            "names": names,
            "n_features": len(names),
            "computation": "replicated from real_predictor.smiles_to_features (silent zero-fill)",
            "diagnostics": fdiag,
            "compute_seconds": round(feat_secs, 1),
        },
        "hparams": {k: v for k, v in hp.items() if k != "n_jobs"},
        "protocol": {
            "cv": f"StratifiedKFold(n_splits={args.cv_folds}, shuffle=True, random_state={args.seed})",
            "test": f"train_test_split(test_size={args.test_size}, stratify=y, random_state={args.seed})",
            "labeling": "median_split",
            "note": (
                "median split 给出 50:50 平衡标签，Accuracy 与 Balanced Accuracy 接近，"
                "不存在类别不平衡虚高问题。这与 train_rf.py（阈值+丢弃中间带，86% 活性）"
                "是两套不同的标注策略，不可混用比较。"
            ),
        },
        "cv_results": cv,
        "final_test_results": fm,
        "comparison_to_readme": {
            "sample_size": {"claimed": 13286, "reproduced": int(len(df)),
                            "match": int(len(df)) == 13286},
            "positive_rate": {"claimed": 0.508, "reproduced": float(y.mean()),
                              "match": abs(float(y.mean()) - 0.508) < 0.005},
            "rf_cv_auc": {"claimed": 0.867, "claimed_std": 0.005,
                          "reproduced": cv["roc_auc"]["mean"], "reproduced_std": cv["roc_auc"]["std"],
                          "match": abs(cv["roc_auc"]["mean"] - 0.867) < 0.01},
            "rf_cv_accuracy": {"claimed": 0.782, "claimed_std": 0.005,
                               "reproduced": cv["accuracy"]["mean"], "reproduced_std": cv["accuracy"]["std"],
                               "match": abs(cv["accuracy"]["mean"] - 0.782) < 0.01},
        },
        "artifacts": {
            "model": _rel(MODEL_OUT),
            "model_sha256": _sha256_file(MODEL_OUT),
        },
        "runtime_seconds": round(elapsed, 2),
    }

    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"  指标: {METRICS_JSON}")

    _section("完成 —— 与 README 声称值对照")
    c = payload["comparison_to_readme"]
    auc_r, auc_s = c["rf_cv_auc"]["reproduced"], c["rf_cv_auc"]["reproduced_std"]
    acc_r, acc_s = c["rf_cv_accuracy"]["reproduced"], c["rf_cv_accuracy"]["reproduced_std"]

    def _mark(ok: bool) -> str:
        return "一致" if ok else "不符"

    _log(f"  {'项':<14} {'README 声称':>18} {'本次复现':>22} {'判定':>6}")
    _log(f"  {'-'*14} {'-'*18} {'-'*22} {'-'*6}")
    _log(f"  {'样本量':<14} {c['sample_size']['claimed']:>18,} "
         f"{c['sample_size']['reproduced']:>22,} {_mark(c['sample_size']['match']):>6}")
    _log(f"  {'活性比例':<14} {c['positive_rate']['claimed']*100:>17.1f}% "
         f"{c['positive_rate']['reproduced']*100:>21.2f}% {_mark(c['positive_rate']['match']):>6}")
    _log(f"  {'RF CV AUC':<14} {'0.867 ± 0.005':>18} "
         f"{f'{auc_r:.4f} ± {auc_s:.4f}':>22} {_mark(c['rf_cv_auc']['match']):>6}")
    _log(f"  {'RF CV Acc':<14} {'0.782 ± 0.005':>18} "
         f"{f'{acc_r:.4f} ± {acc_s:.4f}':>22} {_mark(c['rf_cv_accuracy']['match']):>6}")
    _log("")
    _log(f"  总耗时: {elapsed:.1f}s")
    _log(SEP)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        _log("\n中断。")
        sys.exit(130)
    except Exception as exc:
        import traceback

        _log("")
        _log("=" * 78)
        _log(f"[失败] {type(exc).__name__}: {exc}")
        _log("=" * 78)
        traceback.print_exc()
        sys.exit(1)
