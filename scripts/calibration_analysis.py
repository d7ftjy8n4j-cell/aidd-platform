#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/calibration_analysis.py — P1-01 概率校准分析（legacy 协议）
====================================================================

目的
----
回答审稿人 Q10 / 教学核心问题：「界面上的 0.6，是不是 60% 意义上的 60%？」

方法
----
A. 复现模型（legacy 协议，EXP-010b 同一套 5 折划分）：
   1. 5 折 out-of-fold（OOF）概率 —— 全部 13,286 个样本都有一个「没见过它」的模型给的分数；
   2. Brier score / ECE（10 等宽桶）/ MCE / 可靠性曲线分桶数据；
   3. 后处理校准对照：Platt（logistic）与 Isotonic，**2 折交叉拟合**（一半拟合、另一半评估，
      轮换），杜绝「用评估集拟合校准器」的乐观偏差。
B. 部署 pkl（rf_egfr_model_final.pkl）诊断：
   在数据集 A 上直接评估 Brier/ECE。⚠️ 口径警示：无法重跑 pkl 的训练划分，
   评估集与其训练数据大概率重叠，**B 部分 仅作 UI 概率语义诊断，不进科研口径**。

产物
----
artifacts/calibration_metrics.json
artifacts/calibration_reliability_diagram.png（两联图，图内英文标签避免字体缺字）

用法
----
    python scripts/calibration_analysis.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
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

from train_rf_legacy_protocol import compute_features  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_CSV = REPO_ROOT / "data" / "raw" / "egfr_all_ic50_full.csv"
DEPLOYED_PKL = REPO_ROOT / "rf_egfr_model_final.pkl"
OUT_JSON = ARTIFACT_DIR / "calibration_metrics.json"
OUT_PNG = ARTIFACT_DIR / "calibration_reliability_diagram.png"
HPARAMS = {"n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
           "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True}
N_BINS = 10
SEED = 42

SEP = "=" * 78


def _log(m: str) -> None:
    print(m, flush=True)


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def brier_score(y_true: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean((p - y_true) ** 2))


def reliability_bins(y_true: np.ndarray, p: np.ndarray, n_bins: int = N_BINS) -> List[Dict[str, float]]:
    """等宽桶 [i/n, (i+1)/n)。返回每桶样本数、平均预测概率、实际正例频率。"""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1], right=False), 0, n_bins - 1)
    bins = []
    for b in range(n_bins):
        mask = idx == b
        n = int(mask.sum())
        bins.append({
            "bin_low": float(edges[b]), "bin_high": float(edges[b + 1]),
            "n": n,
            "mean_predicted": float(p[mask].mean()) if n else None,
            "fraction_positive": float(y_true[mask].mean()) if n else None,
        })
    return bins


def ece_from_bins(bins: List[Dict[str, float]]) -> Tuple[float, float]:
    """返回 (ECE, MCE)。空桶跳过。"""
    total = sum(b["n"] for b in bins)
    ece = 0.0
    mce = 0.0
    for b in bins:
        if not b["n"]:
            continue
        gap = abs(b["fraction_positive"] - b["mean_predicted"])
        ece += b["n"] / total * gap
        mce = max(mce, gap)
    return float(ece), float(mce)


def fit_calibrators(p_fit: np.ndarray, y_fit: np.ndarray) -> Dict[str, Any]:
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LogisticRegression

    platt = LogisticRegression(solver="lbfgs")
    platt.fit(p_fit.reshape(-1, 1), y_fit)
    a, b = float(platt.coef_[0][0]), float(platt.intercept_[0])

    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p_fit, y_fit)
    return {"platt": {"a": a, "b": b, "model": platt}, "isotonic": {"model": iso}}


def apply_platt(cal: Dict[str, Any], p: np.ndarray) -> np.ndarray:
    from scipy.special import expit
    return expit(cal["platt"]["a"] * p + cal["platt"]["b"])


def apply_isotonic(cal: Dict[str, Any], p: np.ndarray) -> np.ndarray:
    return cal["isotonic"]["model"].predict(p)


def cross_fit_calibration(p: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """2 折交叉拟合：每半数据用另半拟合的校准器变换，避免自评乐观。"""
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
    out = {"platt": np.empty_like(p), "isotonic": np.empty_like(p)}
    fitted: List[Dict[str, Any]] = []
    for tr, te in skf.split(p, y):
        cal = fit_calibrators(p[tr], y[tr])
        fitted.append({
            "platt": {"a": cal["platt"]["a"], "b": cal["platt"]["b"]},
        })
        out["platt"][te] = apply_platt(cal, p[te])
        out["isotonic"][te] = apply_isotonic(cal, p[te])
    return {"calibrated": out, "fitted_params": fitted}


def evaluate_block(name: str, y_true: np.ndarray, p: np.ndarray) -> Dict[str, Any]:
    bins = reliability_bins(y_true, p)
    ece, mce = ece_from_bins(bins)
    block = {
        "n": int(len(p)),
        "brier": brier_score(y_true, p),
        "ece_10bin": ece,
        "mce_10bin": mce,
        "mean_predicted": float(p.mean()),
        "positive_rate": float(y_true.mean()),
        "bins": bins,
    }
    _log(f"  [{name}] Brier={block['brier']:.4f}  ECE={ece:.4f}  MCE={mce:.4f}  "
         f"mean(p)={block['mean_predicted']:.4f}  pos_rate={block['positive_rate']:.4f}")
    return block


def main() -> int:
    import joblib
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()

    _log(SEP)
    _log("药尘光 · P1-01 概率校准分析（legacy 协议）")
    _log(SEP)

    df = pd.read_csv(DATA_CSV)
    smiles_list = df["SMILES"].astype(str).tolist()
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    med = float(pic50.median())
    y = (pic50.fillna(-1).values >= med).astype(int)
    feature_names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X, feat_diag = compute_features(smiles_list, feature_names)
    _log(f"  数据 {len(df)} 条（median split {med:.4f}，正例率 {y.mean():.4f}），特征 {X.shape[1]} 维")

    # ---- A1. 5 折 OOF 概率（与 EXP-010b 同一划分） ----
    _log("\n[A1] 复现模型 5 折 OOF 概率（StratifiedKFold(5, shuffle, rs=42)）")
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y), dtype=float)
    for k, (tr, te) in enumerate(skf.split(X, y), 1):
        clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **HPARAMS)
        clf.fit(X[tr], y[tr])
        oof[te] = clf.predict_proba(X[te])[:, 1]
        _log(f"  fold {k}: n_train={len(tr)}  fold-AUC={roc_auc_score(y[te], oof[te]):.4f}")
    _log(f"  OOF AUC = {roc_auc_score(y, oof):.4f}（与 EXP-010b CV 0.8680 可比）")

    raw_block = evaluate_block("复现模型 OOF", y, oof)

    # ---- A2. 后处理校准（2 折交叉拟合） ----
    _log("\n[A2] 后处理校准（Platt / Isotonic，2 折交叉拟合防自评）")
    cal = cross_fit_calibration(oof, y)
    platt_block = evaluate_block("Platt 校准后", y, cal["calibrated"]["platt"])
    iso_block = evaluate_block("Isotonic 校准后", y, cal["calibrated"]["isotonic"])

    # ---- B. 部署 pkl 诊断 ----
    pkl_block = None
    pkl_auc = None
    if DEPLOYED_PKL.exists():
        _log("\n[B] 部署 pkl 诊断（⚠️ 评估集与其训练数据可能重叠，仅 UI 语义参考）")
        pkl_model = joblib.load(DEPLOYED_PKL)
        p_pkl = pkl_model.predict_proba(X)[:, 1]
        pkl_auc = float(roc_auc_score(y, p_pkl))
        pkl_block = evaluate_block("部署 pkl 全量", y, p_pkl)
        pkl_block["auc_on_dataset_a"] = pkl_auc
        pkl_block["caveat"] = ("无法重跑 pkl 的训练划分，评估集与其训练数据可能重叠；"
                               "本块仅诊断 UI 概率分布语义，不作为科研口径。")
        qs = np.quantile(p_pkl, [0.05, 0.25, 0.5, 0.75, 0.95])
        pkl_block["probability_quantiles"] = {"q05": float(qs[0]), "q25": float(qs[1]),
                                              "q50": float(qs[2]), "q75": float(qs[3]), "q95": float(qs[4])}
        _log(f"  概率分位数 p5/p25/50/75/p95 = {', '.join(f'{q:.3f}' for q in qs)}")

    # ---- 可靠性图（两联；英文标签避免中文字体缺字） ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharex=True, sharey=True)
        panels = [
            (axes[0], raw_block, platt_block, iso_block,
             "Repro model (5-fold OOF, n=13,286)"),
        ]
        if pkl_block is not None:
            panels.append((axes[1], pkl_block, None, None,
                           "Deployed pkl (diagnostic only)"))
        else:
            axes[1].axis("off")
        for ax, blk, pl, iso, title in panels:
            ax.plot([0, 1], [0, 1], "k--", lw=1, label="Perfect calibration")
            pts = [(b["mean_predicted"], b["fraction_positive"]) for b in blk["bins"] if b["n"]]
            ax.plot([x for x, _ in pts], [yy for _, yy in pts], "o-", color="#1f77b4",
                    label=f"Raw (ECE={blk['ece_10bin']:.3f})")
            if pl is not None:
                pts = [(b["mean_predicted"], b["fraction_positive"]) for b in pl["bins"] if b["n"]]
                ax.plot([x for x, _ in pts], [yy for _, yy in pts], "s--", color="#2ca02c",
                        label=f"Platt (ECE={pl['ece_10bin']:.3f})")
            if iso is not None:
                pts = [(b["mean_predicted"], b["fraction_positive"]) for b in iso["bins"] if b["n"]]
                ax.plot([x for x, _ in pts], [yy for _, yy in pts], "^--", color="#d62728",
                        label=f"Isotonic (ECE={iso['ece_10bin']:.3f})")
            ax.set_xlabel("Mean predicted probability")
            ax.set_ylabel("Observed active fraction")
            ax.set_title(title, fontsize=10)
            ax.legend(fontsize=8, loc="upper left")
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
        fig.suptitle("Reliability diagrams — YaoChenGuang EGFR RF (median-split protocol)",
                     fontsize=11)
        fig.tight_layout()
        fig.savefig(OUT_PNG, dpi=150)
        plt.close(fig)
        _log(f"\n  可靠性图: {OUT_PNG}")
    except Exception as exc:  # 绘图失败不阻断指标落盘
        _log(f"  ⚠️ 绘图失败（不影响 JSON 产物）: {type(exc).__name__}: {exc}")

    payload = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/calibration_analysis.py",
        "generated_by": "改进总控代理 / P1-01",
        "data": {"csv": "data/raw/egfr_all_ic50_full.csv", "csv_sha256": _sha256_file(DATA_CSV),
                 "n_rows": int(len(df)), "labeling": f"median_split (pIC50 >= {med:.4f})",
                 "positive_rate": float(y.mean())},
        "features": {"names": feature_names, "diagnostics": feat_diag},
        "hparams": {**HPARAMS, "random_state": SEED},
        "protocol": {
            "oof": "StratifiedKFold(5, shuffle=True, random_state=42)，与 EXP-010b 同划分",
            "calibration": "Platt / Isotonic，2 折交叉拟合（一半拟合一半评估，轮换）",
            "n_bins": N_BINS,
        },
        "repro_model": {"oof_auc": float(roc_auc_score(y, oof)), "raw": raw_block,
                        "platt_calibrated": platt_block, "isotonic_calibrated": iso_block,
                        "platt_params_crossfit": cal["fitted_params"]},
        "deployed_pkl_diagnostic": pkl_block,
        "runtime_seconds": round(time.perf_counter() - t0, 2),
    }
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"  指标  : {OUT_JSON}")
    _log(SEP)
    return 0


if __name__ == "__main__":
    sys.exit(main())
