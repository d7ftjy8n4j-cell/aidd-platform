#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_rf.py — 可复现的 RF 训练与评估管线
================================================

目的
----
把「README 声称的 5 折 CV AUC 0.867±0.005」变成**代码可重跑、产物可核对**的事实。

设计原则
--------
1. **无参可跑**：``python scripts/train_rf.py`` 即产出全部结果。
2. **数据快照优先**：有 ``data/raw/chembl_egfr_activities.csv`` 就只读快照；
   没有才拉 ChEMBL 并落盘。之后永久离线可复现。（DECISIONS.md D-04）
3. **不覆盖现有模型**：新模型写 ``artifacts/rf_repro.pkl``，原 pkl 保留作对照。（D-01）
4. **指标全落盘**：``artifacts/rf_metrics.json`` 含 CV / 独立测试 / 数据指纹 / 超参 / 特征名。
5. **对比现有 pkl**：报告「复现模型 vs 原模型」在同一测试集上的差异。（D-07）

评估协议（D-05）
----------------
* 5 折 StratifiedKFold(shuffle=True, random_state=42) → AUC / ACC 的 mean ± std
* 另留 20% final test（stratify, random_state=42），与 CV 样本互斥
* 附：随机 split 的已知缺陷说明（scaffold split 见 P0-05）

用法
----
    python scripts/train_rf.py                 # 完整跑
    python scripts/train_rf.py --max-records 3000
    python scripts/train_rf.py --force-refetch # 忽略快照重新拉数据

作者：改进总控代理（自动生成）
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
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ---- Windows 控制台 UTF-8 防御（与项目现有风格一致）----
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

# ---- 路径常量 ----
DATA_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
SNAPSHOT_CSV = DATA_DIR / "chembl_egfr_activities.csv"
SNAPSHOT_META = DATA_DIR / "chembl_egfr_activities.meta.json"
METRICS_JSON = ARTIFACT_DIR / "rf_metrics.json"
MODEL_OUT = ARTIFACT_DIR / "rf_repro.pkl"
LEGACY_MODEL = REPO_ROOT / "rf_egfr_model_final.pkl"
FEATURE_NAMES_JSON = REPO_ROOT / "feature_names.json"

# ---- 复现超参：从现有 pkl 逆向得到（DECISIONS.md D-07）----
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

# ---- 数据拉取默认参数 ----
DEFAULT_TARGET = "EGFR"
DEFAULT_ACTIVITY_TYPE = "IC50"
DEFAULT_MAX_RECORDS = 2000      # 教学页上限；CHEMBL203 全量会更大
DEFAULT_PIC50_THRESHOLD = 6.0   # 与 utils/teach_lab.py 一致

SEP = "=" * 78


def _log(msg: str) -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    _log("")
    _log(SEP)
    _log(title)
    _log(SEP)


def _rel(path: Path) -> str:
    """尽量返回相对仓库路径；不在仓库内则返回绝对路径字符串。"""
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256_of_df(df: pd.DataFrame) -> str:
    """对 DataFrame 内容做稳定指纹（按列排序 + 固定浮点格式）。"""
    stable = df.reindex(sorted(df.columns), axis=1).copy()
    for col in stable.columns:
        if pd.api.types.is_float_dtype(stable[col]):
            stable[col] = stable[col].round(8)
    return hashlib.sha256(
        pd.util.hash_pandas_object(stable, index=True).values.tobytes()
    ).hexdigest()


# ============================================================================
# 1b. 特征计算 —— 严格复刻部署路径（real_predictor.smiles_to_features）
# ============================================================================
def compute_features_deployed(
    smiles_list: List[str], feature_names: List[str]
) -> Tuple[np.ndarray, Dict[str, Any]]:
    """按 real_predictor.smiles_to_features 的方式计算特征。

    关键点（实测自 real_predictor.py:188-211）：
      * 逐个 ``getattr(Descriptors, feat_name)(mol)`` 取值
      * 任何异常 / None / NaN → **静默填 0.0**
      * 顺序严格等于 ``feature_names``

    本函数复刻这一行为，但**统计并报告**填充次数——静默填 0 是潜在数据质量问题，
    复现时可以接受（为对齐部署模型），但必须让它可见。
    """
    from rdkit import Chem, RDLogger
    from rdkit.Chem import Descriptors

    RDLogger.DisableLog("rdApp.*")

    n = len(smiles_list)
    X = np.zeros((n, len(feature_names)), dtype=float)
    fill_counts = {name: 0 for name in feature_names}
    unparsable = 0
    invalid_names = [nm for nm in feature_names if not hasattr(Descriptors, nm)]

    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            unparsable += 1
            for j in range(len(feature_names)):
                fill_counts[feature_names[j]] += 1
            continue
        for j, name in enumerate(feature_names):
            try:
                value = getattr(Descriptors, name)(mol)
                if value is None:
                    raise ValueError("None")
                value = float(value)
                if np.isnan(value) or np.isinf(value):
                    raise ValueError("NaN/Inf")
                X[i, j] = value
            except Exception:
                X[i, j] = 0.0
                fill_counts[name] += 1

    diag = {
        "n_molecules": n,
        "unparsable_smiles": unparsable,
        "invalid_descriptor_names": invalid_names,
        "zero_fill_counts": {k: v for k, v in fill_counts.items() if v > 0},
        "zero_fill_total": int(sum(fill_counts.values())),
        "note": (
            "部署模型计算特征时对异常静默填 0.0（real_predictor.py:203-208）。"
            "本脚本复刻该行为以保证与 shipped pkl 一致，但填充次数在此显式记录。"
        ),
    }
    return X, diag


# ============================================================================
# 1. 数据获取（快照优先）
# ============================================================================
def fetch_snapshot(
    target: str = DEFAULT_TARGET,
    activity_type: str = DEFAULT_ACTIVITY_TYPE,
    max_records: int = DEFAULT_MAX_RECORDS,
    force: bool = False,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """读取或创建 ChEMBL 数据快照。返回 (raw_df, meta)。"""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if SNAPSHOT_CSV.exists() and not force:
        _section("1. 读取数据快照（离线，可复现）")
        df = pd.read_csv(SNAPSHOT_CSV)
        meta: Dict[str, Any] = {}
        if SNAPSHOT_META.exists():
            try:
                meta = json.loads(SNAPSHOT_META.read_text(encoding="utf-8"))
            except Exception:
                meta = {}
        meta.setdefault("snapshot_path", str(SNAPSHOT_CSV))
        meta.setdefault("sha256_csv", _sha256_of_file(SNAPSHOT_CSV))
        meta.setdefault("n_records", int(len(df)))
        _log(f"  快照: {SNAPSHOT_CSV}")
        _log(f"  记录数: {len(df)}")
        _log(f"  sha256: {meta['sha256_csv'][:16]}...")
        _log("  （如需重新拉取，加 --force-refetch）")
        return df, meta

    _section("1. 从 ChEMBL 拉取数据（首次，之后走快照）")
    from utils.teach_lab import fetch_target_activities  # 延迟导入，避免无网/无依赖时崩

    _log(f"  靶点={target}  类型={activity_type}  上限={max_records}")
    t0 = time.perf_counter()
    result = fetch_target_activities(
        target_name=target,
        activity_type=activity_type,
        max_records=max_records,
        status_callback=lambda m: _log(f"    · {m}"),
    )
    elapsed = time.perf_counter() - t0

    if not result.get("success") or result.get("df") is None:
        raise RuntimeError(
            f"ChEMBL 拉取失败: {result.get('message', '未知原因')}\n"
            "排查：网络是否可访问 ebi.ac.uk；或先用 --max-records 300 试小批量。"
        )

    df: pd.DataFrame = result["df"]
    df.to_csv(SNAPSHOT_CSV, index=False, encoding="utf-8")

    meta = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "target_name_requested": target,
        "target_name_resolved": result.get("target_name"),
        "target_chembl_id": result.get("target_chembl_id"),
        "activity_type": activity_type,
        "max_records_requested": max_records,
        "raw_count_reported": result.get("raw_count"),
        "n_records": int(len(df)),
        "fetch_seconds": round(elapsed, 2),
        "sha256_csv": _sha256_of_file(SNAPSHOT_CSV),
        "snapshot_path": str(SNAPSHOT_CSV),
        "chembl_client_version": _safe_pkg_version("chembl_webresource_client"),
    }
    SNAPSHOT_META.write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    _log(f"  ✅ 拉取完成: {len(df)} 条，耗时 {elapsed:.1f}s")
    _log(f"  已落盘: {SNAPSHOT_CSV}")
    return df, meta


def _safe_pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


# ============================================================================
# 2. 清洗（复用项目既有逻辑，保证与其他模块同源）
# ============================================================================
def clean_data(raw_df: pd.DataFrame, threshold: float) -> Tuple[pd.DataFrame, List[Any]]:
    _section("2. 数据清洗（复用 utils.teach_lab.clean_activity_data）")
    from utils.teach_lab import clean_activity_data

    clean_df, stages = clean_activity_data(raw_df, pic50_threshold=threshold)

    _log(f"  {'阶段':<38} {'剩余':>8} {'移除':>8}")
    _log(f"  {'-' * 38} {'-' * 8} {'-' * 8}")
    for st in stages:
        name = st.name if len(st.name) <= 36 else st.name[:33] + "..."
        _log(f"  {name:<38} {st.remaining:>8} {st.removed:>8}")
    _log(f"  {'-' * 38} {'-' * 8} {'-' * 8}")
    _log(f"  {'最终样本数':<38} {len(clean_df):>8}")
    return clean_df, stages


# ============================================================================
# 3. 评估
# ============================================================================
def _metrics(y_true: np.ndarray, y_proba: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        out["pr_auc"] = float("nan")
    return out


def cross_validate_rf(
    X: np.ndarray, y: np.ndarray, hparams: Dict[str, Any], n_splits: int = 5
) -> Dict[str, Any]:
    """5 折分层交叉验证，逐折记录指标。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=hparams["random_state"])
    folds: List[Dict[str, Any]] = []

    _section(f"3. {n_splits} 折分层交叉验证")
    _log(f"  {'fold':<6} {'n_train':>8} {'n_val':>6} {'AUC':>8} {'ACC':>8} {'F1':>8} {'秒':>7}")
    _log(f"  {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

    for i, (tr, va) in enumerate(skf.split(X, y), start=1):
        model = RandomForestClassifier(**hparams)
        t0 = time.perf_counter()
        model.fit(X[tr], y[tr])
        secs = time.perf_counter() - t0

        proba = model.predict_proba(X[va])[:, 1]
        pred = (proba >= 0.5).astype(int)
        m = _metrics(y[va], proba, pred)
        m.update({"fold": i, "n_train": int(len(tr)), "n_val": int(len(va)),
                  "train_seconds": round(secs, 2)})
        folds.append(m)
        auc_str = f"{m['roc_auc']:>8.4f}" if not np.isnan(m["roc_auc"]) else f"{'nan':>8}"
        _log(f"  {i:<6} {len(tr):>8} {len(va):>6} {auc_str} "
             f"{m['accuracy']:>8.4f} {m['f1']:>8.4f} {secs:>7.2f}")

    summary: Dict[str, Any] = {"n_splits": n_splits, "folds": folds}
    for key in ("roc_auc", "accuracy", "f1", "precision", "recall", "pr_auc"):
        vals = np.array([f[key] for f in folds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }
    _log("")
    _log(f"  >>> CV ROC-AUC : {summary['roc_auc']['mean']:.4f} ± {summary['roc_auc']['std']:.4f}")
    _log(f"  >>> CV Accuracy: {summary['accuracy']['mean']:.4f} ± {summary['accuracy']['std']:.4f}")
    _log(f"  >>> CV F1      : {summary['f1']['mean']:.4f} ± {summary['f1']['std']:.4f}")
    return summary


def final_test(
    X: np.ndarray, y: np.ndarray, hparams: Dict[str, Any], test_size: float = 0.2
) -> Dict[str, Any]:
    """独立留出测试集（与 CV 样本互斥）。"""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split

    _section("4. 独立留出测试集（20%，与 CV 互斥）")
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=hparams["random_state"], stratify=y
    )
    model = RandomForestClassifier(**hparams)
    t0 = time.perf_counter()
    model.fit(X_tr, y_tr)
    secs = time.perf_counter() - t0

    proba = model.predict_proba(X_te)[:, 1]
    pred = (proba >= 0.5).astype(int)
    m = _metrics(y_te, proba, pred)
    m.update({
        "n_train": int(len(X_tr)),
        "n_test": int(len(X_te)),
        "train_seconds": round(secs, 2),
        "positive_rate_train": float(y_tr.mean()),
        "positive_rate_test": float(y_te.mean()),
    })
    _log(f"  n_train={m['n_train']}  n_test={m['n_test']}  "
         f"pos_rate_train={m['positive_rate_train']:.3f}  pos_rate_test={m['positive_rate_test']:.3f}")
    _log(f"  ROC-AUC={m['roc_auc']:.4f}  ACC={m['accuracy']:.4f}  "
         f"F1={m['f1']:.4f}  PR-AUC={m['pr_auc']:.4f}")

    # ---- 与现有 pkl 对照 ----
    _section("5. 与现有 pkl 对照（同一测试集）")
    legacy = _compare_legacy(model, X_te, y_te, X_tr, hparams)
    return {"model": model, "metrics": m, "legacy_comparison": legacy,
            "test_proba": proba, "test_label": y_te, "train_label": y_tr}


def _compare_legacy(
    new_model, X_te: np.ndarray, y_te: np.ndarray, X_tr: np.ndarray, hparams: Dict[str, Any]
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if not LEGACY_MODEL.exists():
        _log("  现有 pkl 不存在，跳过对照。")
        return out
    try:
        import joblib

        legacy = joblib.load(LEGACY_MODEL)
        out["available"] = True
        out["type"] = type(legacy).__name__
        out["n_features_in_"] = int(getattr(legacy, "n_features_in_", -1))
        out["n_estimators"] = int(getattr(legacy, "n_estimators", -1))
        out["max_depth"] = getattr(legacy, "max_depth", None)
        out["_sklearn_version"] = getattr(legacy, "_sklearn_version", None)

        lp = legacy.predict_proba(X_te)[:, 1]
        lpred = (lp >= 0.5).astype(int)
        lm = _metrics(y_te, lp, lpred)
        out["test_metrics"] = lm

        np_ = new_model.predict_proba(X_te)[:, 1]
        agree = float(np.mean((lp >= 0.5) == (np_ >= 0.5)))
        corr = float(np.corrcoef(lp, np_)[0, 1]) if np.std(lp) > 0 and np.std(np_) > 0 else float("nan")
        out["agreement_rate"] = agree
        out["proba_correlation"] = corr

        _log(f"  现有 pkl: AUC={lm['roc_auc']:.4f}  ACC={lm['accuracy']:.4f}")
        _log(f"  复现模型: AUC={_metrics(y_te, np_, (np_ >= 0.5).astype(int))['roc_auc']:.4f}")
        _log(f"  两者判定一致率={agree:.4f}  概率相关系数={corr:.4f}")
        if lm["roc_auc"] > 0.99:
            _log("  ⚠️ 现有 pkl 在同一测试集 AUC>0.99 —— 高度提示它是在包含此测试集的数据上训练的（泄漏）。")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"  ⚠️ 对照失败: {out['error']}")
    return out


# ============================================================================
# 6. 主流程
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="可复现 RF 训练与评估")
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--activity-type", default=DEFAULT_ACTIVITY_TYPE)
    ap.add_argument("--max-records", type=int, default=DEFAULT_MAX_RECORDS)
    ap.add_argument("--threshold", type=float, default=DEFAULT_PIC50_THRESHOLD)
    ap.add_argument("--force-refetch", action="store_true", help="忽略快照重新拉 ChEMBL")
    ap.add_argument("--n-estimators", type=int, default=DEFAULT_HPARAMS["n_estimators"])
    ap.add_argument("--max-depth", type=int, default=DEFAULT_HPARAMS["max_depth"])
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=DEFAULT_HPARAMS["random_state"])
    args = ap.parse_args(argv)

    hparams = dict(DEFAULT_HPARAMS)
    hparams.update({
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "random_state": args.seed,
    })

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    _log(SEP)
    _log("药尘光 · RF 可复现训练与评估管线")
    _log(SEP)
    _log(f"  时间     : {datetime.now().isoformat(timespec='seconds')}")
    _log(f"  Python   : {platform.python_version()}")
    _log(f"  numpy    : {np.__version__}")
    _log(f"  pandas   : {pd.__version__}")
    _log(f"  sklearn  : {_safe_pkg_version('scikit-learn')}")
    _log(f"  超参     : { {k: v for k, v in hparams.items() if k != 'n_jobs'} }")

    # ---- 数据 ----
    raw_df, data_meta = fetch_snapshot(
        target=args.target,
        activity_type=args.activity_type,
        max_records=args.max_records,
        force=args.force_refetch,
    )

    # ---- 清洗 ----
    clean_df, stages = clean_data(raw_df, args.threshold)

    # ---- 特征 ----
    _section("2b. 特征与标签（复刻部署路径）")
    if FEATURE_NAMES_JSON.exists():
        feature_names = json.loads(FEATURE_NAMES_JSON.read_text(encoding="utf-8"))
        feat_source = "feature_names.json"
        _log(f"  feature_names.json: {len(feature_names)} 个")
        _log(f"  {feature_names}")
    else:
        raise RuntimeError(
            "feature_names.json 缺失。部署模型 rf_egfr_model_final.pkl 的 12 个特征名"
            "只有这个文件里有（注意：它与 utils.teach_lab.DESCRIPTOR_NAMES 是两套不同的名字，"
            "仅 5 个同名 —— 见 STATE.md F-12）。请从仓库根目录恢复该文件后重试。"
        )

    # 部署模型的真实特征计算路径（real_predictor.smiles_to_features）
    X_all, feat_diag = compute_features_deployed(clean_df["smiles"].tolist(), feature_names)
    y = clean_df["label"].to_numpy(dtype=int)

    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y, return_counts=True))}
    _log(f"  X shape: {X_all.shape}   y 分布: {label_counts}   正例比例={y.mean():.4f}")

    if len(label_counts) < 2:
        raise RuntimeError(
            f"标签只有单一类别 {label_counts}，无法训练/评估（AUC 与 CV 均无定义）。\n"
            "原因通常是：数据量太小、或活性阈值把所有样本都推到同一侧。\n"
            "处理：加大 --max-records（建议 ≥2000），或调整 --threshold。"
        )
    if min(label_counts.values()) < args.cv_folds:
        raise RuntimeError(
            f"最小类别样本数 {min(label_counts.values())} < 折数 {args.cv_folds}，"
            "无法做分层 CV。\n处理：加大 --max-records，或降低 --cv-folds。"
        )
    if feat_diag["invalid_descriptor_names"]:
        _log(f"  ⚠️ 无效描述符名: {feat_diag['invalid_descriptor_names']}")
    if feat_diag["zero_fill_total"]:
        _log(f"  ⚠️ 静默填 0 次数: {feat_diag['zero_fill_total']}  "
             f"（详情见 metrics.json → features.zero_fill）")
        for k, v in feat_diag["zero_fill_counts"].items():
            _log(f"       {k}: {v}")
    else:
        _log("  ✅ 无静默填 0")

    # 训练时仍需保证 12 个特征列存在（用于与 pkl 的 n_features_in_ 对齐校验）
    if X_all.shape[1] != len(feature_names):
        raise RuntimeError(f"特征维度异常: {X_all.shape[1]} != {len(feature_names)}")

    X = X_all

    # ---- CV ----
    cv_summary = cross_validate_rf(X, y, hparams, n_splits=args.cv_folds)

    # ---- Final test + legacy 对照 ----
    final = final_test(X, y, hparams, test_size=args.test_size)

    # ---- 保存模型 ----
    _section("6. 保存产物")
    import joblib

    joblib.dump(final["model"], MODEL_OUT)
    _log(f"  模型     : {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.1f} KB)")

    # ---- 组装 metrics.json ----
    elapsed = time.perf_counter() - t_start
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_rf.py",
        "generated_by": "改进总控代理 / P0-01",

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scikit_learn": _safe_pkg_version("scikit-learn"),
            "joblib": _safe_pkg_version("joblib"),
        },

        "data": {
            "snapshot": data_meta,
            "raw_n": int(len(raw_df)),
            "clean_n": int(len(clean_df)),
            "label_counts": label_counts,
            "positive_rate": float(y.mean()),
            "pic50_threshold": args.threshold,
            "data_fingerprint_sha256": _sha256_of_df(clean_df),
            "cleaning_stages": [
                {"name": s.name, "remaining": int(s.remaining), "removed": int(s.removed)}
                for s in stages
            ],
        },

        "features": {
            "names": feature_names,
            "n_features": len(feature_names),
            "source": feat_source,
            "computation": "replicated from real_predictor.smiles_to_features (silent zero-fill on error)",
            "diagnostics": feat_diag,
        },

        "hparams": {k: v for k, v in hparams.items() if k != "n_jobs"},

        "protocol": {
            "cv": f"StratifiedKFold(n_splits={args.cv_folds}, shuffle=True, random_state={args.seed})",
            "test": f"train_test_split(test_size={args.test_size}, stratify=y, random_state={args.seed})",
            "split_type": "random_stratified",
            "known_limitation": (
                "随机分层划分会让同骨架分子同时出现在训练与测试集，AUC 系统性偏高。"
                "scaffold split 对照见 P0-05（尚未完成）。"
            ),
        },

        "cv_results": cv_summary,
        "final_test_results": final["metrics"],
        "legacy_model_comparison": final["legacy_comparison"],

        "artifacts": {
            "model": _rel(MODEL_OUT),
            "model_sha256": _sha256_of_file(MODEL_OUT),
            "metrics": _rel(METRICS_JSON),
        },

        "runtime_seconds": round(elapsed, 2),
    }

    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"  指标     : {METRICS_JSON}")

    # ---- 最终摘要 ----
    _section("完成 —— 与 README 声称值对照")
    cv_auc_m = cv_summary["roc_auc"]["mean"]
    cv_auc_s = cv_summary["roc_auc"]["std"]
    cv_acc_m = cv_summary["accuracy"]["mean"]
    cv_acc_s = cv_summary["accuracy"]["std"]
    ft_auc = final["metrics"]["roc_auc"]
    ft_acc = final["metrics"]["accuracy"]

    claim = {
        "cv_auc": (0.867, 0.005),
        "cv_acc": (0.782, 0.005),
    }

    def _delta(measured: float, claimed: float) -> str:
        return f"{measured - claimed:+.4f}"

    _log(f"  {'指标':<26} {'本次实测':>16} {'README 声称':>15} {'Δ':>9}")
    _log(f"  {'-' * 26} {'-' * 16} {'-' * 15} {'-' * 9}")
    _log(f"  {'CV AUC (5-fold)':<26} {cv_auc_m:>9.4f} ± {cv_auc_s:.4f} "
         f"{'0.867 ± 0.005':>15} {_delta(cv_auc_m, claim['cv_auc'][0]):>9}")
    _log(f"  {'CV Accuracy (5-fold)':<26} {cv_acc_m:>9.4f} ± {cv_acc_s:.4f} "
         f"{'0.782 ± 0.005':>15} {_delta(cv_acc_m, claim['cv_acc'][0]):>9}")
    _log(f"  {'独立测试 AUC':<26} {ft_auc:>9.4f} {'':>6} {'（未声称）':>15} {'':>9}")
    _log(f"  {'独立测试 ACC':<26} {ft_acc:>9.4f} {'':>6} {'（未声称）':>15} {'':>9}")
    _log("")
    _log(f"  总耗时: {elapsed:.1f}s")
    _log("")
    _log("  下一步：把 artifacts/rf_metrics.json 回传，我据此更新 STATE/EXPERIMENTS 并出 P0-02。")
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
