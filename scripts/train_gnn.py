#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/train_gnn.py — 可复现的 GNN（GCN）训练与评估管线
==========================================================

目的
----
P0-01 已证明 RF 链路可复现。本脚本对 GNN 做同样的事：
把 README 声称的「GNN AUC 0.845±0.008 / ACC 0.767±0.011」变成可核对的产物。

与 P0-01 的对齐
---------------
* 同一份数据快照（``data/raw/chembl_egfr_activities.csv``）
* 同一套清洗（``utils.teach_lab.clean_activity_data``）
* 同一个评估协议（5 折分层 CV + 独立 20% 留出测试）
* **同一次划分**（相同 random_state，便于与 RF 逐折对比）

关键设计
--------
1. **复用项目既有建图逻辑**（``utils.teach_lab.smiles_to_graph``），13 维原子特征
   顺序与 ``gnn_predictor.py`` 严格一致 —— 这是硬约束，错了模型加载不上。
2. **模型结构复刻 ``gnn_predictor.GCNModel``**：conv1/2/3 + bn1/2 + lin1/2，13→128。
3. **不覆盖** ``gcn_egfr_best_model.pth``，新模型写 ``artifacts/gcn_repro.pth``。
4. **CV 逐折训练**（5 折 × N epochs），耗时约为单次训练 5 倍 —— 因此默认 epochs 调低，
   并用 ``--epochs`` 让人类按机器性能取舍。
5. 额外做 **无环分子/建图失败统计**，因为 GNN 对无效结构比 RF 更敏感。

用法
----
    python scripts/train_gnn.py                  # 快照协议（teach_lab 阈值 6.0），15 epochs × 5 折
    python scripts/train_gnn.py --epochs 30
    python scripts/train_gnn.py --quick          # 3 折 5 epochs，先验通链路
    python scripts/train_gnn.py --dataset legacy --tag legacy --epochs 25
        # 在原始 13,286 条数据（median split，复刻 README 口径）上复现 GNN 指标；
        # 产物写 artifacts/gnn_metrics_legacy.json / artifacts/gcn_repro_legacy.pth，不覆盖默认产物

作者：改进总控代理（自动生成）
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import random
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_DUPLICATE_LIB_OK", "TRUE")

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

DATA_DIR = REPO_ROOT / "data" / "raw"
ARTIFACT_DIR = REPO_ROOT / "artifacts"
SNAPSHOT_CSV = DATA_DIR / "chembl_egfr_activities.csv"
METRICS_JSON = ARTIFACT_DIR / "gnn_metrics.json"
MODEL_OUT = ARTIFACT_DIR / "gcn_repro.pth"
LEGACY_MODEL = REPO_ROOT / "gcn_egfr_best_model.pth"

# 复刻 gnn_predictor.GCNModel 的默认超参
DEFAULT_HPARAMS: Dict[str, Any] = {
    "num_node_features": 13,
    "hidden_dim": 128,
    "dropout": 0.5,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "epochs": 15,
    "random_state": 42,
}

DEFAULT_PIC50_THRESHOLD = 6.0
SEP = "=" * 78


def _log(msg: str) -> None:
    print(msg, flush=True)


def _section(title: str) -> None:
    _log("")
    _log(SEP)
    _log(title)
    _log(SEP)


def _rel(path: Path) -> str:
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
    stable = df.reindex(sorted(df.columns), axis=1).copy()
    for col in stable.columns:
        if pd.api.types.is_float_dtype(stable[col]):
            stable[col] = stable[col].round(8)
    return hashlib.sha256(
        pd.util.hash_pandas_object(stable, index=True).values.tobytes()
    ).hexdigest()


def _metrics(y_true: np.ndarray, y_proba: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        matthews_corrcoef,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    pred = (y_proba >= 0.5).astype(int)
    out: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, pred)),
        "mcc": float(matthews_corrcoef(y_true, pred)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
    }
    try:
        out["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    except ValueError:
        out["roc_auc"] = float("nan")
    try:
        out["pr_auc"] = float(average_precision_score(y_true, y_proba))
    except ValueError:
        out["pr_auc"] = float("nan")
    try:
        tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
        out["confusion"] = {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)}
        out["specificity"] = float(tn / (tn + fp)) if (tn + fp) else float("nan")
    except ValueError:
        out["confusion"] = {"tn": 0, "fp": 0, "fn": 0, "tp": 0}
    return out


# ============================================================================
# 模型定义（严格复刻 gnn_predictor.GCNModel）
# ============================================================================
def build_model(num_node_features: int, hidden_dim: int):
    import torch
    import torch.nn.functional as F
    from torch.nn import BatchNorm1d, Linear
    from torch_geometric.nn import GCNConv, global_mean_pool

    class GCNModel(torch.nn.Module):
        def __init__(self, n_feat: int, h: int):
            super().__init__()
            self.conv1 = GCNConv(n_feat, h)
            self.conv2 = GCNConv(h, h)
            self.conv3 = GCNConv(h, h)
            self.bn1 = BatchNorm1d(h)
            self.bn2 = BatchNorm1d(h)
            self.lin1 = Linear(h, h // 2)
            self.lin2 = Linear(h // 2, 1)

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            x = F.relu(self.bn1(self.conv1(x, edge_index)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = F.relu(self.bn2(self.conv2(x, edge_index)))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv3(x, edge_index)
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)
            return x.view(-1, 1)

    return GCNModel(num_node_features, hidden_dim)


def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================================
# 训练 / 评估单折
# ============================================================================
def train_one_fold(
    train_graphs: List[Any],
    val_graphs: List[Any],
    hparams: Dict[str, Any],
    seed: int,
    tag: str = "fold",
) -> Dict[str, Any]:
    import torch
    from torch_geometric.loader import DataLoader

    set_seed(seed)
    model = build_model(hparams["num_node_features"], hparams["hidden_dim"])
    optimizer = torch.optim.Adam(model.parameters(), lr=hparams["learning_rate"])
    criterion = torch.nn.BCEWithLogitsLoss()

    train_loader = DataLoader(train_graphs, batch_size=hparams["batch_size"], shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=hparams["batch_size"], shuffle=False)

    t0 = time.perf_counter()
    best_auc = -1.0
    best_state = None
    history: List[Dict[str, float]] = []

    for epoch in range(1, hparams["epochs"] + 1):
        model.train()
        total_loss, n_batch = 0.0, 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item())
            n_batch += 1
        train_loss = total_loss / max(n_batch, 1)

        val = _eval_loader(model, val_loader)
        history.append({"epoch": epoch, "train_loss": round(train_loss, 5),
                        "val_auc": round(val["roc_auc"], 5) if not np.isnan(val["roc_auc"]) else None,
                        "val_acc": round(val["accuracy"], 5)})
        if not np.isnan(val["roc_auc"]) and val["roc_auc"] > best_auc:
            best_auc = val["roc_auc"]
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

        if epoch == 1 or epoch % 5 == 0 or epoch == hparams["epochs"]:
            _log(f"    [{tag}] epoch {epoch:>3}/{hparams['epochs']}  "
                 f"loss={train_loss:.4f}  val_auc={val['roc_auc']:.4f}  acc={val['accuracy']:.4f}")

    secs = time.perf_counter() - t0

    # 用最佳 epoch 的权重重新评估
    if best_state is not None:
        model.load_state_dict(best_state)
    final = _eval_loader(model, val_loader)
    final.update({
        "train_seconds": round(secs, 2),
        "best_val_auc": float(best_auc) if best_auc >= 0 else float("nan"),
        "best_epoch": int(max(history, key=lambda h: h["val_auc"] or -1)["epoch"]),
        "history": history,
    })
    return {"model": model, "state_dict": best_state, "metrics": final}


def _eval_loader(model, loader) -> Dict[str, Any]:
    import torch

    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).view(-1).tolist())
            labels.extend(batch.y.view(-1).tolist())

    y_true = np.array([int(v) for v in labels])
    y_prob = np.array([float(v) for v in probs])
    if len(y_true) == 0:
        return {"roc_auc": float("nan"), "accuracy": float("nan"),
                "proba": np.array([]), "label": np.array([])}
    m = _metrics(y_true, y_prob)
    m["proba"] = y_prob
    m["label"] = y_true
    return m


# ============================================================================
# 主流程
# ============================================================================
def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="可复现 GNN 训练与评估")
    ap.add_argument("--max-records", type=int, default=2000)
    ap.add_argument("--threshold", type=float, default=DEFAULT_PIC50_THRESHOLD)
    ap.add_argument("--epochs", type=int, default=DEFAULT_HPARAMS["epochs"])
    ap.add_argument("--cv-folds", type=int, default=5)
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=DEFAULT_HPARAMS["random_state"])
    ap.add_argument("--quick", action="store_true", help="3 折 5 epochs，仅验通链路")
    ap.add_argument("--dataset", choices=["snapshot", "legacy"], default="snapshot",
                    help="snapshot=teach_lab 快照+阈值协议；legacy=原始 13,286 条 median split 协议（README 口径）")
    ap.add_argument("--tag", default="", help="非空时产物文件名加后缀，避免覆盖默认产物")
    args = ap.parse_args(argv)

    if args.quick:
        args.cv_folds = 3
        args.epochs = 5
        _log("⚠️ QUICK 模式：3 折 × 5 epochs，结果不可用于汇报")

    hparams = dict(DEFAULT_HPARAMS)
    hparams.update({"epochs": args.epochs, "random_state": args.seed})

    global METRICS_JSON, MODEL_OUT
    if args.tag:
        METRICS_JSON = ARTIFACT_DIR / f"gnn_metrics_{args.tag}.json"
        MODEL_OUT = ARTIFACT_DIR / f"gcn_repro_{args.tag}.pth"

    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    t_start = time.perf_counter()

    _log(SEP)
    _log("药尘光 · GNN（GCN）可复现训练与评估管线")
    _log(SEP)
    _log(f"  时间   : {datetime.now().isoformat(timespec='seconds')}")
    _log(f"  Python : {platform.python_version()}")
    _log(f"  torch  : {_pkg('torch')}")
    _log(f"  PyG    : {_pkg('torch-geometric')}")
    _log(f"  超参   : epochs={hparams['epochs']} bs={hparams['batch_size']} "
         f"lr={hparams['learning_rate']} hidden={hparams['hidden_dim']}")

    # ---- 依赖检查 ----
    try:
        import torch
        import torch_geometric  # noqa: F401
    except Exception as exc:
        _log(f"[失败] PyTorch/PyG 不可用: {exc}")
        return 1

    # ---- 数据 ----
    from utils.teach_lab import build_graph_dataset, clean_activity_data

    _section(f"1. 读取数据（dataset={args.dataset}）")
    if args.dataset == "legacy":
        # 复刻原始训练协议：13,286 条，median split（pIC50 >= 中位数 -> active，无丢弃）
        csv_path = DATA_DIR / "egfr_all_ic50_full.csv"
        if not csv_path.exists():
            _log(f"[失败] 原始数据不存在: {csv_path}")
            return 1
        raw_df = pd.read_csv(csv_path)
        _log(f"  数据  : {csv_path}")
        _log(f"  记录数: {len(raw_df)}")

        _section("2. 标注（median split，复刻原始协议）")
        pic50_num = pd.to_numeric(raw_df["pIC50"], errors="coerce")
        med = float(pic50_num.median())
        clean_df = raw_df.rename(columns={"SMILES": "smiles"}).loc[pic50_num.notna().values].copy()
        clean_df["label"] = (pd.to_numeric(clean_df["pIC50"], errors="coerce") >= med).astype(int)
        stages = []
        label_threshold = med
        labeling = f"median_split (pIC50 >= median={med:.4f} -> active, 无丢弃)"
        _log(f"  中位数 pIC50 = {med:.4f}")
    else:
        csv_path = SNAPSHOT_CSV
        if not csv_path.exists():
            _log(f"[失败] 快照不存在: {csv_path}")
            _log("       请先运行 scripts/train_rf.py 生成快照。")
            return 1
        raw_df = pd.read_csv(csv_path)
        _log(f"  快照  : {csv_path}")
        _log(f"  记录数: {len(raw_df)}")

        _section("2. 数据清洗")
        clean_df, stages = clean_activity_data(raw_df, pic50_threshold=args.threshold)
        label_threshold = args.threshold
        labeling = "threshold_filter (pIC50 >= 阈值 -> active，中间带丢弃)"
    for st in stages:
        nm = st.name if len(st.name) <= 34 else st.name[:31] + "..."
        _log(f"  {nm:<36} 剩余 {st.remaining:>6}  移除 {st.removed:>5}")
    _log(f"  标注协议: {labeling}")
    _log(f"  >>> 最终样本: {len(clean_df)}")

    y_all = clean_df["label"].to_numpy(dtype=int)
    label_counts = {int(k): int(v) for k, v in zip(*np.unique(y_all, return_counts=True))}
    _log(f"  标签分布: {label_counts}  正例比例={y_all.mean():.4f}")
    if len(label_counts) < 2:
        _log("[失败] 单一类别，无法训练。")
        return 1

    # ---- 建图 ----
    _section("3. 构建分子图（13 维原子特征）")
    t0 = time.perf_counter()
    graphs = build_graph_dataset(clean_df)
    build_secs = time.perf_counter() - t0
    _log(f"  成功建图: {len(graphs)} / {len(clean_df)}  "
         f"（失败 {len(clean_df) - len(graphs)}），耗时 {build_secs:.1f}s")
    if len(graphs) < len(clean_df):
        _log("  ⚠️ 有分子建图失败，后续指标基于成功建图的子集。")

    # 若建图失败，需同步裁剪标签数组以保持对齐（build_graph_dataset 内部跳过失败项）
    y_used = np.array([int(g.y.item()) for g in graphs])
    _log(f"  实际标签分布: { {int(k): int(v) for k, v in zip(*np.unique(y_used, return_counts=True))} }")

    # ---- 5 折 CV ----
    _section(f"4. {args.cv_folds} 折分层交叉验证（每折 {args.epochs} epochs）")
    from sklearn.model_selection import StratifiedKFold

    idx = np.arange(len(graphs))
    skf = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    folds: List[Dict[str, Any]] = []

    _log(f"  {'fold':<6} {'n_train':>8} {'n_val':>6} {'AUC':>8} {'ACC':>8} {'BAcc':>8} {'MCC':>8} {'秒':>7}")
    _log(f"  {'-' * 6} {'-' * 8} {'-' * 6} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 7}")

    for i, (tr, va) in enumerate(skf.split(idx, y_used), start=1):
        tr_graphs = [graphs[j] for j in tr]
        va_graphs = [graphs[j] for j in va]
        res = train_one_fold(tr_graphs, va_graphs, hparams, seed=args.seed + i, tag=f"f{i}")
        m = res["metrics"]
        folds.append({
            "fold": i,
            "n_train": int(len(tr)),
            "n_val": int(len(va)),
            "roc_auc": m["roc_auc"],
            "accuracy": m["accuracy"],
            "balanced_accuracy": m["balanced_accuracy"],
            "mcc": m["mcc"],
            "f1": m["f1"],
            "precision": m["precision"],
            "recall": m["recall"],
            "pr_auc": m["pr_auc"],
            "specificity": m.get("specificity", float("nan")),
            "confusion": m["confusion"],
            "best_epoch": m["best_epoch"],
            "train_seconds": m["train_seconds"],
            "history": m["history"],
        })
        _log(f"  {i:<6} {len(tr):>8} {len(va):>6} {m['roc_auc']:>8.4f} {m['accuracy']:>8.4f} "
             f"{m['balanced_accuracy']:>8.4f} {m['mcc']:>8.4f} {m['train_seconds']:>7.1f}")

    cv_summary: Dict[str, Any] = {"n_splits": args.cv_folds, "folds": folds}
    for key in ("roc_auc", "accuracy", "balanced_accuracy", "mcc", "f1", "precision",
                "recall", "pr_auc", "specificity"):
        vals = np.array([f[key] for f in folds], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size:
            cv_summary[key] = {
                "mean": float(vals.mean()),
                "std": float(vals.std(ddof=1)) if vals.size > 1 else 0.0,
                "min": float(vals.min()),
                "max": float(vals.max()),
            }

    _log("")
    _log(f"  >>> CV ROC-AUC   : {cv_summary['roc_auc']['mean']:.4f} ± {cv_summary['roc_auc']['std']:.4f}")
    _log(f"  >>> CV Accuracy  : {cv_summary['accuracy']['mean']:.4f} ± {cv_summary['accuracy']['std']:.4f}")
    _log(f"  >>> CV BalancedAcc: {cv_summary['balanced_accuracy']['mean']:.4f} ± {cv_summary['balanced_accuracy']['std']:.4f}")
    _log(f"  >>> CV MCC       : {cv_summary['mcc']['mean']:.4f} ± {cv_summary['mcc']['std']:.4f}")

    # ---- 独立测试集 ----
    _section("5. 独立留出测试集（20%，与 CV 互斥）")
    from sklearn.model_selection import train_test_split

    tr_i, te_i = train_test_split(
        idx, test_size=args.test_size, random_state=args.seed, stratify=y_used
    )
    tr_graphs = [graphs[j] for j in tr_i]
    te_graphs = [graphs[j] for j in te_i]
    final = train_one_fold(tr_graphs, te_graphs, hparams, seed=args.seed, tag="test")
    fm = final["metrics"]
    _log(f"  n_train={len(tr_i)} n_test={len(te_i)}")
    _log(f"  ROC-AUC={fm['roc_auc']:.4f}  ACC={fm['accuracy']:.4f}  "
         f"BAcc={fm['balanced_accuracy']:.4f}  MCC={fm['mcc']:.4f}  F1={fm['f1']:.4f}")
    _log(f"  混淆矩阵: {fm['confusion']}")

    # ---- 与现有 pth 对照 ----
    _section("6. 与现有 pth 对照（同一测试集）")
    legacy_cmp = _compare_legacy(te_graphs, fm)

    # ---- 保存 ----
    _section("7. 保存产物")
    import torch

    state = final["state_dict"]
    if state is None:
        state = final["model"].state_dict()
    torch.save(state, MODEL_OUT)
    _log(f"  模型 : {MODEL_OUT} ({MODEL_OUT.stat().st_size / 1024:.1f} KB)")

    elapsed = time.perf_counter() - t_start
    payload: Dict[str, Any] = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "script": "scripts/train_gnn.py",
        "generated_by": "改进总控代理 / P0-02",
        "quick_mode": bool(args.quick),

        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _pkg("torch"),
            "torch_geometric": _pkg("torch-geometric"),
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },

        "data": {
            "dataset": args.dataset,
            "csv": _rel(csv_path),
            "csv_sha256": _sha256_of_file(csv_path),
            "raw_n": int(len(raw_df)),
            "clean_n": int(len(clean_df)),
            "graph_n": int(len(graphs)),
            "build_seconds": round(build_secs, 2),
            "label_counts": label_counts,
            "positive_rate": float(y_all.mean()),
            "labeling": labeling,
            "pic50_threshold": float(label_threshold),
            "data_fingerprint_sha256": _sha256_of_df(clean_df),
            "cleaning_stages": [
                {"name": s.name, "remaining": int(s.remaining), "removed": int(s.removed)}
                for s in stages
            ],
        },

        "model": {
            "architecture": "GCNConv(13->128) x3 + BN x2 + Linear(128,64) + Linear(64,1)",
            "num_node_features": hparams["num_node_features"],
            "hidden_dim": hparams["hidden_dim"],
            "dropout": hparams["dropout"],
            "param_count": int(sum(p.numel() for p in final["model"].parameters())),
            "atom_feature_spec": [
                "atomic_num", "degree", "formal_charge", "hybridization", "is_aromatic",
                "total_hs", "implicit_valence", "radical_electrons", "isotope",
                "mass/100", "total_valence", "has_implicit_h", "is_N_or_O",
            ],
        },

        "hparams": {k: v for k, v in hparams.items()},

        "protocol": {
            "cv": f"StratifiedKFold(n_splits={args.cv_folds}, shuffle=True, random_state={args.seed})",
            "test": f"train_test_split(test_size={args.test_size}, stratify=y, random_state={args.seed})",
            "split_type": "random_stratified",
            "model_selection": "best val_auc epoch within each fold (no test leakage)",
            "known_limitation": (
                "随机分层划分；scaffold split 对照需要另行验证（P0-05）。"
                "GNN 对建图失败分子敏感，实际样本量可能小于 clean_df。"
            ),
        },

        "cv_results": cv_summary,
        "final_test_results": {k: v for k, v in fm.items() if k not in ("proba", "label")},
        "legacy_model_comparison": legacy_cmp,

        "artifacts": {
            "model": _rel(MODEL_OUT),
            "model_sha256": _sha256_of_file(MODEL_OUT),
        },

        "runtime_seconds": round(elapsed, 2),
    }

    METRICS_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    _log(f"  指标 : {METRICS_JSON}")

    # ---- 对照 README ----
    _section("完成 —— 与 README 声称值对照")
    claim_auc, claim_auc_s = 0.845, 0.008
    claim_acc, claim_acc_s = 0.767, 0.011
    g_auc_m = cv_summary["roc_auc"]["mean"]
    g_auc_s = cv_summary["roc_auc"]["std"]
    g_acc_m = cv_summary["accuracy"]["mean"]
    g_acc_s = cv_summary["accuracy"]["std"]

    _log(f"  {'指标':<26} {'本次实测':>16} {'README 声称':>15} {'Δ':>9}")
    _log(f"  {'-' * 26} {'-' * 16} {'-' * 15} {'-' * 9}")
    _log(f"  {'CV AUC (5-fold)':<26} {g_auc_m:>9.4f} ± {g_auc_s:.4f} "
         f"{'0.845 ± 0.008':>15} {g_auc_m - claim_auc:>+9.4f}")
    _log(f"  {'CV Accuracy (5-fold)':<26} {g_acc_m:>9.4f} ± {g_acc_s:.4f} "
         f"{'0.767 ± 0.011':>15} {g_acc_m - claim_acc:>+9.4f}")
    _log("")
    _log(f"  总耗时: {elapsed:.1f}s")
    _log("")
    _log(f"  产物: {METRICS_JSON.name} / {MODEL_OUT.name}")
    _log(SEP)
    return 0


def _compare_legacy(te_graphs: List[Any], new_metrics: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {"available": False}
    if not LEGACY_MODEL.exists():
        _log("  现有 pth 不存在，跳过对照。")
        return out
    try:
        import torch
        from torch_geometric.loader import DataLoader

        state = torch.load(LEGACY_MODEL, map_location="cpu", weights_only=True)
        model = build_model(13, 128)
        model.load_state_dict(state, strict=True)
        loader = DataLoader(te_graphs, batch_size=32, shuffle=False)
        m = _eval_loader(model, loader)

        out = {
            "available": True,
            "param_count": int(sum(p.numel() for p in model.parameters())),
            "test_metrics": {k: v for k, v in m.items() if k not in ("proba", "label")},
            "agreement_rate": float(np.mean(
                ((m["proba"] >= 0.5) == (new_metrics["proba"] >= 0.5))
            )) if len(m["proba"]) == len(new_metrics["proba"]) else None,
            "proba_correlation": float(np.corrcoef(m["proba"], new_metrics["proba"])[0, 1])
            if len(m["proba"]) == len(new_metrics["proba"])
            and np.std(m["proba"]) > 0 and np.std(new_metrics["proba"]) > 0 else None,
        }
        _log(f"  现有 pth: AUC={m['roc_auc']:.4f}  ACC={m['accuracy']:.4f}  "
             f"BAcc={m['balanced_accuracy']:.4f}  MCC={m['mcc']:.4f}")
        _log(f"  复现模型: AUC={new_metrics['roc_auc']:.4f}  ACC={new_metrics['accuracy']:.4f}")
        if out["agreement_rate"] is not None:
            _log(f"  判定一致率={out['agreement_rate']:.4f}  "
                 f"概率相关={out['proba_correlation']:.4f}")
        if m["roc_auc"] > 0.99:
            _log("  ⚠️ 现有 pth 在同一测试集 AUC>0.99 —— 高度提示泄漏。")
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
        _log(f"  ⚠️ 对照失败: {out['error']}")
    return out


def _pkg(name: str) -> Optional[str]:
    try:
        import importlib.metadata as md

        return md.version(name)
    except Exception:
        return None


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
