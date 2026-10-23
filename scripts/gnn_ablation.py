#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/gnn_ablation.py — P1-07 GNN 超参消融矩阵（legacy 协议）
================================================================
**诚实口径**：单次 80/20 划分（rs=42）+ 固定 8 epochs 取末轮指标。
目的仅是**配置相对排序**（哪个方向值得投入），不产出可对外声称的最终指标；
最终指标以 EXP-019（5 折 × 25 epochs）为准。此口径限制如实写入产物与文档。

8 个配置：层数（2/3/4）× hidden（64/128/256）× dropout（0.2/0.5/0.7）× lr（1e-3/3e-3）。
产物：artifacts/gnn_ablation_metrics.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import warnings
warnings.filterwarnings("ignore")
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_DUPLICATE_LIB_OK", "TRUE")
if sys.platform == "win32":
    for _s in (sys.stdout, sys.stderr):
        try:
            _s.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

from sklearn.model_selection import train_test_split  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_CSV = REPO_ROOT / "data" / "raw" / "egfr_all_ic50_full.csv"
OUT_JSON = ARTIFACT_DIR / "gnn_ablation_metrics.json"
SEED = 42
EPOCHS = 8
BATCH = 32

CONFIGS = [
    {"id": "base_L3_h128_d0.5_lr1e-3", "layers": 3, "hidden": 128, "dropout": 0.5, "lr": 1e-3},
    {"id": "L2_h128_d0.5", "layers": 2, "hidden": 128, "dropout": 0.5, "lr": 1e-3},
    {"id": "L4_h128_d0.5", "layers": 4, "hidden": 128, "dropout": 0.5, "lr": 1e-3},
    {"id": "L3_h64_d0.5", "layers": 3, "hidden": 64, "dropout": 0.5, "lr": 1e-3},
    {"id": "L3_h256_d0.5", "layers": 3, "hidden": 256, "dropout": 0.5, "lr": 1e-3},
    {"id": "L3_h128_d0.2", "layers": 3, "hidden": 128, "dropout": 0.2, "lr": 1e-3},
    {"id": "L3_h128_d0.7", "layers": 3, "hidden": 128, "dropout": 0.7, "lr": 1e-3},
    {"id": "L3_h128_d0.5_lr3e-3", "layers": 3, "hidden": 128, "dropout": 0.5, "lr": 3e-3},
]


def build_model(layers: int, hidden: int, dropout: float):
    import torch
    import torch.nn.functional as F
    from torch.nn import BatchNorm1d, Linear, ModuleList
    from torch_geometric.nn import GCNConv, global_mean_pool

    class FlexGCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.convs = ModuleList([GCNConv(13, hidden)] +
                                    [GCNConv(hidden, hidden) for _ in range(layers - 1)])
            self.bns = ModuleList([BatchNorm1d(hidden) for _ in range(layers - 1)])
            self.lin1 = Linear(hidden, hidden // 2)
            self.lin2 = Linear(hidden // 2, 1)
            self.dropout = dropout

        def forward(self, data):
            x, edge_index, batch = data.x, data.edge_index, data.batch
            for i, conv in enumerate(self.convs):
                x = conv(x, edge_index)
                if i < layers - 1:
                    x = F.relu(self.bns[i](x))
                    x = F.dropout(x, p=self.dropout, training=self.training)
            x = global_mean_pool(x, batch)
            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            return self.lin2(x).view(-1, 1)

    return FlexGCN()


def train_eval(config: Dict[str, Any], train_graphs, test_graphs) -> Dict[str, Any]:
    import torch
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 roc_auc_score)
    from torch_geometric.loader import DataLoader

    torch.manual_seed(SEED)
    np.random.seed(SEED)
    model = build_model(config["layers"], config["hidden"], config["dropout"])
    opt = torch.optim.Adam(model.parameters(), lr=config["lr"])
    crit = torch.nn.BCEWithLogitsLoss()
    tr_loader = DataLoader(train_graphs, batch_size=BATCH, shuffle=True)
    te_loader = DataLoader(test_graphs, batch_size=BATCH, shuffle=False)

    curve = []
    t0 = time.perf_counter()
    for epoch in range(1, EPOCHS + 1):
        model.train()
        for batch in tr_loader:
            opt.zero_grad()
            loss = crit(model(batch), batch.y.view(-1, 1))
            loss.backward()
            opt.step()
        model.eval()
        probs, labels = [], []
        with torch.no_grad():
            for batch in te_loader:
                probs.extend(torch.sigmoid(model(batch)).view(-1).tolist())
                labels.extend(batch.y.view(-1).tolist())
        y = np.array(labels, dtype=int)
        p = np.array(probs)
        auc = float(roc_auc_score(y, p))
        curve.append({"epoch": epoch, "test_auc": round(auc, 4)})
        print(f"    [{config['id']}] ep{epoch}: AUC={auc:.4f}", flush=True)

    pred = (p >= 0.5).astype(int)
    return {
        "config": config,
        "final_epoch_auc": float(roc_auc_score(y, p)),
        "final_epoch_bacc": float(balanced_accuracy_score(y, pred)),
        "final_epoch_mcc": float(matthews_corrcoef(y, pred)),
        "param_count": int(sum(q.numel() for q in model.parameters())),
        "curve": curve,
        "seconds": round(time.perf_counter() - t0, 1),
    }


def main() -> int:
    import torch  # noqa: F401
    from utils.teach_lab import build_graph_dataset

    df = pd.read_csv(DATA_CSV)
    pic50 = pd.to_numeric(df["pIC50"], errors="coerce")
    med = float(pic50.median())
    clean = df.rename(columns={"SMILES": "smiles"}).loc[pic50.notna().values].copy()
    clean["label"] = (pd.to_numeric(clean["pIC50"], errors="coerce") >= med).astype(int)
    print(f"建图（13,286）…", flush=True)
    graphs = build_graph_dataset(clean)
    y_all = np.array([int(g.y.item()) for g in graphs])
    tr_i, te_i = train_test_split(np.arange(len(graphs)), test_size=0.2,
                                  random_state=SEED, stratify=y_all)
    tr_graphs = [graphs[j] for j in tr_i]
    te_graphs = [graphs[j] for j in te_i]
    print(f"单次划分: train={len(tr_i)} test={len(te_i)}，{len(CONFIGS)} 配置 × {EPOCHS} epochs", flush=True)

    results = []
    for cfg in CONFIGS:
        print(f"  配置 {cfg['id']}", flush=True)
        results.append(train_eval(cfg, tr_graphs, te_graphs))

    results.sort(key=lambda r: -r["final_epoch_auc"])
    print("\n== 排序（末轮 test AUC）==")
    for r in results:
        print(f"  {r['config']['id']:<28} AUC={r['final_epoch_auc']:.4f} "
              f"BAcc={r['final_epoch_bacc']:.4f} MCC={r['final_epoch_mcc']:.4f} "
              f"params={r['param_count']} {r['seconds']}s")

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-07",
        "honest_caveat": "单次划分 + 固定 8 epochs 末轮指标，仅用于配置相对排序；不可对外声称，最终口径见 EXP-019",
        "protocol": {"split": "train_test_split(0.2, rs=42, stratify)", "epochs": EPOCHS,
                     "batch": BATCH, "seed": SEED},
        "results": results,
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
