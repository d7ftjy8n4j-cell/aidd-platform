#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
scripts/external_validation.py — P1-06 外部（时序）验证（legacy 协议）
======================================================================
流程：
1. 用本地 2026-10 快照（1,998 条）与 legacy 13,286 条求规范化 SMILES 补集；
2. 若补集不足（实测为 0 —— 快照被 legacy 全覆盖，此发现本身入档），直接 REST 分页
   拉取 ChEMBL 全宇宙（CHEMBL203 · IC50 · relation='=' · assay_type=B · Homo sapiens，
   共 17,686 条），落盘 data/raw/chembl_egfr_full_2026.csv，再求补集；
3. 外部集标签沿用固定决策规则 pIC50 >= 7.0（与训练协议一致，不重取中位数）；
4. 在 legacy 全量上重训 RF（legacy 超参），在外部集上评估。

⚠️ 诚实口径：这是「时间外推」（同靶点同活性类型、不同爬取时间），非跨靶点泛化。
产物：artifacts/external_validation_metrics.json + data/raw/chembl_egfr_full_2026.csv
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
for p in (str(REPO_ROOT), str(SCRIPTS_DIR)):
    if p not in sys.path:
        sys.path.insert(0, p)

from train_rf_legacy_protocol import compute_features  # noqa: E402

ARTIFACT_DIR = REPO_ROOT / "artifacts"
DATA_DIR = REPO_ROOT / "data" / "raw"
LEGACY_CSV = DATA_DIR / "egfr_all_ic50_full.csv"
SNAPSHOT_CSV = DATA_DIR / "chembl_egfr_activities.csv"
FULL_CSV = DATA_DIR / "chembl_egfr_full_2026.csv"
FULL_META = DATA_DIR / "chembl_egfr_full_2026.meta.json"
OUT_JSON = ARTIFACT_DIR / "external_validation_metrics.json"
SEED = 42
THRESHOLD = 7.0
HP = {"n_estimators": 200, "max_depth": 15, "max_features": "sqrt",
      "min_samples_leaf": 2, "criterion": "gini", "bootstrap": True}
BASE_URL = "https://www.ebi.ac.uk/chembl/api/data/activity.json"
MIN_EXTERNAL = 300


def _sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canon(s: str):
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog("rdApp.*")
    try:
        m = Chem.MolFromSmiles(s)
        return Chem.MolToSmiles(m) if m else None
    except Exception:
        return None


def fetch_full_universe() -> pd.DataFrame:
    """REST 分页拉全 CHEMBL203 IC50 relation='=' assay_type=B Homo sapiens 记录。"""
    page, offset, rows, total = 0, 0, [], None
    while True:
        params = {
            "target_chembl_id": "CHEMBL203", "type": "IC50", "relation": "=",
            "assay_type": "B", "target_organism": "Homo sapiens",
            "standard_units": "nM", "limit": 1000, "offset": offset,
            "only": "activity_id,molecule_chembl_id,standard_value,standard_units,pChembL_value",
        }
        r = requests.get(BASE_URL, params=params, timeout=60)
        r.raise_for_status()
        j = r.json()
        page_recs = j.get("activities", [])
        if total is None:
            total = int(j.get("page_meta", {}).get("total_count", 0))
            print(f"  全宇宙 total_count = {total}", flush=True)
        rows.extend(page_recs)
        print(f"  activity 分页 {page + 1}: 累计 {len(rows)}/{total}", flush=True)
        offset += 1000
        page += 1
        if not page_recs or offset >= total:
            break
        time.sleep(0.5)

    df = pd.DataFrame(rows)
    df = df[pd.to_numeric(df["standard_value"], errors="coerce").notna()]
    df = df.sort_values("standard_value").drop_duplicates(subset=["molecule_chembl_id"], keep="first")
    df["pIC50"] = 9.0 - np.log10(pd.to_numeric(df["standard_value"], errors="coerce"))
    # 活动数据先落盘缓存，结构请求失败不必重拉 18 页
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    df[["molecule_chembl_id", "standard_value", "pIC50"]].to_csv(
        DATA_DIR / "_chembl_universe_activities_cache.csv", index=False)

    # 批量取 SMILES（200/批：URL 过长会 400；带重试）
    mids = df["molecule_chembl_id"].tolist()
    smiles_map = {}
    batch = 200
    n_batches = math.ceil(len(mids) / batch)
    for i in range(0, len(mids), batch):
        chunk = mids[i:i + batch]
        got = None
        for attempt in range(3):
            try:
                r = requests.get(
                    "https://www.ebi.ac.uk/chembl/api/data/molecule.json",
                    params={"molecule_chembl_id__in": ",".join(chunk), "limit": batch,
                            "only": "molecule_chembl_id,molecule_structures"},
                    timeout=120)
                r.raise_for_status()
                got = r.json()
                break
            except Exception as exc:
                print(f"    结构批次 {i // batch + 1} 第 {attempt + 1} 次失败: {exc}", flush=True)
                time.sleep(3 * (attempt + 1))
        if got:
            for mol in got.get("molecules", []):
                s = (mol.get("molecule_structures") or {}).get("canonical_smiles")
                if s:
                    smiles_map[mol["molecule_chembl_id"]] = s
        if (i // batch) % 10 == 0 or i + batch >= len(mids):
            print(f"  structure 批次 {i // batch + 1}/{n_batches}（累计 SMILES {len(smiles_map)}）", flush=True)
        time.sleep(0.4)
    df["SMILES"] = df["molecule_chembl_id"].map(smiles_map)
    df = df.dropna(subset=["SMILES"])
    out = df[["SMILES", "pIC50"]].reset_index(drop=True)
    return out


def main() -> int:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import (balanced_accuracy_score, matthews_corrcoef,
                                 roc_auc_score)

    legacy = pd.read_csv(LEGACY_CSV)
    snap = pd.read_csv(SNAPSHOT_CSV)
    if "SMILES" not in snap.columns:
        snap = snap.rename(columns={"smiles": "SMILES"})

    legacy_canon = {c for c in (canon(s) for s in legacy["SMILES"].astype(str)) if c}

    def complement(df: pd.DataFrame) -> pd.DataFrame:
        d = df.copy()
        d["canon"] = [canon(s) for s in d["SMILES"].astype(str)]
        d = d.dropna(subset=["canon", "pIC50"]).drop_duplicates(subset=["canon"])
        return d[~d["canon"].isin(legacy_canon)]

    snap_overlap = len(snap) - len(complement(snap))
    ext = complement(snap)
    source = "snapshot"
    print(f"快照 {len(snap)} 条，与 legacy 重合 {snap_overlap} 条，补集 {len(ext)} 条", flush=True)

    if len(ext) < MIN_EXTERNAL:
        print(f"补集不足 {MIN_EXTERNAL}（快照被 legacy 全覆盖——发现入档），拉取全宇宙…", flush=True)
        if not FULL_CSV.exists():
            t0 = time.perf_counter()
            full = fetch_full_universe()
            FULL_CSV.write_text(full.to_csv(index=False), encoding="utf-8")
            FULL_META.write_text(json.dumps({
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "query": "CHEMBL203 · IC50 · relation='=' · assay_type=B · Homo sapiens · nM",
                "n_rows": int(len(full)),
                "sha256_csv": _sha256_file(FULL_CSV),
                "fetch_seconds": round(time.perf_counter() - t0, 1),
                "purpose": "P1-06 时序外推集（与 legacy 求补）",
            }, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"  全宇宙落盘 {len(full)} 条，耗时 {time.perf_counter() - t0:.0f}s", flush=True)
        full = pd.read_csv(FULL_CSV)
        ext = complement(full)
        source = "full_universe_2026"
        print(f"全宇宙 {len(full)} 条，与 legacy 重合 {len(full) - len(ext)} 条，外部集 {len(ext)} 条", flush=True)

    if len(ext) == 0 or source == "snapshot":
        print("[失败] 无外部数据可用（网络失败或补集仍为 0），见产物中的 findings")
        payload = {"findings": {"snapshot_overlap": snap_overlap,
                                "note": "2026-10 teach_lab top-2000 快照被 legacy 13,286 全覆盖"}}
        ARTIFACT_DIR.mkdir(exist_ok=True)
        OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return 1

    y_ext = (pd.to_numeric(ext["pIC50"], errors="coerce").fillna(-1).values >= THRESHOLD).astype(int)
    names = json.loads((REPO_ROOT / "feature_names.json").read_text(encoding="utf-8"))
    X_ext, diag = compute_features(ext["SMILES"].astype(str).tolist(), names)

    y_all = (pd.to_numeric(legacy["pIC50"], errors="coerce").fillna(-1).values >= THRESHOLD).astype(int)
    X_all, _ = compute_features(legacy["SMILES"].astype(str).tolist(), names)
    clf = RandomForestClassifier(random_state=SEED, n_jobs=-1, **HP)
    clf.fit(X_all, y_all)
    p_ext = clf.predict_proba(X_ext)[:, 1]
    pred_ext = (p_ext >= 0.5).astype(int)

    result = {
        "n_external": int(len(ext)),
        "positive_rate_external": float(y_ext.mean()),
        "positive_rate_legacy": float(y_all.mean()),
        "auc": float(roc_auc_score(y_ext, p_ext)) if len(np.unique(y_ext)) > 1 else None,
        "balanced_accuracy": float(balanced_accuracy_score(y_ext, pred_ext)),
        "mcc": float(matthews_corrcoef(y_ext, pred_ext)),
        "accuracy": float((pred_ext == y_ext).mean()),
        "brier": float(np.mean((p_ext - y_ext) ** 2)),
        "pic50_external_describe": {k: round(float(v), 3) for k, v in
                                    pd.to_numeric(ext["pIC50"], errors="coerce").describe().items()},
    }
    print(json.dumps({k: v for k, v in result.items() if "describe" not in k},
                     ensure_ascii=False, indent=2))

    payload = {
        "schema_version": "1.0",
        "generated_by": "改进总控代理 / P1-06",
        "caveat": "时间外推（同靶点 IC50，不同爬取时间），非跨靶点泛化；标签用固定阈值 7.0",
        "findings": {"snapshot_overlap": snap_overlap,
                     "note": "2026-10 teach_lab top-2000 快照被 legacy 13,286 全覆盖——legacy 数据对当前 ChEMBL top 记录的覆盖率 100%"},
        "data": {
            "legacy_csv_sha256": _sha256_file(LEGACY_CSV),
            "external_source": source,
            "full_universe_csv_sha256": _sha256_file(FULL_CSV) if FULL_CSV.exists() else None,
            "legacy_n": int(len(legacy)),
        },
        "external_results": result,
        "feature_diagnostics": diag,
        "hparams": {**HP, "random_state": SEED},
    }
    ARTIFACT_DIR.mkdir(exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"产物: {OUT_JSON}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
