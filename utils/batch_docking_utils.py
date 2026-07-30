# utils/batch_docking_utils.py
"""
批量分子对接工具模块 (Smina)

基于 docking_utils.py 中的单配体对接函数，
提供批量 PDBQT 生成、并行对接、结果汇总等高层封装。

使用方式:
    from utils.batch_docking_utils import run_batch_docking
    result = run_batch_docking(pdb_id="2ITO", smiles_list=[...])
"""

import os
import tempfile
import logging
import shutil
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
import pandas as pd

# 复用单配体模块中的核心函数
from docking_utils import (
    _download_pdb,
    _parse_pdb_for_docking,
    pdb_to_pdbqt,
    smiles_to_pdbqt,
    run_smina,
    parse_smina_output,
    _ensure_openbabel,
)

# ---------- 批量配体准备 ----------

def batch_prepare_ligands(
    smiles_list: List[str],
    output_dir: str,
    pH: float = 7.4,
) -> pd.DataFrame:
    """
    将 SMILES 列表批量转换为 PDBQT 文件。

    参数
    ----------
    smiles_list : list[str]
        配体 SMILES 字符串列表
    output_dir : str
        输出目录路径
    pH : float
        质子化状态 pH，默认 7.4

    返回
    -------
    pd.DataFrame
        列: name, smiles, pdbqt_path
    """
    if not _ensure_openbabel():
        raise ImportError("openbabel 未安装，请执行: conda install -c conda-forge openbabel")

    os.makedirs(output_dir, exist_ok=True)
    records = []

    for i, smi in enumerate(smiles_list):
        smi = smi.strip()
        if not smi:
            continue
        name = f"ligand_{i + 1:03d}"
        pdbqt_path = os.path.join(output_dir, f"{name}.pdbqt")

        try:
            smiles_to_pdbqt(smi, pdbqt_path, pH=pH)
            records.append({
                "name": name,
                "smiles": smi,
                "pdbqt_path": pdbqt_path,
            })
            logging.info(f"  配体准备成功: {name}")
        except Exception as e:
            logging.warning(f"  配体 {name} 准备失败 ({smi[:30]}...): {e}")

    if not records:
        raise RuntimeError("所有配体 PDBQT 准备均失败，请检查 SMILES 格式或 openbabel 安装")

    return pd.DataFrame(records)


# ---------- 单配体对接 (并行工作单元) ----------

def dock_single_ligand(
    ligand_row: Dict,
    receptor_pdbqt: str,
    pocket_center: List[float],
    pocket_size: List[float],
    output_dir: str,
    num_poses: int = 9,
    exhaustiveness: int = 8,
) -> Optional[Dict]:
    """
    对单个配体执行 Smina 对接（用于 ThreadPoolExecutor 并行）。

    返回:
        dict or None -- 对接成功则返回结果字典，失败返回 None
    """
    name = ligand_row["name"]
    ligand_pdbqt = ligand_row["pdbqt_path"]
    out_sdf = os.path.join(output_dir, f"{name}_poses.sdf")

    try:
        output_text = run_smina(
            ligand_pdbqt,
            receptor_pdbqt,
            out_sdf,
            pocket_center,
            pocket_size,
            num_poses=num_poses,
            exhaustiveness=exhaustiveness,
        )
        results = parse_smina_output(output_text)

        if results:
            best_score = min(r["affinity"] for r in results)
            worst_score = max(r["affinity"] for r in results)
        else:
            best_score = worst_score = None

        return {
            "name": name,
            "smiles": ligand_row["smiles"],
            "best_score": best_score,
            "worst_score": worst_score,
            "pose_count": len(results),
            "poses": results,
            "output_sdf": out_sdf,
            "success": True,
            "error": None,
        }
    except Exception as e:
        return {
            "name": name,
            "smiles": ligand_row["smiles"],
            "best_score": None,
            "worst_score": None,
            "pose_count": 0,
            "poses": [],
            "output_sdf": None,
            "success": False,
            "error": str(e),
        }


# ---------- 批量对接 ----------

def batch_dock(
    ligands_df: pd.DataFrame,
    receptor_pdbqt: str,
    pocket_center: List[float],
    pocket_size: List[float],
    output_dir: str,
    num_poses: int = 9,
    exhaustiveness: int = 8,
    n_jobs: int = 1,
    progress_callback=None,
) -> pd.DataFrame:
    """
    批量执行分子对接（支持并行 Smina 调用）。

    参数
    ----------
    ligands_df : pd.DataFrame
        batch_prepare_ligands() 的输出
    receptor_pdbqt : str
        受体蛋白 PDBQT 文件路径
    pocket_center : list
        盒子中心 [x, y, z]
    pocket_size : list
        盒子尺寸 [sx, sy, sz]
    output_dir : str
        对接结果输出目录
    num_poses : int
        每个配体保留构象数，默认 9
    exhaustiveness : int
        搜索精度，默认 8
    n_jobs : int
        并行线程数（默认 1=串行）
    progress_callback : callable, optional
        每完成一个配体调用 callback(completed, total)

    返回
    -------
    pd.DataFrame
        按 best_score 升序排列的对接结果表
    """
    os.makedirs(output_dir, exist_ok=True)

    ligand_rows = ligands_df.to_dict("records")
    total = len(ligand_rows)
    results = []

    if n_jobs > 1:
        # 并行模式: ThreadPoolExecutor (Smina 是子进程 I/O，线程安全)
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            futures = {}
            for row in ligand_rows:
                future = executor.submit(
                    dock_single_ligand,
                    row, receptor_pdbqt,
                    pocket_center, pocket_size,
                    output_dir, num_poses, exhaustiveness,
                )
                futures[future] = row["name"]

            completed = 0
            for future in as_completed(futures):
                res = future.result()
                if res:
                    results.append(res)
                completed += 1
                if progress_callback:
                    progress_callback(completed, total)
    else:
        # 串行模式
        for idx, row in enumerate(ligand_rows):
            res = dock_single_ligand(
                row, receptor_pdbqt,
                pocket_center, pocket_size,
                output_dir, num_poses, exhaustiveness,
            )
            if res:
                results.append(res)
            completed = idx + 1
            if progress_callback:
                progress_callback(completed, total)

    if not results:
        raise RuntimeError("所有对接任务均失败")

    df = pd.DataFrame(results)
    # 按最佳打分升序（负值越小结合越强）
    df = df.sort_values("best_score", ascending=True, na_position="last").reset_index(drop=True)
    df.index = df.index + 1  # 排名从 1 开始
    df.index.name = "排名"
    return df


# ---------- 统计摘要 ----------

def generate_comparison_stats(results_df: pd.DataFrame) -> Dict:
    """
    生成批量对接结果统计摘要。

    返回字典:
        count, n_success, n_failed,
        best_score, worst_score, mean_score, std_score,
        top_5, bottom_5
    """
    success_df = results_df[results_df["success"] == True]
    failed_df = results_df[results_df["success"] != True]

    scores = success_df["best_score"].dropna().values
    stats = {
        "count": len(results_df),
        "n_success": len(success_df),
        "n_failed": len(failed_df),
    }

    if len(scores) > 0:
        stats.update({
            "best_score": float(scores.min()),
            "worst_score": float(scores.max()),
            "mean_score": float(scores.mean()),
            "std_score": float(scores.std()),
        })
        stats["top_5"] = (
            success_df.nsmallest(5, "best_score")[["name", "smiles", "best_score"]]
            .to_dict("records")
        )
        stats["bottom_5"] = (
            success_df.nlargest(5, "best_score")[["name", "smiles", "best_score"]]
            .to_dict("records")
        )
    else:
        stats.update({
            "best_score": None, "worst_score": None,
            "mean_score": None, "std_score": None,
            "top_5": [], "bottom_5": [],
        })

    return stats


# ---------- 批量对接主流程 ----------

def run_batch_docking(
    pdb_id: Optional[str] = None,
    pdb_content: Optional[bytes] = None,
    smiles_list: Optional[List[str]] = None,
    ligand_resname: Optional[str] = None,
    num_poses: int = 9,
    exhaustiveness: int = 8,
    n_jobs: int = 1,
    progress_callback=None,
) -> Dict:
    """
    批量分子对接主流程：下载/读取蛋白 → 口袋检测 → 批量 PDBQT 转换 → 批量对接

    参数
    ----------
    pdb_id : str, optional
        PDB ID（如 "2ITO"）
    pdb_content : bytes, optional
        上传的 PDB 文件内容
    smiles_list : list[str]
        配体 SMILES 列表
    ligand_resname : str, optional
        配体残基名，不提供则自动检测
    num_poses : int
        每个配体保留构象数，默认 9
    exhaustiveness : int
        搜索精度，默认 8
    n_jobs : int
        并行 Smina 进程数，默认 1（串行）
    progress_callback : callable, optional
        callback(stage, completed, total)

    返回
    -------
    dict
        {
            "pdb_id": str,
            "pocket": {"center": [...], "size": [...]},
            "ligand_count": int,
            "results_df": pd.DataFrame (按 best_score 升序),
            "stats": dict (统计摘要),
            "temp_dir": str (临时目录路径, 调用者负责清理),
        }
    """
    if not _ensure_openbabel():
        raise ImportError("缺少 openbabel，请执行: conda install -c conda-forge openbabel")

    if not smiles_list:
        raise ValueError("请提供至少一个配体 SMILES")

    # 工作目录
    work_dir = tempfile.mkdtemp(prefix="batch_docking_")
    tmp = Path(work_dir)

    try:
        # ---- 阶段 1: 获取蛋白 + 解析口袋 ----
        if progress_callback:
            progress_callback("protein", 0, 1)

        if pdb_id:
            raw_pdb_path = _download_pdb(pdb_id)
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=5.0
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
        elif pdb_content:
            raw_pdb_path = os.path.join(work_dir, "input_protein.pdb")
            with open(raw_pdb_path, "wb") as f:
                f.write(pdb_content)
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=5.0
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
        else:
            raise ValueError("请提供 PDB ID 或 PDB 文件内容")

        if progress_callback:
            progress_callback("protein", 1, 1)

        # ---- 阶段 2: 蛋白 → PDBQT ----
        if progress_callback:
            progress_callback("receptor_pdbqt", 0, 1)

        receptor_pdbqt = str(tmp / "receptor.pdbqt")
        pdb_to_pdbqt(str(protein_pdb), receptor_pdbqt)

        if progress_callback:
            progress_callback("receptor_pdbqt", 1, 1)

        # ---- 阶段 3: 批量配体 SMILES → PDBQT ----
        if progress_callback:
            progress_callback("ligands", 0, len(smiles_list))

        ligand_dir = str(tmp / "ligands")
        ligands_df = batch_prepare_ligands(smiles_list, ligand_dir)

        if progress_callback:
            progress_callback("ligands", len(ligands_df), len(smiles_list))

        # ---- 阶段 4: 批量对接 ----
        docking_out_dir = str(tmp / "docking_results")

        def _dock_progress(completed, total):
            if progress_callback:
                progress_callback("docking", completed, total)

        results_df = batch_dock(
            ligands_df,
            receptor_pdbqt,
            pocket["center"],
            pocket["size"],
            docking_out_dir,
            num_poses=num_poses,
            exhaustiveness=exhaustiveness,
            n_jobs=n_jobs,
            progress_callback=_dock_progress,
        )

        # ---- 阶段 5: 统计 ----
        stats = generate_comparison_stats(results_df)

        # ---- 打包 SDF 数据用于下载 ----
        for _, row in results_df.iterrows():
            if row["success"] and row.get("output_sdf") and os.path.exists(row["output_sdf"]):
                with open(row["output_sdf"], "rb") as f:
                    results_df.at[row.name, "_sdf_bytes"] = f.read()

        return {
            "pdb_id": pdb_id or "上传的PDB文件",
            "ligand_resname": ligand_resname,
            "pocket": pocket,
            "ligand_count": len(ligands_df),
            "results_df": results_df,
            "stats": stats,
            "temp_dir": work_dir,
        }

    except Exception:
        # 出错时清理临时目录
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        raise


def cleanup_batch_docking(result: Dict):
    """清理批量对接产生的临时文件"""
    temp_dir = result.get("temp_dir")
    if temp_dir and os.path.isdir(temp_dir):
        shutil.rmtree(temp_dir, ignore_errors=True)
