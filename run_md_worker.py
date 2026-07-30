#!/usr/bin/env python
"""
run_md_worker.py — 独立 MD 模拟工作进程
由 Streamlit 页面通过 subprocess 调用，避免 GIL 锁死。

用法:
    python run_md_worker.py '<json_params>'

JSON 参数:
{
    "pdb_id": "3POZ",
    "pdb_content": null,
    "ligand_resname": "03P",
    "ligand_smiles": "CC(C)(O)CC(=O)...",
    "total_steps": 5000,
    "write_interval": 100,
    "temperature": 300.0,
    "padding": 1.0,
    "ionic_strength": 0.15,
    "ph": 7.0,
    "output_dir": "/tmp/md_xxx"
}

进度通信: 将进度百分比写入 {output_dir}/progress.txt
结果通信: 将结果 JSON 写入 {output_dir}/result.json
"""

import sys
import json
import os
import math
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 将项目根目录加入 sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# numpy 用于 sanitize 中的 ndarray 检查
try:
    import numpy as np
except ImportError:
    np = None


def write_progress(output_dir: str, progress: float, status: str = ""):
    """写入进度文件（手动 flush 确保页面轮询立即可读）"""
    os.makedirs(output_dir, exist_ok=True)
    progress_file = os.path.join(output_dir, "progress.txt")
    with open(progress_file, "w") as f:
        f.write(f"{progress:.4f}\n{status}")
        f.flush()


def init_energy_log(output_dir: str) -> str:
    """初始化能量日志 CSV，写入表头并 flush，返回文件路径"""
    csv_path = os.path.join(output_dir, "energy_log.csv")
    with open(csv_path, "w") as f:
        f.write("step,potential_energy_kjmol,temperature_k\n")
        f.flush()
    return csv_path


def append_energy_row(csv_path: str, step: int, energy: float, temp: float):
    """追加一行能量数据并立即 flush（供 progress_callback 调用）"""
    if math.isnan(energy) or math.isinf(energy):
        energy = 0.0
    if math.isnan(temp) or math.isinf(temp):
        temp = 0.0
    with open(csv_path, "a") as f:
        f.write(f"{step},{energy:.2f},{temp:.2f}\n")
        f.flush()


def _sanitize_float(val):
    """将 NaN/Inf 转为 None，避免 json.dump 崩溃"""
    if isinstance(val, float):
        if math.isnan(val) or math.isinf(val):
            return None
    return val


def _sanitize_for_json(obj):
    """递归清理对象中的 NaN/Inf 浮点数"""
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    elif isinstance(obj, float):
        return _sanitize_float(obj)
    elif np is not None and isinstance(obj, np.ndarray):
        return [_sanitize_float(float(v)) for v in obj.tolist()]
    return obj


def write_result(output_dir: str, result: dict):
    """写入最终结果 JSON（NaN/Inf 安全 + status 字段）"""
    result_file = os.path.join(output_dir, "result.json")
    safe = {"status": result.get("status", "success")}
    for k, v in result.items():
        if k in ("topology_pdb", "trajectory_xtc", "mean_pdb_path",
                 "num_atoms", "log", "tmpdir", "analysis",
                 "energy_log_csv", "error_message"):
            safe[k] = v
    clean = _sanitize_for_json(safe)
    with open(result_file, "w") as f:
        json.dump(clean, f, indent=2, ensure_ascii=False)


def main():
    if len(sys.argv) < 2:
        logger.error("缺少 JSON 参数")
        sys.exit(1)

    params = json.loads(sys.argv[1])
    output_dir = params.get("output_dir", "/tmp/md_worker")
    os.makedirs(output_dir, exist_ok=True)

    logger.info(f"MD Worker 启动，输出目录: {output_dir}")

    try:
        from md_utils import run_md_simulation, analyze_trajectory

        # 处理上传的 PDB 文件
        pdb_content = params.get("pdb_content")
        pdb_content_path = params.get("pdb_content_path")
        if pdb_content_path and os.path.exists(pdb_content_path):
            with open(pdb_content_path, "rb") as f:
                pdb_content = f.read()

        write_progress(output_dir, 0.0, "准备体系...")

        # 初始化能量日志（写表头 + flush）
        energy_csv = init_energy_log(output_dir)

        def progress_cb(step, total, energy, temp):
            pct = step / total
            write_progress(output_dir, pct,
                           f"模拟中: {step}/{total} 步 | 势能: {energy:,.0f} kJ/mol | 温度: {temp:.0f} K")
            # 实时追加能量数据（flush 在 append_energy_row 内部）
            append_energy_row(energy_csv, step, energy, temp)

        result = run_md_simulation(
            pdb_id=params.get("pdb_id"),
            pdb_content=pdb_content,
            ligand_resname=params.get("ligand_resname", "03P"),
            ligand_smiles=params.get("ligand_smiles"),
            total_steps=params.get("total_steps", 5000),
            write_interval=params.get("write_interval", 100),
            temperature=params.get("temperature", 300.0),
            padding=params.get("padding", 1.0),
            ionic_strength=params.get("ionic_strength", 0.15),
            ph=params.get("ph", 7.0),
            progress_callback=progress_cb,
        )

        write_progress(output_dir, 0.95, "分析轨迹 + 导出代表性结构...")

        # 轨迹分析
        try:
            analysis = analyze_trajectory(
                result["topology_pdb"],
                result["trajectory_xtc"],
                ligand_resname=params.get("ligand_resname", "03P"),
            )
            result["analysis"] = analysis

            # 导出代表性帧 PDB
            import mdtraj as md
            traj = md.load(result["trajectory_xtc"], top=result["topology_pdb"])
            rep_frame = analysis.get("representative_frame", 0)
            rep_pdb_path = os.path.join(output_dir, "representative_frame.pdb")
            traj[rep_frame].save(rep_pdb_path)
            result["mean_pdb_path"] = rep_pdb_path
            logger.info(f"代表性帧 #{rep_frame} 已导出: {rep_pdb_path}")
        except Exception as e:
            logger.warning(f"轨迹分析失败: {e}")
            result["analysis"] = None

        result["energy_log_csv"] = energy_csv  # 已由 progress_cb 实时写入
        result["status"] = "success"
        write_result(output_dir, result)
        write_progress(output_dir, 1.0, "✅ 模拟完成")
        logger.info("MD Worker 完成")

    except Exception as e:
        logger.exception(f"MD Worker 失败: {e}")
        error_result = {
            "status": "error",
            "error_message": str(e),
        }
        write_result(output_dir, error_result)
        write_progress(output_dir, -1.0, f"❌ 失败: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
