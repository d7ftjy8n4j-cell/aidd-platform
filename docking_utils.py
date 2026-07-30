# docking_utils.py
"""
分子对接工具模块 (Smina / AutoDock Vina)
封装 PDB→PDBQT 转换、SMILES→PDBQT 转换、结合口袋计算、Smina 对接调用与结果解析。

使用方式:
    from docking_utils import run_docking
    result = run_docking(pdb_id="2ITO", ligand_smiles="COC1=CC=C2...")
"""

import os
import tempfile
import subprocess
import urllib.request
from pathlib import Path

import streamlit as st

# ---------- 懒加载重型依赖 ----------

_openbabel_available = None


def _ensure_openbabel():
    """确保 openbabel 已安装并可用（懒加载）"""
    global _openbabel_available
    if _openbabel_available is not None:
        return _openbabel_available
    try:
        from openbabel import pybel  # noqa: F811
        _openbabel_available = True
    except ImportError:
        _openbabel_available = False
    return _openbabel_available


_nglview_available = None


def _ensure_nglview():
    """确保 nglview 已安装（懒加载）"""
    global _nglview_available
    if _nglview_available is not None:
        return _nglview_available
    try:
        import nglview  # noqa: F811
        _nglview_available = True
    except ImportError:
        _nglview_available = False
    return _nglview_available


# ---------- 1. PDB → PDBQT 转换（蛋白） ----------

def pdb_to_pdbqt(pdb_path: str, pdbqt_path: str, pH: float = 7.4):
    """
    将 PDB 文件转换为 PDBQT 格式（AutoDock 家族所需）

    Parameters
    ----------
    pdb_path : str
        输入 PDB 文件路径
    pdbqt_path : str
        输出 PDBQT 文件路径
    pH : float
        质子化状态 pH，默认 7.4
    """
    if not _ensure_openbabel():
        raise ImportError("openbabel 未安装，请执行: pip install openbabel")

    from openbabel import pybel

    molecule = list(pybel.readfile("pdb", str(pdb_path)))[0]
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)


# ---------- 2. SMILES → PDBQT 转换（配体） ----------

def smiles_to_pdbqt(smiles: str, pdbqt_path: str, pH: float = 7.4):
    """
    将 SMILES 字符串转换为 PDBQT 格式（3D 构象生成 + 加氢 + 电荷）

    Parameters
    ----------
    smiles : str
        配体 SMILES 字符串
    pdbqt_path : str
        输出 PDBQT 文件路径
    pH : float
        质子化状态 pH，默认 7.4
    """
    if not _ensure_openbabel():
        raise ImportError("openbabel 未安装，请执行: pip install openbabel")

    from openbabel import pybel

    molecule = pybel.readstring("smi", smiles)
    molecule.OBMol.CorrectForPH(pH)
    molecule.addh()
    molecule.make3D(forcefield="mmff94s", steps=10000)
    for atom in molecule.atoms:
        atom.OBAtom.GetPartialCharge()
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)


# ---------- 3. 从 PDB ID 下载结构 ----------

def fetch_structure(pdb_id: str):
    """
    从 RCSB PDB 下载结构，返回 MDAnalysis Universe

    Parameters
    ----------
    pdb_id : str
        4 位 PDB ID（如 "2ITO"）

    Returns
    -------
    mda.Universe 或类似结构对象
    """
    try:
        from opencadd.structure.core import Structure
    except ImportError:
        raise ImportError("opencadd 未安装，请执行: pip install opencadd")

    structure = Structure.from_pdbid(pdb_id)
    # 确保有 elements 属性
    if not hasattr(structure.atoms, "elements"):
        structure.add_TopologyAttr("elements", structure.atoms.types)
    return structure


# ---------- 4. 计算结合口袋（基于共晶配体） ----------

def calculate_pocket_from_ligand(structure, ligand_resname: str, buffer: float = 5.0):
    """
    基于共晶配体坐标计算对接盒子的中心与尺寸

    Parameters
    ----------
    structure : MDAnalysis Universe
        蛋白-配体复合物结构
    ligand_resname : str
        配体残基名（如 "IRE"）
    buffer : float
        盒子各方向缓冲距离（Å），默认 5.0

    Returns
    -------
    dict : {"center": [x, y, z], "size": [sx, sy, sz]}
    """
    ligand = structure.select_atoms(f"resname {ligand_resname}")
    if len(ligand) == 0:
        raise ValueError(f"未找到配体残基: {ligand_resname}，请检查残基名或手动指定")

    positions = ligand.positions
    center = (positions.max(axis=0) + positions.min(axis=0)) / 2
    size = positions.max(axis=0) - positions.min(axis=0) + buffer

    return {
        "center": center.tolist(),
        "size": size.tolist(),
    }


# ---------- 5. 执行 Smina 对接 ----------

def run_smina(
    ligand_path: str,
    protein_path: str,
    out_path: str,
    pocket_center: list,
    pocket_size: list,
    num_poses: int = 10,
    exhaustiveness: int = 8,
) -> str:
    """
    调用 Smina 命令行执行分子对接

    Parameters
    ----------
    ligand_path : str
        配体 PDBQT 文件路径
    protein_path : str
        蛋白 PDBQT 文件路径
    out_path : str
        输出 SDF 文件路径
    pocket_center : list
        盒子中心 [x, y, z]
    pocket_size : list
        盒子尺寸 [sx, sy, sz]
    num_poses : int
        保留构象数，默认 10
    exhaustiveness : int
        搜索精度（4-32），默认 8

    Returns
    -------
    str : smina 标准输出文本
    """
    cmd = [
        "smina",
        "--ligand", str(ligand_path),
        "--receptor", str(protein_path),
        "--out", str(out_path),
        "--center_x", str(pocket_center[0]),
        "--center_y", str(pocket_center[1]),
        "--center_z", str(pocket_center[2]),
        "--size_x", str(pocket_size[0]),
        "--size_y", str(pocket_size[1]),
        "--size_z", str(pocket_size[2]),
        "--num_modes", str(num_poses),
        "--exhaustiveness", str(exhaustiveness),
    ]
    output_text = subprocess.check_output(cmd, universal_newlines=True)
    return output_text


# ---------- 6. 解析 Smina 输出 ----------

def parse_smina_output(output_text: str):
    """
    从 Smina 标准输出中提取各构象结合能和 RMSD

    Parameters
    ----------
    output_text : str
        smina 命令行标准输出

    Returns
    -------
    list[dict] : [{"mode": 1, "affinity": -9.2, "rmsd_lb": 0.0, "rmsd_ub": 0.0}, ...]
    """
    lines = output_text.strip().split("\n")
    results = []
    in_table = False

    for line in lines:
        if "mode | affinity" in line or "-----+" in line:
            in_table = True
            continue
        if in_table and line.strip():
            parts = line.split()
            if len(parts) >= 4 and parts[0].isdigit():
                results.append({
                    "mode": int(parts[0]),
                    "affinity": float(parts[1]),
                    "rmsd_lb": float(parts[2]),
                    "rmsd_ub": float(parts[3]),
                })

    return results


# ---------- 7. 主对接流程 ----------

def run_docking(
    pdb_id: str = None,
    pdb_content: bytes = None,
    ligand_smiles: str = None,
    ligand_resname: str = None,
    num_poses: int = 10,
    exhaustiveness: int = 8,
):
    """
    完整的分子对接流程：下载/读取蛋白 → 格式转换 → 口袋计算 → Smina 对接 → 结果解析

    Parameters
    ----------
    pdb_id : str, optional
        PDB ID（如 "2ITO"）
    pdb_content : bytes, optional
        上传的 PDB 文件内容
    ligand_smiles : str
        配体 SMILES 字符串
    ligand_resname : str, optional
        配体残基名。不提供则自动检测（非蛋白、非水的第一个残基）
    num_poses : int
        保留构象数，默认 10
    exhaustiveness : int
        搜索精度，默认 8

    Returns
    -------
    dict : {
        "results":      对接结果列表,
        "view":         nglview 3D 可视化对象,
        "sdf_data":     SDF 文件二进制内容,
        "output_text":  smina 原始输出文本,
    }
    """
    if not _ensure_openbabel():
        raise ImportError("openbabel 未安装，请执行: pip install openbabel")
    if not _ensure_nglview():
        raise ImportError("nglview 未安装，请执行: pip install nglview")

    import nglview as nv
    from opencadd.structure.core import Structure

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ----- 步骤 1: 获取蛋白结构 -----
        if pdb_id:
            structure = fetch_structure(pdb_id)
            protein_pdb = tmp / "protein.pdb"
            structure.select_atoms("protein").write(str(protein_pdb))

            # 自动检测配体残基名（非蛋白、非水的第一个残基）
            if ligand_resname is None:
                not_protein = structure.select_atoms("not protein and not resname HOH")
                if len(not_protein) > 0:
                    all_resnames = set(not_protein.resnames)
                    if all_resnames:
                        ligand_resname = sorted(all_resnames)[0]
                else:
                    raise ValueError(
                        "未找到共晶配体残基。请手动指定 ligand_resname 参数"
                    )

        elif pdb_content:
            protein_pdb = tmp / "protein.pdb"
            with open(protein_pdb, "wb") as f:
                f.write(pdb_content)
            # 从已保存文件读取结构以检测配体
            structure = Structure.from_pdb(str(protein_pdb))
            if not hasattr(structure.atoms, "elements"):
                structure.add_TopologyAttr("elements", structure.atoms.types)

            if ligand_resname is None:
                not_protein = structure.select_atoms("not protein and not resname HOH")
                if len(not_protein) > 0:
                    all_resnames = set(not_protein.resnames)
                    if all_resnames:
                        ligand_resname = sorted(all_resnames)[0]
        else:
            raise ValueError("请提供 PDB ID 或 PDB 文件内容")

        # ----- 步骤 2: 蛋白转 PDBQT -----
        protein_pdbqt = tmp / "protein.pdbqt"
        pdb_to_pdbqt(str(protein_pdb), str(protein_pdbqt))

        # ----- 步骤 3: 配体转 PDBQT -----
        if not ligand_smiles:
            raise ValueError("请提供配体 SMILES 字符串")
        ligand_pdbqt = tmp / "ligand.pdbqt"
        smiles_to_pdbqt(ligand_smiles, str(ligand_pdbqt))

        # ----- 步骤 4: 计算结合口袋 -----
        pocket = calculate_pocket_from_ligand(structure, ligand_resname)

        # ----- 步骤 5: 执行 Smina 对接 -----
        sdf_out = tmp / "docking_poses.sdf"
        output_text = run_smina(
            str(ligand_pdbqt),
            str(protein_pdbqt),
            str(sdf_out),
            pocket["center"],
            pocket["size"],
            num_poses,
            exhaustiveness,
        )

        # ----- 步骤 6: 解析结果 -----
        results = parse_smina_output(output_text)

        # ----- 步骤 7: 创建 NGLView 3D 可视化 -----
        view = nv.show_file(str(sdf_out))
        view.add_representation("cartoon", selection="protein")
        view.add_representation("licorice", selection="ligand")

        # ----- 读取 SDF 用于下载 -----
        with open(sdf_out, "rb") as f:
            sdf_data = f.read()

        return {
            "results": results,
            "view": view,
            "sdf_data": sdf_data,
            "output_text": output_text,
        }
