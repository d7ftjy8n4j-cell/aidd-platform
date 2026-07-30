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

import numpy as np
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


# ---------- 3. PDB 下载与纯文本解析 ----------

PDB_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pdb_cache")


def _download_pdb(pdb_id: str) -> str:
    """从 RCSB 下载 PDB 文件并缓存到本地临时目录"""
    os.makedirs(PDB_CACHE_DIR, exist_ok=True)
    pdb_path = os.path.join(PDB_CACHE_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        urllib.request.urlretrieve(pdb_url, pdb_path)
    return pdb_path


def _parse_pdb_for_docking(pdb_path: str, ligand_resname: str = None, buffer: float = 5.0):
    """
    纯文本解析 PDB 文件，不依赖 opencadd / biopython。

    1. 提取所有 ATOM 行 → 写入蛋白 PDB 临时文件
    2. 查找 HETATM 行（非水） → 计算配体坐标口袋

    返回:
        (protein_pdb_path, pocket_dict, detected_ligand_resname)
    """
    protein_lines = []
    ligand_coords = []
    detected_resname = ligand_resname

    with open(pdb_path, "r") as f:
        for line in f:
            if line.startswith("ATOM") or line.startswith("HETATM"):
                rec = line[0:6].strip()
                resname = line[17:20].strip()

                if rec == "ATOM":
                    protein_lines.append(line)
                elif rec == "HETATM" and resname != "HOH":
                    # 发现第一个非水配体时自动确定残基名
                    if detected_resname is None:
                        detected_resname = resname
                    if ligand_resname is None or resname == ligand_resname:
                        try:
                            x = float(line[30:38])
                            y = float(line[38:46])
                            z = float(line[46:54])
                            ligand_coords.append([x, y, z])
                        except ValueError:
                            continue

    if not protein_lines:
        raise ValueError(f"PDB 文件中没有蛋白 ATOM 记录: {pdb_path}")

    # 写蛋白 PDB
    fd, protein_pdb = tempfile.mkstemp(suffix=".pdb")
    with os.fdopen(fd, "w") as f:
        f.writelines(protein_lines)

    # 计算口袋
    if len(ligand_coords) == 0:
        raise ValueError(
            f"未找到共晶配体残基 '{detected_resname or ligand_resname}'。"
            f" 请手动指定 ligand_resname 参数或使用包含配体的 PDB。"
        )

    coords = np.array(ligand_coords)
    center = ((coords.max(axis=0) + coords.min(axis=0)) / 2).tolist()
    size = (coords.max(axis=0) - coords.min(axis=0) + buffer).tolist()

    pocket = {"center": center, "size": size}
    return protein_pdb, pocket, detected_resname


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

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ----- 步骤 1: 获取蛋白 PDB + 解析口袋 -----
        if pdb_id:
            raw_pdb_path = _download_pdb(pdb_id)
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=5.0
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
        elif pdb_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmpf:
                tmpf.write(pdb_content)
                raw_pdb_path = tmpf.name
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=5.0
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
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

        # ----- 步骤 4: 执行 Smina 对接 (口袋已在步骤 1 解析) -----
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

        # ----- 步骤 5: 解析结果 -----
        results = parse_smina_output(output_text)

        # ----- 步骤 6: NGLView 3D 可视化 -----
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
