# docking_utils.py
"""
分子对接工具模块 (Smina / AutoDock Vina)
封装 PDB→PDBQT 转换、SMILES→PDBQT 转换、结合口袋计算、Smina 对接调用与结果解析。

使用方式:
    from docking_utils import run_docking
    result = run_docking(pdb_id="2ITO", ligand_smiles="COC1=CC=C2...")
"""

import os
import re
import tempfile
import subprocess
import urllib.request
from pathlib import Path

import numpy as np
import streamlit as st

# ---------- 懒加载重型依赖 ----------

_openbabel_available = None


def _ensure_openbabel():
    """确保 openbabel 已安装并可用（懒加载）。

    注意：在 `import openbabel` **之前**必须先补齐 conda 激活等价物
    （PATH / BABEL_DATADIR）——否则 Open Babel 的插件与数据目录找不到，
    会表现为"pdb/sdf/mol2 格式未注册"，`pybel.readfile("pdb", ...)` 直接报错。
    """
    global _openbabel_available
    if _openbabel_available is not None:
        return _openbabel_available
    try:
        from utils.runtime_env import ensure_conda_runtime_env

        ensure_conda_runtime_env()
    except Exception:  # 不在包内被导入时（如直接 python docking_utils.py）忽略
        pass
    try:
        from openbabel import pybel  # noqa: F811
        _openbabel_available = True
    except ImportError:
        _openbabel_available = False
    return _openbabel_available


_py3dmol_available = None


def _ensure_py3dmol():
    """确保 py3Dmol 已安装（懒加载）"""
    global _py3dmol_available
    if _py3dmol_available is not None:
        return _py3dmol_available
    try:
        import py3Dmol  # noqa: F811
        _py3dmol_available = True
    except ImportError:
        _py3dmol_available = False
    return _py3dmol_available


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
    # NOTE: GetPartialCharge() 只读取不设置，无实际操作；PDBQT 写出时 OpenBabel 自动计算电荷
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
    # NOTE: 原代码的 GetPartialCharge() 只读取不设置，是无操作；
    # OpenBabel 写 PDBQT 时会自动计算 Gasteiger 电荷
    molecule.write("pdbqt", str(pdbqt_path), overwrite=True)


# ---------- 3. PDB 下载与纯文本解析 ----------

PDB_CACHE_DIR = os.path.join(tempfile.gettempdir(), "pdb_cache")


def _download_pdb(pdb_id: str) -> str:
    """从 RCSB 下载 PDB 文件并缓存到本地临时目录"""
    # 严格校验 PDB ID（4 位字母数字，首字符为数字），防止路径遍历
    if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb_id):
        raise ValueError(f"无效的 PDB ID: {pdb_id!r}（应为 4 位字母数字，如 3POZ）")
    os.makedirs(PDB_CACHE_DIR, exist_ok=True)
    pdb_path = os.path.join(PDB_CACHE_DIR, f"{pdb_id}.pdb")
    if not os.path.exists(pdb_path):
        pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        try:
            urllib.request.urlretrieve(pdb_url, pdb_path)
        except Exception as e:
            raise RuntimeError(f"PDB 下载失败: {pdb_id}: {e}") from e
        # 校验下载内容确实是 PDB 数据（404 页面会被 RCSB 返回 HTML）
        try:
            with open(pdb_path, "r", errors="replace") as f:
                head = f.read(200).lstrip()
        except Exception:
            head = ""
        if not head.startswith(("HEADER", "ATOM", "REMARK", "CRYST1", "MODEL", "TITLE")):
            os.unlink(pdb_path)
            raise ValueError(f"PDB ID {pdb_id} 无效或返回内容不是 PDB 数据")
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

def find_smina_executable():
    """查找 smina 可执行文件，返回绝对路径；找不到返回 None。

    为什么要这么麻烦：Smina 是 C++ 原生二进制（PyPI 上没有对应的 pip 包），
    而且用 Streamlit 启动时，**子进程的 PATH 未必包含 conda 环境的 Library\\bin**
    （Windows 下 smina.exe 恰恰就装在那里）。只查 PATH 会把"已安装"误判成"未安装"，
    所以这里额外扫描当前环境与同机其它 conda 环境目录。
    """
    import shutil as _shutil
    import sys as _sys

    # 1) 先查 PATH（conda activate 后 Scripts / Library\\bin 都在 PATH 上）
    for name in ("smina", "smina.exe"):
        found = _shutil.which(name)
        if found:
            return found

    # 2) 再扫环境目录：当前环境 + CONDA_PREFIX + 同机其它 envs/*
    prefixes = []
    for env_var in ("CONDA_PREFIX", "VIRTUAL_ENV"):
        value = os.environ.get(env_var)
        if value:
            prefixes.append(value)
    prefixes.append(_sys.prefix)
    base_prefix = getattr(_sys, "base_prefix", None)
    if base_prefix and base_prefix not in prefixes:
        prefixes.append(base_prefix)
    envs_dir = os.path.join(os.path.dirname(_sys.prefix), "envs")
    if os.path.isdir(envs_dir):
        try:
            for env_name in sorted(os.listdir(envs_dir)):
                prefixes.append(os.path.join(envs_dir, env_name))
        except OSError:
            pass

    relative_dirs = (
        ("Library", "bin"),      # conda on Windows
        ("Scripts",),            # venv / conda scripts
        ("bin",),                # conda on Linux / venv
        ("Library", "usr", "bin"),
    )
    for prefix in prefixes:
        for rel in relative_dirs:
            for exe_name in ("smina", "smina.exe"):
                candidate = os.path.join(prefix, *rel, exe_name)
                if os.path.isfile(candidate):
                    return candidate

    # 3) 最后看几个常见系统路径（官方静态二进制常放这里）
    for candidate in ("/usr/local/bin/smina", "/usr/bin/smina", "/opt/smina/smina"):
        if os.path.isfile(candidate):
            return candidate
    return None


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
    # 用 find_smina_executable() 解析绝对路径：只查 PATH 会在
    # "conda 装了但 PATH 里没有 Library\\bin" 的情况下误报未安装。
    smina_exe = find_smina_executable()
    if smina_exe is None:
        raise FileNotFoundError(
            "未找到 smina 可执行文件。它不是 pip 包（PyPI 上没有 smina 这个项目，"
            "pip install smina 会 404），只能用 conda 安装或放置官方静态二进制：\n"
            "  conda install -c conda-forge smina\n"
            "  或从 https://sourceforge.net/projects/smina/ 下载 smina.static 到 /usr/local/bin/smina"
            "若确认已安装仍报此错，通常是启动应用时没有激活对应的 conda 环境。"
        )

    cmd = [
        smina_exe,
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
    output_text = subprocess.check_output(
        cmd, universal_newlines=True, timeout=1800
    )  # 30 分钟超时，避免 Smina 挂起卡死页面
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
    buffer: float = 5.0,
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
    buffer : float
        口袋缓冲距离 (Å)，默认 5.0

    Returns
    -------
    dict : {
        "results":      对接结果列表,
        "view":         py3Dmol 3D 可视化对象,
        "sdf_data":     SDF 文件二进制内容,
        "output_text":  smina 原始输出文本,
    }
    """
    if not _ensure_openbabel():
        raise ImportError("openbabel 未安装，请执行: pip install openbabel")
    if not _ensure_py3dmol():
        raise ImportError("py3Dmol 未安装，请执行: pip install py3Dmol")

    import py3Dmol

    # 跟踪需要清理的临时文件（mkstemp/NamedTemporaryFile 不会自动删除）
    _temp_files = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        # ----- 步骤 1: 获取蛋白 PDB + 解析口袋 -----
        if pdb_id:
            raw_pdb_path = _download_pdb(pdb_id)
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=buffer
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
        elif pdb_content:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdb") as tmpf:
                tmpf.write(pdb_content)
                raw_pdb_path = tmpf.name
                _temp_files.append(raw_pdb_path)
            protein_pdb, pocket, detected_resname = _parse_pdb_for_docking(
                raw_pdb_path, ligand_resname, buffer=buffer
            )
            if ligand_resname is None:
                ligand_resname = detected_resname
        else:
            raise ValueError("请提供 PDB ID 或 PDB 文件内容")

        # _parse_pdb_for_docking 用 mkstemp 创建的蛋白 PDB 也需清理
        _temp_files.append(protein_pdb)

        try:
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

            # ----- 步骤 6: py3Dmol 3D 可视化 -----
            with open(sdf_out, "r") as f:
                sdf_str = f.read()
            view = py3Dmol.view(width=800, height=600)
            view.addModel(sdf_str, 'sdf')
            view.setStyle({'model': -1}, {'stick': {}})
            view.zoomTo()

            # ----- 读取 SDF 用于下载 -----
            with open(sdf_out, "rb") as f:
                sdf_data = f.read()

            return {
                "results": results,
                "view": view,
                "sdf_data": sdf_data,
                "output_text": output_text,
            }
        finally:
            # 清理系统 temp 目录下的临时 PDB（避免长时运行累积泄漏）
            for _p in _temp_files:
                try:
                    if _p and os.path.exists(_p):
                        os.unlink(_p)
                except Exception:
                    pass
