"""
md_utils.py — 分子动力学模拟引擎
基于 OpenMM + PDBFixer + MDTraj + RDKit
参考: TeachOpenCADD T019 (https://github.com/volkamerlab/TeachOpenCADD)

工作流:
  1. 下载/上传 PDB → 2. 准备蛋白 (PDBFixer) → 3. 准备配体 (RDKit)
  → 4. 合并体系 (MDTraj) → 5. 力场设置 (AMBER + GAFF)
  → 6. 溶剂化 → 7. 能量最小化 → 8. MD 模拟 → 9. 输出轨迹
"""

import os
import sys
import tempfile
import logging
import time
from pathlib import Path
from io import StringIO

import numpy as np
import requests
import streamlit as st

logger = logging.getLogger(__name__)

# ========== 懒加载重型依赖 ==========
_DEPS_AVAILABLE = None
_DEPS_ERRORS = {}

def _check_md_deps():
    """检查 MD 模拟所需的全部依赖"""
    global _DEPS_AVAILABLE, _DEPS_ERRORS
    if _DEPS_AVAILABLE is not None:
        return _DEPS_AVAILABLE

    deps = {
        "rdkit": ["rdkit", "Chem"],
        "pdbfixer": ["pdbfixer", "PDBFixer"],
        "openmm": ["openmm", "app"],
        "openmm.unit": ["openmm", "unit"],
        "openff.toolkit": ["openff.toolkit.topology", "Molecule"],
        "openmmforcefields": ["openmmforcefields.generators", "GAFFTemplateGenerator"],
        "mdtraj": ["mdtraj", "reporters"],
    }

    all_ok = True
    for name, (mod, attr) in deps.items():
        try:
            __import__(mod)
            _DEPS_ERRORS[name] = None
        except ImportError as e:
            _DEPS_ERRORS[name] = str(e)
            all_ok = False
            logger.warning(f"MD 依赖 {name} 不可用: {e}")

    _DEPS_AVAILABLE = all_ok
    return all_ok


def _get_dep_errors():
    """返回缺失的依赖列表"""
    _check_md_deps()
    return {k: v for k, v in _DEPS_ERRORS.items() if v is not None}


# ========== 1. PDB 下载 ==========

def download_pdb(pdb_id: str, save_dir: str = None) -> str:
    """从 RCSB 下载 PDB 文件，返回本地路径"""
    if save_dir is None:
        save_dir = tempfile.gettempdir()
    os.makedirs(save_dir, exist_ok=True)
    pdb_path = os.path.join(save_dir, f"{pdb_id}.pdb")

    if not os.path.exists(pdb_path):
        url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
        r = requests.get(url, timeout=30)
        r.raise_for_status()
        with open(pdb_path, "wb") as f:
            f.write(r.content)
        logger.info(f"PDB 已下载: {pdb_path}")

    return pdb_path


# ========== 2. 蛋白准备 (PDBFixer) ==========

def prepare_protein(
    pdb_path: str,
    ignore_missing_residues: bool = True,
    ignore_terminal_missing_residues: bool = True,
    ph: float = 7.0,
):
    """
    使用 PDBFixer 准备蛋白结构：
    - 去除杂原子（配体等）
    - 补全缺失原子和氢
    - 替换非标准残基
    """
    import pdbfixer
    fixer = pdbfixer.PDBFixer(str(pdb_path))
    fixer.removeHeterogens()  # 去除配体等杂原子
    fixer.findMissingResidues()

    # 可选：忽略末端缺失残基
    if ignore_terminal_missing_residues:
        chains = list(fixer.topology.chains())
        keys = list(fixer.missingResidues.keys())
        for key in keys:
            chain = chains[key[0]]
            if key[1] == 0 or key[1] == len(list(chain.residues())):
                del fixer.missingResidues[key]

    if ignore_missing_residues:
        fixer.missingResidues = {}

    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(ph)
    return fixer


# ========== 3. 配体准备 (RDKit) ==========

def prepare_ligand(pdb_path: str, resname: str, smiles: str):
    """
    使用 RDKit 准备配体：
    - 从 PDB 中提取配体
    - 用 SMILES 模板修正键级
    - 加氢
    """
    from rdkit import Chem
    from rdkit.Chem import AllChem

    mol = Chem.MolFromPDBFile(str(pdb_path))
    if mol is None:
        raise ValueError(f"无法从 {pdb_path} 解析分子")

    mol_split = Chem.rdmolops.SplitMolByPDBResidues(mol)

    if resname not in mol_split:
        available = list(mol_split.keys())
        raise KeyError(f"残基 '{resname}' 不在 PDB 中。可用残基: {available}")

    ligand = mol_split[resname]
    ligand = Chem.RemoveHs(ligand)

    # 用 SMILES 模板修正键级
    reference = Chem.MolFromSmiles(smiles)
    if reference is None:
        raise ValueError(f"无效的 SMILES: {smiles}")

    prepared = AllChem.AssignBondOrdersFromTemplate(reference, ligand)
    prepared.AddConformer(ligand.GetConformer(0))

    # 加氢
    prepared = Chem.rdmolops.AddHs(prepared, addCoords=True)
    prepared = Chem.MolFromMolBlock(Chem.MolToMolBlock(prepared))

    return prepared


# ========== 4. RDKit → OpenMM 转换 ==========

def rdkit_to_openmm(rdkit_mol, name: str = "LIG"):
    """将 RDKit 分子转为 OpenMM Modeller 对象"""
    import openmm.app as app
    from openff.toolkit.topology import Molecule

    off_mol = Molecule.from_rdkit(rdkit_mol)
    off_mol.name = name

    # 命名原子
    elem_count = {}
    for off_atom, rd_atom in zip(off_mol.atoms, rdkit_mol.GetAtoms()):
        elem = rd_atom.GetSymbol()
        elem_count[elem] = elem_count.get(elem, 0) + 1
        off_atom.name = f"{elem}{elem_count[elem]}"

    off_top = off_mol.to_topology()
    mol_top = off_top.to_openmm()
    mol_pos = off_mol.conformers[0].to("nanometers")

    return app.Modeller(mol_top, mol_pos)


# ========== 5. 合并蛋白 + 配体 ==========

def merge_protein_and_ligand(protein_fixer, ligand_modeller):
    """用 MDTraj 合并蛋白和配体"""
    import mdtraj as md
    from openmm import unit

    # 合并拓扑
    prot_top = md.Topology.from_openmm(protein_fixer.topology)
    lig_top = md.Topology.from_openmm(ligand_modeller.topology)
    complex_top = prot_top.join(lig_top).to_openmm()

    # 合并坐标
    n_total = len(protein_fixer.positions) + len(ligand_modeller.positions)
    complex_pos = unit.Quantity(np.zeros([n_total, 3]), unit=unit.nanometers)
    complex_pos[:len(protein_fixer.positions)] = protein_fixer.positions
    complex_pos[len(protein_fixer.positions):] = ligand_modeller.positions

    return complex_top, complex_pos


# ========== 6. 力场生成 ==========

def generate_forcefield(rdkit_mol=None,
                        protein_ff: str = "amber14-all.xml",
                        solvent_ff: str = "amber14/tip3pfb.xml"):
    """生成力场，可选注册小分子 GAFF 参数"""
    import openmm.app as app
    from openmmforcefields.generators import GAFFTemplateGenerator
    from openff.toolkit.topology import Molecule

    forcefield = app.ForceField(protein_ff, solvent_ff)

    if rdkit_mol is not None:
        gaff = GAFFTemplateGenerator(
            forcefield="gaff-2.2.20",
            molecules=Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True),
        )
        forcefield.registerTemplateGenerator(gaff.generator)

    return forcefield


# ========== 7. 主模拟函数 ==========

def run_md_simulation(
    pdb_id: str = None,
    pdb_content: bytes = None,
    ligand_resname: str = "03P",
    ligand_smiles: str = None,
    total_steps: int = 5000,
    write_interval: int = 500,
    temperature: float = 300.0,
    padding: float = 1.0,
    ionic_strength: float = 0.15,
    ph: float = 7.0,
    progress_callback=None,
) -> dict:
    """
    运行完整的 MD 模拟流程。

    Parameters
    ----------
    pdb_id : str
        PDB ID（如 3POZ），与 pdb_content 二选一
    pdb_content : bytes
        上传的 PDB 文件内容
    ligand_resname : str
        配体残基三字母名（默认 03P = TAK-285）
    ligand_smiles : str
        配体的正确 SMILES（用于键级修正），默认使用 3POZ 配体 03P
    total_steps : int
        模拟总步数（每步 2 fs）
    write_interval : int
        轨迹写入间隔
    temperature : float
        模拟温度 (K)
    padding : float
        溶剂盒子 padding (nm)
    ionic_strength : float
        离子强度 (M)
    ph : float
        pH 值
    progress_callback : callable
        进度回调函数 (step, total, energy, temp)

    Returns
    -------
    dict : {
        "topology_pdb": str,        # 能量最小化后的拓扑 PDB 路径
        "trajectory_xtc": str,      # 轨迹文件路径
        "num_atoms": int,           # 总原子数
        "log": str,                 # 模拟日志
    }
    """
    import openmm as mm
    import openmm.app as app
    from openmm import unit
    import mdtraj as md

    # ---- 1. 获取 PDB ----
    tmpdir = tempfile.mkdtemp(prefix="md_")
    if pdb_id:
        pdb_path = download_pdb(pdb_id, tmpdir)
    elif pdb_content:
        pdb_path = os.path.join(tmpdir, "input.pdb")
        content = pdb_content.decode("utf-8") if isinstance(pdb_content, bytes) else pdb_content
        with open(pdb_path, "w") as f:
            f.write(content)
    else:
        raise ValueError("必须提供 pdb_id 或 pdb_content")

    # ---- 2. 准备蛋白 ----
    protein = prepare_protein(pdb_path, ph=ph, ignore_missing_residues=False)

    # ---- 3. 准备配体 ----
    if ligand_smiles is None:
        # 默认使用 3POZ 中 03P (TAK-285) 的 SMILES
        ligand_smiles = "CC(C)(O)CC(=O)NCCn1ccc2ncnc(Nc3ccc(Oc4cccc(c4)C(F)(F)F)c(Cl)c3)c12"

    rdkit_lig = prepare_ligand(pdb_path, ligand_resname, ligand_smiles)

    # ---- 4. 转换 + 合并 ----
    omm_lig = rdkit_to_openmm(rdkit_lig, ligand_resname)
    complex_top, complex_pos = merge_protein_and_ligand(protein, omm_lig)

    # ---- 5. 力场 + 溶剂化 ----
    forcefield = generate_forcefield(rdkit_lig)
    modeller = app.Modeller(complex_top, complex_pos)
    modeller.addSolvent(
        forcefield,
        padding=padding * unit.nanometers,
        ionicStrength=ionic_strength * unit.molar,
    )

    # ---- 6. 创建 System + Integrator ----
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME)
    integrator = mm.LangevinIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds,
    )
    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    # ---- 7. 能量最小化 ----
    simulation.minimizeEnergy()
    topo_path = os.path.join(tmpdir, "topology.pdb")
    with open(topo_path, "w") as f:
        app.PDBFile.writeFile(
            simulation.topology,
            simulation.context.getState(getPositions=True, enforcePeriodicBox=True).getPositions(),
            file=f,
            keepIds=True,
        )

    # ---- 8. 设置 Reporter ----
    traj_path = os.path.join(tmpdir, "trajectory.xtc")
    simulation.reporters.append(
        md.reporters.XTCReporter(file=traj_path, reportInterval=write_interval)
    )

    # 日志捕获
    log_stream = StringIO()
    simulation.reporters.append(
        app.StateDataReporter(
            log_stream,
            max(1, write_interval // 10),
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=True,
            speed=True,
            totalSteps=total_steps,
            separator="\t",
        )
    )

    # ---- 9. 运行模拟 ----
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # 分批运行以支持进度回调
    batch_size = max(1, min(100, total_steps // 20))
    for step_start in range(0, total_steps, batch_size):
        n = min(batch_size, total_steps - step_start)
        simulation.step(n)
        if progress_callback:
            state = simulation.context.getState(getEnergy=True)
            energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            progress_callback(
                step_start + n, total_steps,
                energy,
                temperature
            )

    # ---- 10. 导出代表性结构 PDB ----
    # 注：不对坐标直接求平均（会拉断化学键），此处保存第一帧拓扑供可视化。
    # 真正的代表性帧由 analyze_trajectory() 计算，在 worker 脚本中导出。
    topo_path = os.path.join(tmpdir, "topology.pdb")
    # (topo_path 已在最小化后保存，此处保留变量名兼容性)
    mean_pdb_path = topo_path  # 默认用拓扑PDB；worker 会覆写为代表性帧

    return {
        "topology_pdb": topo_path,
        "trajectory_xtc": traj_path,
        "mean_pdb_path": mean_pdb_path,
        "num_atoms": modeller.topology.getNumAtoms(),
        "log": log_stream.getvalue(),
        "tmpdir": tmpdir,
    }


# ========== 8. 保存合并后的蛋白-配体 PDB ==========

def save_merged_pdb(protein_fixer, rdkit_ligand, output_path: str):
    """
    将 PDBFixer 蛋白和 RDKit 配体合并保存为单个 PDB 文件。
    供「蛋白-配体作用分析」等下游页面使用。

    Returns
    -------
    str : 输出文件路径
    """
    from openmm import unit
    import mdtraj as md

    # 蛋白 → MDTraj topology
    prot_md = md.Topology.from_openmm(protein_fixer.topology)
    prot_xyz = np.array(protein_fixer.positions.value_in_unit(unit.nanometers))
    prot_traj = md.Trajectory(prot_xyz.reshape(1, -1, 3), prot_md)

    # 配体 → MDTraj
    omm_lig_modeller = rdkit_to_openmm(rdkit_ligand)
    lig_md = md.Topology.from_openmm(omm_lig_modeller.topology)
    lig_xyz = np.array(omm_lig_modeller.positions.value_in_unit(unit.nanometers))
    lig_traj = md.Trajectory(lig_xyz.reshape(1, -1, 3), lig_md)

    # 合并并保存
    merged = prot_traj.stack(lig_traj)
    merged.save(output_path)
    return output_path


# ========== 9. 轨迹分析 ==========

def analyze_trajectory(topology_pdb: str, trajectory_xtc: str, ligand_resname: str = "03P"):
    """
    对 MD 轨迹进行物理严谨的分析。

    关键修正 (vs 初版):
    - 配体 RMSD: 先将轨迹对齐到蛋白结合位点骨架 (Cα)，再计算配体重原子 RMSD
    - 代表性结构: 选取最接近平均坐标的那一帧，而非对坐标直接求平均（避免化学键拉断）

    Returns
    -------
    dict : {
        "rmsd_protein": np.ndarray,       # 蛋白骨架 RMSD (Å)
        "rmsd_ligand": np.ndarray,        # 配体重原子 RMSD (Å, 对齐后)
        "rmsf": np.ndarray,               # 残基 RMSF (Å)
        "time_ps": np.ndarray,            # 时间轴 (ps)
        "residue_ids": np.ndarray,        # 残基编号
        "n_frames": int,                  # 帧数
        "representative_frame": int,      # 代表性帧索引 (0-based)
    }
    """
    import mdtraj as md

    traj = md.load(trajectory_xtc, top=topology_pdb)
    n_frames = traj.n_frames
    if n_frames < 2:
        return {
            "rmsd_protein": np.zeros(n_frames),
            "rmsd_ligand": np.zeros(n_frames),
            "rmsf": np.array([]),
            "time_ps": np.arange(n_frames) * (traj.timestep * 1000),
            "residue_ids": np.array([]),
            "n_frames": n_frames,
            "representative_frame": 0,
        }

    # ---- 1. 选取蛋白骨架用于对齐 ----
    protein_atoms = traj.topology.select("protein and backbone")
    if len(protein_atoms) == 0:
        protein_atoms = traj.topology.select("protein and name CA")

    # ---- 2. 对齐轨迹到首帧（基于蛋白骨架） ----
    traj_aligned = traj.superpose(traj, frame=0, atom_indices=protein_atoms)

    # ---- 3. 蛋白骨架 RMSD (Å) ----
    rmsd_protein = md.rmsd(traj_aligned, traj_aligned, frame=0,
                           atom_indices=protein_atoms) * 10  # nm → Å

    # ---- 4. 配体重原子 RMSD (对齐后) ----
    # 先尝试指定残基名，找不到则搜索非蛋白非水的重原子
    ligand_atoms = traj.topology.select(f"resname {ligand_resname} and element != H")
    if len(ligand_atoms) == 0:
        # 试试常见配体残基名
        for guess in ["LIG", "UNL", "UNK", "DRG"]:
            ligand_atoms = traj.topology.select(f"resname {guess} and element != H")
            if len(ligand_atoms) > 0:
                break
    if len(ligand_atoms) == 0:
        # 最后尝试：非蛋白、非水、非离子的重原子
        ligand_atoms = traj.topology.select(
            "not protein and not water and not type Na+ Cl- K+ and element != H"
        )

    if len(ligand_atoms) > 0:
        rmsd_ligand = md.rmsd(traj_aligned, traj_aligned, frame=0,
                              atom_indices=ligand_atoms) * 10
    else:
        rmsd_ligand = np.zeros(n_frames)

    # ---- 5. 蛋白残基 RMSF ----
    if len(protein_atoms) > 0:
        # 用对齐后的轨迹计算
        rmsf_all = md.rmsf(traj_aligned, traj_aligned, atom_indices=protein_atoms) * 10
        # 每个残基取第一个原子
        residue_ids_all = np.array([traj.topology.atom(i).residue.resSeq
                                    for i in protein_atoms])
        _, unique_idx = np.unique(residue_ids_all, return_index=True)
        rmsf = rmsf_all[unique_idx]
        residue_ids = residue_ids_all[unique_idx]
    else:
        rmsf = np.array([])
        residue_ids = np.array([])

    # ---- 6. 找代表性结构（最接近平均坐标的帧） ----
    # 不对坐标直接求平均（会拉断化学键），而是找离"平均构象"最近的帧
    if len(protein_atoms) > 0:
        avg_coords = np.mean(traj_aligned.xyz[:, protein_atoms, :], axis=0)  # (n_atoms, 3)
        frame_dists = np.zeros(n_frames)
        for i in range(n_frames):
            diff = traj_aligned.xyz[i, protein_atoms, :] - avg_coords
            frame_dists[i] = np.sqrt(np.mean(diff ** 2))
        representative_frame = int(np.argmin(frame_dists))
    else:
        representative_frame = n_frames // 2  # 中位数帧

    # ---- 7. 时间轴 ----
    time_ps = np.arange(n_frames) * (traj.timestep * 1000)  # ps

    return {
        "rmsd_protein": rmsd_protein,
        "rmsd_ligand": rmsd_ligand,
        "rmsf": rmsf,
        "time_ps": time_ps,
        "residue_ids": residue_ids,
        "n_frames": n_frames,
        "representative_frame": representative_frame,
    }
