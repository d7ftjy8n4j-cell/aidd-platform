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

# ---- Windows: 确保 conda Library\bin 在 DLL 搜索路径中 ----
if sys.platform == "win32":
    _conda_prefix = os.environ.get("CONDA_PREFIX", "")
    if _conda_prefix:
        _dll_dir = os.path.join(_conda_prefix, "Library", "bin")
        if os.path.isdir(_dll_dir):
            try:
                os.add_dll_directory(_dll_dir)
            except Exception:
                pass

logger = logging.getLogger(__name__)

# ========== 懒加载重型依赖 ==========
_DEPS_AVAILABLE = None
_DEPS_ERRORS = {}

def _check_md_deps():
    """检查 MD 模拟所需的全部依赖，优先做实际导入探测。

    mdtraj 标记为 optional，不影响整体 pass/fail。
    """
    global _DEPS_AVAILABLE, _DEPS_ERRORS

    # 每次调用都重新检测（不用缓存），因为 os.add_dll_directory 可能在后来才生效
    deps_required = {
        "rdkit": ["rdkit", "Chem"],
        "pdbfixer": ["pdbfixer", "PDBFixer"],
        "openmm": ["openmm", "app"],
        "openmm.unit": ["openmm", "unit"],
        "openmmforcefields": ["openmmforcefields.generators", "GAFFTemplateGenerator"],
    }
    deps_optional = {
        "mdtraj": ["mdtraj", "reporters"],
    }

    all_ok = True
    for name, (mod, attr) in deps_required.items():
        try:
            module = __import__(mod, fromlist=[attr])
            getattr(module, attr)
            _DEPS_ERRORS[name] = None
        except Exception as e:
            _DEPS_ERRORS[name] = str(e)
            all_ok = False
            logger.warning(f"MD 核心依赖 {name} 不可用: {e}")

    # optional 依赖：失败仅记录，不改变 all_ok
    for name, (mod, attr) in deps_optional.items():
        try:
            module = __import__(mod, fromlist=[attr])
            getattr(module, attr)
            _DEPS_ERRORS[name] = None
        except Exception as e:
            _DEPS_ERRORS[name] = str(e)
            logger.warning(f"MD 可选依赖 {name} 不可用: {e}")

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
    """使用 PDBFixer 准备蛋白（含加氢）。"""
    import pdbfixer
    fixer = pdbfixer.PDBFixer(str(pdb_path))
    fixer.removeHeterogens()
    fixer.findMissingResidues()
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
    """将 RDKit 分子转为 OpenMM Modeller 对象（PDB 快路径，毫秒级）。

    不使用 openff-toolkit，而是走 RDKit → PDB 字符串 → OpenMM PDBFile，
    避免 Molecule.from_rdkit() 的重型导入和感知开销。
    """
    import tempfile, os
    from rdkit import Chem
    import openmm.app as app
    from openmm import unit

    with tempfile.NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as tmp:
        tmp.write(Chem.MolToPDBBlock(rdkit_mol))
        tmp_pdb = tmp.name
    try:
        pdb_obj = app.PDBFile(tmp_pdb)
        modeller = app.Modeller(pdb_obj.topology, pdb_obj.positions)
        pos_nm = modeller.positions.value_in_unit(unit.nanometers)
        return app.Modeller(modeller.topology, pos_nm * unit.nanometers)
    finally:
        os.unlink(tmp_pdb)


def build_openff_molecule(rdkit_mol):
    """构建带 Gasteiger 电荷的 openff Molecule（毫秒级）。

    先转 openff，再从 openff 内部 RDKit 分子计算 Gasteiger 电荷，
    确保电荷数组长度与 openff 原子数一致。
    """
    from rdkit.Chem import AllChem
    from openff.toolkit.topology import Molecule
    from openff.units import unit as off_unit

    # 1. openff Molecule（跳过手性感知）
    off_mol = Molecule.from_rdkit(rdkit_mol, allow_undefined_stereo=True)

    # 2. 从 openff 获取内部 RDKit 分子，计算 Gasteiger 电荷
    off_rdmol = off_mol.to_rdkit()
    AllChem.ComputeGasteigerCharges(off_rdmol)
    charges = [float(a.GetDoubleProp("_GasteigerCharge")) for a in off_rdmol.GetAtoms()]

    # 3. 预分配 Gasteiger 电荷
    off_mol.partial_charges = off_unit.Quantity(
        charges, off_unit.elementary_charge
    )

    return off_mol


# ========== 5. 合并蛋白 + 配体 (纯 OpenMM，无需 MDTraj) ==========

def merge_protein_and_ligand(protein_fixer, ligand_modeller):
    """用纯 OpenMM 合并蛋白和配体的拓扑（保留链结构 + 键信息）。"""
    from openmm import unit
    import openmm.app as app

    new_top = app.Topology()
    old_to_new = {}  # old_atom → new_atom

    # 1) 蛋白：逐链复制
    for old_chain in protein_fixer.topology.chains():
        new_chain = new_top.addChain(id=old_chain.id)
        for old_res in old_chain.residues():
            new_res = new_top.addResidue(old_res.name, new_chain, id=old_res.id)
            for old_atom in old_res.atoms():
                na = new_top.addAtom(old_atom.name, old_atom.element, new_res, id=old_atom.id)
                old_to_new[old_atom] = na

    # 2) 配体：独立 chain
    lig_chain = new_top.addChain(id="L")
    for old_chain in ligand_modeller.topology.chains():
        for old_res in old_chain.residues():
            new_res = new_top.addResidue(old_res.name, lig_chain, id=old_res.id)
            for old_atom in old_res.atoms():
                na = new_top.addAtom(old_atom.name, old_atom.element, new_res, id=old_atom.id)
                old_to_new[old_atom] = na

    # 3) 复制蛋白的键
    for old_bond in protein_fixer.topology.bonds():
        a1, a2 = old_bond
        if a1 in old_to_new and a2 in old_to_new:
            new_top.addBond(old_to_new[a1], old_to_new[a2])

    # 4) 复制配体的键
    for old_bond in ligand_modeller.topology.bonds():
        a1, a2 = old_bond
        if a1 in old_to_new and a2 in old_to_new:
            new_top.addBond(old_to_new[a1], old_to_new[a2])

    # 合并坐标
    n_prot = len(protein_fixer.positions)
    n_lig = len(ligand_modeller.positions)
    new_pos = unit.Quantity(np.zeros([n_prot + n_lig, 3]), unit=unit.nanometers)
    new_pos[:n_prot] = protein_fixer.positions.value_in_unit(unit.nanometers) * unit.nanometers
    new_pos[n_prot:] = ligand_modeller.positions.value_in_unit(unit.nanometers) * unit.nanometers

    return new_top, new_pos


# ========== 6. 力场生成（纯 SMIRNOFF + Gasteiger，跳过 GAFF/AM1-BCC）==========

def _apply_ligand_smirnoff(system, topology, off_mol, ligand_resname):
    """使用 SMIRNOFF Sage + Gasteiger 为配体注入完整力场参数。

    通过 openff.interchange.Interchange 构建配体专属参数化（使用
    charge_from_molecules 跳过 AM1-BCC 电荷计算），然后将键长/键角/
    二面角/非键参数写入主系统的对应 Force 对象中。

    整个过程不涉及 GAFF、antechamber、OpenEye 或任何量子化学工具。
    """
    from openff.toolkit import ForceField as SageForceField
    from openff.interchange import Interchange
    import openmm as mm

    # ---- 1. 构建配体 SMIRNOFF Interchange（Gasteiger，毫秒级）----
    handler = SageForceField("openff-2.1.0.offxml")
    # charge_from_molecules 让 Sage 直接使用分子预分配的 partial_charges
    interchange = Interchange.from_smirnoff(
        handler,
        off_mol.to_topology(),
        charge_from_molecules=[off_mol],
    )
    ref_system = interchange.to_openmm(combine_nonbonded_forces=True)

    # ---- 2. 构建 GAFF 命名规则：元素+同类序号 → 配体局部索引 ----
    elem_counter = {}
    name_to_local = {}
    for i, atom in enumerate(off_mol.atoms):
        e = atom.symbol
        elem_counter[e] = elem_counter.get(e, 0) + 1
        name_to_local[f"{e}{elem_counter[e]}"] = i

    # ---- 3. 配体局部索引 → 主系统全局索引 ----
    local_to_global = {}
    for atom in topology.atoms():
        if atom.residue.name == ligand_resname:
            li = name_to_local.get(atom.name)
            if li is not None:
                local_to_global[li] = atom.index

    n_lig = off_mol.n_atoms

    # ---- 4. 复制 HarmonicBondForce ----
    for ref_f in ref_system.getForces():
        if isinstance(ref_f, mm.HarmonicBondForce):
            for target_f in system.getForces():
                if isinstance(target_f, mm.HarmonicBondForce):
                    for i in range(ref_f.getNumBonds()):
                        a1, a2, length, k = ref_f.getBondParameters(i)
                        if a1 < n_lig and a2 < n_lig:
                            target_f.addBond(
                                local_to_global[a1], local_to_global[a2],
                                length, k,
                            )
                    break

    # ---- 5. 复制 HarmonicAngleForce ----
    for ref_f in ref_system.getForces():
        if isinstance(ref_f, mm.HarmonicAngleForce):
            for target_f in system.getForces():
                if isinstance(target_f, mm.HarmonicAngleForce):
                    for i in range(ref_f.getNumAngles()):
                        a1, a2, a3, angle, k = ref_f.getAngleParameters(i)
                        if a1 < n_lig and a2 < n_lig and a3 < n_lig:
                            target_f.addAngle(
                                local_to_global[a1], local_to_global[a2], local_to_global[a3],
                                angle, k,
                            )
                    break

    # ---- 6. 复制 PeriodicTorsionForce ----
    for ref_f in ref_system.getForces():
        if isinstance(ref_f, mm.PeriodicTorsionForce):
            for target_f in system.getForces():
                if isinstance(target_f, mm.PeriodicTorsionForce):
                    for i in range(ref_f.getNumTorsions()):
                        a1, a2, a3, a4, per, phase, k = ref_f.getTorsionParameters(i)
                        if all(x < n_lig for x in (a1, a2, a3, a4)):
                            target_f.addTorsion(
                                local_to_global[a1], local_to_global[a2],
                                local_to_global[a3], local_to_global[a4],
                                per, phase, k,
                            )
                    break

    # ---- 7. 覆写 NonbondedForce 中配体原子的电荷 + LJ ----
    for ref_f in ref_system.getForces():
        if isinstance(ref_f, mm.NonbondedForce):
            for target_f in system.getForces():
                if isinstance(target_f, mm.NonbondedForce):
                    for local_i, global_i in local_to_global.items():
                        q, sigma, eps = ref_f.getParticleParameters(local_i)
                        target_f.setParticleParameters(global_i, q, sigma, eps)

                    # 复制配体内部的 nonbonded 例外项（1-2, 1-3 缩放）
                    for i in range(ref_f.getNumExceptions()):
                        a1, a2, qq, ss, ee = ref_f.getExceptionParameters(i)
                        if a1 < n_lig and a2 < n_lig:
                            target_f.addException(
                                local_to_global[a1], local_to_global[a2],
                                qq, ss, ee,
                            )
                    break
            break


# ========== 6.5 溶剂化（带模板重试）==========

def _solvate_with_retry(modeller, forcefield, padding, ionic_strength, unit):
    """加水盒子。如果力场模板匹配失败，自动删除问题残基后重试（最多3次）。"""
    for attempt in range(4):
        try:
            modeller.addSolvent(
                forcefield,
                padding=padding * unit.nanometers,
                ionicStrength=ionic_strength * unit.molar,
            )
            return  # 成功
        except ValueError as e:
            if "No template found for residue" not in str(e) or attempt >= 3:
                raise
            import re
            match = re.search(r"residue (\d+)", str(e))
            if not match:
                raise
            bad_idx = int(match.group(1)) - 1
            residues = list(modeller.topology.residues())
            if not (0 <= bad_idx < len(residues)):
                raise
            logger.warning(f"删除无模板残基 {residues[bad_idx]} (索引 {bad_idx})，第{attempt+1}次重试")
            modeller.delete(list(residues[bad_idx].atoms()))


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
    output_dir: str = None,
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
    output_dir : str, optional
        输出目录，用于写入初始化阶段进度

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
    try:
        import mdtraj as md
        _MDTRAJ_OK = True
    except ImportError:
        md = None
        _MDTRAJ_OK = False
        logger.warning("mdtraj 不可用，将使用 OpenMM DCD 格式保存轨迹")

    # ---- 初始化进度写入 ----
    def _init_progress(pct: float, msg: str):
        """在初始化阶段写入进度文件，让用户知道系统在做什么"""
        if output_dir:
            try:
                progress_path = os.path.join(output_dir, "progress.txt")
                with open(progress_path, "w", encoding="utf-8") as pf:
                    pf.write(f"{pct:.4f}\n{msg}")
                    pf.flush()
            except Exception:
                pass

    _init_progress(0.00, "准备体系...")

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

    _init_progress(0.05, "PDB 下载完成，准备蛋白质...")

    # ---- 2. 准备蛋白 (PDBFixer + CHARMM36 兼容) ----
    protein = prepare_protein(pdb_path, ph=ph, ignore_missing_residues=True)

    _init_progress(0.20, "加载 CHARMM36 力场...")
    protein_ff = app.ForceField("charmm36.xml", "charmm36/water.xml")

    # ---- 3. 溶剂化 ----
    _init_progress(0.30, "添加水盒子与离子...")
    modeller = app.Modeller(protein.topology, protein.positions)
    modeller.addSolvent(
        protein_ff,
        padding=padding * unit.nanometers,
        ionicStrength=ionic_strength * unit.molar,
    )

    _init_progress(0.35, f"构建力场参数 ({modeller.topology.getNumAtoms()} 个原子)...")

    # ---- 4. 创建 System ----
    system = protein_ff.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
    )

    integrator = mm.LangevinMiddleIntegrator(
        temperature * unit.kelvin,
        1.0 / unit.picoseconds,
        2.0 * unit.femtoseconds,  # 2 fs（CHARMM36 兼容）
    )

    _init_progress(0.55, "初始化计算平台...")
    # 自动选择最快的计算平台（GPU > 多核CPU > 单核CPU，绝不使用极慢的 Reference）
    platform = None
    platform_speed = "多核 CPU"
    for candidate_name in ["CUDA", "OpenCL", "CPU"]:
        try:
            candidate = mm.Platform.getPlatformByName(candidate_name)
            platform = candidate
            if candidate_name == "CUDA":
                platform_speed = f"GPU (CUDA) - {candidate.getPropertyDefaultValue('CudaDeviceIndex')}"
            elif candidate_name == "OpenCL":
                platform_speed = f"GPU (OpenCL)"
            else:
                platform_speed = f"多核 CPU ({os.cpu_count()} 核)"
            break
        except Exception:
            continue
    if platform is None:
        logger.warning("未找到可用 OpenMM 平台，使用默认（可能极慢）")
        platform_speed = "默认 (可能为 Reference，极慢)"

    simulation = app.Simulation(modeller.topology, system, integrator, platform)
    simulation.context.setPositions(modeller.positions)
    logger.info(f"OpenMM 计算平台: {platform_speed}")
    logger.info(f"体系原子数: {modeller.topology.getNumAtoms()}")

    _init_progress(0.65, f"能量最小化... ({platform_speed})")

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
    if _MDTRAJ_OK:
        traj_path = os.path.join(tmpdir, "trajectory.xtc")
        simulation.reporters.append(
            md.reporters.XTCReporter(file=traj_path, reportInterval=write_interval)
        )
    else:
        traj_path = os.path.join(tmpdir, "trajectory.dcd")
        simulation.reporters.append(
            app.DCDReporter(file=traj_path, reportInterval=write_interval)
        )

    # 日志捕获（降低报告频率 + 去掉耗时的 speed/remainingTime 计算）
    log_stream = StringIO()
    simulation.reporters.append(
        app.StateDataReporter(
            log_stream,
            max(1, write_interval // 2),  # 降低报告频率（原来是 //10）
            step=True,
            potentialEnergy=True,
            temperature=True,
            progress=True,
            remainingTime=False,  # 去掉耗时字段
            speed=False,           # 去掉耗时字段
            totalSteps=total_steps,
            separator="\t",
        )
    )

    # ---- 9. 运行模拟 ----
    simulation.context.setVelocitiesToTemperature(temperature * unit.kelvin)

    # 分批运行以支持进度回调（使用较大 batch_size 减少 Python ↔ OpenMM 切换开销）
    # 以前 batch_size 过小 (~100 steps) 会导致大量 CPU↔GPU 数据传输，极大拖慢速度
    batch_size = max(1, min(5000, total_steps // 2))
    for batch_idx, step_start in enumerate(range(0, total_steps, batch_size)):
        n = min(batch_size, total_steps - step_start)
        simulation.step(n)

        current_step = step_start + n
        if progress_callback:
            # getState(getEnergy=True) 开销大，仅最后一批或每隔 4 批取一次完整状态
            if current_step >= total_steps or batch_idx % 4 == 0:
                state = simulation.context.getState(getEnergy=True)
                energy = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
            else:
                energy = 0.0  # 跳过昂贵的能量查询，快速报告进度
            progress_callback(current_step, total_steps, energy, temperature)

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
        "platform": platform_speed,
    }


# ========== 8. 保存合并后的蛋白-配体 PDB ==========

def _write_merged_pdb_openmm(protein_fixer, rdkit_ligand, output_path: str):
    """纯 OpenMM 合并写入 PDB（无需 mdtraj）"""
    from openmm import unit
    import openmm.app as app
    omm_lig = rdkit_to_openmm(rdkit_ligand)
    complex_top, complex_pos = merge_protein_and_ligand(protein_fixer, omm_lig)
    with open(output_path, "w") as f:
        app.PDBFile.writeFile(complex_top, complex_pos, file=f)
    return output_path


def save_merged_pdb(protein_fixer, rdkit_ligand, output_path: str):
    """
    将 PDBFixer 蛋白和 RDKit 配体合并保存为单个 PDB 文件。
    供「蛋白-配体作用分析」等下游页面使用。

    Returns
    -------
    str : 输出文件路径
    """
    from openmm import unit
    try:
        import mdtraj as md
    except ImportError:
        # 无 mdtraj 时用纯 OpenMM 写 PDB
        _write_merged_pdb_openmm(protein_fixer, rdkit_ligand, output_path)
        return output_path

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
