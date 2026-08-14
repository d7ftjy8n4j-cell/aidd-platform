# utils/mmgbsa_utils.py
"""
MM-GBSA 结合自由能估算核心引擎

基于 OpenMM + MDTraj，从分子动力学轨迹计算蛋白-配体结合自由能。

ΔG_bind = G_complex - G_receptor - G_ligand

技术路线（单轨迹协议）：
1. 从 MD 轨迹中均匀采样帧
2. **一次性**构建 3 个 GBSA-OBC1 隐式溶剂系统（复合物/受体/配体）
3. 循环中仅调用 simulation.context.setPositions() 更新坐标
4. 直接计算当前帧的势能（不做能量最小化，保留 MD 采样构象）
5. 统计分析所有帧的 ΔG

性能关键优化：
- 力场 (amber14-all.xml) 只加载 3 次（三种系统各一次），而非 N_frames * 3 次
- 消除了逐帧写临时 PDB 的文件 I/O 瓶颈
- 直接通过 NumPy 数组切片提取蛋白/配体坐标
"""

import logging
import numpy as np
import os
import sys

# ---- Windows: 修复 OpenMP 冲突 ----
if sys.platform == "win32":
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

logger = logging.getLogger(__name__)


# ============================================================================
#  Amber OBC1 (igb=5) GB 原子半径 (单位: nm)
#  来源: AmberTools 标准参数，Authored by Onufriev, Bashford & Case
#  顺序按原子序数索引，未列出的元素默认使用碳半径 0.17 nm
# ============================================================================
_GB_RADIUS_MAP = {
    1:  0.12,   # H  (Amber: 1.2 Angstrom)
    6:  0.17,   # C  (Amber: 1.7 Angstrom)
    7:  0.155,  # N  (Amber: 1.55 Angstrom)
    8:  0.15,   # O  (Amber: 1.5 Angstrom)
    9:  0.15,   # F  (Amber: 1.5 Angstrom)
    15: 0.185,  # P  (Amber: 1.85 Angstrom)
    16: 0.18,   # S  (Amber: 1.8 Angstrom)
    17: 0.17,   # Cl (Amber: 1.7 Angstrom)
    35: 0.19,   # Br (Amber: 1.9 Angstrom)
    53: 0.21,   # I  (Amber: 2.1 Angstrom)
}
_GB_RADIUS_DEFAULT = 0.17  # nm


# ============================================================================
#  配体索引识别
# ============================================================================

def get_ligand_indices(traj, ligand_resname: str):
    """
    从 MDTraj 轨迹中根据残基名找到配体原子索引。
    避免 MDTraj select() 的 ast.parse bug（残基名含前导零如 "03P" 会被误解析为数字）。
    """
    idx = np.array([], dtype=np.int64)
    for res in traj.topology.residues:
        if res.name.upper() == ligand_resname.upper():
            idx = np.array([a.index for a in res.atoms], dtype=np.int64)
            break
    if len(idx) == 0:
        available = sorted(set(r.name for r in traj.topology.residues))
        raise ValueError(
            f"在轨迹中找不到残基名为 '{ligand_resname}' 的配体。\n"
            f"可用残基名: {available}"
        )
    return idx


# ============================================================================
#  核心：构建 GBSA-OBC1 系统（仅调用一次，后续复用）
# ============================================================================

def _build_cached_gb_system(openmm_topology, initial_positions, temperature=300.0):
    """
    为一个 OpenMM Topology 构建完整的 GBSA-OBC1 隐式溶剂系统。

    此函数是整个性能优化的核心——它只在计算开始时调用一次，
    后续逐帧仅通过 simulation.context.setPositions() 更新坐标。

    技术细节：
    1. 加载 amber14-all.xml + tip3p.xml 力场
    2. 创建 CutoffNonPeriodic 系统
    3. 从 NonbondedForce 提取每个原子的真实电荷
    4. 移除 NonbondedForce，添加 GBSAOBC1Force（隐式溶剂）
    5. GB 半径根据原子元素查表映射（不依赖 vdW sigma）

    重要：不做能量最小化！
    MM-GBSA 单轨迹协议要求保留 MD 采样的真实构象。
    最小化会扭曲侧链/配体构象，导致 ΔG 偏向过负。

    Parameters
    ----------
    openmm_topology : openmm.app.Topology
        子系统的 OpenMM 拓扑对象
    initial_positions : openmm.unit.Quantity
        初始坐标（仅用于构建 Simulation Context，后续会被覆盖）
    temperature : float
        模拟温度 (K)，用于 LangevinIntegrator

    Returns
    -------
    openmm.app.Simulation
        已就绪的 Simulation 对象（Context 中已设置初始坐标）
    """
    import openmm as mm
    from openmm import app, unit
    from openmm.app.internal.customgbforces import GBSAOBC1Force

    # ---- 1. 加载力场 ----
    forcefield = app.ForceField("amber14-all.xml", "amber14/tip3p.xml")

    # ---- 2. 创建 System ----
    system = forcefield.createSystem(
        openmm_topology,
        nonbondedMethod=app.CutoffNonPeriodic,
        nonbondedCutoff=1.0 * unit.nanometer,
        constraints=app.HBonds,
    )

    # ---- 3. 从 NonbondedForce 提取真实电荷 ----
    n_particles = system.getNumParticles()
    atom_charges = [0.0] * n_particles  # 占位，确保索引安全

    forces = list(system.getForces())
    for f in forces:
        if isinstance(f, mm.NonbondedForce):
            for j in range(f.getNumParticles()):
                charge, _, _ = f.getParticleParameters(j)
                atom_charges[j] = charge.value_in_unit(unit.elementary_charge)
            system.removeForce(f)
            break

    # ---- 4. 获取原子元素信息（用于 GB 半径映射）----
    # 用户上传 PDB 可能缺少元素列，此时 atom.element 为 None，默认按碳处理
    element_numbers = [
        getattr(atom.element, "atomic_number", 6) for atom in openmm_topology.atoms()
    ]

    # 安全检查：粒子数与元素数一致
    if len(element_numbers) != len(atom_charges):
        raise RuntimeError(
            f"拓扑粒子数 ({len(element_numbers)}) 与电荷数 ({len(atom_charges)}) 不一致，"
            f"请检查输入 PDB/轨迹文件"
        )

    # ---- 5. 添加 GBSA-OBC1 隐式溶剂力场 ----
    gb_force = GBSAOBC1Force()
    for chg, an in zip(atom_charges, element_numbers):
        radius = _GB_RADIUS_MAP.get(an, _GB_RADIUS_DEFAULT)
        # scalingFactor=0.8 是 OBC1 模型标准值
        gb_force.addParticle(chg, radius, 0.8)
    system.addForce(gb_force)

    # ---- 6. 创建 Simulation Context ----
    temperature_k = temperature * unit.kelvin
    friction_coeff = 1.0 / unit.picosecond
    step_size = 0.002 * unit.picosecond

    integrator = mm.LangevinIntegrator(temperature_k, friction_coeff, step_size)
    simulation = app.Simulation(openmm_topology, system, integrator)
    simulation.context.setPositions(initial_positions)

    # NOTE: 不做 energy minimization
    # 原因见函数文档顶部注释。如需消除极端原子碰撞，
    # 可在 frame loop 中手动调用，但默认不执行。

    return simulation


def _get_frame_energy(simulation, positions) -> float:
    """
    设置坐标并直接获取势能（不最小化）。

    Parameters
    ----------
    simulation : openmm.app.Simulation
        已缓存的 Simulation 对象
    positions : openmm.unit.Quantity
        当前帧的原子坐标，形状 (n_atoms, 3) 单位 nm

    Returns
    -------
    float
        势能 (kcal/mol)
    """
    from openmm import unit

    simulation.context.setPositions(positions)
    state = simulation.context.getState(getEnergy=True)
    return state.getPotentialEnergy().value_in_unit(unit.kilocalories_per_mole)


# ============================================================================
#  主函数：从轨迹计算 MM-GBSA
# ============================================================================

def run_mmgbsa(
    topology_file: str,
    trajectory_file: str,
    ligand_resname: str = "03P",
    n_frames: int = 20,
    temperature: float = 300.0,
    progress_callback=None,
):
    """
    从 MD 轨迹计算 MM-GBSA 结合自由能。

    单轨迹协议 (Single-Trajectory Protocol):
    ΔG_bind = <G_complex> - <G_receptor> - <G_ligand>

    三部分能量均来自同一条 MD 轨迹，分别计算复合物构象、
    蛋白构象（从复合物中去除配体）和配体构象的势能。
    此方法假设蛋白在结合/未结合状态下构象变化可忽略。

    Performance:
    - 力场仅加载 3 次（三类系统各一次），而非 N_frames * 3 次
    - 无临时 PDB 文件 I/O
    - 无能量最小化
    - 20 帧计算约 10-30 秒（取决于体系大小）

    Parameters
    ----------
    topology_file : str
        拓扑文件路径（.pdb）
    trajectory_file : str
        轨迹文件路径（.dcd 或 .xtc）
    ligand_resname : str
        PDB 中配体残基名，默认 "03P"（EGFR 抑制剂常用名）
    n_frames : int
        从轨迹中均匀采样的帧数（建议 10-50）
    temperature : float
        模拟温度 (K)，默认 300K
    progress_callback : callable or None
        进度回调 (current, total, traj_index) -> bool
        返回 True 继续，False 停止

    Returns
    -------
    dict
        {
            "delta_g_mean": float,          # 平均结合自由能 (kcal/mol)
            "delta_g_std": float,           # 标准差
            "delta_g_all": list[float],     # 各帧 ΔG 列表
            "g_complex_all": list[float],   # 各帧复合物 G
            "g_receptor_all": list[float],  # 各帧受体 G
            "g_ligand_all": list[float],    # 各帧配体 G
            "n_frames": int,
            "frame_indices": list[int],
            "ligand_resname": str,
        }
    """
    import mdtraj as md
    from openmm import unit

    # ---- 1. 加载轨迹 ----
    logger.info(f"加载轨迹: {topology_file} / {trajectory_file}")
    try:
        traj = md.load(trajectory_file, top=topology_file)
    except Exception as e:
        raise RuntimeError(f"无法加载轨迹文件: {e}")

    logger.info(f"  共 {traj.n_frames} 帧，{traj.n_atoms} 个原子")

    # 空轨迹防护：避免后续 traj.xyz[0] 抛 IndexError
    if traj.n_frames == 0:
        raise ValueError("轨迹文件为空，未包含任何帧，无法计算 MM-GBSA")

    # ---- 2. 识别蛋白 & 配体原子索引 ----
    # 避免 MDTraj select() 的 ast.parse bug
    protein_indices = np.array([a.index for a in traj.topology.atoms if a.residue.is_protein], dtype=np.int64)
    ligand_indices = get_ligand_indices(traj, ligand_resname)

    logger.info(
        f"  蛋白原子: {len(protein_indices)}  配体原子: {len(ligand_indices)}"
    )

    if len(ligand_indices) == 0:
        raise ValueError(f"配体残基 '{ligand_resname}' 没有原子")
    if len(protein_indices) == 0:
        logger.warning("轨迹中未检测到蛋白原子（protein selection 为空），将使用所有非配体原子作为受体")
        all_idx = set(range(traj.n_atoms))
        lig_set = set(ligand_indices.tolist())
        protein_indices = np.array(
            sorted(all_idx - lig_set), dtype=np.int64
        )

    # ---- 3. 均匀采样帧 ----
    if traj.n_frames <= n_frames:
        frame_indices = np.arange(traj.n_frames, dtype=int)
    else:
        frame_indices = np.linspace(0, traj.n_frames - 1, n_frames, dtype=int)

    n_actual = len(frame_indices)
    logger.info(f"  将计算 {n_actual} 帧 (请求 {n_frames})")

    # ================================================================
    #  阶段 A: 一次性构建 3 个 GBSA 系统（关键性能优化！）
    #  力场只加载 3 次，而非 n_actual * 3 次
    # ================================================================
    logger.info("  [1/3] 构建复合物 GB 系统...")

    # Complex：仅蛋白+配体原子（剔除水/离子），保证与 receptor/ligand 子系统原子集一致，
    # 满足单轨迹协议的抵消要求（显式溶剂 MD 轨迹含水和离子，若不剔除会系统性偏差 ΔG_bind）
    complex_indices = np.array(
        sorted(set(protein_indices.tolist()) | set(ligand_indices.tolist())),
        dtype=np.int64,
    )
    sub_traj_complex = traj.atom_slice(complex_indices)
    omm_top_complex = sub_traj_complex.topology.to_openmm()
    first_pos = traj.xyz[0][complex_indices] * unit.nanometer
    sim_complex = _build_cached_gb_system(omm_top_complex, first_pos, temperature)

    logger.info(f"  [2/3] 构建受体 GB 系统 ({len(protein_indices)} 原子)...")

    # Receptor：仅蛋白原子
    rec_indices = np.asarray(protein_indices, dtype=int)
    sub_traj_rec = traj.atom_slice(rec_indices)
    omm_top_receptor = sub_traj_rec.topology.to_openmm()
    rec_first_pos = traj.xyz[0][rec_indices] * unit.nanometer
    sim_receptor = _build_cached_gb_system(omm_top_receptor, rec_first_pos, temperature)

    logger.info(f"  [3/3] 构建配体 GB 系统 ({len(ligand_indices)} 原子)...")

    # Ligand：仅配体原子
    lig_indices_arr = np.asarray(ligand_indices, dtype=int)
    sub_traj_lig = traj.atom_slice(lig_indices_arr)
    omm_top_ligand = sub_traj_lig.topology.to_openmm()
    lig_first_pos = traj.xyz[0][lig_indices_arr] * unit.nanometer
    sim_ligand = _build_cached_gb_system(omm_top_ligand, lig_first_pos, temperature)

    # ================================================================
    #  阶段 B: 逐帧更新坐标计算能量（仅 setPositions + getState）
    # ================================================================
    logger.info(f"  开始逐帧能量计算 ({n_actual} 帧)...")

    delta_g_list = []
    g_complex_list = []
    g_receptor_list = []
    g_ligand_list = []

    for step, fi in enumerate(frame_indices):
        # --- Complex ---
        pos_full = traj.xyz[fi][complex_indices] * unit.nanometer
        g_c = _get_frame_energy(sim_complex, pos_full)

        # --- Receptor ---
        pos_rec = traj.xyz[fi][rec_indices] * unit.nanometer
        g_r = _get_frame_energy(sim_receptor, pos_rec)

        # --- Ligand ---
        pos_lig = traj.xyz[fi][lig_indices_arr] * unit.nanometer
        g_l = _get_frame_energy(sim_ligand, pos_lig)

        dg = g_c - g_r - g_l

        delta_g_list.append(dg)
        g_complex_list.append(g_c)
        g_receptor_list.append(g_r)
        g_ligand_list.append(g_l)

        logger.debug(
            f"    帧 {step + 1}/{n_actual}: "
            f"G_complex={g_c:.1f}  G_rec={g_r:.1f}  G_lig={g_l:.1f}  "
            f"ΔG={dg:.2f} kcal/mol"
        )

        # 进度回调（支持取消）
        if progress_callback:
            if not progress_callback(step + 1, n_actual, int(fi)):
                logger.info("  用户取消计算")
                break

    # ---- 统计分析 ----
    if not delta_g_list:
        raise ValueError("未计算任何帧（可能被用户取消），无法统计 ΔG")
    delta_g_arr = np.array(delta_g_list)
    delta_g_mean = float(np.mean(delta_g_arr))
    delta_g_std = float(np.std(delta_g_arr))

    logger.info(
        f"  ✓ ΔG = {delta_g_mean:.2f} ± {delta_g_std:.2f} kcal/mol  "
        f"(n={len(delta_g_list)})"
    )

    return {
        "delta_g_mean": delta_g_mean,
        "delta_g_std": delta_g_std,
        "delta_g_all": [float(v) for v in delta_g_list],
        "g_complex_all": [float(v) for v in g_complex_list],
        "g_receptor_all": [float(v) for v in g_receptor_list],
        "g_ligand_all": [float(v) for v in g_ligand_list],
        "n_frames": len(delta_g_list),
        "frame_indices": [int(f) for f in frame_indices[:len(delta_g_list)]],
        "ligand_resname": ligand_resname,
    }
