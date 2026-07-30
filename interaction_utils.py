# interaction_utils.py
"""
蛋白-配体相互作用 & 激酶结合模式相似性 工具模块

功能：
1. 通用蛋白-配体相互作用分析 (基于 PLIP + NGLView)
2. 激酶 IFP 指纹相似性分析 (基于 opencadd + KLIFS)
"""
import os
import tempfile
import urllib.request
import logging
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------- 懒加载标记 ----------
_PLIP_AVAILABLE = False
_NGLVIEW_AVAILABLE = False
_OPENCADD_AVAILABLE = False
_SEABORN_AVAILABLE = False
_PLIP_IMPORT_ERROR = None
_NGLVIEW_IMPORT_ERROR = None
_OPENCADD_IMPORT_ERROR = None
_SEABORN_IMPORT_ERROR = None


def _ensure_plip():
    """延迟导入 PLIP，避免阻塞非 PLIP 页面"""
    global _PLIP_AVAILABLE, _PLIP_IMPORT_ERROR
    if not _PLIP_AVAILABLE and _PLIP_IMPORT_ERROR is None:
        try:
            from plip.structure.preparation import PDBComplex  # noqa: F401
            from plip.exchange.report import BindingSiteReport  # noqa: F401
            _PLIP_AVAILABLE = True
        except ImportError as e:
            _PLIP_IMPORT_ERROR = str(e)
            logger.warning(f"PLIP 导入失败: {e}")
    return _PLIP_AVAILABLE


def _ensure_nglview():
    """延迟导入 NGLView"""
    global _NGLVIEW_AVAILABLE, _NGLVIEW_IMPORT_ERROR
    if not _NGLVIEW_AVAILABLE and _NGLVIEW_IMPORT_ERROR is None:
        try:
            import nglview as nv  # noqa: F401
            _NGLVIEW_AVAILABLE = True
        except ImportError as e:
            _NGLVIEW_IMPORT_ERROR = str(e)
            logger.warning(f"NGLView 导入失败: {e}")
    return _NGLVIEW_AVAILABLE


def _ensure_opencadd():
    """延迟导入 opencadd"""
    global _OPENCADD_AVAILABLE, _OPENCADD_IMPORT_ERROR
    if not _OPENCADD_AVAILABLE and _OPENCADD_IMPORT_ERROR is None:
        try:
            from opencadd.databases.klifs import setup_remote  # noqa: F401
            _OPENCADD_AVAILABLE = True
        except ImportError as e:
            _OPENCADD_IMPORT_ERROR = str(e)
            logger.warning(f"opencadd 导入失败: {e}")
    return _OPENCADD_AVAILABLE


def _ensure_seaborn():
    """延迟导入 seaborn"""
    global _SEABORN_AVAILABLE, _SEABORN_IMPORT_ERROR
    if not _SEABORN_AVAILABLE and _SEABORN_IMPORT_ERROR is None:
        try:
            import seaborn as sns  # noqa: F401
            _SEABORN_AVAILABLE = True
        except ImportError as e:
            _SEABORN_IMPORT_ERROR = str(e)
            logger.warning(f"seaborn 导入失败: {e}")
    return _SEABORN_AVAILABLE


# ========== 模块1：通用蛋白-配体相互作用 (PLIP) ==========

def analyze_plip(pdb_id=None, pdb_content=None):
    """
    输入 PDB ID 或 PDB 文件内容，返回相互作用分析结果。

    返回:
        interactions_df : pd.DataFrame  相互作用表格
        html_str        : str           NGLView 3D 可视化 HTML
        pdb_path        : str           本地临时 PDB 文件路径
    """
    if not _ensure_plip():
        raise ImportError(
            f"PLIP 未安装或导入失败。请运行: pip install plip\n错误详情: {_PLIP_IMPORT_ERROR}"
        )
    if not _ensure_nglview():
        raise ImportError(
            f"NGLView 未安装或导入失败。请运行: pip install nglview\n错误详情: {_NGLVIEW_IMPORT_ERROR}"
        )

    from plip.structure.preparation import PDBComplex
    from plip.exchange.report import BindingSiteReport
    import nglview as nv

    # 1. 获取 PDB 文件
    if pdb_id:
        pdb_dir = tempfile.gettempdir()
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            urllib.request.urlretrieve(pdb_url, pdb_path)
        logger.info(f"PDB 文件已下载: {pdb_path}")
    elif pdb_content:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdb', mode='w') as tmp:
            if isinstance(pdb_content, bytes):
                pdb_content = pdb_content.decode('utf-8')
            tmp.write(pdb_content)
            pdb_path = tmp.name
    else:
        raise ValueError("必须提供 pdb_id 或 pdb_content")

    # 2. PLIP 分析
    complex_obj = PDBComplex()
    complex_obj.load_pdb(pdb_path)
    complex_obj.analyze()
    logger.info(f"PLIP 分析完成，发现 {len(complex_obj.interaction_sets)} 个结合位点")

    # 3. 提取所有相互作用
    rows = []
    for site_id, site in complex_obj.interaction_sets.items():
        report = BindingSiteReport(site)
        # 氢键
        for h in report.hbonds_pairs:
            rows.append({
                "结合位点": site_id, "类型": "氢键",
                "蛋白残基": h[0], "配体原子": h[1]
            })
        # 疏水作用
        for h in report.hydrophobic_pairs:
            rows.append({
                "结合位点": site_id, "类型": "疏水作用",
                "蛋白残基": h[0], "配体原子": h[1]
            })
        # 盐桥
        for s in report.saltbridge_pairs:
            rows.append({
                "结合位点": site_id, "类型": "盐桥",
                "蛋白残基": s[0], "配体原子": s[1]
            })
        # pi-pi 堆积
        for p in report.pistacking_pairs:
            rows.append({
                "结合位点": site_id, "类型": "π-π堆积",
                "蛋白残基": p[0], "配体原子": p[1]
            })
        # 卤键
        for x in report.halogen_pairs:
            rows.append({
                "结合位点": site_id, "类型": "卤键",
                "蛋白残基": x[0], "配体原子": x[1]
            })

    df = pd.DataFrame(rows)
    logger.info(f"共提取 {len(df)} 条相互作用记录")

    # 4. 生成 NGLView 3D 可视化
    view = nv.show_file(pdb_path)
    view.add_representation('cartoon', selection='protein', color='sstruc')
    view.add_representation('licorice', selection='ligand')
    view.add_representation('ball+stick', selection='ligand')
    html_str = view._repr_html_()

    return df, html_str, pdb_path


# ========== 模块2：激酶 IFP 指纹相似性 (KLIFS) ==========

@st.cache_data(ttl=86400)  # 缓存 24 小时
def fetch_klifs_ifps(kinase_names):
    """
    从 KLIFS 数据库获取激酶的相互作用指纹 (IFP)。

    参数:
        kinase_names : list[str]  激酶名称列表

    返回:
        pd.DataFrame  包含 IFP 和结构信息的完整数据框
    """
    if not _ensure_opencadd():
        raise ImportError(
            f"opencadd 未安装或导入失败。请运行: pip install opencadd\n错误详情: {_OPENCADD_IMPORT_ERROR}"
        )

    from opencadd.databases.klifs import setup_remote

    session = setup_remote()

    # 获取结构
    structures = session.structures.by_kinase_name(kinase_names=kinase_names)
    logger.info(f"从 KLIFS 获取到 {len(structures)} 个原始结构")

    # 过滤高质量结构（DFG-in, 分辨率 ≤ 3.0 Å, 质量分 ≥ 6, 人源）
    structures = structures[
        (structures["species.klifs"] == "Human") &
        (structures["structure.dfg"] == "in") &
        (structures["structure.resolution"] <= 3.0) &
        (structures["structure.qualityscore"] >= 6)
    ]
    logger.info(f"过滤后保留 {len(structures)} 个高质量结构")

    if structures.empty:
        return pd.DataFrame()

    # 获取 IFP
    structure_ids = structures["structure.klifs_id"].tolist()
    ifps = session.interactions.by_structure_klifs_id(structure_ids)

    # 合并结构与 IFP
    result = ifps.merge(structures, on="structure.klifs_id", how="inner")
    logger.info(f"IFP 合并完成，共 {len(result)} 条记录")
    return result


def compute_ifp_distance_matrix(ifp_df):
    """
    根据 IFP 数据框计算 Jaccard 距离矩阵。

    返回:
        dist_matrix : np.ndarray   距离矩阵
        labels      : list[str]    标签列表
    """
    # 提取 IFP 字符串 (0/1 字符串)
    ifp_col = None
    for col in ["interaction.fingerprint", "ifp"]:
        if col in ifp_df.columns:
            ifp_col = col
            break

    if ifp_col is None:
        raise ValueError("IFP 数据框中未找到指纹列 (interaction.fingerprint / ifp)")

    # 构建唯一标识
    id_cols = [c for c in ["structure.klifs_id", "kinase.klifs_name"] if c in ifp_df.columns]
    ifp_series = ifp_df.set_index(id_cols)[ifp_col]

    # 转成布尔矩阵
    ifp_matrix = np.array([
        [c == '1' for c in str(fp_str)] for fp_str in ifp_series.values
    ], dtype=bool)

    # Jaccard 距离
    dist_matrix = pairwise_distances(ifp_matrix, metric='jaccard')

    # 标签
    labels = [str(idx) for idx in ifp_series.index]

    return dist_matrix, labels


def plot_ifp_heatmap(dist_matrix, labels):
    """
    绘制 IFP 距离矩阵热图。

    返回:
        fig : matplotlib.figure.Figure
    """
    if not _ensure_seaborn():
        raise ImportError(
            f"seaborn 未安装。请运行: pip install seaborn\n错误详情: {_SEABORN_IMPORT_ERROR}"
        )
    import seaborn as sns

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(
        dist_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis_r',
        annot=True,
        fmt='.2f',
        ax=ax,
        linewidths=0.5
    )
    ax.set_title("激酶结合模式相似性 (Jaccard距离, 越小越相似)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
