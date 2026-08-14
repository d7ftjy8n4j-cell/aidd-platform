# interaction_utils.py
"""
蛋白-配体相互作用 & 激酶结合模式相似性 工具模块

功能：
1. 通用蛋白-配体相互作用分析 (基于 PLIP + NGLView)
2. 激酶 IFP 指纹相似性分析 (基于 KLIFS REST API)
"""
import os
import re
import tempfile
import urllib.request
import logging
import numpy as np
import pandas as pd
import requests
import streamlit as st
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# ---------- 懒加载标记 ----------
_PLIP_AVAILABLE = False
_PY3DMOL_AVAILABLE = False
_SEABORN_AVAILABLE = False
_PLIP_IMPORT_ERROR = None
_PY3DMOL_IMPORT_ERROR = None
_SEABORN_IMPORT_ERROR = None


def _ensure_plip():
    """延迟导入 PLIP，避免阻塞非 PLIP 页面"""
    global _PLIP_AVAILABLE, _PLIP_IMPORT_ERROR
    if not _PLIP_AVAILABLE and _PLIP_IMPORT_ERROR is None:
        try:
            from plip.structure.preparation import PDBComplex  # noqa: F401
            _PLIP_AVAILABLE = True
        except ImportError as e:
            _PLIP_IMPORT_ERROR = str(e)
            logger.warning(f"PLIP 导入失败: {e}")
    return _PLIP_AVAILABLE


def _ensure_py3dmol():
    """延迟导入 py3Dmol"""
    global _PY3DMOL_AVAILABLE, _PY3DMOL_IMPORT_ERROR
    if not _PY3DMOL_AVAILABLE and _PY3DMOL_IMPORT_ERROR is None:
        try:
            import py3Dmol  # noqa: F401
            _PY3DMOL_AVAILABLE = True
        except ImportError as e:
            _PY3DMOL_IMPORT_ERROR = str(e)
            logger.warning(f"py3Dmol 导入失败: {e}")
    return _PY3DMOL_AVAILABLE


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
    if not _ensure_py3dmol():
        raise ImportError(
            f"py3Dmol 未安装或导入失败。请运行: pip install py3Dmol\n错误详情: {_PY3DMOL_IMPORT_ERROR}"
        )

    from plip.structure.preparation import PDBComplex
    import py3Dmol

    # 1. 获取 PDB 文件
    if pdb_id:
        # 严格校验 PDB ID，防止路径遍历
        if not re.fullmatch(r"[0-9][A-Za-z0-9]{3}", pdb_id):
            raise ValueError(f"无效的 PDB ID: {pdb_id!r}（应为 4 位字母数字，如 3POZ）")
        pdb_dir = tempfile.gettempdir()
        pdb_path = os.path.join(pdb_dir, f"{pdb_id}.pdb")
        if not os.path.exists(pdb_path):
            pdb_url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
            resp = requests.get(pdb_url, timeout=30)
            resp.raise_for_status()
            if not resp.text.lstrip().startswith(("HEADER", "ATOM", "REMARK", "CRYST1", "MODEL", "TITLE")):
                raise ValueError(f"PDB ID {pdb_id} 无效或返回内容不是 PDB 数据")
            with open(pdb_path, "w", encoding="utf-8") as f:
                f.write(resp.text)
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

    # 3. 提取所有相互作用 (PLIP 3.0 API)
    rows = []
    for site_id, site in complex_obj.interaction_sets.items():
        # 氢键（蛋白供体 + 配体供体）
        for h in (site.hbonds_pdon or []) + (site.hbonds_ldon or []):
            rows.append({
                "结合位点": site_id, "类型": "氢键",
                "蛋白残基": f"{h.restype}{h.resnr}{h.reschain}",
                "配体原子": f"{getattr(h, 'restype_l', '')}{getattr(h, 'resnr_l', '')}"
            })
        # 疏水作用
        for h in (site.hydrophobic_contacts or []):
            rows.append({
                "结合位点": site_id, "类型": "疏水作用",
                "蛋白残基": f"{h.restype}{h.resnr}{h.reschain}",
                "配体原子": f"{getattr(h, 'restype_l', '')}{getattr(h, 'resnr_l', '')}"
            })
        # 盐桥
        for s in (site.saltbridge_lneg or []) + (site.saltbridge_pneg or []):
            rows.append({
                "结合位点": site_id, "类型": "盐桥",
                "蛋白残基": f"{s.restype}{s.resnr}{s.reschain}",
                "配体原子": f"{getattr(s, 'restype_l', '')}{getattr(s, 'resnr_l', '')}"
            })
        # pi-pi 堆积
        for p in (site.pistacking or []):
            rows.append({
                "结合位点": site_id, "类型": "π-π堆积",
                "蛋白残基": f"{p.restype}{p.resnr}{p.reschain}",
                "配体原子": f"{getattr(p, 'restype_l', '')}{getattr(p, 'resnr_l', '')}"
            })
        # 卤键
        for x in (site.halogen_bonds or []):
            rows.append({
                "结合位点": site_id, "类型": "卤键",
                "蛋白残基": f"{x.restype}{x.resnr}{x.reschain}",
                "配体原子": f"{getattr(x, 'restype_l', '')}{getattr(x, 'resnr_l', '')}"
            })

    df = pd.DataFrame(rows)
    logger.info(f"共提取 {len(df)} 条相互作用记录")

    # 4. 生成 py3Dmol 3D 可视化
    with open(pdb_path, 'r') as f:
        pdb_str = f.read()
    view = py3Dmol.view(width=800, height=550)
    view.addModel(pdb_str, 'pdb')
    view.setStyle({'model': -1}, {'cartoon': {'color': 'spectrum'}})
    view.setStyle({'hetflag': True}, {'stick': {'radius': 0.3}})
    view.zoomTo()
    html_str = view._make_html()

    return df, html_str, pdb_path


# ========== 模块2：激酶 IFP 指纹相似性 (KLIFS REST API) ==========

# ---- KLIFS API 常量 ----
KLIFS_BASE_URL = "https://klifs.vu-compmedchem.nl/api/v2"


def _klifs_api_get(endpoint: str, params: dict = None):
    """
    统一的 KLIFS API 调用封装，带缓存。
    请求失败时抛出异常（st.cache_data 默认不缓存异常），
    避免临时故障被缓存 24 小时；调用方负责捕获并降级。
    """
    url = f"{KLIFS_BASE_URL}/{endpoint.lstrip('/')}"
    resp = requests.get(url, params=params, timeout=30)
    resp.raise_for_status()
    return resp.json()


def _fetch_kinase_id(kinase_name: str) -> int | None:
    """
    通过激酶名称查找 KLIFS 内部 ID。
    KLIFS API: GET /kinases?kinase_name={name}
    """
    try:
        kinases = _klifs_api_get("kinases", {"kinase_name": kinase_name})
    except requests.RequestException as e:
        logger.error(f"KLIFS 激酶查询失败 [{kinase_name}]: {e}")
        return None
    if kinases and isinstance(kinases, list) and len(kinases) > 0:
        return kinases[0].get("kinase_ID")
    return None


def _fetch_structures_for_kinase_id(kinase_id: int) -> list:
    """
    获取指定激酶的高质量结构列表。
    KLIFS API: GET /structures?kinase_ID={id}
    过滤条件：人源、分辨率 ≤ 3.0 Å、质量分 ≥ 6、DFG-in
    """
    try:
        structures = _klifs_api_get("structures", {"kinase_ID": kinase_id})
    except requests.RequestException as e:
        logger.error(f"KLIFS 结构查询失败 [kinase_ID={kinase_id}]: {e}")
        return []
    if not structures or not isinstance(structures, list):
        return []

    filtered = []
    for s in structures:
        res = s.get("resolution")
        qs = s.get("quality_score")
        # KLIFS v2 API 对 NMR/预测结构可能返回 null，需显式判空
        if (s.get("species") == "Human"
                and res is not None and float(res) <= 3.0
                and qs is not None and float(qs) >= 6
                and s.get("DFG") == "in"):
            filtered.append(s)
    return filtered


def _fetch_ifp_for_structure(structure_id: int) -> str | None:
    """
    获取某个结构的相互作用指纹 (IFP)。
    KLIFS API: GET /interactions/structure?structure_ID={id}
    返回 85 位的 0/1 字符串。
    """
    try:
        data = _klifs_api_get("interactions/structure", {"structure_ID": structure_id})
    except requests.RequestException as e:
        logger.error(f"KLIFS IFP 查询失败 [structure_ID={structure_id}]: {e}")
        return None
    if data:
        # KLIFS API 返回单条记录（dict）而非列表
        if isinstance(data, dict):
            return data.get("fingerprint", None)
        elif isinstance(data, list) and len(data) > 0:
            return data[0].get("fingerprint", None)
    return None


@st.cache_data(ttl=86400)  # 缓存 24 小时
def fetch_klifs_ifps(kinase_names):
    """
    使用 KLIFS REST API 获取激酶的相互作用指纹 (IFP)。

    参数:
        kinase_names : list[str]  激酶名称列表（如 ["EGFR", "ErbB2"]）

    返回:
        pd.DataFrame  包含 IFP 和结构信息的数据框
                      列: structure_klifs_id, kinase_name, pdb_id, resolution, interaction_fingerprint
    """
    all_rows = []

    for kinase_name in kinase_names:
        # Step 1: 查找激酶 ID
        kinase_id = _fetch_kinase_id(kinase_name)
        if kinase_id is None:
            logger.warning(f"在 KLIFS 中未找到激酶: {kinase_name}")
            continue

        # Step 2: 获取结构列表
        structures = _fetch_structures_for_kinase_id(kinase_id)
        logger.info(f"[{kinase_name}] 找到 {len(structures)} 个高质量结构")

        # Step 3: 逐个获取 IFP
        for s in structures:
            sid = s.get("structure_ID")
            if not sid:
                continue
            ifp_str = _fetch_ifp_for_structure(sid)
            if ifp_str:
                all_rows.append({
                    "structure.klifs_id": sid,
                    "kinase.klifs_name": kinase_name,
                    "pdb_id": s.get("pdb", ""),
                    "resolution": s.get("resolution", None),
                    "interaction.fingerprint": ifp_str,
                })

    if not all_rows:
        logger.warning("未获取到任何 IFP 数据")
        return pd.DataFrame()

    result = pd.DataFrame(all_rows)
    logger.info(f"共获取 {len(result)} 条 IFP 记录，覆盖 {result['kinase.klifs_name'].nunique()} 个激酶")
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
