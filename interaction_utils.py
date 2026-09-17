# interaction_utils.py
"""
蛋白-配体相互作用 & 激酶结合模式相似性 工具模块

功能：
1. 通用蛋白-配体相互作用分析 (基于 PLIP + NGLView)
2. 激酶 IFP 指纹相似性分析 (基于 KLIFS REST API)
"""
import os
import re
import time
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
# KLIFS 已从 klifs.vu-compmedchem.nl 迁移到 klifs.net，并把 REST 端点从 v1 风格
# 改成了函数式路径（官方 Swagger：https://klifs.net/swagger_v2/swagger.json，basePath=/api_v2）：
#     kinases                -> kinase_ID
#     structures             -> structures_list
#     interactions/structure -> interactions_get_IFP
# 而且 IFP 的返回字段名从 fingerprint 改成了 IFP。旧域名只会 301 到 klifs.net，
# 旧路径在新站返回 400 ["KLIFS error: No correct function calls were specified."]，
# 页面因此表现为"未获取到任何结构数据"（网络本身是通的）。
KLIFS_BASE_URL = "https://klifs.net/api_v2"

#: 每个激酶最多取多少个高质量结构（热门靶点有几百个结构，逐个取 IFP 会把页面拖死）
KLIFS_MAX_STRUCTURES_PER_KINASE = 100

#: 批量取 IFP 时每批的结构数（KLIFS 支持 structure_ID 逗号分隔）
KLIFS_IFP_BATCH_SIZE = 50


def _klifs_api_get(endpoint: str, params: dict = None, max_attempts: int = 3):
    """
    统一的 KLIFS API 调用封装。

    * 4xx（端点/参数写错）**不重试**，直接把 KLIFS 的响应体带进异常，
      便于一眼看出"契约又变了"（例如 400 ["KLIFS error: No correct function calls were specified."]）；
    * 连接被重置 / 超时 / 5xx 属于瞬时报错，最多重试 max_attempts 次（实测 KLIFS
      在连续请求后会 Reset 连接，不重试会导致整个激酶被静默跳过）。

    请求最终失败时抛出异常（st.cache_data 默认不缓存异常），避免故障被缓存 24 小时。
    """
    url = f"{KLIFS_BASE_URL}/{endpoint.lstrip('/')}"
    last_error = None
    for attempt in range(1, max_attempts + 1):
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code >= 400:
                raise requests.HTTPError(
                    f"KLIFS {resp.status_code} @ {resp.url} :: {resp.text[:200]}",
                    response=resp,
                )
            return resp.json()
        except requests.HTTPError as exc:
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status is not None and 400 <= status < 500:
                raise  # 契约/参数问题：重试没有意义
            last_error = exc
        except (requests.ConnectionError, requests.Timeout) as exc:
            last_error = exc

        if attempt < max_attempts:
            logger.warning(
                "KLIFS 请求失败（第 %s/%s 次，%s）：%s",
                attempt, max_attempts, endpoint, last_error,
            )
            time.sleep(1.5 * attempt)

    raise last_error if last_error is not None else RuntimeError("KLIFS 请求失败")


def _to_float(value, default: float) -> float:
    """KLIFS 的数值字段常以字符串返回（"1.7" / "8"），统一安全转 float。"""
    try:
        if value is None or value == "":
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _fetch_kinase_id(kinase_name: str) -> int | None:
    """
    通过激酶名称查找 KLIFS 内部 ID。
    KLIFS API: GET /kinase_ID?kinase_name={name}&species=Human
    """
    try:
        kinases = _klifs_api_get(
            "kinase_ID", {"kinase_name": kinase_name, "species": "Human"}
        )
    except requests.RequestException as e:
        logger.error(f"KLIFS 激酶查询失败 [{kinase_name}]: {e}")
        return None
    if isinstance(kinases, dict):  # 防御：某些版本可能返回单条 dict
        kinases = [kinases]
    if isinstance(kinases, list):
        for item in kinases:
            if isinstance(item, dict) and item.get("kinase_ID"):
                return item.get("kinase_ID")
    return None


def _fetch_structures_for_kinase_id(kinase_id: int) -> list:
    """
    获取指定激酶的高质量结构列表。
    KLIFS API: GET /structures_list?kinase_ID={id}
    过滤条件：人源、分辨率 ≤ 3.0 Å、质量分 ≥ 6、DFG-in
    """
    try:
        structures = _klifs_api_get("structures_list", {"kinase_ID": kinase_id})
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


def _fetch_ifps_batch(structure_ids) -> dict:
    """
    批量获取多个结构的相互作用指纹 (IFP)。

    KLIFS API: GET /interactions_get_IFP?structure_ID=1,2,3
    （实测支持逗号分隔的 ID 列表，可用一批请求取代几十次单条请求）

    返回:
        {structure_ID(int): "0101..."}

    注意：v2 的字段名是 **IFP**（旧版叫 fingerprint）；指纹长度随 KLIFS 版本变化
    （当前 595 位），所以下游只按"0/1 字符串"处理，不写死长度。
    """
    ids = []
    for sid in structure_ids:
        try:
            ids.append(int(sid))
        except (TypeError, ValueError):
            continue

    result = {}
    for start in range(0, len(ids), KLIFS_IFP_BATCH_SIZE):
        chunk = ids[start:start + KLIFS_IFP_BATCH_SIZE]
        try:
            data = _klifs_api_get(
                "interactions_get_IFP",
                {"structure_ID": ",".join(str(i) for i in chunk)},
            )
        except requests.RequestException as e:
            logger.error(f"KLIFS IFP 查询失败 [structure_IDs={chunk[:5]}...]: {e}")
            continue
        if isinstance(data, dict):
            data = [data]
        for item in data or []:
            if not isinstance(item, dict):
                continue
            sid = item.get("structure_ID")
            fingerprint = item.get("IFP") or item.get("fingerprint")
            if sid is not None and fingerprint:
                result[int(sid)] = fingerprint
    return result


def _fetch_ifp_for_structure(structure_id: int) -> str | None:
    """获取单个结构的 IFP（内部走批量接口）。"""
    return _fetch_ifps_batch([structure_id]).get(int(structure_id))


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
        # Step 1: 查找激酶 ID（KLIFS v2: /kinase_ID?kinase_name=...&species=Human）
        kinase_id = _fetch_kinase_id(kinase_name)
        if kinase_id is None:
            logger.warning(f"在 KLIFS 中未找到激酶: {kinase_name}")
            continue

        # Step 2: 获取结构列表（KLIFS v2: /structures_list?kinase_ID=...）
        structures = _fetch_structures_for_kinase_id(kinase_id)
        logger.info(f"[{kinase_name}] 找到 {len(structures)} 个高质量结构")

        # 结构太多时按"质量分高、分辨率低"优先，并限制数量：
        # EGFR 这类热门靶点有几百个合格结构，全量取 IFP 会让页面等好几分钟。
        structures = sorted(
            structures,
            key=lambda s: (
                -_to_float(s.get("quality_score"), 0.0),
                _to_float(s.get("resolution"), 99.0),
            ),
        )[:KLIFS_MAX_STRUCTURES_PER_KINASE]
        logger.info(f"[{kinase_name}] 取前 {len(structures)} 个结构参与 IFP 比对")

        # Step 3: 批量获取 IFP（KLIFS v2: /interactions_get_IFP?structure_ID=1,2,3）
        ifps = _fetch_ifps_batch([s.get("structure_ID") for s in structures])
        for s in structures:
            try:
                sid = int(s.get("structure_ID"))
            except (TypeError, ValueError):
                continue
            ifp_str = ifps.get(sid)
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


def compute_kinase_distance_matrix(ifp_df):
    """
    把结构级 IFP 聚合成 **激酶级** 距离矩阵。

    页面要回答的问题是"哪些激酶的结合模式彼此相似"（选择性/脱靶风险），
    所以矩阵元素取两个激酶全部高质量结构对的平均 Jaccard 距离：
        dist[i][j] = mean( Jaccard(IFF_i 的每个结构, IFP_j 的每个结构) )，对角为 0

    为什么不直接用 compute_ifp_distance_matrix：那个是**逐结构**的，
    4 个激酶就能产生两三百行 → 热图既画不动也读不懂。

    返回:
        dist_matrix : np.ndarray   (n_kinase, n_kinase)
        labels      : list[str]    激酶名（按字母序）
    """
    ifp_col = None
    for col in ["interaction.fingerprint", "ifp"]:
        if col in ifp_df.columns:
            ifp_col = col
            break
    if ifp_col is None:
        raise ValueError("IFP 数据框中未找到指纹列 (interaction.fingerprint / ifp)")

    kinase_col = "kinase.klifs_name"
    if kinase_col not in ifp_df.columns:
        # 没有激酶列时退回逐结构矩阵，避免页面直接崩掉
        return compute_ifp_distance_matrix(ifp_df)

    groups = {}
    for kinase, sub in ifp_df.groupby(kinase_col):
        groups[str(kinase)] = np.array(
            [[c == "1" for c in str(fp)] for fp in sub[ifp_col]], dtype=bool
        )

    labels = sorted(groups)
    n = len(labels)
    dist_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            a, b = groups[labels[i]], groups[labels[j]]
            if a.shape[1] != b.shape[1]:
                # 指纹长度不一致（极少见，例如不同 KLIFS 版本的数据混在一起）
                # 截断到公共长度再比，避免 jaccard 直接抛 ValueError
                width = min(a.shape[1], b.shape[1])
                a, b = a[:, :width], b[:, :width]
            distance = float(pairwise_distances(a, b, metric="jaccard").mean())
            dist_matrix[i, j] = distance
            dist_matrix[j, i] = distance
    return dist_matrix, labels


def _matplotlib_cjk_font() -> str | None:
    """为 matplotlib 找一个能显示中文的字体名；找不到返回 None。

    背景：matplotlib 默认字体不含 CJK，中文标题会渲染成"豆腐块"并刷一堆
    "Glyph xxxx missing from current font" 警告。
    """
    try:
        from matplotlib import font_manager
    except Exception:
        return None
    candidates = [
        "Microsoft YaHei",   # Windows
        "SimHei",            # Windows
        "PingFang SC",       # macOS
        "Noto Sans CJK SC",  # Linux
        "Source Han Sans SC",
        "WenQuanYi Micro Hei",
        "Arial Unicode MS",
    ]
    try:
        available = {font.name for font in font_manager.fontManager.ttflist}
    except Exception:
        return None
    for name in candidates:
        if name in available:
            return name
    return None


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

    # 仅在小矩阵上标数字：几十行以上的矩阵标注会拖慢渲染且根本看不清
    annotate = len(labels) <= 15
    fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(labels) + 3), max(5.0, 0.8 * len(labels) + 3)))
    sns.heatmap(
        dist_matrix,
        xticklabels=labels,
        yticklabels=labels,
        cmap='viridis_r',
        annot=annotate,
        fmt='.2f' if annotate else '',
        ax=ax,
        linewidths=0.5
    )
    cjk_font = _matplotlib_cjk_font()
    if cjk_font:
        plt.rcParams["font.sans-serif"] = [cjk_font] + list(
            plt.rcParams.get("font.sans-serif", [])
        )
        plt.rcParams["axes.unicode_minus"] = False
        title = "激酶结合模式相似性（Jaccard 距离，越小越相似）"
    else:
        # 系统没有中文字体时退回英文标题，总比一片豆腐块好
        title = "Kinase binding-mode similarity (Jaccard distance, lower = more similar)"
    ax.set_title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    return fig
