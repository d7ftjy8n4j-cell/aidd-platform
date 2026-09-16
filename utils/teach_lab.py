# utils/teach_lab.py
"""
🎓 教学实验室（Teach Lab）核心逻辑
=================================

定位：这是**教学**模块，不是科研模块。核心目标是「让用户看清答案是怎么来的」——
把药物发现的完整流程拆成四步闭环，每一步都可见、可解释：

    步骤 1  从 ChEMBL 下载真实活性数据（真实数据，绝不模拟）
    步骤 2  清洗数据（每一步的数据流失都记录为 DataStage，便于可视化）
    步骤 3  现场训练两个小模型：随机森林（秒级）+ 图神经网络 GNN（分钟级）
    步骤 4  用刚训练好的模型预测新分子

与项目已有训练脚本/预测器的硬约束（改错会直接报错，勿动）：
    * GNN 原子特征固定 **13 维**（匹配 gcn_egfr_best_model.pth 中 conv1.lin.weight 形状 [128, 13]）
    * GNN 层名固定 **conv1/conv2/conv3 与 bn1/bn2**，不用 ModuleList（否则 state_dict 键名不匹配）
    * 计时统一用 ``time.perf_counter()``，**不用** ``torch.cuda.Event``
    * ChEMBL 靶点查询必须用 ``search()`` 端点（get()/filter() 对靶点名检索不稳定）
    * torch / torch_geometric 全部延迟导入，缺失时 ``TORCH_AVAILABLE=False``，
      不会影响随机森林流程

作者：dadamingli
"""

import itertools
import logging
import math
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# ============================================================================
# 可选依赖：RDKit / ChEMBL / PyTorch（全部延迟或防御式导入）
# ============================================================================
try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, rdMolDescriptors

    RDKIT_AVAILABLE = True
    RDKIT_IMPORT_ERROR: Optional[str] = None
except Exception as _rdkit_err:  # pragma: no cover - 取决于环境
    RDKIT_AVAILABLE = False
    RDKIT_IMPORT_ERROR = str(_rdkit_err)
    logger.error("RDKit 不可用，教学实验室的分子解析功能将失效: %s", _rdkit_err)

try:
    from chembl_webresource_client.new_client import new_client
    from chembl_webresource_client.settings import Settings

    CHEMBL_AVAILABLE = True
    CHEMBL_IMPORT_ERROR: Optional[str] = None
except Exception as _chembl_err:  # pragma: no cover - 取决于环境
    CHEMBL_AVAILABLE = False
    CHEMBL_IMPORT_ERROR = str(_chembl_err)
    logger.error("chembl_webresource_client 不可用: %s", _chembl_err)

# ---- PyTorch / PyTorch Geometric：延迟导入（GNN 不可用时 RF 流程照常工作）----
try:
    import torch
    import torch.nn.functional as F
    from torch.nn import BatchNorm1d, Linear
    from torch_geometric.data import Data
    from torch_geometric.nn import GCNConv, global_mean_pool

    try:
        from torch_geometric.loader import DataLoader  # PyG >= 2.0
    except Exception:  # pragma: no cover - 兼容旧版 PyG
        from torch_geometric.data import DataLoader  # type: ignore[no-redef]

    TORCH_AVAILABLE = True
    TORCH_IMPORT_ERROR: Optional[str] = None
except Exception as _torch_err:  # pragma: no cover - 取决于环境
    TORCH_AVAILABLE = False
    TORCH_IMPORT_ERROR = str(_torch_err)
    logger.warning("PyTorch/PyG 不可用，GNN 部分将自动禁用: %s", _torch_err)


# ============================================================================
# 常量
# ============================================================================
#: 12 个 RDKit 分子描述符（顺序即特征向量顺序，训练与预测必须一致）
DESCRIPTOR_NAMES: List[str] = [
    "MolWt",            # 分子量
    "LogP",             # 脂水分配系数
    "HBD",              # 氢键供体数
    "HBA",              # 氢键受体数
    "TPSA",             # 拓扑极性表面积
    "RotatableBonds",   # 可旋转键数
    "AromaticRings",    # 芳香环数
    "HeavyAtoms",       # 重原子数
    "FractionCSP3",     # sp3 碳比例
    "RingCount",        # 环总数
    "NumHeteroatoms",   # 杂原子数
    "MolMR",            # 摩尔折射率
]

#: 描述符中文说明（教学展示用）
DESCRIPTOR_LABELS: Dict[str, str] = {
    "MolWt": "分子量 MolWt",
    "LogP": "脂水分配系数 LogP",
    "HBD": "氢键供体数 HBD",
    "HBA": "氢键受体数 HBA",
    "TPSA": "拓扑极性表面积 TPSA",
    "RotatableBonds": "可旋转键数",
    "AromaticRings": "芳香环数",
    "HeavyAtoms": "重原子数",
    "FractionCSP3": "sp3 碳比例",
    "RingCount": "环总数",
    "NumHeteroatoms": "杂原子数",
    "MolMR": "摩尔折射率 MolMR",
}

#: GNN 原子特征维度（与 gcn_egfr_best_model.pth 严格一致，禁止改成 14）
ATOM_FEATURE_DIM = 13

#: 默认活性阈值：pIC50 >= 阈值 记活性，<= 阈值-1 记非活性，中间丢弃
DEFAULT_PIC50_THRESHOLD = 6.0

#: 教学用示例分子（吉非替尼，EGFR 抑制剂）
EXAMPLE_SMILES = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"

#: ChEMBL 靶点/活性端点单次请求的分子结构批量大小
_STRUCTURE_CHUNK_SIZE = 100

#: ChEMBL API 单次请求的最大返回条数（客户端默认只有 20，导致下载奇慢）
_CHEMBL_PAGE_LIMIT = 1000


def _configure_chembl_client(client_endpoints: Optional[List[Any]] = None) -> None:
    """把 ChEMBL 客户端的单页上限调到 1000（幂等）。

    背景（实测根因）：``chembl_webresource_client`` 的 ``Settings.MAX_LIMIT`` 默认是 **20**，
    即每次 HTTP 请求只能拿回 20 条记录。下载 300 个分子要 15+ 次请求，
    下载 2000 个分子要 500 次请求（实测从十几秒恶化到十分钟以上）。
    ChEMBL REST API 单页上限为 1000，因此把客户端页大小调到 1000，
    同样的数据量只需 1~10 次请求。

    注意：客户端在 **导入时** 就用 ``Settings.MAX_LIMIT`` 构造好了各端点的查询对象
    （``chembl_webresource_client/url_query.py`` 里 ``self.limit = Settings.Instance().MAX_LIMIT``），
    所以只改 ``Settings`` 不够，必须同时修正已存在端点的 ``query.limit``。

    Args:
        client_endpoints: 需要调大页大小的端点（target / activity / molecule 等）。
    """
    if not CHEMBL_AVAILABLE:
        return
    try:
        settings = Settings.Instance()
        if int(getattr(settings, "MAX_LIMIT", 0)) < _CHEMBL_PAGE_LIMIT:
            settings.MAX_LIMIT = _CHEMBL_PAGE_LIMIT
    except Exception as exc:  # 配置失败不应阻断下载，只是慢一些
        logger.warning("调整 ChEMBL Settings.MAX_LIMIT 失败: %s", exc)

    for endpoint in client_endpoints or []:
        try:
            query = getattr(endpoint, "query", None)
            if query is not None and int(getattr(query, "limit", 0)) < _CHEMBL_PAGE_LIMIT:
                query.limit = _CHEMBL_PAGE_LIMIT
        except Exception as exc:
            logger.warning("调整 ChEMBL 端点页大小失败（不影响功能，只是下载较慢）: %s", exc)


# ============================================================================
# 数据结构
# ============================================================================
@dataclass
class DataStage:
    """清洗流程中的一个阶段（教学核心：让数据流失看得见）。

    Attributes:
        name: 阶段名称，如「去重」。
        description: 该阶段在做什么（一句话讲清原理）。
        remaining: 该阶段结束后剩余的样本数。
        removed: 该阶段移除的样本数。
        note: 备注（例如移除原因统计），可为空。
    """

    name: str
    description: str
    remaining: int
    removed: int
    note: str = ""

    @property
    def retention(self) -> float:
        """相对上一阶段的保留率（0~1）；上一阶段为空时返回 0.0。"""
        before = self.remaining + self.removed
        if before <= 0:
            return 0.0
        return self.remaining / before


@dataclass
class TrainingResult:
    """一次现场训练的结果（RF 与 GNN 共用）。

    Attributes:
        model_name: 模型名称，如「随机森林 (RF)」。
        auc: 测试集 AUC（单类别测试集时为 float('nan')）。
        accuracy: 测试集准确率（阈值 0.5）。
        n_train: 训练集样本数。
        n_test: 测试集样本数。
        n_features: 输入特征维度（RF=12，GNN=13）。
        extra: 附加内容，键名约定见各训练函数的 docstring。
    """

    model_name: str
    auc: float
    accuracy: float
    n_train: int
    n_test: int
    n_features: int = 0
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def auc_text(self) -> str:
        """AUC 的展示文本（NaN 时给出友好提示）。"""
        if self.auc is None or (isinstance(self.auc, float) and math.isnan(self.auc)):
            return "无法计算（测试集只有单一类别）"
        return f"{self.auc:.4f}"


# ============================================================================
# 环境状态
# ============================================================================
def get_teach_lab_status() -> Dict[str, Any]:
    """返回教学实验室各可选依赖的可用状态，供页面顶部提示使用。

    Returns:
        字典，包含 ``rdkit`` / ``chembl`` / ``torch`` 三个布尔值及错误信息。
    """
    return {
        "rdkit": RDKIT_AVAILABLE,
        "chembl": CHEMBL_AVAILABLE,
        "torch": TORCH_AVAILABLE,
        "rdkit_error": RDKIT_IMPORT_ERROR,
        "chembl_error": CHEMBL_IMPORT_ERROR,
        "torch_error": TORCH_IMPORT_ERROR,
    }


# ============================================================================
# 通用小工具：SMILES 与描述符
# ============================================================================
def canonicalize_smiles(smiles: Any) -> Optional[str]:
    """把 SMILES 标准化为 RDKit 规范形式。

    Args:
        smiles: 原始 SMILES 字符串（可能为 None / 非字符串）。

    Returns:
        规范化 SMILES；无法解析时返回 None。
    """
    if not RDKIT_AVAILABLE or not isinstance(smiles, str) or not smiles.strip():
        return None
    try:
        mol = Chem.MolFromSmiles(smiles)
    except Exception:  # pragma: no cover - RDKit 极少抛异常，防御式兜底
        return None
    if mol is None:
        return None
    try:
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:  # pragma: no cover
        return smiles.strip()


def _descriptor_functions() -> Dict[str, Callable[[Any], float]]:
    """构建「描述符名称 -> 计算函数」映射（每次调用重建，避免模块导入即依赖 RDKit）。"""
    if not RDKIT_AVAILABLE:
        return {}
    return {
        "MolWt": lambda m: float(Descriptors.MolWt(m)),
        "LogP": lambda m: float(Descriptors.MolLogP(m)),
        "HBD": lambda m: float(Descriptors.NumHDonors(m)),
        "HBA": lambda m: float(Descriptors.NumHAcceptors(m)),
        "TPSA": lambda m: float(Descriptors.TPSA(m)),
        "RotatableBonds": lambda m: float(Descriptors.NumRotatableBonds(m)),
        # CalcNumAromaticRings 比 Descriptors.NumAromaticRings 更早版本就存在，兼容性更稳
        "AromaticRings": lambda m: float(rdMolDescriptors.CalcNumAromaticRings(m)),
        "HeavyAtoms": lambda m: float(m.GetNumHeavyAtoms()),
        "FractionCSP3": lambda m: float(Descriptors.FractionCSP3(m)),
        "RingCount": lambda m: float(Descriptors.RingCount(m)),
        "NumHeteroatoms": lambda m: float(Descriptors.NumHeteroatoms(m)),
        "MolMR": lambda m: float(Descriptors.MolMR(m)),
    }


def compute_descriptors(smiles: str) -> Optional[Dict[str, float]]:
    """计算一个分子的 12 个 RDKit 描述符。

    Args:
        smiles: 分子 SMILES。

    Returns:
        形如 ``{"MolWt": 446.9, ...}`` 的字典；SMILES 无效或 RDKit 缺失时返回 None。
        单个描述符计算失败的条目记为 ``float('nan')``。
    """
    if not RDKIT_AVAILABLE:
        return None
    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        return None

    feats: Dict[str, float] = {}
    for name, func in _descriptor_functions().items():
        try:
            feats[name] = float(func(mol))
        except Exception as exc:  # 单个描述符异常不影响整体流程
            logger.debug("描述符 %s 计算失败 (%s): %s", name, smiles[:30], exc)
            feats[name] = float("nan")

    # 保证 12 个键齐全且顺序与 DESCRIPTOR_NAMES 一致
    return {name: feats.get(name, float("nan")) for name in DESCRIPTOR_NAMES}


def pic50_from_nm(value_nm: float) -> float:
    """把 nM 单位的活性值换算为 pIC50。

    pIC50 = 9 - log10(IC50[nM])（IC50=1 nM 时 pIC50=9，IC50=1 uM 时 pIC50=6）。

    Args:
        value_nm: 以 nM 为单位的活性值。

    Returns:
        pIC50；输入非正数时返回 ``float('nan')``。
    """
    try:
        value = float(value_nm)
    except (TypeError, ValueError):
        return float("nan")
    if value <= 0:
        return float("nan")
    return 9.0 - math.log10(value)


# ============================================================================
# 步骤 1：从 ChEMBL 下载真实活性数据
# ============================================================================
def fetch_target_activities(
    target_name: str,
    activity_type: str = "IC50",
    max_records: int = 300,
    organism: str = "Homo sapiens",
    status_callback: Optional[Callable[[str], None]] = None,
) -> Dict[str, Any]:
    """从 ChEMBL 按靶点名下载真实活性数据并换算 pIC50。

    流程（与 utils/data_fetcher.py 保持一致，便于教学对照）：
        1. ``target.search(名称)`` 找到靶点（**必须用 search 端点**）
        2. 过滤物种，优先选 SINGLE PROTEIN
        3. ``activity.filter(...)`` 拉取指定类型的活性记录，只保留 nM 单位
        4. 按分子去重、计算 pIC50、截断到 max_records
        5. 批量拉取分子结构（canonical SMILES）并合并

    Args:
        target_name: 靶点名称，如 "EGFR"、"HER2"、"BTK"。
        activity_type: 活性类型，IC50 / Ki / Kd / EC50。
        max_records: 最多返回多少条（个分子）记录，页面限制 50~2000。
        organism: 物种，默认人类。
        status_callback: 进度回调，接收一句中文进度文本（页面用于显示状态）。

    Returns:
        字典：
            ``success``          是否成功
            ``message``          给人看的提示信息
            ``df``               成功时为 DataFrame，列见下
            ``target_name``      实际命中的靶点名
            ``target_chembl_id`` ChEMBL 靶点 ID
            ``activity_type``    活性类型
            ``raw_count``        原始活性记录数（未过滤前）

        ``df`` 的列：``molecule_chembl_id`` / ``smiles`` / ``standard_value`` /
        ``standard_units`` / ``pIC50`` / ``activity_type`` / ``target_name`` /
        ``target_chembl_id``
    """

    def _notify(msg: str) -> None:
        logger.info(msg)
        if status_callback is not None:
            try:
                status_callback(msg)
            except Exception:  # 回调失败不能影响主流程
                pass

    outcome: Dict[str, Any] = {
        "success": False,
        "message": "",
        "df": None,
        "target_name": target_name,
        "target_chembl_id": None,
        "activity_type": activity_type,
        "raw_count": 0,
    }

    if not CHEMBL_AVAILABLE:
        outcome["message"] = (
            "chembl_webresource_client 未安装或导入失败。"
            "请运行: pip install \"chembl_webresource_client>=0.10.9\""
        )
        return outcome

    if not target_name or not target_name.strip():
        outcome["message"] = "请先填写靶点名称（例如 EGFR）。"
        return outcome

    max_records = int(max(1, min(int(max_records), 2000)))
    target_name = target_name.strip()

    try:
        targets_api = new_client.target
        activities_api = new_client.activity
        molecules_api = new_client.molecule
    except Exception as exc:  # pragma: no cover - 网络/客户端初始化失败
        outcome["message"] = f"ChEMBL 客户端初始化失败: {exc}"
        return outcome

    # 先把三个端点的页大小都调到 1000（默认 20 会让下载慢十倍以上）
    _configure_chembl_client([targets_api, activities_api, molecules_api])

    try:
        # ---- 1. 靶点检索：必须用 search()，get()/filter() 对名称检索不稳定 ----
        _notify(f"① 正在 ChEMBL 检索靶点「{target_name}」…")
        target_records = [
            dict(rec)
            for rec in targets_api.search(target_name).only(
                "target_chembl_id", "pref_name", "organism", "target_type"
            )
        ]
        if not target_records:
            outcome["message"] = f"ChEMBL 未找到靶点「{target_name}」，请检查拼写（如 EGFR / HER2 / BTK）。"
            return outcome

        target_df = pd.DataFrame.from_records(target_records)
        if "organism" in target_df.columns:
            human_df = target_df[target_df["organism"] == organism]
            target_df = human_df if not human_df.empty else target_df

        # 优先选择单蛋白（SINGLE PROTEIN）靶点，避免命中复合物/细胞系
        if "target_type" in target_df.columns:
            single = target_df[target_df["target_type"] == "SINGLE PROTEIN"]
            if not single.empty:
                target_df = single

        target_row = target_df.iloc[0]
        target_chembl_id = str(target_row["target_chembl_id"])
        pref_name = str(target_row.get("pref_name") or target_name)
        outcome["target_name"] = pref_name
        outcome["target_chembl_id"] = target_chembl_id
        _notify(f"   → 命中靶点：{pref_name} ({target_chembl_id})")

        # ---- 2. 活性数据：只取标准类型、标准关系、结合测定 ----
        _notify(f"② 正在拉取 {activity_type} 活性记录…")
        activities = activities_api.filter(
            target_chembl_id=target_chembl_id,
            type=activity_type,
            relation="=",
            assay_type="B",
            target_organism=organism,
        ).only(
            "activity_id",
            "molecule_chembl_id",
            "standard_value",
            "standard_units",
            "standard_type",
            "relation",
        )

        # 教学场景只要有限样本：主动截断，避免把整靶点（可能上万条）全拉下来。
        # 截断靠迭代器 islice 完成，并且已把客户端页大小调到 1000，所以只发少量请求。
        raw_records = [dict(rec) for rec in itertools.islice(iter(activities), max_records * 5)]
        outcome["raw_count"] = len(raw_records)
        if not raw_records:
            outcome["message"] = f"未找到「{pref_name}」的 {activity_type} 活性数据，请换活性类型或靶点再试。"
            return outcome

        bio_df = pd.DataFrame.from_records(raw_records)

        # ---- 3. 清洗原始活性：nM 单位 -> 数值 -> 去重 -> pIC50 ----
        bio_df = bio_df[bio_df.get("standard_units") == "nM"]
        if bio_df.empty:
            outcome["message"] = f"没有 nM 单位的 {activity_type} 数据，请尝试其他活性类型。"
            return outcome

        bio_df = bio_df.copy()
        bio_df["standard_value"] = pd.to_numeric(bio_df["standard_value"], errors="coerce")
        bio_df = bio_df.dropna(subset=["standard_value", "molecule_chembl_id"])
        bio_df = bio_df[bio_df["standard_value"] > 0]
        if bio_df.empty:
            outcome["message"] = "有效数值型活性记录为 0，请稍后重试或更换靶点。"
            return outcome

        bio_df = bio_df.drop_duplicates(subset=["molecule_chembl_id"], keep="first")
        bio_df["pIC50"] = bio_df["standard_value"].apply(pic50_from_nm)
        bio_df = bio_df.dropna(subset=["pIC50"])

        # 按活性从强到弱排序，再在整个活性区间上等间隔抽样。
        # 教学关键：绝不能只取“活性最强的前 N 个”（那会得到只有活性样本的单类别数据集，
        # 模型无法训练）；等间隔抽样能同时保留活性/非活性/中间地带，标签分布更真实。
        bio_df = bio_df.sort_values("pIC50", ascending=False).reset_index(drop=True)
        if len(bio_df) > max_records:
            pick = np.linspace(0, len(bio_df) - 1, num=max_records).round().astype(int)
            bio_df = bio_df.iloc[np.unique(pick)].reset_index(drop=True)
        _notify(
            f"   → 得到 {len(bio_df)} 个分子的 {activity_type} 数据"
            f"（pIC50 {bio_df['pIC50'].min():.2f} ~ {bio_df['pIC50'].max():.2f}，“从最强到最弱”均匀覆盖）"
        )

        # ---- 4. 批量补齐分子结构（canonical SMILES）----
        chembl_ids = bio_df["molecule_chembl_id"].astype(str).tolist()
        structure_rows: List[Dict[str, Any]] = []
        total_chunks = (len(chembl_ids) + _STRUCTURE_CHUNK_SIZE - 1) // _STRUCTURE_CHUNK_SIZE
        for chunk_index, start in enumerate(range(0, len(chembl_ids), _STRUCTURE_CHUNK_SIZE), start=1):
            chunk = chembl_ids[start:start + _STRUCTURE_CHUNK_SIZE]
            _notify(f"③ 正在获取分子结构（第 {chunk_index}/{total_chunks} 批）…")
            try:
                recs = molecules_api.filter(molecule_chembl_id__in=chunk).only(
                    "molecule_chembl_id", "molecule_structures"
                )
            except Exception as exc:  # 单批失败不影响其它批次
                logger.warning("批量获取分子结构失败（%d-%d）: %s", start, start + len(chunk), exc)
                continue
            for rec in recs:
                row = dict(rec)
                structures = row.get("molecule_structures")
                row["smiles"] = (
                    structures.get("canonical_smiles") if isinstance(structures, dict) else None
                )
                structure_rows.append(row)

        if not structure_rows:
            outcome["message"] = "未能获取分子结构数据（ChEMBL 结构端点无返回），请重试。"
            return outcome

        struct_df = pd.DataFrame.from_records(structure_rows)
        struct_df = struct_df[["molecule_chembl_id", "smiles"]].dropna(subset=["smiles"])
        struct_df["molecule_chembl_id"] = struct_df["molecule_chembl_id"].astype(str)

        merged = pd.merge(bio_df, struct_df, on="molecule_chembl_id", how="inner")
        if merged.empty:
            outcome["message"] = "活性数据与分子结构无法匹配（SMILES 缺失），请重试。"
            return outcome

        merged["activity_type"] = activity_type
        merged["target_name"] = pref_name
        merged["target_chembl_id"] = target_chembl_id

        result_df = merged[[
            "molecule_chembl_id", "smiles", "standard_value", "standard_units",
            "pIC50", "activity_type", "target_name", "target_chembl_id",
        ]].reset_index(drop=True)

        outcome["df"] = result_df
        outcome["success"] = True
        outcome["message"] = (
            f"成功下载 {len(result_df)} 个分子的 {activity_type} 数据"
            f"（靶点 {pref_name}，ChEMBL 原始记录 {outcome['raw_count']} 条，"
            f"pIC50 覆盖 {result_df['pIC50'].min():.2f} ~ {result_df['pIC50'].max():.2f}）"
        )
        _notify("④ 下载完成 ✅")
        return outcome

    except Exception as exc:  # 网络异常等：给教学友好提示
        logger.error("ChEMBL 数据下载失败: %s", exc, exc_info=True)
        outcome["message"] = f"ChEMBL 数据下载失败：{exc}"
        return outcome


# ============================================================================
# 步骤 2：数据清洗（用 DataStage 记录每一步的数据流失）
# ============================================================================
def clean_activity_data(
    raw_df: pd.DataFrame,
    pic50_threshold: float = DEFAULT_PIC50_THRESHOLD,
    progress_callback: Optional[Callable[[str, float], None]] = None,
) -> Tuple[pd.DataFrame, List[DataStage]]:
    """清洗活性数据并打标签，逐步记录样本流失。

    清洗顺序（每一步都生成一个 DataStage）：
        1. 原始数据
        2. 去重（同一 SMILES 只保留一次；以 RDKit 规范 SMILES 为准）
        3. SMILES 有效性检查（RDKit 解析）
        4. 活性阈值过滤（pIC50 >= 阈值 → 活性；<= 阈值-1 → 非活性；中间丢弃）
        5. 分子描述符计算（12 个 RDKit 描述符，计算失败的分子移除）

    Args:
        raw_df: 步骤 1 的原始数据，必须含 ``smiles`` 与 ``pIC50`` 列。
        pic50_threshold: 活性阈值，默认 6.0（相当于 1 uM，即 1 微摩尔/升）。
        progress_callback: ``(阶段名称, 进度 0~1)`` 回调，页面用于显示进度条。

    Returns:
        ``(clean_df, stages)``：
            ``clean_df`` 列 = smiles / pic50 / label / 12 个描述符（保留 molecule_chembl_id 若存在）
            ``stages`` 为 DataStage 列表，顺序即清洗顺序
    """
    if not RDKIT_AVAILABLE:
        raise RuntimeError("RDKit 不可用，无法清洗数据。请安装 rdkit-pypi==2022.9.5。")
    if raw_df is None or raw_df.empty:
        raise ValueError("原始数据为空，请先在步骤 1 下载数据。")
    if "smiles" not in raw_df.columns or "pIC50" not in raw_df.columns:
        raise ValueError("原始数据缺少 smiles / pIC50 列，无法清洗。")

    stages: List[DataStage] = []
    total_steps = 5

    def _report(step: int, name: str) -> None:
        if progress_callback is not None:
            try:
                progress_callback(name, step / total_steps)
            except Exception:
                pass

    # ---- 阶段 1：原始数据 ----
    _report(1, "原始数据")
    work = raw_df.copy()
    work["smiles"] = work["smiles"].astype(str)
    work["pic50"] = pd.to_numeric(work["pIC50"], errors="coerce")
    work = work.dropna(subset=["pic50"])
    stages.append(DataStage(
        name="原始数据",
        description="从 ChEMBL 拉取的真实活性数据，活性值已换算为 pIC50（pIC50 = 9 - log10(IC50[nM])）。",
        remaining=len(work),
        removed=0,
        note="pIC50 越大代表活性越强；pIC50=6 约等于 1 uM（1 微摩尔/升）。",
    ))

    # ---- 阶段 2：去重（完全相同的 SMILES 字符串只保留一条）----
    _report(2, "去重")
    before = len(work)
    # 按 pIC50 降序后 keep="first"：同一记录的多次测定保留活性最强的那条
    work = work.sort_values("pic50", ascending=False).drop_duplicates(
        subset=["smiles"], keep="first"
    )
    stages.append(DataStage(
        name="去重",
        description="完全相同的 SMILES 字符串只保留一条（同一分子的多次测定取 pIC50 最高的一次）。",
        remaining=len(work),
        removed=before - len(work),
        note="ChEMBL 中同一化合物常有多个测定结果；不去重会让「同一个分子」被重复计权，模型会偏向它。",
    ))

    # ---- 阶段 3：SMILES 有效性检查（并用规范 SMILES 再归一一次）----
    _report(3, "SMILES 有效性检查")
    before = len(work)
    work = work.copy()
    work["canonical_smiles"] = work["smiles"].apply(canonicalize_smiles)
    invalid_count = int(work["canonical_smiles"].isna().sum())
    work = work.dropna(subset=["canonical_smiles"])
    valid_count = len(work)
    # 不同写法可能指向同一分子（如 "Oc1ccccc1" 与 "c1ccccc1O"），用规范 SMILES 再去重一次
    work = work.drop_duplicates(subset=["canonical_smiles"], keep="first")
    rewritten_count = valid_count - len(work)
    # 统一改用 RDKit 规范 SMILES：后续描述符计算与建图都基于它
    work["smiles"] = work["canonical_smiles"]
    work = work.drop(columns=["canonical_smiles"])
    stages.append(DataStage(
        name="SMILES 有效性检查",
        description=(
            "用 RDKit 解析每一条 SMILES：解析失败的直接移除；解析成功的统一改写为规范 SMILES"
            "（同一分子的不同写法会归一成同一个）。"
        ),
        remaining=len(work),
        removed=before - len(work),
        note=(
            f"其中 {invalid_count} 条无法被 RDKit 解析（多为含金属/聚合物/异常盐形式的记录），"
            f"{rewritten_count} 条是同一分子的不同写法。"
        ),
    ))

    # ---- 阶段 4：活性阈值过滤 ----
    _report(4, f"活性阈值过滤 (pIC50 {pic50_threshold:g})")
    before = len(work)
    threshold = float(pic50_threshold)
    work = work.copy()
    work["label"] = np.where(
        work["pic50"] >= threshold, 1,
        np.where(work["pic50"] <= threshold - 1.0, 0, -1),
    )
    ambiguous = int((work["label"] == -1).sum())
    work = work[work["label"] != -1]
    work["label"] = work["label"].astype(int)
    stages.append(DataStage(
        name=f"活性阈值过滤 (pIC50 ≥ {threshold:g} → 活性)",
        description=(
            f"pIC50 ≥ {threshold:g} 记为活性(1)，pIC50 ≤ {threshold - 1.0:g} 记为非活性(0)，"
            "中间地带样本语义模糊，主动丢弃。"
        ),
        remaining=len(work),
        removed=before - len(work),
        note=(
            f"其中 {ambiguous} 条落在中间地带（{threshold - 1.0:g} < pIC50 < {threshold:g}）。"
            "阈值越靠近中间地带，标签越模糊——这正是真实数据标注的难点。"
        ),
    ))

    # ---- 阶段 5：描述符计算 ----
    _report(5, "分子描述符计算")
    before = len(work)
    descriptor_rows: List[Dict[str, float]] = []
    failed = 0
    for smi in work["smiles"].tolist():
        feats = compute_descriptors(smi)
        if feats is None or any(pd.isna(v) for v in feats.values()):
            failed += 1
            descriptor_rows.append({name: float("nan") for name in DESCRIPTOR_NAMES})
        else:
            descriptor_rows.append(feats)

    desc_df = pd.DataFrame(descriptor_rows, columns=DESCRIPTOR_NAMES).reset_index(drop=True)
    work = pd.concat([work.reset_index(drop=True), desc_df], axis=1)
    work = work.dropna(subset=DESCRIPTOR_NAMES)
    stages.append(DataStage(
        name="分子描述符计算",
        description="计算 12 个 RDKit 分子描述符（分子量/LogP/HBD/HBA/TPSA/可旋转键/芳香环/重原子/sp3 碳/环数/杂原子/摩尔折射率）。",
        remaining=len(work),
        removed=before - len(work),
        note=f"共 {failed} 条记录存在描述符计算失败或缺失值，已移除（避免用 0 填充误导模型）。",
    ))

    keep_cols = [c for c in ["molecule_chembl_id", "smiles", "pic50", "label"] if c in work.columns]
    clean_df = work[keep_cols + DESCRIPTOR_NAMES].reset_index(drop=True)

    if clean_df.empty:
        raise ValueError(
            "清洗后没有剩余样本：请降低活性阈值、增大下载量，或换一个靶点/Bioactivity 类型。"
        )

    logger.info("清洗完成: %d 条样本，%d 个清洗阶段", len(clean_df), len(stages))
    return clean_df, stages


def summarize_labels(clean_df: pd.DataFrame) -> Dict[str, int]:
    """统计样本标签分布。

    Args:
        clean_df: 清洗后的数据框（需含 ``label`` 列）。

    Returns:
        ``{"total": n, "active": n1, "inactive": n0}``
    """
    if clean_df is None or clean_df.empty or "label" not in clean_df.columns:
        return {"total": 0, "active": 0, "inactive": 0}
    labels = clean_df["label"].astype(int)
    return {
        "total": int(len(labels)),
        "active": int((labels == 1).sum()),
        "inactive": int((labels == 0).sum()),
    }


# ============================================================================
# 步骤 3.1：随机森林（秒级）
# ============================================================================
def train_rf_model(
    clean_df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
    n_estimators: int = 100,
    max_depth: int = 10,
) -> TrainingResult:
    """用 12 个描述符现场训练一个随机森林分类器。

    Args:
        clean_df: 步骤 2 输出的数据框（12 个描述符 + label）。
        test_size: 测试集比例，默认 0.2（8:2 划分）。
        random_state: 随机种子，固定以便教学复现。
        n_estimators: 树的数量（默认 100）。
        max_depth: 树的最大深度（默认 10）。

    Returns:
        TrainingResult，``extra`` 键：
            ``model``               训练好的 RandomForestClassifier
            ``feature_importances`` ``[(描述符名, 重要性), ...]`` 按重要性降序
            ``roc``                  ``(fpr, tpr)`` 测试集 ROC 曲线点
            ``test_proba``/``test_label``  测试集预测概率/真实标签
            ``train_seconds``       训练耗时（time.perf_counter 计时）
            ``positive_rate``       训练集活性样本比例
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
    from sklearn.model_selection import train_test_split

    if clean_df is None or clean_df.empty:
        raise ValueError("请先在步骤 2 完成数据清洗。")

    X = clean_df[DESCRIPTOR_NAMES].to_numpy(dtype=float)
    y = clean_df["label"].to_numpy(dtype=int)

    if len(np.unique(y)) < 2:
        raise ValueError(
            "标签只有单一类别，无法训练/评估（AUC 无定义）。"
            "请降低活性阈值或增大下载量，让活性与非活性样本都存在。"
        )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # 计时：明确使用 time.perf_counter()（不涉及 GPU，也不使用 torch.cuda.Event）
    t0 = time.perf_counter()
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - t0

    proba = model.predict_proba(X_test)[:, 1]
    pred = (proba >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_test, pred))
    try:
        auc = float(roc_auc_score(y_test, proba))
        fpr, tpr, _ = roc_curve(y_test, proba)
        roc = (fpr.tolist(), tpr.tolist())
    except ValueError:  # 测试集单类别
        auc = float("nan")
        roc = ([], [])

    importances = sorted(
        zip(DESCRIPTOR_NAMES, [float(v) for v in model.feature_importances_]),
        key=lambda item: item[1],
        reverse=True,
    )

    logger.info(
        "RF 训练完成: AUC=%.4f, Acc=%.4f, 训练 %d / 测试 %d, 耗时 %.2fs",
        auc, accuracy, len(X_train), len(X_test), train_seconds,
    )

    return TrainingResult(
        model_name="随机森林 (RF)",
        auc=auc,
        accuracy=accuracy,
        n_train=int(len(X_train)),
        n_test=int(len(X_test)),
        n_features=len(DESCRIPTOR_NAMES),
        extra={
            "model": model,
            "feature_importances": importances,
            "roc": roc,
            "test_proba": proba.tolist(),
            "test_label": y_test.tolist(),
            "train_seconds": train_seconds,
            "positive_rate": float(y_train.mean()),
            "n_estimators": n_estimators,
            "max_depth": max_depth,
        },
    )


def predict_smiles_rf(result: TrainingResult, smiles: str) -> Dict[str, Any]:
    """用步骤 3.1 训练出的 RF 模型预测一个分子。

    Args:
        result: ``train_rf_model`` 的返回结果。
        smiles: 待预测分子的 SMILES。

    Returns:
        ``{"success", "probability_active", "label", "descriptors", "error"}``
    """
    model = result.extra.get("model") if result is not None else None
    if model is None:
        return {"success": False, "error": "尚未训练随机森林模型，请先完成步骤 3.1。", "smiles": smiles}

    feats = compute_descriptors(smiles)
    if feats is None:
        return {"success": False, "error": "SMILES 无法被 RDKit 解析，请检查输入。", "smiles": smiles}
    if any(pd.isna(v) for v in feats.values()):
        return {"success": False, "error": "该分子的描述符存在缺失值，无法预测。", "smiles": smiles}

    vector = np.array([[feats[name] for name in DESCRIPTOR_NAMES]], dtype=float)
    try:
        proba = float(model.predict_proba(vector)[0, 1])
    except Exception as exc:
        return {"success": False, "error": f"预测失败: {exc}", "smiles": smiles}

    return {
        "success": True,
        "smiles": smiles,
        "probability_active": proba,
        "label": "活性" if proba >= 0.5 else "非活性",
        "descriptors": feats,
        "error": None,
    }


# ============================================================================
# 步骤 3.2：图神经网络 GNN（分钟级，可选）
# ============================================================================
if TORCH_AVAILABLE:

    class TeachGCN(torch.nn.Module):
        """教学用 GCN 分类器，结构与项目训练脚本**完全一致**。

        前向传播：
            conv1 → bn1 → ReLU → Dropout(0.5)
            conv2 → bn2 → ReLU → Dropout(0.5)
            conv3 → global_mean_pool → lin1 → ReLU → Dropout(0.5) → lin2

        层名必须是 conv1/conv2/conv3、bn1/bn2（不用 ModuleList），
        否则与 gcn_egfr_best_model.pth 的 state_dict 键名不匹配。
        """

        def __init__(
            self,
            num_node_features: int = ATOM_FEATURE_DIM,
            hidden_dim: int = 128,
        ) -> None:
            super().__init__()
            self.conv1 = GCNConv(num_node_features, hidden_dim)
            self.conv2 = GCNConv(hidden_dim, hidden_dim)
            self.conv3 = GCNConv(hidden_dim, hidden_dim)
            self.bn1 = BatchNorm1d(hidden_dim)
            self.bn2 = BatchNorm1d(hidden_dim)
            self.lin1 = Linear(hidden_dim, hidden_dim // 2)
            self.lin2 = Linear(hidden_dim // 2, 1)

        def forward(self, data: Any) -> Any:
            """前向传播，返回形状 [batch_size, 1] 的 logit。"""
            x, edge_index, batch = data.x, data.edge_index, data.batch

            x = self.conv1(x, edge_index)
            x = self.bn1(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            x = self.conv2(x, edge_index)
            x = self.bn2(x)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)

            x = self.conv3(x, edge_index)

            x = global_mean_pool(x, batch)

            x = F.relu(self.lin1(x))
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.lin2(x)

            return x.view(-1, 1)

else:  # pragma: no cover - 取决于环境
    TeachGCN = None  # type: ignore[assignment,misc]


def smiles_to_graph(smiles: str, label: Optional[int] = None) -> Any:
    """把 SMILES 转成 PyG ``Data`` 对象（原子特征固定 13 维）。

    13 维原子特征顺序（与 gnn_predictor.py 严格一致）：
        [原子序数, 度数, 形式电荷, 杂化类型, 芳香性, 总氢数,
         隐式价, 自由基电子数, 同位素, 质量/100, 总价, 是否有隐式氢, 是否为N/O]

    Args:
        smiles: 分子 SMILES。
        label: 可选标签（0/1），会写入 ``data.y``。

    Returns:
        ``torch_geometric.data.Data``。

    Raises:
        RuntimeError: PyTorch/PyG 不可用。
        ValueError: SMILES 无效或分子无原子。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch / PyTorch Geometric 不可用，无法进行 GNN 流程。"
            f"原始错误: {TORCH_IMPORT_ERROR}"
        )

    mol = Chem.MolFromSmiles(smiles) if isinstance(smiles, str) else None
    if mol is None:
        raise ValueError(f"无效的 SMILES: {smiles}")

    atom_features: List[List[float]] = []
    for atom in mol.GetAtoms():
        atom_features.append([
            float(atom.GetAtomicNum()),                      # 1. 原子序数
            float(atom.GetDegree()),                         # 2. 度（连接数）
            float(atom.GetFormalCharge()),                   # 3. 形式电荷
            float(atom.GetHybridization().real),             # 4. 杂化类型
            float(atom.GetIsAromatic()),                     # 5. 芳香性
            float(atom.GetTotalNumHs()),                     # 6. 总氢数
            float(atom.GetImplicitValence()),                # 7. 隐式价
            float(atom.GetNumRadicalElectrons()),            # 8. 自由基电子数
            float(atom.GetIsotope()),                        # 9. 同位素
            float(atom.GetMass() / 100.0),                   # 10. 原子质量（/100 归一化）
            float(atom.GetTotalValence()),                   # 11. 总价
            1.0 if atom.GetNumImplicitHs() > 0 else 0.0,     # 12. 是否有隐式氢
            1.0 if atom.GetAtomicNum() in (7, 8) else 0.0,   # 13. 是否为 N/O
        ])

    if not atom_features:
        raise ValueError(f"分子没有原子: {smiles}")

    x = torch.tensor(atom_features, dtype=torch.float)

    # 无向图：每条键建两条有向边（GCNConv 不支持有向消息聚合方向区分）
    edges: List[List[int]] = []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edges.append([i, j])
        edges.append([j, i])
    if not edges:  # 单原子分子（如 [Na+]）：自环兜底，避免空 edge_index
        edges = [[0, 0]]
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

    data = Data(x=x, edge_index=edge_index)
    if label is not None:
        data.y = torch.tensor([float(label)], dtype=torch.float)
    return data


def build_graph_dataset(
    clean_df: pd.DataFrame,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> List[Any]:
    """把清洗后的数据框批量转成 PyG 图数据集。

    Args:
        clean_df: 步骤 2 输出的数据框。
        progress_callback: ``(已完成, 总数)`` 回调，页面用于显示转化进度。

    Returns:
        ``Data`` 列表，每个图带 ``y`` 标签。
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError("PyTorch / PyTorch Geometric 不可用，无法构建图数据集。")
    if clean_df is None or clean_df.empty:
        raise ValueError("请先在步骤 2 完成数据清洗。")

    smiles_list = clean_df["smiles"].astype(str).tolist()
    labels = clean_df["label"].astype(int).tolist()
    total = len(smiles_list)

    graphs: List[Any] = []
    for idx, (smi, lab) in enumerate(zip(smiles_list, labels), start=1):
        try:
            graphs.append(smiles_to_graph(smi, label=lab))
        except ValueError as exc:
            logger.warning("跳过无法建图的分子: %s", exc)
        if progress_callback is not None and (idx % 10 == 0 or idx == total):
            try:
                progress_callback(idx, total)
            except Exception:
                pass

    if not graphs:
        raise ValueError("没有任何分子能成功转换为图，请检查数据。")
    return graphs


def _evaluate_gnn(model: Any, loader: Any) -> Dict[str, Any]:
    """在给定 DataLoader 上评估 GNN，返回概率、标签与指标。

    Returns:
        ``{"auc": float, "accuracy": float, "proba": np.ndarray, "label": np.ndarray}``
        （单类别时 auc 为 nan，proba/label 始终是 np.ndarray，便于调用方统一处理）
    """
    from sklearn.metrics import accuracy_score, roc_auc_score

    model.eval()
    probs: List[float] = []
    labels: List[int] = []
    with torch.no_grad():
        for batch in loader:
            logits = model(batch)
            probs.extend(torch.sigmoid(logits).view(-1).tolist())
            labels.extend(batch.y.view(-1).tolist())

    y_true = np.array([int(v) for v in labels])
    y_prob = np.array([float(v) for v in probs])
    if len(y_true) == 0:
        return {
            "auc": float("nan"),
            "accuracy": float("nan"),
            "proba": np.array([], dtype=float),
            "label": np.array([], dtype=int),
        }

    pred = (y_prob >= 0.5).astype(int)
    accuracy = float(accuracy_score(y_true, pred))
    if len(np.unique(y_true)) < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(y_true, y_prob))
    return {"auc": auc, "accuracy": accuracy, "proba": y_prob, "label": y_true}


def train_gnn_model(
    clean_df: pd.DataFrame,
    epochs: int = 20,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    test_size: float = 0.2,
    random_state: int = 42,
    hidden_dim: int = 128,
    epoch_callback: Optional[Callable[[int, float, float], None]] = None,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> TrainingResult:
    """现场训练一个 GCN（教学用，分钟级）。

    Args:
        clean_df: 步骤 2 输出的数据框。
        epochs: 训练轮数（页面 slider 控制，建议 5~50）。
        batch_size: 批大小。
        learning_rate: Adam 学习率。
        test_size: 测试集比例（8:2 分层划分）。
        random_state: 随机种子（固定以便教学复现）。
        hidden_dim: 隐藏层维度（与既有模型一致为 128）。
        epoch_callback: **每个 epoch 结束后的回调** ``(epoch, train_loss, test_auc)``，
            页面用它实时刷新 Loss / Val AUC 曲线。
        progress_callback: 图数据转化进度 ``(已完成, 总数)``。

    Returns:
        TrainingResult，``extra`` 键：
            ``model``         训练好的 TeachGCN
            ``history``       ``{"loss": [...], "val_auc": [...]}`` 逐 epoch 记录
            ``roc``           ``(fpr, tpr)``
            ``test_proba``/``test_label``
            ``train_seconds`` 总训练耗时（time.perf_counter 计时，不用 torch.cuda.Event）
            ``epochs``/``learning_rate``/``batch_size``/``hidden_dim``
    """
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch / PyTorch Geometric 不可用，无法训练 GNN。"
            f"原始错误: {TORCH_IMPORT_ERROR}"
        )
    from sklearn.metrics import roc_curve
    from sklearn.model_selection import train_test_split

    if clean_df is None or clean_df.empty:
        raise ValueError("请先在步骤 2 完成数据清洗。")
    epochs = int(max(1, epochs))

    # ---- 固定随机种子，保证教学可复现 ----
    np.random.seed(random_state)
    torch.manual_seed(random_state)

    # ---- 1. 构建图数据集 ----
    graphs = build_graph_dataset(clean_df, progress_callback=progress_callback)
    labels = np.array([int(g.y.item()) for g in graphs])
    if len(np.unique(labels)) < 2:
        raise ValueError("标签只有单一类别，无法训练 GNN，请调整活性阈值或增大下载量。")

    # ---- 2. 分层划分 8:2 ----
    indices = np.arange(len(graphs))
    idx_train, idx_test = train_test_split(
        indices, test_size=test_size, random_state=random_state, stratify=labels
    )
    train_graphs = [graphs[i] for i in idx_train]
    test_graphs = [graphs[i] for i in idx_test]

    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_graphs, batch_size=batch_size, shuffle=False)

    # ---- 3. 建模与优化器（结构与既有训练脚本一致）----
    model = TeachGCN(num_node_features=ATOM_FEATURE_DIM, hidden_dim=hidden_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = torch.nn.BCEWithLogitsLoss()

    history: Dict[str, List[float]] = {"loss": [], "val_auc": []}

    # 计时统一使用 time.perf_counter()（本项目曾因 torch.cuda.Event 在 CPU 环境踩坑）
    t0 = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        seen = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch)
            loss = criterion(out, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * int(batch.num_graphs)
            seen += int(batch.num_graphs)

        train_loss = total_loss / max(seen, 1)
        metrics = _evaluate_gnn(model, test_loader)
        val_auc = float(metrics["auc"])
        history["loss"].append(train_loss)
        history["val_auc"].append(val_auc)

        logger.info("GNN epoch %d/%d loss=%.4f val_auc=%.4f", epoch, epochs, train_loss, val_auc)

        # 每个 epoch 回调一次，页面据此实时画曲线
        if epoch_callback is not None:
            try:
                epoch_callback(epoch, train_loss, val_auc)
            except Exception as exc:  # 回调异常不应中断训练
                logger.warning("epoch_callback 执行失败: %s", exc)

    train_seconds = time.perf_counter() - t0

    # ---- 4. 训练结束后的最终评估 ----
    final = _evaluate_gnn(model, test_loader)
    y_true = np.asarray(final["label"], dtype=int)
    y_prob = np.asarray(final["proba"], dtype=float)
    if len(np.unique(y_true)) >= 2:
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc = (fpr.tolist(), tpr.tolist())
    else:
        roc = ([], [])

    logger.info(
        "GNN 训练完成: AUC=%.4f, Acc=%.4f, 训练 %d / 测试 %d, 耗时 %.1fs",
        final["auc"], final["accuracy"], len(idx_train), len(idx_test), train_seconds,
    )

    return TrainingResult(
        model_name="图神经网络 (GCN)",
        auc=float(final["auc"]),
        accuracy=float(final["accuracy"]),
        n_train=int(len(idx_train)),
        n_test=int(len(idx_test)),
        n_features=ATOM_FEATURE_DIM,
        extra={
            "model": model,
            "history": history,
            "roc": roc,
            "test_proba": y_prob.tolist(),
            "test_label": y_true.tolist(),
            "train_seconds": train_seconds,
            "epochs": epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "hidden_dim": hidden_dim,
            # 说明：这里没有迁移加载项目的预训练权重，模型是"现场从零训练"的
            "from_scratch": True,
        },
    )


def predict_smiles_gnn(result: TrainingResult, smiles: str) -> Dict[str, Any]:
    """用步骤 3.2 现场训练的 GNN 预测一个分子。

    Args:
        result: ``train_gnn_model`` 的返回结果。
        smiles: 待预测分子的 SMILES。

    Returns:
        ``{"success", "probability_active", "label", "num_atoms", "error"}``
    """
    if not TORCH_AVAILABLE:
        return {"success": False, "error": "PyTorch / PyG 不可用，无法使用 GNN 预测。", "smiles": smiles}
    model = result.extra.get("model") if result is not None else None
    if model is None:
        return {"success": False, "error": "尚未训练 GNN 模型，请先完成步骤 3.2。", "smiles": smiles}

    try:
        data = smiles_to_graph(smiles)
    except ValueError as exc:
        return {"success": False, "error": str(exc), "smiles": smiles}

    try:
        model.eval()
        with torch.no_grad():
            logit = model(data)
            proba = float(torch.sigmoid(logit).view(-1)[0].item())
    except Exception as exc:
        return {"success": False, "error": f"GNN 预测失败: {exc}", "smiles": smiles}

    return {
        "success": True,
        "smiles": smiles,
        "probability_active": proba,
        "label": "活性" if proba >= 0.5 else "非活性",
        "num_atoms": int(data.x.size(0)),
        "num_bonds": int(data.edge_index.size(1) // 2),
        "error": None,
    }


# ============================================================================
# 自测入口：python -m utils.teach_lab（不依赖 Streamlit）
# ============================================================================
if __name__ == "__main__":  # pragma: no cover
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    print("=" * 64)
    print("🎓 教学实验室 核心逻辑自测")
    print("=" * 64)
    print("依赖状态:", get_teach_lab_status())

    # 用示例分子验证描述符与建图
    desc = compute_descriptors(EXAMPLE_SMILES)
    print(f"\n示例分子（吉非替尼）描述符: {desc}")
    print(f"pIC50(100 nM) = {pic50_from_nm(100):.2f}")

    if TORCH_AVAILABLE:
        g = smiles_to_graph(EXAMPLE_SMILES, label=1)
        print(f"图数据: 节点 {g.x.shape}（应为 13 维）, 边 {g.edge_index.shape}, y={g.y}")
        model = TeachGCN()
        model.eval()
        print("前向传播输出形状:", tuple(model(g).shape), "（应为 [1, 1]）")
    print("\n✅ 自测结束")
