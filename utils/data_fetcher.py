"""
数据获取模块 - 从 ChEMBL 和 PubChem 获取化合物数据
支持按靶点检索、相似性搜索和批量获取，为自动化流程提供数据源。
"""

import time
import requests
import pandas as pd
import math
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from urllib.parse import quote
import logging

try:
    from chembl_webresource_client.new_client import new_client
    CHEMBL_AVAILABLE = True
except ImportError:
    CHEMBL_AVAILABLE = False
    logging.warning("chembl_webresource_client 未安装，ChEMBL 功能不可用")

logger = logging.getLogger(__name__)


@dataclass
class CompoundRecord:
    """单个化合物记录

    Attributes
    ----------
    smiles : str
        规范 SMILES 字符串
    chembl_id : Optional[str]
        ChEMBL 数据库分子 ID
    pubchem_cid : Optional[str]
        PubChem 数据库化合物 CID
    activity_value : Optional[float]
        pIC50 或 IC50 (nM) 值
    activity_type : Optional[str]
        活性类型（IC50, EC50, Ki 等）
    target_name : Optional[str]
        靶点名称
    source : str
        数据来源标识（"ChEMBL" / "PubChem" / "upload"）
    raw_data : Dict[str, Any]
        原始数据字典，保留完整字段
    """
    smiles: str
    chembl_id: Optional[str] = None
    pubchem_cid: Optional[str] = None
    activity_value: Optional[float] = None
    activity_type: Optional[str] = None
    target_name: Optional[str] = None
    source: str = "unknown"
    raw_data: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FetchResult:
    """数据获取结果

    Attributes
    ----------
    success : bool
        操作是否成功
    query : str
        查询关键词
    source : str
        数据来源（"ChEMBL" / "PubChem"）
    compounds : List[CompoundRecord]
        获取到的化合物列表
    total_count : int
        有效化合物数量
    error : Optional[str]
        失败时的错误信息
    fetch_time : float
        获取耗时（秒）
    """
    success: bool
    query: str
    source: str
    compounds: List[CompoundRecord] = field(default_factory=list)
    total_count: int = 0
    error: Optional[str] = None
    fetch_time: float = 0.0


class DataFetcher:
    """整合 ChEMBL 和 PubChem 的数据获取工具

    提供从公开数据库获取化合物及活性数据的能力，支持：
    - 按靶点名称从 ChEMBL 检索活性化合物
    - 按 SMILES 相似性从 PubChem 检索类似化合物

    Examples
    --------
    >>> fetcher = DataFetcher()
    >>> result = fetcher.fetch_by_target("EGFR", max_compounds=50)
    >>> for comp in result.compounds:
    ...     print(comp.smiles, comp.activity_value)
    """

    def __init__(self):
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": (
                "Mozilla/5.0 (compatible; AiEgfrPlatform/2.0; "
                "+https://ai-egfr-platform.streamlit.app)"
            )
        })

        if CHEMBL_AVAILABLE:
            self.targets_api = new_client.target
            self.compounds_api = new_client.molecule
            self.bioactivities_api = new_client.activity
        else:
            self.targets_api = None
            self.compounds_api = None
            self.bioactivities_api = None

    # --------------------- ChEMBL 相关 ---------------------
    def fetch_by_target(
        self,
        target_name: str,
        activity_type: str = "IC50",
        max_compounds: int = 100,
        min_pic50: Optional[float] = None,
        organism: str = "Homo sapiens"
    ) -> FetchResult:
        """按靶点名称从 ChEMBL 获取化合物及活性数据

        Parameters
        ----------
        target_name : str
            靶点名称（如 "EGFR", "BRCA1"）
        activity_type : str
            活性数据类型（IC50 / EC50 / Ki / Kd）
        max_compounds : int
            最大返回化合物数量
        min_pic50 : Optional[float]
            最小 pIC50 阈值（None 表示不过滤）
        organism : str
            物种筛选，默认 "Homo sapiens"

        Returns
        -------
        FetchResult
        """
        if not CHEMBL_AVAILABLE:
            return FetchResult(
                success=False, query=target_name, source="ChEMBL",
                error="chembl_webresource_client 库未安装，请运行: pip install chembl_webresource_client"
            )

        start_time = time.time()
        try:
            # 1. 查询靶点
            targets = self.targets_api.get(
                pref_name__icontains=target_name,
                organism=organism
            ).only("target_chembl_id", "pref_name", "organism", "target_type")
            targets_df = pd.DataFrame.from_records(targets)
            if targets_df.empty:
                return FetchResult(
                    success=False,
                    query=target_name,
                    source="ChEMBL",
                    error=f"未找到靶点: {target_name} (organism={organism})"
                )

            # 优先选择 SINGLE PROTEIN 类型
            if "target_type" in targets_df.columns:
                target_row = targets_df[targets_df["target_type"] == "SINGLE PROTEIN"]
                if target_row.empty:
                    target_row = targets_df.iloc[[0]]
            else:
                target_row = targets_df.iloc[[0]]

            target_chembl_id = target_row.iloc[0]["target_chembl_id"]
            pref_name = target_row.iloc[0]["pref_name"]
            logger.info(f"找到靶点: {pref_name} ({target_chembl_id})")

            # 2. 查询生物活性
            # 注意: chembl_webresource_client 的 QuerySet 会自动分页遍历全部结果，
            # 底层 ChEMBL API 每页默认返回 100 条，通过 offset 自动翻页直到数据耗尽。
            # 转换为 DataFrame 时会消费全部数据。
            bioactivities = self.bioactivities_api.filter(
                target_chembl_id=target_chembl_id,
                type=activity_type,
                relation="=",
                assay_type="B",
                target_organism=organism
            ).only(
                "activity_id",
                "molecule_chembl_id",
                "standard_value",
                "standard_units",
                "standard_type",
                "relation"
            )
            bio_df = pd.DataFrame.from_records(bioactivities)
            raw_count = len(bio_df)
            logger.info(f"ChEMBL 原始生物活性数据: {raw_count} 条")

            if bio_df.empty:
                return FetchResult(
                    success=False,
                    query=target_name,
                    source="ChEMBL",
                    error=f"未找到 {activity_type} 活性数据"
                )

            # 3. 预处理：只保留 nM 单位，转换标准值，计算 pIC50
            bio_df = bio_df[bio_df["standard_units"] == "nM"]
            logger.info(f"过滤 nM 单位后: {len(bio_df)} 条")

            if bio_df.empty:
                return FetchResult(
                    success=False,
                    query=target_name,
                    source="ChEMBL",
                    error=f"没有 nM 单位的 {activity_type} 数据，请尝试其他活性类型"
                )

            bio_df = bio_df.astype({"standard_value": "float64"})
            bio_df.dropna(subset=["standard_value"], inplace=True)

            # 去重（每个分子保留第一条记录）
            bio_df.drop_duplicates("molecule_chembl_id", keep="first", inplace=True)
            logger.info(f"去重后: {len(bio_df)} 条")

            if min_pic50 is not None:
                # 计算 pIC50 = 9 - log10(IC50_nM)
                bio_df["pIC50"] = bio_df["standard_value"].apply(
                    lambda x: 9 - math.log10(x) if x > 0 else float("-inf")
                )
                before_pic50_filter = len(bio_df)
                bio_df = bio_df[bio_df["pIC50"] >= min_pic50]
                logger.info(
                    f"pIC50 >= {min_pic50} 过滤后: {before_pic50_filter} -> {len(bio_df)} 条"
                )

            # 在去重和 pIC50 过滤之后再截断，确保返回的是高质量的去重结果
            bio_df = bio_df.head(max_compounds)
            logger.info(f"截断至最多 {max_compounds} 条后: {len(bio_df)} 条")

            # 4. 获取分子 SMILES
            chembl_ids = bio_df["molecule_chembl_id"].tolist()
            if not chembl_ids:
                if min_pic50 is not None:
                    detail = (
                        f"pIC50 >= {min_pic50} 过滤后无符合条件的分子"
                        f"（共处理 {raw_count} 条原始数据，"
                        f"尝试降低 pIC50 阈值或检查靶点名称）"
                    )
                else:
                    detail = "没有符合条件的分子"
                return FetchResult(
                    success=False,
                    query=target_name,
                    source="ChEMBL",
                    error=detail
                )

            # 批量获取分子结构
            compounds = self.compounds_api.filter(
                molecule_chembl_id__in=chembl_ids
            ).only("molecule_chembl_id", "molecule_structures")
            comp_list = list(compounds)
            comp_df = pd.DataFrame.from_records(comp_list)

            # 提取规范 SMILES
            comp_df["smiles"] = comp_df["molecule_structures"].apply(
                lambda x: x.get("canonical_smiles") if isinstance(x, dict) else None
            )
            comp_df = comp_df[["molecule_chembl_id", "smiles"]]
            comp_df.dropna(subset=["smiles"], inplace=True)

            # 5. 合并
            merged = pd.merge(
                bio_df, comp_df,
                on="molecule_chembl_id", how="inner"
            )
            if merged.empty:
                return FetchResult(
                    success=False,
                    query=target_name,
                    source="ChEMBL",
                    error="未能获取分子结构数据"
                )

            # 6. 构建结果
            records: List[CompoundRecord] = []
            for _, row in merged.iterrows():
                pic50_val = row.get("pIC50", None)
                records.append(CompoundRecord(
                    smiles=row["smiles"],
                    chembl_id=row["molecule_chembl_id"],
                    activity_value=(
                        float(pic50_val) if pic50_val is not None
                        else float(row.get("standard_value", 0))
                    ),
                    activity_type=activity_type,
                    target_name=pref_name,
                    source="ChEMBL",
                    raw_data=row.to_dict()
                ))

            elapsed = time.time() - start_time
            logger.info(f"ChEMBL 检索完成: {len(records)} 条结果, 耗时 {elapsed:.1f}s")

            return FetchResult(
                success=True,
                query=target_name,
                source="ChEMBL",
                compounds=records,
                total_count=len(records),
                fetch_time=elapsed
            )

        except Exception as e:
            logger.error(f"ChEMBL 数据获取失败: {e}", exc_info=True)
            return FetchResult(
                success=False,
                query=target_name,
                source="ChEMBL",
                error=str(e)
            )

    # --------------------- PubChem 相关 ---------------------
    def fetch_similar_by_smiles(
        self,
        smiles: str,
        threshold: int = 90,
        max_records: int = 20
    ) -> FetchResult:
        """使用 PubChem 相似性搜索获取类似化合物

        采用异步任务模式：先提交相似性搜索任务获取 ListKey，
        然后轮询获取 CID 列表，最后批量获取 SMILES。

        Parameters
        ----------
        smiles : str
            查询分子的 SMILES 字符串
        threshold : int
            相似度阈值（70-100），默认 90
        max_records : int
            最大返回化合物数量，默认 20

        Returns
        -------
        FetchResult
        """
        start_time = time.time()
        try:
            # 1. 创建异步任务
            escaped = quote(smiles)
            url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"similarity/smiles/{escaped}/JSON"
                f"?Threshold={threshold}&MaxRecords={max_records}"
            )
            r = self.session.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()

            if "Waiting" not in data:
                return FetchResult(
                    success=False,
                    query=smiles[:50],
                    source="PubChem",
                    error="未获得异步任务 key"
                )
            list_key = data["Waiting"]["ListKey"]

            # 2. 轮询获取结果
            poll_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"listkey/{list_key}/cids/JSON"
            )
            attempts = 30  # 最多等待 60 秒
            cids: List[int] = []
            while attempts > 0:
                resp = self.session.get(poll_url, timeout=30)
                resp.raise_for_status()
                resp_data = resp.json()
                if "IdentifierList" in resp_data:
                    cids = resp_data["IdentifierList"]["CID"]
                    break
                attempts -= 1
                time.sleep(2)

            if not cids:
                return FetchResult(
                    success=False,
                    query=smiles[:50],
                    source="PubChem",
                    error="未找到相似化合物或轮询超时"
                )

            # 3. 批量获取 SMILES
            cid_str = ",".join(map(str, cids[:max_records]))
            prop_url = (
                f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/"
                f"cid/{cid_str}/property/CanonicalSMILES/JSON"
            )
            prop_resp = self.session.get(prop_url, timeout=30)
            prop_resp.raise_for_status()
            prop_data = prop_resp.json()

            props = prop_data.get("PropertyTable", {}).get("Properties", [])
            records: List[CompoundRecord] = []
            for item in props:
                s = item.get("CanonicalSMILES")
                if s:
                    records.append(CompoundRecord(
                        smiles=s,
                        pubchem_cid=str(item.get("CID", "")),
                        source="PubChem"
                    ))

            elapsed = time.time() - start_time
            logger.info(f"PubChem 相似性搜索完成: {len(records)} 条结果, 耗时 {elapsed:.1f}s")

            return FetchResult(
                success=True,
                query=smiles[:50],
                source="PubChem",
                compounds=records,
                total_count=len(records),
                fetch_time=elapsed
            )

        except requests.RequestException as e:
            logger.error(f"PubChem 网络请求失败: {e}")
            return FetchResult(
                success=False,
                query=smiles[:50],
                source="PubChem",
                error=f"网络请求失败: {e}"
            )
        except Exception as e:
            logger.error(f"PubChem 相似性搜索失败: {e}", exc_info=True)
            return FetchResult(
                success=False,
                query=smiles[:50],
                source="PubChem",
                error=str(e)
            )
