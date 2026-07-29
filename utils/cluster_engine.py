"""
分子聚类引擎 - 基于 Butina 算法

对化合物库进行结构聚类，识别化学空间中的骨架多样性，
支持代表性分子筛选和降维可视化。
"""

from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass, field
from rdkit import Chem, DataStructs
from rdkit.ML.Cluster import Butina
from rdkit.Chem import rdFingerprintGenerator
import logging

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """单个聚类结果

    Attributes
    ----------
    cluster_id : int
        簇编号（从 0 开始）
    member_indices : List[int]
        簇成员在原始数据集中的索引
    centroid_index : int
        簇中心分子索引
    size : int
        簇大小
    intra_similarities : List[float]
        簇内各分子与中心的 Tanimoto 相似度
    representative_mols : List[Chem.Mol]
        簇的代表性分子（最多 10 个）
    """
    cluster_id: int
    member_indices: List[int]
    centroid_index: int
    size: int = 0
    intra_similarities: List[float] = field(default_factory=list)
    representative_mols: List[Chem.Mol] = field(default_factory=list)


@dataclass
class ClusteringSummary:
    """聚类汇总结果

    Attributes
    ----------
    n_clusters : int
        簇总数
    n_singletons : int
        单例簇数量（大小为 1 的孤立分子）
    cluster_sizes : List[int]
        各簇大小列表（按从大到小排序）
    largest_cluster_size : int
        最大簇的大小
    results : List[ClusterResult]
        所有聚类结果
    """
    n_clusters: int
    n_singletons: int
    cluster_sizes: List[int]
    largest_cluster_size: int
    results: List[ClusterResult]


class ClusterEngine:
    """基于 Butina 算法的分子聚类引擎

    Butina 算法是一种高效的层次聚类方法，专为化学空间设计。
    它使用 Tanimoto 距离（1 - Tanimoto 相似度）作为距离度量，
    通过阈值 cutoff 控制簇的紧密度。

    Parameters
    ----------
    fingerprint_type : str
        指纹类型，"morgan"（Morgan/Circular）或 "rdkit"（拓扑指纹）
    radius : int
        Morgan 指纹半径，默认 2
    n_bits : int
        指纹位长，默认 2048

    Examples
    --------
    >>> engine = ClusterEngine()
    >>> mols = [Chem.MolFromSmiles(s) for s in smiles_list]
    >>> summary = engine.cluster(mols, cutoff=0.3)
    >>> print(f"共 {summary.n_clusters} 个簇")
    """

    def __init__(
        self,
        fingerprint_type: str = "morgan",
        radius: int = 2,
        n_bits: int = 2048
    ):
        self.fingerprint_type = fingerprint_type
        self.radius = radius
        self.n_bits = n_bits
        self._fingerprint_generator = self._get_generator()

    def _get_generator(self):
        """获取指纹生成器"""
        if self.fingerprint_type == "morgan":
            return rdFingerprintGenerator.GetMorganGenerator(
                radius=self.radius, fpSize=self.n_bits
            )
        else:
            return rdFingerprintGenerator.GetRDKitFPGenerator(
                maxPath=5, fpSize=self.n_bits
            )

    def _tanimoto_distance_matrix(self, fp_list: List) -> List[float]:
        """计算距离矩阵（上三角展开列表）

        Butina 算法期望的输入格式：扁平化的上三角距离列表。
        D = [1 - Tanimoto(fp_i, fp_j) for i > j]

        Parameters
        ----------
        fp_list : List
            指纹列表

        Returns
        -------
        List[float]
            扁平化的上三角距离列表，长度为 n*(n-1)/2
        """
        distance_matrix: List[float] = []
        n = len(fp_list)
        for i in range(1, n):
            similarities = DataStructs.BulkTanimotoSimilarity(
                fp_list[i], fp_list[:i]
            )
            distance_matrix.extend([1.0 - x for x in similarities])
        logger.debug(
            f"距离矩阵: {len(distance_matrix)} 个元素 (n={n})"
        )
        return distance_matrix

    def cluster(
        self,
        molecules: List[Chem.Mol],
        cutoff: float = 0.3,
        ids: Optional[List[str]] = None
    ) -> ClusteringSummary:
        """执行 Butina 聚类

        处理流程：
        1. 为所有分子计算分子指纹
        2. 计算成对 Tanimoto 距离矩阵
        3. 使用 Butina 算法进行聚类
        4. 计算簇内统计并构建结果

        Parameters
        ----------
        molecules : List[Chem.Mol]
            RDKit Mol 对象列表
        cutoff : float
            距离阈值，簇内分子到中心的距离需 < cutoff
            对应相似度 > 1 - cutoff
        ids : Optional[List[str]]
            分子标识符列表，默认为 mol_0, mol_1, ...

        Returns
        -------
        ClusteringSummary
            包含聚类统计和每个簇的详细信息
        """
        if ids is None:
            ids = [f"mol_{i}" for i in range(len(molecules))]

        n_molecules = len(molecules)
        logger.info(f"开始聚类: {n_molecules} 个分子, cutoff={cutoff}")

        # 1. 计算指纹
        fps = [
            self._fingerprint_generator.GetFingerprint(mol)
            for mol in molecules
        ]

        # 2. 计算距离矩阵
        distance_matrix = self._tanimoto_distance_matrix(fps)

        # 3. Butina 聚类
        # ClusterData(data, nPts, distThresh, isDistData=False)
        clusters = Butina.ClusterData(
            distance_matrix, n_molecules, cutoff, isDistData=True
        )
        clusters = sorted(clusters, key=len, reverse=True)
        logger.info(f"Butina 聚类完成: {len(clusters)} 个簇")

        # 4. 构建结果对象
        results: List[ClusterResult] = []
        for cid, cluster_indices in enumerate(clusters):
            # 簇中心 = 列表第一个元素（Butina 算法的约定）
            centroid_idx = cluster_indices[0]
            centroid_fp = fps[centroid_idx]

            # 计算簇内各成员与中心的相似度
            intra_sim: List[float] = []
            for idx in cluster_indices[1:]:
                sim = DataStructs.TanimotoSimilarity(centroid_fp, fps[idx])
                intra_sim.append(sim)

            # 取前 10 个代表分子
            rep_mols = [
                molecules[i] for i in cluster_indices[:10]
            ]

            results.append(ClusterResult(
                cluster_id=cid,
                member_indices=cluster_indices,
                centroid_index=centroid_idx,
                size=len(cluster_indices),
                intra_similarities=intra_sim,
                representative_mols=rep_mols,
            ))

        singletons = sum(1 for r in results if r.size == 1)
        return ClusteringSummary(
            n_clusters=len(results),
            n_singletons=singletons,
            cluster_sizes=[r.size for r in results],
            largest_cluster_size=results[0].size if results else 0,
            results=results,
        )

    def get_cluster_centers(
        self,
        summary: ClusteringSummary,
        molecules: List[Chem.Mol]
    ) -> List[Chem.Mol]:
        """返回每个簇的中心分子"""
        return [molecules[r.centroid_index] for r in summary.results]

    def get_representative_subset(
        self,
        summary: ClusteringSummary,
        molecules: List[Chem.Mol],
        max_compounds: int = 1000
    ) -> Tuple[List[int], List[Chem.Mol]]:
        """从每个簇中挑选代表性分子

        策略：
        1. 先选中所有簇中心（保证覆盖全部化学空间）
        2. 剩余额度按簇大小分配，每个簇取最多 10 个与中心最相似的成员
        3. 总量不超过 max_compounds

        Parameters
        ----------
        summary : ClusteringSummary
            聚类结果
        molecules : List[Chem.Mol]
            原始分子列表
        max_compounds : int
            最大返回数量，默认 1000

        Returns
        -------
        Tuple[List[int], List[Chem.Mol]]
            (选中索引列表, 选中分子列表)
        """
        selected_indices: List[int] = []
        selected_mols: List[Chem.Mol] = []

        # 先选所有簇中心
        for r in summary.results:
            selected_indices.append(r.centroid_index)
            selected_mols.append(molecules[r.centroid_index])

        remaining = max_compounds - len(selected_indices)
        if remaining <= 0:
            return selected_indices[:max_compounds], selected_mols[:max_compounds]

        # 按簇排序（已在 cluster() 中按大小排好）
        for r in summary.results:
            if remaining <= 0:
                break

            # 排除中心
            others = [i for i in r.member_indices if i != r.centroid_index]
            if not others:
                continue

            # 对非中心成员按与中心的相似度排序
            centroid_fp = self._fingerprint_generator.GetFingerprint(
                molecules[r.centroid_index]
            )
            others_sim: List[Tuple[float, int]] = []
            for idx in others:
                sim = DataStructs.TanimotoSimilarity(
                    centroid_fp,
                    self._fingerprint_generator.GetFingerprint(molecules[idx])
                )
                others_sim.append((sim, idx))

            # 相似度从高到低
            others_sim.sort(reverse=True, key=lambda x: x[0])

            # 每个簇最多 10 个
            take = min(10, len(others_sim), remaining)
            for _, idx in others_sim[:take]:
                selected_indices.append(idx)
                selected_mols.append(molecules[idx])

            remaining -= take

        logger.info(
            f"代表性筛选: {len(summary.results)} 个簇 -> "
            f"{len(selected_indices)} 个代表分子"
        )
        return selected_indices, selected_mols
