# mcs_utils.py
"""
最大公共子结构 (MCS) 工具模块

基于 RDKit rdFMCS 实现，用于识别一组分子的核心骨架。
"""

from typing import List, Optional, Tuple

import streamlit as st
from rdkit import Chem
from rdkit.Chem import rdFMCS, Draw


def compute_mcs(
    smiles_list: List[str],
    threshold: float = 1.0,
    ring_matches_ring: bool = True,
    match_valences: bool = False,
    timeout: int = 30,
) -> Tuple[Optional[str], Optional[int], Optional[int], Optional[str]]:
    """
    计算一组分子的最大公共子结构 (MCS)。

    Parameters
    ----------
    smiles_list : List[str]
        SMILES 字符串列表
    threshold : float
        分子必须共享 MCS 的最小比例 (0-1)，值越小 MCS 越大
    ring_matches_ring : bool
        环是否必须匹配环（避免芳香环匹配到脂肪环）
    match_valences : bool
        是否匹配化合价
    timeout : int
        超时时间（秒）

    Returns
    -------
    (smarts, num_atoms, num_bonds, error_msg)
    """
    # 转换 SMILES → Mol
    mols = []
    invalid = []
    for i, smi in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            invalid.append(i)
        else:
            mols.append(mol)

    if invalid:
        return None, None, None, f"无效 SMILES（索引 {invalid}）"

    if len(mols) < 2:
        return None, None, None, "至少需要 2 个有效分子"

    try:
        mcs_result = rdFMCS.FindMCS(
            mols,
            threshold=threshold,
            ringMatchesRingOnly=ring_matches_ring,
            matchValences=match_valences,
            timeout=timeout,
        )

        if mcs_result is None or mcs_result.numAtoms == 0:
            return None, None, None, "未找到公共子结构（尝试降低 threshold）"

        return (
            mcs_result.smartsString,
            mcs_result.numAtoms,
            mcs_result.numBonds,
            None,
        )

    except Exception as e:
        return None, None, None, f"MCS 计算失败: {str(e)}"


def highlight_mcs_in_molecules(
    smiles_list: List[str],
    mcs_smarts: str,
    size: int = 350,
):
    """
    在每个分子中高亮显示 MCS 子结构。

    Returns
    -------
    list of PIL.Image
    """
    mols = [
        Chem.MolFromSmiles(smi)
        for smi in smiles_list
        if Chem.MolFromSmiles(smi) is not None
    ]
    if not mols:
        return []

    query = Chem.MolFromSmarts(mcs_smarts)
    if query is None:
        return []

    images = []
    for mol in mols:
        matches = mol.GetSubstructMatches(query)
        highlight_atoms = list(matches[0]) if matches else []
        img = Draw.MolToImage(mol, highlightAtoms=highlight_atoms, size=(size, size))
        images.append(img)

    return images


def get_mcs_smarts_as_mol(mcs_smarts: str):
    """将 MCS SMARTS 转换为 RDKit Mol 对象并生成 2D 渲染图"""
    try:
        mol = Chem.MolFromSmarts(mcs_smarts)
        if mol:
            return Draw.MolToImage(mol, size=(400, 200))
    except Exception:
        pass
    return None
