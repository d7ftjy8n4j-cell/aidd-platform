# utils/knime_export_utils.py
"""
KNIME 工作流导出工具 - 将平台数据导出为 KNIME 兼容格式
参考: TeachOpenCADD-KNIME (https://hub.knime.com/volkamerlab/space/TeachOpenCADD)
"""

import os
import json
import io
import zipfile
import logging
from typing import Dict, Optional, Any
from datetime import datetime

import pandas as pd

# 列名映射: 平台内部名 → KNIME 兼容名 (与 TeachOpenCADD-KNIME W1-W8 对齐)
COLUMN_MAPPING = {
    "smiles": "SMILES", "canonical": "Canonical_SMILES",
    "name": "Molecule_Name", "mol_id": "Molecule_ID",
    "activity": "pIC50", "prediction": "Predicted_Activity",
    "probability": "Activity_Probability",
    "rf_pred": "RF_Prediction", "gnn_pred": "GNN_Prediction",
    "rf_prob": "RF_Probability", "gnn_prob": "GNN_Probability",
    "lipinski_pass": "Lipinski_Pass", "lipinski_violations": "Lipinski_Violations",
    "pains_pass": "PAINS_Pass",
    "mw": "Molecular_Weight", "logp": "LogP",
    "hbd": "Num_H_Donors", "hba": "Num_H_Acceptors",
    "rotatable_bonds": "Num_Rotatable_Bonds", "tpsa": "TPSA",
    "docking_score": "Vina_Score", "docking_rank": "Docking_Rank",
    "cluster": "Cluster_ID", "is_center": "Is_Cluster_Center",
    "similarity": "Tanimoto_Similarity", "target_smiles": "Query_SMILES",
    "mmgbsa_dg": "MMGBSA_dG_Bind", "rmsd_mean": "RMSD_Mean",
    "rmsd_std": "RMSD_Std",
}

WORKFLOW_SUGGESTIONS = {
    "docking": "W8 (蛋白-配体相互作用分析)",
    "prediction": "W7 (机器学习 / QSAR 建模)",
    "screening": "W2 (ADME/类药性) + W3 (有害子结构过滤)",
    "clustering": "W5 (化合物聚类)",
    "similarity": "W4 (化合物相似性搜索)",
    "mmgbsa": "W8 (蛋白-配体结合能分析)",
    "general": "W1-W8 (根据分析目标选择)",
}


def _infer_module(df: pd.DataFrame) -> str:
    """根据列名推断数据来源"""
    cols = set(df.columns)
    for key in ["docking_score", "mmgbsa_dg", "rmsd_mean", "prediction",
                 "lipinski_pass", "cluster", "similarity"]:
        if key in cols:
            return {
                "docking_score": "docking", "mmgbsa_dg": "mmgbsa",
                "rmsd_mean": "md", "prediction": "prediction",
                "lipinski_pass": "screening", "cluster": "clustering",
                "similarity": "similarity",
            }[key]
    return "general"


class KNIMEExporter:
    """KNIME 工作流导出器"""

    def __init__(self, data: pd.DataFrame, metadata: Optional[Dict] = None):
        if data is None or data.empty:
            raise ValueError("导出数据为空")
        self.data = data.copy()
        self.metadata = metadata or {}
        self.module_type = _infer_module(self.data)
        self._normalize_smiles_col()

    def _normalize_smiles_col(self):
        """自动检测并重命名 SMILES 列"""
        for name in ["smiles", "canonical", "SMILES", "Canonical_SMILES"]:
            if name in self.data.columns and name != "smiles":
                self.data.rename(columns={name: "smiles"}, inplace=True)
                return
            elif name in self.data.columns:
                return
        for col in self.data.columns:
            if "smil" in col.lower():
                self.data.rename(columns={col: "smiles"}, inplace=True)
                return
        raise ValueError(f"未找到 SMILES 列，可用列: {list(self.data.columns)[:12]}")

    def to_csv_bytes(self) -> bytes:
        """导出为 KNIME 兼容 CSV (内存字节)"""
        rename = {c: COLUMN_MAPPING[c] for c in self.data.columns if c in COLUMN_MAPPING}
        return self.data.rename(columns=rename).to_csv(index=False).encode("utf-8")

    def to_workflow_zip(self, workflow_name: str = "药尘光_分析结果") -> bytes:
        """打包完整工作流为 ZIP"""
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.writestr(f"{workflow_name}_data.csv", self.to_csv_bytes())
            zf.writestr(f"{workflow_name}_metadata.json",
                        json.dumps(self._build_metadata(workflow_name),
                                   indent=2, ensure_ascii=False))
            zf.writestr(f"{workflow_name}.knime_flow",
                        json.dumps(self._build_knime_desc(workflow_name),
                                   indent=2, ensure_ascii=False))
        buf.seek(0)
        return buf.getvalue()

    def _build_metadata(self, name: str) -> Dict:
        return {
            "workflow_name": name,
            "description": f"药尘光 2.0 {self.module_type} 分析结果",
            "export_time": datetime.now().isoformat(),
            "source_platform": "药尘光 2.0 (my-egfr-v2)",
            "module_type": self.module_type,
            "teachopencadd_knime_hub": "https://hub.knime.com/volkamerlab/space/TeachOpenCADD",
            "data_shape": list(self.data.shape),
            "columns": list(self.data.columns),
            "recommended_workflow": WORKFLOW_SUGGESTIONS.get(
                self.module_type, WORKFLOW_SUGGESTIONS["general"]),
            "user_metadata": self.metadata,
        }

    def _build_knime_desc(self, name: str) -> Dict:
        return {
            "format_version": "2.1",
            "name": name,
            "description": f"药尘光 2.0 导出的 {self.module_type} 数据",
            "compatible_knime": ["4.3+", "5.0+"],
            "teachopencadd_ref": "https://hub.knime.com/volkamerlab/space/TeachOpenCADD",
            "recommended_workflow": WORKFLOW_SUGGESTIONS.get(
                self.module_type, WORKFLOW_SUGGESTIONS["general"]),
            "import_steps": [
                "1. 在 KNIME 中新建工作流",
                f"2. 拖入 File Reader 节点，加载 {name}_data.csv",
                "3. 使用 KNIME RDKit 节点进行分子指纹计算",
                "4. 参考推荐工作流进行后续分析",
            ],
        }
