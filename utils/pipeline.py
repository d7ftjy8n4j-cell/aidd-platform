# utils/pipeline.py
"""
自动化药物发现流程编排器
依次执行：双模型预测 → ADME/Ro5筛选 → 不良子结构筛查 → 药效团匹配 → 相似性搜索
"""

import sys
import os
import logging
import traceback
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from dataclasses import dataclass, field, asdict


@dataclass
class SingleMoleculeResult:
    """单个分子的完整分析结果（dataclass）
    
    提供结构化类型提示，便于 IDE 代码补全和后续维护。
    可通过 asdict() 转为普通字典，或直接使用属性访问。
    """
    smiles: str
    """分子 SMILES 字符串"""
    timestamp: str = ""
    """分析时间戳 (ISO 8601)"""
    steps: Dict[str, Any] = field(default_factory=dict)
    """各步骤详细结果: {'rf': {...}, 'gnn': {...}, ...}"""
    summary: Dict[str, Any] = field(default_factory=dict)
    """汇总判定: {'rf_consensus': ..., 'final_verdict': ..., ...}"""
    errors: List[str] = field(default_factory=list)
    """执行过程中的错误信息列表"""
    success: bool = True
    """整体流程是否成功（所有启用的步骤均无致命错误）"""
    
    def to_dict(self) -> Dict[str, Any]:
        """转为 JSON 兼容的字典"""
        return asdict(self)
    
    def __getitem__(self, key: str) -> Any:
        """支持 dict-style 访问，兼容旧代码"""
        return getattr(self, key)
    
    def get(self, key: str, default: Any = None) -> Any:
        """支持 dict-style .get()，兼容旧代码"""
        return getattr(self, key, default)

# 将父目录加入路径，以便导入根目录下的模块
_parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _parent_dir not in sys.path:
    sys.path.insert(0, _parent_dir)

from real_predictor import RealEGFRPredictor
from chem_filter import ADMEFilter, SubstructureFilter

logger = logging.getLogger(__name__)

# ---------- 可选模块 ----------
GCN_AVAILABLE = True
try:
    from gnn_predictor import GCNPredictor
except Exception as e:
    GCN_AVAILABLE = False
    logger.warning(f"GNN预测器不可用: {e}")

PHARMACOPHORE_AVAILABLE = True
try:
    from pharmacophore_streamlit import StreamlitPharmacophore
except Exception as e:
    PHARMACOPHORE_AVAILABLE = False
    logger.warning(f"药效团模块不可用: {e}")

SIMILARITY_AVAILABLE = True
try:
    from chem_insight_safe import SafeChemInsightEngine
except Exception as e:
    SIMILARITY_AVAILABLE = False
    logger.warning(f"相似性搜索模块不可用: {e}")


class Pipeline:
    """自动化药物发现流程编排器
    
    将双模型预测、ADME筛选、不良子结构、药效团匹配和相似性搜索
    组合为一个可配置的执行流程。
    """
    
    def __init__(self):
        # 各组件懒加载
        self._rf_predictor: Optional[RealEGFRPredictor] = None
        self._gnn_predictor = None           # GCNPredictor or None
        self._adme_filter: Optional[ADMEFilter] = None
        self._substructure_filter: Optional[SubstructureFilter] = None
        self._pharmacophore = None            # StreamlitPharmacophore or None
        self._similarity_engine = None        # SafeChemInsightEngine or None
        self._reference_smiles: Optional[List[str]] = None
    
    # ---------------- 懒加载方法 ----------------
    def _ensure_rf(self) -> RealEGFRPredictor:
        if self._rf_predictor is None:
            self._rf_predictor = RealEGFRPredictor()
            logger.info("随机森林预测器已初始化")
        return self._rf_predictor
    
    def _ensure_gnn(self):
        if not GCN_AVAILABLE:
            return None
        if self._gnn_predictor is None:
            self._gnn_predictor = GCNPredictor(device='cpu')
            logger.info("GNN预测器已初始化")
        return self._gnn_predictor
    
    def _ensure_adme(self) -> ADMEFilter:
        if self._adme_filter is None:
            self._adme_filter = ADMEFilter()
        return self._adme_filter
    
    def _ensure_substructure(self) -> SubstructureFilter:
        if self._substructure_filter is None:
            self._substructure_filter = SubstructureFilter()
            logger.info("子结构筛选器已初始化")
        return self._substructure_filter
    
    def _ensure_pharmacophore(self):
        if not PHARMACOPHORE_AVAILABLE:
            return None
        if self._pharmacophore is None:
            self._pharmacophore = StreamlitPharmacophore()
            logger.info("药效团分析器已初始化")
        return self._pharmacophore
    
    def _ensure_similarity(self):
        if not SIMILARITY_AVAILABLE:
            return None
        if self._similarity_engine is None:
            self._similarity_engine = SafeChemInsightEngine()
            logger.info("相似性搜索引擎已初始化")
        return self._similarity_engine
    
    # ---------------- 单分子流程 ----------------
    def run_single_molecule(
        self,
        smiles: str,
        enable_rf: bool = True,
        enable_gnn: bool = True,
        enable_adme: bool = True,
        enable_substructure: bool = True,
        enable_pharmacophore: bool = False,
        enable_similarity: bool = False,
    ) -> SingleMoleculeResult:
        """对单个 SMILES 执行完整分析流程
        
        Parameters
        ----------
        smiles : str
            分子的 SMILES 字符串
        enable_rf, enable_gnn, enable_adme, enable_substructure,
        enable_pharmacophore, enable_similarity : bool
            各步骤开关
        
        Returns
        -------
        SingleMoleculeResult
            包含 steps（每步详细结果）、summary（汇总判定）、
            errors（错误列表）、success（整体成功标志）的结构化结果
        """
        steps: Dict[str, Any] = {}
        summary: Dict[str, Any] = {}
        errors: List[str] = []
        
        # ---- 步骤1：随机森林预测 ----
        if enable_rf:
            try:
                rf = self._ensure_rf()
                rf_res = rf.predict(smiles)
                if "error" in rf_res:
                    error_msg = f"RF预测返回错误: {rf_res['error']}"
                    steps["rf"] = {"error": rf_res["error"]}
                    summary["rf_consensus"] = "Error"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                else:
                    steps["rf"] = {
                        "prediction": rf_res.get("prediction"),
                        "probability": rf_res.get("probability_active"),
                        "confidence": rf_res.get("confidence"),
                        "prediction_label": "Active" if rf_res.get("prediction") == 1 else "Inactive",
                    }
                    summary["rf_consensus"] = (
                        "Active" if rf_res.get("prediction") == 1 else "Inactive"
                    )
                    logger.info(f"RF预测完成: {summary['rf_consensus']}")
            except Exception as e:
                error_msg = f"RF预测异常: {e.__class__.__name__}: {e}"
                steps["rf"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["rf_consensus"] = "Error"
                errors.append(error_msg)
                logger.error(f"RF预测异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 步骤2：GNN预测 ----
        if enable_gnn:
            try:
                gnn = self._ensure_gnn()
                if gnn is None:
                    error_msg = "GNN预测器不可用（模块导入失败）"
                    steps["gnn"] = {"error": error_msg}
                    summary["gnn_consensus"] = "N/A"
                    errors.append(error_msg)
                    logger.warning(error_msg)
                else:
                    gnn_res = gnn.predict(smiles)
                    if not gnn_res.get("success", True):
                        error_msg = f"GNN预测失败: {gnn_res.get('error', '未知错误')}"
                        steps["gnn"] = {"error": error_msg}
                        summary["gnn_consensus"] = "Error"
                        errors.append(error_msg)
                        logger.warning(error_msg)
                    else:
                        steps["gnn"] = {
                            "prediction": gnn_res.get("prediction"),
                            "probability": gnn_res.get("probability_active"),
                            "confidence": gnn_res.get("confidence"),
                            "prediction_label": "Active" if gnn_res.get("prediction") == 1 else "Inactive",
                        }
                        summary["gnn_consensus"] = (
                            "Active" if gnn_res.get("prediction") == 1 else "Inactive"
                        )
                        logger.info(f"GNN预测完成: {summary['gnn_consensus']}")
            except Exception as e:
                error_msg = f"GNN预测异常: {e.__class__.__name__}: {e}"
                steps["gnn"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["gnn_consensus"] = "Error"
                errors.append(error_msg)
                logger.error(f"GNN预测异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 步骤3：ADME/Ro5 ----
        if enable_adme:
            try:
                adme = self._ensure_adme()
                ro5_series = adme.calculate_ro5_properties(smiles)
                # 将 pd.Series 转为普通字典
                adme_dict = {
                    "MW": float(ro5_series.get("MW", 0)) if ro5_series.get("MW") is not None else None,
                    "HBA": int(ro5_series.get("HBA", 0)) if ro5_series.get("HBA") is not None else None,
                    "HBD": int(ro5_series.get("HBD", 0)) if ro5_series.get("HBD") is not None else None,
                    "LogP": float(ro5_series.get("LogP", 0)) if ro5_series.get("LogP") is not None else None,
                    "Pass_Ro5": bool(ro5_series.get("Pass_Ro5", False)),
                }
                steps["adme"] = adme_dict
                summary["adme_pass"] = adme_dict["Pass_Ro5"]
                logger.info(f"ADME完成: Pass_Ro5={adme_dict['Pass_Ro5']}")
            except Exception as e:
                error_msg = f"ADME计算异常: {e.__class__.__name__}: {e}"
                steps["adme"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["adme_pass"] = False
                errors.append(error_msg)
                logger.error(f"ADME异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 步骤4：不良子结构 ----
        if enable_substructure:
            try:
                subst = self._ensure_substructure()
                subst_res = subst.check_single_molecule(smiles)
                has_pains = subst_res.get("PAINS_found", False)
                has_brenk = subst_res.get("Brenk_found", False)
                steps["substructure"] = {
                    "PAINS_found": has_pains,
                    "PAINS_names": subst_res.get("PAINS_names", []),
                    "Brenk_found": has_brenk,
                    "Brenk_names": subst_res.get("Brenk_names", []),
                }
                summary["substructure_pass"] = not (has_pains or has_brenk)
                logger.info(f"子结构筛查完成: PAINS={has_pains}, Brenk={has_brenk}")
            except Exception as e:
                error_msg = f"子结构筛查异常: {e.__class__.__name__}: {e}"
                steps["substructure"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["substructure_pass"] = False
                errors.append(error_msg)
                logger.error(f"子结构筛查异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 步骤5：药效团匹配 ----
        if enable_pharmacophore:
            try:
                pharm = self._ensure_pharmacophore()
                if pharm is None:
                    error_msg = "药效团模块不可用（模块导入失败）"
                    steps["pharmacophore"] = {"error": error_msg}
                    summary["pharmacophore_matched"] = False
                    errors.append(error_msg)
                    logger.warning(error_msg)
                else:
                    # StreamlitPharmacophore 需要加载分子后提取特征
                    n_loaded = pharm.load_molecules_from_smiles([smiles], names=["Query"])
                    if n_loaded > 0:
                        features = pharm.extract_pharmacophore_features()
                        pharm_dict = {
                            "matched": len(features) > 0 and len(features[0]) > 0,
                            "feature_count": len(features[0]) if features else 0,
                            "features": [
                                {"type": f.get("type"), "strength": f.get("strength")}
                                for f in (features[0] if features else [])
                            ],
                        }
                        steps["pharmacophore"] = pharm_dict
                        summary["pharmacophore_matched"] = pharm_dict["matched"]
                        logger.info(f"药效团匹配完成: {pharm_dict['feature_count']} 个特征")
                    else:
                        error_msg = "无法解析分子构象"
                        steps["pharmacophore"] = {"error": error_msg}
                        summary["pharmacophore_matched"] = False
                        errors.append(error_msg)
            except Exception as e:
                error_msg = f"药效团匹配异常: {e.__class__.__name__}: {e}"
                steps["pharmacophore"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["pharmacophore_matched"] = False
                errors.append(error_msg)
                logger.error(f"药效团匹配异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 步骤6：相似性搜索 ----
        if enable_similarity:
            try:
                sim_engine = self._ensure_similarity()
                if sim_engine is None:
                    error_msg = "相似性搜索模块不可用（模块导入失败）"
                    steps["similarity"] = {"error": error_msg}
                    summary["similarity_count"] = 0
                    errors.append(error_msg)
                    logger.warning(error_msg)
                else:
                    similar = sim_engine.safe_find_similar_compounds(smiles, top_n=5)
                    sim_list = []
                    for s in similar:
                        sim_list.append({
                            "smiles": s.get("smiles", ""),
                            "similarity": round(s.get("similarity", 0), 4),
                            "name": s.get("name", ""),
                            "is_active": s.get("is_active", False),
                        })
                    steps["similarity"] = {
                        "count": len(sim_list),
                        "results": sim_list,
                    }
                    summary["similarity_count"] = len(sim_list)
                    logger.info(f"相似性搜索完成: {len(sim_list)} 个相似分子")
            except Exception as e:
                error_msg = f"相似性搜索异常: {e.__class__.__name__}: {e}"
                steps["similarity"] = {"error": error_msg, "traceback": traceback.format_exc()}
                summary["similarity_count"] = 0
                errors.append(error_msg)
                logger.error(f"相似性搜索异常 traceback:\n{traceback.format_exc()}")
        
        # ---- 综合判定：先构建结果对象 ----
        result = SingleMoleculeResult(
            smiles=smiles,
            timestamp=datetime.now().isoformat(),
            steps=steps,
            summary=summary,
            errors=errors,
        )
        
        # 判断双模型一致性
        rf_pred = steps.get("rf", {}).get("prediction")
        gnn_pred = steps.get("gnn", {}).get("prediction")
        rf_ok = steps.get("rf", {}).get("success", rf_pred is not None)
        gnn_ok = steps.get("gnn", {}).get("success", gnn_pred is not None)
        # ADME/子结构：仅当步骤实际执行（无 error 键）时才算作“通过关卡”
        adme_ran = "error" not in steps.get("adme", {})
        subst_ran = "error" not in steps.get("substructure", {})
        adme_pass = summary.get("adme_pass", False) if adme_ran else True
        subst_pass = summary.get("substructure_pass", False) if subst_ran else True

        if enable_rf and enable_gnn:
            if rf_pred is not None and gnn_pred is not None:
                if rf_pred == gnn_pred:
                    if rf_pred == 1 and adme_pass and subst_pass:
                        result.summary["final_verdict"] = "✅ 推荐候选分子"
                    elif rf_pred == 1:
                        result.summary["final_verdict"] = "⚠️ 活性但成药性不佳"
                    else:
                        result.summary["final_verdict"] = "❌ 非活性"
                else:
                    result.summary["final_verdict"] = "⚡ 双模型结果分歧，需人工判断"
            else:
                result.summary["final_verdict"] = "⚠️ 部分模型预测失败，无法综合判定"
        elif enable_rf and not enable_gnn:
            if rf_pred is None or not rf_ok:
                result.summary["final_verdict"] = "⚠️ RF预测失败，无法判定"
            elif rf_pred == 1 and adme_pass and subst_pass:
                result.summary["final_verdict"] = "✅ RF预测活性 + 成药性通过"
            elif rf_pred == 1:
                result.summary["final_verdict"] = "⚠️ RF预测活性但成药性不佳"
            else:
                result.summary["final_verdict"] = "❌ RF预测非活性"
        elif enable_gnn and not enable_rf:
            if gnn_pred is None or not gnn_ok:
                result.summary["final_verdict"] = "⚠️ GNN预测失败，无法判定"
            elif gnn_pred == 1 and adme_pass and subst_pass:
                result.summary["final_verdict"] = "✅ GNN预测活性 + 成药性通过"
            elif gnn_pred == 1:
                result.summary["final_verdict"] = "⚠️ GNN预测活性但成药性不佳"
            else:
                result.summary["final_verdict"] = "❌ GNN预测非活性"
        else:
            result.summary["final_verdict"] = "未启用任何预测模型"
        
        # 整体成功：所有启用步骤均无致命错误
        result.success = len(errors) == 0
        
        logger.info(f"最终判定: {result.summary['final_verdict']}")
        return result
    
    # ---------------- 批量处理 ----------------
    def run_batch(self, smiles_list: List[str], **kwargs) -> List[SingleMoleculeResult]:
        """批量处理多个 SMILES
        
        Parameters
        ----------
        smiles_list : List[str]
            SMILES 字符串列表
        **kwargs : 
            传递给 run_single_molecule 的步骤开关
        
        Returns
        -------
        List[Dict]
            每个分子的分析结果列表
        """
        results = []
        for smi in smiles_list:
            res = self.run_single_molecule(smi, **kwargs)
            results.append(res)
        return results
    
    # ---------------- 报告生成 ----------------
    def generate_report(self, result: SingleMoleculeResult, format: str = "markdown") -> str:
        """生成可读报告
        
        Parameters
        ----------
        result : dict
            run_single_molecule 的返回结果
        format : str
            "markdown" 或 "text"
        
        Returns
        -------
        str
            格式化报告文本
        """
        if format == "markdown":
            return self._generate_markdown_report(result)
        else:
            return self._generate_text_report(result)
    
    def _generate_markdown_report(self, result: SingleMoleculeResult) -> str:
        lines = []
        lines.append(f"# 自动化药物发现报告")
        lines.append(f"**SMILES**: `{result['smiles']}`")
        lines.append(f"**时间**: {result['timestamp']}")
        lines.append("")
        lines.append("---")
        lines.append("")
        
        # 汇总
        lines.append("## 🏷️ 综合判定")
        lines.append(f"**{result['summary'].get('final_verdict', 'N/A')}**")
        lines.append("")
        
        # 各步骤
        lines.append("## 📋 步骤详情")
        lines.append("")
        
        step_labels = {
            "rf": "🌲 随机森林预测",
            "gnn": "🧠 GNN预测",
            "adme": "💊 ADME/Ro5 筛选",
            "substructure": "⚠️ 不良子结构筛查",
            "pharmacophore": "🎯 药效团匹配",
            "similarity": "🔍 相似性搜索",
        }
        
        for step, data in result["steps"].items():
            label = step_labels.get(step, step.upper())
            lines.append(f"### {label}")
            if "error" in data:
                lines.append(f"❌ 错误: {data['error']}")
            elif step == "rf" or step == "gnn":
                lines.append(f"- 预测: {data.get('prediction_label', 'N/A')}")
                prob = data.get('probability')
                if prob is not None:
                    lines.append(f"- 活性概率: {prob:.4f}")
                lines.append(f"- 置信度: {data.get('confidence', 'N/A')}")
            elif step == "adme":
                lines.append(f"- 分子量 (MW): {data.get('MW', 'N/A')}")
                lines.append(f"- LogP: {data.get('LogP', 'N/A')}")
                lines.append(f"- 氢键受体 (HBA): {data.get('HBA', 'N/A')}")
                lines.append(f"- 氢键供体 (HBD): {data.get('HBD', 'N/A')}")
                lines.append(f"- Ro5通过: {'✅' if data.get('Pass_Ro5') else '❌'}")
            elif step == "substructure":
                lines.append(f"- PAINS: {'⚠️ 发现' if data.get('PAINS_found') else '✅ 通过'} {data.get('PAINS_names', [])}")
                lines.append(f"- Brenk: {'⚠️ 发现' if data.get('Brenk_found') else '✅ 通过'} {data.get('Brenk_names', [])}")
            elif step == "pharmacophore":
                lines.append(f"- 匹配: {'✅' if data.get('matched') else '❌'}")
                lines.append(f"- 特征数: {data.get('feature_count', 0)}")
            elif step == "similarity":
                lines.append(f"- 相似分子数: {data.get('count', 0)}")
                for sim in data.get("results", []):
                    lines.append(f"  - `{sim.get('smiles', '')}` (相似度: {sim.get('similarity', 0):.3f})")
            lines.append("")
        
        lines.append("---")
        lines.append(f"*报告由药尘光自动化流程生成于 {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*")
        
        return "\n".join(lines)
    
    def _generate_text_report(self, result: SingleMoleculeResult) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("自动化药物发现报告")
        lines.append("=" * 60)
        lines.append(f"SMILES: {result['smiles']}")
        lines.append(f"时间: {result['timestamp']}")
        lines.append(f"综合判定: {result['summary'].get('final_verdict', 'N/A')}")
        lines.append("-" * 60)
        for step, data in result["steps"].items():
            lines.append(f"\n[{step.upper()}]")
            for k, v in data.items():
                lines.append(f"  {k}: {v}")
        lines.append("=" * 60)
        return "\n".join(lines)
    
    # ---------------- 汇总DataFrame ----------------
    def results_to_dataframe(self, results: List[SingleMoleculeResult]) -> pd.DataFrame:
        """将批量结果转为汇总 DataFrame
        
        Parameters
        ----------
        results : List[Dict]
            run_batch 的返回结果
        
        Returns
        -------
        pd.DataFrame
        """
        rows = []
        for r in results:
            steps = r.get("steps", {})
            summary = r.get("summary", {})
            
            # RF 概率
            rf_data = steps.get("rf", {})
            rf_prob = rf_data.get("probability") if "error" not in rf_data else None
            
            # GNN 概率
            gnn_data = steps.get("gnn", {})
            gnn_prob = gnn_data.get("probability") if "error" not in gnn_data else None
            
            rows.append({
                "SMILES": r["smiles"],
                "RF预测": summary.get("rf_consensus", "N/A"),
                "RF活性概率": f"{rf_prob:.4f}" if rf_prob is not None else "N/A",
                "GNN预测": summary.get("gnn_consensus", "N/A"),
                "GNN活性概率": f"{gnn_prob:.4f}" if gnn_prob is not None else "N/A",
                "ADME通过": "✅" if summary.get("adme_pass") else "❌" if summary.get("adme_pass") is False else "N/A",
                "子结构通过": "✅" if summary.get("substructure_pass") else "❌" if summary.get("substructure_pass") is False else "N/A",
                "最终判定": summary.get("final_verdict", "N/A"),
            })
        return pd.DataFrame(rows)
