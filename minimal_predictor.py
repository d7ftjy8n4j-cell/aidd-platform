"""minimal_predictor.py — 最简兜底预测器（降级链最后一层）
====================================================

定位（P0-03，DECISIONS D-17）
----------------------------
当前两层（真实模型 → 规则基线）都不可用时才启用的**确定性启发式打分器**。
它**不是机器学习模型**，也不含任何随机数——同一条 SMILES 任何时候、任何进程
重启后得到的都是同一个数。

诚实性约定：
1. 结果必须带 ``predictor_tier="minimal"`` 与醒目的 ``note``，UI 据此向学生声明
   「这不是模型预测」。
2. 禁止在此层引入 ``hash()``/随机数造概率（F-08 历史教训：Python 对 str 的
   ``hash()`` 每个进程带盐，会导致同一分子重启后预测值漂移）。

仅依赖标准库，保证在 numpy / RDKit / torch 全部缺失的极端环境下仍可用。
"""

# 层级标签（供 UI 显示与测试断言）
TIER_REAL = "real"
TIER_FALLBACK = "fallback"
TIER_MINIMAL = "minimal"


class MinimalEGFRPredictor:
    """确定性启发式打分（非 ML）。规则透明、可解释、可复现。"""

    def __init__(self):
        self.predictor_tier = TIER_MINIMAL
        self.feature_names = ["SMILES长度", "碳原子数", "氮原子数", "氧原子数"]

    def predict(self, smiles):
        if not smiles or len(smiles) < 2:
            return {"error": "无效的SMILES字符串"}

        length = len(smiles)
        c_count = smiles.count('C')
        n_count = smiles.count('N')
        o_count = smiles.count('O')

        score = 0.5
        if 30 <= length <= 80:
            score += 0.15
        if n_count >= 2:
            score += 0.15
        if o_count >= 1:
            score += 0.10
        probability = max(0.1, min(0.9, score))

        return {
            "success": True,
            "smiles": smiles,
            "prediction": 1 if probability > 0.5 else 0,
            "probability_active": probability,
            "confidence": "中",
            "predictor_tier": TIER_MINIMAL,
            "explanation": {
                "top_features": ["SMILES长度", "氮原子数", "氧原子数"],
                "top_importance": [0.5, 0.3, 0.2],
                "values": {"SMILES长度": length, "氮原子数": n_count, "氧原子数": o_count}
            },
            "features_used": self.feature_names,
            "feature_values": [length, c_count, n_count, o_count],
            "note": "最简兜底：确定性启发式规则打分，非机器学习模型，结果仅供界面演示，不具科研效力",
        }
