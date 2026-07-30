# utils/uncertainty_utils.py
"""
模型预测不确定性估计

通过集成方法 (Ensemble) 量化预测的可信度：
- 利用随机森林内部的 n_estimators 棵决策树
- 每棵树独立预测，树间方差 = 模型不确定性
- 标准差小 → 多棵树达成共识 → 预测可信
- 标准差大 → 树间分歧严重 → 预测需谨慎

注意：这是模型层面的**认知不确定性** (epistemic uncertainty)，
反映模型对该特征空间的"熟悉程度"，而非实验误差。
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


def predict_with_uncertainty(model, features: np.ndarray):
    """
    使用随机森林的每棵决策树独立预测，量化不确定性。

    不重新训练模型，直接利用已有的 `.estimators_` 属性。

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        已训练的随机森林模型
    features : np.ndarray, shape (n_samples, n_features)
        分子描述符矩阵

    Returns
    -------
    dict
        {
            "mean_proba": np.ndarray,   # 平均活性概率
            "std_proba":  np.ndarray,   # 预测标准差
            "tree_probas": np.ndarray,  # 每棵树的预测 (n_trees, n_samples)
            "n_trees": int,
        }
    """
    if not hasattr(model, "estimators_"):
        raise ValueError("模型没有 .estimators_ 属性，不是 sklearn RandomForest")

    n_trees = len(model.estimators_)
    n_samples = features.shape[0] if features.ndim == 2 else 1

    if features.ndim == 1:
        features = features.reshape(1, -1)

    tree_probas = np.zeros((n_trees, n_samples))
    for i, tree in enumerate(model.estimators_):
        tree_probas[i, :] = tree.predict_proba(features)[:, 1]

    mean_proba = np.mean(tree_probas, axis=0)
    std_proba = np.std(tree_probas, axis=0)

    return {
        "mean_proba": mean_proba,
        "std_proba": std_proba,
        "tree_probas": tree_probas,
        "n_trees": n_trees,
    }


def get_confidence_level(std_proba: float) -> tuple:
    """
    根据标准差判断预测置信度。

    Parameters
    ----------
    std_proba : float
        预测概率的标准差 (0~0.5)

    Returns
    -------
    (level: str, icon: str, message: str)
    """
    if std_proba < 0.05:
        return (
            "高",
            "✅",
            "各棵决策树预测高度一致，分子落于模型「熟悉」的特征空间，"
            "预测结果非常可信。",
        )
    elif std_proba < 0.10:
        return (
            "中",
            "⚠️",
            "模型内部存在一定分歧，分子可能处于训练数据的边界区域，"
            "建议结合其他证据谨慎参考。",
        )
    else:
        return (
            "低",
            "❌",
            "多棵决策树预测结果分歧显著，分子可能偏离训练数据分布较远，"
            "强烈建议通过实验加以验证。",
        )


def plot_uncertainty_distribution(unc_result: dict, sample_idx: int = 0):
    """
    绘制每棵树预测概率的分布直方图。

    Parameters
    ----------
    unc_result : dict
        predict_with_uncertainty() 的返回值
    sample_idx : int

    Returns
    -------
    matplotlib.figure.Figure
    """
    probas = unc_result["tree_probas"][:, sample_idx]
    mean_val = unc_result["mean_proba"][sample_idx]
    std_val = unc_result["std_proba"][sample_idx]
    n_trees = unc_result["n_trees"]

    level, icon, _ = get_confidence_level(std_val)

    fig, ax = plt.subplots(figsize=(7, 3.5))

    ax.hist(
        probas, bins=max(15, n_trees // 8),
        color="#2ecc71", alpha=0.7, edgecolor="#27ae60", linewidth=0.5,
    )
    ax.axvline(
        mean_val, color="#e74c3c", linewidth=2.5,
        label=f"Mean = {mean_val:.3f}  ± {std_val:.3f}",
    )
    ax.axvline(
        mean_val - std_val, color="#e74c3c", linestyle="--", linewidth=1,
        alpha=0.5,
    )
    ax.axvline(
        mean_val + std_val, color="#e74c3c", linestyle="--", linewidth=1,
        alpha=0.5,
    )

    ax.set_xlabel("Predicted Activity Probability")
    ax.set_ylabel(f"# Trees (out of {n_trees})")
    ax.set_title(f"Tree Prediction Distribution  ({icon} Confidence: {level})")
    ax.legend(loc="upper right")
    plt.tight_layout()
    return fig


def format_uncertainty_summary(unc_result: dict, smiles_label: str = "") -> str:
    """
    生成不确定性评估的中文摘要。

    Parameters
    ----------
    unc_result : dict
    smiles_label : str
        分子标签（可选）

    Returns
    -------
    str
    """
    mean_val = unc_result["mean_proba"][0]
    std_val = unc_result["std_proba"][0]
    n_trees = unc_result["n_trees"]
    level, icon, msg = get_confidence_level(std_val)

    if smiles_label:
        header = f"**不确定性评估：`{smiles_label}`**"
    else:
        header = "**不确定性评估**"

    return "\n".join([
        header,
        "",
        f"- 平均活性概率：**{mean_val:.4f}** ± {std_val:.4f}",
        f"- 参与决策的树：{n_trees} 棵",
        f"- 置信度等级：{icon} **{level}**",
        "",
        msg,
        "",
        f"*计算方式：随机森林 {n_trees} 棵决策树各自预测活性概率，"
        f"以标准差衡量树间共识程度。*",
    ])
