# utils/shap_utils.py
"""
SHAP 模型解释性分析工具

使用 SHAP (SHapley Additive exPlanations) TreeExplainer 来
解释随机森林模型对 EGFR 抑制剂活性的预测。

核心原理：
- SHAP 值基于博弈论中的 Shapley 值，将模型预测分解为各特征的贡献
- TreeExplainer 利用树结构直接计算精确 SHAP 值，无需采样
- 正值：该特征推高活性预测 → 有利于结合
- 负值：该特征拉低活性预测 → 不利于结合

由于平台不分发原始训练数据，本模块工作方式如下：
1. TreeExplainer 从树模型结构直接提取期望值（base value）
2. 对用户输入的分子计算其描述符向量
3. 输出该分子的 SHAP 瀑布图 + 全局条形图
"""

import logging
import os
import subprocess
import sys
import threading

# 必须在导入 numpy 之前设置：允许 MKL(Intel OpenMP) 与 numba(LLVM OpenMP)
# 两个 OpenMP 运行时共存，避免 OMP Error #15 → 原生崩溃 0xC06D007F
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

# Matplotlib 中文字体设置
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================
# SHAP 可用性探测（Windows 原生崩溃隔离）
# ============================================================
# 背景：Windows 下 `import shap` 可能因 BLAS/DLL 冲突触发原生崩溃
# 0xC06D007F（延迟加载 DLL 失败）。该异常发生在 C/C++ 层，无法被
# Python 的 try/except 捕获，会直接终止整个进程。
# 因此必须在【子进程】中先探测 shap 是否可导入——子进程崩溃只影响
# 子进程本身，从而保证主应用进程永不因 shap 崩溃。
_shap_ready = None  # None=未探测, True/False=探测结果缓存
_shap_lock = threading.Lock()


def _probe_shap(timeout=60):
    """在子进程中尝试 import shap，返回是否可用（隔离原生崩溃）。"""
    code = "import shap; print('SHAP_PROBE_OK')"
    kwargs = {}
    if os.name == "nt":
        # 避免崩溃时弹出 Windows 错误报告窗口
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    try:
        p = subprocess.run(
            [sys.executable, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            **kwargs,
        )
        ok = p.returncode == 0 and "SHAP_PROBE_OK" in p.stdout
    except Exception:
        ok = False
    if not ok:
        logger.warning(
            "SHAP 子进程探测失败（可能为 BLAS/DLL 冲突），"
            "将自动降级为 RF 原生特征重要性展示"
        )
    return ok


def is_shap_available(timeout=60):
    """判断 shap 是否可在当前进程安全导入（结果缓存 + 后台预探测）。"""
    global _shap_ready
    if _shap_ready is None:
        with _shap_lock:
            if _shap_ready is None:
                _shap_ready = _probe_shap(timeout)
    return _shap_ready


def _start_background_probe():
    """应用启动后即开始后台探测，避免首次使用时阻塞 UI。"""

    def _run():
        try:
            is_shap_available()
        except Exception:
            pass

    threading.Thread(target=_run, daemon=True).start()


_start_background_probe()


def get_shap_explainer(model):
    """
    为一个 sklearn RandomForestClassifier 创建 SHAP TreeExplainer。

    TreeExplainer 专为树模型优化，比 KernelExplainer 快 10-100 倍。
    不需要训练数据作为背景——期望值直接从树结构中提取。

    注意：Windows 下 import shap 可能触发不可捕获的原生崩溃，
    调用前请先通过 is_shap_available() 确认可用性。

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        已训练好的随机森林模型

    Returns
    -------
    shap.TreeExplainer
    """
    if not is_shap_available():
        raise RuntimeError(
            "SHAP 在当前环境不可用（Windows BLAS/DLL 兼容问题），"
            "请使用 plot_feature_importance_fallback 降级展示"
        )
    import shap

    return shap.TreeExplainer(model)


def compute_shap_for_sample(
    explainer, features: np.ndarray, feature_names: list
):
    """
    对一组样本计算 SHAP 值。

    Parameters
    ----------
    explainer : shap.TreeExplainer
    features : np.ndarray, shape (n_samples, n_features)
        分子描述符矩阵
    feature_names : list[str]

    Returns
    -------
    dict
        {
            "shap_values":    np.ndarray, 正类 SHAP 值 (n_samples, n_features)
            "base_value":     float,      模型期望值（base value）
            "feature_names":  list[str],
            "feature_values": np.ndarray,
        }
    """
    # 确保特征维度正确
    if features.ndim == 1:
        features = features.reshape(1, -1)

    # 计算 SHAP 值
    raw_shap = explainer.shap_values(features)

    # shap 版本差异处理：
    # - 旧版：list（二分类 [neg, pos]）或 2D ndarray (n_samples, n_features)
    # - 新版 (shap>=0.46)：3D ndarray (n_samples, n_features, n_classes)
    if isinstance(raw_shap, list):
        shap_vals = raw_shap[1]  # 正类 = 活性分子
        base_val = (
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )
    elif raw_shap.ndim == 3:
        # (n_samples, n_features, n_classes) → 取正类（最后一列）
        shap_vals = raw_shap[..., -1]
        ev = explainer.expected_value
        if isinstance(ev, (list, np.ndarray)) and np.ndim(ev) > 0:
            base_val = float(np.asarray(ev).reshape(-1)[-1])
        else:
            base_val = float(ev)
    else:
        shap_vals = raw_shap
        base_val = (
            explainer.expected_value
            if not isinstance(explainer.expected_value, (list, np.ndarray))
            else float(np.mean(explainer.expected_value))
        )

    return {
        "shap_values": shap_vals,
        "base_value": float(base_val),
        "feature_names": feature_names,
        "feature_values": features,
    }


def plot_shap_waterfall(shap_result: dict, sample_idx: int = 0):
    """
    绘制单个分子的 SHAP 瀑布图。

    瀑布图直观展示从基准预测到最终预测的逐步贡献：
    - 基准线 (base value): 模型在不了解该分子时的平均预测
    - 红色条: 正向贡献（推高活性预测）
    - 蓝色条: 负向贡献（拉低活性预测）
    - 最终线 (f(x)): 模型对该分子的实际预测

    Parameters
    ----------
    shap_result : dict
        compute_shap_for_sample() 的返回值
    sample_idx : int
        解释第几个样本（默认第一个）

    Returns
    -------
    matplotlib.figure.Figure
    """
    import shap

    shap_vals = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    base_value = shap_result["base_value"]
    feature_vals = shap_result["feature_values"]

    if isinstance(shap_vals, np.ndarray) and shap_vals.ndim == 2:
        sv = shap_vals[sample_idx]
    else:
        sv = shap_vals

    if isinstance(feature_vals, np.ndarray) and feature_vals.ndim == 2:
        fv = feature_vals[sample_idx]
    else:
        fv = feature_vals

    fig, ax = plt.subplots(figsize=(8, len(feature_names) * 0.35 + 1))

    shap.waterfall_plot(
        shap.Explanation(
            values=sv,
            base_values=base_value,
            data=fv,
            feature_names=feature_names,
        ),
        show=False,
    )
    plt.tight_layout()
    return fig


def plot_shap_bar(shap_result: dict, top_n: int = None):
    """
    绘制 SHAP 特征重要性条形图。

    按 |SHAP| 均值降序排列，展示对活性预测影响最大的 Top-N 描述符。
    单个分子时为绝对值排序。

    Parameters
    ----------
    shap_result : dict
    top_n : int or None
        显示 top N 个特征（None 表示全部）

    Returns
    -------
    matplotlib.figure.Figure
    """
    shap_vals = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]

    if shap_vals.ndim == 1:
        shap_vals = shap_vals.reshape(1, -1)

    # 均值绝对 SHAP 值
    mean_abs = np.mean(np.abs(shap_vals), axis=0)
    n_all = len(feature_names)
    if top_n is None:
        top_n = min(12, n_all)

    # 排序取 Top-N
    idx_sorted = np.argsort(mean_abs)[::-1]
    idx_top = idx_sorted[:top_n]

    top_names = [feature_names[i] for i in idx_top]
    top_values = [mean_abs[i] for i in idx_top]

    # 颜色：按方向
    mean_raw = np.mean(shap_vals, axis=0)
    colors = ["#e74c3c" if mean_raw[i] > 0 else "#3498db" for i in idx_top]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    # 不要双重反转：barh 从下往上绘制，配合 invert_yaxis 让最重要的特征显示在最上方
    ax.barh(range(top_n), top_values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("mean(|SHAP value|)")
    ax.set_title("SHAP Feature Importance (EGFR Activity)")
    ax.axvline(x=0, color="gray", linewidth=0.5)
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def plot_feature_importance_fallback(model, feature_names, top_n=None):
    """
    SKlearn 原生特征重要性（Gini importance）降级展示。

    当 SHAP 因 Windows BLAS/DLL 兼容问题不可用时，使用随机森林自带的
    feature_importances_ 提供替代的特征贡献分析，保证解释性功能始终可用。

    Parameters
    ----------
    model : sklearn.ensemble.RandomForestClassifier
        已训练好的随机森林模型
    feature_names : list[str]
        描述符名称列表
    top_n : int or None
        显示 top N 个特征（None 表示最多显示 12 个）

    Returns
    -------
    matplotlib.figure.Figure or None
        模型无可解释特征重要性时返回 None
    """
    importances = getattr(model, "feature_importances_", None)
    if importances is None or importances.size == 0:
        return None
    n_all = len(feature_names)
    if top_n is None:
        top_n = min(12, n_all)
    idx = np.argsort(importances)[::-1][:top_n]
    top_names = [feature_names[i] for i in idx]
    top_values = [float(importances[i]) for i in idx]

    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.35)))
    colors = plt.cm.YlOrRd(np.linspace(0.25, 0.9, top_n))
    ax.barh(range(top_n), top_values, color=colors)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(top_names)
    ax.set_xlabel("Gini importance (RF feature_importances_)")
    ax.set_title("特征重要性（RF Gini）· SHAP 不可用时的降级展示")
    ax.invert_yaxis()
    plt.tight_layout()
    return fig


def format_shap_insights(shap_result: dict, sample_idx: int = 0) -> str:
    """
    将 SHAP 结果转化为人类可读的文字摘要。

    Parameters
    ----------
    shap_result : dict
    sample_idx : int

    Returns
    -------
    str
        中文摘要文本
    """
    shap_vals = shap_result["shap_values"]
    feature_names = shap_result["feature_names"]
    feature_vals = shap_result["feature_values"]
    base_value = shap_result["base_value"]

    if shap_vals.ndim == 2:
        sv = shap_vals[sample_idx]
        fv = feature_vals[sample_idx]
    else:
        sv = shap_vals
        fv = feature_vals

    # TreeExplainer 的 shap_values/expected_value 默认在 log-odds 空间，
    # 需先经 sigmoid 转概率再与 0.5 阈值比较
    logit = base_value + float(np.sum(sv))
    final_pred = 1 / (1 + np.exp(-logit))
    pred_label = "活性的可能性较高" if final_pred > 0.5 else "活性的可能性较低"

    # Top 3 正向和负向贡献（特征数可能不足 3，需截断防越界）
    n_feat = len(sv)
    idx_pos = np.argsort(sv)[::-1][:min(3, n_feat)]
    idx_neg = np.argsort(sv)[:min(3, n_feat)]

    lines = [
        f"**SHAP 解释摘要** (基准 logit = {base_value:.3f})",
        f"- 最终预测概率 = **{final_pred:.3f}** → 该分子被预测为 **{pred_label}**",
    ]

    top_contributions = sv[idx_pos]
    if top_contributions[0] > 0:
        lines.append("\n**主要正向贡献（推高活性预测）：**")
        for i in idx_pos:
            if sv[i] > 0:
                lines.append(
                    f"  - `{feature_names[i]}` = {fv[i]:.3g}  "
                    f"(贡献 +{sv[i]:.4f})"
                )

    # 无负向贡献时不输出空标题
    if np.any(sv[idx_neg] < 0):
        lines.append("")
        lines.append("**主要负向贡献（拉低活性预测）：**")
        for i in idx_neg:
            if sv[i] < 0:
                lines.append(
                    f"  - `{feature_names[i]}` = {fv[i]:.3g}  "
                    f"(贡献 {sv[i]:.4f})"
                )

    return "\n".join(lines)
