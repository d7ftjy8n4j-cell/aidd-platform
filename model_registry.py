"""model_registry.py — 模型性能指标单一事实源（SSOT）
====================================================

本文件是 README / app.py UI / gnn_predictor.py 运行时三处显示指标的**唯一来源**。

纪律（P0-08，DECISIONS D-16）：
1. 任何数值只能来自 ``artifacts/*_metrics.json``——它们由 ``scripts/`` 下的训练脚本
   一条命令重跑得出，含数据 sha256 指纹，协议与逐折明细见 ``docs/MODEL_CARD.md``。
2. 改这里的数字之前，必须先重跑训练脚本、更新 artifacts/，再同步本文件与
   ``tests/test_model_registry.py`` 的对账单测（单测会在 artifacts 缺失时跳过）。
3. 历史声称值（首次训练、无产物出处）一律不得写回本文件；它们只允许出现在
   MODEL_CARD 的「原始 vs 复现」对照列里。

当前协议（两套模型同协议同 split，可直接逐折比较，见 EXP-023）：
    数据集 A：ChEMBL CHEMBL203 IC50，13,286 条，median split（pIC50 ≥ 中位数 7.0），
    5 折 StratifiedKFold(shuffle=True, random_state=42)。
"""

# ---- 随机森林（scripts/train_rf_legacy_protocol.py → artifacts/rf_legacy_protocol_metrics.json）----
RF_PERF = {
    "auc": 0.8680,               # cv_results.roc_auc.mean = 0.86798379
    "auc_std": 0.0050,           # cv_results.roc_auc.std  = 0.00495039
    "accuracy": 0.7851,          # cv_results.accuracy.mean = 0.78511172
    "accuracy_std": 0.0073,      # cv_results.accuracy.std  = 0.00730843
    "balanced_accuracy": 0.7845,  # cv_results.balanced_accuracy.mean
    "mcc": 0.5712,               # cv_results.mcc.mean
    "feature_count": 12,         # feature_names.json（12 个 RDKit 描述符）
    "n_trees": 200,              # n_estimators（pkl 逆向 + 复现脚本一致）
}

# ---- GNN/GCN（scripts/train_gnn.py --dataset legacy → artifacts/gnn_metrics_legacy.json，EXP-019）----
GNN_PERF = {
    "auc": 0.8517,               # cv_results.roc_auc.mean = 0.85168...
    "auc_std": 0.0061,           # cv_results.roc_auc.std
    "accuracy": 0.7719,          # cv_results.accuracy.mean
    "accuracy_std": 0.0061,      # cv_results.accuracy.std
    "balanced_accuracy": 0.7714,  # cv_results.balanced_accuracy.mean
    "mcc": 0.5463,               # cv_results.mcc.mean
    "node_features": "13维",
}

# 原始训练的声称值（来源：gcn_cv_results.csv / README 初版）。
# 仅作对照展示用，不是本 registry 的权威值；权威值是上面两个 dict。
ORIGINAL_CLAIMS = {
    "rf_auc": 0.867, "rf_accuracy": 0.782,
    "gnn_auc": 0.845, "gnn_accuracy": 0.767,
}
