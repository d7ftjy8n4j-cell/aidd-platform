"""pytest 单测 —— model_registry.py 指标对账（P0-08）。

目的
----
保证「README / app.py UI / gnn_predictor.py 运行时」三处显示的模型指标
与 ``artifacts/*_metrics.json`` 里的复现实测值一致——防止再出现
0.8695 / 0.8628 / 0.8081 那样无产出处、互相矛盾的孤立数字（F-03/04/05）。

artifacts 缺失时跳过（例如云端精简部署），但只要 artifacts 在，registry 就不许漂移。

运行（仓库根目录）::

    set KMP_DUPLICATE_LIB_OK=TRUE
    python.exe -m pytest tests/test_model_registry.py -q
"""

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import model_registry  # noqa: E402

ARTIFACTS = REPO_ROOT / "artifacts"
TOL = 1e-3  # registry 存 4 位小数，允许舍入误差


def _load(name: str) -> dict:
    path = ARTIFACTS / name
    if not path.exists():
        pytest.skip(f"artifacts/{name} 不存在（精简部署？），对账跳过")
    return json.loads(path.read_text(encoding="utf-8"))


def test_rf_registry_matches_artifact():
    j = _load("rf_legacy_protocol_metrics.json")
    cv = j["cv_results"]
    assert j["data"]["n_rows"] == 13286
    assert abs(model_registry.RF_PERF["auc"] - cv["roc_auc"]["mean"]) < TOL
    assert abs(model_registry.RF_PERF["auc_std"] - cv["roc_auc"]["std"]) < TOL
    assert abs(model_registry.RF_PERF["accuracy"] - cv["accuracy"]["mean"]) < TOL
    assert abs(model_registry.RF_PERF["accuracy_std"] - cv["accuracy"]["std"]) < TOL
    assert abs(model_registry.RF_PERF["balanced_accuracy"] - cv["balanced_accuracy"]["mean"]) < TOL
    assert abs(model_registry.RF_PERF["mcc"] - cv["mcc"]["mean"]) < TOL


def test_gnn_registry_matches_artifact():
    j = _load("gnn_metrics_legacy.json")
    cv = j["cv_results"]
    assert j["data"]["graph_n"] == 13286
    assert abs(model_registry.GNN_PERF["auc"] - cv["roc_auc"]["mean"]) < TOL
    assert abs(model_registry.GNN_PERF["auc_std"] - cv["roc_auc"]["std"]) < TOL
    assert abs(model_registry.GNN_PERF["accuracy"] - cv["accuracy"]["mean"]) < TOL
    assert abs(model_registry.GNN_PERF["accuracy_std"] - cv["accuracy"]["std"]) < TOL
    assert abs(model_registry.GNN_PERF["balanced_accuracy"] - cv["balanced_accuracy"]["mean"]) < TOL
    assert abs(model_registry.GNN_PERF["mcc"] - cv["mcc"]["mean"]) < TOL


def test_registry_values_sane():
    """指标必须落在合理区间，防止手滑写错小数点。"""
    for perf in (model_registry.RF_PERF, model_registry.GNN_PERF):
        assert 0.5 < perf["auc"] < 1.0
        assert 0.5 < perf["accuracy"] < 1.0
        assert 0.0 < perf["auc_std"] < 0.1
        assert 0.0 < perf["mcc"] < 1.0
    assert model_registry.RF_PERF["feature_count"] == 12
    assert model_registry.RF_PERF["n_trees"] == 200


def test_original_claims_unchanged():
    """历史声称值只允许原样保存在 ORIGINAL_CLAIMS 作对照，禁止冒充权威值。"""
    assert model_registry.ORIGINAL_CLAIMS == {
        "rf_auc": 0.867, "rf_accuracy": 0.782,
        "gnn_auc": 0.845, "gnn_accuracy": 0.767,
    }
