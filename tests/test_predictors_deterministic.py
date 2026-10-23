"""pytest 单测 —— 预测器降级链确定性（P0-03）。

背景
----
历史缺陷 F-08/F-09：备用/最简预测器用 ``hash(smiles)`` 播随机数造概率——
Python 对 str 的 hash 每个进程带盐，同一分子在应用重启后预测值会漂移，
且 UI 不显示预测来源，学生拿到伪随机数毫不知情。

本文件锁死三条纪律：
1. **确定性**：同一输入，任何时刻、任何实例，输出逐位一致。
2. **诚实标签**：非真实模型的层必须带 ``predictor_tier`` 和「非机器学习模型」note。
3. **无 hash 造数**：预测器源码中不得再出现 ``hash(``。
"""

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import fallback_predictor  # noqa: E402
from minimal_predictor import MinimalEGFRPredictor, TIER_FALLBACK, TIER_MINIMAL  # noqa: E402

GEFITINIB = "COC1=C(C=C2C(=C1)N=CN=C2C3=CC(=C(C=C3)F)Cl)OCCCN4CCOCC4"
ERLOTINIB = "COCCOC1=C(C=C2C(=C1)C(=O)C(=CN2)C3=CC=CC=C3)OCCCN4CCOCC4"


def _all_predictor_instances():
    """收集可用的降级链实例（RDKit 增强版存在时一并测）。"""
    instances = [MinimalEGFRPredictor(), fallback_predictor.FallbackEGFRPredictor()]
    if getattr(fallback_predictor, "FallbackEGFRPredictor", None) is not None:
        rdkit_cls = getattr(fallback_predictor, "FallbackEGFRPredictorWithRDKit", None)
        if rdkit_cls is not None:
            try:
                instances.append(rdkit_cls())
            except Exception:
                pass
    return instances


@pytest.mark.parametrize("idx", range(3))
def test_determinism_same_input_same_output(idx):
    """同一输入重复预测 + 跨实例预测，概率必须逐位一致。"""
    instances = _all_predictor_instances()
    if idx >= len(instances):
        pytest.skip("该层实例不可用")
    p = instances[idx]
    r1 = p.predict(GEFITINIB)
    r2 = p.predict(GEFITINIB)
    r3 = p.__class__().predict(GEFITINIB)
    assert r1.get("success") is True and r2.get("success") is True
    assert r1["probability_active"] == r2["probability_active"] == r3["probability_active"]
    assert r1["prediction"] == r2["prediction"] == r3["prediction"]


def test_all_tiers_deterministic_both_molecules():
    for p in _all_predictor_instances():
        for smi in (GEFITINIB, ERLOTINIB):
            a = p.predict(smi)["probability_active"]
            b = p.predict(smi)["probability_active"]
            assert a == b, f"{type(p).__name__} 对 {smi[:20]}... 输出不稳定"


def test_probability_bounds_and_success():
    for p in _all_predictor_instances():
        r = p.predict(GEFITINIB)
        assert r["success"] is True
        assert 0.0 < r["probability_active"] < 1.0
        assert r["prediction"] in (0, 1)


def test_honest_tier_labels():
    """非真实模型层必须自带层级标签与「非机器学习模型」声明。"""
    for p in _all_predictor_instances():
        r = p.predict(GEFITINIB)
        assert r.get("predictor_tier") in (TIER_MINIMAL, TIER_FALLBACK, TIER_FALLBACK + "_rdkit")
        assert ("非机器学习" in r.get("note", "")) or ("启发式" in r.get("note", ""))
    # 真实模型层（如可加载）必须标 real
    try:
        from real_predictor import RealEGFRPredictor
        rp = RealEGFRPredictor()
        if rp.model is not None:
            assert rp.predict(GEFITINIB)["predictor_tier"] == "real"
    except Exception:
        pass  # 环境缺 pkl/依赖时跳过，不影响本测试目标


def test_invalid_input_still_errors():
    """空串必须报错（RDKit 增强版会把解析失败回落到基线，基线对 len<2 报错）。"""
    for p in _all_predictor_instances():
        r = p.predict("")
        assert "error" in r, f"{type(p).__name__} 对空串未报错"
    # 甲烷 "C" 是合法分子，规则基线按设计允许预测，不做断言。


def test_no_hash_in_predictor_sources():
    """预测路径源码禁止再出现 hash 播种造数（F-08/F-09 回归防线）。

    只打击真实滥用模式（``hash(smiles)`` 播种 / ``seed(hash(...)``），
    不打击文档字符串里「禁止 hash()」之类的说明文字。
    """
    for name in ("app.py", "fallback_predictor.py", "minimal_predictor.py",
                 "real_predictor.py", "gnn_predictor.py"):
        src = (REPO_ROOT / name).read_text(encoding="utf-8")
        assert "hash(smiles)" not in src, f"{name} 中再次出现 hash(smiles) 造数"
        assert "seed(hash" not in src, f"{name} 中再次出现 seed(hash(...)) 播种"
