"""剩余业务模块的边界回归测试。"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def test_batch_filter_keeps_invalid_smiles_aligned():
    from chem_filter import SubstructureFilter

    frame = pd.DataFrame({"smiles": ["CCO", "not_a_smiles", "c1ccccc1"]})
    clean, labeled, _, _ = SubstructureFilter().filter_dataframe(frame.copy())

    assert len(labeled) == len(frame)
    assert labeled.loc[1, "Filter_Status"] == "Invalid SMILES"
    assert "not_a_smiles" not in clean["smiles"].tolist()


def test_gnn_batch_predict_validates_batch_size_and_empty_input():
    from gnn_predictor import GCNPredictor

    predictor = GCNPredictor.__new__(GCNPredictor)
    predictor.predict = lambda smiles: {"smiles": smiles}

    assert predictor.batch_predict([]) == []
    with pytest.raises(ValueError, match="batch_size"):
        predictor.batch_predict(["CCO"], batch_size=0)


def test_structure_visualizer_handles_timeout_and_bad_upload(monkeypatch):
    import structure_viz

    calls = []

    def fake_get(url, timeout=None):
        calls.append(timeout)
        return SimpleNamespace(status_code=200, text="HEADER test\n")

    monkeypatch.setattr(structure_viz.requests, "get", fake_get)
    visualizer = structure_viz.StructureVisualizer()
    assert visualizer.load_from_pdb_id("3POZ") is True
    assert calls == [30]

    class BadUpload:
        name = "broken.pdb"

        def getvalue(self):
            return b"\xff"

    assert structure_viz.StructureVisualizer().load_from_file(BadUpload()) is False


def test_md_simulation_rejects_nonpositive_parameters():
    import md_utils

    with pytest.raises(ValueError, match="total_steps"):
        md_utils.run_md_simulation(total_steps=0)
    with pytest.raises(ValueError, match="write_interval"):
        md_utils.run_md_simulation(total_steps=10, write_interval=0)


def test_md_dependency_probe_is_cached(monkeypatch):
    import builtins
    import md_utils

    first = md_utils._check_md_deps()

    def fail_import(*args, **kwargs):
        raise AssertionError("依赖探测未使用缓存")

    monkeypatch.setattr(builtins, "__import__", fail_import)
    assert md_utils._check_md_deps() is first
