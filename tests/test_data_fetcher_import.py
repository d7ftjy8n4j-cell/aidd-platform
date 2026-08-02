import importlib.util
from pathlib import Path


def test_data_fetcher_handles_chembl_import_error(monkeypatch):
    module_path = Path(__file__).resolve().parents[1] / "utils" / "data_fetcher.py"
    real_import = __import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name.startswith("chembl_webresource_client"):
            raise AttributeError("module 'bson' has no attribute 'encode'")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr("builtins.__import__", fake_import)

    spec = importlib.util.spec_from_file_location("data_fetcher_test_module", module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)

    assert module.CHEMBL_AVAILABLE is False
