import importlib

def test_imports():
    """The main module should import without errors."""
    importlib.import_module("v100_monitor_V1_0")