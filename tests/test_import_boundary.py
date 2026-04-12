import ast
from pathlib import Path


FORBIDDEN = ("chess", "recon_lite_chess", "recon_lite_hector", "demos")


def test_core_has_no_domain_imports():
    root = Path(__file__).resolve().parents[1] / "src" / "recon_lite"
    violations = []
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(), filename=str(path))
        for node in ast.walk(tree):
            module = None
            if isinstance(node, ast.Import):
                for alias in node.names:
                    module = alias.name
                    if module.split(".")[0] in FORBIDDEN:
                        violations.append((path, module))
            elif isinstance(node, ast.ImportFrom) and node.module:
                module = node.module
                if module.split(".")[0] in FORBIDDEN:
                    violations.append((path, module))

    assert violations == []
