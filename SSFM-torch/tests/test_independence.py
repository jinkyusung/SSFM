import ast
from pathlib import Path


FORBIDDEN = {"jax", "equinox", "diffrax", "optax", "jaxtyping"}


def test_package_has_no_reference_runtime_imports():
    package = Path(__file__).parents[1] / "ssfm_torch"
    imported = set()
    for path in package.glob("*.py"):
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
    assert imported.isdisjoint(FORBIDDEN), imported & FORBIDDEN


def test_project_dependencies_are_independent():
    project = (Path(__file__).parents[1] / "pyproject.toml").read_text().lower()
    assert all(name not in project for name in FORBIDDEN)

