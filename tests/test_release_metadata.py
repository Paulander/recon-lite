from pathlib import Path
import re

import recon_lite


ROOT = Path(__file__).resolve().parents[1]


def test_pyproject_version_matches_package_version():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version = "([^"]+)"$', pyproject, re.MULTILINE)

    assert match is not None
    assert match.group(1) == recon_lite.__version__


def test_release_metadata_is_present():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'license = "MIT"' in pyproject
    assert "[project.urls]" in pyproject
    assert "Development Status :: 3 - Alpha" in pyproject
