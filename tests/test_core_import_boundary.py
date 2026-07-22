# Copyright 2025-2026 EUMETSAT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Static test: plugin source must only import from firecube.ingestor.api or firecube.core.api.

Firecube exposes exactly two public import surfaces that plugins may depend on:

* ``firecube.ingestor.api``
* ``firecube.core.api``

This test walks every ``.py`` file under ``src/firecube_mtg_fci_l1c/``, parses it
with :mod:`ast`, and asserts that no ``from firecube.X import ...`` or
``import firecube.X`` statement reaches any other firecube submodule.
"""

from __future__ import annotations

import ast
import pathlib

# Only add to this set if an import CANNOT be migrated to the public API,
# with a documented reason in the comment. MUST stay empty for a clean codebase.
ALLOWED_DEEP_IMPORTS: frozenset[str] = frozenset()

APPROVED_FIRECUBE_MODULES: frozenset[str] = frozenset(
    {"firecube.ingestor.api", "firecube.core.api"}
)

# Any firecube.* module starting with one of these prefixes is explicitly
# considered an internal submodule and forbidden to plugins.
FORBIDDEN_PREFIXES: tuple[str, ...] = (
    "firecube.core.",
    "firecube.ingestor.runtime",
    "firecube.ingestor.contracts",
    "firecube.ingestor.cli",
    "firecube.ingestor.tracing",
)

SRC_ROOT: pathlib.Path = (
    pathlib.Path(__file__).parent.parent / "src" / "firecube_mtg_fci_l1c"
)


def _collect_firecube_imports(
    py_file: pathlib.Path,
) -> list[tuple[str, int]]:
    tree = ast.parse(py_file.read_text(encoding="utf-8"))
    found: list[tuple[str, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module == "firecube" or node.module.startswith("firecube."):
                found.append((node.module, node.lineno))
        elif isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "firecube" or name.startswith("firecube."):
                    found.append((name, node.lineno))
    return found


def test_no_deep_firecube_imports() -> None:
    """All firecube imports must come from one of the approved public APIs."""
    violations: list[str] = []
    py_files = sorted(SRC_ROOT.rglob("*.py"))
    assert py_files, f"No Python files found under {SRC_ROOT}"

    for py_file in py_files:
        if "__pycache__" in py_file.parts:
            continue
        rel = py_file.relative_to(SRC_ROOT.parent.parent)
        for module, lineno in _collect_firecube_imports(py_file):
            if module in APPROVED_FIRECUBE_MODULES:
                continue
            if module in ALLOWED_DEEP_IMPORTS:
                continue
            if any(module.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
                violations.append(
                    f"{rel}:{lineno}: forbidden import of '{module}' "
                    f"(internal submodule)"
                )
            else:
                # Any other firecube.* not in the approved list is suspicious.
                violations.append(
                    f"{rel}:{lineno}: unapproved firecube module '{module}' "
                    f"(must be one of {sorted(APPROVED_FIRECUBE_MODULES)})"
                )

    assert not violations, (
        "Deep firecube imports found (plugins may only import from "
        f"{sorted(APPROVED_FIRECUBE_MODULES)}):\n  " + "\n  ".join(violations)
    )
