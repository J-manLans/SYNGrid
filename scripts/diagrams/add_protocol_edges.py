"""Draw the rebuilt strain's class diagrams as Mermaid, with protocol realization arrows (..|>).

pyreverse, from pylint, draws the diagrams into `output/diagrams/`, leaving out `legacy/`. It
draws an arrow only for a base class a class names, so a class that matches a `typing.Protocol`
structurally gets none. This script then reads the source with `ast`, so it imports nothing,
finds every Protocol, and finds every class whose members cover all of a Protocol's members. It
appends an `Impl ..|> Proto` edge to each diagram that holds the class, and leaves out an edge
the diagram already has.

Matching is by member name, not by signature, and a class is known by its bare name.

pylint is not a dependency of the project or of CI. Install it once with the `diagrams` extra,
then run from any folder:

    pip install -e ".[diagrams]"
    python scripts/diagrams/add_protocol_edges.py
"""

from __future__ import annotations

import ast
import importlib.util
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src" / "syn_grid"
DIAGRAM_DIR = ROOT / "output" / "diagrams"

SKIP_DIRS = {"legacy", "__pycache__"}
IGNORED_MEMBERS = {"__init__", "__slots__", "__doc__", "__module__"}


@dataclass
class ClassInfo:
    name: str
    bases: list[str]
    members: set[str] = field(default_factory=set)
    is_protocol: bool = False


# ================= #
#     Scanning      #
# ================= #


def _base_name(node: ast.expr) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Subscript):  # Protocol[T], Generic[T]
        return _base_name(node.value)
    return ""


def _collect_members(cls: ast.ClassDef) -> set[str]:
    members: set[str] = set()
    for item in cls.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            members.add(item.name)
            # self.x = ... inside a method counts as an attribute
            for sub in ast.walk(item):
                targets: list[ast.expr] = []
                if isinstance(sub, ast.Assign):
                    targets = sub.targets
                elif isinstance(sub, (ast.AnnAssign, ast.AugAssign)):
                    targets = [sub.target]
                for t in targets:
                    if (
                        isinstance(t, ast.Attribute)
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"
                    ):
                        members.add(t.attr)
        elif isinstance(item, ast.AnnAssign) and isinstance(item.target, ast.Name):
            members.add(item.target.id)
        elif isinstance(item, ast.Assign):
            for t in item.targets:
                if isinstance(t, ast.Name):
                    members.add(t.id)
    return members - IGNORED_MEMBERS


def scan(src: Path) -> dict[str, ClassInfo]:
    classes: dict[str, ClassInfo] = {}
    for path in sorted(src.rglob("*.py")):
        if SKIP_DIRS & set(path.parts):
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                bases = [b for b in map(_base_name, node.bases) if b]
                classes[node.name] = ClassInfo(
                    name=node.name,
                    bases=bases,
                    members=_collect_members(node),
                    is_protocol="Protocol" in bases,
                )
    return classes


# ================= #
#     Matching      #
# ================= #


def all_members(
    name: str, classes: dict[str, ClassInfo], seen: set[str] | None = None
) -> set[str]:
    seen = seen if seen is not None else set()
    if name in seen or name not in classes:
        return set()
    seen.add(name)
    info = classes[name]
    out = set(info.members)
    for b in info.bases:
        out |= all_members(b, classes, seen)
    return out


def ancestors(name: str, classes: dict[str, ClassInfo]) -> set[str]:
    out: set[str] = set()
    stack = list(classes[name].bases) if name in classes else []
    while stack:
        b = stack.pop()
        if b in out:
            continue
        out.add(b)
        if b in classes:
            stack.extend(classes[b].bases)
    return out


def find_realizations(classes: dict[str, ClassInfo]) -> list[tuple[str, str]]:
    protocols = {n: c for n, c in classes.items() if c.is_protocol}
    edges: list[tuple[str, str]] = []
    for proto, pinfo in protocols.items():
        required = pinfo.members
        if not required:
            continue
        matches = {
            n
            for n, c in classes.items()
            if not c.is_protocol
            # A class that names the protocol as a base already has pyreverse's arrow.
            and proto not in ancestors(n, classes)
            and required <= all_members(n, classes)
        }
        # Only the topmost match gets an arrow; its subclasses inherit the relation.
        for n in sorted(matches):
            if not (ancestors(n, classes) & matches):
                edges.append((n, proto))
    return edges


# ================= #
#     Patching      #
# ================= #


def patch_mmd(path: Path, edges: list[tuple[str, str]]) -> list[str]:
    text = path.read_text(encoding="utf-8")
    present = set(re.findall(r"^\s*class (\w+)", text, re.MULTILINE))
    drawn = set(re.findall(r"^\s*(\w+ \.\.\|> \w+)\s*$", text, re.MULTILINE))
    added: list[str] = []
    lines: list[str] = []
    for impl, proto in edges:
        edge = f"{impl} ..|> {proto}"
        if impl not in present or edge in drawn:
            continue
        if proto not in present:  # the protocol is in another diagram
            lines.append(f"  class {proto}")
            lines.append(f"  <<interface>> {proto}")
            present.add(proto)
        lines.append(f"  {edge}")
        added.append(edge)
    if lines:
        path.write_text(
            text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )
    return added


def draw_diagrams() -> list[Path]:
    """Run pyreverse with this interpreter and return the diagrams it wrote."""
    if importlib.util.find_spec("pylint") is None:
        sys.exit(
            "pyreverse is missing: it comes with pylint, which is not installed in this "
            f"environment ({sys.executable}).\n"
            'Install it with: pip install -e ".[diagrams]"'
        )
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pylint import run_pyreverse; run_pyreverse()",
            "-o",
            "mmd",
            "-d",
            str(DIAGRAM_DIR),
            "--ignore",
            "legacy",
            str(SRC_DIR),
        ],
        check=True,
    )
    return sorted(DIAGRAM_DIR.glob("*.mmd"))


def main() -> None:
    files = draw_diagrams()
    edges = find_realizations(scan(SRC_DIR))
    for f in files:
        added = patch_mmd(f, edges)
        print(f"{f.relative_to(ROOT)}: {len(added)} protocol edge(s)")
        for a in added:
            print(f"    {a}")


if __name__ == "__main__":
    main()
