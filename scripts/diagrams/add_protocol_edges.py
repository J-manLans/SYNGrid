"""Draw the rebuilt strain's class diagrams as Mermaid: the whole strain, and one diagram per group.

pyreverse, from pylint, draws the whole strain into `output/diagrams/classes.mmd`, leaving out
`legacy/`, with private attributes and class names only (`-f ALL -k`). It also writes
`packages.mmd`, the modules and their imports, from which this script drops every module no
import touches. This script then reads the source with `ast`, so it imports nothing, and adds
two kinds of arrow pyreverse cannot see:

- **Protocol arrows** (`Impl ..|> Proto`). pyreverse draws an arrow only for a base class a class
  names, so a class that matches a `typing.Protocol` structurally gets none. A class whose members
  cover all of a Protocol's members gets one; matching is by member name, not by signature.
- **Handed-in arrows** (`Part --o Owner : attr`). pyreverse draws an attribute's arrow only where
  it sees the class built, as in `self.x = SomeClass()`, so what a class is handed has none. For
  each constructor, every parameter stored on `self` as it is, and every `self.x: T` the
  constructor annotates, gets an arrow to the hinted type, usually a protocol, which is what the
  class knows. A dataclass's or a pydantic model's fields count as its constructor's parameters,
  and `self.x = param.field()` keeps what `field`, a `Callable` the param's class annotates,
  returns, as the environment keeps the world its scenario builds.
  Lists, tuples, dicts, `Final` and unions are unwrapped to what they hold; a `Callable` is not
  followed, since it is called, not held.

Both kinds are added to every diagram that holds the class, and an arrow the diagram already has
is left out. Then each group in `GROUPS` is cut out of the whole diagram into
`classes_<name>.mmd`: its own classes, every arrow that touches one of them, and the classes at
the other end of those arrows, marked `<<outside>>` as neighbours outside the group. A group also gets a
use arrow (`User ..> Used`) for each class a method of one of its classes takes or returns, where
no other arrow joins the two, so it shows what it works with, such as the orbs the digesters
digest. The scenario group also gets `..> : «create»` arrows from each recipe and population to
the classes it makes, the digesters a world recipe has its bundles make among them.

`legend.mmd` draws every kind of arrow once, between boxes named for the part each end plays,
with an example from the code.

A class is known by its bare name.

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
WHOLE = DIAGRAM_DIR / "classes.mmd"
PACKAGES = DIAGRAM_DIR / "packages.mmd"
LEGEND = DIAGRAM_DIR / "legend.mmd"

# Every arrow the diagrams use, between boxes named for the part each end plays, with its UML
# name and an example from the code. Mermaid draws the arrowhead or diamond at the second class written.
LEGEND_TEXT = """classDiagram
  class Subclass
  class BaseClass
  Subclass --|> BaseClass : Generalization - inherits from, as TierOrb from BaseOrb

  class Implementer
  class Protocol
  <<interface>> Protocol
  Implementer ..|> Protocol : Realization - matches the protocol, as TierDigester matches OrbDigester

  class Part
  class Owner
  Part --* Owner : Composition - is built by the owner and lives in it, as Timer in BaseOrb

  class HandedIn
  class Holder
  HandedIn --o Holder : Aggregation - is handed to the holder and kept, as SynergyDroid by GridWorld

  class Keeper
  class Field
  Keeper --> Field : Association - keeps a field of this type, as Scenario its ObservationRules

  class User
  class Used
  User ..> Used : Dependency - takes or returns it in a method, as a digester takes a BaseOrb

  class Maker
  class Made
  Maker ..> Made : «create» Dependency, instantiation - makes it, as WorldRecipe makes a GridWorld

  class Neighbour
  <<outside>> Neighbour
"""

SKIP_DIRS = {"legacy", "__pycache__"}
IGNORED_MEMBERS = {"__init__", "__slots__", "__doc__", "__module__"}
# Bases whose class-level annotations are the constructor's parameters.
FIELD_BASES = {"BaseModel", "NamedTuple"}
# Annotations whose contents are not held: a callable is called, a ClassVar is not per instance.
NOT_HELD = {"Callable", "ClassVar", "type"}
# Classes that make what a world holds: their «create» arrows go in the scenario group.
CREATORS = re.compile(r"(Recipe|Population)$")


@dataclass(frozen=True)
class Group:
    """One diagram cut out of the whole: a name, and the source paths it covers.

    A path, relative to `src/syn_grid/`, is a folder when it ends in `/` and a file otherwise.
    """

    name: str
    covers: tuple[str, ...]
    leaves_out: tuple[str, ...] = ()
    creates: bool = False


# The diagrams beside the whole one, grouped as the owner thinks of the code. `rendering`, `plot`
# and `utils` show only in the whole one.
GROUPS: tuple[Group, ...] = (
    Group("orbs", ("core/orbs/", "core/utils/timer.py")),
    Group("droid_and_digestion", ("core/droid/",)),
    Group("grid_world", ("core/grid_world.py",)),
    Group(
        "scenario",
        ("scenario/",),
        # A family's config models are config, not how a world is put together.
        leaves_out=(
            "scenario/families/goal/config.py",
            "scenario/families/goal/single_tier_chain/config.py",
        ),
        creates=True,
    ),
    Group(
        "observation",
        ("gymnasium/observation_space/", "scenario/blocks/observation.py"),
    ),
    Group("environment", ("gymnasium/environment.py",)),
    Group(
        "runners",
        ("runners/", "gymnasium/utils/episode_logging/"),
        # The extractor joins once it is rewritten, in phase 7b.
        leaves_out=("runners/agent_runners/utils/extractors.py",),
    ),
)


@dataclass
class ClassInfo:
    name: str
    path: str
    bases: list[str]
    members: set[str] = field(default_factory=set)
    is_protocol: bool = False
    is_dataclass: bool = False
    node: ast.ClassDef | None = None


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
    if isinstance(node, ast.Call):  # @dataclass(frozen=True)
        return _base_name(node.func)
    return ""


def _self_target(node: ast.expr) -> str | None:
    if (
        isinstance(node, ast.Attribute)
        and isinstance(node.value, ast.Name)
        and node.value.id == "self"
    ):
        return node.attr
    return None


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
                    if (name := _self_target(t)) is not None:
                        members.add(name)
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
                    path=path.relative_to(src).as_posix(),
                    bases=bases,
                    members=_collect_members(node),
                    is_protocol="Protocol" in bases,
                    is_dataclass="dataclass" in map(_base_name, node.decorator_list),
                    node=node,
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


def find_realizations(classes: dict[str, ClassInfo]) -> list[str]:
    """Return an `Impl ..|> Proto` edge for each class that matches a protocol it does not name."""
    protocols = {n: c for n, c in classes.items() if c.is_protocol}
    edges: list[str] = []
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
                edges.append(f"{n} ..|> {proto}")
    return edges


def _held_types(annotation: ast.expr | None, known: set[str]) -> list[str]:
    """The known classes an annotation holds, with containers, `Final` and unions unwrapped."""
    if annotation is None:
        return []
    if isinstance(annotation, ast.Constant) and isinstance(annotation.value, str):
        try:
            annotation = ast.parse(annotation.value, mode="eval").body
        except SyntaxError:
            return []
    found: list[str] = []

    def visit(node: ast.expr) -> None:
        if isinstance(node, ast.Subscript):
            if _base_name(node.value) in NOT_HELD:
                return
            visit(node.slice)
        elif isinstance(node, (ast.Tuple, ast.List)):
            for elt in node.elts:
                visit(elt)
        elif isinstance(node, ast.BinOp):  # X | None
            visit(node.left)
            visit(node.right)
        else:
            name = _base_name(node)
            if name in known and name not in found:
                found.append(name)

    visit(annotation)
    return found


def _handed_in(
    info: ClassInfo, classes: dict[str, ClassInfo]
) -> list[tuple[str, ast.expr | None]]:
    """The (attribute, annotation) pairs a class's constructor is handed and keeps."""
    assert info.node is not None
    pairs: list[tuple[str, ast.expr | None]] = []
    field_based = info.is_dataclass or bool(
        FIELD_BASES & (ancestors(info.name, classes) | set(info.bases))
    )
    for item in info.node.body:
        if (
            field_based
            and isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
        ):
            pairs.append((item.target.id, item.annotation))
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            params = {
                a.arg: a.annotation for a in item.args.args[1:] + item.args.kwonlyargs
            }
            for sub in ast.walk(item):
                if isinstance(sub, ast.AnnAssign) and (
                    name := _self_target(sub.target)
                ):
                    pairs.append((name, sub.annotation))
                elif (
                    isinstance(sub, ast.Assign)
                    and isinstance(sub.value, ast.Name)
                    and sub.value.id in params
                ):
                    for t in sub.targets:
                        if name := _self_target(t):
                            pairs.append((name, params[sub.value.id]))
                elif isinstance(sub, ast.Assign) and (
                    made := _made_by_param(sub.value, params, classes)
                ):
                    for t in sub.targets:
                        if name := _self_target(t):
                            pairs.append((name, made))
    return pairs


def _made_by_param(
    value: ast.expr, params: dict[str, ast.expr | None], classes: dict[str, ClassInfo]
) -> ast.expr | None:
    """What `param.field()` returns, where the param's class annotates `field` as a `Callable`.

    It is how a constructor keeps what a handed-in recipe makes, as `self.world =
    scenario.build_world()`.
    """
    if not (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Attribute)
        and isinstance(value.func.value, ast.Name)
        and value.func.value.id in params
    ):
        return None
    hinted = _held_types(params[value.func.value.id], set(classes))
    if len(hinted) != 1 or (info := classes[hinted[0]]).node is None:
        return None
    for item in info.node.body:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == value.func.attr
            and isinstance(item.annotation, ast.Subscript)
            and _base_name(item.annotation.value) == "Callable"
            and isinstance(item.annotation.slice, ast.Tuple)
            and len(item.annotation.slice.elts) == 2
        ):
            return item.annotation.slice.elts[1]
    return None


def find_handed_in(classes: dict[str, ClassInfo]) -> list[str]:
    """Return a `Part --o Owner : attr` edge for each class a constructor is handed and keeps."""
    known = set(classes)
    edges: list[str] = []
    for owner in sorted(classes):
        for attr, annotation in _handed_in(classes[owner], classes):
            for part in _held_types(annotation, known):
                if part != owner:
                    edges.append(f"{part} --o {owner} : {attr}")
    return edges


def find_uses(classes: dict[str, ClassInfo]) -> list[str]:
    """Return a `User ..> Used` edge for each class a method takes or returns.

    They go only in the groups, where they show the neighbours a group works with, such as the
    orbs the digesters digest. In the whole diagram they would bury the arrows of what is held.
    """
    known = set(classes)
    edges: list[str] = []
    by_class: dict[str, set[str]] = {}
    for user in sorted(classes):
        info = classes[user]
        if info.node is None:
            continue
        used: list[str] = []
        for item in info.node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                args = item.args.posonlyargs + item.args.args + item.args.kwonlyargs
                for arg in args:
                    used += _held_types(arg.annotation, known)
                used += _held_types(item.returns, known)
        by_class[user] = set(used) - {user}
    for user in sorted(by_class):
        # A use a base class already has is inherited, and not drawn again.
        inherited = set().union(
            *(by_class.get(a, set()) for a in ancestors(user, classes))
        )
        for name in sorted(by_class[user] - inherited):
            edges.append(f"{user} ..> {name}")
    return edges


def find_creations(classes: dict[str, ClassInfo]) -> list[str]:
    """Return a `Maker ..> Made : «create»` edge for each class a recipe or a population makes.

    What it makes is every known class it calls by name, what its `__call__` returns, and what a
    `Callable` field it calls returns, as a world recipe calls each bundle's `make_digester`. A
    field is found by its name, and only when one class has a `Callable` field of that name.
    """
    known = set(classes)
    makers: dict[str, list[ast.expr]] = {}
    for info in classes.values():
        for item in info.node.body if info.node else []:
            if (
                isinstance(item, ast.AnnAssign)
                and isinstance(item.target, ast.Name)
                and isinstance(item.annotation, ast.Subscript)
                and _base_name(item.annotation.value) == "Callable"
                and isinstance(item.annotation.slice, ast.Tuple)
                and len(item.annotation.slice.elts) == 2
            ):
                makers.setdefault(item.target.id, []).append(
                    item.annotation.slice.elts[1]
                )
    edges: list[str] = []
    for maker in sorted(classes):
        info = classes[maker]
        if not CREATORS.search(maker) or info.is_protocol or info.node is None:
            continue
        made: list[str] = []
        for item in info.node.body:
            if not isinstance(item, ast.FunctionDef):
                continue
            if item.name == "__call__":
                made += _held_types(item.returns, known)
            for sub in ast.walk(item):
                if (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Name)
                    and sub.func.id in known
                ):
                    made.append(sub.func.id)
                elif (
                    isinstance(sub, ast.Call)
                    and isinstance(sub.func, ast.Attribute)
                    and len(makers.get(sub.func.attr, [])) == 1
                ):
                    made += _held_types(makers[sub.func.attr][0], known)
        for name in dict.fromkeys(made):
            if name != maker:
                edges.append(f"{maker} ..> {name} : «create»")
    return edges


# ================= #
#     Patching      #
# ================= #

EDGE = re.compile(r"^\s*(\w+) (\S+) (\w+)(?: : (.+?))?\s*$")
CLASS = re.compile(r"^\s*class (\w+)", re.MULTILINE)


def _edge_key(edge: str) -> tuple[str, str, str] | None:
    """An edge's two ends and its label, so one drawn with another arrow counts as the same.

    pyreverse writes an association owner first (`Owner --> Part`) and a composition or an
    aggregation part first (`Part --* Owner`), so the ends are put in one order, part first.
    """
    m = EDGE.match(edge)
    if m is None or not re.search(r"[-.]", m.group(2)):
        return None
    first, arrow, second = m.group(1), m.group(2), m.group(3)
    if arrow == "-->":
        first, second = second, first
    return first, second, m.group(4) or ""


def patch_mmd(path: Path, edges: list[str], interface: bool) -> list[str]:
    """Append each edge whose first class the diagram holds and that it does not already draw."""
    text = path.read_text(encoding="utf-8")
    present = set(CLASS.findall(text))
    drawn = {k for line in text.splitlines() if (k := _edge_key(line))}
    added: list[str] = []
    lines: list[str] = []
    for edge in edges:
        key = _edge_key(edge)
        assert key is not None
        first, _, second = edge.split(" ", 3)[:3]
        if first not in present or key in drawn or (key[0], key[1], "") in drawn:
            continue
        if second not in present:  # the other class is not drawn here
            lines.append(f"  class {second}")
            if interface:
                lines.append(f"  <<interface>> {second}")
            present.add(second)
        lines.append(f"  {edge}")
        drawn.add(key)
        added.append(edge)
    if lines:
        path.write_text(
            text.rstrip("\n") + "\n" + "\n".join(lines) + "\n", encoding="utf-8"
        )
    return added


def _in_group(path: str, group: Group) -> bool:
    if path in group.leaves_out:
        return False
    return any(
        path.startswith(c) if c.endswith("/") else path == c for c in group.covers
    )


def write_group(
    group: Group,
    whole: str,
    classes: dict[str, ClassInfo],
    creations: list[str],
    uses: list[str],
) -> Path:
    """Cut a group out of the whole diagram, with its neighbours outside it marked `<<outside>>`.

    The mark is an annotation, the oldest Mermaid syntax for one: a `style` line breaks older
    renderers, such as Obsidian's.

    A use arrow is drawn only from a class inside the group, and only between two classes no
    other arrow joins, either way round: what the group works with, not everything that works
    with it.
    """
    inside = {n for n, c in classes.items() if _in_group(c.path, group)}
    inside &= set(CLASS.findall(whole))
    edges = [line.strip() for line in whole.splitlines() if _edge_key(line)]
    if group.creates:
        edges += creations
    joined = {frozenset(k[:2]) for e in edges if (k := _edge_key(e))}
    for use in uses:
        key = _edge_key(use)
        assert key is not None
        if key[0] in inside and frozenset(key[:2]) not in joined:
            edges.append(use)
            joined.add(frozenset(key[:2]))
    keys = {e: k for e in edges if (k := _edge_key(e))}
    kept = [e for e, k in keys.items() if k[0] in inside or k[1] in inside]
    outside = sorted({end for e in kept for end in keys[e][:2] if end not in inside})
    interfaces = set(re.findall(r"<<interface>> (\w+)", whole)) | {
        n for n, c in classes.items() if c.is_protocol
    }

    lines = ["classDiagram"]
    for name in sorted(inside) + outside:
        lines.append(f"  class {name}")
        # One word: Obsidian's Mermaid takes no comma or space in an annotation. Outside the
        # group, being outside is what matters; a protocol shows as one in its own group.
        if name in outside:
            lines.append(f"  <<outside>> {name}")
        elif name in interfaces:
            lines.append(f"  <<interface>> {name}")
    lines += [f"  {e}" for e in kept]

    out = DIAGRAM_DIR / f"classes_{group.name}.mmd"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def drop_unconnected(path: Path) -> list[str]:
    """Remove every box no arrow touches, and return their names.

    In the module diagram these are packages nothing imports by their own name, since imports
    name the module that holds a class, and empty packages kept for later.
    """
    lines = path.read_text(encoding="utf-8").splitlines()
    connected = {end for line in lines if (k := _edge_key(line)) for end in k[:2]}
    kept: list[str] = []
    dropped: list[str] = []
    skip_close = False
    for line in lines:
        if skip_close:
            skip_close = line.strip() != "}"
            continue
        m = CLASS.match(line)
        if m and m.group(1) not in connected:
            dropped.append(m.group(1))
            skip_close = line.rstrip().endswith("{")
            continue
        kept.append(line)
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")
    return dropped


def draw_diagrams() -> list[Path]:
    """Run pyreverse with this interpreter and return the diagrams it wrote."""
    if importlib.util.find_spec("pylint") is None:
        sys.exit(
            "pyreverse is missing: it comes with pylint, which is not installed in this "
            f"environment ({sys.executable}).\n"
            'Install it with: pip install -e ".[diagrams]"'
        )
    DIAGRAM_DIR.mkdir(parents=True, exist_ok=True)
    for old in DIAGRAM_DIR.glob("classes_*.mmd"):
        old.unlink()
    subprocess.run(
        [
            sys.executable,
            "-c",
            "from pylint import run_pyreverse; run_pyreverse()",
            "-o",
            "mmd",
            # Every attribute, private ones too, so their "has a" arrows show; class names only.
            "-f",
            "ALL",
            "-k",
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
    classes = scan(SRC_DIR)
    realizations = find_realizations(classes)
    handed_in = find_handed_in(classes)
    for f in files:
        added = patch_mmd(f, realizations, interface=True)
        added += patch_mmd(f, handed_in, interface=False)
        print(f"{f.relative_to(ROOT)}: {len(added)} edge(s) added")
        for a in added:
            print(f"    {a}")
    LEGEND.write_text(LEGEND_TEXT, encoding="utf-8")
    print(f"{LEGEND.relative_to(ROOT)}: what each arrow means")
    dropped = drop_unconnected(PACKAGES)
    print(
        f"{PACKAGES.relative_to(ROOT)}: {len(dropped)} unconnected module(s) dropped: {', '.join(dropped)}"
    )

    whole = WHOLE.read_text(encoding="utf-8")
    creations = find_creations(classes)
    uses = find_uses(classes)
    for group in GROUPS:
        out = write_group(group, whole, classes, creations, uses)
        text = out.read_text(encoding="utf-8")
        n_classes = len(re.findall(r"^\s*class ", text, re.MULTILINE))
        n_dashed = len(re.findall(r"^\s*<<[^>]*outside>> ", text, re.MULTILINE))
        n_edges = sum(1 for line in text.splitlines() if _edge_key(line))
        print(
            f"{out.relative_to(ROOT)}: {n_classes - n_dashed} classes, {n_dashed} outside, {n_edges} edges"
        )


if __name__ == "__main__":
    main()
