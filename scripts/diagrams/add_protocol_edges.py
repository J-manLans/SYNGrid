"""Draw the rebuilt strain's class diagrams as Mermaid: the whole strain, and one diagram per group.

pyreverse, from pylint, draws the whole strain into `output/diagrams/classes.mmd`, leaving out
`legacy/`, with private attributes and class names only (`-f ALL -k`). It also writes
`packages.mmd`, the modules and their imports, from which this script drops every module no
import touches. This script then reads the source with `ast`, so it imports nothing, and adds
arrows pyreverse cannot see:

- **Protocol arrows** (`Impl ..|> Proto`). pyreverse draws an arrow only for a base class a class
  names, so a class that matches a `typing.Protocol` structurally gets none. A class whose members
  cover all of a Protocol's members gets one; matching is by member name, not by signature.
- **What a class keeps.** pyreverse draws an attribute's arrow only where it sees the class built,
  as in `self.x = SomeClass()`, or a field whose type is a plain class. The script adds, in
  pyreverse's own forms:
  - an aggregation (`Part --o Owner : attr`) for what a constructor is handed and keeps: a
    parameter stored on `self`, an empty container it annotates and fills, as the engine's dict
    of digesters, `cast(T, ...)`, a field of a
    parameter (`self._termination = scenario.termination`), or what a `Callable` field of a
    parameter returns (`self.world = scenario.build_world()`);
  - an association (`Owner --> Part : attr`) for a dataclass's or a pydantic model's field;
  - a dependency (`Owner ..> Made : attr`) for a field that is a `Callable`, on what it returns,
    as an entry's `scoring` returns a tier digester.
  The arrow goes to the hinted type, usually a protocol, which is what the class knows; for a
  stored parameter, to the parameter's hint, the narrower one. Lists, tuples, `Final`, unions
  and module-level type aliases are unwrapped, a mapping to its values; what is inside a
  `Callable` is followed only to what it returns.

These are added to every diagram that holds the class, and an arrow the diagram already has
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
  User ..> Used : Dependency - takes or returns it, or holds a callable that makes it, as a digester takes a BaseOrb

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
MAPPINGS = {"dict", "Dict", "Mapping", "MutableMapping"}
# Classes that make what a world holds: their «create» arrows go in the scenario group.
CREATORS = re.compile(r"(Recipe|Population)$")
# Module-level type aliases, `X = A | B` or `type X = ...`, filled in by `scan`.
ALIASES: dict[str, ast.expr] = {}


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
        for stmt in tree.body:
            if isinstance(stmt, ast.TypeAlias) and isinstance(stmt.name, ast.Name):
                ALIASES[stmt.name.id] = stmt.value
            elif (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and isinstance(stmt.value, ast.BinOp)
            ):
                ALIASES[stmt.targets[0].id] = stmt.value
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
    """The known classes an annotation holds.

    Containers, `Final`, unions and the type aliases in `ALIASES` are unwrapped; what is inside
    a `Callable`, a `ClassVar` or a `type[...]` is not held.
    """
    if annotation is None:
        return []
    found: list[str] = []
    seen_aliases: set[str] = set()

    def visit(node: ast.expr) -> None:
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            try:
                visit(ast.parse(node.value, mode="eval").body)
            except SyntaxError:
                return
        elif isinstance(node, ast.Subscript):
            base = _base_name(node.value)
            if base in NOT_HELD:
                return
            if base in MAPPINGS and isinstance(node.slice, ast.Tuple):
                visit(
                    node.slice.elts[-1]
                )  # a mapping holds its values; its keys index them
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
            if name in ALIASES and name not in seen_aliases:
                seen_aliases.add(name)
                visit(ALIASES[name])
            elif name in known and name not in found:
                found.append(name)

    visit(annotation)
    return found


def _callable_returns(annotation: ast.expr | None) -> list[ast.expr]:
    """What each `Callable` in an annotation returns, following a callable that returns one."""
    if annotation is None:
        return []
    out: list[ast.expr] = []

    def visit(node: ast.expr) -> None:
        if isinstance(node, ast.BinOp):
            visit(node.left)
            visit(node.right)
        elif (
            isinstance(node, ast.Subscript)
            and _base_name(node.value) == "Callable"
            and isinstance(node.slice, ast.Tuple)
            and len(node.slice.elts) == 2
        ):
            returns = node.slice.elts[1]
            out.append(returns)
            visit(returns)

    visit(annotation)
    return out


def _field_annotation(info: ClassInfo, name: str) -> ast.expr | None:
    """The annotation a class gives one of its class-level fields."""
    for item in info.node.body if info.node else []:
        if (
            isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
            and item.target.id == name
        ):
            return item.annotation
    return None


def _taken_from_param(
    value: ast.expr, params: dict[str, ast.expr | None], classes: dict[str, ClassInfo]
) -> ast.expr | None:
    """The type a constructor keeps from a param it is handed, other than the param itself.

    - `cast(T, ...)`: T.
    - `param.field`: the field's annotation in the param's class, as `self._termination =
      scenario.termination`.
    - `param.field()`: what the field, a `Callable`, returns, as `self.world =
      scenario.build_world()`.
    """
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id == "cast"
        and value.args
    ):
        return value.args[0]
    called = isinstance(value, ast.Call)
    target = value.func if isinstance(value, ast.Call) else value
    if not (
        isinstance(target, ast.Attribute)
        and isinstance(target.value, ast.Name)
        and target.value.id in params
    ):
        return None
    hinted = _held_types(params[target.value.id], set(classes))
    if len(hinted) != 1:
        return None
    annotation = _field_annotation(classes[hinted[0]], target.attr)
    if not called:
        return annotation
    returns = _callable_returns(annotation)
    return returns[0] if returns else None


def _kept(
    info: ClassInfo, classes: dict[str, ClassInfo]
) -> list[tuple[str, ast.expr | None, bool]]:
    """What a class keeps: (attribute, annotation, whether it is a field).

    A field is a dataclass's or a pydantic model's class-level field. Everything else is what
    the constructor is handed and keeps: a parameter stored on `self`, a `self.x: T` it
    annotates, or what it takes from a parameter (see `_taken_from_param`).
    """
    assert info.node is not None
    kept: list[tuple[str, ast.expr | None, bool]] = []
    field_based = info.is_dataclass or bool(
        FIELD_BASES & (ancestors(info.name, classes) | set(info.bases))
    )
    for item in info.node.body:
        if (
            field_based
            and isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
        ):
            kept.append((item.target.id, item.annotation, True))
        if isinstance(item, ast.FunctionDef) and item.name == "__init__":
            params = {
                a.arg: a.annotation for a in item.args.args[1:] + item.args.kwonlyargs
            }
            for sub in ast.walk(item):
                if isinstance(sub, ast.AnnAssign):
                    targets, value = [sub.target], sub.value
                elif isinstance(sub, ast.Assign):
                    targets, value = sub.targets, sub.value
                else:
                    continue
                if value is None:
                    continue
                if isinstance(value, ast.Name) and value.id in params:
                    # The param's hint is the narrower one, as `kind: DirectKind` against
                    # `self.KIND: Final[OrbKind]`.
                    annotation = params[value.id] or getattr(sub, "annotation", None)
                elif isinstance(sub, ast.AnnAssign) and isinstance(
                    value, (ast.Dict, ast.List, ast.Set, ast.Tuple)
                ):
                    # An empty container the constructor fills from what it is handed, as
                    # the engine's dict of digesters.
                    annotation = sub.annotation
                else:
                    annotation = _taken_from_param(value, params, classes)
                if annotation is None:
                    continue
                for t in targets:
                    if name := _self_target(t):
                        kept.append((name, annotation, False))
    return kept


def find_kept(classes: dict[str, ClassInfo]) -> list[str]:
    """Return an arrow for each class another class keeps, in pyreverse's own forms.

    - A field: an association, `Owner --> Part : attr`, as pyreverse draws a field it sees.
    - What a constructor is handed and keeps: an aggregation, `Part --o Owner : attr`.
    - A field that is a `Callable`: a dependency on what it returns, `Owner ..> Made : attr`,
      as an entry's `scoring` returns a tier digester.
    """
    known = set(classes)
    edges: list[str] = []
    for owner in sorted(classes):
        for attr, annotation, is_field in _kept(classes[owner], classes):
            for part in _held_types(annotation, known):
                if part != owner:
                    edges.append(
                        f"{owner} --> {part} : {attr}"
                        if is_field
                        else f"{part} --o {owner} : {attr}"
                    )
            if is_field:
                for returns in _callable_returns(annotation):
                    for made in _held_types(returns, known):
                        if made != owner:
                            edges.append(f"{owner} ..> {made} : {attr}")
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
    kept = find_kept(classes)
    for f in files:
        added = patch_mmd(f, realizations, interface=True)
        added += patch_mmd(f, kept, interface=False)
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
