"""Draw the rebuilt strain's class diagrams by layer, as Mermaid drafts of the draw.io drawing.

A layer is a top-level package of `src/syn_grid/`: core, scenario, gymnasium, runners and
rendering. The script reads the source with `ast`, so it imports nothing and needs no
dependency, and writes into `output/diagrams/layers/`:

- `overview.mmd`: every layer as a container, the external libraries as single boxes, and only
  the arrows that cross from one layer to another;
- `<layer>.mmd`: one layer in full, with every arrow that touches one of its classes and the
  classes at the other end in containers of their own layers.

The relations, strongest first, and only the strongest kept for a pair:

- **inherit** (`Child --|> Parent`): a base class the class names.
- **realize** (`Impl ..|> Protocol`): a class whose members, `self.x` set in its methods
  included, cover a `typing.Protocol`'s, drawn only on the topmost class that matches.
- **compose** (`Owner *-- Part`): `self.x = Part(...)`, also inside a list, a dict or a
  comprehension, and `self.x: Part = maker(...)`, built through a class held in a variable.
- **assoc** (`Owner --> Part`): what a class keeps that it did not build: `self.x: Part`, a
  parameter stored on `self`, a dataclass's or a pydantic model's field, and what a constructor
  takes from a parameter, as `self.world = scenario.build_world()`.
- **use** (`User ..> Used : Use`): any other reference to a project class in the class body,
  string annotations included. A field typed `Callable[..., Part]` is a use, not an association.

A use is left out where a stronger arrow joins the two classes either way round, and where a
base class or a protocol the class realizes already has it. The classes in `LEFT_OUT` are not
drawn. Gymnasium and Stable-Baselines3 are single boxes: a class that subclasses one of their
classes inherits the box, and a class that names one otherwise uses it.

Inside a layer the classes are grouped as `SUBGROUPS` lists them, such as core's world, orbs and
droid. A group is a container named for its layer, as `core_orbs`, beside its layer's other
groups; `NESTED = True` writes it inside the layer's container, which is the picture meant, where
the renderer takes a namespace inside another.

Mermaid draws the containers with `namespace`, which older renderers may reject; `NAMESPACES =
False` draws every class loose and marks a neighbour from another layer `<<outside>>`. A protocol is
marked `<<protocol>>`, an abstract class `<<abstract>>`, a library `<<library>>`.

Run from any folder:

    python scripts/diagrams/layers.py
"""

from __future__ import annotations

import ast
import sys
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from add_protocol_edges import (
    SRC_DIR,
    ClassInfo,
    _base_name,
    _callable_returns,
    _held_types,
    _self_target,
    _taken_from_param,
    ancestors,
    find_realizations,
    scan,
)

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "output" / "diagrams" / "layers"

# ================= #
#     Settings      #
# ================= #

# The layers, in the order the overview lists them. A class outside them is not drawn.
LAYERS = ("runners", "gymnasium", "scenario", "core", "rendering")

# Left out of every diagram: whole folders, a name suffix, enums, and small value types.
LEFT_OUT_PATHS = ("config/",)
LEFT_OUT_SUFFIXES = ("Conf",)
LEFT_OUT_ENUMS = True
LEFT_OUT = {"StepResult", "EpisodeOutcome", "LogKey", "Lock"}

# The libraries drawn as one box each, by the top-level module their names come from.
LIBRARIES = {
    "gymnasium": "Gymnasium",
    "stable_baselines3": "StableBaselines3",
    "sb3_contrib": "StableBaselines3",
}

# Inside a layer, the classes are grouped as the owner thinks of the code: a name, and the paths
# it covers, relative to `src/syn_grid/`, a folder when it ends in `/` and a file otherwise. A
# class of the layer that no group covers sits in the layer itself.
SUBGROUPS: dict[str, tuple[tuple[str, tuple[str, ...]], ...]] = {
    "core": (
        ("world", ("core/grid_world.py",)),
        ("orbs", ("core/orbs/", "core/utils/timer.py")),
        ("droid", ("core/droid/",)),
    ),
    "gymnasium": (
        ("environment", ("gymnasium/environment.py",)),
        ("observation", ("gymnasium/observation_space/",)),
        ("logging", ("gymnasium/utils/episode_logging/",)),
    ),
    "scenario": (
        (
            "frame",
            ("scenario/scenario.py", "scenario/entry.py", "scenario/registry.py"),
        ),
        ("blocks", ("scenario/blocks/",)),
        ("families", ("scenario/families/",)),
    ),
    "runners": (
        ("agent_runners", ("runners/agent_runners/",)),
        ("human_play", ("runners/human_runner/",)),
    ),
}

NAMESPACES = True
# True writes a group as a namespace inside its layer's namespace, which is the picture meant,
# but Mermaid may not take a namespace inside another. False writes the groups side by side,
# each named for its layer, as `core_orbs`.
NESTED = False

# ================= #
#     Relations     #
# ================= #

STRENGTH = {"inherit": 5, "realize": 4, "compose": 3, "assoc": 2, "use": 1}
ARROW = {
    "inherit": "{a} --|> {b}",
    "realize": "{a} ..|> {b}",
    "compose": "{a} *-- {b} : {label}",
    "assoc": "{a} --> {b} : {label}",
    "use": "{a} ..> {b} : Use",
}
ENUM_BASES = {"Enum", "IntEnum", "StrEnum", "Flag", "IntFlag"}


@dataclass(frozen=True)
class Edge:
    kind: str
    source: str
    target: str
    label: str = ""

    def mermaid(self) -> str:
        line = ARROW[self.kind].format(a=self.source, b=self.target, label=self.label)
        return line.removesuffix(" : ")


def layer_of(info: ClassInfo) -> str | None:
    top = info.path.split("/", 1)[0]
    return top if top in LAYERS else None


def subgroup_of(info: ClassInfo) -> str | None:
    """The group a class is in inside its layer, or None when no group covers it."""
    for name, covers in SUBGROUPS.get(layer_of(info) or "", ()):
        if any(
            info.path.startswith(c) if c.endswith("/") else info.path == c
            for c in covers
        ):
            return name
    return None


def left_out(info: ClassInfo, classes: dict[str, ClassInfo]) -> bool:
    return (
        info.name in LEFT_OUT
        or info.path.startswith(LEFT_OUT_PATHS)
        or info.name.endswith(LEFT_OUT_SUFFIXES)
        or (
            LEFT_OUT_ENUMS
            and bool(ENUM_BASES & (set(info.bases) | ancestors(info.name, classes)))
        )
        or layer_of(info) is None
    )


def is_abstract(info: ClassInfo) -> bool:
    assert info.node is not None
    if "ABC" in info.bases:
        return True
    return any(
        isinstance(item, ast.FunctionDef)
        and "abstractmethod" in map(_base_name, item.decorator_list)
        for item in info.node.body
    )


def library_names(src: Path) -> dict[str, dict[str, str]]:
    """For each module, the names it imports from a library, mapped to the library's box."""
    out: dict[str, dict[str, str]] = {}
    for path in sorted(src.rglob("*.py")):
        if {"legacy", "__pycache__"} & set(path.parts):
            continue
        names: dict[str, str] = {}
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            if isinstance(node, ast.ImportFrom) and node.module:
                box = LIBRARIES.get(node.module.split(".", 1)[0])
                if box:
                    for alias in node.names:
                        names[alias.asname or alias.name] = box
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    box = LIBRARIES.get(alias.name.split(".", 1)[0])
                    if box:
                        names[alias.asname or alias.name.split(".", 1)[0]] = box
        out[path.relative_to(src).as_posix()] = names
    return out


def _root_name(node: ast.expr) -> str:
    while isinstance(node, (ast.Attribute, ast.Subscript)):  # gym.Wrapper[ObsType, ...]
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def _composed(value: ast.expr, known: set[str]) -> list[str]:
    """The known classes a value builds, also inside a literal or a comprehension."""
    found: list[str] = []
    for node in ast.walk(value):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Name)
            and node.func.id in known
            and node.func.id not in found
        ):
            found.append(node.func.id)
    # A call nested in another call's arguments is what that call is handed, not what is built.
    if (
        isinstance(value, ast.Call)
        and isinstance(value.func, ast.Name)
        and value.func.id in known
    ):
        return [value.func.id]
    return found


def class_edges(
    info: ClassInfo, classes: dict[str, ClassInfo], libs: dict[str, str]
) -> list[Edge]:
    """Every relation one class has, before the strongest is chosen."""
    assert info.node is not None
    known = set(classes)
    name = info.name
    edges: list[Edge] = []
    node = info.node

    # inherit
    for base in node.bases:
        b = _base_name(base)
        if b in known:
            edges.append(Edge("inherit", name, b))
        elif (box := libs.get(_root_name(base)) or libs.get(b)) is not None:
            edges.append(Edge("inherit", name, box))

    field_based = info.is_dataclass or bool(
        {"BaseModel", "NamedTuple"} & (ancestors(name, classes) | set(info.bases))
    )
    for item in node.body:
        # fields: assoc, or use for a Callable
        if (
            field_based
            and isinstance(item, ast.AnnAssign)
            and isinstance(item.target, ast.Name)
        ):
            label = item.target.id
            for part in _held_types(item.annotation, known):
                edges.append(Edge("assoc", name, part, label))
            for returns in _callable_returns(item.annotation):
                for made in _held_types(returns, known):
                    edges.append(Edge("use", name, made))

        if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        params = {
            a.arg: a.annotation for a in item.args.args[1:] + item.args.kwonlyargs
        }
        for sub in ast.walk(item):
            if isinstance(sub, ast.AnnAssign):
                targets, value, annotation = [sub.target], sub.value, sub.annotation
            elif isinstance(sub, ast.Assign):
                targets, value, annotation = sub.targets, sub.value, None
            else:
                continue
            attrs = [a for t in targets if (a := _self_target(t))]
            if not attrs:
                continue
            built = _composed(value, known) if value is not None else []
            for attr in attrs:
                for part in built:
                    edges.append(Edge("compose", name, part, attr))
                if built:
                    continue
                kept: ast.expr | None = None
                if isinstance(value, ast.Name) and value.id in params:
                    kept = params[value.id] or annotation
                elif value is not None and item.name == "__init__":
                    kept = _taken_from_param(value, params, classes)
                if (
                    kept is None
                    and isinstance(value, ast.Call)
                    and annotation is not None
                ):
                    # Built through a class held in a variable, as the handler builds its
                    # perception from the class `PERCEPTIONS` names: the annotation says what.
                    for part in _held_types(annotation, known):
                        edges.append(Edge("compose", name, part, attr))
                    continue
                for part in _held_types(kept or annotation, known):
                    edges.append(Edge("assoc", name, part, attr))

    # use: every other reference to a project class or a library in the class body
    for sub in ast.walk(node):
        refs: list[str] = []
        if isinstance(sub, ast.Name):
            refs = [sub.id]
        elif isinstance(sub, ast.Attribute):
            refs = [sub.attr, _root_name(sub)]
        elif isinstance(sub, ast.Constant) and isinstance(sub.value, str):
            try:
                parsed = ast.parse(sub.value, mode="eval").body
            except SyntaxError:
                continue
            refs = [n.id for n in ast.walk(parsed) if isinstance(n, ast.Name)]
        for ref in refs:
            if ref in known:
                edges.append(Edge("use", name, ref))
            elif ref in libs:
                edges.append(Edge("use", name, libs[ref]))
    return [e for e in edges if e.target != name]


def relations(classes: dict[str, ClassInfo]) -> list[Edge]:
    libs_by_module = library_names(SRC_DIR)
    raw: list[Edge] = []
    for info in classes.values():
        raw += class_edges(info, classes, libs_by_module.get(info.path, {}))
    for edge in find_realizations(classes):
        impl, proto = edge.split(" ..|> ")
        raw.append(Edge("realize", impl, proto))

    drawn = {n for n, c in classes.items() if not left_out(c, classes)} | set(
        LIBRARIES.values()
    )
    raw = [e for e in raw if e.source in drawn and e.target in drawn]

    # The strongest relation for each ordered pair, the first label met for it.
    best: dict[tuple[str, str], Edge] = {}
    for e in raw:
        key = (e.source, e.target)
        if key not in best or STRENGTH[e.kind] > STRENGTH[best[key].kind]:
            best[key] = e
    edges = list(best.values())

    # A use is dropped where a stronger arrow joins the pair either way round.
    strong = {frozenset((e.source, e.target)) for e in edges if e.kind != "use"}
    edges = [
        e
        for e in edges
        if e.kind != "use" or frozenset((e.source, e.target)) not in strong
    ]

    # A use is dropped where a base class or a realized protocol already has an arrow to it.
    realized = {(e.source, e.target) for e in edges if e.kind == "realize"}
    out_of = {}
    for e in edges:
        out_of.setdefault(e.source, set()).add(e.target)
    kept: list[Edge] = []
    for e in edges:
        if e.kind == "use":
            parents = ancestors(e.source, classes) | {
                p for s, p in realized if s == e.source
            }
            if any(e.target in out_of.get(p, set()) for p in parents):
                continue
        kept.append(e)
    return sorted(kept, key=lambda e: (-STRENGTH[e.kind], e.source, e.target))


# ================= #
#      Writing      #
# ================= #


def mark(name: str, classes: dict[str, ClassInfo]) -> str | None:
    if name in set(LIBRARIES.values()):
        return "library"
    info = classes.get(name)
    if info is None:
        return None
    if info.is_protocol:
        return "protocol"
    if is_abstract(info):
        return "abstract"
    return None


def write(
    path: Path,
    boxes: dict[str, str | None],
    edges: list[Edge],
    classes: dict[str, ClassInfo],
    home: str | None,
) -> None:
    """Write one diagram: `boxes` maps each class to its layer, or None for a library.

    `home` is the layer the diagram is about; without namespaces, a class from another layer is
    marked `<<outside>>`.
    """
    lines = ["classDiagram"]
    marks: list[str] = []
    if NAMESPACES:
        for layer in LAYERS:
            members = sorted(n for n, lay in boxes.items() if lay == layer)
            if not members:
                continue
            groups: dict[str | None, list[str]] = {}
            for n in members:
                groups.setdefault(subgroup_of(classes[n]), []).append(n)
            order = [g for g, _ in SUBGROUPS.get(layer, ()) if g in groups]
            if NESTED:
                lines.append(f"  namespace {layer} {{")
                lines += [f"    class {n}" for n in groups.get(None, [])]
                for g in order:
                    lines.append(f"    namespace {g} {{")
                    lines += [f"      class {n}" for n in groups[g]]
                    lines.append("    }")
                lines.append("  }")
                continue
            if None in groups:
                lines.append(f"  namespace {layer} {{")
                lines += [f"    class {n}" for n in groups[None]]
                lines.append("  }")
            for g in order:
                lines.append(f"  namespace {layer}_{g} {{")
                lines += [f"    class {n}" for n in groups[g]]
                lines.append("  }")
        loose = sorted(n for n, lay in boxes.items() if lay is None)
        lines += [f"  class {n}" for n in loose]
        for n in sorted(boxes):
            if (m := mark(n, classes)) is not None:
                marks.append(f"  <<{m}>> {n}")
    else:
        for n in sorted(boxes):
            lines.append(f"  class {n}")
            m = (
                "outside"
                if home is not None and boxes[n] not in (home, None)
                else mark(n, classes)
            )
            if m is not None:
                marks.append(f"  <<{m}>> {n}")
    lines += marks
    lines += [f"  {e.mermaid()}" for e in edges]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    classes = scan(SRC_DIR)
    edges = relations(classes)
    layer = {n: layer_of(c) for n, c in classes.items() if not left_out(c, classes)}
    for box in set(LIBRARIES.values()):
        layer[box] = None

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for old in OUT_DIR.glob("*.mmd"):
        old.unlink()

    crossing = [e for e in edges if layer[e.source] != layer[e.target]]
    write(OUT_DIR / "overview.mmd", layer, crossing, classes, home=None)
    print(
        f"{(OUT_DIR / 'overview.mmd').relative_to(ROOT)}: {len(layer)} boxes, {len(crossing)} crossing arrows"
    )

    for name in LAYERS:
        inside = {n for n, lay in layer.items() if lay == name}
        touching = [e for e in edges if e.source in inside or e.target in inside]
        boxes = {n: layer[n] for e in touching for n in (e.source, e.target)} | {
            n: name for n in inside
        }
        out = OUT_DIR / f"{name}.mmd"
        write(out, boxes, touching, classes, home=name)
        print(
            f"{out.relative_to(ROOT)}: {len(inside)} classes, {len(boxes) - len(inside)} outside, {len(touching)} arrows"
        )

    counts: dict[str, int] = {}
    for e in edges:
        counts[e.kind] = counts.get(e.kind, 0) + 1
    print("relations:", ", ".join(f"{k} {counts.get(k, 0)}" for k in STRENGTH))


if __name__ == "__main__":
    main()
