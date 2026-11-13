"""Mesh output contract shared by function-based and stateful generators."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class PolyMesh:
    """An OpenFOAM mesh directory, not a blockMeshDict or intermediate CAD file.

    File presence is checked here; physical/topological quality is checked with
    checkMesh and problem-specific diagnostics before running the solver.
    """
    directory: Path
    required_files = ("points", "faces", "owner", "neighbour", "boundary")

    def __post_init__(self):
        path = Path(self.directory).resolve()
        missing = [name for name in self.required_files if not (path / name).is_file()]
        if missing:
            raise ValueError(f"Incomplete polyMesh in {path}: {', '.join(missing)}")
        object.__setattr__(self, "directory", path)

    @classmethod
    def from_case(cls, case):
        return cls(Path(case) / "constant/polyMesh")

    def checksums(self):
        return {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                for p in sorted(self.directory.iterdir()) if p.is_file()}


class MeshGenerator(Protocol):
    """A callable contract; plain functions satisfy it without subclassing."""
    def __call__(self, case: Path, geometry, config) -> PolyMesh: ...


def block_mesh(case, runtime, info, args=()) -> PolyMesh:
    runtime.run(case, "blockMesh", args, info=info)
    return PolyMesh.from_case(case)
