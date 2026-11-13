"""Shared execution lifecycle; physics and result extraction stay in problems."""

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import subprocess
import time
from typing import Callable
import uuid

from .runtime import Runtime, SimulationError


def write_json(path, payload):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, allow_nan=False)
        stream.write("\n")


@dataclass(frozen=True)
class CaseDefinition:
    """Problem-owned hooks and data consumed by CaseEvaluator.

    validate checks geometry and pinned sources without executing a solver.
    execute meshes/solves/extracts results using the resolved runtime, updating
    the supplied result so partial diagnostics survive a failure.
    """
    name: str
    config_type: type
    inputs: frozenset[str]
    commands: tuple[str, ...]
    validate: Callable
    execute: Callable
    generate: Callable
    extract: Callable
    revision: str | None = None
    execution_outputs: tuple[str, ...] = ()
    timeout_s: float = 1800


class CaseEvaluator:
    """Create unique cases, validate inputs, run once, and retain failure records."""

    def __init__(self, definition: CaseDefinition, runtime=None):
        self.definition = definition
        self.runtime = runtime or Runtime(timeout_s=definition.timeout_s)

    def evaluate(self, *case_args, output_root="runs"):
        case = Path(output_root).resolve() / (self.definition.name + "-" + uuid.uuid4().hex)
        self.definition.generate(case, *case_args)
        return self.run_case(case)

    def _load_case(self, case):
        """Verify the same immutable inputs for execution and offline extraction."""
        spec = self.definition
        manifest = json.loads((case / "case.json").read_text(encoding="utf-8"))
        config = spec.config_type(**manifest["config"])
        hashes = manifest.get("input_sha256", {})
        if (manifest.get("problem") != spec.name
                or manifest.get("config_sha256") != config.fingerprint
                or set(hashes) != spec.inputs
                or (spec.revision is not None and manifest.get("model_revision") != spec.revision)):
            raise ValueError(f"Invalid {spec.name} manifest")
        for name, digest in hashes.items():
            relative = Path(name)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Invalid case input path: {name}")
            if hashlib.sha256((case / relative).read_bytes()).hexdigest() != digest:
                raise ValueError(f"Case input changed: {name}; generate a fresh case")
        spec.validate(case, manifest, config)
        return manifest, config

    def run_case(self, case):
        case = Path(case).resolve()
        spec = self.definition
        manifest, config = self._load_case(case)
        outputs = ("postProcessing", "result.json") + spec.execution_outputs
        if any((case / name).exists() for name in outputs):
            raise FileExistsError("Case already contains execution data; generate a fresh case")
        for path in case.iterdir():
            try:
                old_fields = path.is_dir() and float(path.name) > 0
            except ValueError:
                old_fields = False
            if old_fields:
                raise FileExistsError("Case contains old fields; generate a fresh case")
        with (case / ".execution-started").open("x", encoding="utf-8") as stream:
            stream.write(datetime.now(timezone.utc).isoformat() + "\n")
        result = dict(manifest, status="failed", case_directory=str(case),
                      created_utc=datetime.now(timezone.utc).isoformat())
        started = time.monotonic()
        try:
            info = self.runtime.resolve(commands=spec.commands)
            result["runtime"] = info
            spec.execute(case, config, manifest, self.runtime, info, result)
            result.update({"status": "ok", **spec.extract(case, config)})
        except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
            result.update(status="failed", error=str(exc))
            raise SimulationError(f"{spec.name} failed in {case}: {exc}", case_directory=case) from exc
        except KeyboardInterrupt:
            result.update(status="failed", error="Interrupted by user")
            raise
        finally:
            result["wall_time_s"] = time.monotonic() - started
            write_json(case / "result.json", result)
        return result

    def extract_results(self, case):
        """Read saved OpenFOAM output without launching CFD or writing files."""
        case = Path(case).resolve()
        _, config = self._load_case(case)
        return self.definition.extract(case, config)
