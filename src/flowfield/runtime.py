"""Run v2312 serial utilities with bounded time and per-case logs."""

from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

DEFAULT_IMAGE = "opencfd/openfoam-default:2312"
DEFAULT_COMMANDS = {"blockMesh", "checkMesh", "simpleFoam"}
COMMANDS = DEFAULT_COMMANDS | {"gmshToFoam"}


class SimulationError(RuntimeError):
    """A CFD execution or verification failure; never an analytical fallback."""

    def __init__(self, message, *, case_directory=None):
        super().__init__(message)
        self.case_directory = str(case_directory) if case_directory is not None else None


@dataclass(frozen=True)
class Runtime:
    backend: str = "auto"
    image: str = DEFAULT_IMAGE
    timeout_s: float = 600

    def __post_init__(self):
        if self.backend not in ("auto", "native", "docker"):
            raise ValueError("backend must be auto, native or docker")
        if not math.isfinite(self.timeout_s) or self.timeout_s <= 0:
            raise ValueError("timeout_s must be positive and finite")

    def resolve(self, commands=None):
        commands = DEFAULT_COMMANDS if commands is None else set(commands)
        if not commands or not commands <= COMMANDS:
            raise ValueError("Unsupported OpenFOAM command set")
        backend = self.backend
        if backend == "auto":
            backend = "native" if os.environ.get("WM_PROJECT_VERSION") in ("2312", "v2312") else "docker"
        if backend == "native":
            if os.environ.get("WM_PROJECT_VERSION") not in ("2312", "v2312"):
                raise SimulationError("Source OpenCFD OpenFOAM v2312 before selecting native execution")
            missing = [command for command in sorted(commands) if not shutil.which(command)]
            if missing:
                raise SimulationError(f"Missing native OpenFOAM commands: {missing}")
            info = {"backend": backend, "openfoam_version": os.environ["WM_PROJECT_VERSION"],
                    "executables": {command: shutil.which(command) for command in sorted(commands)}}
        else:
            if not shutil.which("docker"):
                raise SimulationError("Docker is unavailable; load OpenFOAM v2312 or install/start Docker")
            try:
                probe = subprocess.run(["docker", "image", "inspect", self.image],
                    check=True, capture_output=True, text=True, timeout=min(self.timeout_s, 30))
                details = json.loads(probe.stdout)[0]
            except (OSError, subprocess.SubprocessError, ValueError, IndexError) as exc:
                raise SimulationError(f"Cannot inspect Docker image {self.image}; start Docker and pull it explicitly") from exc
            # Execute the inspected immutable image ID, even if its tag changes later.
            info = {"backend": backend, "image": self.image, "image_id": details["Id"],
                    "repo_digests": details.get("RepoDigests", []), "architecture": details.get("Architecture")}
        return info

    def run(self, case, command, args=(), *, info=None):
        if command not in COMMANDS:
            raise ValueError(f"Unsupported OpenFOAM command: {command}")
        info = info or self.resolve()
        case = Path(case).resolve()
        name = "flowfield-" + uuid.uuid4().hex
        if info["backend"] == "native":
            argv = [info["executables"][command], "-case", str(case), *args]
        else:
            if "," in str(case):
                raise SimulationError("Docker bind paths cannot contain commas")
            argv = ["docker", "run", "--rm", "--pull", "never", "--init", "--name", name,
                "--user", f"{os.getuid()}:{os.getgid()}", "--env", "HOME=/tmp", "--env", "USER=foam",
                "--workdir", "/case", "--mount", f"type=bind,source={case},target=/case",
                "--entrypoint", "/bin/bash", info["image_id"], "-c",
                'flowfield_args=("$@"); set --; source /usr/lib/openfoam/openfoam2312/etc/bashrc || exit $?; '
                '[[ "$WM_PROJECT_VERSION" == v2312 || "$WM_PROJECT_VERSION" == 2312 ]] || exit 90; '
                'exec "${flowfield_args[@]}"', "bash", command, "-case", "/case", *args]
        log = case / f"log.{command}"
        try:
            with log.open("x", encoding="utf-8") as stream:
                subprocess.run(argv, stdout=stream, stderr=subprocess.STDOUT, check=True,
                               timeout=self.timeout_s, cwd=case)
        except (subprocess.SubprocessError, OSError, KeyboardInterrupt) as exc:
            if info["backend"] == "docker":
                # A killed Docker client need not kill its container. Clean only our own named run.
                try:
                    subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=30)
                except (subprocess.SubprocessError, OSError):
                    pass
            if isinstance(exc, KeyboardInterrupt):
                raise
            raise SimulationError(f"{command} failed ({type(exc).__name__}); inspect {log}") from exc
        return log
