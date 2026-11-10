"""Run the shared FlowField experiment runner from any working directory."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from flowfield.experiments import main


if __name__ == "__main__":
    raise SystemExit(main())
