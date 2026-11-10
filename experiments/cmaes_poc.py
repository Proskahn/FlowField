"""Budget-limited CMA-ES / feasible discrete suite entry point."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from flowfield.optimization import main

if __name__ == '__main__':
    raise SystemExit(main())
