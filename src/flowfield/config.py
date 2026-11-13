"""JSON transport for validated, problem-specific configuration objects."""

import json
from pathlib import Path


def load_config(config_type, source=None, **overrides):
    """Load JSON or a mapping, then apply the config's physical validation."""
    if source is None:
        values = {}
    elif isinstance(source, config_type):
        values = source.to_dict()
    elif isinstance(source, (str, Path)):
        values = json.loads(Path(source).read_text(encoding="utf-8"))
    else:
        values = source
    if not isinstance(values, dict):
        raise ValueError("Configuration must be a JSON object or a mapping")
    return config_type(**(values | overrides))
