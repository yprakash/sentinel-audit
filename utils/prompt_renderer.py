import json
from typing import Any


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def render_prompt(template: str, **kwargs) -> str:
    """
    Centralized prompt renderer.
    Handles JSON serialization safely.
    Prevents accidental f-string injection.
    """
    serialized_kwargs = {
        k: to_pretty_json(v) if not isinstance(v, str) else v
        for k, v in kwargs.items()
    }

    return template.format(**serialized_kwargs)
