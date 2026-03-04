import json
from string import Template
from typing import Any


def to_pretty_json(data: Any) -> str:
    return json.dumps(data, indent=2, ensure_ascii=False)


def render_prompt(template: str, **kwargs) -> str:
    serialized_kwargs = {
        k: to_pretty_json(v) if not isinstance(v, str) else v
        for k, v in kwargs.items()
    }

    tmpl = Template(template)

    # safe_substitute avoids KeyError if a variable is missing
    return tmpl.safe_substitute(**serialized_kwargs)
