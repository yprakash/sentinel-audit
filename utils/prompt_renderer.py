import json
from string import Template
from typing import Any

from pydantic import BaseModel


def normalize(data: Any):
    if isinstance(data, BaseModel):
        return data.model_dump(exclude_none=True)

    if isinstance(data, list):
        return [normalize(v) for v in data]

    if isinstance(data, dict):
        return {
            k: normalize(v)
            for k, v in data.items()
        }

    return data


def to_pretty_json(data: Any) -> str:
    return json.dumps(
        normalize(data),
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def render_prompt(template: str, **kwargs) -> str:
    serialized_kwargs = {
        k: (
            v
            if isinstance(v, str)
            else to_pretty_json(v)
        )
        for k, v in kwargs.items()
    }

    tmpl = Template(template)

    # safe_substitute avoids KeyError if a variable is missing
    return tmpl.safe_substitute(**serialized_kwargs)
