from enum import Enum
from pathlib import Path
from typing import Any

from pydantic import BaseModel


def jsonable(data: Any):
    if isinstance(data, BaseModel):
        return data.model_dump(mode="json")

    if isinstance(data, type) and issubclass(data, BaseModel):
        return data.__name__

    if isinstance(data, Enum):
        return data.value

    if isinstance(data, Path):
        return str(data)

    if isinstance(data, dict):
        return {
            k: jsonable(v)
            for k, v in data.items()
        }

    if isinstance(data, list):
        return [jsonable(v) for v in data]

    if isinstance(data, tuple):
        return [jsonable(v) for v in data]

    return data
