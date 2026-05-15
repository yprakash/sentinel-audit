from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

_jinja_env = None
TEMPLATES_DIR = Path("templates")


def build_jinja_env() -> Environment:
    global _jinja_env
    if _jinja_env:
        return _jinja_env

    _jinja_env = Environment(
        loader=FileSystemLoader(TEMPLATES_DIR),
        undefined=StrictUndefined,
        trim_blocks=True,
        lstrip_blocks=True,
        autoescape=False,
    )
    _jinja_env.filters["severity_badge"] = severity_badge
    _jinja_env.filters["newline_to_br"] = newline_to_br

    return _jinja_env


def severity_badge(value: str) -> str:
    mapping = {
        "critical": "🔴 Critical",
        "high": "🟠 High",
        "medium": "🟡 Medium",
        "low": "🔵 Low",
        "informational": "⚪ Informational",
    }
    return mapping.get(value.lower(), value)


def newline_to_br(value: str) -> str:
    return value.replace("\n", "<br>")


def render_template(
        template_name: str,
        context: dict[str, Any],
) -> str:
    env = build_jinja_env()

    template = env.get_template(template_name)

    return template.render(**context)


def render_to_file(
        template_name: str,
        context: dict[str, Any],
        output_path: str | Path,
) -> Path:
    rendered = render_template(template_name, context)

    output_path = Path(output_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(rendered, encoding="utf-8")

    return output_path
