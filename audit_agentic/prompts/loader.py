"""Jinja2 prompt loader.

All prompts live under prompt_template/ as .jinja2 files. Python code only
loads + renders. Add a new prompt by dropping a new .jinja2 file in.

Custom filters:
* ``tojson`` is overridden to use ensure_ascii=False so Chinese characters
  render as-is (Jinja2's built-in tojson escapes non-ASCII).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined, TemplateNotFound


def _tojson_zh(value: Any, indent: int | None = None) -> str:
    """tojson that preserves Chinese (ensure_ascii=False)."""
    return json.dumps(value, ensure_ascii=False, indent=indent, sort_keys=True)


class PromptTemplateLoader:
    def __init__(self, template_dir: str | Path):
        self.template_dir = Path(template_dir).resolve()
        if not self.template_dir.exists():
            raise FileNotFoundError(f"Template dir not found: {self.template_dir}")
        self._env = Environment(
            loader=FileSystemLoader(str(self.template_dir)),
            autoescape=False,
            keep_trailing_newline=True,
            undefined=StrictUndefined,
        )
        # override built-in tojson to keep Chinese readable
        self._env.filters["tojson"] = _tojson_zh

    def render(self, template_name: str, **kwargs: Any) -> str:
        # Accept both "main_router" and "main_router.jinja2"
        if not template_name.endswith(".jinja2"):
            template_name = f"{template_name}.jinja2"
        try:
            tmpl = self._env.get_template(template_name)
        except TemplateNotFound as e:
            raise FileNotFoundError(
                f"Prompt template '{template_name}' not found under {self.template_dir}"
            ) from e
        return tmpl.render(**kwargs)

    def list_templates(self) -> list[str]:
        return sorted(self._env.list_templates(extensions=["jinja2"]))


def default_loader() -> PromptTemplateLoader:
    """Return loader pointing at audit_agentic/prompt_template/."""
    here = Path(__file__).resolve().parent.parent
    return PromptTemplateLoader(here / "prompt_template")
