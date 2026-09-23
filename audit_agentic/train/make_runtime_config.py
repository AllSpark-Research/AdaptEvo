"""Create a run-local config with optional LLM endpoint/model overrides.

The eval scripts consume one config file containing ``main_agent`` and
``planner_agent`` sections. This helper keeps ``audit_agentic/config.json`` as
the student/default config, while making it cheap to run the same pipeline with
a teacher endpoint.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


TEACHER_DEFAULTS = {
    "api_base": "http://192.0.2.1:21237/v1",
    "model": "Qwen3.5-397B-A17B-FP8",
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base-config", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument(
        "--profile",
        choices=["student", "teacher", "custom"],
        default="student",
        help="student keeps base config unless explicit overrides are passed; teacher uses built-in teacher defaults.",
    )
    p.add_argument("--api-base", default="", help="Override OpenAI-compatible /v1 base URL.")
    p.add_argument("--model", default="", help="Override served model name.")
    p.add_argument("--timeout", type=float, default=0, help="Override request timeout when >0.")
    p.add_argument("--max-retries", type=int, default=0, help="Override max retries when >0.")
    p.add_argument(
        "--agents",
        default="main_agent,planner_agent,judge_agent",
        help="Comma-separated config sections to override.",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg: dict[str, Any] = json.loads(args.base_config.read_text(encoding="utf-8"))
    out_cfg = copy.deepcopy(cfg)

    overrides: dict[str, Any] = {}
    if args.profile == "teacher":
        overrides.update(TEACHER_DEFAULTS)
    if args.api_base:
        overrides["api_base"] = args.api_base
    if args.model:
        overrides["model"] = args.model
    if args.timeout > 0:
        overrides["timeout"] = args.timeout
    if args.max_retries > 0:
        overrides["max_retries"] = args.max_retries

    agents = [x.strip() for x in args.agents.split(",") if x.strip()]
    for name in agents:
        if name not in out_cfg or not isinstance(out_cfg[name], dict):
            continue
        out_cfg[name].update(overrides)

    out_cfg["_runtime_profile"] = {
        "profile": args.profile,
        "base_config": str(args.base_config),
        "overrides": overrides,
        "agents": agents,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_cfg, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out_cfg["_runtime_profile"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
