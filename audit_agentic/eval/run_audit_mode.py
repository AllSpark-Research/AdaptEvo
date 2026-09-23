"""Unified launcher for the supported audit evaluation modes.

The repository has three maintained evaluation routes:

* ``four-stage``: Router -> Planner -> Tools -> Final.
* ``four-stage-history-gate``: Four-stage plus a mandatory Train-history
  boundary review after Final.
* ``agentscope``: one AgentScope ReAct agent with native tool calls and
  structured final output.

The retired hand-written Native Agent loop is intentionally not exposed here.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any


AUDIT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_ROOT = AUDIT_ROOT.parent
AGENTICRL_ROOT = PROJECT_ROOT.parent
FOUR_STAGE_MODE = "four-stage"
FOUR_STAGE_HISTORY_GATE_MODE = "four-stage-history-gate"
AGENTSCOPE_MODE = "agentscope"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Four-stage, Four-stage + History Gate, or AgentScope."
    )
    parser.add_argument(
        "--mode",
        choices=[FOUR_STAGE_MODE, FOUR_STAGE_HISTORY_GATE_MODE, AGENTSCOPE_MODE],
        required=True,
    )
    parser.add_argument("--input", default=str(AUDIT_ROOT / "data" / "test_local.jsonl"))
    parser.add_argument("--config", default=str(AUDIT_ROOT / "config.json"))
    parser.add_argument("--output-root", default=str(AUDIT_ROOT / "traces" / "eval_runs"))
    parser.add_argument("--tag", default="")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument(
        "--row-indices",
        default="",
        help="Comma-separated original JSONL row indices for AgentScope eval.",
    )
    parser.add_argument(
        "--sources",
        default="",
        help="Comma-separated source allowlist for AgentScope eval.",
    )
    parser.add_argument("--concurrency", type=int)
    parser.add_argument("--router-concurrency", type=int)
    parser.add_argument("--planner-concurrency", type=int)
    parser.add_argument("--final-concurrency", type=int)
    parser.add_argument("--tool-concurrency", type=int)
    parser.add_argument("--image-max-tokens", type=int, default=448)
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument(
        "--experience-mode",
        choices=["auto", "none", "final", "personalized", "all"],
        default="auto",
        help=(
            "four-stage: none/final/personalized/all; AgentScope: none/final. "
            "auto resolves to personalized for four-stage and final for AgentScope."
        ),
    )
    parser.add_argument(
        "--experience-path",
        default=str(
            AUDIT_ROOT / "environment" / "cache" / "experience" / "audit_experience.json"
        ),
    )
    parser.add_argument("--rule-cache")
    parser.add_argument("--tool-cache")
    parser.add_argument(
        "--tool-cache-version",
        default=os.environ.get("AUDIT_TOOL_CACHE_VERSION", "tool-cache-v2-example_user"),
    )
    parser.add_argument(
        "--tool-cache-identity-mode",
        choices=["default", "note_history"],
        default=os.environ.get("AUDIT_TOOL_CACHE_IDENTITY_MODE", "default"),
    )
    parser.add_argument("--image-cache")
    parser.add_argument("--blob-index")
    parser.add_argument("--plagiarism-cache")
    parser.add_argument(
        "--enable-rule-images",
        action=argparse.BooleanOptionalAction,
        default=None,
    )

    # Four-stage History Gate controls.
    parser.add_argument(
        "--history-data",
        default=str(AUDIT_ROOT / "data" / "train_local.jsonl"),
    )
    parser.add_argument(
        "--history-context-mode",
        choices=["legacy_compact", "full_multimodal"],
        default="legacy_compact",
    )
    parser.add_argument("--history-cases-per-class", type=int, default=3)
    parser.add_argument("--history-concurrency", type=int, default=64)
    parser.add_argument("--history-image-max-tokens", type=int, default=448)
    parser.add_argument("--history-gate-max-tokens", type=int, default=2048)

    # AgentScope-specific controls.
    parser.add_argument("--thinking", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--max-iters", type=int, default=8)
    parser.add_argument("--structured-output-grace-iters", type=int, default=2)
    parser.add_argument("--router-mode", choices=["none", "agentscope"], default="none")
    parser.add_argument("--router-bypass-max-labels", type=int, default=8)
    parser.add_argument(
        "--rule-loading-mode",
        choices=["all", "preview_tool"],
        default="preview_tool",
    )
    parser.add_argument("--rule-preview-max-chars", type=int, default=700)
    parser.add_argument("--detail-rule-max-labels", type=int, default=6)
    parser.add_argument("--parallel-tool-calls", action="store_true")
    parser.add_argument(
        "--context-compression",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Enable AgentScope context compression. When omitted, preserve "
            "the evaluator default."
        ),
    )
    parser.add_argument(
        "--compression-trigger-tokens",
        type=int,
        default=0,
        help=(
            "Absolute AgentScope compression threshold. 0 keeps the native "
            "80%% context-window threshold."
        ),
    )
    parser.add_argument(
        "--compression-profile",
        choices=["default", "audit"],
        default="default",
        help="AgentScope compression prompt/schema profile.",
    )
    parser.add_argument("--full-trace", action="store_true")
    parser.add_argument(
        "--agentscope-python",
        help="Python interpreter containing AgentScope; defaults to the shared repository venv.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def positive(name: str, value: int | float | None) -> None:
    if value is not None and value < 1:
        raise SystemExit(f"ERROR: {name} must be >= 1")


def sha256_file(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_experience(mode: str, requested: str) -> str:
    if requested == "auto":
        return (
            "personalized"
            if mode in {FOUR_STAGE_MODE, FOUR_STAGE_HISTORY_GATE_MODE}
            else "final"
        )
    if mode == AGENTSCOPE_MODE and requested not in {"none", "final"}:
        raise SystemExit(
            "ERROR: AgentScope has one decision agent, so --experience-mode only accepts none or final"
        )
    return requested


def build_environment(args: argparse.Namespace) -> tuple[dict[str, str], dict[str, str]]:
    env = os.environ.copy()
    overrides: dict[str, str] = {}
    path_mapping = {
        "AUDIT_RULE_CACHE_PATH": args.rule_cache,
        "AUDIT_TOOL_RESULT_CACHE_PATH": args.tool_cache,
        "IMAGE_CACHE_DIR": args.image_cache,
        "BLOB_INDEX_PATH": args.blob_index,
        "PLAGIARISM_CACHE_PATH": args.plagiarism_cache,
    }
    for key, value in path_mapping.items():
        if value:
            overrides[key] = str(Path(value).expanduser().resolve())
    literal_mapping = {
        "AUDIT_TOOL_CACHE_VERSION": args.tool_cache_version,
        "AUDIT_TOOL_CACHE_IDENTITY_MODE": args.tool_cache_identity_mode,
    }
    for key, value in literal_mapping.items():
        if value:
            overrides[key] = str(value)
    if args.enable_rule_images is not None:
        overrides["AUDIT_ENABLE_RULE_IMAGES"] = "1" if args.enable_rule_images else "0"
    env.update(overrides)
    return env, overrides


def build_four_stage_command(
    args: argparse.Namespace,
    input_path: Path,
    config_path: Path,
    output_root: Path,
    tag: str,
    experience_mode: str,
    experience_path: Path,
) -> list[str]:
    unsupported = {
        "--thinking": args.thinking,
        "--temperature": args.temperature,
        "--max-tokens": args.max_tokens,
        "--parallel-tool-calls": args.parallel_tool_calls,
        "--retry-failed": args.retry_failed,
        "--full-trace": args.full_trace,
    }
    used = [name for name, value in unsupported.items() if value is not None and value is not False]
    if used:
        raise SystemExit(
            f"ERROR: four-stage does not accept AgentScope-only options: {', '.join(used)}"
        )

    command = [
        "bash",
        str(AUDIT_ROOT / "train" / "four_stage" / "run_all_steps.sh"),
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--output-root",
        str(output_root),
        "--tag",
        tag,
        "--limit",
        str(args.limit),
        "--image-max-tokens",
        str(args.image_max_tokens),
        "--experience-mode",
        experience_mode,
        "--experience-path",
        str(experience_path),
        "--resume" if args.resume else "--no-resume",
    ]
    shared = args.concurrency
    values = {
        "--router-concurrency": args.router_concurrency or shared,
        "--planner-concurrency": args.planner_concurrency or shared,
        "--final-concurrency": args.final_concurrency or shared,
        "--tool-concurrency": args.tool_concurrency,
    }
    for name, value in values.items():
        if value is not None:
            command.extend([name, str(value)])
    if args.mode == FOUR_STAGE_HISTORY_GATE_MODE:
        command.extend(
            [
                "--history-gate",
                "--history-data",
                str(Path(args.history_data).expanduser().resolve()),
                "--history-context-mode",
                args.history_context_mode,
                "--history-cases-per-class",
                str(args.history_cases_per_class),
                "--history-concurrency",
                str(args.history_concurrency),
                "--history-image-max-tokens",
                str(args.history_image_max_tokens),
                "--history-gate-max-tokens",
                str(args.history_gate_max_tokens),
            ]
        )
    return command


def resolve_agentscope_python(args: argparse.Namespace) -> Path:
    configured = args.agentscope_python or os.environ.get("AGENTSCOPE_PYTHON")
    if configured:
        return Path(configured).expanduser().absolute()
    return (AGENTICRL_ROOT / "agentscope" / ".venv" / "bin" / "python").absolute()


def build_agentscope_command(
    args: argparse.Namespace,
    input_path: Path,
    config_path: Path,
    run_dir: Path,
    experience_mode: str,
    experience_path: Path,
) -> tuple[list[str], dict[str, Any]]:
    stage_concurrency = {
        "--router-concurrency": args.router_concurrency,
        "--planner-concurrency": args.planner_concurrency,
        "--final-concurrency": args.final_concurrency,
        "--tool-concurrency": args.tool_concurrency,
    }
    used = [name for name, value in stage_concurrency.items() if value is not None]
    if used:
        raise SystemExit(
            f"ERROR: AgentScope uses only --concurrency; unsupported: {', '.join(used)}"
        )

    python = resolve_agentscope_python(args)
    if not python.is_file():
        raise SystemExit(f"ERROR: AgentScope Python not found: {python}")

    thinking = True if args.thinking is None else bool(args.thinking)
    rule_cache = Path(
        args.rule_cache
        or os.environ.get("AUDIT_RULE_CACHE_PATH")
        or AUDIT_ROOT / "environment" / "cache" / "rules" / "test"
    ).expanduser().resolve()
    tool_cache = Path(
        args.tool_cache
        or os.environ.get("AUDIT_TOOL_RESULT_CACHE_PATH")
        or AUDIT_ROOT / "environment" / "cache" / "test_jhx"
    ).expanduser().resolve()
    command = [
        str(python),
        "-m",
        "audit_agentic.eval.run_agentscope_agent_eval",
        "--input",
        str(input_path),
        "--config",
        str(config_path),
        "--output-dir",
        str(run_dir),
        "--limit",
        str(args.limit),
        "--image-max-tokens",
        str(args.image_max_tokens),
        "--timeout",
        str(args.timeout),
        "--max-iters",
        str(args.max_iters),
        "--structured-output-grace-iters",
        str(args.structured_output_grace_iters),
        "--router-mode",
        args.router_mode,
        "--router-bypass-max-labels",
        str(args.router_bypass_max_labels),
        "--rule-loading-mode",
        args.rule_loading_mode,
        "--rule-preview-max-chars",
        str(args.rule_preview_max_chars),
        "--detail-rule-max-labels",
        str(args.detail_rule_max_labels),
        "--rule-cache",
        str(rule_cache),
        "--tool-cache",
        str(tool_cache),
        "--tool-cache-version",
        args.tool_cache_version,
        "--tool-cache-identity-mode",
        args.tool_cache_identity_mode,
        "--experience-path",
        str(experience_path),
        "--experience" if experience_mode == "final" else "--no-experience",
        "--thinking" if thinking else "--no-thinking",
        "--resume" if args.resume else "--no-resume",
    ]
    if args.concurrency is not None:
        command.extend(["--concurrency", str(args.concurrency)])
    if args.row_indices:
        command.extend(["--row-indices", args.row_indices])
    if args.sources:
        command.extend(["--sources", args.sources])
    if args.retry_failed:
        command.append("--retry-failed")
    if args.temperature is not None:
        command.extend(["--temperature", str(args.temperature)])
    if args.max_tokens is not None:
        command.extend(["--max-tokens", str(args.max_tokens)])
    if args.parallel_tool_calls:
        command.append("--parallel-tool-calls")
    if args.context_compression is not None:
        command.append(
            "--context-compression"
            if args.context_compression
            else "--no-context-compression"
        )
    if args.compression_trigger_tokens:
        command.extend(
            [
                "--compression-trigger-tokens",
                str(args.compression_trigger_tokens),
            ]
        )
    command.extend(["--compression-profile", args.compression_profile])
    if args.full_trace:
        command.append("--full-trace")

    effective = {
        "python": str(python),
        "thinking": thinking,
        "router_mode": args.router_mode,
        "rule_loading_mode": args.rule_loading_mode,
            "rule_cache": str(rule_cache),
            "tool_cache": str(tool_cache),
            "tool_cache_version": args.tool_cache_version,
            "tool_cache_identity_mode": args.tool_cache_identity_mode,
        "context_compression": (
            True
            if args.context_compression is None
            else bool(args.context_compression)
        ),
        "compression_trigger_tokens": args.compression_trigger_tokens,
        "compression_profile": args.compression_profile,
        "row_indices": args.row_indices,
        "sources": args.sources,
    }
    return command, effective


def main() -> None:
    args = parse_args()
    for name in (
        "concurrency",
        "router_concurrency",
        "planner_concurrency",
        "final_concurrency",
        "tool_concurrency",
        "image_max_tokens",
        "max_tokens",
        "timeout",
        "max_iters",
        "structured_output_grace_iters",
        "router_bypass_max_labels",
        "rule_preview_max_chars",
        "detail_rule_max_labels",
        "history_cases_per_class",
        "history_concurrency",
        "history_image_max_tokens",
        "history_gate_max_tokens",
    ):
        positive(f"--{name.replace('_', '-')}", getattr(args, name))
    if args.limit < 0:
        raise SystemExit("ERROR: --limit must be >= 0")
    if args.mode != AGENTSCOPE_MODE and (args.row_indices or args.sources):
        raise SystemExit(
            "ERROR: --row-indices and --sources are currently supported only "
            "in AgentScope mode"
        )
    if args.compression_trigger_tokens < 0:
        raise SystemExit("ERROR: --compression-trigger-tokens must be >= 0")
    if (
        args.mode != AGENTSCOPE_MODE
        and (
            args.context_compression is not None
            or args.compression_trigger_tokens
            or args.compression_profile != "default"
        )
    ):
        raise SystemExit(
            "ERROR: context compression options are only supported in "
            "AgentScope mode"
        )
    if args.context_compression is False and args.compression_trigger_tokens:
        raise SystemExit(
            "ERROR: --compression-trigger-tokens requires "
            "--context-compression"
        )

    input_path = Path(args.input).expanduser().resolve()
    config_path = Path(args.config).expanduser().resolve()
    experience_path = Path(args.experience_path).expanduser().resolve()
    output_root = Path(args.output_root).expanduser().resolve()
    for name, path in (("input", input_path), ("config", config_path)):
        if not path.is_file():
            raise SystemExit(f"ERROR: {name} file not found: {path}")
    if args.mode == FOUR_STAGE_HISTORY_GATE_MODE:
        history_path = Path(args.history_data).expanduser().resolve()
        if not history_path.is_file():
            raise SystemExit(f"ERROR: history data file not found: {history_path}")

    experience_mode = resolve_experience(args.mode, args.experience_mode)
    if experience_mode != "none" and not experience_path.is_file():
        raise SystemExit(f"ERROR: experience file not found: {experience_path}")

    tag = args.tag or f"{args.mode}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    run_dir = output_root / tag
    run_dir.mkdir(parents=True, exist_ok=True)
    env, env_overrides = build_environment(args)

    mode_config: dict[str, Any] = {}
    if args.mode in {FOUR_STAGE_MODE, FOUR_STAGE_HISTORY_GATE_MODE}:
        command = build_four_stage_command(
            args,
            input_path,
            config_path,
            output_root,
            tag,
            experience_mode,
            experience_path,
        )
        if args.mode == FOUR_STAGE_HISTORY_GATE_MODE:
            mode_config = {
                "history_data": str(Path(args.history_data).expanduser().resolve()),
                "history_context_mode": args.history_context_mode,
                "history_cases_per_class": args.history_cases_per_class,
                "history_concurrency": args.history_concurrency,
                "history_image_max_tokens": args.history_image_max_tokens,
                "history_gate_max_tokens": args.history_gate_max_tokens,
            }
    else:
        command, mode_config = build_agentscope_command(
            args,
            input_path,
            config_path,
            run_dir,
            experience_mode,
            experience_path,
        )

    launcher_config: dict[str, Any] = {
        "created_at": datetime.now().astimezone().isoformat(),
        "mode": args.mode,
        "tag": tag,
        "run_dir": str(run_dir),
        "input": str(input_path),
        "input_sha256": sha256_file(input_path),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "experience_mode": experience_mode,
        "experience_path": str(experience_path),
        "experience_sha256": sha256_file(experience_path),
        "environment_overrides": env_overrides,
        "mode_config": mode_config,
        "command": command,
        "command_display": shlex.join(command),
        "dry_run": args.dry_run,
    }
    config_output = run_dir / "launcher_config.json"
    config_output.write_text(
        json.dumps(launcher_config, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(launcher_config, ensure_ascii=False, indent=2))
    if args.dry_run:
        return
    completed = subprocess.run(command, cwd=PROJECT_ROOT, env=env, check=False)
    raise SystemExit(completed.returncode)


if __name__ == "__main__":
    main()
