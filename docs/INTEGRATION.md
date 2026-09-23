# Tool and Evaluation Integration

## Data contract

One evaluation row is a JSON object with `note`, `images`, and `metadata`.
`metadata` supplies `note_id`, `source`, `candidate_labels`, and optional `labels`
for evaluation. Image paths must be readable on the machine running the agent.
Do not expose GT annotations or human-review reasons in the policy-model prompt.

```json
{
  "note": "A synthetic example, not a real post.",
  "images": [],
  "metadata": {
    "note_id": "synthetic-001",
    "source": "synthetic",
    "candidate_labels": ["通过", "DEMO_UNVERIFIED"],
    "labels": ["通过"]
  }
}
```

Training additionally supports `metadata.audit_input` as the canonical structured
input. Four review votes live under `metadata.four_round_labels.rounds`; the rule
judge verdict lives under `metadata.rule_consistency_judge.verdict`. Inspect
`AuditInput.from_data_row` and `extract_four_round_labels` for the exact contract.

## Add a tool

Subclass `audit_agentic.environment.tools.base_tool.BaseTool`, then implement:

1. `name`, `description`, and `public_input_schema` (JSON Schema).
2. `run(args) -> dict`: retrieve structured evidence.
3. `render(result) -> ToolRenderResult`: render text plus images. Each `<image>`
   marker must correspond to exactly one image, in the same order.

Register the instance in a mapping and pass the same mapping to `ToolExecutor`
and `AgentScopeAuditAgent`. See `examples/synthetic_audit.py`. The public schema
should omit runtime-bound identifiers: the adapter supplies case context.
Distinguish service failure, empty evidence, and a verified negative result.

The included business tools demonstrate adapter shapes, not public services.
Online business access is disabled by default. If you implement online adapters,
configure the relevant endpoints explicitly: `AUDIT_ONLINE_SERVICE_BASE`,
`AUDIT_RULE_SERVICE_URL`, `AUDIT_FACTOR_SERVICE_URL`, or
`AUDIT_MEDIA_RESOLVER_URL`. Optional tokens belong in environment variables,
not source files. Cached-only operation is recommended for reproducibility.

## AgentScope evaluation

After preparing your own JSONL, rule cache, and tool cache, create
`audit_agentic/config.local.json` based on `config.example.json`. Set
`main_agent.api_base` and `main_agent.model`; set `api_key` to an empty string
and `api_key_env` to `OPENAI_API_KEY` to use a secret environment variable.

```bash
export PYTHONPATH="$PWD:$PWD/agentscope/src:${PYTHONPATH:-}"
python -m audit_agentic.eval.run_agentscope_agent_eval \
  --input /path/to/eval.jsonl \
  --config audit_agentic/config.local.json \
  --rule-cache /path/to/rules \
  --tool-cache /path/to/tool-cache \
  --output-dir local_artifacts/eval \
  --no-experience --concurrency 8 --max-tokens 8192 --full-trace
```

Use `--help` to inspect all options. The internal Source400/test-local launchers
and dataset-specific selection logic are intentionally not included. The generic
entry point above is retained. An output directory must belong to only one
evaluation configuration; use a new directory when changing the model or prompt.

## Experience and judging

The agent supports queue-level experience and label-level experience in rule
previews. Supply your own file via `AUDIT_EXPERIENCE_PATH`; no learned experience
is distributed. Freeze selected versions before held-out evaluation.

Process judging should receive the full observable trajectory, including image
content, tool evidence and final decision. Do not send only generated reasoning
or infer image access from a model name. Verify media counts, successful image
resolution, structured-score parsing, and failure rates before training.

Prompt variants and candidate rubric dimensions are in `prompt_templates`.
Do not compare process-score means across different judges or rubrics as though
they were a calibrated common scale. Validate discriminative power, correctness,
and possible reward hacking together.
