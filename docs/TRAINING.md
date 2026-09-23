# Training Integration

This release includes the Relax framework source and audit integration, not
internal scheduler jobs or a portable one-command 32-GPU environment. Use the
upstream Relax installation guide to provision compatible PyTorch/CUDA, Ray,
Megatron and SGLang components. Model parallelism, tokenizer configuration,
checkpoint conversion and shared storage must match your deployment.

## Managed agent

Relax launches `audit_agentic/relax_app/run_agentscope_agent_app.sh` per managed
session. It supplies `RELAX_INPUT_JSON`, `RELAX_OUTPUT_JSON`, `RELAX_BASE_URL` and
`RELAX_SESSION_ID`. The adapter uses the session ID for the managed model calls,
exports the final reward and metadata, and preserves canonical multimodal
history for training. The process judge is a separate service, not part of the
trainable rollout token stream.

Use these arguments as an **integration fragment** in your validated Relax job;
they are not a complete launch command:

```bash
--use-agentic-rollout \
--agent-command "bash $PWD/audit_agentic/relax_app/run_agentscope_agent_app.sh" \
--agent-cwd "$PWD" \
--agentic-custom-advantage-path audit_agentic.relax_app.reward_agentscope.advantage_func
```

The framework entry module is `relax.entrypoints.train`. Keep the same absolute
checkout and data/media mounts available on every worker. Set `AGENTSCOPE_PYTHON`
to the interpreter containing inference dependencies and `AGENTSCOPE_ROOT` to
the bundled source directory. Both the controller and agents need this checkout
on `PYTHONPATH`.

## Reward and ablation controls

| Environment variable | Meaning |
|---|---|
| `AUDIT_ADAPTIVE_REWARD_ENABLED` | Enable confidence-aware mixture |
| `AUDIT_GT_CONFIDENCE_RHO` | Minimum process weight at full confidence, original formula |
| `AUDIT_GT_CONFIDENCE_GAMMA` | Confidence-to-weight exponent |
| `AUDIT_GT_WEIGHT_MIN`, `AUDIT_GT_WEIGHT_MAX` | Optional bounded-weight variant |
| `AUDIT_AGENTSCOPE_EXPERIENCE` | Enable/disable experience injection |
| `AUDIT_AGENTSCOPE_RULE_LOADING_MODE` | `all` or `preview_tool` |
| `AUDIT_PROCESS_JUDGE_ENABLED` | Enable external process judging |
| `AUDIT_PROCESS_JUDGE_BASE_URL` | OpenAI-compatible judge URL |
| `AUDIT_PROCESS_JUDGE_MODEL` | Explicit judge model name |
| `AUDIT_PROCESS_JUDGE_API_KEY` | Optional runtime secret |
| `AUDIT_PROCESS_JUDGE_PROMPT_TEMPLATE` | Explicit rubric template path |
| `AUDIT_PROCESS_JUDGE_THINKING` | Provider-specific thinking switch |
| `AUDIT_PROCESS_JUDGE_REASONING_EFFORT` | Optional provider-supported effort level |
| `AUDIT_PROCESS_JUDGE_MAX_TOKENS` | Judge generation budget |
| `AUDIT_PROCESS_JUDGE_TEMPERATURE` | Judge sampling temperature |
| `AUDIT_PROCESS_JUDGE_TIMEOUT` | Read timeout in seconds |
| `AUDIT_PROCESS_JUDGE_MAX_RETRIES` | Client retry setting |
| `AUDIT_PROCESS_JUDGE_LOG_PATH` | Private JSONL diagnostics output |

For a fixed-weight experiment, set both bounded-weight limits to the same
value and explicitly enable adaptive rewards. For example, `MIN=MAX=0.8`
gives outcome/process weights `0.8/0.2`, independent of GT confidence. Keep the
dataset, policy initialization, rollout parameters, filter settings, and
experience switch identical across ablations.

Missing judge scores are reported as unavailable. The arithmetic helper alone
does not automatically discard groups; select an appropriate availability
filter from `audit_agentic/rewards/dynamic_sampling_filters.py` when required.
Outcome-variance filtering and judge-availability filtering answer different
questions; record both settings in experiment metadata.

GRPO/GSPO and clipping options are implemented in Relax. See
`--advantage-estimator`, `--eps-clip`, `--eps-clip-high`, and adaptive clipping
arguments in `relax_agentic/relax/utils/arguments.py`. Do not assume GSPO is
equivalent to changing a single flag under every backend.

## Before a distributed run

1. Validate one synthetic/authorized case and a complete tool trajectory.
2. Check images resolve in both policy and judge workers, including tool images.
3. Verify judge JSON schema, score range, retries, timeout and missing-score rate.
4. Verify a group exports the intended number of trainable trajectories.
5. Check reward components, protocol penalties and filter counters in logs.
6. Run a short GPU smoke test and a save/resume checkpoint test.

CPU unit tests in this release do not establish multi-node GPU compatibility.
