# AdaptEvo

Research code for content-moderation agents with confidence-aware outcome
rewards, trajectory-level process judging, and experience/rubric evolution.
The agent runs on **AgentScope**; distributed RL integration uses **Relax**.

This is a sanitized source release, not a copy of the internal experiment
workspace. No real posts, human annotations, business rules, learned experience,
model checkpoints, evaluation outputs, credentials, or cluster configurations
are distributed. Examples are synthetic.

## Components

| Directory | Purpose |
|---|---|
| `audit_agentic/agents` | AgentScope reasoning/tool loop, structured decisions, multimodal preparation |
| `audit_agentic/environment` | Tool interfaces, rule retrieval, experience injection, cache adapters |
| `audit_agentic/rewards` | GT confidence, adaptive reward, process judge client and score parsing |
| `audit_agentic/relax_app` | Managed rollout protocol, reward export, multimodal wire-history handling |
| `audit_agentic/eval` | Dataset evaluation, metrics, provider adapters |
| `prompt_templates` | Process rubrics and candidate-dimension pool |
| `scripts` | Confidence-data preparation and judge diagnostics |
| `agentscope` | Vendored, modified AgentScope source snapshot |
| `relax_agentic` | Vendored, modified Relax source snapshot |
| `examples` | Synthetic inference and reward examples |

## Quick Start

Use Python 3.11 or 3.12 in an isolated environment. GPU training dependencies
are separate from inference dependencies.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ./agentscope
pip install -r requirements-eval.txt
export PYTHONPATH="$PWD:$PWD/agentscope/src:$PWD/relax_agentic:${PYTHONPATH:-}"

# No API, production data, or GPU required.
python -m examples.reward_smoke
python -m pytest audit_agentic/tests -q

# Requires a user-provided, tool-calling OpenAI-compatible model service.
export OPENAI_BASE_URL='http://localhost:8000/v1'
export OPENAI_MODEL='your-model-name'
export OPENAI_API_KEY='your-api-key'
python -m examples.synthetic_audit
```

The synthetic example uses a locally defined rule and a deterministic tool.
It does not query business services. Live model output is not deterministic.

## Reward Design

For four human-review rounds, let `p_k` be the fraction assigned to label set
`k`. Human agreement is `c_h = 1 - H(p) / log(4)`. Rule-consistency verdicts
map to `g = 1.0 / 0.6 / 0.2` for supported / ambiguous / unsupported, and
`c = c_h * g`.

In the original confidence-aware variant, the outcome weight is
`w_out = (1 - rho) * c**gamma`, and `w_proc = 1 - w_out`.
The code also supports bounded-weight and fixed-weight ablations. Format
penalties and missing-judge handling are explicit, not silently treated as a
successful process score. See [training integration](docs/TRAINING.md).

The original process rubric scores factual grounding, rule fidelity, evidence
coverage, tool use, and decision consistency. Their default weights are
`0.25, 0.25, 0.20, 0.15, 0.15`. Use a multimodal judge when evidence contains
images. A high process score is not evidence that the final label is correct.

Experience should be extracted and selected using **training data only**.
Keep held-out test data out of candidate generation and selection. Candidate
rubric dimensions should be validated in batches on held-out training traces;
high variance alone does not establish validity.

## Integration and Release Scope

- [Tool and evaluation integration](docs/INTEGRATION.md)
- [Training integration and prerequisites](docs/TRAINING.md)
- [Experience and rubric evolution workflow](docs/EVOLUTION.md)
- [Public-release boundaries and verification](docs/PUBLIC_RELEASE.md)
- [Third-party notices](THIRD_PARTY_NOTICES.md)

Business tool implementations are integration references. Their original
services and caches are not public dependencies; replace them with your own
adapters or provide offline fixtures. The distributed training stack requires
compatible CUDA, PyTorch, Ray, Megatron and SGLang installations. This release
does not provision a cluster or claim independently reproduced paper results.

## Licensing

Vendored components retain their existing licenses and copyright notices.
A project-wide license for the AdaptEvo-specific code has not yet been chosen;
public repository visibility alone does not grant a new license. See
[third-party notices](THIRD_PARTY_NOTICES.md) before redistributing.
