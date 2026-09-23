#!/usr/bin/env bash
# Entry point for one audit_agentic managed session under relax_agentic.
#
# Relax passes RELAX_INPUT_JSON, RELAX_OUTPUT_JSON, RELAX_BASE_URL, and
# RELAX_SESSION_ID. We map them to OpenAI-compatible env vars so the agent's
# AsyncOpenAI client works, then delegate to audit_agentic.relax_app.agent.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"

# Map relax env → OpenAI-compatible env (same as dapo_ma/run_agent_app.sh)
export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"

# Make sure audit_agentic is importable.
export PYTHONPATH="${SCRIPT_DIR}/../..:${PYTHONPATH:-}"

python3 -m audit_agentic.relax_app.agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
