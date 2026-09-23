#!/usr/bin/env bash
# Managed Relax entry point for the single-agent AgentScope audit runtime.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
PROJECT_ROOT="$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)"
AGENTSCOPE_ROOT="${AGENTSCOPE_ROOT:-${PROJECT_ROOT}/agentscope}"
AGENTSCOPE_PYTHON="${AGENTSCOPE_PYTHON:-$(command -v python3)}"

if [[ ! -x "${AGENTSCOPE_PYTHON}" ]]; then
    echo "AgentScope Python is not executable: ${AGENTSCOPE_PYTHON}" >&2
    exit 1
fi

export OPENAI_BASE_URL="${RELAX_BASE_URL}"
export OPENAI_API_KEY="${RELAX_SESSION_ID}"
export PYTHONPATH="${PROJECT_ROOT}:${AGENTSCOPE_ROOT}/src:${PYTHONPATH:-}"

"${AGENTSCOPE_PYTHON}" -m audit_agentic.relax_app.agentscope_agent \
    --input-json "${RELAX_INPUT_JSON}" \
    --output-json "${RELAX_OUTPUT_JSON}"
