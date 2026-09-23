# Third-Party Notices

This repository includes modified source snapshots used by the research
integration. They are not represented as unmodified upstream releases.

| Component | Public upstream | Included license |
|---|---|---|
| AgentScope | https://github.com/agentscope-ai/agentscope | `agentscope/LICENSE` (Apache-2.0) |
| Relax | https://github.com/redai-infra/Relax | `relax_agentic/LICENSE` (Apache-2.0) |

Existing file-level notices, attribution, and package metadata are retained.
Snapshot export modifications include replacing deployment-specific paths and
endpoints, removing embedded credentials, and making service access explicit.
Additional research changes include the AgentScope/Relax audit protocol and
multimodal history integration. The snapshots are not guaranteed to match the
latest public upstream APIs.

No new license is asserted here for AdaptEvo-specific code or third-party data.
External models, datasets and service APIs remain subject to their own terms.
