# Experience and Rubric Evolution

This document describes the experimental workflow, not an automated end-to-end
evolution service. The private traces and selected experience files are omitted.

## Experience evolution

1. Split training cases by source/queue. Deduplicate by item identity before
   creating discovery and validation partitions; freeze held-out tests.
2. Generate multiple discovery trajectories per training case (for example,
   four independent rollouts) and retain tool calls, results, images, predictions
   and parsing/transport status separately.
3. Analyze recurring errors within one queue. Human-review reasons can suggest
   evidence or rule checks, but are not automatically reliable rules. Do not
   convert inaccessible evidence into a factual conclusion.
4. Propose a concise queue-level experience candidate: describe the trigger,
   necessary checks, boundary conditions and counterexamples. Avoid memorizing
   case identifiers, quotations, or a fixed label answer.
5. Evaluate candidates on the frozen training-validation partition under the
   same model and inference settings. Compare outcome metrics and successful
   coverage; report infrastructure failures separately.
6. Limit the search budget (for example, at most five candidate versions per
   queue), stop early for a predefined meaningful improvement, and retain the
   best validated candidate or the no-experience baseline.
7. Freeze the experience artifact before held-out evaluation. Test results may
   be reported, but must not select, rewrite, or reject individual experiences.

`audit_agentic/environment/audit_experience.py` provides the runtime loader.
Label-level guidance can also be attached to matching rule previews. Formal
rules and observed tool evidence must take priority over experience hints.

## Rubric evolution

Use `prompt_templates/rubric_pool` as candidate dimensions rather than asking
a judge to emit every possible score at once. Keep an unchanged anchor set of
training trajectories, including correct, incorrect, uncertain and failed-tool
cases. Verify evidence completeness before interpreting score differences.

Validate small batches of candidate dimensions. For each dimension measure
score distribution, between-trajectory variance, repeated-judgment consistency,
redundancy with existing dimensions, and alignment with manually checked process
defects. A noisy dimension can have high variance without useful discrimination.

Freeze the selected dimensions, score grid and deterministic aggregation weights
as a versioned rubric. Re-score the same anchor set before changing training.
Do not silently compare aggregate rewards from different versions as a common
calibrated quantity. Preserve per-dimension scores, judge model/configuration,
media counts, failure status and the rubric version for subsequent analysis.

Outcome performance remains an independent check: a rising process reward with
flat or falling outcome reward can indicate an easy/saturated judge or a policy
exploiting the rubric. It is a diagnostic signal, not proof of reward hacking.
