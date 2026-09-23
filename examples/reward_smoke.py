"""Exercise confidence and reward arithmetic without network access."""

import json

from audit_agentic.rewards.reward_adaptive import compute_final_reward


def main():
    rows = []
    for votes in (["A"] * 4, ["A", "A", "B", "B"], ["A", "B", "C", "D"]):
        reward = compute_final_reward(
            human_labels=votes,
            rule_verdict="supported",
            gt="A",
            prediction="A",
            process_reward_value=0.6,
            rho=0.1,
            gamma=1.0,
        )
        rows.append({"votes": votes, "reward": reward})
    assert rows[0]["reward"]["gt_confidence"] == 1.0
    assert rows[1]["reward"]["gt_confidence"] == 0.5
    assert rows[2]["reward"]["gt_confidence"] == 0.0
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
