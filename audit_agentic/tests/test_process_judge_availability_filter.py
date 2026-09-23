import unittest
from types import SimpleNamespace

from audit_agentic.rewards.dynamic_sampling_filters import (
    check_process_judge_available,
    check_reward_nonzero_std_and_process_judge,
)


def sample(reward=1.0, enabled=1, success=1, invalid=0):
    return SimpleNamespace(
        metadata={"process_judge/enabled": enabled, "process_judge/success": success,
                  "agentscope/invalid_final_output": invalid},
        get_reward_value=lambda args: reward,
    )


class AvailabilityFilterTests(unittest.TestCase):
    def test_equal_successful_scores_are_kept(self):
        for reward in [0.0, 1.0, -.1]:
            group = [sample(reward=reward), sample(reward=reward)]
            self.assertTrue(check_process_judge_available(None, group).keep)
            self.assertFalse(check_reward_nonzero_std_and_process_judge(None, group).keep)

    def test_judge_failure_drops_whole_group(self):
        group = [sample(), sample(success=0)]
        for fn in [check_process_judge_available, check_reward_nonzero_std_and_process_judge]:
            result = fn(None, group)
            self.assertFalse(result.keep)
            self.assertEqual(result.reason, "process_judge_failed")

    def test_invalid_agent_final_stays_trainable(self):
        self.assertTrue(check_process_judge_available(None, [sample(success=0, invalid=1)]).keep)

    def test_disabled_judge_stays_trainable(self):
        self.assertTrue(check_process_judge_available(None, [sample(enabled=0, success=0)]).keep)

    def test_old_filter_still_accepts_nonzero_spread(self):
        self.assertTrue(check_reward_nonzero_std_and_process_judge(None, [sample(.2), sample(.9)]).keep)


if __name__ == "__main__":
    unittest.main()
