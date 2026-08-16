import unittest
import random

from mvp import ARMS, ToyProgram, UCBPolicy, run_trial

class MVPTests(unittest.TestCase):
    def test_ucb_visits_every_arm_before_exploitation(self):
        p = UCBPolicy()
        chosen = []
        for t in range(1, len(ARMS) + 1):
            a = p.choose(t)
            chosen.append(a)
            p.update(a, 0.0)
        self.assertEqual(set(chosen), set(ARMS))

    def test_toy_program_observation_schema(self):
        env = ToyProgram("users", random.Random(1))
        obs = env.execute("mutate_user_id")
        for key in ("forbidden_flow", "security_delta", "input_distance", "csa", "reward"):
            self.assertIn(key, obs)
        self.assertEqual(obs["input_distance"], 1.0)

    def test_program_state_is_local(self):
        a = run_trial("users", "ucb", 0, 30, 7)
        b = run_trial("reports", "ucb", 0, 30, 7)
        self.assertEqual(a.target, "users")
        self.assertEqual(b.target, "reports")

if __name__ == "__main__":
    unittest.main()
