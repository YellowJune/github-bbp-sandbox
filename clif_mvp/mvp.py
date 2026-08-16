from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from dataclasses import dataclass, asdict
from pathlib import Path

ARMS = [
    "mutate_user_id", "mutate_role", "mutate_mode", "mutate_page",
    "mutate_limit", "mutate_sort", "mutate_token", "mutate_scope",
]

TARGETS = {
    "users": {"hot": {"mutate_user_id": 0.55, "mutate_scope": 0.20}, "base": 0.015},
    "reports": {"hot": {"mutate_mode": 0.50, "mutate_page": 0.22}, "base": 0.015},
    "tokens": {"hot": {"mutate_token": 0.52, "mutate_role": 0.18}, "base": 0.015},
}

@dataclass
class TrialResult:
    target: str
    policy: str
    trial: int
    budget: int
    discoveries: int
    first_discovery: int | None
    reward_sum: float

class ToyProgram:
    """Instrumented, deliberately synthetic security-flow environment."""

    def __init__(self, name: str, rng: random.Random):
        self.name = name
        self.rng = rng
        self.spec = TARGETS[name]
        self.seen_signatures: set[str] = set()

    def execute(self, action: str) -> dict:
        p = self.spec["hot"].get(action, self.spec["base"])
        hit = self.rng.random() < p
        trace_novelty = self.rng.random() * 0.05
        if hit:
            signature = f"{self.name}:{action}:{self.rng.randrange(5)}"
            new = signature not in self.seen_signatures
            self.seen_signatures.add(signature)
            security_delta = 1.0 if new else 0.25
            forbidden_flow = True
        else:
            signature = None
            new = False
            security_delta = 0.0
            forbidden_flow = False

        input_distance = 1.0
        csa = security_delta / input_distance
        reward = csa + 0.05 * trace_novelty
        return {
            "action": action,
            "forbidden_flow": forbidden_flow,
            "new_signature": new,
            "signature": signature,
            "security_delta": security_delta,
            "input_distance": input_distance,
            "csa": csa,
            "trace_novelty": trace_novelty,
            "reward": reward,
        }

class RandomPolicy:
    def __init__(self, rng: random.Random):
        self.rng = rng

    def choose(self, t: int) -> str:
        return self.rng.choice(ARMS)

    def update(self, action: str, reward: float) -> None:
        pass

class UCBPolicy:
    """Tiny tabula-rasa program-local learner."""

    def __init__(self):
        self.n = {a: 0 for a in ARMS}
        self.q = {a: 0.0 for a in ARMS}

    def choose(self, t: int) -> str:
        unseen = [a for a in ARMS if self.n[a] == 0]
        if unseen:
            return unseen[0]
        log_t = math.log(max(2, t))
        return max(ARMS, key=lambda a: self.q[a] + 1.4 * math.sqrt(log_t / self.n[a]))

    def update(self, action: str, reward: float) -> None:
        self.n[action] += 1
        n = self.n[action]
        self.q[action] += (reward - self.q[action]) / n

def run_trial(target: str, policy_name: str, trial: int, budget: int, seed: int) -> TrialResult:
    env_rng = random.Random(seed * 1009 + trial * 31 + hash(target) % 997)
    policy_rng = random.Random(seed * 2027 + trial * 43 + hash(policy_name) % 991)
    env = ToyProgram(target, env_rng)
    policy = UCBPolicy() if policy_name == "ucb" else RandomPolicy(policy_rng)

    discoveries = 0
    first = None
    reward_sum = 0.0
    for t in range(1, budget + 1):
        action = policy.choose(t)
        obs = env.execute(action)
        reward = float(obs["reward"])
        policy.update(action, reward)
        reward_sum += reward
        if obs["new_signature"]:
            discoveries += 1
            if first is None:
                first = t

    return TrialResult(target, policy_name, trial, budget, discoveries, first, reward_sum)

def summarize(rows: list[TrialResult]) -> dict:
    out = {"targets": {}, "aggregate": {}}
    for target in TARGETS:
        out["targets"][target] = {}
        for policy in ("random", "ucb"):
            rr = [r for r in rows if r.target == target and r.policy == policy]
            firsts = [r.first_discovery if r.first_discovery is not None else r.budget + 1 for r in rr]
            out["targets"][target][policy] = {
                "mean_discoveries": statistics.mean(r.discoveries for r in rr),
                "stdev_discoveries": statistics.pstdev(r.discoveries for r in rr),
                "mean_first_discovery": statistics.mean(firsts),
                "mean_reward": statistics.mean(r.reward_sum for r in rr),
            }

    for policy in ("random", "ucb"):
        rr = [r for r in rows if r.policy == policy]
        firsts = [r.first_discovery if r.first_discovery is not None else r.budget + 1 for r in rr]
        out["aggregate"][policy] = {
            "mean_discoveries": statistics.mean(r.discoveries for r in rr),
            "mean_first_discovery": statistics.mean(firsts),
            "mean_reward": statistics.mean(r.reward_sum for r in rr),
        }

    r = out["aggregate"]["random"]
    u = out["aggregate"]["ucb"]
    out["aggregate"]["relative"] = {
        "discovery_lift_pct": 100.0 * (u["mean_discoveries"] / r["mean_discoveries"] - 1.0),
        "first_discovery_reduction_pct": 100.0 * (1.0 - u["mean_first_discovery"] / r["mean_first_discovery"]),
        "reward_lift_pct": 100.0 * (u["mean_reward"] / r["mean_reward"] - 1.0),
    }
    return out

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--trials", type=int, default=80)
    ap.add_argument("--budget", type=int, default=120)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--out", type=Path, default=Path("artifacts/clif_mvp_results.json"))
    args = ap.parse_args()

    rows: list[TrialResult] = []
    for target in TARGETS:
        for policy in ("random", "ucb"):
            for trial in range(args.trials):
                rows.append(run_trial(target, policy, trial, args.budget, args.seed))

    payload = {
        "mvp": "CLIF program-local online learner",
        "pretraining": False,
        "cross_program_state_reuse": False,
        "learner": "UCB1-style online mutation-family learner",
        "trials_per_target_policy": args.trials,
        "execution_budget": args.budget,
        "seed": args.seed,
        "summary": summarize(rows),
        "rows": [asdict(r) for r in rows],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {args.out}")

if __name__ == "__main__":
    main()
