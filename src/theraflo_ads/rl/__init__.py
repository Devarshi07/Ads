"""Reinforcement learning (design doc: Reinforcement Learning, Offline RL Dataset, RL Safety).

- `dataset`: logged transitions per user from the decision log.
- `fqi`: offline fitted-Q iteration with a support constraint, evaluated with OPE.
- `ppo`: on-policy PPO for a linear-softmax policy, run only inside the simulator.
- `safety`: vulnerable-cohort evals that reject a policy regardless of revenue.
"""
