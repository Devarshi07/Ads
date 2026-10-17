"""Plain-text report for a simulation run."""

from __future__ import annotations

from theraflo_ads.ope.estimators import (
    OPEEstimate,
    doubly_robust,
    ips,
    samples_from_decisions,
    snips,
    user_clusters,
)
from theraflo_ads.ope.targets import category_reward_model, target_from_policy
from theraflo_ads.policy.policies import EpsilonGreedyPolicy, Policy, UniformPolicy
from theraflo_ads.policy.scoring import HeuristicScorer
from theraflo_ads.simulator.runner import SimResult


def ope_targets(result: SimResult) -> dict[str, Policy]:
    scorer = HeuristicScorer(result.world.catalog, result.world.priors)
    return {
        "greedy heuristic (epsilon=0)": EpsilonGreedyPolicy(scorer, epsilon=0.0),
        "uniform random": UniformPolicy(),
    }


def evaluate_targets(result: SimResult) -> dict[str, list[OPEEstimate]]:
    samples = samples_from_decisions(result.decisions, result.rewards)
    reward_model = category_reward_model(samples, result.world.catalog)
    estimates = {}
    users = user_clusters(samples)
    for name, policy in ope_targets(result).items():
        target = target_from_policy(policy, result.world.catalog)
        estimates[name] = [
            ips(samples, target, clusters=users),
            snips(samples, target, clusters=users),
            doubly_robust(samples, target, reward_model, clusters=users),
        ]
    return estimates


def format_report(result: SimResult) -> str:
    config = result.world.config
    decisions = result.decisions
    served = sum(1 for d in decisions if d.impression_id)
    clicks = sum(result.clicks.values())
    lines = [
        f"Theraflo Ads simulation: {config.n_users} users, {config.n_days} days, "
        f"seed {config.seed}, epsilon {config.epsilon}, stateless={config.stateless}",
        "",
        "Traffic",
        f"  ad requests         {len(decisions):>10,}",
        f"  ads served          {served:>10,}  ({served / max(len(decisions), 1):.1%})",
        f"  clicks              {clicks:>10,}  (CTR {clicks / max(served, 1):.2%})",
        "  served_by           " + _counts(result.served_by),
        "  blocked             " + _counts(result.blocked),
        "",
        "Revenue",
        f"  platform revenue    ${result.platform_revenue:>12,.2f}",
        f"  billed              ${result.billed:>12,.2f}",
        f"  overdelivery        ${result.overdelivery:>12,.2f}  (absorbed, never billed)",
        f"  value / decision    ${result.policy_value():>12.5f}",
        "  conversions         " + _counts(result.conversion_outcomes),
        "",
        "Safety (oracle checks, every count must be 0)",
        "  violations          " + (_counts(result.violations) or "none"),
        "",
        "Vulnerable-cohort parity (the policy never sees this flag)",
    ]
    for name, stats in result.cohorts.items():
        lines.append(
            f"  {name:<12} ad load {stats.ad_load:6.1%}   impulse share {stats.impulse_share:6.1%}"
            f"   ({stats.served:,} served / {stats.opportunities:,} opportunities)"
        )
    lines += ["", "Retention by holdout (long-term cost of ads)"]
    for holdout, (retained, total) in result.retention.items():
        rate = retained / total if total else 0.0
        lines.append(f"  {holdout:<20} {rate:6.1%}  ({retained:,}/{total:,} users)")
    lines += ["", "Off-policy estimates of alternative policies (from logged decisions)"]
    for name, estimates in evaluate_targets(result).items():
        lines.append(f"  {name}")
        for e in estimates:
            verdict = "conclusive" if e.conclusive else f"INCONCLUSIVE: {e.reason}"
            lines.append(
                f"    {e.estimator:<6} ${e.value:.5f}  95% CI [${e.ci_low:.5f}, ${e.ci_high:.5f}]"
                f"  ESS {e.ess:,.0f}/{e.n:,}  {verdict}"
            )
    return "\n".join(lines)


def _counts(counter) -> str:
    return ", ".join(f"{key}={value:,}" for key, value in counter.most_common())
