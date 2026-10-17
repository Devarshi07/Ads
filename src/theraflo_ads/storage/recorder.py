"""Decision recording (design doc: Counterfactual Logging, Failure Hierarchy).

The decision row, the impression's reward label, and a decision message for
downstream consumers (analytics sink, drift monitor) commit in one unit of
work. If it fails, the DecisionService serves NO_AD.
"""

from __future__ import annotations

from theraflo_ads.attribution.service import AttributionPolicy, AttributionService
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.policy.decision import DecisionOutcome
from theraflo_ads.storage.uow import UnitOfWorkFactory


class DecisionRecorder:
    def __init__(
        self, uow_factory: UnitOfWorkFactory, policy: AttributionPolicy | None = None
    ) -> None:
        self._uow = uow_factory
        self._policy = policy or AttributionPolicy()

    def __call__(self, outcome: DecisionOutcome) -> None:
        decision = outcome.decision
        with self._uow() as uow:
            uow.decisions.append(decision)
            AttributionService(uow.attribution, self._policy).register(outcome)
            uow.add_message(
                Message(
                    f"decision:{decision.request_id}",
                    MessageType.DECISION,
                    decision.to_log(),
                    decision.timestamp,
                )
            )
