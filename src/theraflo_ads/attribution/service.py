"""Last-click attribution and reward labels (design doc: Attribution, Delayed Rewards).

Two kinds of operations:
- Facts, recorded synchronously by the API: an impression was served, a click
  got its click_id, a conversion postback was classified, a refund requested.
- Accounting, applied asynchronously by the worker: revenue and label state.
  Each accounting step is idempotent, so messages can be redelivered safely.

A label is only usable for training once it is mature: the click window plus
a grace period for late clicks has passed. Until then, an impression with no
conversion is not a negative.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from theraflo_ads.domain import PricingModel

if TYPE_CHECKING:
    from theraflo_ads.policy.decision import DecisionOutcome


class LabelState(StrEnum):
    PENDING = "pending"
    PARTIALLY_OBSERVED = "partially_observed"
    FINALIZED = "finalized"
    REVERSED = "reversed"


OPEN_STATES = frozenset({LabelState.PENDING, LabelState.PARTIALLY_OBSERVED})


class ConversionOutcome(StrEnum):
    ATTRIBUTED = "attributed"
    DUPLICATE = "duplicate"
    EXPIRED = "expired"
    INVALID = "invalid"
    UNKNOWN_CLICK = "unknown_click"
    FORBIDDEN = "forbidden"


@dataclass(frozen=True)
class AttributionPolicy:
    click_window: timedelta = timedelta(days=7)
    late_click_grace: timedelta = timedelta(days=1)
    version: str = "last_click_7d_v1"

    @property
    def maturity_delay(self) -> timedelta:
        return self.click_window + self.late_click_grace


@dataclass
class ImpressionLabel:
    impression_id: str
    request_id: str
    ad_id: str
    campaign_id: str
    advertiser_id: str
    pricing_model: PricingModel
    price: float
    budget_key: str
    budget_limit: float
    reservation_id: str | None
    served_at: datetime
    viewable: bool = False
    click_id: str | None = None
    clicked_at: datetime | None = None
    click_accounted: bool = False
    conversions: int = 0
    conversion_value: float = 0.0  # advertiser order value (ROAS guardrails only)
    platform_revenue: float = 0.0  # what Theraflo earns: the reward
    state: LabelState = LabelState.PENDING


@dataclass(frozen=True)
class ConversionRecord:
    advertiser_id: str
    order_id: str
    click_id: str
    impression_id: str | None
    value: float
    payout: float
    occurred_at: datetime
    received_at: datetime
    outcome: ConversionOutcome
    accounted: bool = False
    reversal_requested: bool = False
    reversed: bool = False


class AttributionRepository(Protocol):
    def get_label(self, impression_id: str) -> ImpressionLabel | None: ...

    def save_label(self, label: ImpressionLabel) -> None: ...

    def impression_for_click(self, click_id: str) -> str | None: ...

    def save_click(self, click_id: str, impression_id: str) -> None: ...

    def get_order(self, advertiser_id: str, order_id: str) -> ConversionRecord | None: ...

    def insert_order(self, record: ConversionRecord) -> bool:
        """False if (advertiser_id, order_id) already exists."""
        ...

    def update_order(self, record: ConversionRecord) -> None: ...

    def labels_served_before(
        self, cutoff: datetime, *, open_only: bool = False
    ) -> list[ImpressionLabel]: ...

    def orders(self) -> list[ConversionRecord]: ...


class AttributionService:
    def __init__(
        self, repository: AttributionRepository, policy: AttributionPolicy | None = None
    ) -> None:
        self._repo = repository
        self.policy = policy or AttributionPolicy()

    # ----- facts (API path) -----

    def register(self, outcome: DecisionOutcome) -> ImpressionLabel | None:
        served = outcome.served
        if served is None:
            return None
        existing = self._repo.get_label(served.impression_id)
        if existing is not None:
            return existing
        label = ImpressionLabel(
            impression_id=served.impression_id,
            request_id=outcome.decision.request_id,
            ad_id=served.ad.ad_id,
            campaign_id=served.campaign.campaign_id,
            advertiser_id=served.campaign.advertiser_id,
            pricing_model=served.campaign.pricing_model,
            price=served.campaign.price,
            budget_key=served.budget_key,
            budget_limit=served.budget_limit,
            reservation_id=served.reservation_id,
            served_at=outcome.decision.timestamp,
        )
        self._repo.save_label(label)
        return label

    def label(self, impression_id: str) -> ImpressionLabel | None:
        return self._repo.get_label(impression_id)

    def label_for_click(self, click_id: str) -> ImpressionLabel | None:
        impression_id = self._repo.impression_for_click(click_id)
        return self._repo.get_label(impression_id) if impression_id else None

    def assign_click(self, impression_id: str, at: datetime) -> tuple[str, bool]:
        """Returns (click_id, first_click). Only the first click per impression counts."""
        label = self._require(impression_id)
        if label.click_id is not None:
            return label.click_id, False
        label.click_id = uuid4().hex
        label.clicked_at = at
        self._repo.save_label(label)
        self._repo.save_click(label.click_id, impression_id)
        return label.click_id, True

    def classify_conversion(
        self,
        *,
        advertiser_id: str,
        click_id: str,
        order_id: str,
        value: float,
        occurred_at: datetime,
        received_at: datetime,
    ) -> ConversionOutcome:
        label = self.label_for_click(click_id)
        if label is None:
            return ConversionOutcome.UNKNOWN_CLICK
        if label.advertiser_id != advertiser_id:
            return ConversionOutcome.FORBIDDEN
        if self._repo.get_order(advertiser_id, order_id) is not None:
            return ConversionOutcome.DUPLICATE
        # Explicit, not an assert: asserts vanish under `python -O`.
        if label.clicked_at is None or occurred_at < label.clicked_at:
            outcome = ConversionOutcome.INVALID
        elif occurred_at - label.clicked_at > self.policy.click_window:
            outcome = ConversionOutcome.EXPIRED
        else:
            outcome = ConversionOutcome.ATTRIBUTED
        attributed = outcome is ConversionOutcome.ATTRIBUTED
        payout = label.price if attributed and label.pricing_model is PricingModel.CPA else 0.0
        record = ConversionRecord(
            advertiser_id,
            order_id,
            click_id,
            label.impression_id,
            value,
            payout,
            occurred_at,
            received_at,
            outcome,
        )
        if not self._repo.insert_order(record):
            return ConversionOutcome.DUPLICATE  # lost a race with a concurrent postback
        return outcome

    def request_reversal(self, advertiser_id: str, order_id: str) -> ConversionRecord | None:
        """Refund / chargeback. Returns the record to reverse, or None if nothing to reverse."""
        record = self._repo.get_order(advertiser_id, order_id)
        if (
            record is None
            or record.reversal_requested
            or record.outcome is not ConversionOutcome.ATTRIBUTED
        ):
            return None
        record = replace(record, reversal_requested=True)
        self._repo.update_order(record)
        return record

    # ----- accounting (worker path, idempotent) -----

    def account_viewable(self, impression_id: str, revenue: float) -> bool:
        label = self._require(impression_id)
        if label.viewable:
            return False
        label.viewable = True
        label.platform_revenue += revenue
        self._repo.save_label(label)
        return True

    def account_click(self, impression_id: str, revenue: float) -> bool:
        label = self._require(impression_id)
        if label.click_accounted:
            return False
        label.click_accounted = True
        label.platform_revenue += revenue
        self._repo.save_label(label)
        return True

    def account_conversion(self, advertiser_id: str, order_id: str) -> bool:
        record = self._repo.get_order(advertiser_id, order_id)
        # A refund processed before the conversion was accounted wins: never count it.
        if record is None or record.accounted or record.reversed or record.impression_id is None:
            return False
        label = self._require(record.impression_id)
        label.conversions += 1
        label.conversion_value += record.value
        label.platform_revenue += record.payout
        if label.state is LabelState.PENDING:
            label.state = LabelState.PARTIALLY_OBSERVED
        self._repo.save_label(label)
        self._repo.update_order(replace(record, accounted=True))
        return True

    def account_reversal(self, advertiser_id: str, order_id: str) -> bool:
        record = self._repo.get_order(advertiser_id, order_id)
        if record is None or record.reversed or record.impression_id is None:
            return False
        if record.accounted:
            label = self._require(record.impression_id)
            label.conversions -= 1
            label.conversion_value -= record.value
            label.platform_revenue -= record.payout
            label.state = LabelState.REVERSED
            self._repo.save_label(label)
        self._repo.update_order(replace(record, reversed=True))
        return True

    # ----- maturity -----

    def is_mature(self, label: ImpressionLabel, now: datetime) -> bool:
        return now >= label.served_at + self.policy.maturity_delay

    def finalize(self, now: datetime) -> int:
        """Close labels whose window has passed. Returns how many were finalized."""
        finalized = 0
        cutoff = now - self.policy.maturity_delay
        for label in self._repo.labels_served_before(cutoff, open_only=True):
            label.state = LabelState.FINALIZED
            self._repo.save_label(label)
            finalized += 1
        return finalized

    def mature_labels(self, now: datetime) -> list[ImpressionLabel]:
        return self._repo.labels_served_before(now - self.policy.maturity_delay)

    def rewards_by_request(self, now: datetime) -> dict[str, float]:
        """Platform revenue per request, mature labels only."""
        return {label.request_id: label.platform_revenue for label in self.mature_labels(now)}

    def conversion_outcomes(self) -> list[ConversionRecord]:
        return self._repo.orders()

    def _require(self, impression_id: str) -> ImpressionLabel:
        label = self._repo.get_label(impression_id)
        if label is None:
            raise UnknownImpression(impression_id)
        return label


class UnknownImpression(KeyError):
    """No label for this impression. Permanent: retrying cannot help."""
