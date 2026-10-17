"""Accounting worker (design doc: Message Processing, Delivery Semantics, Billing).

Consumes event messages: bills first, then updates the reward label, each
inside a unit of work. Both steps are idempotent (billing by idempotency key,
labels by their accounted flags), so at-least-once delivery is safe: a crash
between the two steps is fixed by redelivery.
"""

from __future__ import annotations

from datetime import datetime

from theraflo_ads.attribution.service import (
    AttributionPolicy,
    AttributionService,
    ImpressionLabel,
    UnknownImpression,
)
from theraflo_ads.counters.budget import BudgetStore
from theraflo_ads.domain import PricingModel
from theraflo_ads.events.bus import Bus, Message, MessageType
from theraflo_ads.storage.uow import UnitOfWorkFactory


class AccountingHandler:
    def __init__(
        self,
        *,
        uow_factory: UnitOfWorkFactory,
        budget: BudgetStore,
        policy: AttributionPolicy | None = None,
    ) -> None:
        self._uow = uow_factory
        self._budget = budget
        self._policy = policy or AttributionPolicy()

    def subscribe(self, bus: Bus) -> None:
        bus.subscribe(MessageType.AD_VIEWABLE, self.on_viewable)
        bus.subscribe(MessageType.AD_CLICKED, self.on_click)
        bus.subscribe(MessageType.CONVERSION, self.on_conversion)
        bus.subscribe(MessageType.CONVERSION_REVERSED, self.on_reversal)

    def on_viewable(self, message: Message) -> None:
        payload = message.payload
        at = datetime.fromisoformat(payload["at"])
        with self._uow() as uow:
            attribution = AttributionService(uow.attribution, self._policy)
            label = _require(attribution, payload["impression_id"])
            if label.viewable:
                return
            revenue = 0.0
            if payload.get("billable", True) and label.pricing_model is PricingModel.CPM:
                revenue = label.price / 1000.0
                self._budget.settle(
                    label.reservation_id,
                    label.budget_key,
                    revenue,
                    label.budget_limit,
                    at,
                    idempotency_key=f"view:{label.impression_id}",
                )
            attribution.account_viewable(label.impression_id, revenue)

    def on_click(self, message: Message) -> None:
        payload = message.payload
        at = datetime.fromisoformat(payload["at"])
        with self._uow() as uow:
            attribution = AttributionService(uow.attribution, self._policy)
            label = _require(attribution, payload["impression_id"])
            if label.click_accounted:
                return
            revenue = 0.0
            if label.pricing_model is PricingModel.CPC:
                revenue = label.price
                self._budget.settle(
                    label.reservation_id,
                    label.budget_key,
                    revenue,
                    label.budget_limit,
                    at,
                    idempotency_key=f"click:{label.impression_id}",
                )
            attribution.account_click(label.impression_id, revenue)

    def on_conversion(self, message: Message) -> None:
        payload = message.payload
        advertiser_id, order_id = payload["advertiser_id"], payload["order_id"]
        received_at = datetime.fromisoformat(payload["received_at"])
        with self._uow() as uow:
            attribution = AttributionService(uow.attribution, self._policy)
            record = uow.attribution.get_order(advertiser_id, order_id)
            if record is None or record.impression_id is None:
                raise UnknownImpression(f"order {advertiser_id}/{order_id}")
            if record.accounted or record.reversed:
                return
            label = _require(attribution, record.impression_id)
            if record.payout > 0:
                self._budget.charge(
                    label.budget_key,
                    record.payout,
                    label.budget_limit,
                    received_at,
                    idempotency_key=f"conv:{advertiser_id}:{order_id}",
                )
            attribution.account_conversion(advertiser_id, order_id)

    def on_reversal(self, message: Message) -> None:
        payload = message.payload
        advertiser_id, order_id = payload["advertiser_id"], payload["order_id"]
        at = datetime.fromisoformat(payload["at"])
        with self._uow() as uow:
            attribution = AttributionService(uow.attribution, self._policy)
            record = uow.attribution.get_order(advertiser_id, order_id)
            if record is None or record.impression_id is None:
                raise UnknownImpression(f"order {advertiser_id}/{order_id}")
            if record.reversed:
                return
            if record.accounted and record.payout > 0:
                label = _require(attribution, record.impression_id)
                self._budget.credit(
                    label.budget_key,
                    record.payout,
                    at,
                    idempotency_key=f"rev:{advertiser_id}:{order_id}",
                )
            attribution.account_reversal(advertiser_id, order_id)


def _require(attribution: AttributionService, impression_id: str) -> ImpressionLabel:
    label = attribution.label(impression_id)
    if label is None:
        raise UnknownImpression(impression_id)
    return label
