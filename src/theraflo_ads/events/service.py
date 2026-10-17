"""Synchronous event recording (design doc: Event API, Idempotency).

The API records facts and enqueues the accounting work in the same unit of
work: the event row (idempotent on event_id) and its outbox message commit
together. Billing never runs on the request path, so a billing outage queues
work instead of failing the client.
"""

from __future__ import annotations

from datetime import datetime

from theraflo_ads.attribution.service import (
    AttributionPolicy,
    AttributionService,
    ConversionOutcome,
)
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.events.schemas import ClientEventType
from theraflo_ads.storage.uow import RejectedEvent, UnitOfWorkFactory

_CLIENT_MESSAGE_TYPES = {
    ClientEventType.AD_VIEWABLE: MessageType.AD_VIEWABLE,
    ClientEventType.AD_HIDDEN: MessageType.AD_HIDDEN,
    ClientEventType.AD_REPORTED: MessageType.AD_REPORTED,
}


class EventService:
    def __init__(
        self, *, uow_factory: UnitOfWorkFactory, policy: AttributionPolicy | None = None
    ) -> None:
        self._uow = uow_factory
        self._policy = policy or AttributionPolicy()

    def ad_for(self, impression_id: str) -> str | None:
        """The served ad for an impression, or None if the impression is unknown."""
        with self._uow() as uow:
            label = uow.attribution.get_label(impression_id)
            return label.ad_id if label else None

    def record_client_event(
        self,
        *,
        event_id: str,
        event_type: ClientEventType,
        impression_id: str,
        ad_id: str,
        at: datetime,
        billable: bool = True,
    ) -> bool:
        """False if event_id was already accepted."""
        payload = {
            "event_id": event_id,
            "impression_id": impression_id,
            "ad_id": ad_id,
            "at": at.isoformat(),
            "billable": billable,
        }
        with self._uow() as uow:
            if not uow.events.accept(event_id, event_type.value, payload):
                return False
            uow.add_message(
                Message(f"evt:{event_id}", _CLIENT_MESSAGE_TYPES[event_type], payload, at)
            )
        return True

    def viewable(self, impression_id: str, ad_id: str, at: datetime, *, billable=True) -> bool:
        """Convenience for server-side callers (the simulator): one viewable per impression."""
        return self.record_client_event(
            event_id=f"viewable-{impression_id}",
            event_type=ClientEventType.AD_VIEWABLE,
            impression_id=impression_id,
            ad_id=ad_id,
            at=at,
            billable=billable,
        )

    def click(self, impression_id: str, at: datetime) -> str:
        with self._uow() as uow:
            attribution = AttributionService(uow.attribution, self._policy)
            click_id, first = attribution.assign_click(impression_id, at)
            if first:
                payload = {
                    "impression_id": impression_id,
                    "click_id": click_id,
                    "at": at.isoformat(),
                }
                uow.add_message(
                    Message(f"click:{impression_id}", MessageType.AD_CLICKED, payload, at)
                )
        return click_id

    def conversion(
        self,
        *,
        advertiser_id: str,
        click_id: str,
        order_id: str,
        value: float,
        occurred_at: datetime,
        received_at: datetime,
    ) -> ConversionOutcome:
        with self._uow() as uow:
            outcome = AttributionService(uow.attribution, self._policy).classify_conversion(
                advertiser_id=advertiser_id,
                click_id=click_id,
                order_id=order_id,
                value=value,
                occurred_at=occurred_at,
                received_at=received_at,
            )
            if outcome is ConversionOutcome.ATTRIBUTED:
                payload = {
                    "advertiser_id": advertiser_id,
                    "order_id": order_id,
                    "received_at": received_at.isoformat(),
                }
                uow.add_message(
                    Message(
                        f"conv:{advertiser_id}:{order_id}",
                        MessageType.CONVERSION,
                        payload,
                        received_at,
                    )
                )
        return outcome

    def reversal(self, *, advertiser_id: str, order_id: str, at: datetime) -> bool:
        with self._uow() as uow:
            record = AttributionService(uow.attribution, self._policy).request_reversal(
                advertiser_id, order_id
            )
            if record is None:
                return False
            payload = {"advertiser_id": advertiser_id, "order_id": order_id, "at": at.isoformat()}
            uow.add_message(
                Message(
                    f"rev:{advertiser_id}:{order_id}",
                    MessageType.CONVERSION_REVERSED,
                    payload,
                    at,
                )
            )
        return True

    def reject(self, event_id: str | None, reason: str, at: datetime) -> None:
        with self._uow() as uow:
            uow.events.reject(RejectedEvent(event_id, reason, at))
