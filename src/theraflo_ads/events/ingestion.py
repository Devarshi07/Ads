"""Client event ingestion (design doc: Event API, Idempotency, Security → Threat Model).

Every client event must present a valid impression token bound to the
authenticated user. Events with missing, forged, or expired tokens are kept
for fraud analysis but never become rewards or billable events.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from theraflo_ads.domain import AdCatalog
from theraflo_ads.events.schemas import ClientEvent, ClientEventType
from theraflo_ads.events.service import EventService
from theraflo_ads.security.identifiers import UserHasher
from theraflo_ads.security.tokens import ImpressionClaims, ImpressionTokenSigner, InvalidToken


class IngestStatus(StrEnum):
    ACCEPTED = "accepted"
    DUPLICATE = "duplicate"
    REJECTED = "rejected"


@dataclass(frozen=True)
class IngestResult:
    status: IngestStatus
    reason: str | None = None
    click_id: str | None = None
    landing_url: str | None = None


class ClientEventIngestor:
    def __init__(
        self,
        *,
        signer: ImpressionTokenSigner,
        hasher: UserHasher,
        events: EventService,
        catalog: AdCatalog,
        on_hidden: Callable[[str, str], None] | None = None,
        on_reported: Callable[[str, str], None] | None = None,
        late_event_grace: timedelta = timedelta(hours=24),
    ) -> None:
        self._signer = signer
        self._hasher = hasher
        self._events = events
        self._catalog = catalog
        self._on_hidden = on_hidden or (lambda user_id, ad_id: None)
        self._on_reported = on_reported or (lambda impression_id, ad_id: None)
        # Mobile clients buffer events offline, so events may arrive after the
        # token expires. Display itself must still happen before expiry.
        self._grace = late_event_grace

    def ingest(self, event: ClientEvent, user_id: str, now: datetime) -> IngestResult:
        claims, reason = self._verify(event.impression_token, user_id, now)
        if claims is None:
            return self._reject(event.event_id, reason, now)
        if claims.impression_id != event.impression_id or claims.ad_id != event.ad_id:
            return self._reject(event.event_id, "claims_mismatch", now)

        accepted = self._events.record_client_event(
            event_id=event.event_id,
            event_type=event.event_type,
            impression_id=claims.impression_id,
            ad_id=claims.ad_id,
            at=now,
            billable=event.client_timestamp <= claims.expires_at,
        )
        if not accepted:
            return IngestResult(IngestStatus.DUPLICATE)
        if event.event_type is ClientEventType.AD_HIDDEN:
            self._on_hidden(user_id, claims.ad_id)
        elif event.event_type is ClientEventType.AD_REPORTED:
            self._on_reported(claims.impression_id, claims.ad_id)
        return IngestResult(IngestStatus.ACCEPTED)

    def click(self, token: str, now: datetime) -> IngestResult:
        """Click redirect. The token alone authorizes the click: it is signed, bound to
        one impression and user, expires, and only the first click is billed."""
        try:
            claims = self._signer.verify(token, now, grace=self._grace)
        except InvalidToken as exc:
            return self._reject(None, exc.reason, now)
        ad_id = self._events.ad_for(claims.impression_id)
        if ad_id is None:
            return self._reject(None, "unknown_impression", now)
        click_id = self._events.click(claims.impression_id, now)
        return IngestResult(
            IngestStatus.ACCEPTED,
            click_id=click_id,
            landing_url=self._catalog.ads[ad_id].landing_url,
        )

    def _verify(
        self, token: str, user_id: str, now: datetime
    ) -> tuple[ImpressionClaims | None, str]:
        try:
            claims = self._signer.verify(token, now, grace=self._grace)
        except InvalidToken as exc:
            return None, exc.reason
        if not self._hasher.matches(user_id, claims.user_hash):
            return None, "user_mismatch"
        if self._events.ad_for(claims.impression_id) is None:
            return None, "unknown_impression"
        return claims, ""

    def _reject(self, event_id: str | None, reason: str, now: datetime) -> IngestResult:
        self._events.reject(event_id, reason, now)
        return IngestResult(IngestStatus.REJECTED, reason=reason)
