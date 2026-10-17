"""Signed impression tokens (design doc: APIs → Rank Ads / Event API).

The rank response carries an HMAC token binding impression, ad, user, and
expiry. The event API only accepts events that present a valid token, so
clicks and impressions cannot be forged.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from theraflo_ads.security.identifiers import MIN_KEY_BYTES

_SEPARATOR = "|"


class InvalidToken(Exception):
    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class ImpressionClaims:
    impression_id: str
    ad_id: str
    user_hash: str
    expires_at: datetime


class ImpressionTokenSigner:
    def __init__(self, key: bytes) -> None:
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(f"token key must be at least {MIN_KEY_BYTES} bytes")
        self._key = key

    def sign(self, claims: ImpressionClaims) -> str:
        fields = (claims.impression_id, claims.ad_id, claims.user_hash)
        if any(_SEPARATOR in f or not f for f in fields):
            raise ValueError("token fields must be non-empty and cannot contain '|'")
        expires = str(int(claims.expires_at.timestamp()))
        payload = _SEPARATOR.join((*fields, expires)).encode()
        return f"{_b64encode(payload)}.{_b64encode(self._mac(payload))}"

    def verify(
        self, token: str, now: datetime, *, grace: timedelta = timedelta(0)
    ) -> ImpressionClaims:
        try:
            payload_part, signature_part = token.split(".")
            payload = _b64decode(payload_part)
            signature = _b64decode(signature_part)
        except (ValueError, binascii.Error) as exc:
            raise InvalidToken("malformed") from exc
        if not hmac.compare_digest(signature, self._mac(payload)):
            raise InvalidToken("bad_signature")
        try:
            impression_id, ad_id, user_hash, expires = payload.decode().split(_SEPARATOR)
            expires_at = datetime.fromtimestamp(int(expires), UTC)
        except (ValueError, UnicodeDecodeError) as exc:
            raise InvalidToken("malformed") from exc
        if now > expires_at + grace:
            raise InvalidToken("expired")
        return ImpressionClaims(impression_id, ad_id, user_hash, expires_at)

    def _mac(self, payload: bytes) -> bytes:
        return hmac.new(self._key, payload, hashlib.sha256).digest()


def _b64encode(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64decode(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
