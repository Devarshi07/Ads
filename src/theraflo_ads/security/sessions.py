"""Session tokens (design doc: Security → Baseline Controls).

Theraflo's auth service issues short-lived HMAC session tokens for the ads API:
base64url("user_id|expires").base64url(signature). The ads service only
verifies them; it never sees passwords or the main app's credentials.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import hmac
from datetime import UTC, datetime, timedelta

from theraflo_ads.security.identifiers import MIN_KEY_BYTES


class SessionTokens:
    def __init__(self, key: bytes, *, ttl: timedelta = timedelta(hours=1)) -> None:
        if len(key) < MIN_KEY_BYTES:
            raise ValueError(f"session key must be at least {MIN_KEY_BYTES} bytes")
        self._key = key
        self._ttl = ttl

    def issue(self, user_id: str, now: datetime) -> str:
        if "|" in user_id or not user_id:
            raise ValueError("user_id must be non-empty and cannot contain '|'")
        expires = int((now + self._ttl).timestamp())
        payload = f"{user_id}|{expires}".encode()
        return f"{_b64(payload)}.{_b64(self._mac(payload))}"

    def verify(self, token: str, now: datetime) -> str | None:
        """The user ID, or None for a malformed, forged, or expired token."""
        try:
            payload_part, signature_part = token.split(".")
            payload = _unb64(payload_part)
            signature = _unb64(signature_part)
            if not hmac.compare_digest(signature, self._mac(payload)):
                return None
            user_id, expires = payload.decode().split("|")
            if now > datetime.fromtimestamp(int(expires), UTC):
                return None
            return user_id
        except (ValueError, binascii.Error, UnicodeDecodeError):
            return None

    def _mac(self, payload: bytes) -> bytes:
        return hmac.new(self._key, b"session\x00" + payload, hashlib.sha256).digest()


def _b64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text: str) -> bytes:
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))
