"""Keyed pseudonymous identifiers (design doc: Security & Threat Model → Identifiers).

A plain SHA-256 of a user ID can be reversed by hashing every known ID, so
user hashes are HMACs under a secret, purpose-specific key. Each hash carries
its key version so keys can rotate without breaking history.
"""

from __future__ import annotations

import hashlib
import hmac
from collections.abc import Mapping

MIN_KEY_BYTES = 32


class UserHasher:
    def __init__(self, purpose: str, keys: Mapping[str, bytes], current_version: str) -> None:
        if current_version not in keys:
            raise ValueError(f"unknown current key version {current_version!r}")
        for version, key in keys.items():
            if ":" in version:
                raise ValueError("key versions cannot contain ':'")
            if len(key) < MIN_KEY_BYTES:
                raise ValueError(f"key {version!r} must be at least {MIN_KEY_BYTES} bytes")
        self._purpose = purpose
        self._keys = dict(keys)
        self._current = current_version

    def hash(self, user_id: str) -> str:
        return self._hash_with(self._current, user_id)

    def matches(self, user_id: str, user_hash: str) -> bool:
        """True if user_hash belongs to user_id under any known key version."""
        version, sep, _ = user_hash.partition(":")
        if not sep or version not in self._keys:
            return False
        return hmac.compare_digest(self._hash_with(version, user_id), user_hash)

    def _hash_with(self, version: str, user_id: str) -> str:
        message = f"{self._purpose}\x00{user_id}".encode()
        digest = hmac.new(self._keys[version], message, hashlib.sha256).hexdigest()
        return f"{version}:{digest[:32]}"
