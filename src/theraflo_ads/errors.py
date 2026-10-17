"""Error taxonomy shared by every layer (design doc: Message Processing).

Transient errors are retried (a message stays on the queue); anything else is
permanent and goes to the dead-letter queue.
"""

from __future__ import annotations


class TransientError(RuntimeError):
    """A dependency is temporarily unavailable. Safe to retry."""


class StorageUnavailable(TransientError):
    """The system of record (Postgres in production) cannot be reached."""


class PermanentError(RuntimeError):
    """Retrying cannot help (malformed message, unknown entity)."""


TRANSIENT_ERRORS: tuple[type[BaseException], ...] = (TransientError, ConnectionError, TimeoutError)
