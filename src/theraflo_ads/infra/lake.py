"""Data-lake writer (design doc: Event Infrastructure → Analytics Sink, DVC Edge Cases).

Appends every message to immutable daily partitions:
    <lake>/<type>/date=YYYY-MM-DD/part-<host>-<pid>.jsonl
Delivery is at-least-once, so partitions can contain duplicates; dataset builds
deduplicate on message_id. Partitions are what DVC versions.

With a pseudonymizer, user hashes are replaced by per-user-key pseudonyms before
anything is written, so deleting a user's key (crypto-shredding) makes every
partition row of that user unlinkable.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

from theraflo_ads.events.bus import Message
from theraflo_ads.privacy.shredding import Pseudonymizer


class LakeWriter:
    def __init__(self, root: Path, pseudonymizer: Pseudonymizer | None = None) -> None:
        self._root = root
        self._part = f"part-{socket.gethostname()}-{os.getpid()}.jsonl"
        self._pseudonymizer = pseudonymizer
        self.written = 0

    def __call__(self, message: Message) -> None:
        if self._pseudonymizer is not None:
            message = self._pseudonymizer.message(message)
        partition = self._root / message.type.value / f"date={message.created_at.date()}"
        partition.mkdir(parents=True, exist_ok=True)
        with (partition / self._part).open("a", encoding="utf-8") as f:
            f.write(message.to_json() + "\n")
        self.written += 1


def read_partitions(root: Path, message_type: str) -> list[Message]:
    """All messages of one type, deduplicated on message_id, in created_at order."""
    seen: dict[str, Message] = {}
    for path in sorted((root / message_type).glob("date=*/*.jsonl")):
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    message = Message.from_json(line)
                    seen.setdefault(message.message_id, message)
    return sorted(seen.values(), key=lambda m: m.created_at)
