"""Right to deletion: pseudonymized snapshots, crypto-shredding, and operational erasure."""

from __future__ import annotations

import json
from datetime import timedelta

from tests.helpers import NOW, make_request, make_stack, make_user
from tests.test_api import ALICE, make_env, rank
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.infra.lake import LakeWriter, read_partitions
from theraflo_ads.privacy.shredding import (
    SHREDDED,
    DeletionService,
    InMemoryDeletionRequests,
    InMemoryUserKeyStore,
    Pseudonymizer,
    memory_erasers,
)


def decision_message(user_hash="k1:alice"):
    return Message(
        "m1",
        MessageType.DECISION,
        {"user_hash": user_hash, "session_id": "s1", "request_id": "r1"},
        NOW,
    )


def test_pseudonyms_are_stable_until_the_key_is_shredded():
    keys = InMemoryUserKeyStore()
    pseudo = Pseudonymizer(keys)
    first = pseudo.pseudonym("k1:alice")
    assert first == pseudo.pseudonym("k1:alice") != pseudo.pseudonym("k1:bob")
    assert first.startswith("p:") and "alice" not in first
    assert keys.shred("k1:alice") and not keys.shred("k1:alice")
    # A new key after deletion: old snapshot rows can never be linked to new ones.
    assert pseudo.pseudonym("k1:alice") != first


def test_snapshot_messages_carry_no_user_hash_or_session(tmp_path):
    keys = InMemoryUserKeyStore()
    writer = LakeWriter(tmp_path, Pseudonymizer(keys))
    writer(decision_message())
    [stored] = read_partitions(tmp_path, "decision")
    raw = "".join(p.read_text() for p in tmp_path.rglob("*.jsonl"))
    assert "k1:alice" not in raw and stored.payload["session_id"] is None
    assert stored.payload["user_hash"] == Pseudonymizer(keys).pseudonym("k1:alice")


def test_deletion_job_shreds_the_key_and_erases_operational_state():
    stack = make_stack()
    user = make_user("k1:alice")
    for i in range(3):
        stack.service.decide(
            make_request(user=user, request_id=f"r{i}", now=NOW + timedelta(minutes=11 * i))
        )
    stack.service.decide(make_request(user=make_user("k1:bob"), request_id="rb"))
    keys = InMemoryUserKeyStore()
    keys.key_for("k1:alice")
    events = []
    service = DeletionService(
        requests=InMemoryDeletionRequests(),
        keys=keys,
        erasers=memory_erasers(stack.platform),
        clock=lambda: NOW,
        on_event=lambda e, d: events.append((e, d)),
    )
    request = service.request("k1:alice")
    assert service.run_pending() == [request.request_id]
    assert "k1:alice" not in keys
    alice = [d for d in stack.decisions if d.request_id.startswith("r") and d.request_id != "rb"]
    assert all(d.user_hash == SHREDDED and d.session_id == SHREDDED for d in alice)
    assert next(d for d in stack.decisions if d.request_id == "rb").user_hash == "k1:bob"
    assert stack.log.for_user("k1:alice", NOW - timedelta(days=1)) == []
    assert stack.log.for_user("k1:bob", NOW - timedelta(days=1))
    [(event, details)] = events
    assert event == "user_deletion_completed" and details["decisions_anonymized"] == 3
    assert "k1:alice" not in json.dumps(details)
    assert service.run_pending() == []  # nothing left pending


def test_delete_me_erases_the_users_decisions_through_the_api():
    client, container, _ = make_env()
    rank(client)
    user_hash = container.hasher.hash("user-alice")
    assert any(d.user_hash == user_hash for d in container.platform.decision_log())
    assert client.delete("/v1/users/me", headers=ALICE).status_code == 202
    assert not any(d.user_hash == user_hash for d in container.platform.decision_log())
    assert user_hash not in str(container.audit)
