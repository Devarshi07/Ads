from datetime import timedelta

from tests.helpers import HASH_KEY, NOW, make_request, make_stack, make_user
from theraflo_ads.events import ClientEvent
from theraflo_ads.events.ingestion import ClientEventIngestor, IngestStatus
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.security.identifiers import UserHasher

HASHER = UserHasher("ads", {"k1": HASH_KEY}, "k1")


def setup(**ingestor_options):
    stack = make_stack(chain=lambda catalog: [PopularityPolicy(["ad_cpm"])])
    user = make_user(HASHER.hash("alice"))
    served = stack.service.decide(make_request(user=user)).served
    ingestor = ClientEventIngestor(
        signer=stack.signer,
        hasher=HASHER,
        events=stack.events,
        catalog=stack.catalog,
        **ingestor_options,
    )
    return stack, served, ingestor


def event(served, event_id="evt_00000001", event_type="AD_VIEWABLE", at=NOW, **overrides):
    payload = {
        "event_id": event_id,
        "event_type": event_type,
        "impression_id": served.impression_id,
        "ad_id": served.ad.ad_id,
        "impression_token": served.token,
        "client_timestamp": at.isoformat(),
    } | overrides
    return ClientEvent.model_validate(payload)


def rejected(stack):
    return [r.reason for r in stack.store.events.rejected]


def test_valid_event_is_accepted_once():
    stack, served, ingestor = setup()
    assert ingestor.ingest(event(served), "alice", NOW).status is IngestStatus.ACCEPTED
    assert ingestor.ingest(event(served), "alice", NOW).status is IngestStatus.DUPLICATE
    assert stack.labels.label(served.impression_id).viewable
    assert stack.budget.billed("c_cpm", NOW.date()) > 0


def test_someone_elses_token_is_rejected():
    stack, served, ingestor = setup()
    result = ingestor.ingest(event(served), "mallory", NOW)
    assert (result.status, result.reason) == (IngestStatus.REJECTED, "user_mismatch")
    assert rejected(stack) == ["user_mismatch"]


def test_forged_token_is_rejected():
    _, served, ingestor = setup()
    forged = served.token[:-4] + "AAAA"
    result = ingestor.ingest(event(served, impression_token=forged), "alice", NOW)
    assert result.reason == "bad_signature"


def test_claims_must_match_the_event():
    _, served, ingestor = setup()
    result = ingestor.ingest(event(served, ad_id="ad_other"), "alice", NOW)
    assert result.reason == "claims_mismatch"


def test_late_events_are_accepted_within_grace_but_not_after():
    _, served, ingestor = setup()
    late = NOW + timedelta(hours=5)
    assert ingestor.ingest(event(served), "alice", late).status is IngestStatus.ACCEPTED
    too_late = NOW + timedelta(days=2)
    result = ingestor.ingest(event(served, event_id="evt_00000002"), "alice", too_late)
    assert result.reason == "expired"


def test_display_after_token_expiry_is_recorded_but_not_billed():
    stack, served, ingestor = setup()
    shown_late = served.expires_at + timedelta(minutes=1)
    result = ingestor.ingest(event(served, at=shown_late), "alice", shown_late)
    assert result.status is IngestStatus.ACCEPTED
    assert stack.budget.billed(served.budget_key, NOW.date()) == 0


def test_hide_and_report_callbacks():
    hidden, reported = [], []
    _, served, ingestor = setup(
        on_hidden=lambda user_id, ad_id: hidden.append((user_id, ad_id)),
        on_reported=lambda impression_id, ad_id: reported.append(ad_id),
    )
    ingestor.ingest(event(served, event_type="AD_HIDDEN"), "alice", NOW)
    ingestor.ingest(event(served, "evt_00000002", event_type="AD_REPORTED"), "alice", NOW)
    assert hidden == [("alice", served.ad.ad_id)]
    assert reported == [served.ad.ad_id]


def test_click_returns_click_id_and_landing_url():
    _, served, ingestor = setup()
    result = ingestor.click(served.token, NOW)
    assert result.status is IngestStatus.ACCEPTED
    assert result.landing_url == served.ad.landing_url
    assert ingestor.click(served.token, NOW).click_id == result.click_id


def test_click_with_bad_token_is_rejected():
    stack, _, ingestor = setup()
    assert ingestor.click("not-a-token", NOW).status is IngestStatus.REJECTED
    assert stack.store.events.rejected[-1].event_id is None
