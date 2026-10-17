"""Creative review, advertiser reports, and "why this ad" through the HTTP API."""

from datetime import timedelta

from fastapi.testclient import TestClient

from tests.helpers import NOW
from tests.test_api import ADMIN, ALICE, BOB, FakeClock, rank
from tests.test_review import StubModel, payload
from theraflo_ads.api import build_dev_container, create_app
from theraflo_ads.experiments import HoldoutConfig
from theraflo_ads.policy import PopularityPolicy
from theraflo_ads.review.creative import ReviewerUnavailable
from theraflo_ads.review.simulated import SimulatedReviewModel

BOOKLY = {"X-Advertiser-Key": "dev-adv-bookly"}
TRAILCO = {"X-Advertiser-Key": "dev-adv-trailco"}


def make_env(reviewer=None):
    clock = FakeClock()
    container = build_dev_container(
        clock=clock,
        user_hash_key=b"u" * 32,
        token_key=b"t" * 32,
        holdouts=HoldoutConfig(uniform_exploration_share=0.0, no_ads_share=0.0),
        chain=lambda catalog: [PopularityPolicy(["ad_bookly", "ad_sleepwell"])],
        reviewer=reviewer or SimulatedReviewModel(),
    )
    return TestClient(create_app(container)), container


def creative_body(ad_id="ad_bookly_fall", **overrides):
    return {
        "ad_id": ad_id,
        "campaign_id": "cmp_bookly",
        "declared_category": "books",
        "headline": "Your next favorite novel",
        "body": "Cozy mysteries for rainy evenings, delivered to your door.",
        "landing_url": "https://bookly.example/fall",
    } | overrides


def test_clean_creative_is_reviewed_after_submission_and_goes_live():
    client, container = make_env()
    response = client.post("/v1/advertiser/creatives", json=creative_body(), headers=BOOKLY)
    assert response.status_code == 202 and response.json()["review_status"] == "pending"
    # The background review ran after the response was sent.
    status = client.get("/v1/advertiser/creatives/ad_bookly_fall", headers=BOOKLY).json()
    assert status == {
        "ad_id": "ad_bookly_fall",
        "review_status": "approved",
        "violations": [],
        "category": "books",
    }
    assert container.catalog.ads["ad_bookly_fall"].review_status.value == "approved"
    assert any(e["event"] == "creative_reviewed" for e in container.audit)


def test_violating_creative_is_rejected_with_policy_labels_only():
    client, _ = make_env()
    body = creative_body("ad_bookly_bad", body="Feeling low tonight? Treat yourself to a book.")
    client.post("/v1/advertiser/creatives", json=body, headers=BOOKLY)
    status = client.get("/v1/advertiser/creatives/ad_bookly_bad", headers=BOOKLY).json()
    assert status["review_status"] == "rejected"
    assert status["violations"] == ["vulnerability_targeting"]
    assert set(status) == {"ad_id", "review_status", "violations", "category"}  # no rationale


def test_submissions_are_scoped_to_the_advertiser():
    client, _ = make_env()
    assert client.post("/v1/advertiser/creatives", json=creative_body()).status_code == 401
    # Another advertiser's campaign looks exactly like an unknown one.
    response = client.post("/v1/advertiser/creatives", json=creative_body(), headers=TRAILCO)
    assert response.status_code == 404
    client.post("/v1/advertiser/creatives", json=creative_body(), headers=BOOKLY)
    assert client.get("/v1/advertiser/creatives/ad_bookly_fall", headers=TRAILCO).status_code == 404
    bad = creative_body(landing_url="http://bookly.example/")
    assert client.post("/v1/advertiser/creatives", json=bad, headers=BOOKLY).status_code == 422


def test_reviewer_outage_queues_for_humans_who_decide_through_the_admin_api():
    client, container = make_env(StubModel(error=ReviewerUnavailable("down")))
    client.post("/v1/advertiser/creatives", json=creative_body(), headers=BOOKLY)
    status = client.get("/v1/advertiser/creatives/ad_bookly_fall", headers=BOOKLY).json()
    assert status["review_status"] == "pending"
    assert client.get("/v1/admin/review-queue", headers=ALICE).status_code == 401
    items = client.get("/v1/admin/review-queue", headers=ADMIN).json()["items"]
    assert [i["ad_id"] for i in items] == ["ad_bookly_fall"]
    detail = client.get("/v1/admin/review-queue/ad_bookly_fall", headers=ADMIN).json()
    assert detail["review"]["error"].startswith("reviewer_unavailable")
    restricted = {"verdict": "approve", "category": "gambling"}
    response = client.post("/v1/admin/review-queue/ad_bookly_fall", json=restricted, headers=ADMIN)
    assert response.status_code == 422
    decided = client.post(
        "/v1/admin/review-queue/ad_bookly_fall", json={"verdict": "approve"}, headers=ADMIN
    )
    assert decided.status_code == 200 and decided.json()["reviewer"] == "dev-admin"
    assert container.catalog.ads["ad_bookly_fall"].review_status.value == "approved"
    again = client.post(
        "/v1/admin/review-queue/ad_bookly_fall", json={"verdict": "reject"}, headers=ADMIN
    )
    assert again.status_code == 404


def test_policy_owners_register_substantiated_claims():
    client, _ = make_env(StubModel(payload(category="books")))
    claim = "Rated 4.9 stars by 80,000 readers."
    # Not a health claim, so unmatched it goes to a human rather than being rejected.
    unsupported = creative_body("ad_b1", body=f"Great reads. {claim}")
    client.post("/v1/advertiser/creatives", json=unsupported, headers=BOOKLY)
    assert (
        client.get("/v1/advertiser/creatives/ad_b1", headers=BOOKLY).json()["review_status"]
        == "pending"
    )
    entry = {
        "claim_id": "clm_b2",
        "advertiser_id": "adv_bookly",
        "text": claim,
        "evidence": "store-ratings-2026.pdf",
    }
    assert client.post("/v1/admin/claims", json=entry, headers=BOOKLY).status_code == 401
    assert client.post("/v1/admin/claims", json=entry, headers=ADMIN).status_code == 201
    supported = creative_body("ad_b2", body=f"Great reads. {claim}")
    client.post("/v1/advertiser/creatives", json=supported, headers=BOOKLY)
    assert (
        client.get("/v1/advertiser/creatives/ad_b2", headers=BOOKLY).json()["review_status"]
        == "approved"
    )


def test_why_this_ad_is_only_for_the_user_who_saw_it():
    client, _ = make_env()
    ad = rank(client)["ads"][0]
    why = client.get(f"/v1/ads/{ad['impression_id']}/why", headers=ALICE)
    assert why.status_code == 200
    body = why.json()
    assert body["factors"][0].startswith("This ad was shown in your home feed")
    assert "Hide this ad" in body["controls"] and body["template_version"] == "why_v1"
    assert client.get(f"/v1/ads/{ad['impression_id']}/why", headers=BOB).status_code == 404
    assert client.get("/v1/ads/imp_unknown/why", headers=ALICE).status_code == 404
    assert client.get(f"/v1/ads/{ad['impression_id']}/why").status_code == 401


def test_advertiser_report_is_aggregated_and_thresholded():
    client, _ = make_env()
    for _ in range(3):
        rank(client)
    params = {
        "start": (NOW - timedelta(days=1)).isoformat(),
        "end": (NOW + timedelta(days=1)).isoformat(),
    }
    response = client.get("/v1/advertiser/report", params=params, headers=BOOKLY)
    assert response.status_code == 200, response.text
    report = response.json()
    assert report["totals"]["impressions"] >= 1
    assert report["segments"] == []  # one user: every segment is below the thresholds
    assert report["suppressed_segments"]
    assert report["narrative_source"] == "template"
    assert "user" not in report["narrative"].lower()
    reversed_period = {"start": params["end"], "end": params["start"]}
    assert (
        client.get("/v1/advertiser/report", params=reversed_period, headers=BOOKLY).status_code
        == 422
    )
    assert client.get("/v1/advertiser/report", params=params).status_code == 401


def test_audit_summary_endpoint():
    client, _ = make_env()
    summary = client.get("/v1/admin/review-audit", headers=ADMIN).json()
    assert summary["audited"] == 0 and summary["false_approval_rate"] is None
