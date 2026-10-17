"""HTTP API (design doc: APIs, Event API, Security & Threat Model, User Controls).

Run locally:   theraflo-ads serve          (or: uvicorn --factory theraflo_ads.api.app:create_app)
Backend:       THERAFLO_ADS_BACKEND=memory (default) | infra
Dev console:   http://localhost:8000/ui    (dev credentials only; see theraflo_ads.api.dev)

Endpoints are sync and run in the threadpool. The container's lock serializes
access to in-memory stores; with the infra backend it is a no-op.

No `from __future__ import annotations` here: FastAPI resolves the dependency
aliases defined inside create_app() at definition time.
"""

import hmac
import logging
import mimetypes
import os
import re
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit
from uuid import uuid4

from fastapi import BackgroundTasks, Depends, FastAPI, Header, HTTPException, Query, Request, status
from fastapi.responses import JSONResponse, RedirectResponse, Response
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from fastapi.staticfiles import StaticFiles
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

from theraflo_ads import __version__
from theraflo_ads.api.container import Container, build_container_from_env
from theraflo_ads.attribution.service import ConversionOutcome
from theraflo_ads.domain import AdRequest, ConsentScope
from theraflo_ads.errors import TransientError
from theraflo_ads.events.ingestion import IngestStatus
from theraflo_ads.events.schemas import ClientEvent, ConversionPostback, ReversalPostback
from theraflo_ads.observability.metrics import served_ad_violations
from theraflo_ads.observability.ratelimit import retry_after_header
from theraflo_ads.review.creative import Creative, SubstantiatedClaim, Verdict, Violation
from theraflo_ads.review.explanations import explain
from theraflo_ads.review.narrator import narrate
from theraflo_ads.review.reports import build_advertiser_report
from theraflo_ads.review.store import QueueKind, SubmissionError

_ID = r"^[A-Za-z0-9_-]+$"
_REQUEST_ID = re.compile(r"^[A-Za-z0-9_-]{8,64}$")
CONSOLE_DIR = Path(__file__).parent / "static" / "console"
# The Windows registry can map .js to text/plain, and browsers refuse such module scripts.
mimetypes.add_type("text/javascript", ".js")
mimetypes.add_type("text/css", ".css")
# The console loads nothing from anywhere else, runs no inline script, and cannot be framed.
CONSOLE_CSP = (
    "default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data:; "
    "connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'"
)
# Some proxies consume Authorization instead of forwarding it: the Kubernetes API server's
# service proxy (LocalStack's Container Apps ingress) authenticates with it, then deletes it.
# With THERAFLO_ADS_AUTH_FALLBACK_HEADER on, the same "Bearer <token>" is also read from this
# header when Authorization is absent. Off by default, and refused in production.
FALLBACK_AUTH_HEADER = "X-Theraflo-Authorization"
access_log = logging.getLogger("theraflo_ads.access")
log = logging.getLogger("theraflo_ads.api")


class RankContext(BaseModel):
    model_config = ConfigDict(extra="ignore")

    device: str | None = Field(default=None, max_length=32)
    # Informational only: the server clock is authoritative.
    timestamp: AwareDatetime | None = None


class RankRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    session_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    placement: str = Field(min_length=1, max_length=64, pattern=_ID)
    context: RankContext = Field(default_factory=RankContext)


class RankedAd(BaseModel):
    impression_id: str
    ad_id: str
    position: int
    impression_token: str
    expires_at: datetime
    click_url: str


class RankResponse(BaseModel):
    request_id: str
    ads: list[RankedAd]
    served_by: str
    policy_version: str | None


class KillSwitchRequest(BaseModel):
    enabled: bool
    reason: str = Field(min_length=3, max_length=200)


class ConsentRequest(BaseModel):
    consent: Literal["personalized", "contextual_only"]


class SuppressionRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=128)
    hours: float = Field(default=24.0, gt=0, le=24 * 30)


class CreativeSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    ad_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    campaign_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    declared_category: str = Field(min_length=1, max_length=64, pattern=_ID)
    headline: str = Field(min_length=1, max_length=200)
    body: str = Field(default="", max_length=2000)
    landing_url: str = Field(min_length=8, max_length=2048, pattern=r"^https://")
    landing_text: str = Field(default="", max_length=5000)


class HumanReviewDecision(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["review", "audit"] = "review"
    verdict: Literal["approve", "reject"]
    category: str | None = Field(default=None, max_length=64)
    violations: list[Violation] = Field(default_factory=list, max_length=10)
    notes: str = Field(default="", max_length=1000)


class ClaimRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    claim_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    advertiser_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    text: str = Field(min_length=3, max_length=1000)
    evidence: str = Field(min_length=3, max_length=500)


def create_app(container: Container | None = None) -> FastAPI:
    c = container or build_container_from_env()
    app = FastAPI(title="Theraflo Ads", version=__version__)
    app.state.container = c
    bearer = HTTPBearer(auto_error=False)
    fallback_header = _auth_fallback_enabled()

    def bearer_token(
        request: Request,
        authorization: Annotated[HTTPAuthorizationCredentials | None, Depends(bearer)],
    ) -> str | None:
        if authorization is not None:
            return authorization.credentials
        if fallback_header:
            scheme, _, token = request.headers.get(FALLBACK_AUTH_HEADER, "").partition(" ")
            if scheme.lower() == "bearer" and token:
                return token
        return None

    Token = Annotated[str | None, Depends(bearer_token)]

    def _bearer(authenticate, token: str | None) -> str:
        principal = authenticate(token)
        if principal is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
        return principal

    def current_user(token: Token) -> str:
        return _bearer(c.authenticate_user, token)

    def admin(token: Token) -> str:
        return _bearer(c.authenticate_admin, token)

    def service(token: Token) -> str:
        return _bearer(c.authenticate_service, token)

    def advertiser(x_advertiser_key: Annotated[str | None, Header()] = None) -> str:
        advertiser_id = c.authenticate_advertiser(x_advertiser_key)
        if advertiser_id is None:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid advertiser key")
        return advertiser_id

    User = Annotated[str, Depends(current_user)]
    Advertiser = Annotated[str, Depends(advertiser)]

    def limit(scope: str, principal: str) -> None:
        decision = c.limiter.check(scope, principal)
        if not decision.allowed:
            c.metrics.rate_limited.labels(scope).inc()
            raise HTTPException(
                status.HTTP_429_TOO_MANY_REQUESTS,
                "rate limited",
                headers={"Retry-After": retry_after_header(decision)},
            )

    @app.middleware("http")
    async def observe(request: Request, call_next):
        started = time.perf_counter()
        incoming = request.headers.get("x-request-id", "")
        request_id = incoming if _REQUEST_ID.match(incoming) else uuid4().hex
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            response.headers["X-Request-ID"] = request_id
            return response
        finally:
            elapsed = time.perf_counter() - started
            # The route template, never the raw path: click URLs carry signed tokens.
            route = getattr(request.scope.get("route"), "path", "unmatched")
            c.metrics.http_requests.labels(route, request.method, str(status_code)).inc()
            c.metrics.http_latency.labels(route).observe(elapsed)
            access_log.info(
                "request",
                extra={
                    "route": route,
                    "method": request.method,
                    "status": status_code,
                    "duration_ms": round(elapsed * 1000, 2),
                    "request_id": request_id,
                },
            )

    def _scrape() -> None:
        c.metrics.kill_switch.set(1 if c.platform.safety.current().kill_switch else 0)
        for name, ok in c.platform.health().items():
            c.metrics.dependency_up.labels(name).set(1 if ok else 0)
        if c.review is not None:
            for kind in QueueKind:
                c.metrics.review_queue.labels(kind.value).set(len(c.review.queue(kind)))

    c.metrics.on_scrape(_scrape)

    @app.get("/metrics", include_in_schema=False)
    def metrics(presented: Token) -> Response:
        """Prometheus scrape. With THERAFLO_ADS_METRICS_TOKEN set, a matching bearer token
        is required; without it, expose this port only on the internal network."""
        token = os.environ.get("THERAFLO_ADS_METRICS_TOKEN")
        if token and not hmac.compare_digest((presented or "").encode(), token.encode()):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, "invalid credentials")
        return Response(c.metrics.render(), media_type="text/plain; version=0.0.4; charset=utf-8")

    @app.exception_handler(TransientError)
    def transient(request: Request, exc: TransientError) -> JSONResponse:
        # A dependency is down: the client keeps the event and retries (idempotent).
        return _retry_later()

    try:  # the infra backend's database errors; absent when SQLAlchemy is not installed
        from sqlalchemy.exc import InterfaceError, OperationalError
    except ImportError:  # pragma: no cover
        pass
    else:

        @app.exception_handler(OperationalError)
        @app.exception_handler(InterfaceError)
        def database_unreachable(request: Request, exc: Exception) -> JSONResponse:
            # Connection-level failures only (found by scripts/chaos.py): retryable 503,
            # never a 500. Integrity errors are bugs and still surface as 500s.
            log.error("database unreachable: %s", type(exc).__name__)
            return _retry_later()

    @app.get("/healthz")
    def healthz() -> dict:
        return {"status": "ok"}

    @app.get("/readyz")
    def readyz() -> JSONResponse:
        checks = dict(c.platform.health())
        ready = all(checks.values())
        code = status.HTTP_200_OK if ready else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse({"ready": ready, "checks": checks}, status_code=code)

    @app.post("/v1/ads/rank", response_model=RankResponse)
    def rank(body: RankRequest, user_id: User) -> RankResponse:
        limit("rank", user_id)
        with c.lock:
            request = AdRequest(
                request_id=uuid4().hex,
                session_id=body.session_id,
                placement_id=body.placement,
                user=c.profiles.context(user_id, device=body.context.device),
                now=c.clock(),
            )
            outcome = c.platform.decisions.decide(request)
        decision, served = outcome.decision, outcome.served
        c.metrics.observe_decision(decision)
        violations = served_ad_violations(
            decision, c.catalog, c.platform.safety.current(), request.user
        )
        if violations:
            # Last gate: an ad that breaks a hard constraint is never sent to the client.
            for kind in violations:
                c.metrics.safety_violations.labels(kind).inc()
            log.error(
                "served ad withheld: hard-constraint violation",
                extra={"violations": violations, "request_id": decision.request_id},
            )
            served = None
        ads = []
        if served is not None:
            ads.append(
                RankedAd(
                    impression_id=served.impression_id,
                    ad_id=served.ad.ad_id,
                    position=served.position,
                    impression_token=served.token,
                    expires_at=served.expires_at,
                    click_url=f"/v1/click/{served.token}",
                )
            )
        return RankResponse(
            request_id=decision.request_id,
            ads=ads,
            served_by=decision.served_by.value,
            policy_version=decision.policy_version,
        )

    @app.post("/v1/events", status_code=status.HTTP_202_ACCEPTED)
    def events(body: ClientEvent, user_id: User) -> JSONResponse:
        limit("events", user_id)
        with c.lock:
            result = c.ingestor.ingest(body, user_id, c.clock())
        c.metrics.client_events.labels(result.status.value).inc()
        code = {
            IngestStatus.ACCEPTED: status.HTTP_202_ACCEPTED,
            IngestStatus.DUPLICATE: status.HTTP_200_OK,
            IngestStatus.REJECTED: status.HTTP_400_BAD_REQUEST,
        }[result.status]
        # Rejection reasons stay server-side (fraud analysis); clients only learn the status.
        return JSONResponse({"status": result.status.value}, status_code=code)

    @app.get("/v1/click/{token}")
    def click(token: str, request: Request):
        limit("click", request.client.host if request.client else "unknown")
        with c.lock:
            result = c.ingestor.click(token, c.clock())
        if result.status is IngestStatus.REJECTED or result.landing_url is None:
            return JSONResponse({"status": "rejected"}, status_code=status.HTTP_400_BAD_REQUEST)
        return RedirectResponse(
            _with_query(result.landing_url, click_id=result.click_id),
            status_code=status.HTTP_302_FOUND,
            # The landing page must not learn which Theraflo screen the user came from.
            headers={"Referrer-Policy": "no-referrer", "Cache-Control": "no-store"},
        )

    @app.post("/v1/postbacks/conversion")
    def conversion(body: ConversionPostback, advertiser_id: Advertiser) -> JSONResponse:
        limit("postback", advertiser_id)
        now = c.clock()
        with c.lock:
            outcome = c.platform.events.conversion(
                advertiser_id=advertiser_id,
                click_id=body.click_id,
                order_id=body.order_id,
                value=body.value,
                occurred_at=body.occurred_at,
                received_at=now,
            )
        if outcome is ConversionOutcome.FORBIDDEN:
            # Answer exactly like an unknown click, so advertisers cannot probe
            # for other advertisers' click IDs.
            c.record_audit("cross_tenant_postback", advertiser_id, click_id=body.click_id)
            outcome = ConversionOutcome.UNKNOWN_CLICK
        code = {
            ConversionOutcome.ATTRIBUTED: status.HTTP_202_ACCEPTED,
            ConversionOutcome.DUPLICATE: status.HTTP_200_OK,
            ConversionOutcome.EXPIRED: status.HTTP_202_ACCEPTED,
            ConversionOutcome.INVALID: status.HTTP_422_UNPROCESSABLE_CONTENT,
            ConversionOutcome.UNKNOWN_CLICK: status.HTTP_404_NOT_FOUND,
        }[outcome]
        return JSONResponse({"status": outcome.value}, status_code=code)

    @app.post("/v1/postbacks/reversal")
    def reversal(body: ReversalPostback, advertiser_id: Advertiser) -> JSONResponse:
        limit("postback", advertiser_id)
        with c.lock:
            reversed_ = c.platform.events.reversal(
                advertiser_id=advertiser_id, order_id=body.order_id, at=c.clock()
            )
        if not reversed_:
            return JSONResponse({"status": "not_found"}, status_code=status.HTTP_404_NOT_FOUND)
        return JSONResponse({"status": "reversed"}, status_code=status.HTTP_202_ACCEPTED)

    @app.put("/v1/users/me/consent")
    def set_consent(body: ConsentRequest, user_id: User) -> dict:
        """Personalization is opt-in; revoking takes effect on the next request."""
        with c.lock:
            c.profiles.set_consent(user_id, ConsentScope(body.consent))
        c.record_audit("consent_changed", "user", consent=body.consent)
        return {"consent": body.consent}

    @app.delete("/v1/users/me", status_code=status.HTTP_202_ACCEPTED)
    def delete_me(user_id: User, background: BackgroundTasks) -> dict:
        """Right to deletion: the ads profile goes now; the deletion job shreds the user's
        key (making dataset snapshots unlinkable) and erases operational state."""
        with c.lock:
            c.profiles.forget(user_id)
        details = {}
        if c.deletions is not None:
            request = c.deletions.request(c.hasher.hash(user_id))
            details["request_id"] = request.request_id
            if c.deletion_inline:

                def run() -> None:
                    with c.lock:
                        c.deletions.run_pending()

                background.add_task(run)
        # The audit trail must not keep the identity the user asked us to delete.
        c.record_audit("user_deletion_requested", "user", **details)
        return {"status": "deletion_scheduled"}

    @app.post("/v1/internal/suppressions", status_code=status.HTTP_204_NO_CONTENT)
    def suppress(body: SuppressionRequest, actor: Annotated[str, Depends(service)]) -> None:
        """Called by the Theraflo backend when a user opens crisis resources. The reason is
        never sent to the ads service."""
        with c.lock:
            c.profiles.suppress(body.user_id, c.clock() + timedelta(hours=body.hours))

    # ----- creative review (Phase 8) -----

    def review_service():
        if c.review is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "creative review disabled")
        return c.review

    @app.post("/v1/advertiser/creatives", status_code=status.HTTP_202_ACCEPTED)
    def submit_creative(
        body: CreativeSubmission, advertiser_id: Advertiser, background: BackgroundTasks
    ) -> dict:
        """The ad enters the catalog as pending and is never served until reviewed. Review is
        asynchronous: a background task here (memory backend) or the review worker."""
        limit("creative", advertiser_id)
        service = review_service()
        try:
            ad = service.submit(Creative(advertiser_id=advertiser_id, **body.model_dump()))
        except SubmissionError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "unknown campaign") from None
        if c.review_inline:
            background.add_task(service.process_pending)
        return {"ad_id": ad.ad_id, "review_status": ad.review_status.value}

    @app.get("/v1/advertiser/creatives")
    def list_creatives(
        advertiser_id: Advertiser, limit: Annotated[int, Query(ge=1, le=200)] = 50
    ) -> dict:
        """This advertiser's creatives, newest first, with their review status."""
        service = review_service()
        items = []
        for creative in service.repository.creatives_for(advertiser_id)[:limit]:
            found = service.status(creative.ad_id, advertiser_id)
            if found is not None:
                submitted = creative.submitted_at.isoformat() if creative.submitted_at else None
                items.append(
                    {
                        **found,
                        "campaign_id": creative.campaign_id,
                        "headline": creative.headline,
                        "submitted_at": submitted,
                    }
                )
        return {"creatives": items}

    @app.get("/v1/advertiser/campaigns")
    def advertiser_campaigns(advertiser_id: Advertiser) -> dict:
        """This advertiser's campaigns, with what was billed today (the durable ledger)."""
        today = c.clock().date()
        catalog = c.catalog
        campaigns = []
        for campaign in catalog.campaigns.values():
            if campaign.advertiser_id != advertiser_id:
                continue
            billed = c.platform.budget.billed(campaign.campaign_id, today)
            campaigns.append(
                {
                    "campaign_id": campaign.campaign_id,
                    "pricing_model": campaign.pricing_model.value,
                    "price": campaign.price,
                    "daily_budget": campaign.daily_budget,
                    "billed_today": billed,
                    "remaining_today": max(campaign.daily_budget - billed, 0.0),
                    "active": campaign.active,
                    "placements": sorted(campaign.placements) if campaign.placements else None,
                    "ads": [
                        {
                            "ad_id": ad.ad_id,
                            "category": ad.category,
                            "review_status": ad.review_status.value,
                        }
                        for ad in catalog.ads.values()
                        if ad.campaign_id == campaign.campaign_id
                    ],
                }
            )
        return {"campaigns": campaigns}

    @app.get("/v1/advertiser/creatives/{ad_id}")
    def creative_status(ad_id: str, advertiser_id: Advertiser) -> dict:
        found = review_service().status(ad_id, advertiser_id)
        if found is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        return found

    @app.get("/v1/advertiser/report")
    def advertiser_report(
        advertiser_id: Advertiser,
        start: Annotated[AwareDatetime, Query()],
        end: Annotated[AwareDatetime, Query()],
    ) -> dict:
        """Aggregated with minimum segment sizes; never user-level, never health-derived."""
        limit("report", advertiser_id)
        if c.report_source is None:
            raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, "reports unavailable")
        if not start < end or end - start > timedelta(days=92):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, "invalid period")
        labels, decisions = c.report_source(advertiser_id, start, end)
        report = build_advertiser_report(advertiser_id, labels, decisions, start=start, end=end)
        narrative = narrate(report, c.narrator)
        return {
            **report.to_dict(),
            "narrative": narrative.text,
            "narrative_source": narrative.source,
        }

    @app.get("/v1/ads/{impression_id}/why")
    def why_this_ad(impression_id: str, user_id: User) -> dict:
        """Templated from the logged decision; only the user who saw the ad can ask."""
        limit("why", user_id)
        decision = c.platform.find_decision(impression_id)
        if decision is None or decision.user_hash != c.hasher.hash(user_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        try:
            explanation = explain(
                decision,
                c.catalog,
                c.taxonomy,
                c.platform.safety.current().personalization_excluded_categories,
            )
        except ValueError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found") from None
        return explanation.to_dict()

    Admin = Annotated[str, Depends(admin)]

    @app.get("/v1/admin/review-queue")
    def review_queue(actor: Admin, kind: Literal["review", "audit"] | None = None) -> dict:
        items = review_service().queue(QueueKind(kind) if kind else None)
        return {"items": [i.to_dict() for i in items]}

    @app.get("/v1/admin/review-queue/{ad_id}")
    def review_item(ad_id: str, actor: Admin) -> dict:
        service = review_service()
        creative = service.repository.creative(ad_id)
        record = service.repository.latest_review(ad_id)
        if creative is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "not found")
        return {"creative": creative.to_dict(), "review": record.to_dict() if record else None}

    @app.post("/v1/admin/review-queue/{ad_id}")
    def decide_review(ad_id: str, body: HumanReviewDecision, actor: Admin) -> dict:
        try:
            decision = review_service().decide(
                ad_id,
                kind=QueueKind(body.kind),
                verdict=Verdict(body.verdict),
                reviewer=actor,
                category=body.category,
                violations=body.violations,
                notes=body.notes,
            )
        except KeyError:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "no open queue item") from None
        except ValueError as exc:
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from None
        c.record_audit("creative_decided", actor, ad_id=ad_id, verdict=body.verdict, kind=body.kind)
        return decision.to_dict()

    @app.get("/v1/admin/review-audit")
    def review_audit(actor: Admin) -> dict:
        """Measured false-approval rate of auto-approved ads, with a 95% interval."""
        return review_service().audit_summary()

    @app.get("/v1/admin/claims")
    def list_claims(
        actor: Admin,
        advertiser_id: Annotated[str, Query(min_length=1, max_length=64, pattern=_ID)],
    ) -> dict:
        claims = review_service().repository.claims_for(advertiser_id)
        return {
            "claims": [
                {
                    "claim_id": claim.claim_id,
                    "advertiser_id": claim.advertiser_id,
                    "text": claim.text,
                    "evidence": claim.evidence,
                }
                for claim in claims
            ]
        }

    @app.post("/v1/admin/claims", status_code=status.HTTP_201_CREATED)
    def add_claim(body: ClaimRequest, actor: Admin) -> dict:
        """Substantiated claims are entered by policy owners after checking the evidence."""
        review_service().repository.add_claim(SubstantiatedClaim(**body.model_dump()))
        c.record_audit(
            "claim_substantiated", actor, claim_id=body.claim_id, advertiser_id=body.advertiser_id
        )
        return {"claim_id": body.claim_id}

    @app.get("/v1/admin/kill-switch")
    def kill_switch_state(actor: Admin) -> dict:
        policy = c.platform.safety.current()
        return {"kill_switch": policy.kill_switch, "safety_policy_version": policy.version}

    @app.get("/v1/admin/audit")
    def audit_log(actor: Admin, limit: Annotated[int, Query(ge=1, le=500)] = 100) -> dict:
        """The audit trail, newest first: reviews, claims, the kill switch, console actions."""
        entries = c.recent_audit(limit)
        return {"entries": [{k: _plain(v) for k, v in e.items()} for e in entries]}

    @app.put("/v1/admin/kill-switch")
    def kill_switch(body: KillSwitchRequest, actor: Annotated[str, Depends(admin)]) -> dict:
        with c.lock:
            policy = c.platform.safety.set_kill_switch(body.enabled)
        c.record_audit("kill_switch", actor, enabled=body.enabled, reason=body.reason)
        return {"kill_switch": policy.kill_switch, "safety_policy_version": policy.version}

    if c.dev_mode:
        _mount_console(app, c, admin)
    return app


def _mount_console(app: FastAPI, c: Container, admin) -> None:
    """The dev console and its API, only with dev credentials (never in production)."""
    from theraflo_ads.api.dev import dev_router
    from theraflo_ads.api.jobs import JobRunner

    reports = Path(os.environ.get("THERAFLO_ADS_REPORTS_DIR", "reports"))
    jobs = JobRunner(Path.cwd(), backend=c.backend)
    app.include_router(dev_router(c, admin, reports_root=reports, jobs=jobs))
    app.mount("/ui", StaticFiles(directory=CONSOLE_DIR, html=True), name="console")

    @app.get("/", include_in_schema=False)
    def home() -> RedirectResponse:
        return RedirectResponse("/ui/")

    @app.middleware("http")
    async def console_headers(request: Request, call_next):
        response = await call_next(request)
        if request.url.path.startswith("/ui"):
            response.headers["Content-Security-Policy"] = CONSOLE_CSP
            response.headers["X-Content-Type-Options"] = "nosniff"
            response.headers["Referrer-Policy"] = "no-referrer"
            response.headers["Cache-Control"] = "no-cache"
        return response


def _auth_fallback_enabled() -> bool:
    value = os.environ.get("THERAFLO_ADS_AUTH_FALLBACK_HEADER", "off").strip().lower()
    if value not in {"1", "on", "true"}:
        return False
    if os.environ.get("THERAFLO_ADS_ENV", "development").lower() == "production":
        raise RuntimeError(
            "THERAFLO_ADS_AUTH_FALLBACK_HEADER is for emulators; it cannot run in production"
        )
    log.warning("bearer tokens are also accepted in %s", FALLBACK_AUTH_HEADER)
    return True


def _plain(value):
    return value.isoformat() if isinstance(value, datetime) else value


def _with_query(url: str, **params: str | None) -> str:
    parts = urlsplit(url)
    query = parse_qsl(parts.query, keep_blank_values=True)
    query += [(k, v) for k, v in params.items() if v is not None]
    return urlunsplit(parts._replace(query=urlencode(query)))


def _retry_later() -> JSONResponse:
    return JSONResponse(
        {"status": "unavailable"},
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        headers={"Retry-After": "5"},
    )
