from datetime import timedelta

import pytest

from tests.helpers import NOW, make_request, make_stack, make_user
from theraflo_ads.counters.frequency import ImpressionRecord
from theraflo_ads.domain import AgeBand, ConsentScope
from theraflo_ads.safety import AdRejection, BlockReason, SafetyPolicy, default_safety_policy
from theraflo_ads.safety.engine import effective_consent


def eligible_ids(stack, request):
    result = stack.engine.evaluate(request)
    return {ad.ad_id for ad in result.eligible}, result


def test_adult_on_allowed_placement_gets_only_safe_approved_active_ads():
    stack = make_stack()
    ids, result = eligible_ids(stack, make_request())
    assert ids == {"ad_cpm", "ad_cpc", "ad_cpa", "ad_dating", "ad_feed"}
    assert result.rejections["ad_gamble"] is AdRejection.RESTRICTED_CATEGORY
    assert result.rejections["ad_inactive"] is AdRejection.CAMPAIGN_INACTIVE
    assert result.rejections["ad_pending"] is AdRejection.NOT_APPROVED


def test_kill_switch_blocks_everything():
    stack = make_stack()
    stack.safety.set_kill_switch(True)
    assert stack.engine.evaluate(make_request()).blocked is BlockReason.KILL_SWITCH


@pytest.mark.parametrize("placement", ["journal", "crisis_resources", "therapy_chat", "unknown"])
def test_placements_off_the_allowlist_are_blocked(placement):
    stack = make_stack()
    result = stack.engine.evaluate(make_request(placement))
    assert result.blocked is BlockReason.PLACEMENT_NOT_ALLOWED


def test_ad_free_placement_cannot_be_put_on_the_allowlist():
    base = default_safety_policy()
    with pytest.raises(ValueError, match="ad-free"):
        SafetyPolicy(
            version="bad",
            ad_placements=base.ad_placements | {"journal"},
            restricted_categories=base.restricted_categories,
            minor_restricted_categories=base.minor_restricted_categories,
            suppression_window=base.suppression_window,
        )


def test_under_13_gets_no_ads():
    stack = make_stack()
    request = make_request(user=make_user(age_band=AgeBand.UNDER_13))
    assert stack.engine.evaluate(request).blocked is BlockReason.UNDER_13


def test_suppression_window_blocks_until_it_ends():
    stack = make_stack()
    user = make_user(suppressed_until=NOW + timedelta(hours=1))
    assert stack.engine.evaluate(make_request(user=user)).blocked is BlockReason.SUPPRESSION_WINDOW
    later = make_request(user=user, now=NOW + timedelta(hours=1))
    assert not stack.engine.evaluate(later).is_blocked


@pytest.mark.parametrize("age_band", [AgeBand.TEEN, AgeBand.UNKNOWN])
def test_minors_and_unknown_age_get_expanded_restrictions_and_contextual_only(age_band):
    stack = make_stack()
    ids, result = eligible_ids(stack, make_request(user=make_user(age_band=age_band)))
    assert "ad_dating" not in ids
    assert result.rejections["ad_dating"] is AdRejection.RESTRICTED_CATEGORY
    assert result.effective_consent is ConsentScope.CONTEXTUAL_ONLY


def test_adult_consent_is_respected():
    assert effective_consent(ConsentScope.PERSONALIZED, AgeBand.ADULT) is ConsentScope.PERSONALIZED
    assert (
        effective_consent(ConsentScope.CONTEXTUAL_ONLY, AgeBand.ADULT)
        is ConsentScope.CONTEXTUAL_ONLY
    )


def test_campaign_placement_targeting():
    stack = make_stack()
    ids, result = eligible_ids(stack, make_request("content_library"))
    assert "ad_feed" not in ids
    assert result.rejections["ad_feed"] is AdRejection.PLACEMENT_TARGETING


def test_hidden_ads_and_advertisers_are_excluded():
    stack = make_stack()
    user = make_user(hidden_ads=frozenset({"ad_cpm"}), hidden_advertisers=frozenset({"adv_b"}))
    ids, result = eligible_ids(stack, make_request(user=user))
    assert {"ad_cpm", "ad_cpc"}.isdisjoint(ids)
    assert result.rejections["ad_cpc"] is AdRejection.HIDDEN_BY_USER


def test_frequency_cap_and_min_gap():
    stack = make_stack()
    user = make_user()
    record = ImpressionRecord(user.user_hash, "ad_cpm", "adv_a", NOW - timedelta(minutes=5))
    stack.frequency.record(record)
    ids, result = eligible_ids(stack, make_request(user=user))
    assert result.rejections["ad_cpm"] is AdRejection.FREQUENCY_CAP  # min gap of 10 minutes
    for minutes in (60, 120):
        stack.frequency.record(
            ImpressionRecord(user.user_hash, "ad_cpm", "adv_a", NOW - timedelta(minutes=minutes))
        )
    later = make_request(user=user, now=NOW + timedelta(hours=1))
    assert stack.engine.evaluate(later).rejections["ad_cpm"] is AdRejection.FREQUENCY_CAP


def test_user_ad_load_cap_blocks_the_request():
    stack = make_stack()
    user = make_user()
    for i in range(stack.frequency.caps.per_user_24h):
        stack.frequency.record(
            ImpressionRecord(user.user_hash, f"other_{i}", f"adv_{i}", NOW - timedelta(hours=1))
        )
    assert stack.engine.evaluate(make_request(user=user)).blocked is BlockReason.AD_LOAD_CAP


def test_unreadable_frequency_state_fails_closed():
    stack = make_stack()
    stack.frequency.available = False
    assert stack.engine.evaluate(make_request()).blocked is BlockReason.FREQUENCY_UNAVAILABLE


def test_cold_counter_cache_with_log_down_fails_closed():
    stack = make_stack()
    stack.frequency.flush()
    stack.log.available = False
    assert stack.engine.evaluate(make_request()).blocked is BlockReason.FREQUENCY_UNAVAILABLE


def test_unreadable_budget_state_fails_closed():
    stack = make_stack()
    stack.budget.available = False
    assert stack.engine.evaluate(make_request()).blocked is BlockReason.BUDGET_UNAVAILABLE


def test_campaign_that_cannot_absorb_one_click_is_ineligible():
    stack = make_stack()
    # c_cpc: $1.00 per click, $50 budget, 2% margin ($1). Leave $1.50 of headroom.
    stack.budget.charge("c_cpc", 48.5, 50.0, NOW, idempotency_key="setup")
    ids, result = eligible_ids(stack, make_request())
    assert "ad_cpc" not in ids
    assert result.rejections["ad_cpc"] is AdRejection.BUDGET


def test_no_eligible_ads_is_a_block_not_an_error():
    stack = make_stack()
    user = make_user(hidden_advertisers=frozenset({"adv_a", "adv_b", "adv_c", "adv_e", "adv_f"}))
    assert stack.engine.evaluate(make_request(user=user)).blocked is BlockReason.NO_ELIGIBLE_ADS
