"""Attribution facts (API path) and accounting (worker path) through the in-process bus."""

from datetime import timedelta

import pytest

from tests.helpers import NOW, make_request, make_stack
from theraflo_ads.attribution import ConversionOutcome, LabelState
from theraflo_ads.events.bus import Message, MessageType
from theraflo_ads.policy import PopularityPolicy


def serve(ad_id):
    """A stack whose only policy serves `ad_id`, plus the served impression."""
    stack = make_stack(chain=lambda catalog: [PopularityPolicy([ad_id])])
    outcome = stack.service.decide(make_request())
    assert outcome.served is not None and outcome.served.ad.ad_id == ad_id
    return stack, outcome.served


def convert(stack, click_id, order_id="o1", advertiser="adv_c", days=1.0):
    at = NOW + timedelta(days=days)
    return stack.events.conversion(
        advertiser_id=advertiser,
        click_id=click_id,
        order_id=order_id,
        value=55.0,
        occurred_at=at,
        received_at=at,
    )


def test_cpm_bills_once_on_first_viewable():
    stack, served = serve("ad_cpm")
    assert stack.events.viewable(served.impression_id, "ad_cpm", NOW)
    assert not stack.events.viewable(served.impression_id, "ad_cpm", NOW)  # same event id
    assert stack.budget.billed("c_cpm", NOW.date()) == pytest.approx(0.01)
    assert stack.budget.reserved("c_cpm", NOW.date()) == 0
    assert stack.labels.label(served.impression_id).platform_revenue == pytest.approx(0.01)


def test_redelivered_messages_never_double_bill():
    stack, served = serve("ad_cpm")
    stack.events.viewable(served.impression_id, "ad_cpm", NOW)
    message = next(m for m in stack.bus.published if m.type is MessageType.AD_VIEWABLE)
    for _ in range(3):  # at-least-once delivery
        stack.accounting.on_viewable(message)
    assert stack.budget.billed("c_cpm", NOW.date()) == pytest.approx(0.01)
    assert stack.labels.label(served.impression_id).platform_revenue == pytest.approx(0.01)


def test_display_after_expiry_is_not_billed():
    stack, served = serve("ad_cpm")
    stack.events.viewable(served.impression_id, "ad_cpm", NOW, billable=False)
    assert stack.budget.billed("c_cpm", NOW.date()) == 0
    assert stack.labels.label(served.impression_id).viewable


def test_cpc_bills_once_per_impression_and_click_ids_are_stable():
    stack, served = serve("ad_cpc")
    first = stack.events.click(served.impression_id, NOW)
    second = stack.events.click(served.impression_id, NOW + timedelta(seconds=5))
    assert first == second
    assert stack.budget.billed("c_cpc", NOW.date()) == pytest.approx(1.0)


def test_cpa_conversion_lifecycle():
    stack, served = serve("ad_cpa")
    click_id = stack.events.click(served.impression_id, NOW)
    label = stack.labels.label(served.impression_id)

    assert convert(stack, click_id) is ConversionOutcome.ATTRIBUTED
    assert label.state is LabelState.PARTIALLY_OBSERVED
    assert label.platform_revenue == pytest.approx(20.0)
    assert label.conversion_value == pytest.approx(55.0)
    assert stack.budget.billed("c_cpa", (NOW + timedelta(days=1)).date()) == pytest.approx(20.0)

    assert convert(stack, click_id) is ConversionOutcome.DUPLICATE
    assert convert(stack, click_id, "o2", days=8) is ConversionOutcome.EXPIRED
    assert convert(stack, click_id, "o3", days=-1) is ConversionOutcome.INVALID
    assert convert(stack, "nope", "o4") is ConversionOutcome.UNKNOWN_CLICK
    assert convert(stack, click_id, "o5", advertiser="adv_a") is ConversionOutcome.FORBIDDEN
    assert label.conversions == 1

    assert stack.events.reversal(advertiser_id="adv_c", order_id="o1", at=NOW)
    assert not stack.events.reversal(advertiser_id="adv_c", order_id="o1", at=NOW)
    assert label.state is LabelState.REVERSED
    assert label.platform_revenue == pytest.approx(0.0)
    assert stack.budget.credited("c_cpa", NOW.date()) == pytest.approx(20.0)


def test_billing_outage_queues_work_instead_of_failing():
    stack, served = serve("ad_cpa")
    click_id = stack.events.click(served.impression_id, NOW)
    stack.budget.available = False
    # The API still answers: the fact is recorded, the accounting waits on the queue.
    assert convert(stack, click_id) is ConversionOutcome.ATTRIBUTED
    assert stack.labels.label(served.impression_id).conversions == 0
    assert stack.bus.counts() == {"pending": 1, "dead_letters": 0}

    stack.budget.available = True
    assert stack.bus.retry_pending() == 1
    assert stack.labels.label(served.impression_id).conversions == 1
    assert stack.budget.billed("c_cpa", (NOW + timedelta(days=1)).date()) == pytest.approx(20.0)


def test_refund_processed_before_its_conversion_is_never_charged():
    stack, served = serve("ad_cpa")
    click_id = stack.events.click(served.impression_id, NOW)
    stack.budget.available = False
    convert(stack, click_id)  # accounting pending
    stack.budget.available = True
    stack.events.reversal(advertiser_id="adv_c", order_id="o1", at=NOW)  # reversal applies first
    stack.bus.retry_pending()
    label = stack.labels.label(served.impression_id)
    assert (label.conversions, label.platform_revenue) == (0, 0.0)
    assert stack.budget.billed("c_cpa", (NOW + timedelta(days=1)).date()) == 0


def test_unknown_impression_is_dead_lettered():
    stack, _ = serve("ad_cpm")
    payload = {"impression_id": "nope", "at": NOW.isoformat()}
    stack.bus.publish(Message("evt:x", MessageType.AD_VIEWABLE, payload, NOW))
    assert stack.bus.counts() == {"pending": 0, "dead_letters": 1}


def test_storage_outage_fails_the_request_so_the_client_retries():
    stack, served = serve("ad_cpm")
    stack.store.available = False
    with pytest.raises(Exception, match="unavailable"):
        stack.events.viewable(served.impression_id, "ad_cpm", NOW)
    stack.store.available = True
    assert stack.events.viewable(served.impression_id, "ad_cpm", NOW)


def test_labels_mature_after_the_window_and_grace():
    stack, served = serve("ad_cpm")
    stack.events.viewable(served.impression_id, "ad_cpm", NOW)
    attribution = stack.labels
    almost = NOW + attribution.policy.maturity_delay - timedelta(seconds=1)
    assert attribution.rewards_by_request(almost) == {}
    assert attribution.finalize(almost) == 0
    mature = NOW + attribution.policy.maturity_delay
    assert attribution.finalize(mature) == 1
    assert attribution.label(served.impression_id).state is LabelState.FINALIZED
    assert attribution.rewards_by_request(mature) == {"r1": pytest.approx(0.01)}
