"""Versioned event contracts (design doc: Data Contracts, Event API)."""

from __future__ import annotations

from enum import StrEnum
from typing import Literal

from pydantic import AwareDatetime, BaseModel, ConfigDict, Field

_ID = r"^[A-Za-z0-9_-]+$"


class ClientEventType(StrEnum):
    AD_VIEWABLE = "AD_VIEWABLE"
    AD_HIDDEN = "AD_HIDDEN"
    AD_REPORTED = "AD_REPORTED"


class ClientEvent(BaseModel):
    """Sent by the Theraflo app. Unknown fields are ignored (additive evolution)."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["client_event.v1"] = "client_event.v1"
    event_id: str = Field(min_length=8, max_length=64, pattern=_ID)
    event_type: ClientEventType
    impression_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    ad_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    impression_token: str = Field(min_length=1, max_length=512)
    client_timestamp: AwareDatetime


class ConversionPostback(BaseModel):
    """Server-to-server postback from an advertiser, keyed by the opaque click_id."""

    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["conversion_postback.v1"] = "conversion_postback.v1"
    click_id: str = Field(min_length=1, max_length=64, pattern=_ID)
    order_id: str = Field(min_length=1, max_length=128, pattern=_ID)
    value: float = Field(ge=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    occurred_at: AwareDatetime


class ReversalPostback(BaseModel):
    model_config = ConfigDict(extra="ignore")

    schema_version: Literal["conversion_reversal.v1"] = "conversion_reversal.v1"
    order_id: str = Field(min_length=1, max_length=128, pattern=_ID)
    reason: str = Field(default="refund", max_length=64)
