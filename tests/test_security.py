import ast
from datetime import timedelta
from pathlib import Path

import pytest

from tests.helpers import NOW, TOKEN_KEY
from theraflo_ads.security import ImpressionClaims, ImpressionTokenSigner, InvalidToken, UserHasher

KEY_1 = b"1" * 32
KEY_2 = b"2" * 32


class TestUserHasher:
    def test_deterministic_and_versioned(self):
        hasher = UserHasher("ads", {"k1": KEY_1}, "k1")
        assert hasher.hash("alice") == hasher.hash("alice")
        assert hasher.hash("alice").startswith("k1:")
        assert hasher.hash("alice") != hasher.hash("bob")

    def test_purposes_cannot_be_joined(self):
        ads = UserHasher("ads", {"k1": KEY_1}, "k1")
        analytics = UserHasher("analytics", {"k1": KEY_1}, "k1")
        assert ads.hash("alice") != analytics.hash("alice")

    def test_rotation_keeps_old_hashes_verifiable(self):
        old = UserHasher("ads", {"k1": KEY_1}, "k1")
        rotated = UserHasher("ads", {"k1": KEY_1, "k2": KEY_2}, "k2")
        legacy_hash = old.hash("alice")
        assert rotated.hash("alice").startswith("k2:")
        assert rotated.matches("alice", legacy_hash)
        assert not rotated.matches("bob", legacy_hash)
        assert not rotated.matches("alice", "k9:" + legacy_hash[3:])
        assert not rotated.matches("alice", "no-version")

    @pytest.mark.parametrize(
        "keys,current",
        [({"k1": b"short"}, "k1"), ({"k1": KEY_1}, "k2"), ({"k:1": KEY_1}, "k:1")],
    )
    def test_rejects_bad_configuration(self, keys, current):
        with pytest.raises(ValueError):
            UserHasher("ads", keys, current)


class TestImpressionTokens:
    claims = ImpressionClaims("imp1", "ad_1", "k1:abc", NOW + timedelta(minutes=10))

    def test_roundtrip(self):
        signer = ImpressionTokenSigner(TOKEN_KEY)
        verified = signer.verify(signer.sign(self.claims), NOW)
        assert verified == self.claims

    def test_tampered_payload_is_rejected(self):
        signer = ImpressionTokenSigner(TOKEN_KEY)
        other = signer.sign(ImpressionClaims("imp2", "ad_1", "k1:abc", self.claims.expires_at))
        token = signer.sign(self.claims)
        forged = other.split(".")[0] + "." + token.split(".")[1]
        with pytest.raises(InvalidToken, match="bad_signature"):
            signer.verify(forged, NOW)

    def test_wrong_key_is_rejected(self):
        token = ImpressionTokenSigner(TOKEN_KEY).sign(self.claims)
        with pytest.raises(InvalidToken, match="bad_signature"):
            ImpressionTokenSigner(b"x" * 32).verify(token, NOW)

    def test_expiry_and_grace(self):
        signer = ImpressionTokenSigner(TOKEN_KEY)
        token = signer.sign(self.claims)
        late = NOW + timedelta(minutes=30)
        with pytest.raises(InvalidToken, match="expired"):
            signer.verify(token, late)
        assert signer.verify(token, late, grace=timedelta(hours=1)).impression_id == "imp1"

    @pytest.mark.parametrize("token", ["", "abc", "a.b.c", "!!!.???"])
    def test_malformed(self, token):
        with pytest.raises(InvalidToken):
            ImpressionTokenSigner(TOKEN_KEY).verify(token, NOW)

    def test_fields_cannot_smuggle_the_separator(self):
        signer = ImpressionTokenSigner(TOKEN_KEY)
        with pytest.raises(ValueError):
            signer.sign(ImpressionClaims("imp|1", "ad_1", "k1:abc", NOW))

    def test_short_key_rejected(self):
        with pytest.raises(ValueError):
            ImpressionTokenSigner(b"short")


SECURITY_CODE = ("security", "privacy", "observability/ratelimit.py", "infra/privacy_store.py")


def test_security_code_never_uses_the_random_module():
    """Bandit's B311 is skipped project-wide because `random` drives exploration and
    simulation; this keeps it out of everything that creates or checks secrets."""
    root = Path(__file__).resolve().parents[1] / "src" / "theraflo_ads"
    files = [
        f
        for target in SECURITY_CODE
        for f in ((root / target).rglob("*.py") if (root / target).is_dir() else [root / target])
    ]
    assert files
    for path in files:
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue

            assert "random" not in names, f"{path} imports random"
