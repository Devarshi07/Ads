"""Command-line helpers that set up a local environment."""

from __future__ import annotations

import pytest

from theraflo_ads.cli import main


def test_gen_keys_writes_an_ascii_env_file(tmp_path, capsys):
    env = tmp_path / ".env"
    main(["gen-keys", "--out", str(env)])
    lines = env.read_bytes().decode("ascii").splitlines()
    names = [line.split("=", 1)[0] for line in lines]
    assert names == [
        "THERAFLO_ADS_USER_HASH_KEY",
        "THERAFLO_ADS_TOKEN_KEY",
        "THERAFLO_ADS_SESSION_KEY",
    ]
    assert all(len(bytes.fromhex(line.split("=", 1)[1])) == 32 for line in lines)
    assert "wrote 3 keys" in capsys.readouterr().out  # the keys themselves are not printed


def test_gen_keys_never_silently_replaces_existing_keys(tmp_path):
    env = tmp_path / ".env"
    env.write_text("THERAFLO_ADS_TOKEN_KEY=keep\n", encoding="ascii")
    with pytest.raises(SystemExit, match="exists"):
        main(["gen-keys", "--out", str(env)])
    assert env.read_text(encoding="ascii") == "THERAFLO_ADS_TOKEN_KEY=keep\n"
    main(["gen-keys", "--out", str(env), "--force"])
    assert "keep" not in env.read_text(encoding="ascii")
