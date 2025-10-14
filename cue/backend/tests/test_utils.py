import asyncio

import pytest

from backend.main import RateLimiter, sanitize_text


def test_sanitize_text_strips_control_characters_and_truncates() -> None:
    raw = "Hello\x01 world\n\nThis is a test" + "!" * 2100
    cleaned = sanitize_text(raw, limit=50)
    assert "\x01" not in cleaned
    assert "  " not in cleaned
    assert len(cleaned) <= 51  # includes ellipsis when truncated
    assert cleaned.endswith("…")


@pytest.mark.asyncio
async def test_rate_limiter_enforces_window(monkeypatch: pytest.MonkeyPatch) -> None:
    current = 100.0

    def fake_monotonic() -> float:
        return current

    monkeypatch.setattr("backend.main.time.monotonic", fake_monotonic)

    limiter = RateLimiter(max_requests=2, window_seconds=10)

    assert await limiter.allow() is True
    assert await limiter.allow() is True
    assert await limiter.allow() is False

    current += 11
    assert await limiter.allow() is True


@pytest.mark.asyncio
async def test_rate_limiter_lock_is_respected(monkeypatch: pytest.MonkeyPatch) -> None:
    # Fast-forward monotonic time on each call to simulate sequential requests
    current = 200.0

    def fake_monotonic() -> float:
        nonlocal current
        current += 0.1
        return current

    monkeypatch.setattr("backend.main.time.monotonic", fake_monotonic)

    limiter = RateLimiter(max_requests=1, window_seconds=1)

    results = await asyncio.gather(*(limiter.allow() for _ in range(3)))

    assert results.count(True) == 1
    assert results.count(False) == 2
