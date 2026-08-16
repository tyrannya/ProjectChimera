"""Telegram notifier: optional, deduplicated, rate limited, secret-free."""

from __future__ import annotations

import logging

import pytest
import requests

from chimera.notify import TelegramNotifier


@pytest.fixture
def sent(monkeypatch):
    """Capture outbound posts instead of making them."""
    calls: list[dict] = []

    class OK:
        status_code = 200

        def raise_for_status(self):
            return None

    def fake_post(url, data=None, timeout=None):
        calls.append({"url": url, "data": data})
        return OK()

    monkeypatch.setattr(requests, "post", fake_post)
    return calls


@pytest.fixture
def notifier(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "12345")
    return TelegramNotifier(dedup_window_s=60, max_per_minute=5)


# --- optional -------------------------------------------------------------
def test_missing_credentials_disable_notifications_without_raising(monkeypatch):
    """A monitoring integration must never stop the bot from starting."""
    monkeypatch.delenv("TELEGRAM_BOT_TOKEN", raising=False)
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    quiet = TelegramNotifier()
    assert quiet.enabled is False
    assert quiet.send("hello") is False
    assert quiet.send_error("something broke") is False


def test_a_partial_configuration_is_disabled(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.delenv("TELEGRAM_CHAT_ID", raising=False)
    assert TelegramNotifier().enabled is False


def test_credentials_enable_it(notifier):
    assert notifier.enabled is True


# --- sending ---------------------------------------------------------------
def test_a_message_is_posted(notifier, sent):
    assert notifier.send("hello") is True
    assert len(sent) == 1
    assert sent[0]["data"]["chat_id"] == "12345"
    assert sent[0]["data"]["text"] == "hello"


def test_html_in_a_message_is_escaped(notifier, sent):
    notifier.send_error("<script>alert(1)</script>")
    assert "&lt;script&gt;" in sent[0]["data"]["text"]


# --- dedup and rate limiting -------------------------------------------------
def test_an_identical_message_is_sent_once_per_window(notifier, sent):
    for _ in range(10):
        notifier.send("the same thing")
    assert len(sent) == 1


def test_repeated_errors_with_the_same_title_send_once(notifier, sent):
    """One failing dependency used to produce one message per occurrence."""
    for i in range(50):
        notifier.send_error("Inference failure", f"attempt {i}")
    assert len(sent) == 1


def test_different_messages_still_get_through(notifier, sent):
    notifier.send("first")
    notifier.send("second")
    assert len(sent) == 2


def test_the_overall_rate_is_capped(notifier, sent):
    for i in range(20):
        notifier.send(f"distinct message {i}")
    assert len(sent) == 5, "max_per_minute must bound the total, not just duplicates"


def test_the_dedup_table_does_not_grow_without_bound(monkeypatch):
    monkeypatch.setenv("TELEGRAM_BOT_TOKEN", "t")
    monkeypatch.setenv("TELEGRAM_CHAT_ID", "c")
    notifier = TelegramNotifier(dedup_window_s=0.0, max_per_minute=10_000)
    monkeypatch.setattr(
        requests, "post", lambda *a, **k: type("R", (), {"raise_for_status": lambda s: None})()
    )
    for i in range(2000):
        notifier.send(f"message {i}")
    assert len(notifier._recent) <= 1024


# --- failure handling ---------------------------------------------------------
def test_a_transport_failure_is_swallowed(notifier, monkeypatch):
    monkeypatch.setattr(
        requests,
        "post",
        lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("down")),
    )
    assert notifier.send("hello") is False


def test_the_token_never_reaches_the_logs(notifier, monkeypatch, caplog):
    """``str(exc)`` from requests often contains the URL, and the URL contains
    the bot token."""

    def explode(*_, **__):
        raise requests.ConnectionError(
            "failed to connect to https://api.telegram.org/bottest-token/sendMessage"
        )

    monkeypatch.setattr(requests, "post", explode)
    with caplog.at_level(logging.ERROR):
        notifier.send("hello")
    assert "test-token" not in caplog.text
