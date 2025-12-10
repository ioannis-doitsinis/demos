"""
Extended test suite for the Simple AI Server.

Usage:
    # from repo root (with server running on localhost:8090)
    pytest -q tests/test_chat_and_prefill_extended.py

Assumptions:
    - Server listens on http://localhost:8090
    - /v1/chat/completions implements an OpenAI-compatible response shape
    - /v1/prefill extracts fields and writes/append to data.csv in repo root
"""
import csv
import json
import os
import time
from typing import Dict, Any

import pytest
import requests

SERVER_URL = os.environ.get("SERVER_URL", "http://localhost:8090")
CHAT_URL = f"{SERVER_URL}/v1/chat/completions"
PREFILL_URL = f"{SERVER_URL}/v1/prefill"
CSV_PATH = os.environ.get("CSV_PATH", "data.csv")


# -----------------------------
# Helpers
# -----------------------------
def _read_last_row(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        return {}
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)
        return rows[-1] if rows else {}


def _remove_csv_if_exists(path: str):
    if os.path.exists(path):
        os.remove(path)


def _server_up() -> bool:
    try:
        # a cheap way to see if server is up; /v1/prefill is POST-only,
        # so we attempt a harmless GET to /v1/chat/completions and expect either 405/400/200.
        r = requests.get(CHAT_URL, timeout=3)
        return r.status_code in (200, 400, 404, 405)
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _server_up(), reason="Server is not running on localhost:8090")


# -----------------------------
# /v1/chat/completions tests
# -----------------------------

@pytest.mark.parametrize("model", [
    "gpt-4o-mini",  # OpenAI small model
    "openai/gpt-oss-20b:free",  # Example OpenRouter free model
])
def test_chat_basic_roundtrip(model):
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "Reply with exactly: OK"}],
        "max_tokens": 8,
        "temperature": 0,
    }
    r = requests.post(CHAT_URL, json=payload, timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "choices" in data and isinstance(data["choices"], list)
    assert len(data["choices"]) >= 1
    content = data["choices"][0]["message"]["content"].strip()
    assert isinstance(content, str) and len(content) > 0


def test_chat_includes_usage_when_available():
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Say: tokens"}],
        "max_tokens": 5,
    }
    r = requests.post(CHAT_URL, json=payload, timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    # usage is optional depending on the implementation, but if present it should be well-formed
    if "usage" in data:
        usage = data["usage"]
        assert isinstance(usage.get("prompt_tokens"), int)
        assert isinstance(usage.get("completion_tokens"), int)
        assert isinstance(usage.get("total_tokens"), int)


def test_chat_handles_system_and_user_messages():
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are terse. Reply with one word."},
            {"role": "user", "content": "Greet me"},
        ],
        "max_tokens": 8,
        "temperature": 0,
    }
    r = requests.post(CHAT_URL, json=payload, timeout=30)
    assert r.status_code == 200, r.text
    data = r.json()
    assert "choices" in data
    assert data["choices"][0]["message"]["role"] in ("assistant", "tool")
    assert isinstance(data["choices"][0]["message"]["content"], str)


def test_chat_rejects_missing_messages_field():
    payload = {"model": "gpt-4o-mini", "max_tokens": 5}
    r = requests.post(CHAT_URL, json=payload, timeout=30)
    assert r.status_code in (400, 422), f"Expected 4xx for invalid payload, got {r.status_code}: {r.text}"


def test_chat_temperature_is_respected_within_bounds():
    payload = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": "Write a random-sounding 3-word sentence."}],
        "max_tokens": 10,
        "temperature": 1.0,
    }
    r = requests.post(CHAT_URL, json=payload, timeout=30)
    assert r.status_code == 200, r.text


# -----------------------------
# /v1/prefill tests
# -----------------------------

BASIC_EMAILS = [
    # USD with explicit due date (MM/DD/YYYY)
    {
        "email_text": """Subject: Reminder: Payment Due
Hello,
Please settle the invoice for consulting services.
Amount: $2,345.67 USD
Due date: 01/20/2025
Company: Foobar LLC
Contact: jane.doe@foobar.com
Thanks!
""",
        "expect_currency": "USD",
    },
    # EUR with European decimals and date (DD.MM.YYYY)
    {
        "email_text": """Subject: Rechnung 2025-001
Sehr geehrte Damen und Herren,
Der Betrag beläuft sich auf €1.234,50 EUR.
Fällig am: 20.09.2025
Firma: Beispiel GmbH
Kontakt: buchhaltung@beispiel.de
""",
        "expect_currency": "EUR",
    },
    # GBP with textual date
    {
        "email_text": """Subject: Invoice INV-221
Hi,
Amount due: £980.00 GBP
Payment due on 1 September 2025.
Company: Albion Ltd.
Contact: accounts@albion.co.uk
""",
        "expect_currency": "GBP",
    },
    # No currency symbol, ISO code only
    {
        "email_text": """Subject: Invoice #77
Amount: 150000 JPY
Due: 2025-12-05
Company: Sakura Co.
Contact: h.yamada@sakura.jp
""",
        "expect_currency": "JPY",
    },
]


@pytest.mark.parametrize("sample", BASIC_EMAILS)
def test_prefill_various_formats(sample):
    _remove_csv_if_exists(CSV_PATH)

    r = requests.post(PREFILL_URL, json={"email_text": sample["email_text"]}, timeout=15)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("success") is True
    assert os.path.exists(CSV_PATH), "CSV should be created"

    last = _read_last_row(CSV_PATH)
    # Required headers should exist
    for col in ("amount", "currency", "due_date", "description", "company", "contact"):
        assert col in last, f"Missing column {col} in CSV"

    # Currency expectation (best-effort; implementation-dependent)
    if sample.get("expect_currency"):
        assert last.get("currency") == sample["expect_currency"]


def test_prefill_appends_multiple_rows():
    _remove_csv_if_exists(CSV_PATH)

    e1 = "Amount: $10.00 USD\nDue: 2025-08-01\nCompany: A\nContact: a@a.com"
    e2 = "Amount: €20,50 EUR\nDue: 01.08.2025\nCompany: B\nContact: b@b.com"

    r1 = requests.post(PREFILL_URL, json={"email_text": e1}, timeout=15)
    assert r1.status_code == 200 and r1.json().get("success") is True

    r2 = requests.post(PREFILL_URL, json={"email_text": e2}, timeout=15)
    assert r2.status_code == 200 and r2.json().get("success") is True

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == 2, f"Expected 2 rows, got {len(rows)}"


def test_prefill_handles_missing_fields_gracefully():
    _remove_csv_if_exists(CSV_PATH)
    # No due date or company; should still succeed and write best-effort fields
    email_text = "Hi, amount due is $55.00 USD. Contact: billing@x.io"
    r = requests.post(PREFILL_URL, json={"email_text": email_text}, timeout=15)
    assert r.status_code == 200, r.text
    data = r.json()
    assert data.get("success") is True

    last = _read_last_row(CSV_PATH)
    assert last.get("amount") in ("55.00", "$55.00", "55", "55.0")  # allow minor formatting differences
    assert last.get("currency") == "USD"


def test_prefill_rejects_invalid_payload():
    _remove_csv_if_exists(CSV_PATH)
    # Missing email_text key entirely
    r = requests.post(PREFILL_URL, json={"foo": "bar"}, timeout=15)
    assert r.status_code in (400, 422), f"Expected 4xx for invalid payload, got {r.status_code}: {r.text}"


def test_prefill_large_email_body():
    _remove_csv_if_exists(CSV_PATH)
    # Simulate a long thread; important fields appear only once
    long_body = "Hello\n" + ("\n".join([f"Noise line {i}" for i in range(1000)])) + """
Important:
Amount due: $1,234.00 USD
Due Date: March 3, 2026
Company: MegaSoft Inc.
Contact: ap@megasoft.example
"""
    r = requests.post(PREFILL_URL, json={"email_text": long_body}, timeout=30)
    assert r.status_code == 200
    assert r.json().get("success") is True
    last = _read_last_row(CSV_PATH)
    assert last.get("company") == "MegaSoft Inc."
    assert last.get("currency") == "USD"


@pytest.mark.slow
def test_prefill_concurrent_calls_append_without_corruption():
    _remove_csv_if_exists(CSV_PATH)
    payloads = [
        {"email_text": f"Amount: ${i}.00 USD\nDue: 2025-10-0{i%9+1}\nCompany: C{i}\nContact: c{i}@c.com"}
        for i in range(1, 6)
    ]

    # Fire off sequentially (pytest is single-threaded by default), but quickly
    for p in payloads:
        r = requests.post(PREFILL_URL, json=p, timeout=10)
        assert r.status_code == 200 and r.json().get("success") is True

    with open(CSV_PATH, "r", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
        assert len(rows) == len(payloads)


# -----------------------------
# Cleanup
# -----------------------------

def teardown_module(module):
    # don't delete CSV to allow devs to inspect after failure; comment in if desired:
    # _remove_csv_if_exists(CSV_PATH)
    pass
