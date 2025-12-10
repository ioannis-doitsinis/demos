"""
AI Server for Chat Completions and Email Prefill Extraction.

Endpoints
---------
1) /v1/chat/completions
   - Supports:
       - Free OpenRouter models (vendor/name:free)
       - Paid OpenAI models (e.g., "gpt-4o-mini", "gpt-4.1", "o3-mini") via OPENAI_API_KEY

2) /v1/prefill
   - Extracts payment-related fields from raw email text and appends a row to data.csv.
   - Fields: amount, currency, due_date, description, company, contact
"""

import os
import csv
import uvicorn
import requests
import re

from fastapi import FastAPI, Body, HTTPException
from fastapi.responses import JSONResponse # for OpenAI exact format

from typing import Any, Dict, Optional, List, Tuple
from pydantic import BaseModel #data validation

from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Provider endpoints
OPENAI_URL = "https://api.openai.com/v1/chat/completions"
OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"

# for debugging.
print("OPENROUTER_API_KEY:", OPENROUTER_API_KEY)
print("OPENAI_API_KEY configured:", OPENAI_API_KEY) 

# sample of free models (OpenRouter)
FREE_MODELS = [
    "openai/gpt-oss-20b:free",
    "meta-llama/llama-3.2-3b-instruct:free",
    "google/gemma-2-9b-it:free",
    "mistralai/mistral-7b-instruct:free",
]

PAID_OPENAI_MODELS = [  # (used only for your own reference)
    "gpt-4.1-mini",
    "gpt-4.1",
    "gpt-4o-mini",
    "gpt-4o",
    "o3-mini",
]

app = FastAPI()


def pick_backend(model: str) -> Tuple[str, str, Dict[str, str]]:
    """
    Decide which upstream provider to use based on the model name.

    Rules:
      - If model ends with ':free' → use OpenRouter (free-tier).
      - Otherwise → use OpenAI with OPENAI_API_KEY.

    Returns:
        (provider_name, url, headers)

    Raises:
        HTTPException: if model is missing/unsupported or API key is not configured.
    """

    if not model:
        raise HTTPException(status_code=400, detail="Missing 'model'")

    # --- OpenRouter free models ---
    if model.endswith(":free"):
        if not OPENROUTER_API_KEY:
                raise HTTPException(status_code=500, detail="OPENROUTER_API_KEY not configured")
        return (
            "openrouter",
            OPENROUTER_URL,
            {
                "Authorization": f"Bearer {OPENROUTER_API_KEY}",
                "Content-Type": "application/json",
                "HTTP-Referer": "http://127.0.0.1:8090",
                "X-Title": "Proxy",
            },
        )     

    # OpenAI direct (budgeted key)
    if not OPENAI_API_KEY:
         # User is trying to use a paid model but there is no API key
        raise HTTPException(status_code=500, detail="OPENAI_API_KEY not configured")
    return (
        "openai",
        OPENAI_URL,
        {"Authorization": f"Bearer {OPENAI_API_KEY}", "Content-Type": "application/json"},
    )

     

# Remove hidden "<think>...</think>" blocks if they slip through
THINK_BLOCK_RE = re.compile(r"<think>.*?</think>\s*", flags=re.IGNORECASE | re.DOTALL)

def _coalesce_choice_content(choice: Dict[str, Any]) -> str:
    """
    Normalize a single choice's content into a clean string and strip think blocks.

    Handles cases where providers return content as a list of segments rather than a string.
    """
    msg = choice.get("message") or {}
    content = msg.get("content")

    if isinstance(content, list):
        parts = []
        for seg in content:
            if isinstance(seg, dict) and isinstance(seg.get("text"), str):
                parts.append(seg["text"])
            elif isinstance(seg, str):
                parts.append(seg)
        content = "\n".join(parts).strip()

    if isinstance(content, str):
        content = THINK_BLOCK_RE.sub("", content).strip()

    return content or ""


def _strip_reasoning_fields(resp: Dict[str, Any]) -> Dict[str, Any]:
    """
    Ensure response content is normalized and chain-of-thought fields are removed.
    """
    for ch in resp.get("choices", []):
        msg = ch.get("message") or {}
        msg["content"] = _coalesce_choice_content(ch)
        msg.pop("reasoning", None)
        ch["message"] = msg
    return resp


def normalize_payload_for_backend(provider: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Return a copy of payload adjusted for the target provider.
    For OpenRouter:
      - Use `max_completion_tokens` instead of `max_tokens`.
      - Add `reasoning` field to hide chain-of-thought unless explicitly requested.
    """
    p = dict(payload)  # shallow copy so caller's body isn't mutated

    if provider == "openrouter":
        # Align token field name
        if "max_tokens" in p and "max_completion_tokens" not in p:
            p["max_completion_tokens"] = p.pop("max_tokens")

        # Give thinking models room to finish; avoid empty/trimmed outputs.
        if not isinstance(p.get("max_completion_tokens"), int) or p["max_completion_tokens"] < 256:
            p["max_completion_tokens"] = 512

        # Hide chain-of-thought by default unless caller explicitly asks otherwise.
        if "reasoning" not in p and "include_reasoning" not in p:
            p["reasoning"] = {"exclude": True, "effort": "low"}  # hide CoT, keep it cheap

    elif provider == "openai":
        # Normalize model names: allow "openai/gpt-4o-mini" as alias for "gpt-4o-mini"
        model_name = p.get("model", "")
        if isinstance(model_name, str) and model_name.startswith("openai/"):
            p["model"] = model_name.split("/", 1)[1]

        # Ensure `max_tokens` exists in a sane range (optional, but nice)
        if "max_completion_tokens" in p and "max_tokens" not in p:
            # If user mistakenly sends OpenRouter-style field, map it back.
            p["max_tokens"] = p.pop("max_completion_tokens")

        if isinstance(p.get("max_tokens"), int) and p["max_tokens"] <= 0:
            p["max_tokens"] = 256

    return p



def try_call_provider(payload: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    """
    Forward the request to the selected provider and capture errors in a uniform shape.

    Returns:
        (json, err)
          - success -> (provider_json, None)
          - failure -> (None, {"status": int|None, "body": str})
    """
    model = payload.get("model", "")
    provider, url, headers = pick_backend(model)

    # normalize per backend (max_completion_tokens or max_tokens)
    fwd = normalize_payload_for_backend(provider, payload)

    try:
        r = requests.post(url, json=fwd, headers=headers, timeout=60)  # wait up to 60secs to make a connection
    except requests.RequestException as e:
        return None, {"status": None, "body": f"Upstream request failed: {e!s}"}

    if r.ok:
        return r.json(), None

    return None, {"status": r.status_code, "body": r.text[:2000]}


def find_available_model(exclude: Optional[str] = None) -> Optional[str]:
    """
    Probe the FREE_MODELS list and return the first model that successfully
    responds to a tiny 'ping' request.

    `exclude`: model to skip (e.g. the one the user asked for that just failed).
    """
    for candidate in FREE_MODELS:
        if exclude and candidate == exclude:
            continue

        probe_body = {
            "model": candidate,
            "messages": [{"role": "user", "content": "ping"}],
            "max_tokens": 1,
        }
        provider_json, err = try_call_provider(probe_body)
        if provider_json is not None:
            print(f"[availability] Model '{candidate}' appears AVAILABLE.")
            return candidate
        else:
            print(f"[availability] Model '{candidate}' unavailable or failed: {err}")

    return None


# ---------- 1: Chat Completions Endpoint ----------
@app.post("/v1/chat/completions")
def chat_completions(body: Dict[str, Any] = Body(...)):
    """
    Multi-model chat completions endpoint.

    Body :
        {
          "model": "deepseek/…:free, "..../....:free",
          "messages": [{"role":"user","content":"..."}],
          "max_tokens": 256,  # optional
          ...
        }

    Returns:
        Upstream provider JSON (normalized to remove hidden reasoning), or a local
        fallback response if all providers fail.
    """

    if "messages" not in body:
        raise HTTPException(status_code=400, detail="Missing 'messages'")

    model = body.get("model", "")
    messages = body.get("messages", [])

    # 1) Primary attempt
    try:
        provider_json, err = try_call_provider(body)
    except HTTPException as exc:
        # We try to suggest an alternative *free* model if possible.
        suggested = find_available_model(exclude=None)
        if suggested:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": "invalid_or_unsupported_model",
                    "message": (
                        f"Model '{model}' cannot be used with this proxy. "
                        f"Try using '{suggested}' (OpenRouter free) "
                        f"or a valid OpenAI model like 'gpt-4o-mini' if you configured OPENAI_API_KEY."
                    ),
                    "requested_model": model,
                    "recommended_model": suggested,
                    "known_free_models": FREE_MODELS,
                },
            )
        # No suggestion available, re-raise original error
        raise exc

    if provider_json is not None:
        return JSONResponse(content=_strip_reasoning_fields(provider_json))

    # 2) Requested model failed, try to find working free model (OpenRouter)
    print("Primary provider error:", err)

    suggested = find_available_model(exclude=model if model.endswith(":free") else None)
    if suggested:
        # We *don't* auto-switch the model; we just tell the user what to use.
        raise HTTPException(
            status_code=503,
            detail={
                "error": "model_unavailable",
                "message": (
                    f"Requested model '{model}' is currently unavailable upstream. "
                    f"If you're using a free model, switch to '{suggested}' or another "
                    f"free model from the list. For paid OpenAI models, check your "
                    f"OPENAI_API_KEY and model name."
                ),
                "requested_model": model,
                "recommended_model": suggested,
                "known_free_models": FREE_MODELS,
                "upstream_error": err,
            },
        )

    # 3) Local fallback
    print("Provider error:", err)  # helpful for debugging

    prompt = ""
    for m in reversed(messages or []):
        if m.get("role") == "user":
            prompt = m.get("content", "")
            break

    return JSONResponse(
        content={
            "id": "chatcmpl-local",
            "object": "chat.completion",
            "created": 0,
            "model": model or "proxy-local",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": f"I’m running in fallback mode. You said: {prompt[:200]}",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 0, "completion_tokens": 20, "total_tokens": 20},
            "system_fingerprint": "local-stub",
        },
        status_code=200,
    )
    
########## --------------------------------------------------------- ##########
########## -------------------Second Endpoint----------------------- ##########
########## --------------------------------------------------------- ##########
########## --------------------------------------------------------- ##########
########## --------------------------------------------------------- ##########

class PrefillRequest(BaseModel):
    """Request schema for /v1/prefill."""
    email_text: str
    model: Optional[str] = None  # accepted but not used yet

class PrefillResponse(BaseModel):
    """Response schema for /v1/prefill."""
    success: bool
    message: str
    data: Optional[Dict[str, Optional[str]]] = None

# Currency helpers
SYM_TO_CODE = {"$": "USD", "€": "EUR", "£": "GBP", "¥": "JPY", "￥": "JPY"}
ISO_CODES = r"USD|EUR|GBP|JPY|AUD|CAD|CHF|CNY|INR|SEK|NOK|DKK|ZAR|NZD|SGD"

def extract_amount_and_currency(text: str):
    """
    Heuristically extract the first (amount, currency) pair from email text.

    Supports:
      - 'Amount Due: $1,500.00 USD'
      - 'Amount Due: 1500.00 USD'
      - '$1,500.00'
      - 'USD 1,500.00' / '1,500.00 USD'
    """
    # Case 1: "Amount Due: $1,500.00 USD"
    m = re.search(
        rf"(?i)amount\s+due[^\n]*?([$€£¥￥])\s*([\d,]+(?:\.\d{2})?)\s*({ISO_CODES})?",
        text
    )
    if m:
        sym, amt, code = m.groups()
        amt = amt.replace(",", "")
        currency = code.upper() if code else SYM_TO_CODE.get(sym)
        return amt, currency

    # Case 2: "Amount Due: 1500.00 USD" (no symbol, just code)
    m = re.search(
        rf"(?i)amount\s+due[^\n]*?([\d,]+(?:\.\d{2})?)\s*({ISO_CODES})",
        text
    )
    if m:
        amt, code = m.groups()
        return amt.replace(",", ""), code.upper()

    # Case 3: first standalone symbol + amount (like "$1,500.00")
    m = re.search(r"([$€£])\s*([\d,]+(?:\.\d{2})?)", text)
    if m:
        sym, amt = m.groups()
        return amt.replace(",", ""), SYM_TO_CODE.get(sym)

    # Case 4: first code + amount (like "USD 1,500.00")
    m = re.search(rf"({ISO_CODES})\s*([\d,]+(?:\.\d{2})?)", text, re.I)
    if m:
        code, amt = m.groups()
        return amt.replace(",", ""), code.upper()

    m = re.search(rf"([\d,]+(?:\.\d{2})?)\s*({ISO_CODES})", text, re.I)
    if m:
        amt, code = m.groups()
        return amt.replace(",", ""), code.upper()

    return None, None


def extract_due_date(text: str) -> Optional[str]:
    """
    Extract a due date string.

    Prefers a line like 'Due Date: January 15, 2025'.
    Falls back to simple date patterns (e.g., 'January 15, 2025', '2025-01-15', '01/15/2025').
    """
    m = re.search(r"(?i)Due\s*Date\s*[:\-]\s*([^\n]+)", text)
    if m:
        return m.group(1).strip()

    
    m = re.search(r"([A-Za-z]{3,9}\s+\d{1,2},\s*\d{4}|\d{4}-\d{2}-\d{2}|\d{1,2}/\d{1,2}/\d{4})", text)
    return m.group(1).strip() if m else None


def extract_subject(text: str) -> Optional[str]:
    """Return the email subject line, if present."""
    m = re.search(r"(?im)^Subject:\s*(.+)$", text)
    return m.group(1).strip() if m else None

def extract_description(text: str) -> Optional[str]:
    """
    Keep it short; default to the Subject line.

    If there is no subject, use the first non-empty paragraph (up to ~180 chars).
    """
    subj = extract_subject(text)
    if subj:
        return subj
    # fallback: first non-empty paragraph
    parts = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    return parts[0][:180] if parts else None


def extract_company(text: str) -> Optional[str]:
    """
    Extract a plausible company name from the email.

    Strategies:
      1) A line starting with 'Company:'.
      2) Subject pattern '... from <Company> - ...'.
      3) Signature/footer lines containing Corp/Inc/LLC/Ltd/GmbH/AG.
      4) Fallback: use the base of the sender's email domain.
    """

    # 1. Explicit "Company:" line
    for line in text.splitlines():
        line = line.strip()
        if line.lower().startswith("company:"):
            return line.split(":", 1)[1].strip()  # everything after the colon

    # 2. Subject line: "... from Acme Corp ..."
    subj = extract_subject(text) or ""
    m = re.search(r"(?i)\bfrom\s+(.+?)(?:\s*[-–—]|$)", subj)
    if m:
        return m.group(1).strip()

    # 3. scan lines (signature often at the bottom)
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    for ln in lines[-12:]:
        if re.search(r"\b(Corp|Inc\.?|LLC|Ltd\.?|GmbH|AG)\b", ln):
            return ln

    # 4. From: header → domain → base name
    m = re.search(r"(?im)^From:\s*([^\n]+)$", text)
    if m:
        frm = m.group(1)
        dom = re.search(r"@([A-Za-z0-9.-]+)", frm)
        if dom:
            base = dom.group(1).split(".")[0]
            return base.capitalize()

    return None


def extract_contact(text: str) -> Optional[str]:
    """
    Extract a contact email or 'From:' line.

    Prefers an explicit 'Email: some@addr' in the signature; otherwise use the 'From:' header.
    """
    m = re.search(r"(?i)Email:\s*([^\s]+@[^\s]+)", text)
    if m:
        return m.group(1).strip()

    m = re.search(r"(?im)^From:\s*(.+)$", text)
    return m.group(1).strip() if m else None


# ----------                     ----------
# ---------- 2. Prefill Endpoint ----------
# ----------                     ----------
@app.post("/v1/prefill", response_model=PrefillResponse)
def prefill(req: PrefillRequest):
    """
    Extract payment-related fields from raw email text and append them to data.csv.

    Returns:
        PrefillResponse: success flag, message, and the extracted data dictionary.
    """

    try:
        text = (req.email_text or "").strip()
        if not text:
            return PrefillResponse(success=False, message="email_text is required")

        amount, currency = extract_amount_and_currency(text)
        due_date = extract_due_date(text)
        description = extract_description(text)
        company = extract_company(text)
        contact = extract_contact(text)

        if not any([amount, currency, due_date, description, company, contact]):
            return PrefillResponse(
                success=False,
                message="could not extract any fields",
                data={"amount": None, "currency": None, "due_date": None,
                      "description": None, "company": None, "contact": None}
            )

        # write minimal CSV row (others empty for now)
        row = {
            "amount": amount or "",
            "currency": currency or "",
            "due_date": due_date or "",
            "description": description or "",
            "company": company or "",
            "contact": contact or "",
        }
        path = "data.csv"
        fields = ["amount", "currency", "due_date", "description", "company", "contact"]
        exists = os.path.exists(path)
        
        with open(path, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            if not exists:
                w.writeheader()
            w.writerow(row)

        return PrefillResponse(success=True, message="data extracted and written", data=row)
    except Exception as e:
        # Ensure we ALWAYS return JSON
        return PrefillResponse(success=False, message=f"something went wrong: {e}")
    



if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="127.0.0.1",
        port=8090,
        reload=True,
    )
