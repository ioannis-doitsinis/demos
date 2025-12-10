# AI Chat & Email Prefill Server

## Overview

This project is a FastAPI-based AI server that:

- Proxies **chat completion** requests to either:
  - Free **OpenRouter** models (e.g. `vendor/name:free`), or
  - Paid **OpenAI** models (e.g. `gpt-4o-mini`, `gpt-4.1`, `o3-mini`)
- Extracts **payment-related fields** from raw email text and appends them to `data.csv`.

It is designed as a small, self-contained service to demonstrate LLM integration, provider routing, graceful fallbacks, and heuristic data extraction.

---

## Endpoints

### 1) Chat Completions

**Path:** `/v1/chat/completions`  
**Method:** `POST`

This endpoint is compatible with the OpenAI ChatCompletion API format.

#### Provider Routing Logic

- If `model` **ends with `:free`** → request is routed to **OpenRouter** using `OPENROUTER_API_KEY`.
- Otherwise → request is routed to **OpenAI** using `OPENAI_API_KEY`.
- If upstream fails:
  - The server probes a list of free OpenRouter models (`FREE_MODELS`) to suggest an available alternative.
  - If nothing works, it sends a **local fallback response** echoing the user’s last message.

#### Example Request

```json
POST /v1/chat/completions
{
  "model": "mistralai/mistral-7b-instruct:free",
  "messages": [
    { "role": "user", "content": "What can you do?" }
  ],
  "max_tokens": 128,
  "temperature": 0.7
}
```

#### Request Format Example

```Json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "model": "mistralai/mistral-7b-instruct:free",
  "choices": [
    {
      "index": 0,
      "message": { "role": "assistant", "content": "I can help you with many tasks..." },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 20,
    "total_tokens": 30
  }
}

```

### 2) Prefill Endpoint

**Path:** `/v1/prefill` 
**Method:** POST  
**Response model:** PrefillResponse

This endpoint does not call an LLM. It uses regular expressions and simple heuristics to extract payment-related fields from raw email text and write them to `data.csv`.

**Fields extracted:**
  - amount
  - currency
  - due_date
  - description
  - company
  - contact

If no fields can be extracted, the endpoint returns a failure response with success: false.

#### Request Format Example

```JSON
POST /v1/prefill
{
  "email_text": "Subject: Invoice\nAmount Due: €1,500.00 EUR\nDue Date: January 15, 2025\nCompany: Acme GmbH\nEmail: billing@acme.com"
}
```

#### Response Format Example

```json
{
  "success": true,
  "message": "data extracted and written",
  "data": {
    "amount": "1500.00",
    "currency": "EUR",
    "due_date": "January 15, 2025",
    "description": "Invoice",
    "company": "Acme GmbH",
    "contact": "billing@acme.com"
  }
}
```

This will append a row to data.csv with the same fields.

## Installation

  1. Clone the project

  ```bash
  git clone https://github.com/ioannis-doitsinis/demos.git
  cd demos
  ```

  2. Create a virtual environment

  ```bash
  python -m venv .venv
  source .venv/bin/activate   # Windows: .venv\Scripts\activate
  ```

  3. Install dependencies
  
  ```bash
  pip install -r requirements.txt
  ```

  4. Configure environment variables

    Create a .env file:

    ```bash
    OPENAI_API_KEY=your_openai_key_here   # optional, needed for paid OpenAI models
    OPENROUTER_API_KEY=your_openrouter_key_here     # optional, needed for :free models via OpenRouter
    ```

  You can use:

  - Only OPENROUTER_API_KEY → for free :free models.
  - Only OPENAI_API_KEY → for paid OpenAI models.
  - Both → for maximum flexibility.

#### Running the Server

  Start FastAPI with Uvicorn:

  ```bash
    uvicorn main:app --reload --host 127.0.0.1 --port 8090
  ```
  Access the API:

  - Base URL: http://127.0.0.1:8090
  - Interactive docs (Swagger UI): http://127.0.0.1:8090/docs


#### Running Tests
  ```bash
  pytest -q tests/test_chat_and_prefill_extended.py
  ```
