"""
Unlike the stock version (utils/ollama.py in stock_agent2), there is NO
silent substitution of defaults on invalid JSON here. The spec requires the exact
opposite: one retry of the request, and if the response is still invalid —
an honest error mark, without plausible placeholder numbers (requirement item 4).
"""
import time
import json

import requests

from config import OLLAMA_URL, MAX_LINE_LOG, THINKING_MODEL_PREFIXES


def _raw_call(prompt: str, model: str, json_schema: dict = None, temperature: float = None) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    if json_schema is not None:
        payload["format"] = json_schema
    if temperature is not None:
        payload["options"] = {"temperature": temperature, "top_p": 0.85}
    # See THINKING_MODEL_PREFIXES in config.py: without this, models like
    # qwen3.5 with json_schema return an empty "response" instead of JSON.
    if model.startswith(THINKING_MODEL_PREFIXES):
        payload["think"] = False
    start = time.time()
    resp = requests.post(OLLAMA_URL, json=payload, timeout=1820)
    resp.raise_for_status()
    elapsed = time.time() - start
    raw = resp.json()["response"].strip()
    ts = f"{elapsed:.1f} sec" if elapsed < 60 else f"{elapsed / 60:.1f} min"
    print(f"⏱ {model} responded in {ts}")
    print(f"   🔍 RAW RESPONSE ({len(raw)} chars) before parsing:")
    for line in raw.split('\n'):
        print(f"   │ {line[:MAX_LINE_LOG]}")
    return raw


def _validate(data, schema: dict) -> tuple:
    """Strict check: all required fields are present, enum fields contain
    ONLY the declared values. No fuzzy matching and no substitution of
    defaults — that is the whole point of an honest error instead of guessing."""
    if not isinstance(data, dict):
        return False, "the response is not a JSON object"

    required = schema.get("required", [])
    properties = schema.get("properties", {})

    missing = [f for f in required if f not in data]
    if missing:
        return False, f"required fields missing: {missing}"

    bad_enums = []
    for field, prop in properties.items():
        if "enum" in prop and field in data and data[field] not in prop["enum"]:
            bad_enums.append((field, data[field], prop["enum"]))
    if bad_enums:
        details = "; ".join(f"{f}={v!r} not in {allowed}" for f, v, allowed in bad_enums)
        return False, f"invalid enum values: {details}"

    return True, ""


def call_ollama_json(prompt: str, model: str, json_schema: dict = None, temperature: float = 0,
                      max_retries: int = 1) -> dict:
    """
    Returns the model's parsed JSON response. On an invalid response makes
    ONE retry; if that is invalid too — returns
    {"_error": True, "_error_message": "...", "_raw": "..."} instead of
    substituting default values. The calling code (agents/*.py) must
    check data.get("_error") and show it honestly in the report, rather than
    treat a partially filled response as a real analysis.
    """
    last_raw = ""
    last_reason = ""
    for attempt in range(max_retries + 1):
        raw = _raw_call(prompt, model, json_schema, temperature)
        last_raw = raw

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as e:
            last_reason = f"invalid JSON: {e}"
            if attempt < max_retries:
                print(f"   ⚠️ {last_reason} — retrying the request ({attempt + 1}/{max_retries})")
                continue
            break

        if json_schema:
            ok, reason = _validate(result, json_schema)
            if not ok:
                last_reason = reason
                if attempt < max_retries:
                    print(f"   ⚠️ Invalid response ({reason}) — retrying the request ({attempt + 1}/{max_retries})")
                    continue
                break

        return result

    print(f"   ❌ The model gave no valid response after {max_retries + 1} attempts: {last_reason}")
    return {"_error": True, "_error_message": last_reason, "_raw": last_raw[:1000]}
