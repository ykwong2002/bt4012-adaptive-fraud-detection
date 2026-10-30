"""Thin layer over the Anthropic Message Batches API: build requests with structured output,
estimate cost, dry-run synchronously, submit in chunks, collect by custom_id, parse JSON safely.
Never sleeps unless wait=True (run that in the background)."""
from __future__ import annotations

import json
import time
from pathlib import Path

# USD per million tokens (first-party API, Oct 2026). Batch API halves both.
PRICES = {"claude-haiku-5-5": (0.10, 0.50), "claude-sonnet-5-5": (2.00, 10.00), "claude-opus-5-5": (4.00, 20.00)}


def make_request(custom_id: str, model: str, system: str, user: str, schema: dict,
                 max_tokens: int = 1024, effort: str = "low") -> dict:
    return {"custom_id": custom_id,
            "params": {"model": model, "max_tokens": max_tokens,
                       "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                       "messages": [{"role": "user", "content": user}],
                       "output_config": {"effort": effort, "format": {"type": "json_schema", "schema": schema}}}}


def estimate_cost(requests: list[dict], out_tokens: int = 150, batch: bool = True) -> dict:
    """Rough: 1 token ~ 4 chars; thinking at low effort adds ~out_tokens."""
    if not requests:
        return {"n": 0, "usd": 0.0}
    model = requests[0]["params"]["model"]
    pin, pout = PRICES.get(model, (2.0, 10.0))
    in_tok = sum(len(r["params"]["system"][0]["text"]) + len(r["params"]["messages"][0]["content"]) for r in requests) / 4
    out_tok = len(requests) * out_tokens * 2
    usd = (in_tok * pin + out_tok * pout) / 1e6 * (0.5 if batch else 1.0)
    return {"n": len(requests), "model": model, "input_tokens_est": int(in_tok), "output_tokens_est": int(out_tok),
            "usd": round(usd, 2), "batch": batch}


def parse_message(msg) -> tuple[dict | None, str]:
    """Returns (parsed_json or None, status) for a Message object."""
    if msg.stop_reason == "refusal":
        return None, "refusal"
    if msg.stop_reason == "max_tokens":
        return None, "max_tokens"
    text = next((b.text for b in msg.content if b.type == "text"), "")
    try:
        return json.loads(text), "ok"
    except json.JSONDecodeError:
        return None, "bad_json"


def dry_run(client, requests: list[dict]) -> list[tuple[str, dict | None, str, dict]]:
    """Synchronous calls for a handful of requests: validates the schema end to end."""
    out = []
    for r in requests:
        msg = client.messages.create(**r["params"])
        data, status = parse_message(msg)
        out.append((r["custom_id"], data, status, {"in": msg.usage.input_tokens, "out": msg.usage.output_tokens}))
    return out


def submit(client, requests: list[dict], state_path: Path, chunk: int = 10000) -> list[str]:
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    ids = []
    for i in range(0, len(requests), chunk):
        part = [Request(custom_id=r["custom_id"], params=MessageCreateParamsNonStreaming(**r["params"]))
                for r in requests[i:i + chunk]]
        b = client.messages.batches.create(requests=part)
        ids.append(b.id)
        print(f"submitted batch {b.id} with {len(part)} requests")
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state = json.loads(state_path.read_text()) if state_path.exists() else {"batches": []}
    state["batches"] += ids
    state_path.write_text(json.dumps(state, indent=2))
    return ids


def collect(client, state_path: Path, wait: bool = False, poll_s: int = 60):
    """Yields (custom_id, parsed_json or None, status) for every ended batch in the state file.
    Batches already collected are recorded in the state file and skipped."""
    state = json.loads(state_path.read_text())
    done = set(state.get("collected", []))
    pending = [b for b in state["batches"] if b not in done]
    while pending:
        still = []
        for bid in pending:
            b = client.messages.batches.retrieve(bid)
            if b.processing_status != "ended":
                print(f"{bid}: {b.processing_status} ({b.request_counts.processing} processing)")
                still.append(bid)
                continue
            print(f"{bid}: ended succeeded={b.request_counts.succeeded} errored={b.request_counts.errored} "
                  f"expired={b.request_counts.expired} canceled={b.request_counts.canceled}")
            for res in client.messages.batches.results(bid):
                if res.result.type == "succeeded":
                    data, status = parse_message(res.result.message)
                    yield res.custom_id, data, status
                else:
                    yield res.custom_id, None, res.result.type
            done.add(bid)
            state["collected"] = sorted(done)
            state_path.write_text(json.dumps(state, indent=2))
        pending = still
        if pending and not wait:
            break
        if pending:
            time.sleep(poll_s)
