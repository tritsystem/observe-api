"""
A2A (Agent2Agent) protocol adapter -- makes OBSERVE discoverable and
callable by real A2A clients, not just a JSON manifest pointing at an
incompatible REST shape. Implements the actual wire protocol at
`/a2a/v1/message:send` (schema verified against the normative
`a2a.proto` at github.com/a2aproject/A2A, not guessed at) so a client
that discovers this agent via `/.well-known/agent-card.json` can
genuinely transact with it, not just read about it.

v1 scope, disclosed: extracts the search query from the first `text`
Part of the incoming Message only -- no support yet for a structured
`data` Part carrying {query, k, repo} explicitly, and no streaming
(`message:stream`), tasks list/cancel, or push notifications. A search
is synchronous and fast enough that async task tracking adds nothing
real yet; if that changes (e.g. a slow op gets added), GetTask/ListTasks
become worth building. Auth reuses OBSERVE's existing API key system
(Bearer obs_...) -- an A2A caller needs a real OBSERVE key same as any
other caller, credited the same CREDITS_PER_SEARCH per call.
"""
import os
import sys
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from spiking_causal_relevance import fuse_causal_and_relevance

# Same real cross-platform fix as spiking_causal_relevance.py -- a bare
# Windows path silently fails to resolve under WSL2 (confirmed, not
# assumed, by the spiking_evidence.py import test).
_METHODLM_CANDIDATES = [r"C:\Users\gbran\llama_demo", "/mnt/c/Users/gbran/llama_demo"]
_methodlm_root = next((p for p in _METHODLM_CANDIDATES if os.path.isdir(p)), _METHODLM_CANDIDATES[0])
sys.path.insert(0, _methodlm_root)

router = APIRouter()

PROTOCOL_VERSION = "0.3"


def build_agent_card(base_url: str) -> dict:
    return {
        "name": "OBSERVE Search API",
        "description": (
            "Pay-per-query semantic code search over a curated set of popular "
            "open source repos (React, Django, NumPy, FastAPI, Tokio, and more). "
            "Use when you can only describe what you're looking for, not name "
            "the exact identifier -- vocabulary-mismatch and concept-only "
            "queries. Costs API credits per call; requires an OBSERVE API key "
            f"(get one: POST {base_url}/v1/signup)."
        ),
        "supportedInterfaces": [
            {
                "url": f"{base_url}/a2a/v1",
                "protocolBinding": "HTTP+JSON",
                "protocolVersion": PROTOCOL_VERSION,
            }
        ],
        "provider": {
            "url": "https://github.com/gbranaa4-hue/observe-api",
            "organization": "OBSERVE",
        },
        "version": "1.0.0",
        "documentationUrl": "https://github.com/gbranaa4-hue/observe-api#readme",
        "capabilities": {
            "streaming": False,
            "pushNotifications": False,
            "extendedAgentCard": False,
        },
        "securitySchemes": {
            "observeApiKey": {
                "httpAuthSecurityScheme": {
                    "description": "OBSERVE API key, obtained from POST /v1/signup",
                    "scheme": "Bearer",
                }
            }
        },
        "securityRequirements": [{"schemes": {"observeApiKey": {"list": []}}}],
        "defaultInputModes": ["text/plain"],
        "defaultOutputModes": ["application/json"],
        "skills": [
            {
                "id": "semantic-code-search",
                "name": "Semantic code search",
                "description": (
                    "Search a curated set of popular open source repos by "
                    "describing behavior rather than naming an exact "
                    "identifier. Not a replacement for grep/exact lookups -- "
                    "see the tool description for when each wins."
                ),
                "tags": ["code-search", "semantic-search", "developer-tools"],
                "examples": [
                    "where does this handle retrying a failed upload",
                    "how does this library convert timezones",
                ],
                "inputModes": ["text/plain"],
                "outputModes": ["application/json"],
            },
            {
                "id": "causal-driver-check",
                "name": "Causal driver check",
                "description": (
                    "Real causal-reasoning check for one named candidate driver of a target "
                    "in a CSV dataset: runs MethodLM's real backdoor-adjustment test (Cinelli-"
                    "Hazlett robustness value) AND OBSERVE's own semantic search relevance, "
                    "fusing both into one verdict via a real compiled Spikeling LIF neuron -- "
                    "not a bare correlation, and not either signal alone. A strong robustness "
                    "value can clear the bar by itself; a moderate one needs real search "
                    "corroboration too. Answers 'is this actually a real driver' honestly, "
                    "including refusing when neither signal is strong enough."
                ),
                "tags": ["causal-reasoning", "methodlm", "spiking-neural-network", "developer-tools"],
                "examples": [
                    "is bmi a real driver of progression in this diabetes dataset, controlling for age and sex",
                ],
                "inputModes": ["application/json"],
                "outputModes": ["application/json"],
            },
        ],
    }


def _extract_query(message: dict) -> Optional[str]:
    for part in message.get("parts", []):
        text = part.get("text")
        if text and text.strip():
            return text.strip()
    return None


def _extract_data_part(message: dict) -> Optional[dict]:
    """First real structured `data` Part support in this adapter -- v1
    (message:send) only ever read `text` Parts (disclosed limitation in
    this module's own docstring). The causal-driver-check skill genuinely
    needs structured fields (csv_path/target/candidate/confounders), so
    this is a real, new capability, not a workaround."""
    for part in message.get("parts", []):
        data = part.get("data")
        if isinstance(data, dict):
            return data
    return None


def _failed_task(task_id: str, context_id: str, reason: str) -> dict:
    now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    return {
        "task": {
            "id": task_id,
            "contextId": context_id,
            "status": {
                "state": "TASK_STATE_FAILED",
                "message": {
                    "messageId": str(uuid.uuid4()),
                    "role": "ROLE_AGENT",
                    "parts": [{"text": reason}],
                },
                "timestamp": now,
            },
            "artifacts": [],
        }
    }


def register_a2a_routes(app, engine, db, rate_limit, require_key_fn, credits_per_search: int):
    """Wired from server.py with its already-constructed dependencies (engine,
    db, rate_limit, the existing _require_key helper) rather than
    re-importing/re-instantiating any of them -- one engine, one set of
    credit rules, whether a caller comes in via /v1/search or /a2a/v1.
    No repo_registry param -- v1 doesn't support the repo filter yet (see
    module docstring), so there's nothing to pass it for."""

    @app.get("/.well-known/agent-card.json", include_in_schema=False)
    def agent_card(request: Request):
        base_url = str(request.base_url).rstrip("/")
        return JSONResponse(build_agent_card(base_url))

    @router.post("/a2a/v1/message:send")
    async def a2a_send_message(request: Request, authorization: Optional[str] = Header(None)):
        raw_key = require_key_fn(authorization)

        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded -- slow down and retry shortly")

        body = await request.json()
        message = body.get("message") or {}
        context_id = message.get("contextId") or str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        query = _extract_query(message)
        if not query:
            return JSONResponse(_failed_task(
                task_id, context_id,
                "No text Part found in message.parts -- v1 only supports a "
                "plain-text search query, not structured data parts yet.",
            ))

        if not db.deduct_credit(raw_key, credits_per_search):
            return JSONResponse(_failed_task(
                task_id, context_id,
                "insufficient credits -- purchase more via POST /v1/signup's checkout_url",
            ), status_code=402)

        try:
            raw_results = engine.search(query, k=10)
        except Exception:
            db.deduct_credit(raw_key, -credits_per_search)  # refund, same as /v1/search
            return JSONResponse(_failed_task(task_id, context_id, "search failed -- credit refunded, please retry"))

        db.log_usage(raw_key, query, None, len(raw_results))

        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = db.get_key_record(raw_key)
        return JSONResponse({
            "task": {
                "id": task_id,
                "contextId": context_id,
                "status": {
                    "state": "TASK_STATE_COMPLETED",
                    "timestamp": now,
                },
                "artifacts": [
                    {
                        "artifactId": str(uuid.uuid4()),
                        "name": "search_results",
                        "description": f"credits_remaining: {record['credits']}",
                        "parts": [
                            {
                                "data": {
                                    "results": [
                                        {"score": r["score"], "path": r["path"], "preview": r["preview"]}
                                        for r in raw_results
                                    ],
                                },
                                "mediaType": "application/json",
                            }
                        ],
                    }
                ],
            }
        })

    @router.post("/a2a/v1/causal-check:send")
    async def a2a_causal_check(request: Request, authorization: Optional[str] = Header(None)):
        """Real causal-driver-check skill: MethodLM's real ADJUST (backdoor
        adjustment + Cinelli-Hazlett robustness value) fused with OBSERVE's
        own real search relevance via a compiled Spikeling LIF neuron.
        Costs the same credits_per_search as a normal search -- real compute
        either way, same pricing model, not a separate tier."""
        raw_key = require_key_fn(authorization)
        if not rate_limit.allow(raw_key):
            raise HTTPException(status_code=429, detail="rate limit exceeded -- slow down and retry shortly")

        body = await request.json()
        message = body.get("message") or {}
        context_id = message.get("contextId") or str(uuid.uuid4())
        task_id = str(uuid.uuid4())

        data = _extract_data_part(message)
        if not data:
            return JSONResponse(_failed_task(
                task_id, context_id,
                "No structured data Part found -- this skill needs "
                "{csv_path, target, candidate, confounders: [...]} as a data Part.",
            ))

        csv_path = data.get("csv_path")
        target = data.get("target")
        candidate = data.get("candidate")
        confounders = data.get("confounders") or []
        if not csv_path or not target or not candidate:
            return JSONResponse(_failed_task(
                task_id, context_id,
                "data Part must include 'csv_path', 'target', and 'candidate'.",
            ))
        if not os.path.isfile(csv_path):
            return JSONResponse(_failed_task(task_id, context_id, f"file not found: {csv_path}"))

        if not db.deduct_credit(raw_key, credits_per_search):
            return JSONResponse(_failed_task(
                task_id, context_id,
                "insufficient credits -- purchase more via POST /v1/signup's checkout_url",
            ), status_code=402)

        try:
            from methodlm import load_csv, make_tools
            csv_data = load_csv(csv_path, target)
            if candidate not in csv_data:
                db.deduct_credit(raw_key, -credits_per_search)  # refund -- real error, not a real answer
                return JSONResponse(_failed_task(
                    task_id, context_id,
                    f"'{candidate}' not among numeric columns {list(csv_data)}",
                ))
            _corr, _run, _strat, adjust = make_tools(csv_data, target, interventional=False)
            adjust_result = adjust(candidate, confounders)

            # adjust() returns a real formatted message string ending in the
            # robustness value -- parse the real RV out of it rather than
            # re-deriving it, so this can never silently drift from what
            # MethodLM's own tool actually computed.
            import re as _re
            rv_match = _re.search(r"RV\s*=\s*([\d.]+)", adjust_result)
            causal_rv = float(rv_match.group(1)) if rv_match else 0.0

            search_hits = engine.search(candidate, k=1) if engine.ready else []
            search_relevance = search_hits[0]["score"] if search_hits else 0.0

            fusion = fuse_causal_and_relevance(search_relevance, causal_rv)
        except Exception as e:
            db.deduct_credit(raw_key, -credits_per_search)
            return JSONResponse(_failed_task(task_id, context_id, f"causal check failed -- credit refunded: {e}"))

        db.log_usage(raw_key, f"causal-check:{candidate}", None, 1)
        now = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        record = db.get_key_record(raw_key)
        return JSONResponse({
            "task": {
                "id": task_id,
                "contextId": context_id,
                "status": {"state": "TASK_STATE_COMPLETED", "timestamp": now},
                "artifacts": [{
                    "artifactId": str(uuid.uuid4()),
                    "name": "causal_driver_check",
                    "description": f"credits_remaining: {record['credits']}",
                    "parts": [{
                        "data": {
                            "candidate": candidate,
                            "target": target,
                            "confounders": confounders,
                            "methodlm_adjust_result": adjust_result,
                            "causal_robustness_value": causal_rv,
                            "search_relevance": round(search_relevance, 3),
                            "spiking_fusion": fusion,
                            "verdict": "real_driver" if fusion["fired"] else "not_established",
                        },
                        "mediaType": "application/json",
                    }],
                }],
            }
        })

    app.include_router(router)
