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
import time
import uuid
from typing import Optional

from fastapi import APIRouter, Header, HTTPException, Request
from fastapi.responses import JSONResponse

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
            }
        ],
    }


def _extract_query(message: dict) -> Optional[str]:
    for part in message.get("parts", []):
        text = part.get("text")
        if text and text.strip():
            return text.strip()
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

    app.include_router(router)
