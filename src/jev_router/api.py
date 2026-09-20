"""Local HTTP API for routing and (when configured) executing model requests."""

import os
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from .models import Decision
from .router import Router


class RouteRequest(BaseModel):
    prompt: str = Field(..., min_length=1, max_length=32_000)
    context: Optional[Dict[str, Any]] = None
    force_model: Optional[str] = None
    allowed: Optional[List[str]] = None
    min_model: Optional[str] = None
    max_model: Optional[str] = None
    include_raw_jev: bool = False


class ExecuteRequest(RouteRequest):
    system: Optional[str] = Field(None, max_length=16_000)


def _decision_body(decision: Decision, *, include_raw_jev: bool = False) -> Dict[str, Any]:
    judgment = decision.judgment
    body = {
        "selected": decision.model.key,
        "provider_model": decision.model.provider_model,
        "reasons": list(decision.reasons),
        "jev_available": decision.jev_available,
        "judgment": None if judgment is None else {
            "choice": judgment.choice,
            "confidence": judgment.confidence,
            "probabilities": judgment.probabilities,
            "deep_reasoning": judgment.deep_reasoning,
            "blast_radius": judgment.blast_radius,
            "ambiguity": judgment.ambiguity,
            "input_tokens": judgment.input_tokens,
            "latency_ms": judgment.latency_ms,
        },
    }
    if include_raw_jev:
        body["raw_jev_response"] = None if judgment is None else judgment.raw_response
    return body


def _options(request: RouteRequest) -> Dict[str, Any]:
    return {
        "context": request.context,
        "force_model": request.force_model,
        "allowed": request.allowed,
        "min_model": request.min_model,
        "max_model": request.max_model,
    }


def create_app(router: Optional[Router] = None) -> FastAPI:
    app = FastAPI(
        title="Jev Routing Harness",
        version="0.1.0",
        description="Route requests with Jev, then optionally execute the selected OpenRouter model.",
    )
    app.state.router = router or Router()

    @app.get("/health")
    def health() -> Dict[str, str]:
        return {"status": "ok"}

    @app.post("/v1/route")
    def route(request: RouteRequest) -> Dict[str, Any]:
        try:
            decision = app.state.router.route(request.prompt, **_options(request))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        return _decision_body(decision, include_raw_jev=request.include_raw_jev)

    @app.post("/v1/execute")
    def execute(request: ExecuteRequest) -> Dict[str, Any]:
        try:
            decision, answer = app.state.router.run(request.prompt, system=request.system, **_options(request))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error)) from error
        except RuntimeError as error:
            # Do not expose exception chains, credentials, or upstream response bodies.
            raise HTTPException(status_code=502, detail=str(error)) from error
        response = _decision_body(decision, include_raw_jev=request.include_raw_jev)
        response["answer"] = answer
        return response

    return app


app = create_app()


def run() -> None:
    """Start a loopback-only server. Put a proxy/auth layer in front for remote use."""
    import uvicorn

    uvicorn.run(
        "jev_router.api:app",
        host=os.getenv("JEV_ROUTER_HOST", "127.0.0.1"),
        port=int(os.getenv("JEV_ROUTER_PORT", "8000")),
        reload=False,
    )
