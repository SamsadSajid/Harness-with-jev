"""The one Jev network call. It fails closed to the policy's safe fallback."""

import json
import time
import urllib.request
from typing import Any, Optional, Sequence

from .config import JEV_MODEL
from .models import Judgment, ModelSpec


class JevJudge:
    def __init__(self, api_key: Optional[str] = None, model: str = JEV_MODEL):
        self.api_key = api_key
        self.model = model

    def ask(self, prompt: str, models: Sequence[ModelSpec], context: Optional[dict] = None) -> Optional[Judgment]:
        if not self.api_key:
            return None
        try:
            from typesafe_sdk import Choice, ChoiceAnswer, Noul, NoulAnswer, Score, ScoreAnswer, TypeSafeClient
        except ImportError:
            # The official SDK currently needs Python 3.10+. Keep a tiny,
            # contract-equivalent standard-library transport so the harness can
            # still be run from constrained Python environments.
            return self._ask_http(prompt, models, context)
        try:
            criteria = {model.key: {"what": model.description} for model in models}
            questions = {
                "model": Choice(
                    instructions=("Choose the cheapest candidate that can completely and correctly finish the request "
                                  "in one attempt. Judge the work required, not response length."),
                    criteria=criteria,
                ),
                "deep_reasoning": Noul(instructions="Does this require multi-step inference, derivation, or trade-off reasoning?"),
                "blast_radius": Score(
                    instructions="How costly is a confidently wrong answer if acted on without review?",
                    criteria=["Trivially recoverable", "Causes meaningful rework", "Risks money, security, data loss, legal exposure, or harm"],
                ),
                "ambiguity": Noul(instructions=("Is the request materially underspecified, so reasonable interpretations "
                                                   "would lead to different work? Referred files and data are available.")),
            }
            state: dict[str, Any] = {"request": prompt}
            if context:
                state["context"] = context
            started = time.monotonic()
            client = TypeSafeClient(api_key=self.api_key, timeout=1.5)
            try:
                response = client.system_one(state, questions, model=self.model)
            finally:
                client.close()
            choice = response.answers["model"]
            reasoning = response.answers["deep_reasoning"]
            blast = response.answers["blast_radius"]
            ambiguity = response.answers["ambiguity"]
            if not (isinstance(choice, ChoiceAnswer) and isinstance(reasoning, NoulAnswer)
                    and isinstance(blast, ScoreAnswer) and isinstance(ambiguity, NoulAnswer)):
                return None
            return Judgment(choice.choice, choice.confidence, dict(choice.probabilities), reasoning.noul,
                            blast.score, ambiguity.noul, response.usage.input_tokens or 0,
                            int((time.monotonic() - started) * 1000))
        except Exception:
            # Prompt text is deliberately not logged: this sits in front of user traffic.
            return None

    def _ask_http(self, prompt: str, models: Sequence[ModelSpec], context: Optional[dict]) -> Optional[Judgment]:
        """Call the documented System One HTTP endpoint when the SDK is unavailable."""
        criteria = {model.key: model.description for model in models}
        state: dict[str, Any] = {"request": prompt}
        if context:
            state["context"] = context
        payload = {
            "model": self.model,
            "state": state,
            "questions": {
                "model": {
                    "type": "choice",
                    "instructions": ("Choose the cheapest candidate that can completely and correctly finish the request "
                                     "in one attempt. Judge the work required, not response length."),
                    "criteria": criteria,
                },
                "deep_reasoning": {
                    "type": "noul",
                    "instructions": "Does this require multi-step inference, derivation, or trade-off reasoning?",
                },
                "blast_radius": {
                    "type": "score",
                    "instructions": "How costly is a confidently wrong answer if acted on without review?",
                    "criteria": ["Trivially recoverable", "Causes meaningful rework",
                                 "Risks money, security, data loss, legal exposure, or harm"],
                },
                "ambiguity": {
                    "type": "noul",
                    "instructions": ("Is the request materially underspecified, so reasonable interpretations "
                                     "would lead to different work? Referred files and data are available."),
                },
            },
        }
        request = urllib.request.Request(
            "https://api.typesafe.ai/v1/systemone",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Authorization": "Bearer " + self.api_key, "Content-Type": "application/json"},
            method="POST",
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=3) as response:
                body = json.load(response)
            answers = body["answers"]
            choice = answers["model"]
            reasoning = answers["deep_reasoning"]
            blast = answers["blast_radius"]
            ambiguity = answers["ambiguity"]
            usage = body.get("usage", {})
            return Judgment(
                choice["choice"], float(choice["confidence"]), dict(choice["probabilities"]),
                float(reasoning["noul"]), float(blast["score"]), float(ambiguity["noul"]),
                int(usage.get("input_tokens") or 0), int((time.monotonic() - started) * 1000),
            )
        except Exception:
            return None
