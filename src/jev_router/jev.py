"""The one Jev network call. It fails closed to the policy's safe fallback."""

import time
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
