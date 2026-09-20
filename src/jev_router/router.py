import os
from typing import Iterable, Optional, Sequence

from .config import DEFAULT_MODELS
from .jev import JevJudge
from .models import Decision, ModelSpec
from .policy import decide
from .providers import run_openrouter


class Router:
    def __init__(self, models: Sequence[ModelSpec] = DEFAULT_MODELS, judge=None):
        self.models = tuple(models)
        self.judge = judge if judge is not None else JevJudge(os.getenv("TYPESAFE_API_KEY"))

    def route(self, prompt: str, *, context: Optional[dict] = None, force_model: Optional[str] = None,
              allowed: Optional[Iterable[str]] = None, min_model: Optional[str] = None,
              max_model: Optional[str] = None) -> Decision:
        judgment = None if force_model else self.judge.ask(prompt, self.models, context)
        return decide(self.models, judgment, force_model=force_model, allowed=allowed,
                      min_model=min_model, max_model=max_model)

    def run(self, prompt: str, *, api_key: Optional[str] = None, system: Optional[str] = None, **route_options):
        decision = self.route(prompt, **route_options)
        key = api_key or os.getenv("OPENROUTER_API_KEY")
        if not key:
            raise RuntimeError("OPENROUTER_API_KEY is required to execute the selected model")
        return decision, run_openrouter(decision.model, prompt, key, system)
