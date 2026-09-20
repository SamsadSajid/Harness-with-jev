import unittest

from fastapi.testclient import TestClient

from jev_router.api import create_app
from jev_router.config import DEFAULT_MODELS
from jev_router.models import Decision, Judgment


class StubRouter:
    def __init__(self):
        self.decision = Decision(
            DEFAULT_MODELS[1],
            ("jev:balanced@0.91",),
            Judgment("balanced", .91, {"balanced": .91}, .6, .7, .1, 42, 123,
                      {"model": "jev-1.13.0", "answers": {"model": {"type": "choice"}}}),
        )

    def route(self, prompt, **options):
        self.last_route = (prompt, options)
        return self.decision

    def run(self, prompt, **options):
        self.last_run = (prompt, options)
        return self.decision, "HARNESS_OK"


class ApiTests(unittest.TestCase):
    def setUp(self):
        self.router = StubRouter()
        self.client = TestClient(create_app(self.router))

    def test_health(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_route_returns_explainable_decision(self):
        response = self.client.post("/v1/route", json={"prompt": "Implement this issue", "max_model": "frontier",
                                                        "include_raw_jev": True})
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["selected"], "balanced")
        self.assertEqual(body["provider_model"], "anthropic/claude-sonnet-4")
        self.assertEqual(body["judgment"]["input_tokens"], 42)
        self.assertEqual(body["raw_jev_response"]["model"], "jev-1.13.0")
        self.assertEqual(self.router.last_route[1]["max_model"], "frontier")

    def test_execute_returns_answer_and_route(self):
        response = self.client.post("/v1/execute", json={"prompt": "Return HARNESS_OK"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["answer"], "HARNESS_OK")

    def test_empty_prompt_is_rejected(self):
        response = self.client.post("/v1/route", json={"prompt": ""})
        self.assertEqual(response.status_code, 422)
