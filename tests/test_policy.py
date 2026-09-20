import unittest

from jev_router.config import DEFAULT_MODELS
from jev_router.models import Judgment
from jev_router.policy import decide


def judgment(choice="fast", **values):
    base = dict(confidence=.95, probabilities={}, deep_reasoning=.05, blast_radius=.1, ambiguity=.05)
    base.update(values)
    return Judgment(choice, **base)


class RoutingPolicyTests(unittest.TestCase):
    def test_missing_jev_fails_to_balanced_not_fast(self):
        self.assertEqual(decide(DEFAULT_MODELS, None).model.key, "balanced")

    def test_low_confidence_cannot_select_fast(self):
        decision = decide(DEFAULT_MODELS, judgment(confidence=.20))
        self.assertEqual(decision.model.key, "balanced")
        self.assertIn("low-confidence-floor:balanced", decision.reasons)

    def test_complex_reasoning_escalates_two_steps_with_risk(self):
        decision = decide(DEFAULT_MODELS, judgment(deep_reasoning=.96, blast_radius=1.8))
        self.assertEqual(decision.model.key, "frontier")
        self.assertIn("hard-reasoning:+1", decision.reasons)

    def test_ambiguous_high_risk_work_escalates_to_frontier(self):
        decision = decide(DEFAULT_MODELS, judgment("balanced", ambiguity=.9, blast_radius=2.0))
        self.assertEqual(decision.model.key, "frontier")

    def test_operator_cap_wins_over_jev(self):
        decision = decide(DEFAULT_MODELS, judgment("extended"), max_model="frontier")
        self.assertEqual(decision.model.key, "frontier")

    def test_risk_does_not_demote_an_extended_choice(self):
        decision = decide(DEFAULT_MODELS, judgment("extended", blast_radius=2.0))
        self.assertEqual(decision.model.key, "extended")

    def test_force_model_skips_judgment_and_is_clamped_by_cap(self):
        decision = decide(DEFAULT_MODELS, None, force_model="extended", max_model="frontier")
        self.assertEqual(decision.model.key, "frontier")
        self.assertEqual(decision.reasons, ("forced:extended", "clamped:frontier"))
