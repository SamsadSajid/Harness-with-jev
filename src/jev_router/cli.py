import argparse
import json
from typing import List

from .models import Judgment
from .router import Router


class FixtureJudge:
    """Used only by demo/tests: makes offline routes repeatable, never production routing."""
    def __init__(self, judgments):
        self.judgments = iter(judgments)

    def ask(self, prompt, models, context=None):
        return next(self.judgments)


def _parser():
    parser = argparse.ArgumentParser(description="Route a request with TypeSafe Jev, then optionally execute it.")
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("route", "run"):
        p = sub.add_parser(name)
        p.add_argument("prompt")
        p.add_argument("--force-model")
        p.add_argument("--max-model")
        p.add_argument("--system")
    sub.add_parser("demo", help="run five deterministic offline routing examples")
    return parser


def _print_decision(decision):
    print(json.dumps({"selected": decision.model.key, "provider_model": decision.model.provider_model,
                      "reasons": list(decision.reasons), "jev_available": decision.jev_available}, indent=2))


def _demo() -> None:
    examples = [
        ("Turn these notes into a bullet list.", Judgment("fast", .94, {}, .05, .1, .05)),
        ("Implement the acceptance criteria in this issue.", Judgment("balanced", .88, {}, .63, .7, .1)),
        ("Diagnose a sporadic production auth failure.", Judgment("balanced", .91, {}, .95, 1.7, .25)),
        ("Choose an irreversible database migration plan.", Judgment("frontier", .89, {}, .88, 2.0, .85)),
        ("Reconcile the complete 1,000-page contract corpus.", Judgment("extended", .93, {}, .82, 1.1, .2)),
    ]
    router = Router(judge=FixtureJudge([item[1] for item in examples]))
    for prompt, _ in examples:
        decision = router.route(prompt)
        print("%s\n  -> %s (%s)\n" % (prompt, decision.model.key, decision.why))


def main(argv: List[str] = None) -> None:
    args = _parser().parse_args(argv)
    if args.command == "demo":
        _demo()
        return
    router = Router()
    if args.command == "route":
        _print_decision(router.route(args.prompt, force_model=args.force_model, max_model=args.max_model))
        return
    decision, answer = router.run(args.prompt, force_model=args.force_model, max_model=args.max_model, system=args.system)
    _print_decision(decision)
    print(answer)


if __name__ == "__main__":
    main()
