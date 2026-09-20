# Jev model-routing harness

An explainable harness that uses TypeSafe **Jev** to decide which downstream model should answer a request, then uses that choice to call an OpenAI-compatible provider through OpenRouter.

Jev belongs in the control plane—not the generation plane. It returns typed, calibrated decisions cheaply and quickly; the selected LLM still does the writing, coding, and tool work. The harness keeps the final routing policy in ordinary Python, which makes it reviewable, replayable, and testable without paying for inference.

## What it decides

One Jev request produces four independent signals:

| Signal | Harness use |
| --- | --- |
| Cheapest capable candidate (`Choice`) | Initial model selection |
| Multi-step reasoning (`Noul`) | Floors at `balanced`; near certainty escalates |
| Blast radius (`Score`) | Escalates work involving money, security, data loss, or harm |
| Material ambiguity (`Noul`) | Escalates rather than giving a confident but wrong interpretation |

The default pool is `fast`, `balanced`, `frontier`, and `extended`. It is configured in `src/jev_router/config.py`; replace the OpenRouter model IDs with the models enabled for your account before production use.

Safety rules are deterministic:

- No Jev key, timeout, malformed response, or outage routes to `balanced`, never the cheapest model.
- Low confidence can only raise the selected capability.
- Operator caps/floors always win; request text cannot override them.
- `--force-model` is explicit control data and is still constrained by `--max-model`.

## Setup

```bash
python3.10 -m venv .venv
. .venv/bin/activate
pip install -e '.[dev]'
cp .env.example .env
```

Set `TYPESAFE_API_KEY` for live Jev decisions. Set `OPENROUTER_API_KEY` only when you want the harness to call the downstream model. `route` works without either key, but correctly reports a conservative fallback instead of pretending to use Jev.

## Use

```bash
# Inspect a live Jev route (no downstream generation)
jev-router route "Explain this stack trace and propose the smallest safe fix."

# Enforce a spend/capability ceiling
jev-router route "Summarize this release note" --max-model balanced

# Route, then execute the chosen OpenRouter model
jev-router run "Design a reversible database migration for tenant isolation."

# Five repeatable, offline examples (fixture judgments; no keys or network)
jev-router demo
```

The `run` command prints the route explanation before the selected model's answer. Do not put secrets, browser content, or untrusted tool output into a concatenated router prompt. Pass them as named `context` fields and keep authorization, path allowlists, and spend caps in code.

## Test

```bash
python3 -m pytest -q
python3 -m jev_router.cli demo
```

The tests cover conservative fallback, low-confidence escalation, deep-reasoning/risk escalation, ambiguity, operator caps, and explicit force routes. The demo covers five representative user inputs without requiring API keys.

## Why Jev here

Jev is a decision model rather than a text model: `Choice`, `Score`, and `Noul` outputs make it a strong fit for bounded control decisions such as model routing. It should not be asked to generate the final response. The policy intentionally treats Jev as an advisor whose structured signals are applied through deterministic controls and logged as an explanation.

Reference reading: [TypeSafe Jev router example](https://github.com/its-panzer/jev-model-router), [Jev routing experiment and ablation](https://github.com/TokenTrim/jev-routing-experiment), and [agent-harness guidance](https://learnjev.com/tutorials/agent-harness).
