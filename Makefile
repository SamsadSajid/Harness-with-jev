.PHONY: serve test

# Start the loopback-only HTTP API with credentials from the ignored .env file.
serve:
	@set -a; . ./.env; set +a; PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m uvicorn jev_router.api:app --host 127.0.0.1 --port 8000

test:
	@PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python3 -m unittest discover -s tests -v
