PYTHON ?= python3.11

.PHONY: install fmt lint test test-integration test-chaos run loadtest benchmark demo

install:
	$(PYTHON) -m pip install -r requirements.txt

fmt:
	ruff format .

lint:
	ruff check .

test:
	$(PYTHON) -m pytest -q -m "not integration"

test-integration:
	RUN_INTEGRATION=1 $(PYTHON) -m pytest -q -m integration

test-chaos:
	RUN_INTEGRATION=1 $(PYTHON) -m pytest -q -m chaos

run:
	docker compose up --build

loadtest:
	$(PYTHON) scripts/load_test.py --url http://localhost:8000/v1/generate --requests 200 --concurrency 20

benchmark:
	$(PYTHON) scripts/benchmark.py --url http://localhost:8000/v1/generate --requests 500 --concurrency 50 --stream --out artifacts/benchmark/latest.json

demo:
	@set -e; \
	docker compose up --build -d; \
	trap 'docker compose down -v' EXIT; \
	$(PYTHON) scripts/demo_stream.py --wait-seconds 90
