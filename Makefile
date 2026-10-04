.PHONY: help install run ask seed demo docs lint format typecheck test coverage check eval clean docker-build docker-up

PYTHON ?= python
APP_MODULE := app.main:app
HOST ?= 0.0.0.0
PORT ?= 8000
QUESTION ?= "What does an out-of-range lab test result mean in general?"
TOP_K ?= 5

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

install: ## Install the package with dev dependencies
	$(PYTHON) -m pip install --upgrade pip
	$(PYTHON) -m pip install -e ".[dev]"

run: ## Start the dev server with auto-reload
	$(PYTHON) -m uvicorn $(APP_MODULE) --host $(HOST) --port $(PORT) --reload

ask: ## Ask one question: make ask QUESTION="..."
	$(PYTHON) -m app.cli ask "$(QUESTION)" --top-k $(TOP_K)

seed: ## Index data/sample_docs into the vector store
	$(PYTHON) -m app.cli seed

search: ## Raw retrieval: make search QUERY="..."
	$(PYTHON) -m app.cli search "$(QUERY)" --top-k $(TOP_K)

demo: ## Scripted end-to-end walkthrough against a running server
	bash scripts/demo.sh

docs: ## Print the local API surface
	@curl -s localhost:$(PORT)/openapi.json | $(PYTHON) -m json.tool | head -60

lint: ## Ruff lint
	$(PYTHON) -m ruff check app tests evals scripts
	$(PYTHON) -m ruff format --check app tests evals scripts

format: ## Ruff autofix + format
	$(PYTHON) -m ruff check --fix app tests evals scripts
	$(PYTHON) -m ruff format app tests evals scripts

typecheck: ## mypy in strict mode
	$(PYTHON) -m mypy app

test: ## pytest with coverage
	$(PYTHON) -m pytest -m "not integration" --cov=app --cov-report=term-missing

coverage: ## Write an HTML coverage report
	$(PYTHON) -m pytest -m "not integration" --cov=app --cov-report=html --cov-report=term-missing
	@echo "HTML report: htmlcov/index.html"

check: lint typecheck test ## Everything CI runs

eval: ## Retrieval quality eval harness
	$(PYTHON) evals/run_evals.py --golden evals/golden_set.json

clean: ## Remove caches and build artifacts
	rm -rf .pytest_cache .mypy_cache .ruff_cache .coverage coverage.xml htmlcov dist build *.egg-info
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} +

docker-build: ## Build the container image
	docker build -t ai-engineer-app .

docker-up: ## Run the stack with docker compose
	docker compose up --build
