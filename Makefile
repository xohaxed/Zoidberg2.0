.PHONY: help setup-dev dev-up dev-down test lint format scan clean

# Default target
help:
	@echo "CloudOpt AI - Development Commands"
	@echo ""
	@echo "Setup & Development:"
	@echo "  make setup-dev     Install development dependencies and pre-commit hooks"
	@echo "  make dev-up        Start all services in development mode"
	@echo "  make dev-down      Stop all development services"
	@echo "  make dev-logs      Show logs from all services"
	@echo ""
	@echo "Testing:"
	@echo "  make test          Run all tests"
	@echo "  make test-api      Run API service tests"
	@echo "  make test-ai       Run AI engine tests"
	@echo "  make test-data     Run data service tests"
	@echo "  make test-web      Run web dashboard tests"
	@echo "  make test-infra    Run infrastructure tests"
	@echo "  make test-coverage Run tests with coverage report"
	@echo ""
	@echo "Code Quality:"
	@echo "  make lint          Run all linters"
	@echo "  make format        Format all code"
	@echo "  make typecheck     Run type checkers"
	@echo ""
	@echo "Security:"
	@echo "  make scan          Run all security scans"
	@echo "  make scan-secrets  Scan for committed secrets"
	@echo "  make scan-deps     Scan dependencies for vulnerabilities"
	@echo "  make scan-iac      Scan infrastructure code"
	@echo "  make scan-images   Scan container images"
	@echo ""
	@echo "Database:"
	@echo "  make db-migrate    Run database migrations"
	@echo "  make db-seed       Seed database with sample data"
	@echo "  make db-reset      Reset database (WARNING: deletes data)"
	@echo ""
	@echo "Utilities:"
	@echo "  make clean         Clean build artifacts and caches"
	@echo "  make docs          Generate documentation"

# Setup & Development
setup-dev:
	@echo "Installing development dependencies..."
	pip install pre-commit poetry
	npm install -g pnpm
	pre-commit install
	cd services/api && poetry install
	cd services/ai-engine && poetry install
	cd web/dashboard && pnpm install
	@echo "✓ Development environment ready!"

dev-up:
	@echo "Starting development environment..."
	docker-compose -f dev/docker-compose.yml up -d
	@echo "✓ Services started!"
	@echo ""
	@echo "Access points:"
	@echo "  API:       http://localhost:8000/docs"
	@echo "  Dashboard: http://localhost:3000"
	@echo "  MLflow:    http://localhost:5000"
	@echo "  Grafana:   http://localhost:3001"

dev-down:
	docker-compose -f dev/docker-compose.yml down

dev-logs:
	docker-compose -f dev/docker-compose.yml logs -f

# Testing
test: test-api test-ai test-data test-web

test-api:
	@echo "Running API tests..."
	cd services/api && poetry run pytest -v

test-ai:
	@echo "Running AI engine tests..."
	cd services/ai-engine && poetry run pytest -v

test-data:
	@echo "Running data service tests..."
	cd services/data && poetry run pytest -v

test-web:
	@echo "Running web tests..."
	cd web/dashboard && pnpm test

test-infra:
	@echo "Running infrastructure tests..."
	cd infra/terraform-modules && go test -v ./...

test-coverage:
	@echo "Running tests with coverage..."
	cd services/api && poetry run pytest --cov=app --cov-report=html --cov-report=term
	cd services/ai-engine && poetry run pytest --cov=cloudopt_ai --cov-report=html --cov-report=term

# Code Quality
lint: lint-python lint-js lint-iac

lint-python:
	@echo "Linting Python code..."
	cd services/api && poetry run ruff check .
	cd services/ai-engine && poetry run ruff check .

lint-js:
	@echo "Linting JavaScript/TypeScript..."
	cd web/dashboard && pnpm lint

lint-iac:
	@echo "Linting Terraform..."
	cd infra/terraform-modules && terraform fmt -check -recursive

format: format-python format-js format-iac

format-python:
	@echo "Formatting Python code..."
	cd services/api && poetry run black . && poetry run isort .
	cd services/ai-engine && poetry run black . && poetry run isort .

format-js:
	@echo "Formatting JavaScript/TypeScript..."
	cd web/dashboard && pnpm format

format-iac:
	@echo "Formatting Terraform..."
	cd infra/terraform-modules && terraform fmt -recursive

typecheck:
	@echo "Running type checkers..."
	cd services/api && poetry run mypy app/
	cd services/ai-engine && poetry run mypy cloudopt_ai/
	cd web/dashboard && pnpm typecheck

# Security
scan: scan-secrets scan-deps scan-iac scan-images

scan-secrets:
	@echo "Scanning for secrets..."
	gitleaks detect --source . --verbose

scan-deps:
	@echo "Scanning dependencies..."
	cd services/api && poetry run pip-audit
	cd web/dashboard && pnpm audit

scan-iac:
	@echo "Scanning infrastructure code..."
	checkov -d infra/terraform-modules --quiet
	tfsec infra/terraform-modules

scan-images:
	@echo "Scanning container images..."
	trivy image cloudopt-ai/api:latest
	trivy image cloudopt-ai/ai-engine:latest

# Database
db-migrate:
	@echo "Running database migrations..."
	cd services/data && poetry run alembic upgrade head

db-seed:
	@echo "Seeding database..."
	cd services/data && poetry run python scripts/seed_db.py

db-reset:
	@echo "⚠️  WARNING: This will delete all data!"
	@read -p "Are you sure? [y/N] " -n 1 -r; \
	echo; \
	if [[ $$REPLY =~ ^[Yy]$$ ]]; then \
		cd services/data && poetry run alembic downgrade base && poetry run alembic upgrade head; \
		echo "✓ Database reset complete"; \
	fi

# Utilities
clean:
	@echo "Cleaning build artifacts..."
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "node_modules" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "dist" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	@echo "✓ Cleaned!"

docs:
	@echo "Generating documentation..."
	cd services/api && poetry run python -m mkdocs build
	@echo "✓ Documentation generated in docs/site/"
