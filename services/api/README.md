# CloudOpt AI - API Service

FastAPI-based REST API service for CloudOpt AI platform.

## Features

- **Auto-Discovery**: Trigger infrastructure discovery across AWS, Azure, GCP
- **Recommendations**: AI-powered optimization recommendations (NSGA-II)
- **Simulation**: What-if scenario analysis
- **Deployment**: Generate and deploy IaC code (Terraform/Pulumi)
- **Status**: Track deployment and optimization status

## Architecture

```
app/
├── api/
│   └── v1/
│       └── endpoints/
│           ├── discovery.py         # Infrastructure discovery
│           ├── recommendations.py   # Optimization recommendations
│           ├── simulation.py        # What-if scenarios
│           ├── deployment.py        # IaC generation & deployment
│           └── status.py            # Status tracking
├── core/
│   ├── config.py                    # Configuration settings
│   ├── security.py                  # Auth & security
│   └── logging.py                   # Structured logging
├── models/                           # SQLAlchemy models
├── schemas/                          # Pydantic schemas
├── services/                         # Business logic
└── main.py                           # FastAPI app entry point
```

## Quick Start

### Installation

```bash
# Install dependencies
poetry install

# Set up environment
cp .env.example .env

# Run database migrations
alembic upgrade head
```

### Development Server

```bash
# Run with hot reload
poetry run uvicorn app.main:app --reload --port 8000

# Access docs
open http://localhost:8000/docs
```

### Testing

```bash
# Run all tests
poetry run pytest

# With coverage
poetry run pytest --cov=app --cov-report=html

# Run specific test file
poetry run pytest tests/test_api.py -v
```

## API Endpoints

### Discovery
- `POST /api/v1/discover` - Start infrastructure discovery
- `GET /api/v1/discover/{discovery_id}` - Get discovery status
- `GET /api/v1/discover/{discovery_id}/resources` - List discovered resources

### Recommendations
- `POST /api/v1/recommend` - Generate optimization recommendations
- `GET /api/v1/recommend/{recommendation_id}` - Get recommendation
- `GET /api/v1/recommend/{recommendation_id}/explain` - Get SHAP explanation

### Simulation
- `POST /api/v1/simulate` - Run what-if simulation
- `GET /api/v1/simulate/{simulation_id}` - Get simulation results

### Deployment
- `POST /api/v1/deploy` - Generate and deploy IaC
- `POST /api/v1/deploy/validate` - Dry-run validation
- `GET /api/v1/deploy/{deployment_id}` - Get deployment status
- `POST /api/v1/deploy/{deployment_id}/rollback` - Rollback deployment

### Status & Health
- `GET /health` - Health check
- `GET /ready` - Readiness check
- `GET /metrics` - Prometheus metrics
- `GET /api/v1/status/{job_id}` - Job status

## Configuration

### Environment Variables

```bash
# Application
DEBUG=false
VERSION=0.1.0
API_V1_PREFIX=/api/v1

# Database
DATABASE_URL=postgresql+asyncpg://user:pass@localhost/cloudopt
REDIS_URL=redis://localhost:6379/0

# Security
SECRET_KEY=your-secret-key-change-in-production
ACCESS_TOKEN_EXPIRE_MINUTES=30

# Cloud Providers
AWS_REGION=us-east-1
AZURE_SUBSCRIPTION_ID=xxx
GCP_PROJECT_ID=xxx

# AI Engine
AI_ENGINE_URL=http://localhost:8001
MLFLOW_TRACKING_URI=http://localhost:5000

# Observability
OTEL_EXPORTER_OTLP_ENDPOINT=http://localhost:4317
PROMETHEUS_MULTIPROC_DIR=/tmp/prometheus
```

## Authentication

API uses OAuth2 with JWT tokens:

```bash
# Get access token
curl -X POST http://localhost:8000/api/v1/auth/token \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "username=user&password=pass"

# Use token
curl -X POST http://localhost:8000/api/v1/recommend \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{"infrastructure_id": "..."}'
```

## Docker

```bash
# Build image
docker build -t cloudopt-ai/api:latest .

# Run container
docker run -p 8000:8000 \
  -e DATABASE_URL=postgresql://... \
  cloudopt-ai/api:latest
```

## Testing Strategy

- **Unit Tests**: Test individual functions and classes
- **Integration Tests**: Test API endpoints with test database
- **Contract Tests**: Verify API contracts with Pact
- **Load Tests**: Locust load testing scenarios

## Monitoring

- **Metrics**: Prometheus metrics at `/metrics`
- **Traces**: OpenTelemetry traces to Jaeger
- **Logs**: Structured JSON logs via structlog
- **Health**: `/health` and `/ready` endpoints

## Contributing

See [CONTRIBUTING.md](../../CONTRIBUTING.md) for guidelines.

## License

MIT License - see [LICENSE](../../LICENSE)
