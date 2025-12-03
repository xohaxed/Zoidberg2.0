# CloudOpt AI

**Production-ready cloud infrastructure optimization platform powered by AI**

[![CI](https://github.com/cloudopt-ai/cloudopt-ai/workflows/CI/badge.svg)](https://github.com/cloudopt-ai/cloudopt-ai/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit&logoColor=white)](https://github.com/pre-commit/pre-commit)

## 🚀 Overview

CloudOpt AI is an intelligent cloud infrastructure optimization platform that automatically discovers, analyzes, and optimizes multi-cloud deployments. It provides:

- **Auto-Discovery**: Automatic infrastructure scanning across AWS, Azure, and GCP
- **AI-Powered Recommendations**: Multi-objective optimization (cost, performance, CO2) using NSGA-II
- **One-Click Deployment**: Generate and deploy validated Terraform/Pulumi code
- **Real-Time Monitoring**: Comprehensive observability with Prometheus, Grafana, and OpenTelemetry
- **Data-Driven Insights**: Pricing catalogs, performance metrics, and predictive analytics

## 📁 Repository Structure

```
cloudopt-ai/
├── .github/              # GitHub workflows, issue templates, PR templates
├── services/             # Backend microservices
│   ├── api/             # FastAPI REST API gateway
│   ├── ai-engine/       # ML models, optimizer, training pipelines
│   └── data/            # Scrapers, ETL, connectors, migrations
├── web/                  # Frontend applications
│   └── dashboard/       # React + TypeScript SPA
├── infra/                # Infrastructure as Code
│   ├── terraform-modules/  # Reusable Terraform modules
│   ├── k8s/               # Kubernetes manifests and Helm charts
│   └── live/              # Environment-specific IaC
├── ci/                   # CI/CD scripts and security scanning
├── docs/                 # Architecture, runbooks, ADRs
├── dev/                  # Local development (docker-compose)
├── scripts/              # Utility scripts for development
├── examples/             # Sample payloads and use cases
└── templates/            # IaC templates (Terraform, Pulumi, CF)
```

## 🛠️ Quick Start

### Prerequisites

- Docker & Docker Compose
- Python 3.11+
- Node.js 18+
- Terraform 1.6+
- Make

### Local Development Setup

```bash
# Clone the repository
git clone https://github.com/cloudopt-ai/cloudopt-ai.git
cd cloudopt-ai

# Install pre-commit hooks
make setup-dev

# Start all services locally
make dev-up

# Run tests
make test

# Access services
# - API: http://localhost:8000/docs
# - Dashboard: http://localhost:3000
# - MLflow: http://localhost:5000
# - Grafana: http://localhost:3001
```

## 📦 Services

### API Service (`services/api/`)
FastAPI-based REST API providing endpoints for recommendations, simulations, and deployments.

**Key Endpoints:**
- `POST /api/v1/discover` - Trigger infrastructure discovery
- `POST /api/v1/recommend` - Get optimization recommendations
- `POST /api/v1/simulate` - Run what-if scenarios
- `POST /api/v1/deploy` - Deploy optimized infrastructure

### AI Engine (`services/ai-engine/`)
Machine learning models and multi-objective optimizer for infrastructure recommendations.

**Features:**
- NSGA-II multi-objective optimization
- Cost, performance, and CO2 prediction models
- SHAP explainability integration
- MLflow experiment tracking

### Data Service (`services/data/`)
Data ingestion, ETL pipelines, and pricing catalog management.

**Components:**
- Cloud provider price scrapers
- Observability connectors (CloudWatch, Prometheus, Datadog)
- Airflow DAG orchestration
- Data quality validation (Great Expectations)

### Dashboard (`web/dashboard/`)
React + TypeScript SPA for visualization and interaction.

**Features:**
- Infrastructure topology viewer
- Recommendation comparison UI
- Deployment workflow
- Real-time monitoring dashboards

## 🏗️ Infrastructure

### Terraform Modules (`infra/terraform-modules/`)
Reusable, tested Terraform modules:
- `network/` - VPC, subnets, security groups
- `compute/` - EC2, ECS, Lambda
- `db/` - RDS, DynamoDB, Aurora
- `iam/` - Roles, policies, service accounts
- `monitoring/` - CloudWatch, Prometheus, Grafana

### Kubernetes (`infra/k8s/`)
- Helm charts for each service
- Namespace configurations
- Service meshes (Istio/Linkerd)
- Chaos engineering (Chaos Mesh)
- GitOps with ArgoCD

## 🧪 Testing Strategy

- **Unit Tests**: pytest (Python), Jest (TypeScript)
- **Integration Tests**: testcontainers, docker-compose
- **IaC Tests**: Terratest, checkov, tfsec
- **E2E Tests**: Playwright
- **Security Scans**: Trivy, Snyk, Gitleaks

```bash
# Run all tests
make test

# Run specific test suites
make test-api
make test-ai-engine
make test-web
make test-infra
```

## 🔒 Security

- No secrets in repository (use `.env` templates)
- Secret management with AWS Secrets Manager / HashiCorp Vault
- Pre-commit hooks for secret scanning (gitleaks)
- Security scanning in CI (Trivy, Snyk)
- IaC security policies (checkov, tfsec)

See [SECURITY.md](SECURITY.md) for reporting vulnerabilities.

## 📊 Observability

- **Metrics**: Prometheus + Grafana
- **Traces**: OpenTelemetry + Jaeger
- **Logs**: Loki + FluentBit
- **Dashboards**: Pre-configured Grafana dashboards in `infra/monitoring/`

## 🤝 Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes with tests
4. Run pre-commit checks (`pre-commit run --all-files`)
5. Commit using conventional commits (`git commit -m 'feat: add amazing feature'`)
6. Push and create a Pull Request

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file.

## 📚 Documentation

- [Architecture Overview](docs/architecture.md)
- [API Documentation](services/api/README.md)
- [Deployment Guide](docs/runbooks/deployment.md)
- [Onboarding Guide](docs/onboarding.md)
- [Decision Records](docs/decision-records/)

## 🆘 Support

- 📧 Email: support@cloudopt-ai.example
- 💬 Slack: [Join our community](#)
- 🐛 Issues: [GitHub Issues](https://github.com/cloudopt-ai/cloudopt-ai/issues)

## 🗺️ Roadmap

- [x] Auto-discovery for AWS, Azure, GCP
- [x] Multi-objective optimization engine
- [x] Terraform code generation
- [ ] Pulumi support
- [ ] CloudFormation support
- [ ] FinOps integration
- [ ] Carbon footprint tracking
- [ ] Multi-region optimization
- [ ] Cost anomaly detection
- [ ] Automated remediation

---

**Built with ❤️ by the CloudOpt AI team**
