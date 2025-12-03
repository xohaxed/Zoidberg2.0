# CloudOpt AI - AI Engine Service

Machine Learning engine for cloud infrastructure optimization.

## Features

- **Multi-Objective Optimization**: NSGA-II algorithm balancing cost, performance, and CO2
- **Predictive Models**: Cost, latency, and carbon footprint prediction
- **Model Training**: Automated training pipelines with MLflow tracking
- **Explainability**: SHAP-based model explanations
- **Model Serving**: FastAPI endpoints for inference

## Models

### Cost Predictor
- Predicts monthly cost for infrastructure configurations
- Features: instance types, regions, storage, data transfer
- Algorithm: Gradient Boosting (XGBoost)

### Performance Predictor
- Predicts P95 latency and throughput
- Features: workload characteristics, infrastructure config
- Algorithm: Neural Network (PyTorch)

### Carbon Predictor
- Estimates CO2 emissions
- Features: region carbon intensity, energy consumption
- Data: Electricity Maps API

## Quick Start

```bash
# Install dependencies
poetry install

# Train models
poetry run python cloudopt_ai/model/train.py

# Start model serving
poetry run uvicorn cloudopt_ai.serve:app --port 8001
```

## Optimization

```python
from cloudopt_ai.optimizer import NSGA2Optimizer

optimizer = NSGA2Optimizer(population_size=100, n_generations=50)

recommendations = optimizer.optimize(
    current_infrastructure=infrastructure_config,
    objectives=["cost", "performance", "co2"],
    constraints={"max_budget": 10000, "max_latency_p95": 200}
)

# Returns 3-5 Pareto-optimal solutions
for plan in recommendations:
    print(f"Cost: ${plan['objectives']['cost']}/mo")
    print(f"Latency P95: {plan['objectives']['latency_p95']}ms")
    print(f"Carbon: {plan['objectives']['carbon_kg']}kg CO2/mo")
```

## MLflow Tracking

```bash
# Start MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --port 5000

# Track experiments
export MLFLOW_TRACKING_URI=http://localhost:5000
poetry run python cloudopt_ai/model/train.py
```

## Testing

```bash
poetry run pytest tests/ -v
poetry run pytest --cov=cloudopt_ai
```
