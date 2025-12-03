"""API integration tests."""

import pytest
from httpx import AsyncClient


@pytest.mark.asyncio
async def test_health_check(async_client: AsyncClient):
    """Test health check endpoint."""
    response = await async_client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


@pytest.mark.asyncio
async def test_readiness_check(async_client: AsyncClient):
    """Test readiness check endpoint."""
    response = await async_client.get("/ready")
    assert response.status_code == 200


@pytest.mark.asyncio
async def test_recommend_endpoint_requires_auth(async_client: AsyncClient):
    """Test that recommendations endpoint requires authentication."""
    response = await async_client.post(
        "/api/v1/recommend/",
        json={"infrastructure_id": "test-id", "objectives": ["cost"]},
    )
    assert response.status_code == 401
