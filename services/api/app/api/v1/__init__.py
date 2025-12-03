"""
API v1 Router - Main router for all v1 endpoints
"""

from fastapi import APIRouter

from app.api.v1.endpoints import discovery, recommendations, simulation, deployment, status

api_router = APIRouter()

# Include all endpoint routers
api_router.include_router(discovery.router, prefix="/discover", tags=["Discovery"])
api_router.include_router(recommendations.router, prefix="/recommend", tags=["Recommendations"])
api_router.include_router(simulation.router, prefix="/simulate", tags=["Simulation"])
api_router.include_router(deployment.router, prefix="/deploy", tags=["Deployment"])
api_router.include_router(status.router, prefix="/status", tags=["Status"])
