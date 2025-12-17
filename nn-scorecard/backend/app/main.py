"""
FastAPI Application Entry Point

This module initializes the FastAPI application, configures CORS,
registers routers, and sets up the application lifecycle.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.config import settings
from app.routers import upload, training, results, scoring

app = FastAPI(
    title="Neural Network Scorecard API",
    description="API for training and deploying neural network credit scorecards",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register routers
app.include_router(upload.router, prefix="/api/upload", tags=["upload"])
app.include_router(training.router, prefix="/api/training", tags=["training"])
app.include_router(results.router, prefix="/api/results", tags=["results"])
app.include_router(scoring.router, prefix="/api/scoring", tags=["scoring"])


# Debug: Print all routes on startup
@app.on_event("startup")
async def startup_event():
    print("\n=== REGISTERED ROUTES ===")
    for route in app.routes:
        if hasattr(route, 'methods') and hasattr(route, 'path'):
            methods = ', '.join(sorted(route.methods))
            print(f"  {methods:20} {route.path}")
    print("=========================\n")


@app.get("/")
async def root():
    """Root endpoint for health check."""
    return {"message": "Neural Network Scorecard API", "status": "running"}


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy"}

