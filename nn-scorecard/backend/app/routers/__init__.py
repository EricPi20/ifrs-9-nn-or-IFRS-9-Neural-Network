"""
API Routers

This package contains all FastAPI route handlers for different API endpoints.
"""

from app.routers import upload, training, results, scoring

__all__ = ["upload", "training", "results", "scoring"]

