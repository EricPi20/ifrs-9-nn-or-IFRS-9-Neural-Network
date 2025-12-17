"""
Job Storage Utility

This module provides in-memory storage for training jobs.
In production, this should be replaced with Redis or a database.
"""

from typing import Dict, Any
from datetime import datetime

# In-memory job storage (in production, use Redis or database)
training_jobs: Dict[str, Dict[str, Any]] = {}


def get_training_jobs() -> Dict[str, Dict[str, Any]]:
    """Get the training jobs dictionary."""
    return training_jobs


def get_training_job(job_id: str) -> Dict[str, Any]:
    """Get a specific training job."""
    return training_jobs.get(job_id)


def create_training_job(job_id: str, file_id: str, config: Dict) -> None:
    """Create a new training job."""
    training_jobs[job_id] = {
        "status": "pending",
        "created_at": datetime.now(),
        "config": config,
        "file_id": file_id
    }


def update_training_job(job_id: str, updates: Dict[str, Any]) -> None:
    """Update a training job with new information."""
    if job_id in training_jobs:
        training_jobs[job_id].update(updates)

