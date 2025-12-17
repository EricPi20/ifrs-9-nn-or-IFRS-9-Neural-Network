"""
Score Calculation Endpoints

This module handles endpoints for calculating credit scores using
trained scorecard models.
"""

from fastapi import APIRouter, HTTPException
from typing import List, Dict
from pydantic import BaseModel

from app.routers.training import training_jobs
from app.services.model_storage import ModelStorage
from app.services.scorecard import ScorecardGenerator, Scorecard, FeatureScore, BinScore
from app.core.constants import INPUT_SCALE_FACTOR

router = APIRouter()
storage = ModelStorage()
generator = ScorecardGenerator(scale_factor=INPUT_SCALE_FACTOR)


class ScoreRequest(BaseModel):
    records: List[Dict[str, float]]


class ScoreResult(BaseModel):
    total_score: int
    breakdown: Dict[str, int]
    risk_level: str


class ScoreResponse(BaseModel):
    scores: List[ScoreResult]


def get_risk_level(score: int) -> str:
    """Map score to risk level."""
    if score >= 80:
        return "EXCELLENT"
    elif score >= 60:
        return "GOOD"
    elif score >= 40:
        return "FAIR"
    elif score >= 20:
        return "POOR"
    else:
        return "VERY_POOR"


@router.post("/{job_id}/score", response_model=ScoreResponse)
async def calculate_scores(job_id: str, request: ScoreRequest):
    """
    Calculate scores for one or more records.
    
    Each record should contain ORIGINAL CSV values (standardized log odds × -50).
    The scorecard was generated to work with these original values.
    
    Example:
    {
        "records": [
            {
                "feature_payment": -40,   # Original CSV value
                "feature_util": 25,       # Original CSV value
                ...
            }
        ]
    }
    
    Returns:
    {
        "scores": [
            {
                "total_score": 75,  # 0-100, higher = lower risk
                "breakdown": {
                    "feature_1": 12,
                    "feature_2": -5,
                    ...
                },
                "risk_level": "GOOD"
            }
        ]
    }
    """
    # Get scorecard
    scorecard_data = None
    
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job['status'] != 'completed':
            raise HTTPException(status_code=400, detail=f"Model not ready. Status: {job['status']}")
        scorecard_data = job['result']['scorecard']
    else:
        try:
            checkpoint = storage.load_checkpoint(job_id)
            scorecard_data = checkpoint['metadata'].get('scorecard_output')
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
    
    if not scorecard_data:
        raise HTTPException(status_code=500, detail="Scorecard not found in model")
    
    # Reconstruct Scorecard object from dict
    features = []
    for fd in scorecard_data.get('features', []):
        bins = []
        for b in fd.get('bins', []):
            bins.append(BinScore(
                bin_index=b.get('bin_index', 0),
                input_value=b['input_value'],
                bin_label=b.get('bin_label', ''),
                raw_points=b.get('raw_points', 0.0),
                scaled_points=b['scaled_points'],
                count_train=b.get('count_train', 0),
                count_test=b.get('count_test', 0),
                bad_rate_train=b.get('bad_rate_train', 0.0),
                bad_rate_test=b.get('bad_rate_test', 0.0)
            ))
        features.append(FeatureScore(
            feature_name=fd['feature_name'],
            weight=fd['weight'],
            weight_normalized=fd.get('weight_normalized', fd['weight']),  # Handle backward compatibility
            importance_rank=fd.get('importance_rank', 0),
            bins=bins,
            min_points=fd.get('min_points', 0),
            max_points=fd.get('max_points', 0)
        ))
    
    scorecard = Scorecard(
        segment=scorecard_data.get('segment', ''),
        model_type=scorecard_data.get('model_type', 'neural_network'),
        scale_factor=scorecard_data.get('scale_factor', 1.0),
        offset=scorecard_data.get('offset', 0.0),
        input_scale_factor=scorecard_data.get('input_scale_factor', INPUT_SCALE_FACTOR),
        features=features
    )
    
    # Calculate scores
    results = []
    for record in request.records:
        total_score, breakdown = generator.calculate_score(scorecard, record)
        results.append(ScoreResult(
            total_score=total_score,
            breakdown=breakdown,
            risk_level=get_risk_level(total_score)
        ))
    
    return ScoreResponse(scores=results)


@router.get("/{job_id}/score-breakdown")
async def get_score_breakdown_template(job_id: str):
    """
    Get template showing how score is calculated.
    
    Returns all features with their bins and points.
    """
    scorecard_data = None
    
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job['status'] != 'completed':
            raise HTTPException(status_code=400, detail=f"Model not ready. Status: {job['status']}")
        scorecard_data = job['result']['scorecard']
    else:
        try:
            checkpoint = storage.load_checkpoint(job_id)
            scorecard_data = checkpoint['metadata'].get('scorecard_output')
        except FileNotFoundError:
            raise HTTPException(status_code=404, detail="Model not found")
    
    if not scorecard_data:
        raise HTTPException(status_code=500, detail="Scorecard not found in model")
    
    # Format for display
    breakdown = {
        'score_range': {'min': 0, 'max': 100},
        'calculation': 'Total Score = Σ(Feature Points), clamped to [0, 100]',
        'features': []
    }
    
    for feat in scorecard_data.get('features', []):
        breakdown['features'].append({
            'name': feat['feature_name'],
            'weight': feat['weight'],
            'bins': [
                {
                    'label': b.get('bin_label', ''),
                    'input_value': b['input_value'],
                    'points': b['scaled_points']
                }
                for b in feat.get('bins', [])
            ]
        })
    
    return breakdown
