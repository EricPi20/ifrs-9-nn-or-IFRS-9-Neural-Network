from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
import os
from typing import Optional

from .training import training_jobs
from ..services.model_storage import ModelStorage
from ..config import settings

router = APIRouter()
storage = ModelStorage()


@router.get("")
async def list_results(segment: Optional[str] = Query(None)):
    """List all trained models."""
    models = storage.list_models(segment=segment)
    return {'models': models}


@router.get("/{job_id}")
async def get_results(job_id: str):
    """Get complete training results."""
    # Check in-memory first
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job.get('status') != 'completed':
            raise HTTPException(400, f"Training not complete. Status: {job.get('status')}")
        
        # Import helper functions from training router
        from .training import generate_scorecard, generate_validation_data
        
        # Get scorecard
        scorecard = job.get('scorecard')
        if not scorecard:
            scorecard = generate_scorecard(job_id, job)
            training_jobs[job_id]['scorecard'] = scorecard
        
        # Get validation data
        validation = job.get('validation_data')
        if not validation:
            validation = generate_validation_data(job)
            training_jobs[job_id]['validation_data'] = validation
        
        return {
            'job_id': job_id,
            'status': 'completed',
            'scorecard': scorecard,
            'validation': validation,
        }
    
    # Check storage
    try:
        checkpoint = storage.load_checkpoint(job_id)
        return checkpoint['metadata']
    except FileNotFoundError:
        raise HTTPException(404, "Model not found")


@router.get("/{job_id}/scorecard")
async def get_scorecard(job_id: str):
    """
    Get scorecard with all features and bin points.
    
    Response includes:
    - Per-feature weights
    - Per-bin: input_value, scaled_points, bin_label
    - Score range info
    """
    print(f"[RESULTS] Getting scorecard for job: {job_id}")
    
    # Check in-memory first
    if job_id in training_jobs:
        job = training_jobs[job_id]
        print(f"[RESULTS] Job found in memory. Status: {job.get('status')}")
        print(f"[RESULTS] Job keys: {list(job.keys())}")
        
        if job.get('status') != 'completed':
            raise HTTPException(
                status_code=400, 
                detail=f"Training not complete. Status: {job.get('status')}"
            )
        
        # Check if result exists
        if 'result' not in job:
            print(f"[RESULTS] ERROR: 'result' key not found in job")
            print(f"[RESULTS] Available keys: {list(job.keys())}")
            raise HTTPException(
                status_code=500,
                detail="Training result not found. The job may have completed but result was not saved."
            )
        
        result = job['result']
        print(f"[RESULTS] Result keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        # Check if scorecard exists
        if 'scorecard' not in result:
            print(f"[RESULTS] ERROR: 'scorecard' key not found in result")
            print(f"[RESULTS] Available result keys: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
            raise HTTPException(
                status_code=404,
                detail="Scorecard not generated. The training completed but scorecard was not created."
            )
        
        scorecard = result['scorecard']
        if scorecard is None:
            raise HTTPException(
                status_code=404,
                detail="Scorecard is None. The training completed but scorecard generation failed."
            )
        
        print(f"[RESULTS] Scorecard found with {len(scorecard.get('features', []))} features")
        
        return {
            'scorecard': scorecard,
            'score_range': {
                'min': 0,
                'max': 100,
                'interpretation': 'Higher score = Lower risk'
            }
        }
    
    # Check storage
    print(f"[RESULTS] Job not in memory, checking storage...")
    try:
        checkpoint = storage.load_checkpoint(job_id)
        scorecard_data = checkpoint['metadata'].get('scorecard_output') or checkpoint['metadata'].get('scorecard')
        
        if not scorecard_data:
            print(f"[RESULTS] ERROR: Scorecard not found in checkpoint metadata")
            print(f"[RESULTS] Metadata keys: {list(checkpoint['metadata'].keys())}")
            raise HTTPException(
                status_code=404,
                detail="Scorecard not found in stored model"
            )
        
        print(f"[RESULTS] Scorecard found in storage")
        return {
            'scorecard': scorecard_data,
            'score_range': {
                'min': 0,
                'max': 100,
                'interpretation': 'Higher score = Lower risk'
            }
        }
    except FileNotFoundError:
        print(f"[RESULTS] ERROR: Model checkpoint not found")
        print(f"[RESULTS] Available jobs in memory: {list(training_jobs.keys())}")
        raise HTTPException(
            status_code=404,
            detail=f"Model not found. Available jobs: {list(training_jobs.keys())}"
        )
    except KeyError as e:
        print(f"[RESULTS] ERROR: KeyError accessing checkpoint: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing stored model data: {str(e)}"
        )
    except Exception as e:
        print(f"[RESULTS] ERROR: Unexpected error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail=f"Internal server error: {str(e)}"
        )


@router.get("/{job_id}/metrics")
async def get_metrics(job_id: str):
    """Get detailed metrics."""
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job['status'] != 'completed':
            raise HTTPException(400, f"Training not complete. Status: {job['status']}")
        
        result = job['result']
        return {
            'metrics': result['metrics'],
            'history_summary': {
                'total_epochs': len(result['history']['epochs']),
                'best_epoch': result['history']['best_epoch'],
                'best_test_ar': result['history']['best_test_ar']
            }
        }
    
    raise HTTPException(404, "Model not found")


@router.get("/{job_id}/export")
async def export_results(
    job_id: str,
    format: str = Query("excel", regex="^(excel|json)$")
):
    """Export training report."""
    if format == "excel":
        output_path = os.path.join(settings.EXPORT_DIR, f"{job_id}_report.xlsx")
        
        try:
            storage.export_to_excel(job_id, output_path)
        except FileNotFoundError:
            raise HTTPException(404, "Model not found")
        
        return FileResponse(
            output_path,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            filename=f"RIFT_Scorecard_{job_id}.xlsx"
        )
    else:
        # JSON export
        try:
            checkpoint = storage.load_checkpoint(job_id)
            return checkpoint['metadata']
        except FileNotFoundError:
            raise HTTPException(404, "Model not found")


@router.delete("/{job_id}")
async def delete_result(job_id: str):
    """Delete a model."""
    # Remove from in-memory
    if job_id in training_jobs:
        del training_jobs[job_id]
    
    # Remove from storage
    deleted = storage.delete_model(job_id)
    
    if not deleted and job_id not in training_jobs:
        raise HTTPException(404, "Model not found")
    
    return {"status": "deleted", "job_id": job_id}
