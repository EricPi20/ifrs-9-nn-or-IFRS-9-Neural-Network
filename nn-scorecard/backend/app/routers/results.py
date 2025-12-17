from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
import os
import io
import csv
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


@router.get("/{job_id}/download-scorecard-csv")
async def download_scorecard_csv(job_id: str):
    """Download scorecard data as CSV including feature importance and scorepoints per bin."""
    # Get scorecard and metrics
    scorecard_data = None
    metrics = None
    
    # Check in-memory first
    if job_id in training_jobs:
        job = training_jobs[job_id]
        if job.get('status') != 'completed':
            raise HTTPException(400, f"Training not complete. Status: {job.get('status')}")
        
        result = job.get('result', {})
        scorecard_data = result.get('scorecard') or job.get('scorecard')
        metrics = result.get('metrics') or job.get('current_metrics', {})
        
        # Try to get metrics from history if not available
        if not metrics or not metrics.get('train_auc'):
            history = result.get('history', {}).get('epochs', []) or job.get('history', [])
            if history and len(history) > 0:
                last_epoch = history[-1]
                if isinstance(last_epoch, dict):
                    metrics = {
                        'train_auc': last_epoch.get('train_auc', 0),
                        'test_auc': last_epoch.get('test_auc', 0),
                        'train_ar': last_epoch.get('train_ar', 0),
                        'test_ar': last_epoch.get('test_ar', 0),
                    }
    else:
        # Check storage
        try:
            checkpoint = storage.load_checkpoint(job_id)
            metadata = checkpoint['metadata']
            scorecard_data = metadata.get('scorecard_output') or metadata.get('scorecard')
            metrics = metadata.get('metrics', {})
        except FileNotFoundError:
            raise HTTPException(404, "Model not found")
    
    if not scorecard_data:
        raise HTTPException(404, "Scorecard not found")
    
    # Extract metrics
    train_auc = metrics.get('train_auc', 0) if metrics else 0
    test_auc = metrics.get('test_auc', 0) if metrics else 0
    train_ar = metrics.get('train_ar', 0) if metrics else 0
    test_ar = metrics.get('test_ar', 0) if metrics else 0
    # Gini = AR (they're the same: Gini = 2*AUC - 1 = AR)
    train_gini = train_ar
    test_gini = test_ar
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Header row with training info
    writer.writerow(['Training ID', job_id])
    writer.writerow(['Train AUC', f'{train_auc:.6f}'])
    writer.writerow(['Test AUC', f'{test_auc:.6f}'])
    writer.writerow(['Train AR (Gini)', f'{train_ar:.6f}'])
    writer.writerow(['Test AR (Gini)', f'{test_ar:.6f}'])
    writer.writerow([])  # Empty row
    
    # Feature Importance Table
    writer.writerow(['=== FEATURE IMPORTANCE TABLE ==='])
    writer.writerow(['Feature Name', 'Weight (%)', 'Importance Rank'])
    
    features = scorecard_data.get('features', [])
    for feature in features:
        writer.writerow([
            feature.get('feature_name', ''),
            feature.get('weight', 0),
            feature.get('importance_rank', 0)
        ])
    
    writer.writerow([])  # Empty row
    
    # Scorepoints per Feature per Bin
    writer.writerow(['=== SCOREPOINTS PER FEATURE PER BIN ==='])
    writer.writerow([
        'Feature Name',
        'Bin Index',
        'Bin Label',
        'Input Value',
        'Scaled Points',
        'Count Train',
        'Count Test',
        'Bad Rate Train',
        'Bad Rate Test'
    ])
    
    for feature in features:
        feature_name = feature.get('feature_name', '')
        bins = feature.get('bins', [])
        
        for bin_data in bins:
            writer.writerow([
                feature_name,
                bin_data.get('bin_index', ''),
                bin_data.get('bin_label', ''),
                bin_data.get('input_value', ''),
                bin_data.get('scaled_points', ''),
                bin_data.get('count_train', 0),
                bin_data.get('count_test', 0),
                f"{bin_data.get('bad_rate_train', 0):.6f}",
                f"{bin_data.get('bad_rate_test', 0):.6f}"
            ])
    
    # Prepare response
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename="scorecard_{job_id}.csv"'
        }
    )


@router.get("/{job_id}/download-config-csv")
async def download_config_csv(job_id: str):
    """Download training configuration as CSV with training ID and metrics."""
    # Get config and metrics
    config_data = None
    metrics = None
    
    # Check in-memory first
    if job_id in training_jobs:
        job = training_jobs[job_id]
        config_data = job.get('config', {})
        result = job.get('result', {})
        metrics = result.get('metrics') or job.get('current_metrics', {})
        
        # Try to get metrics from history if not available
        if not metrics or not metrics.get('train_auc'):
            history = result.get('history', {}).get('epochs', []) or job.get('history', [])
            if history and len(history) > 0:
                last_epoch = history[-1]
                if isinstance(last_epoch, dict):
                    metrics = {
                        'train_auc': last_epoch.get('train_auc', 0),
                        'test_auc': last_epoch.get('test_auc', 0),
                        'train_ar': last_epoch.get('train_ar', 0),
                        'test_ar': last_epoch.get('test_ar', 0),
                    }
    else:
        # Check storage
        try:
            checkpoint = storage.load_checkpoint(job_id)
            metadata = checkpoint['metadata']
            config_data = metadata.get('config', {})
            metrics = metadata.get('metrics', {})
        except FileNotFoundError:
            raise HTTPException(404, "Model not found")
    
    if not config_data:
        raise HTTPException(404, "Configuration not found")
    
    # Extract metrics
    train_auc = metrics.get('train_auc', 0) if metrics else 0
    test_auc = metrics.get('test_auc', 0) if metrics else 0
    train_ar = metrics.get('train_ar', 0) if metrics else 0
    test_ar = metrics.get('test_ar', 0) if metrics else 0
    
    # Create CSV content
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Training ID and Metrics
    writer.writerow(['Training ID', job_id])
    writer.writerow(['Train AUC', f'{train_auc:.6f}'])
    writer.writerow(['Test AUC', f'{test_auc:.6f}'])
    writer.writerow(['Train AR (Gini)', f'{train_ar:.6f}'])
    writer.writerow(['Test AR (Gini)', f'{test_ar:.6f}'])
    writer.writerow([])  # Empty row
    
    # Configuration Details
    writer.writerow(['=== TRAINING CONFIGURATION ==='])
    
    # Data Configuration
    writer.writerow(['Data Configuration', ''])
    writer.writerow(['Segment', config_data.get('segment', 'ALL')])
    writer.writerow(['Test Size', f"{config_data.get('test_size', 0.25) * 100:.0f}%"])
    writer.writerow(['Random Seed', config_data.get('random_seed', 42)])
    writer.writerow(['Stratified Split', 'Yes' if config_data.get('stratified_split', True) else 'No'])
    writer.writerow(['Selected Features', ', '.join(config_data.get('selected_features', []))])
    writer.writerow([])
    
    # Network Architecture
    network = config_data.get('network', {})
    writer.writerow(['Network Architecture', ''])
    writer.writerow(['Model Type', network.get('model_type', 'neural_network')])
    writer.writerow(['Hidden Layers', ' → '.join(map(str, network.get('hidden_layers', [])))])
    writer.writerow(['Activation', network.get('activation', 'relu')])
    writer.writerow(['Skip Connection', 'Yes' if network.get('skip_connection', False) else 'No'])
    writer.writerow([])
    
    # Regularization
    regularization = config_data.get('regularization', {})
    writer.writerow(['Regularization', ''])
    writer.writerow(['Dropout Rate', regularization.get('dropout_rate', 0.3)])
    writer.writerow(['L1 Lambda', regularization.get('l1_lambda', 0.0)])
    writer.writerow(['L2 Lambda', regularization.get('l2_lambda', 0.001)])
    writer.writerow([])
    
    # Training Parameters
    writer.writerow(['Training Parameters', ''])
    writer.writerow(['Learning Rate', config_data.get('learning_rate', 0.001)])
    writer.writerow(['Batch Size', config_data.get('batch_size', 32)])
    writer.writerow(['Epochs', config_data.get('epochs', 100)])
    writer.writerow(['Loss Function', config_data.get('loss_function', 'bce')])
    writer.writerow(['Use Class Weights', 'Yes' if config_data.get('use_class_weights', False) else 'No'])
    writer.writerow([])
    
    # Early Stopping
    early_stopping = config_data.get('early_stopping', {})
    writer.writerow(['Early Stopping', ''])
    writer.writerow(['Enabled', 'Yes' if early_stopping.get('enabled', False) else 'No'])
    if early_stopping.get('enabled', False):
        writer.writerow(['Patience', early_stopping.get('patience', 10)])
        writer.writerow(['Min Delta', early_stopping.get('min_delta', 0.001)])
    
    # Prepare response
    output.seek(0)
    
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode('utf-8')),
        media_type='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename="config_{job_id}.csv"'
        }
    )


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
