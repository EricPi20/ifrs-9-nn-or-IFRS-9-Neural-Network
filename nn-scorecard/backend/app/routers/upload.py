"""
File Upload Endpoints

This module handles file upload endpoints for CSV data files.
The input CSV contains features where each value is:
  Input = (log_odds_for_bin - mean) / std × (-50)

The data is ALREADY binned and transformed. No WoE transformation needed.
"""

from fastapi import APIRouter, UploadFile, File, HTTPException, Query
from fastapi.responses import JSONResponse
import pandas as pd
import numpy as np
import uuid
import os
import aiofiles
from pathlib import Path
from typing import Optional, List, Dict

from app.config import settings

router = APIRouter()

# In-memory storage (use database in production)
uploaded_files: Dict[str, Dict] = {}


def initialize_uploaded_files():
    """Scan upload directory and rebuild file index on startup."""
    upload_dir = settings.UPLOAD_DIR
    if isinstance(upload_dir, Path):
        upload_dir = str(upload_dir.resolve())
    else:
        upload_dir = str(upload_dir)
    
    if not os.path.exists(upload_dir):
        os.makedirs(upload_dir, exist_ok=True)
        return
    
    target_col = settings.DEFAULT_TARGET_COLUMN
    segment_col = settings.DEFAULT_SEGMENT_COLUMN
    id_col = settings.DEFAULT_ID_COLUMN
    
    for filename in os.listdir(upload_dir):
        if filename.endswith('.csv'):
            file_id = filename.replace('.csv', '')
            file_path = os.path.join(upload_dir, filename)
            
            try:
                df = pd.read_csv(file_path)
                
                # Validate target exists
                if target_col not in df.columns:
                    print(f"Skipping {filename}: target column '{target_col}' not found")
                    continue
                
                # Identify feature columns
                exclude_cols = {target_col, segment_col, id_col}
                feature_cols = [c for c in df.columns if c not in exclude_cols and c in df.columns]
                
                # Get segments
                segments = ['ALL']
                if segment_col in df.columns:
                    segments.extend(df[segment_col].unique().tolist())
                
                # Analyze features
                feature_analysis = []
                for col in feature_cols:
                    unique_values = sorted(df[col].dropna().unique().tolist())
                    feature_analysis.append({
                        'name': col,
                        'num_bins': len(unique_values),
                        'unique_values': unique_values,
                        'min_value': min(unique_values) if unique_values else 0,
                        'max_value': max(unique_values) if unique_values else 0,
                        'mean_value': float(df[col].mean()),
                        'correlation': float(df[col].corr(df[target_col])) if len(unique_values) > 1 else 0
                    })
                
                # Sort by absolute correlation
                feature_analysis.sort(key=lambda x: abs(x['correlation']), reverse=True)
                
                uploaded_files[file_id] = {
                    'path': file_path,
                    'filename': filename,
                    'segments': segments,
                    'feature_names': feature_cols,
                    'feature_analysis': feature_analysis
                }
                print(f"Restored file: {file_id} ({filename})")
            except Exception as e:
                print(f"Error loading {filename}: {e}")


# Initialize on module load
initialize_uploaded_files()


@router.post("")
async def upload_file(file: UploadFile = File(...)):
    """
    Upload CSV file for scorecard development.
    
    Expected CSV format:
    - Feature columns: Values are standardized log odds × (-50)
      (already binned and transformed upstream)
    - Target column: 0 (Good) or 1 (Bad)
    - Optional: segment column
    - Optional: account_id column
    
    Feature values are DISCRETE (one value per bin).
    """
    if not file.filename.endswith('.csv'):
        raise HTTPException(400, "Only CSV files are supported")
    
    # Read file content
    content = await file.read()
    
    # Validate file size
    file_size_mb = len(content) / (1024 * 1024)
    max_size_mb = settings.MAX_UPLOAD_SIZE_MB
    if file_size_mb > max_size_mb:
        raise HTTPException(
            413,
            f"File size ({file_size_mb:.2f}MB) exceeds maximum allowed size ({max_size_mb}MB)"
        )
    
    file_id = str(uuid.uuid4())
    file_path = os.path.join(settings.UPLOAD_DIR, f"{file_id}.csv")
    
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(content)
    
    # Read and validate
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        os.remove(file_path)
        raise HTTPException(400, f"Failed to parse CSV: {str(e)}")
    
    # Identify columns
    target_col = settings.DEFAULT_TARGET_COLUMN
    segment_col = settings.DEFAULT_SEGMENT_COLUMN
    id_col = settings.DEFAULT_ID_COLUMN
    
    # Validate target exists
    if target_col not in df.columns:
        os.remove(file_path)
        raise HTTPException(400, f"Target column '{target_col}' not found")
    
    # Validate target is binary
    unique_targets = df[target_col].unique()
    if not set(unique_targets).issubset({0, 1}):
        os.remove(file_path)
        raise HTTPException(400, "Target must contain only 0 and 1")
    
    # Identify feature columns (exclude target, segment, id)
    exclude_cols = {target_col, segment_col, id_col}
    feature_cols = [c for c in df.columns if c not in exclude_cols and c in df.columns]
    
    # Get segments
    segments = ['ALL']
    if segment_col in df.columns:
        segments.extend(df[segment_col].unique().tolist())
    
    # Analyze features
    feature_analysis = []
    for col in feature_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        feature_analysis.append({
            'name': col,
            'num_bins': len(unique_values),
            'unique_values': unique_values,  # These are the transformed log odds × -50
            'min_value': min(unique_values) if unique_values else 0,
            'max_value': max(unique_values) if unique_values else 0,
            'mean_value': float(df[col].mean()),
            'correlation': float(df[col].corr(df[target_col])) if len(unique_values) > 1 else 0
        })
    
    # Sort by absolute correlation
    feature_analysis.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    # Calculate segment stats
    segment_stats = []
    for seg in segments:
        if seg == 'ALL':
            seg_df = df
        else:
            seg_df = df[df[segment_col] == seg]
        
        segment_stats.append({
            'segment': seg,
            'count': len(seg_df),
            'bad_count': int(seg_df[target_col].sum()),
            'good_count': int((seg_df[target_col] == 0).sum()),
            'bad_rate': float(seg_df[target_col].mean())
        })
    
    # Store file info
    uploaded_files[file_id] = {
        'path': file_path,
        'filename': file.filename,
        'segments': segments,
        'feature_names': feature_cols,
        'feature_analysis': feature_analysis
    }
    
    return {
        'file_id': file_id,
        'filename': file.filename,
        'num_records': len(df),
        'num_features': len(feature_cols),
        'segments': segments,
        'segment_stats': segment_stats,
        'features': feature_analysis,
        'target_stats': {
            'good_count': int((df[target_col] == 0).sum()),
            'bad_count': int(df[target_col].sum()),
            'bad_rate': float(df[target_col].mean())
        }
    }


@router.get("/{file_id}/segments")
async def get_segments(file_id: str):
    """Get segment statistics."""
    # Try to get from memory first
    if file_id in uploaded_files:
        file_path = uploaded_files[file_id]['path']
    else:
        # Fallback: try to find the file in upload directory
        # Convert to absolute path to avoid relative path issues
        upload_dir = settings.UPLOAD_DIR
        if isinstance(upload_dir, Path):
            upload_dir = str(upload_dir.resolve())
        else:
            upload_dir = str(upload_dir)
        
        file_path = os.path.join(upload_dir, f"{file_id}.csv")
        
        # Also try relative path from current working directory
        if not os.path.exists(file_path):
            # Try multiple possible paths
            possible_paths = [
                os.path.join("data", "uploads", f"{file_id}.csv"),
                os.path.join(os.getcwd(), "data", "uploads", f"{file_id}.csv"),
                os.path.abspath(os.path.join("data", "uploads", f"{file_id}.csv")),
            ]
            
            found = False
            for path in possible_paths:
                if os.path.exists(path):
                    file_path = os.path.abspath(path)
                    found = True
                    break
            
            if not found:
                # Get current working directory for debugging
                cwd = os.getcwd()
                all_tried = [file_path] + possible_paths
                raise HTTPException(404, f"File not found: {file_id}. CWD: {cwd}. Tried: {all_tried}")
    
    df = pd.read_csv(file_path)
    target_col = settings.DEFAULT_TARGET_COLUMN
    segment_col = settings.DEFAULT_SEGMENT_COLUMN
    
    # Validate target exists
    if target_col not in df.columns:
        raise HTTPException(400, f"Target column '{target_col}' not found in file")
    
    segments = ['ALL']
    if segment_col in df.columns:
        segments.extend(df[segment_col].unique().tolist())
    
    stats = []
    for seg in segments:
        seg_df = df if seg == 'ALL' else df[df[segment_col] == seg]
        stats.append({
            'segment': seg,
            'count': len(seg_df),
            'bad_count': int(seg_df[target_col].sum()),
            'bad_rate': float(seg_df[target_col].mean())
        })
    
    return {'segments': stats}


@router.get("/{file_id}/features")
async def get_features(
    file_id: str,
    segment: Optional[str] = Query(None)
):
    """
    Get feature analysis including unique bin values.
    
    Each feature's unique_values are the discrete input values
    (standardized log odds × -50) for each bin.
    """
    # Fallback: check if file exists on disk
    if file_id not in uploaded_files:
        upload_dir = settings.UPLOAD_DIR
        if isinstance(upload_dir, Path):
            upload_dir = str(upload_dir.resolve())
        else:
            upload_dir = str(upload_dir)
        
        file_path = os.path.join(upload_dir, f"{file_id}.csv")
        
        if os.path.exists(file_path):
            # Rebuild file info
            df = pd.read_csv(file_path)
            target_col = settings.DEFAULT_TARGET_COLUMN
            segment_col = settings.DEFAULT_SEGMENT_COLUMN
            id_col = settings.DEFAULT_ID_COLUMN
            
            # Identify feature columns
            exclude_cols = {target_col, segment_col, id_col}
            feature_cols = [c for c in df.columns if c not in exclude_cols and c in df.columns]
            
            # Get segments
            segments = ['ALL']
            if segment_col in df.columns:
                segments.extend(df[segment_col].unique().tolist())
            
            # Analyze features
            feature_analysis = []
            for col in feature_cols:
                unique_values = sorted(df[col].dropna().unique().tolist())
                feature_analysis.append({
                    'name': col,
                    'num_bins': len(unique_values),
                    'unique_values': unique_values,
                    'min_value': min(unique_values) if unique_values else 0,
                    'max_value': max(unique_values) if unique_values else 0,
                    'mean_value': float(df[col].mean()),
                    'correlation': float(df[col].corr(df[target_col])) if len(unique_values) > 1 else 0
                })
            
            # Sort by absolute correlation
            feature_analysis.sort(key=lambda x: abs(x['correlation']), reverse=True)
            
            uploaded_files[file_id] = {
                'path': file_path,
                'filename': f"{file_id}.csv",
                'segments': segments,
                'feature_names': feature_cols,
                'feature_analysis': feature_analysis
            }
        else:
            raise HTTPException(404, "File not found")
    
    df = pd.read_csv(uploaded_files[file_id]['path'])
    target_col = settings.DEFAULT_TARGET_COLUMN
    segment_col = settings.DEFAULT_SEGMENT_COLUMN
    
    # Filter by segment
    if segment and segment != 'ALL' and segment_col in df.columns:
        df = df[df[segment_col] == segment]
    
    feature_cols = uploaded_files[file_id]['feature_names']
    
    features = []
    for col in feature_cols:
        unique_values = sorted(df[col].dropna().unique().tolist())
        
        # Calculate stats per unique value (bin)
        bin_stats = []
        for val in unique_values:
            mask = df[col] == val
            bin_stats.append({
                'input_value': val,  # This is standardized log odds × -50
                'count': int(mask.sum()),
                'bad_count': int(df.loc[mask, target_col].sum()),
                'bad_rate': float(df.loc[mask, target_col].mean()) if mask.sum() > 0 else 0
            })
        
        features.append({
            'name': col,
            'num_bins': len(unique_values),
            'unique_values': unique_values,
            'bin_stats': bin_stats,
            'correlation': float(df[col].corr(df[target_col])) if len(unique_values) > 1 else 0
        })
    
    features.sort(key=lambda x: abs(x['correlation']), reverse=True)
    
    return {
        'segment': segment or 'ALL',
        'num_records': len(df),
        'features': features
    }


@router.get("/{file_id}/bin-mapping")
async def get_bin_mapping(
    file_id: str,
    segment: Optional[str] = Query(None)
):
    """
    Get mapping of input values to bin labels (if provided separately).
    
    This endpoint can be used to associate human-readable bin labels
    with the transformed input values.
    """
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    
    # Check if bin mapping file exists
    mapping_path = uploaded_files[file_id]['path'].replace('.csv', '_bin_mapping.csv')
    
    if os.path.exists(mapping_path):
        mapping_df = pd.read_csv(mapping_path)
        return {'bin_mapping': mapping_df.to_dict('records')}
    
    # If no mapping file, return just the unique values
    df = pd.read_csv(uploaded_files[file_id]['path'])
    feature_cols = uploaded_files[file_id]['feature_names']
    
    mapping = {}
    for col in feature_cols:
        unique_vals = sorted(df[col].dropna().unique().tolist())
        mapping[col] = [
            {'input_value': v, 'bin_label': f'Bin {i+1} (value: {v:.1f})'}
            for i, v in enumerate(unique_vals)
        ]
    
    return {'bin_mapping': mapping}


@router.post("/analyze-file")
async def analyze_file(file_path: str):
    """Analyze uploaded file and return available segments and features."""
    try:
        df = pd.read_csv(file_path)
        
        # Detect segment column
        segment_col = None
        possible_segment_cols = [
            'segment', 'SEGMENT', 'Segment', 
            'customer_segment', 'cust_segment',
            'seg', 'category', 'group', 'type'
        ]
        
        for col in df.columns:
            if col in possible_segment_cols or 'segment' in col.lower():
                segment_col = col
                break
        
        segments = []
        if segment_col:
            segment_counts = df[segment_col].value_counts().to_dict()
            segments = [
                {'value': str(k), 'count': int(v)} 
                for k, v in segment_counts.items()
            ]
        
        # Detect target column
        target_col = None
        for col in df.columns:
            col_lower = col.lower()
            if col_lower in ['target', 'bad_flag', 'bad', 'default', 'is_bad', 'y']:
                target_col = col
                break
            if 'bad' in col_lower or 'target' in col_lower:
                target_col = col
                break
        
        # Get numeric features (exclude target and segment)
        exclude = set()
        if target_col:
            exclude.add(target_col)
        if segment_col:
            exclude.add(segment_col)
        
        features = [
            {
                'name': col,
                'dtype': str(df[col].dtype),
                'nunique': int(df[col].nunique()),
                'sample_values': df[col].head(3).tolist(),
            }
            for col in df.columns
            if col not in exclude and df[col].dtype in ['int64', 'float64', 'int32', 'float32']
        ]
        
        return {
            'total_rows': len(df),
            'total_columns': len(df.columns),
            'segment_column': segment_col,
            'segments': segments,
            'target_column': target_col,
            'features': features,
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{file_id}")
async def delete_upload(file_id: str):
    """Delete uploaded file."""
    if file_id not in uploaded_files:
        raise HTTPException(404, "File not found")
    
    file_path = uploaded_files[file_id]['path']
    if os.path.exists(file_path):
        os.remove(file_path)
    
    del uploaded_files[file_id]
    return {"status": "deleted", "file_id": file_id}
