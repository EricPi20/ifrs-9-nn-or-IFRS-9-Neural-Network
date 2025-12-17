"""
Tests for Upload API Endpoints

Tests for file upload and feature analysis endpoints.
Verifies that discrete bin values are correctly handled.
"""

import pytest
import pandas as pd
import io
import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient
from fastapi import FastAPI
import importlib.util

# Import upload router directly without going through __init__.py
upload_module_path = Path(__file__).parent.parent / "app" / "routers" / "upload.py"
spec = importlib.util.spec_from_file_location("upload", upload_module_path)
upload_module = importlib.util.module_from_spec(spec)
sys.modules["upload"] = upload_module
spec.loader.exec_module(upload_module)

upload_router = upload_module.router
uploaded_files = upload_module.uploaded_files

# Create a minimal app with only the upload router to avoid import issues
app = FastAPI()
app.include_router(upload_router, prefix="/api/upload", tags=["upload"])

client = TestClient(app)


@pytest.fixture
def test_csv_data():
    """Create test CSV with discrete bin values."""
    # Create data with discrete bin values
    # feature_1: [-40, 25, 15, -10]
    # feature_2: [-30, 20, 10, -5]
    # target: 0 or 1
    
    data = {
        'feature_1': [-40, 25, 15, -10, -40, 25, 15, -10, -40, 25],
        'feature_2': [-30, 20, 10, -5, -30, 20, 10, -5, -30, 20],
        'target': [1, 0, 0, 1, 1, 0, 0, 1, 1, 0]
    }
    
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue()
    
    return csv_content.encode('utf-8')


@pytest.fixture(autouse=True)
def cleanup_uploads():
    """Clean up uploaded files after each test."""
    yield
    # Clear in-memory storage
    uploaded_files.clear()
    
    # Clean up any uploaded files
    from app.config import settings
    upload_dir = settings.UPLOAD_DIR
    if upload_dir.exists():
        for file in upload_dir.glob("*.csv"):
            try:
                file.unlink()
            except:
                pass


def test_upload_file_post(test_csv_data):
    """Test POST /api/upload returns file_id and features with unique_values."""
    # Create a file-like object
    files = {
        'file': ('test_data.csv', test_csv_data, 'text/csv')
    }
    
    response = client.post("/api/upload", files=files)
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert 'file_id' in data
    assert isinstance(data['file_id'], str)
    assert len(data['file_id']) > 0
    
    assert 'filename' in data
    assert data['filename'] == 'test_data.csv'
    
    assert 'num_records' in data
    assert data['num_records'] == 10
    
    assert 'num_features' in data
    assert data['num_features'] == 2
    
    # Verify features structure
    assert 'features' in data
    assert isinstance(data['features'], list)
    assert len(data['features']) == 2
    
    # Find feature_1 and feature_2
    feature_1 = next(f for f in data['features'] if f['name'] == 'feature_1')
    feature_2 = next(f for f in data['features'] if f['name'] == 'feature_2')
    
    # Verify feature_1 has correct unique_values (DISCRETE, not continuous)
    assert 'unique_values' in feature_1
    assert feature_1['unique_values'] == [-40, -10, 15, 25]  # Sorted
    assert feature_1['num_bins'] == 4
    
    # Verify these are exactly the discrete input values, not continuous
    # Check that all values are integers (or exact floats)
    for val in feature_1['unique_values']:
        assert isinstance(val, (int, float))
        # Verify it's one of the exact input values
        assert val in [-40, 25, 15, -10]
    
    # Verify feature_2 has correct unique_values (DISCRETE, not continuous)
    assert 'unique_values' in feature_2
    assert feature_2['unique_values'] == [-30, -5, 10, 20]  # Sorted
    assert feature_2['num_bins'] == 4
    
    # Verify these are exactly the discrete input values, not continuous
    for val in feature_2['unique_values']:
        assert isinstance(val, (int, float))
        # Verify it's one of the exact input values
        assert val in [-30, 20, 10, -5]
    
    # Verify other feature fields
    assert 'min_value' in feature_1
    assert 'max_value' in feature_1
    assert 'mean_value' in feature_1
    assert 'correlation' in feature_1
    
    assert feature_1['min_value'] == -40
    assert feature_1['max_value'] == 25
    
    # Verify segment_stats
    assert 'segment_stats' in data
    assert isinstance(data['segment_stats'], list)
    assert len(data['segment_stats']) >= 1
    
    # Verify target_stats
    assert 'target_stats' in data
    assert 'good_count' in data['target_stats']
    assert 'bad_count' in data['target_stats']
    assert 'bad_rate' in data['target_stats']
    
    return data['file_id']


def test_get_features_endpoint(test_csv_data):
    """Test GET /api/upload/{file_id}/features returns bin-level statistics."""
    # First upload a file
    files = {
        'file': ('test_data.csv', test_csv_data, 'text/csv')
    }
    
    upload_response = client.post("/api/upload", files=files)
    assert upload_response.status_code == 200
    file_id = upload_response.json()['file_id']
    
    # Now get features
    response = client.get(f"/api/upload/{file_id}/features")
    
    assert response.status_code == 200
    data = response.json()
    
    # Verify response structure
    assert 'segment' in data
    assert data['segment'] == 'ALL'
    
    assert 'num_records' in data
    assert data['num_records'] == 10
    
    assert 'features' in data
    assert isinstance(data['features'], list)
    assert len(data['features']) == 2
    
    # Find feature_1 and feature_2
    feature_1 = next(f for f in data['features'] if f['name'] == 'feature_1')
    feature_2 = next(f for f in data['features'] if f['name'] == 'feature_2')
    
    # Verify feature_1 has unique_values (DISCRETE)
    assert 'unique_values' in feature_1
    assert feature_1['unique_values'] == [-40, -10, 15, 25]  # Sorted
    assert feature_1['num_bins'] == 4
    
    # Verify bin_stats exist and are correct
    assert 'bin_stats' in feature_1
    assert isinstance(feature_1['bin_stats'], list)
    assert len(feature_1['bin_stats']) == 4
    
    # Verify each bin_stat has the correct structure
    for bin_stat in feature_1['bin_stats']:
        assert 'input_value' in bin_stat
        assert 'count' in bin_stat
        assert 'bad_count' in bin_stat
        assert 'bad_rate' in bin_stat
        
        # Verify input_value is one of the discrete values
        assert bin_stat['input_value'] in [-40, -10, 15, 25]
        assert isinstance(bin_stat['input_value'], (int, float))
        
        # Verify counts are non-negative integers
        assert isinstance(bin_stat['count'], int)
        assert bin_stat['count'] >= 0
        assert isinstance(bin_stat['bad_count'], int)
        assert bin_stat['bad_count'] >= 0
        assert 0.0 <= bin_stat['bad_rate'] <= 1.0
    
    # Verify feature_2 has unique_values (DISCRETE)
    assert 'unique_values' in feature_2
    assert feature_2['unique_values'] == [-30, -5, 10, 20]  # Sorted
    assert feature_2['num_bins'] == 4
    
    # Verify bin_stats for feature_2
    assert 'bin_stats' in feature_2
    assert isinstance(feature_2['bin_stats'], list)
    assert len(feature_2['bin_stats']) == 4
    
    for bin_stat in feature_2['bin_stats']:
        assert 'input_value' in bin_stat
        # Verify input_value is one of the discrete values
        assert bin_stat['input_value'] in [-30, -5, 10, 20]
        assert isinstance(bin_stat['input_value'], (int, float))


def test_unique_values_are_discrete_not_continuous(test_csv_data):
    """Verify that unique_values are discrete input values, NOT continuous."""
    # Upload file
    files = {
        'file': ('test_data.csv', test_csv_data, 'text/csv')
    }
    
    upload_response = client.post("/api/upload", files=files)
    assert upload_response.status_code == 200
    file_id = upload_response.json()['file_id']
    
    # Get features
    response = client.get(f"/api/upload/{file_id}/features")
    assert response.status_code == 200
    data = response.json()
    
    feature_1 = next(f for f in data['features'] if f['name'] == 'feature_1')
    feature_2 = next(f for f in data['features'] if f['name'] == 'feature_2')
    
    # Verify feature_1 unique_values are exactly the discrete input values
    expected_values_1 = sorted([-40, 25, 15, -10])
    assert feature_1['unique_values'] == expected_values_1
    
    # Verify no intermediate values (proving they're discrete, not continuous)
    # The unique_values should only contain the exact input values
    for val in feature_1['unique_values']:
        assert val in [-40, -10, 15, 25], f"Value {val} is not one of the discrete input values"
    
    # Verify feature_2 unique_values are exactly the discrete input values
    expected_values_2 = sorted([-30, 20, 10, -5])
    assert feature_2['unique_values'] == expected_values_2
    
    # Verify no intermediate values
    for val in feature_2['unique_values']:
        assert val in [-30, -5, 10, 20], f"Value {val} is not one of the discrete input values"
    
    # Verify bin_stats input_value matches unique_values exactly
    bin_input_values_1 = [b['input_value'] for b in feature_1['bin_stats']]
    assert sorted(bin_input_values_1) == expected_values_1
    
    bin_input_values_2 = [b['input_value'] for b in feature_2['bin_stats']]
    assert sorted(bin_input_values_2) == expected_values_2


def test_bin_stats_correctness(test_csv_data):
    """Verify bin_stats contain correct counts and bad rates."""
    # Upload file
    files = {
        'file': ('test_data.csv', test_csv_data, 'text/csv')
    }
    
    upload_response = client.post("/api/upload", files=files)
    assert upload_response.status_code == 200
    file_id = upload_response.json()['file_id']
    
    # Get features
    response = client.get(f"/api/upload/{file_id}/features")
    assert response.status_code == 200
    data = response.json()
    
    feature_1 = next(f for f in data['features'] if f['name'] == 'feature_1')
    
    # Verify bin_stats for feature_1
    # Expected counts:
    # -40: appears 3 times, all with target=1 (bad_count=3)
    # -10: appears 2 times, all with target=1 (bad_count=2)
    # 15: appears 2 times, all with target=0 (bad_count=0)
    # 25: appears 3 times, all with target=0 (bad_count=0)
    
    bin_stats_dict = {b['input_value']: b for b in feature_1['bin_stats']}
    
    assert bin_stats_dict[-40]['count'] == 3
    assert bin_stats_dict[-40]['bad_count'] == 3
    assert bin_stats_dict[-40]['bad_rate'] == 1.0
    
    assert bin_stats_dict[-10]['count'] == 2
    assert bin_stats_dict[-10]['bad_count'] == 2
    assert bin_stats_dict[-10]['bad_rate'] == 1.0
    
    assert bin_stats_dict[15]['count'] == 2
    assert bin_stats_dict[15]['bad_count'] == 0
    assert bin_stats_dict[15]['bad_rate'] == 0.0
    
    assert bin_stats_dict[25]['count'] == 3
    assert bin_stats_dict[25]['bad_count'] == 0
    assert bin_stats_dict[25]['bad_rate'] == 0.0


def test_upload_invalid_file():
    """Test upload with invalid file type."""
    files = {
        'file': ('test.txt', b'not a csv', 'text/plain')
    }
    
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Only CSV files are supported" in response.json()['detail']


def test_upload_missing_target_column():
    """Test upload with missing target column."""
    data = {
        'feature_1': [1, 2, 3],
        'feature_2': [4, 5, 6]
    }
    df = pd.DataFrame(data)
    csv_buffer = io.StringIO()
    df.to_csv(csv_buffer, index=False)
    csv_content = csv_buffer.getvalue().encode('utf-8')
    
    files = {
        'file': ('test_data.csv', csv_content, 'text/csv')
    }
    
    response = client.post("/api/upload", files=files)
    assert response.status_code == 400
    assert "Target column 'target' not found" in response.json()['detail']


def test_get_features_nonexistent_file():
    """Test GET /api/upload/{file_id}/features with non-existent file_id."""
    response = client.get("/api/upload/nonexistent-id/features")
    assert response.status_code == 404
    assert "File not found" in response.json()['detail']

