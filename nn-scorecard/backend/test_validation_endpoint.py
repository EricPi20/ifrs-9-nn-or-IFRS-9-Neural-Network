"""
Simple test script to verify the validation endpoint works.
Run this after starting the backend server.
"""

import requests
import json

BASE_URL = "http://localhost:8000"

def test_validation_endpoint():
    """Test the validation endpoint."""
    
    print("=" * 60)
    print("Testing Validation Endpoint")
    print("=" * 60)
    
    # Test 1: Test endpoint (should always work)
    print("\n1. Testing /api/training/test/validation endpoint...")
    try:
        response = requests.get(f"{BASE_URL}/api/training/test/validation")
        print(f"   Status Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"   Status: {data.get('status')}")
            print(f"   Message: {data.get('message')}")
            if 'sample' in data:
                print(f"   Sample metrics: {data['sample'].get('metrics')}")
                print(f"   Histogram bins: {data['sample'].get('num_histogram_bins')}")
                print(f"   CAP curve points: {data['sample'].get('num_cap_points')}")
                print(f"   Score bands: {data['sample'].get('num_score_bands')}")
            print("   ✓ Test endpoint works!")
        else:
            print(f"   ✗ Failed with status {response.status_code}")
            print(f"   Response: {response.text}")
    except requests.exceptions.ConnectionError:
        print("   ✗ Cannot connect to server. Is the backend running?")
        print("   Start it with: uvicorn app.main:app --reload --port 8000")
    except Exception as e:
        print(f"   ✗ Error: {str(e)}")
    
    # Test 2: Try with a real job ID (if available)
    print("\n2. Testing /api/training/{job_id}/validation endpoint...")
    print("   (This requires a completed training job)")
    print("   To test with a real job, replace JOB_ID below with an actual job ID")
    
    # Uncomment and replace with actual job ID to test:
    # job_id = "your-job-id-here"
    # try:
    #     response = requests.get(f"{BASE_URL}/api/training/{job_id}/validation")
    #     print(f"   Status Code: {response.status_code}")
    #     if response.status_code == 200:
    #         data = response.json()
    #         print(f"   ✓ Validation endpoint works!")
    #         print(f"   Metrics: {data.get('metrics')}")
    #     else:
    #         print(f"   ✗ Failed: {response.text}")
    # except Exception as e:
    #     print(f"   ✗ Error: {str(e)}")
    
    print("\n" + "=" * 60)
    print("Test Complete")
    print("=" * 60)

if __name__ == "__main__":
    test_validation_endpoint()

