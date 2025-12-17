/**
 * API Service
 * 
 * Service for making HTTP requests to the backend API.
 */

import { TrainingConfig, ModelResults, UploadResponse, TrainingResponse, FeatureInfo } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';

class ApiService {
  private async request<T>(
    endpoint: string,
    options: RequestInit = {}
  ): Promise<T> {
    const url = `${API_BASE_URL}${endpoint}`;
    const response = await fetch(url, {
      ...options,
      headers: {
        'Content-Type': 'application/json',
        ...options.headers,
      },
    });

    if (!response.ok) {
      const errorData = await response.json().catch(() => ({}));
      console.error('API Error:', response.status, errorData);
      
      // Extract error message from FastAPI response
      let errorMessage: string;
      if (errorData.detail) {
        if (Array.isArray(errorData.detail)) {
          // FastAPI validation errors are arrays
          errorMessage = errorData.detail.map((d: any) => 
            `${d.loc?.join('.') || 'field'}: ${d.msg || JSON.stringify(d)}`
          ).join(', ');
        } else {
          errorMessage = String(errorData.detail);
        }
      } else {
        errorMessage = `HTTP error! status: ${response.status}`;
      }
      
      const error = new Error(errorMessage);
      (error as any).detail = errorData.detail;
      throw error;
    }

    return response.json();
  }

  async uploadFile(file: File): Promise<UploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    console.log('📤 Uploading file to:', `${API_BASE_URL}/api/upload/`);

    let response: Response;
    try {
      response = await fetch(`${API_BASE_URL}/api/upload/`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
        const errorMessage = error.detail || `Upload failed with status ${response.status}`;
        console.error('❌ Upload error:', errorMessage);
        throw new Error(errorMessage);
      }
    } catch (err) {
      if (err instanceof TypeError && err.message.includes('fetch')) {
        console.error('❌ Network error - Backend not reachable:', API_BASE_URL);
        throw new Error(
          `Failed to connect to backend at ${API_BASE_URL}. ` +
          `Make sure the backend is running on port 8000. ` +
          `Check: http://localhost:8000/health`
        );
      }
      throw err;
    }

    const uploadData = await response.json();
    console.log('📤 Upload response from backend:', uploadData);
    console.log('📤 Has segment_stats?', !!uploadData.segment_stats);
    console.log('📤 segment_stats length:', uploadData.segment_stats?.length);
    
    // ALWAYS try to fetch segment stats if missing or empty
    const needsStats = !uploadData.segment_stats || 
                       uploadData.segment_stats.length === 0 ||
                       (uploadData.segment_stats.length > 0 && uploadData.segment_stats.every((s: any) => !s.count || s.count === 0));
    
    if (uploadData.file_id && needsStats) {
      try {
        console.log('🔍 Fetching segment stats from API for file_id:', uploadData.file_id);
        const segmentsResponse = await fetch(`${API_BASE_URL}/api/upload/${uploadData.file_id}/segments`);
        console.log('📡 Segments response status:', segmentsResponse.status);
        
        if (segmentsResponse.ok) {
          const segmentsData = await segmentsResponse.json();
          console.log('📊 Segment stats response:', segmentsData);
          console.log('📊 Has segments array?', !!segmentsData.segments);
          console.log('📊 Segments array length:', segmentsData.segments?.length);
          
          if (segmentsData.segments && Array.isArray(segmentsData.segments) && segmentsData.segments.length > 0) {
            uploadData.segment_stats = segmentsData.segments;
            console.log('✅ Added segment_stats to uploadData:', uploadData.segment_stats);
          } else {
            console.warn('⚠️ Segment stats array is missing or empty');
          }
        } else {
          const errorText = await segmentsResponse.text();
          console.warn('❌ Failed to fetch segment stats, status:', segmentsResponse.status, 'error:', errorText);
        }
      } catch (e) {
        console.error('❌ Error fetching segment stats:', e);
      }
    } else {
      console.log('ℹ️ Segment stats already present in response:', uploadData.segment_stats);
    }
    
    console.log('📤 Final uploadData before return:', {
      file_id: uploadData.file_id,
      hasSegmentStats: !!uploadData.segment_stats,
      segmentStatsLength: uploadData.segment_stats?.length,
      segmentStats: uploadData.segment_stats
    });
    
    return uploadData;
  }
  
  async getSegmentStats(fileId: string) {
    const response = await fetch(`${API_BASE_URL}/api/upload/${fileId}/segments`, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get segment stats' }));
      throw new Error(error.detail || 'Failed to get segment stats');
    }

    return response.json();
  }

  async getFeatures(fileId: string, segment?: string): Promise<{ segment: string; num_records: number; features: FeatureInfo[] }> {
    const segmentParam = segment && segment !== 'ALL' ? `?segment=${encodeURIComponent(segment)}` : '';
    const response = await fetch(`${API_BASE_URL}/api/upload/${fileId}/features${segmentParam}`, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Failed to get features' }));
      throw new Error(error.detail || 'Failed to get features');
    }

    return response.json();
  }

  async startTraining(
    fileId: string,
    config: TrainingConfig
  ): Promise<TrainingResponse> {
    console.log('='.repeat(50));
    console.log('[API] startTraining received:', config);
    console.log('[API] Segment in data:', config.segment);
    console.log('='.repeat(50));
    
    // Extract segment directly from config
    const segment = config.segment || 'ALL';
    
    // Transform config to match backend schema
    // Backend expects file_path instead of file_id, and specific config structure
    // Construct file_path from file_id (pattern: data/uploads/{file_id}.csv)
    // Backend UPLOAD_DIR defaults to ./data/uploads, so relative path should work
    const file_path = `data/uploads/${fileId}.csv`;
    
    // Transform config to ensure it matches backend TrainingConfig schema
    // Backend router expects: segment, selected_features, test_size, stratified_split,
    // epochs, batch_size, learning_rate, random_seed, network, regularization, early_stopping,
    // loss_function, use_class_weights
    const transformedConfig = {
      segment: segment,  // CRITICAL: Include segment at top level of config
      selected_features: Array.isArray(config.selected_features) && config.selected_features.length > 0 
        ? config.selected_features 
        : [],
      test_size: config.test_size ?? 0.3,
      stratified_split: true, // Backend expects this field
      epochs: config.epochs || 100,
      batch_size: config.batch_size || 32,
      learning_rate: config.learning_rate || 0.001,
      random_seed: config.random_seed || 42,
      loss_function: config.loss?.loss_type || 'bce',  // Map loss.loss_type to loss_function
      use_class_weights: config.use_class_weights ?? false,
      network: {
        model_type: config.network?.model_type || 'neural_network',
        hidden_layers: config.network?.hidden_layers || [16, 8],
        activation: config.network?.activation || 'relu',
        skip_connection: config.network?.skip_connection ?? false,
      },
      regularization: {
        dropout_rate: config.network?.dropout_rate ?? ((config.regularization as Record<string, any>)?.dropout_rate) ?? 0.3,
        l1_lambda: config.regularization?.l1_lambda ?? 0.0,
        l2_lambda: config.regularization?.l2_lambda ?? 0.001,
      } as any,
      early_stopping: {
        enabled: config.early_stopping?.enabled ?? false,
        patience: config.early_stopping?.patience || 10,
        min_delta: config.early_stopping?.min_delta ?? 0.001,
      },
    };
    
    console.log('[API] Loss function:', transformedConfig.loss_function);
    console.log('[API] Use class weights:', transformedConfig.use_class_weights);
    
    // Build request body matching backend schema exactly
    const requestBody = {
      file_path: file_path,
      config: transformedConfig,
    };
    
    console.log('[API] Final request body:', JSON.stringify(requestBody, null, 2));
    console.log('[API] Segment in request:', requestBody.config.segment);
    console.log('='.repeat(50));
    
    // Pass data directly - don't transform!
    return this.request<TrainingResponse>('/api/training', {
      method: 'POST',
      body: JSON.stringify(requestBody),
    });
  }

  async getTrainingStatus(jobId: string) {
    return this.request(`/api/training/${jobId}/status`, {
      method: 'GET',
    });
  }

  async cancelTraining(jobId: string) {
    return this.request(`/api/training/${jobId}/cancel`, {
      method: 'POST',
    });
  }

  async getTrainingHistory(jobId: string) {
    return this.request(`/api/training/${jobId}/history`, {
      method: 'GET',
    });
  }

  async getModelResults(jobId: string): Promise<ModelResults> {
    return this.request<ModelResults>(`/api/results/${jobId}/metrics`, {
      method: 'GET',
    });
  }

  async getScorecard(jobId: string): Promise<{ scorecard: any; score_range: { min: number; max: number; interpretation?: string } }> {
    console.log('API: Getting scorecard for job:', jobId);
    // Use /training/ path instead of /results/
    const response = await this.request<{ scorecard: any; score_range: { min: number; max: number; interpretation?: string } }>(`/api/training/${jobId}/scorecard`, {
      method: 'GET',
    });
    console.log('API: Scorecard response:', response);
    return response;
  }

  async getCompleteResults(jobId: string): Promise<any> {
    return this.request(`/api/results/${jobId}`, {
      method: 'GET',
    });
  }

  async generateScorecard(jobId: string) {
    return this.request(`/api/training/${jobId}/scorecard`, {
      method: 'POST',
    });
  }

  async calculateScore(jobId: string, features: Record<string, number>) {
    return this.request('/api/scoring/calculate', {
      method: 'POST',
      body: JSON.stringify({
        job_id: jobId,
        features,
      }),
    });
  }

  async getValidationMetrics(jobId: string) {
    console.log('API: Getting validation metrics for job:', jobId);
    const response = await this.request(`/api/training/${jobId}/validation`, {
      method: 'GET',
    });
    return response;
  }

  async uploadOutOfTimeValidation(jobId: string, file: File) {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${API_BASE_URL}/api/training/${jobId}/out-of-time-validation`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Upload failed' }));
      throw new Error(error.detail || 'Upload failed');
    }

    return response.json();
  }

  async getOutOfTimeValidation(jobId: string) {
    return this.request(`/api/training/${jobId}/out-of-time-validation`, {
      method: 'GET',
    });
  }

  async downloadScorecardCSV(jobId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/api/results/${jobId}/download-scorecard-csv`;
    const response = await fetch(url, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Download failed' }));
      throw new Error(error.detail || 'Download failed');
    }

    return response.blob();
  }

  async downloadConfigCSV(jobId: string): Promise<Blob> {
    const url = `${API_BASE_URL}/api/results/${jobId}/download-config-csv`;
    const response = await fetch(url, {
      method: 'GET',
    });

    if (!response.ok) {
      const error = await response.json().catch(() => ({ detail: 'Download failed' }));
      throw new Error(error.detail || 'Download failed');
    }

    return response.blob();
  }
}

export const api = new ApiService();

