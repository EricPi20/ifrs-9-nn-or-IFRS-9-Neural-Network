/**
 * Training Page
 * 
 * Page for configuring and starting model training.
 */

import React, { useState, useEffect, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { TrainingConfigForm } from '../components/config/TrainingConfigForm';
import { TrainingProgress } from '../components/training/TrainingProgress';
import { TrainingConfig, UploadResponse } from '../types';
import { useApi } from '../hooks/useApi';
import { Alert } from '../components/common';
import { addResult } from '../stores/resultsStore';

interface StoredTrainingConfig {
  file_id: string;
  filename: string;
  segment: string;
  selected_features: string[];
  num_records: number;
  num_features: number;
  segment_stats: Array<{ segment: string; count: number; bad_count: number; bad_rate: number }>;
  features?: Array<{ name: string; correlation: number }>;
}

interface EpochMetric {
  epoch: number;
  train_loss: number;
  test_loss: number;
  train_ar: number;
  test_ar: number;
  train_auc: number;
  test_auc: number;
  train_ks: number;
  test_ks: number;
}

interface TrainingStatus {
  job_id: string;
  status: string;
  progress: number;
  message: string;
  current_epoch?: number;
  total_epochs?: number;
  current_metrics?: {
    train_loss: number;
    test_loss: number;
    train_ar: number;
    test_ar: number;
    train_auc: number;
    test_auc: number;
    train_ks: number;
    test_ks: number;
  };
  history?: EpochMetric[];
  error?: string;
}

interface TrainingResult {
  job_id: string;
  config: TrainingConfig | null;
  metrics?: {
    train_ar?: number;
    test_ar?: number;
    train_auc?: number;
    test_auc?: number;
    train_loss?: number;
    test_loss?: number;
    train_ks?: number;
    test_ks?: number;
  };
  completed_at: string;
}

export const Training: React.FC = () => {
  const navigate = useNavigate();
  const [jobId, setJobId] = useState<string | null>(null);
  const [trainingStatus, setTrainingStatus] = useState<TrainingStatus | null>(null);
  const [storedConfig, setStoredConfig] = useState<StoredTrainingConfig | null>(null);
  const [segments, setSegments] = useState<string[]>(['ALL']);
  const [features, setFeatures] = useState<string[]>([]);
  const [error, setError] = useState<string | null>(null);
  const { startTraining, getTrainingStatus, cancelTraining, getTrainingHistory } = useApi();
  
  // New state for config management and history
  const [lastConfig, setLastConfig] = useState<TrainingConfig | null>(null);
  const [showConfig, setShowConfig] = useState<boolean>(true);
  const [trainingHistory, setTrainingHistory] = useState<TrainingResult[]>([]);
  
  // Use ref to track completion state to prevent polling after completion
  const isCompletedRef = useRef(false);

  useEffect(() => {
    // Load training config from sessionStorage
    try {
      const stored = sessionStorage.getItem('rift_training_config');
      if (!stored) {
        setError('No training configuration found. Please upload a file first.');
        return;
      }

      const config: StoredTrainingConfig = JSON.parse(stored);
      setStoredConfig(config);
      setFeatures(config.selected_features || []);

      // Load segments from upload response
      const uploadResponseStr = sessionStorage.getItem('uploadResponse');
      if (uploadResponseStr) {
        const uploadResponse: UploadResponse = JSON.parse(uploadResponseStr);
        // Deduplicate segments: ensure 'ALL' is first, then add others if not already present
        const segmentsFromApi = uploadResponse.segments || [];
        const uniqueSegments = ['ALL', ...segmentsFromApi.filter(seg => seg !== 'ALL')];
        setSegments([...new Set(uniqueSegments)]);
      } else {
        // Fallback: use segments from stored config
        const segmentList = config.segment_stats?.map(s => s.segment) || [];
        // Deduplicate segments: ensure 'ALL' is first, then add others if not already present
        const uniqueSegments = ['ALL', ...segmentList.filter(seg => seg !== 'ALL')];
        setSegments([...new Set(uniqueSegments)]);
      }

      // Load last training config from sessionStorage if available
      const lastConfigStr = sessionStorage.getItem('rift_last_training_config');
      if (lastConfigStr) {
        try {
          const lastConfigParsed = JSON.parse(lastConfigStr);
          setLastConfig(lastConfigParsed);
        } catch (e) {
          console.warn('Failed to parse last training config:', e);
        }
      }
    } catch (e) {
      console.error('Failed to load training config:', e);
      setError('Failed to load training configuration.');
    }
  }, []);

  // Poll for training status when jobId is set
  useEffect(() => {
    if (!jobId) return;

    // Reset completion flag for new job
    isCompletedRef.current = false;

    // Only set initial status if not already set (to avoid overwriting immediate status from handleStartTraining)
    if (!trainingStatus || trainingStatus.job_id !== jobId) {
      setTrainingStatus({
        job_id: jobId,
        status: 'queued',
        progress: 0,
        message: 'Training job queued...'
      });
    }

    let isMounted = true;
    let timeoutId: NodeJS.Timeout | null = null;
    let pollCount = 0;

    const pollStatus = async () => {
      // Check ref to prevent polling after completion
      if (isCompletedRef.current || !isMounted) {
        console.log('Polling stopped: training already completed or component unmounted');
        return;
      }

      try {
        const statusResponse = await getTrainingStatus(jobId) as TrainingStatus;
        pollCount++;
        console.log(`Poll #${pollCount}:`, statusResponse.status, `Epoch ${statusResponse.current_epoch || 0}/${statusResponse.total_epochs || '?'}`, `Progress: ${statusResponse.progress}%`);
        
        if (!isMounted) return;
        
        setTrainingStatus(statusResponse);
        
        // Stop polling if completed or failed
        if (statusResponse.status === 'completed' || statusResponse.status === 'failed') {
          console.log(`Training ${statusResponse.status}, stopping polling`);
          isCompletedRef.current = true;
          
          // If completed, try to fetch history if not already included
          if (statusResponse.status === 'completed') {
            if (!statusResponse.history || statusResponse.history.length === 0) {
              try {
                const historyResponse: any = await getTrainingHistory(jobId);
                if (historyResponse && historyResponse.history) {
                  // history.history.epochs is the array of epoch metrics
                  const epochs = historyResponse.history.epochs || historyResponse.history;
                  if (Array.isArray(epochs) && epochs.length > 0) {
                    // Update status with history
                    setTrainingStatus({ ...statusResponse, history: epochs });
                  }
                }
              } catch (err) {
                console.error('Failed to fetch training history:', err);
                // History might not be available yet, that's okay
              }
            }
            
            // Save result to persistent storage
            if (lastConfig && statusResponse.current_metrics) {
              const result = {
                job_id: jobId,
                created_at: new Date().toISOString(),
                segment: lastConfig.segment || statusResponse.config?.segment || 'ALL',
                config: {
                  learning_rate: statusResponse.config?.learning_rate || lastConfig.learning_rate || 0.001,
                  hidden_layers: statusResponse.config?.network?.hidden_layers || lastConfig.network?.hidden_layers || [16, 8],
                  activation: statusResponse.config?.network?.activation || lastConfig.network?.activation || 'relu',
                  epochs: statusResponse.config?.epochs || lastConfig.epochs || 100,
                  batch_size: statusResponse.config?.batch_size || lastConfig.batch_size || 32,
                  dropout_rate: statusResponse.config?.regularization?.dropout_rate || lastConfig.network?.dropout_rate || lastConfig.regularization?.dropout_rate || 0.3,
                  l2_lambda: statusResponse.config?.regularization?.l2_lambda || lastConfig.regularization?.l2_lambda || 0.001,
                  skip_connection: statusResponse.config?.network?.skip_connection || lastConfig.network?.skip_connection || false,
                  random_seed: statusResponse.config?.random_seed || lastConfig.random_seed || 42,
                  test_size: statusResponse.config?.test_size || lastConfig.test_size || 0.25,
                  selected_features: statusResponse.config?.selected_features || lastConfig.selected_features || [],
                  stratified_split: statusResponse.config?.stratified_split ?? lastConfig.stratified_split ?? true,
                  early_stopping_enabled: statusResponse.config?.early_stopping?.enabled || lastConfig.early_stopping?.enabled || false,
                  early_stopping_patience: statusResponse.config?.early_stopping?.patience || lastConfig.early_stopping?.patience || 10,
                  loss_type: statusResponse.config?.loss?.loss_type || lastConfig.loss?.loss_type || 'combined',
                  loss_alpha: statusResponse.config?.loss?.loss_alpha ?? lastConfig.loss?.loss_alpha ?? 0.3,
                  auc_gamma: statusResponse.config?.loss?.auc_gamma ?? lastConfig.loss?.auc_gamma ?? 2.0,
                },
                metrics: {
                  test_auc: statusResponse.current_metrics?.test_auc || 0,
                  test_ar: statusResponse.current_metrics?.test_ar || 0,
                  test_ks: statusResponse.current_metrics?.test_ks || 0,
                  train_auc: statusResponse.current_metrics?.train_auc || 0,
                  train_ar: statusResponse.current_metrics?.train_ar || 0,
                  final_loss: statusResponse.current_metrics?.test_loss || 0,
                },
              };
              
              addResult(result);
              console.log('[Training] Result saved:', result.job_id);
            }
            
            // Add to training history when completed
            setTrainingHistory(prev => {
              // Check if this job is already in history
              const exists = prev.some(run => run.job_id === jobId);
              if (!exists && lastConfig) {
                return [...prev, {
                  job_id: jobId,
                  config: lastConfig,
                  metrics: statusResponse.current_metrics,
                  completed_at: new Date().toISOString(),
                }];
              }
              return prev;
            });
          }
          
          // Handle failed status
          if (statusResponse.status === 'failed' && statusResponse.error) {
            setError(statusResponse.error);
          }
          
          // Don't schedule another poll
          return;
        }
        
        // Poll faster during active training, slower when queued/preparing
        const pollInterval = statusResponse.status === 'training' ? 500 : 1000; // 500ms during training!
        
        // Continue polling only for active statuses
        if (['queued', 'preparing', 'running', 'training', 'generating_scorecard'].includes(statusResponse.status)) {
          timeoutId = setTimeout(pollStatus, pollInterval);
        }
      } catch (err: any) {
        console.error('Failed to get training status:', err);
        
        // Don't retry if already completed or component unmounted
        if (isCompletedRef.current || !isMounted) {
          console.log('Not retrying: training completed or component unmounted');
          return;
        }
        
        // If we get a 404, it might mean the job was cleared (e.g., server restart)
        // Check if we have a completed status in our state
        if (err.message && err.message.includes('404')) {
          console.warn('Received 404 for job status. Job may have been cleared.');
          // If we have a completed status, mark as completed and stop
          if (trainingStatus?.status === 'completed' || trainingStatus?.status === 'failed') {
            console.log('Job was cleared but we have final status, stopping polling');
            isCompletedRef.current = true;
            return;
          }
        }
        
        // Only retry if we don't have a completed status yet
        // Use a small retry limit to avoid infinite loops
        if (trainingStatus?.status !== 'completed' && trainingStatus?.status !== 'failed') {
          // Retry with backoff on error (max 3 retries)
          timeoutId = setTimeout(pollStatus, 3000);
        } else {
          console.log('Not retrying: status is already', trainingStatus?.status);
          isCompletedRef.current = true;
        }
      }
    };

    // Start polling immediately
    pollStatus();

    // Cleanup on unmount or when jobId changes
    return () => {
      isMounted = false;
      isCompletedRef.current = true; // Mark as completed to stop any pending polls
      if (timeoutId) {
        clearTimeout(timeoutId);
      }
    };
  }, [jobId]); // Only depend on jobId - this prevents re-triggering on status changes

  const handleStartTraining = async (config: TrainingConfig) => {
    console.log('========================================');
    console.log('[TRAINING] handleStartTraining called');
    console.log('[TRAINING] Received config from form:', config);
    console.log('[TRAINING] FileUpload segment:', storedConfig?.segment);
    console.log('[TRAINING] FileUpload features:', storedConfig?.selected_features?.length || 0);
    console.log('========================================');
    
    if (!storedConfig) {
      console.error('[TRAINING] No storedConfig found');
      setError('No file configuration found. Please upload a file first.');
      return;
    }

    // USE THE SEGMENT AND FEATURES FROM FILE UPLOAD, NOT FROM FORM
    const correctSegment = storedConfig.segment || 'ALL';
    const correctFeatures = storedConfig.selected_features || [];
    
    if (!correctFeatures.length) {
      setError('No features selected. Please select features from the uploaded file.');
      return;
    }
    
    console.log('[TRAINING] Using segment from FileUpload:', correctSegment);
    console.log('[TRAINING] Using features from FileUpload:', correctFeatures);
    console.log('========================================');

    // Save config for later reuse (but override segment and features with correct ones from FileUpload)
    const configWithCorrectValues = {
      ...config,
      segment: correctSegment,  // FROM FILE UPLOAD
      selected_features: correctFeatures,  // FROM FILE UPLOAD
    };
    setLastConfig(configWithCorrectValues);
    
    // Also save to sessionStorage for persistence across page refresh
    sessionStorage.setItem('rift_last_training_config', JSON.stringify(configWithCorrectValues));
    
    // Hide config, show progress
    setShowConfig(false);

    console.log('[TRAINING] Starting training with file_id:', storedConfig.file_id);
    console.log('[TRAINING] Passing segment to API:', correctSegment);
    console.log('[TRAINING] Passing features to API:', correctFeatures.length, 'features');
    
    try {
      setError(null);
      setTrainingStatus(null); // Reset status for new training
      isCompletedRef.current = false; // Reset completion flag for new training
      console.log('[TRAINING] Calling startTraining API...');
      
      // Use the segment and features from FileUpload (storedConfig), not from form
      const trainingData = {
        ...config,
        segment: correctSegment,  // FROM FILE UPLOAD
        selected_features: correctFeatures,  // FROM FILE UPLOAD
      };
      
      console.log('[TRAINING] Training data being sent:', {
        segment: trainingData.segment,
        selected_features: trainingData.selected_features?.length || 0,
        epochs: trainingData.epochs,
      });
      console.log('[TRAINING] Full request data:', JSON.stringify(trainingData, null, 2));
      
      const response = await startTraining(storedConfig.file_id, trainingData);
      console.log('[TRAINING] Training started, job_id:', response.job_id);
      
      // Set job ID - this triggers the polling useEffect
      setJobId(response.job_id);
      
      // ALSO set initial status immediately so UI shows right away
      setTrainingStatus({
        job_id: response.job_id,
        status: 'queued',
        progress: 0,
        current_epoch: 0,
        total_epochs: config.epochs,
        history: [],
        current_metrics: null,
        message: 'Starting training...',
      });
    } catch (err: any) {
      console.error('[TRAINING] Training failed:', err);
      // Handle object errors properly
      const errorMessage = typeof err === 'object' 
        ? (err.detail || err.message || JSON.stringify(err))
        : String(err);
      setError(errorMessage || 'Failed to start training');
      setShowConfig(true); // Show config again on error
    }
  };

  const handleModifyAndRetrain = () => {
    setShowConfig(true);
    setJobId(null);
    setTrainingStatus(null);
    isCompletedRef.current = false;
    // lastConfig is preserved, will be passed to form
  };

  const handleTrainNewModel = () => {
    setShowConfig(true);
    setJobId(null);
    setTrainingStatus(null);
    isCompletedRef.current = false;
    setLastConfig(null); // Clear config for fresh start
    sessionStorage.removeItem('rift_last_training_config');
  };

  const handleCancelTraining = async () => {
    if (!jobId) return;
    
    try {
      setError(null);
      await cancelTraining(jobId);
      // Status will update on next poll
    } catch (err: any) {
      console.error('Failed to cancel training:', err);
      const errorMessage = typeof err === 'object' 
        ? (err.detail || err.message || JSON.stringify(err))
        : String(err);
      setError(errorMessage || 'Failed to cancel training');
    }
  };

  if (error && !storedConfig) {
    return (
      <div className="container mx-auto px-4 py-8">
        <h1 className="text-3xl font-bold mb-8">Train Model</h1>
        <div className="max-w-4xl mx-auto">
          <Alert variant="error" title="Configuration Error">
            {error}
            <div className="mt-4">
              <button
                onClick={() => navigate('/upload')}
                className="px-4 py-2 bg-[#1E3A5F] text-white rounded-lg hover:bg-[#2D4A6F]"
              >
                Go to Upload Page
              </button>
            </div>
          </Alert>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Train Model</h1>
      
      {error && (
        <div className="max-w-4xl mx-auto mb-6">
          <Alert variant="error" onClose={() => setError(null)}>
            {typeof error === 'object' ? JSON.stringify(error) : error}
          </Alert>
        </div>
      )}

      <div className="max-w-4xl mx-auto space-y-6">
        {/* Configuration Form */}
        {showConfig && storedConfig ? (
          <TrainingConfigForm
            defaultValues={lastConfig || undefined}
            uploadedFilePath={`data/uploads/${storedConfig.file_id}.csv`}
            selectedFeatures={storedConfig.selected_features || []}
            segment={storedConfig.segment || 'ALL'}
            onSubmit={handleStartTraining}
          />
        ) : showConfig ? (
          <div>Loading configuration...</div>
        ) : null}

        {/* Training Progress */}
        {!showConfig && (jobId || trainingStatus) && (
          <>
            {trainingStatus ? (
              <TrainingProgress
                jobId={trainingStatus.job_id}
                status={trainingStatus.status}
                progress={trainingStatus.progress}
                message={trainingStatus.message || 'Initializing training...'}
                currentEpoch={trainingStatus.current_epoch}
                totalEpochs={trainingStatus.total_epochs}
                currentMetrics={trainingStatus.current_metrics}
                history={trainingStatus.history}
                error={trainingStatus.error}
                onCancel={handleCancelTraining}
                onModifyAndRetrain={handleModifyAndRetrain}
                onTrainNewModel={handleTrainNewModel}
              />
            ) : (
              <TrainingProgress
                jobId={jobId!}
                status="queued"
                progress={0}
                message="Initializing training..."
              />
            )}

            {/* Completion Actions */}
            {trainingStatus?.status === 'completed' && (
              <div className="bg-green-50 border border-green-200 rounded-xl p-6">
                <div className="flex items-center gap-3 mb-4">
                  <div className="w-10 h-10 bg-green-100 rounded-full flex items-center justify-center">
                    <span className="text-green-600 text-xl">✓</span>
                  </div>
                  <div>
                    <h3 className="text-lg font-semibold text-green-800">
                      Training Completed!
                    </h3>
                    <p className="text-green-700 text-sm">
                      Test AR: {trainingStatus.current_metrics?.test_ar?.toFixed(4) ?? 'N/A'} | 
                      Test AUC: {trainingStatus.current_metrics?.test_auc?.toFixed(4) ?? 'N/A'}
                    </p>
                    <p className="text-gray-500 text-xs mt-1">
                      Random Seed: <code className="bg-gray-100 px-1 rounded">{lastConfig?.random_seed ?? 42}</code>
                      <span className="ml-2">(Use same seed to reproduce results)</span>
                    </p>
                  </div>
                </div>
                
                {/* Action Buttons */}
                <div className="flex flex-wrap gap-3 mt-4">
                  <button
                    onClick={() => navigate(`/results/${jobId}/scorecard`)}
                    className="flex-1 bg-[#1E3A5F] text-white py-3 px-6 rounded-lg 
                               font-semibold hover:bg-[#2D4A6F] transition-colors"
                  >
                    View Scorecard & Results →
                  </button>
                  
                  <button
                    onClick={handleModifyAndRetrain}
                    className="flex-1 bg-[#38B2AC] text-white py-3 px-6 rounded-lg 
                               font-semibold hover:bg-[#2D9A94] transition-colors"
                  >
                    🔄 Modify & Retrain
                  </button>
                  
                  <button
                    onClick={handleTrainNewModel}
                    className="px-6 py-3 border border-gray-300 rounded-lg 
                               font-medium hover:bg-gray-50 transition-colors"
                  >
                    Start Fresh
                  </button>
                </div>
              </div>
            )}

            {/* Failed - Allow retry */}
            {trainingStatus?.status === 'failed' && (
              <div className="bg-red-50 border border-red-200 rounded-xl p-6">
                <h3 className="text-lg font-semibold text-red-800">Training Failed</h3>
                <p className="text-red-700 text-sm mt-1">{trainingStatus.error || 'An error occurred during training'}</p>
                
                <div className="flex gap-3 mt-4">
                  <button
                    onClick={handleModifyAndRetrain}
                    className="bg-[#1E3A5F] text-white py-2 px-6 rounded-lg font-medium hover:bg-[#2D4A6F] transition-colors"
                  >
                    ← Back to Configuration
                  </button>
                </div>
              </div>
            )}
          </>
        )}

        {/* Training History Comparison */}
        {trainingHistory.length > 1 && (
          <div className="bg-white rounded-xl shadow-md p-6 mt-6">
            <h3 className="text-lg font-semibold text-[#1E3A5F] mb-4">
              Training Run Comparison
            </h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b">
                    <th className="text-left py-2 px-3">Run</th>
                    <th className="text-left py-2 px-3">Seed</th>
                    <th className="text-left py-2 px-3">Config</th>
                    <th className="text-right py-2 px-3">Test AR</th>
                    <th className="text-right py-2 px-3">Test AUC</th>
                    <th className="text-right py-2 px-3">Actions</th>
                  </tr>
                </thead>
                <tbody>
                  {trainingHistory.map((run, idx) => (
                    <tr key={run.job_id} className="border-b hover:bg-gray-50">
                      <td className="py-2 px-3">#{idx + 1}</td>
                      <td className="py-2 px-3 font-mono text-xs text-gray-600">
                        {run.config?.random_seed ?? 42}
                      </td>
                      <td className="py-2 px-3 text-xs text-gray-500">
                        LR: {run.config?.learning_rate?.toFixed(4) ?? 'N/A'}, 
                        Layers: {run.config?.network?.hidden_layers?.length ?? 0}
                      </td>
                      <td className="py-2 px-3 text-right font-mono">
                        {run.metrics?.test_ar?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-2 px-3 text-right font-mono">
                        {run.metrics?.test_auc?.toFixed(4) ?? 'N/A'}
                      </td>
                      <td className="py-2 px-3 text-right">
                        <button
                          onClick={() => navigate(`/results/${run.job_id}/scorecard`)}
                          className="text-[#38B2AC] hover:underline text-xs"
                        >
                          View
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

