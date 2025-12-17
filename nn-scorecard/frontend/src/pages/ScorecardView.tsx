// frontend/src/pages/ScorecardView.tsx

import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Download } from 'lucide-react';
import { api } from '../services/api';
import ScorecardDisplay from '../components/results/ScorecardDisplay';
import ValidationMetrics from '../components/results/ValidationMetrics';
import OutOfTimeValidation from '../components/results/OutOfTimeValidation';
import { Scorecard } from '../types';

// Transformation function to convert raw weights to percentages and translate points to 0-100 range
const transformScorecard = (rawScorecard: any): Scorecard => {
  console.log('[TRANSFORM] Raw scorecard:', rawScorecard);
  
  const features = rawScorecard.features || rawScorecard.scorecard?.features || [];
  
  // STEP 1: Calculate weight percentages
  const rawWeights = features.map((f: any) => Math.abs(f.weight || f.weight_normalized || 0));
  const totalWeight = rawWeights.reduce((sum: number, w: number) => sum + w, 0) || 1;
  
  console.log('[TRANSFORM] Raw weights:', rawWeights);
  console.log('[TRANSFORM] Total weight:', totalWeight);
  
  // STEP 2: Transform each feature
  const transformedFeatures = features.map((f: any, index: number) => {
    const rawWeight = Math.abs(f.weight || f.weight_normalized || 0);
    const weightPercent = (rawWeight / totalWeight) * 100;
    
    // Get bin points for this feature
    const bins = f.bins || [];
    const binPoints = bins.map((b: any) => b.scaled_points || b.raw_points || 0);
    const featureMin = binPoints.length > 0 ? Math.min(...binPoints) : 0;
    const featureMax = binPoints.length > 0 ? Math.max(...binPoints) : 0;
    const featureRange = featureMax - featureMin || 1;
    
    // Transform bins - translate to positive range (0 to weightPercent)
    // This ensures all points are positive and each feature contributes up to its weight percentage
    const transformedBins = bins.map((b: any, binIndex: number) => {
      const rawPoints = b.scaled_points || b.raw_points || 0;
      
      // Normalize within feature's range (0 to 1), then scale by weight percentage
      // This maps the feature's raw point range to [0, weightPercent]
      const normalized = featureRange !== 0 ? (rawPoints - featureMin) / featureRange : 0.5;
      const scaledPoints = Math.round(normalized * weightPercent);
      
      return {
        ...b,
        bin_index: binIndex,
        scaled_points: Math.max(0, scaledPoints), // Ensure non-negative
        raw_points: rawPoints,
        // Preserve count and bad_rate properties if they exist
        count_train: b.count_train,
        count_test: b.count_test,
        bad_rate_train: b.bad_rate_train,
        bad_rate_test: b.bad_rate_test,
      };
    });
    
    const scaledBinPoints = transformedBins.map((b: any) => b.scaled_points);
    
    return {
      feature_name: f.feature_name || f.name || `feature_${index}`,
      weight: Math.round(weightPercent),  // Percentage rounded to whole number
      weight_normalized: rawWeight,
      importance_rank: index + 1,
      min_points: scaledBinPoints.length > 0 ? Math.min(...scaledBinPoints) : 0,
      max_points: scaledBinPoints.length > 0 ? Math.max(...scaledBinPoints) : 0,
      bins: transformedBins,
    };
  });
  
  // Sort by weight percentage (highest first)
  transformedFeatures.sort((a: any, b: any) => b.weight - a.weight);
  
  // Re-assign importance ranks after sorting
  transformedFeatures.forEach((f: any, i: number) => {
    f.importance_rank = i + 1;
  });
  
  // Verify sum
  const totalPercent = transformedFeatures.reduce((sum: number, f: any) => sum + f.weight, 0);
  console.log('[TRANSFORM] Total weight percent:', totalPercent, '(should be ~100)');
  
  // Adjust last feature to make exactly 100% if needed
  if (Math.abs(totalPercent - 100) > 0.5 && transformedFeatures.length > 0) {
    const diff = 100 - totalPercent;
    transformedFeatures[transformedFeatures.length - 1].weight += diff;
  }
  
  // Calculate total min/max points (should be 0 to ~100)
  const totalMinPoints = transformedFeatures.reduce((sum: number, f: any) => sum + f.min_points, 0);
  const totalMaxPoints = transformedFeatures.reduce((sum: number, f: any) => sum + f.max_points, 0);
  
  console.log('[TRANSFORM] Transformed features:', transformedFeatures);
  console.log('[TRANSFORM] Score range:', totalMinPoints, 'to', totalMaxPoints);
  
  // Get metrics
  const metrics = rawScorecard.metrics || rawScorecard.scorecard?.metrics || {
    train_auc: 0.5, test_auc: 0.5, train_ar: 0, test_ar: 0, train_ks: 0, test_ks: 0
  };
  
  // Calculate raw score range for reference (sum of all feature raw min/max)
  let globalMinRaw = 0;
  let globalMaxRaw = 0;
  features.forEach((f: any) => {
    const binPoints = (f.bins || []).map((b: any) => b.scaled_points || b.raw_points || 0);
    if (binPoints.length > 0) {
      globalMinRaw = Math.min(globalMinRaw, ...binPoints);
      globalMaxRaw = Math.max(globalMaxRaw, ...binPoints);
    }
  });
  const rawRange = globalMaxRaw - globalMinRaw || 1;
  
  return {
    job_id: rawScorecard.model_id || rawScorecard.job_id || '',
    segment: rawScorecard.segment || 'ALL',
    model_type: 'neural_network',
    created_at: rawScorecard.created_at || new Date().toISOString(),
    
    score_min: 0,
    score_max: 100,
    min_possible_score: Math.round(totalMinPoints),
    max_possible_score: Math.round(totalMaxPoints),
    
    raw_min: globalMinRaw,
    raw_max: globalMaxRaw,
    scale_factor: 100 / rawRange,
    offset: 0,
    input_scale_factor: 50,
    
    features: transformedFeatures,
    metrics: metrics,
    data_stats: rawScorecard.data_stats || rawScorecard.scorecard?.data_stats,
  };
};

const ScorecardView: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [scorecard, setScorecard] = useState<Scorecard | null>(null);
  const [validation, setValidation] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeTab, setActiveTab] = useState<'scorecard' | 'validation' | 'out-of-time'>('scorecard');
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchData = async () => {
      if (!jobId) return;
      
      setLoading(true);
      setError(null);
      
      try {
        console.log('[SCORECARD VIEW] Fetching scorecard for jobId:', jobId);
        const scorecardData = await api.getScorecard(jobId);
        console.log('[SCORECARD VIEW] Scorecard response:', scorecardData);
        
        if (!scorecardData) {
          throw new Error('No scorecard data received from API');
        }
        
        // Handle nested structure - response might be {scorecard: {...}} or direct
        const rawScorecard = scorecardData.scorecard || scorecardData;
        console.log('[SCORECARD VIEW] Raw scorecard:', rawScorecard);
        
        if (!rawScorecard || !rawScorecard.features) {
          throw new Error('Invalid scorecard data: missing features');
        }
        
        // Fetch complete results to get metrics and other metadata
        try {
          const resultsResponse = await api.getCompleteResults(jobId);
          console.log('Results response:', resultsResponse);
          
          // Extract metrics - prefer scorecard metrics if available and valid
          let metrics = rawScorecard.metrics || {
            train_auc: 0,
            test_auc: 0,
            train_ar: 0,
            test_ar: 0,
            train_ks: 0,
            test_ks: 0,
          };
          
          // If scorecard metrics are zeros or missing, try to get from history
          if (!metrics.train_auc || metrics.train_auc === 0) {
            if (resultsResponse.history && resultsResponse.history.epochs && resultsResponse.history.epochs.length > 0) {
              // Get metrics from best epoch or last epoch
              const bestEpochIdx = resultsResponse.history.best_epoch 
                ? Math.min(resultsResponse.history.best_epoch - 1, resultsResponse.history.epochs.length - 1)
                : resultsResponse.history.epochs.length - 1;
              const bestEpoch = resultsResponse.history.epochs[bestEpochIdx];
              
              if (bestEpoch) {
                metrics = {
                  train_auc: bestEpoch.train_auc || 0,
                  test_auc: bestEpoch.test_auc || 0,
                  train_ar: bestEpoch.train_ar || 0,
                  test_ar: bestEpoch.test_ar || 0,
                  train_ks: bestEpoch.train_ks || 0,
                  test_ks: bestEpoch.test_ks || 0,
                };
              }
            } else if (resultsResponse.metrics) {
              // Fallback to test metrics only
              metrics = {
                train_auc: resultsResponse.metrics.auc_roc || 0,
                test_auc: resultsResponse.metrics.auc_roc || 0,
                train_ar: resultsResponse.metrics.gini_ar || 0,
                test_ar: resultsResponse.metrics.gini_ar || 0,
                train_ks: resultsResponse.metrics.ks_statistic || 0,
                test_ks: resultsResponse.metrics.ks_statistic || 0,
              };
            }
          }
          
          console.log('[SCORECARD VIEW] Final metrics:', metrics);
          
          // TRANSFORM the scorecard - convert raw weights to percentages and translate points
          const transformedScorecard = transformScorecard({
            ...rawScorecard,
            model_id: resultsResponse.job_id || jobId,
            segment: resultsResponse.segment || rawScorecard.segment,
            metrics: metrics,
            created_at: resultsResponse.created_at || rawScorecard.created_at,
          });
          
          console.log('Transformed scorecard:', transformedScorecard);
          
          setScorecard(transformedScorecard);
        } catch (resultsErr) {
          console.warn('Failed to fetch complete results, using scorecard only:', resultsErr);
          // TRANSFORM the scorecard even if we can't get full results
          const transformedScorecard = transformScorecard({
            ...rawScorecard,
            model_id: jobId,
            segment: rawScorecard.segment || 'ALL',
            metrics: {
              train_auc: 0,
              test_auc: 0,
              train_ar: 0,
              test_ar: 0,
              train_ks: 0,
              test_ks: 0,
            },
            created_at: rawScorecard.created_at || new Date().toISOString(),
          });
          
          setScorecard(transformedScorecard);
        }
        
        // Try to get validation metrics
        try {
          console.log('[SCORECARD VIEW] Fetching validation metrics for jobId:', jobId);
          const validationData = await api.getValidationMetrics(jobId);
          console.log('[SCORECARD VIEW] Validation metrics response:', validationData);
          setValidation(validationData);
        } catch (e: any) {
          console.warn('[SCORECARD VIEW] Validation metrics not available:', e.message || e);
          // Don't set error state for validation - it's optional
        }
      } catch (err: any) {
        console.error('[SCORECARD VIEW] Error loading scorecard:', err);
        const errorMessage = err.message || err.detail || 'Failed to load scorecard';
        setError(errorMessage);
        // If it's a 404, provide helpful message
        if (errorMessage.includes('404') || errorMessage.includes('not found')) {
          setError('Training job not found. The backend may have restarted. Please check if the training completed successfully.');
        }
      } finally {
        setLoading(false);
      }
    };
    
    fetchData();
  }, [jobId]);

  const handleDownloadCSV = async () => {
    if (!jobId) return;
    
    setDownloading(true);
    try {
      const blob = await api.downloadScorecardCSV(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `scorecard_${jobId}.csv`;
      document.body.appendChild(a);
      a.click();
      window.URL.revokeObjectURL(url);
      document.body.removeChild(a);
    } catch (err: any) {
      console.error('Failed to download CSV:', err);
      alert(`Failed to download CSV: ${err.message || 'Unknown error'}`);
    } finally {
      setDownloading(false);
    }
  };

  if (error) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-red-50 border border-red-200 rounded-lg p-6">
          <h3 className="text-red-800 font-semibold mb-2">Error Loading Scorecard</h3>
          <p className="text-red-700 mb-4">{error}</p>
          <Link
            to="/results"
            className="inline-flex items-center gap-2 text-red-600 hover:text-red-800"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Results
          </Link>
        </div>
      </div>
    );
  }

  return (
    <div className="container mx-auto px-4 py-8">
      {/* Header */}
      <div className="flex items-center gap-4 mb-6">
        <Link
          to="/results"
          className="p-2 hover:bg-gray-100 rounded-lg transition-colors"
        >
          <ArrowLeft className="w-5 h-5 text-gray-600" />
        </Link>
        <div className="flex-1">
          <h1 className="text-2xl font-bold text-gray-800">Credit Scorecard</h1>
          <p className="text-sm text-gray-500">
            Training ID: <code className="bg-gray-100 px-2 py-0.5 rounded">{jobId}</code>
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleDownloadCSV}
            disabled={downloading || !scorecard}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            {downloading ? 'Downloading...' : 'Download CSV'}
          </button>
          <Link
            to={`/results/${jobId}/config`}
            className="px-4 py-2 bg-gray-100 text-gray-700 rounded-lg hover:bg-gray-200 text-sm"
          >
            View Config
          </Link>
        </div>
      </div>

      {/* Tab Navigation - ALWAYS VISIBLE */}
      <div className="flex gap-1 mb-6 bg-gray-100 p-1 rounded-lg w-fit">
        <button
          onClick={() => setActiveTab('scorecard')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'scorecard'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          📊 Credit Scorecard
        </button>
        <button
          onClick={() => setActiveTab('validation')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'validation'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          📈 Validation Metrics
        </button>
        <button
          onClick={() => setActiveTab('out-of-time')}
          className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
            activeTab === 'out-of-time'
              ? 'bg-white text-blue-600 shadow-sm'
              : 'text-gray-600 hover:text-gray-800'
          }`}
        >
          🔍 Out-of-Time Validation
        </button>
      </div>

      {/* Content */}
      {loading ? (
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      ) : (
        <>
          {activeTab === 'scorecard' && (
            scorecard ? (
              <ScorecardDisplay scorecard={scorecard} />
            ) : (
              <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6 text-center">
                <p className="text-yellow-700">Scorecard data is not available for this training run.</p>
              </div>
            )
          )}
          
          {activeTab === 'validation' && (
            validation ? (
              <ValidationMetrics jobId={jobId!} />
            ) : (
              <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
                Validation metrics not available for this training run.
              </div>
            )
          )}
          
          {activeTab === 'out-of-time' && (
            <OutOfTimeValidation jobId={jobId!} />
          )}
        </>
      )}
    </div>
  );
};

export default ScorecardView;

