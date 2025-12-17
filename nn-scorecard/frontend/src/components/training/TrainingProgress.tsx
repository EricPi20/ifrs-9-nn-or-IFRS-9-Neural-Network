/**
 * Training Progress Component
 * 
 * Displays real-time training progress and metrics.
 */

import React from 'react';
import { Card, LoadingSpinner, Button, MetricCard } from '../common';
import { TrainingHistoryCharts } from './TrainingHistoryCharts';

interface TrainingProgressProps {
  jobId: string;
  status: string;
  progress: number;
  message: string;
  currentEpoch?: number;
  totalEpochs?: number;
  currentMetrics?: {
    train_ar?: number;
    test_ar?: number;
    train_auc?: number;
    test_auc?: number;
    train_loss?: number;
    test_loss?: number;
    train_ks?: number;
    test_ks?: number;
  };
  history?: Array<{
    epoch: number;
    train_loss: number;
    test_loss: number;
    train_ar: number;
    test_ar: number;
    train_auc: number;
    test_auc: number;
    train_ks: number;
    test_ks: number;
  }>;
  error?: string;
  onCancel?: () => void;
  onModifyAndRetrain?: () => void;
  onTrainNewModel?: () => void;
}

export const TrainingProgress: React.FC<TrainingProgressProps> = ({
  jobId,
  status,
  progress,
  message,
  currentEpoch,
  totalEpochs,
  currentMetrics,
  history,
  error,
  onCancel,
  onModifyAndRetrain,
  onTrainNewModel,
}) => {
  const isRunning = status === 'queued' || status === 'preparing' || status === 'training' || status === 'generating_scorecard';
  const isTraining = status === 'training';
  const canCancel = isRunning && onCancel;
  
  return (
    <div className="space-y-6">
      {/* Header with Job ID and Status */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <div className="flex justify-between items-center mb-4">
          <div>
            <h2 className="text-lg font-semibold text-[#1E3A5F]">
              Training Job: {jobId.substring(0, 8)}...
            </h2>
            <div className="flex items-center gap-2 mt-1">
              <p className="text-sm text-gray-500">
                Status: <span className="font-medium capitalize">{status}</span>
              </p>
              {isTraining && (
                <div className="flex items-center gap-2 text-sm text-[#38B2AC]">
                  <span className="relative flex h-3 w-3">
                    <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-[#38B2AC] opacity-75"></span>
                    <span className="relative inline-flex rounded-full h-3 w-3 bg-[#38B2AC]"></span>
                  </span>
                  <span>Training in progress...</span>
                </div>
              )}
            </div>
          </div>
          <div className="text-right">
            <div className="text-2xl font-bold text-[#38B2AC]">
              {progress.toFixed(0)}%
            </div>
            <div className="text-sm text-gray-500">
              Epoch {currentEpoch || 0} / {totalEpochs || '?'}
            </div>
          </div>
        </div>

        {/* Progress Bar */}
        <div className="h-3 bg-gray-200 rounded-full overflow-hidden">
          <div
            className="h-full bg-gradient-to-r from-[#1E3A5F] to-[#38B2AC] transition-all duration-300"
            style={{ width: `${Math.min(progress, 100)}%` }}
          />
        </div>

        {isRunning && !isTraining && (
          <div className="flex items-center gap-2 mt-4">
            <LoadingSpinner size="sm" />
            <span className="text-gray-600">{message}</span>
          </div>
        )}
      </div>

      {/* Live Metrics Cards */}
      {currentMetrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Train AR"
            value={currentMetrics.train_ar?.toFixed(4) ?? 'N/A'}
            color="navy"
          />
          <MetricCard
            label="Test AR"
            value={currentMetrics.test_ar?.toFixed(4) ?? 'N/A'}
            color="teal"
          />
          <MetricCard
            label="Train AUC"
            value={currentMetrics.train_auc?.toFixed(4) ?? 'N/A'}
            color="navy"
          />
          <MetricCard
            label="Test AUC"
            value={currentMetrics.test_auc?.toFixed(4) ?? 'N/A'}
            color="teal"
          />
        </div>
      )}
          
      {/* Live Training History Charts */}
      {history && history.length > 0 && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <TrainingHistoryCharts history={history} />
        </div>
      )}

      {/* Completion Message - Note: Action buttons are now handled in parent component */}
      {status === 'completed' && !onModifyAndRetrain && (
        <div className="bg-green-50 border border-green-200 rounded-lg p-4">
          <h3 className="text-green-800 font-semibold">
            ✓ Training completed successfully!
          </h3>
          <p className="text-green-700 text-sm mt-1">
            {message || (currentMetrics ? `Best model saved. Final Test AR: ${currentMetrics.test_ar?.toFixed(4)}` : 'Training completed.')}
          </p>
          <button
            onClick={() => window.location.href = `/results/${jobId}`}
            className="mt-4 bg-[#1E3A5F] text-white px-6 py-2 rounded-lg font-medium hover:bg-[#2D4A6F] transition-colors"
          >
            View Scorecard & Results →
          </button>
        </div>
      )}

      {/* Error Message */}
      {status === 'failed' && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-semibold">✗ Training failed</h3>
          <p className="text-red-700 text-sm mt-1">{error || message}</p>
        </div>
      )}

      {/* Cancelled Message */}
      {status === 'cancelled' && (
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4">
          <h3 className="text-yellow-800 font-semibold">Training cancelled</h3>
          <p className="text-yellow-700 text-sm mt-1">{message}</p>
        </div>
      )}

      {/* Cancel Button */}
      {canCancel && (
        <div className="flex justify-center">
          <Button 
            variant="danger" 
            onClick={onCancel}
            className="min-w-[140px]"
          >
            Cancel Training
          </Button>
        </div>
      )}

      {/* Training History Charts */}
      {history && history.length > 0 && (
        <Card>
          <TrainingHistoryCharts history={history} />
        </Card>
      )}
    </div>
  );
};
