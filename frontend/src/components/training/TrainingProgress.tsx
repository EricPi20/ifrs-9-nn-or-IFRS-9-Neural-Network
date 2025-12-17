import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, 
  Tooltip, Legend, ResponsiveContainer 
} from 'recharts';
import { Card, Button, Badge, MetricCard } from '../common';

interface EpochMetrics {
  epoch: number;
  train_loss: number;
  test_loss: number;
  train_auc: number;
  test_auc: number;
  train_ar: number;
  test_ar: number;
  train_ks: number;
  test_ks: number;
}

interface TrainingStatus {
  job_id: string;
  status: 'queued' | 'preparing' | 'training' | 'generating_scorecard' | 'completed' | 'failed' | 'cancelled';
  current_epoch: number;
  total_epochs: number;
  current_metrics: {
    train_ar: number;
    test_ar: number;
    train_auc: number;
    test_auc: number;
  } | null;
  history?: EpochMetrics[];
  error?: string;
}

interface Props {
  status: TrainingStatus;
  onCancel?: () => void;
}

const statusColors: Record<string, string> = {
  queued: 'bg-gray-100 text-gray-800',
  preparing: 'bg-blue-100 text-blue-800',
  training: 'bg-amber-100 text-amber-800',
  generating_scorecard: 'bg-purple-100 text-purple-800',
  completed: 'bg-green-100 text-green-800',
  failed: 'bg-red-100 text-red-800',
  cancelled: 'bg-gray-100 text-gray-800'
};

const statusLabels: Record<string, string> = {
  queued: 'Queued',
  preparing: 'Preparing Data',
  training: 'Training',
  generating_scorecard: 'Generating Scorecard',
  completed: 'Completed',
  failed: 'Failed',
  cancelled: 'Cancelled'
};

export const TrainingProgress: React.FC<Props> = ({ status, onCancel }) => {
  const progress = status.total_epochs > 0 
    ? (status.current_epoch / status.total_epochs) * 100 
    : 0;

  const isActive = ['queued', 'preparing', 'training', 'generating_scorecard'].includes(status.status);

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-[#1E3A5F]">Training Progress</h2>
          <p className="text-sm text-gray-500">Job ID: {status.job_id}</p>
        </div>
        <Badge className={statusColors[status.status]}>
          {isActive && (
            <span className="inline-block w-2 h-2 bg-current rounded-full mr-2 animate-pulse" />
          )}
          {statusLabels[status.status]}
        </Badge>
      </div>

      {/* Progress Bar */}
      {status.status === 'training' && (
        <Card>
          <div className="space-y-3">
            <div className="flex justify-between text-sm">
              <span>Epoch {status.current_epoch} / {status.total_epochs}</span>
              <span className="font-mono">{progress.toFixed(0)}%</span>
            </div>
            <div className="h-4 bg-gray-200 rounded-full overflow-hidden">
              <div
                className="h-full bg-gradient-to-r from-[#1E3A5F] to-[#38B2AC] transition-all duration-500"
                style={{ width: `${progress}%` }}
              />
            </div>
          </div>
        </Card>
      )}

      {/* Current Metrics */}
      {status.current_metrics && (
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <MetricCard
            label="Train AUC"
            value={status.current_metrics.train_auc.toFixed(4)}
            color="navy"
          />
          <MetricCard
            label="Test AUC"
            value={status.current_metrics.test_auc.toFixed(4)}
            color="teal"
          />
          <MetricCard
            label="Train AR (Gini)"
            value={status.current_metrics.train_ar.toFixed(4)}
            color="navy"
          />
          <MetricCard
            label="Test AR (Gini)"
            value={status.current_metrics.test_ar.toFixed(4)}
            color="teal"
          />
        </div>
      )}

      {/* Training History Chart */}
      {status.history && status.history.length > 0 && (
        <Card title="Training History">
          <div className="h-80">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={status.history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="epoch" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Epoch', position: 'bottom', offset: -5 }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  domain={[0, 1]}
                />
                <Tooltip
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #E2E8F0',
                    borderRadius: '8px'
                  }}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="train_ar"
                  name="Train AR"
                  stroke="#1E3A5F"
                  strokeWidth={2}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="test_ar"
                  name="Test AR"
                  stroke="#38B2AC"
                  strokeWidth={2}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </Card>
      )}

      {/* Error Display */}
      {status.status === 'failed' && status.error && (
        <Card className="border-red-200 bg-red-50">
          <div className="text-red-700">
            <h4 className="font-semibold">Training Failed</h4>
            <p className="text-sm mt-1">{status.error}</p>
          </div>
        </Card>
      )}

      {/* Cancel Button */}
      {isActive && onCancel && (
        <div className="flex justify-center">
          <Button variant="danger" onClick={onCancel}>
            Cancel Training
          </Button>
        </div>
      )}
    </div>
  );
};

export default TrainingProgress;

