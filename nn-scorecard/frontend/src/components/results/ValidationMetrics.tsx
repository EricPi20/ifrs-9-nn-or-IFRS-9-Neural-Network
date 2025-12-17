/**
 * Validation Metrics Component
 * 
 * Displays validation metrics including histograms, bad rates, ROC curves, and score bands.
 */

import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { Loader2 } from 'lucide-react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';

interface ValidationMetricsProps {
  jobId: string;
}

export const ValidationMetrics: React.FC<ValidationMetricsProps> = ({ jobId }) => {
  const [validationData, setValidationData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    fetchMetrics();
  }, [jobId]);

  const fetchMetrics = async () => {
    try {
      setLoading(true);
      const response = await api.getValidationMetrics(jobId);
      setValidationData(response);
      setError(null);
    } catch (err: any) {
      setError(err.message || 'Failed to load validation metrics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-blue-600" />
      </div>
    );
  }

  if (error && !validationData) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-6">
        <p className="text-red-700">{error}</p>
      </div>
    );
  }

  if (!validationData) {
    return (
      <div className="bg-gray-50 rounded-lg p-8 text-center text-gray-500">
        Validation metrics not available for this training run.
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Model Performance Metrics (Validation Data)</h2>

        {/* Metrics Cards */}
        {validationData?.metrics && (
          <div className="grid grid-cols-2 gap-4 mb-6">
            <div className="bg-gradient-to-br from-blue-50 to-blue-100 rounded-lg p-6">
              <p className="text-sm text-blue-600 font-medium mb-1">AUC</p>
              <p className="text-4xl font-bold text-blue-700">
                {(validationData.metrics.auc * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-blue-500 mt-1">Area Under Curve</p>
            </div>
            <div className="bg-gradient-to-br from-green-50 to-green-100 rounded-lg p-6">
              <p className="text-sm text-green-600 font-medium mb-1">AR (Gini)</p>
              <p className="text-4xl font-bold text-green-700">
                {(validationData.metrics.ar * 100).toFixed(1)}%
              </p>
              <p className="text-xs text-green-500 mt-1">Accuracy Ratio</p>
            </div>
          </div>
        )}

        {error && (
          <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-4 mb-4">
            <p className="text-yellow-700 text-sm">{error}</p>
          </div>
        )}
      </div>

      {/* Histogram */}
      {validationData?.histogram && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Score Distribution Histogram</h3>
          <ResponsiveContainer width="100%" height={400}>
            <BarChart data={validationData.histogram.bin_labels.map((label: string, i: number) => ({
              label,
              good: validationData.histogram.good_counts[i],
              bad: validationData.histogram.bad_counts[i],
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" angle={-45} textAnchor="end" height={80} />
              <YAxis />
              <Tooltip />
              <Legend />
              <Bar dataKey="good" stackId="a" fill="#10b981" name="Good" />
              <Bar dataKey="bad" stackId="a" fill="#ef4444" name="Bad" />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Bad Rates Chart */}
      {validationData?.histogram && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Bad Rate by Score Band</h3>
          <ResponsiveContainer width="100%" height={400}>
            <ComposedChart data={validationData.histogram.bin_labels.map((label: string, i: number) => ({
              label,
              count: validationData.histogram.total_counts[i],
              bad_rate: validationData.histogram.bad_rate[i],
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis dataKey="label" angle={-45} textAnchor="end" height={80} />
              <YAxis yAxisId="left" label={{ value: 'Bad Rate (%)', angle: -90, position: 'insideLeft' }} />
              <YAxis yAxisId="right" orientation="right" label={{ value: 'Count', angle: 90, position: 'insideRight' }} />
              <Tooltip />
              <Legend />
              <Bar yAxisId="right" dataKey="count" fill="#93c5fd" name="Count" opacity={0.6} />
              <Line 
                yAxisId="left"
                type="monotone" 
                dataKey="bad_rate" 
                stroke="#3b82f6" 
                strokeWidth={2}
                name="Bad Rate"
                dot={false}
              />
            </ComposedChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ROC Curve */}
      {validationData?.roc_curve && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">ROC Curve</h3>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={validationData.roc_curve.fpr.map((fpr: number, i: number) => ({
              fpr,
              tpr: validationData.roc_curve.tpr[i],
              diagonal: validationData.roc_curve.diagonal[i],
            }))}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis label={{ value: 'False Positive Rate (%)', position: 'insideBottom', offset: -5 }} />
              <YAxis label={{ value: 'True Positive Rate (%)', angle: -90, position: 'insideLeft' }} />
              <Tooltip />
              <Legend />
              <Line
                type="monotone"
                dataKey="tpr"
                stroke="#3b82f6"
                strokeWidth={2}
                name={`Validation (AUC: ${(validationData.metrics.auc * 100).toFixed(2)}%)`}
                dot={false}
              />
              <Line
                type="monotone"
                dataKey="diagonal"
                stroke="#9ca3af"
                strokeWidth={1}
                strokeDasharray="5 5"
                name="Random (AUC: 0.5)"
                dot={false}
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Score Bands */}
      {validationData?.score_bands && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Count per Score Band</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Score Band</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Count</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Good</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Bad</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Bad Rate</th>
                </tr>
              </thead>
              <tbody>
                {validationData.score_bands.map((band: any, i: number) => (
                  <tr key={i} className="border-b hover:bg-gray-50">
                    <td className="py-3 px-4 font-medium text-gray-700">{band.range}</td>
                    <td className="py-3 px-4 text-right text-gray-600">{band.total.toLocaleString()}</td>
                    <td className="py-3 px-4 text-right text-gray-600">{band.good.toLocaleString()}</td>
                    <td className="py-3 px-4 text-right text-gray-600">{band.bad.toLocaleString()}</td>
                    <td className="py-3 px-4 text-right text-gray-600">{band.bad_rate.toFixed(2)}%</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </div>
  );
};

export default ValidationMetrics;
