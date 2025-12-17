/**
 * Out-of-Time Validation Component
 * 
 * Allows uploading a test dataset, scores all records, and displays
 * validation metrics including histogram, ROC curve, score bands, and bad rates.
 */

import React, { useState, useEffect } from 'react';
import { api } from '../../services/api';
import { Loader2, Upload, FileCheck } from 'lucide-react';
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer, ComposedChart
} from 'recharts';

interface OutOfTimeValidationProps {
  jobId: string;
}

export const OutOfTimeValidation: React.FC<OutOfTimeValidationProps> = ({ jobId }) => {
  const [file, setFile] = useState<File | null>(null);
  const [validationData, setValidationData] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [loadingSaved, setLoadingSaved] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [uploaded, setUploaded] = useState(false);

  // Load saved out-of-time validation results on mount
  useEffect(() => {
    const loadSavedResults = async () => {
      try {
        setLoadingSaved(true);
        const savedData = await api.getOutOfTimeValidation(jobId);
        setValidationData(savedData);
        setUploaded(true);
      } catch (err: any) {
        // Not found is OK - means no saved results yet
        if (!err.message?.includes('not found')) {
          console.warn('Failed to load saved out-of-time validation:', err);
        }
      } finally {
        setLoadingSaved(false);
      }
    };

    if (jobId) {
      loadSavedResults();
    }
  }, [jobId]);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      const selectedFile = e.target.files[0];
      if (!selectedFile.name.endsWith('.csv')) {
        setError('Please upload a CSV file');
        return;
      }
      setFile(selectedFile);
      setError(null);
      setUploaded(false);
      setValidationData(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file first');
      return;
    }

    setLoading(true);
    setError(null);

    try {
      const response = await api.uploadOutOfTimeValidation(jobId, file);
      setValidationData(response);
      setUploaded(true);
    } catch (err: any) {
      setError(err.message || 'Failed to process out-of-time validation');
      setValidationData(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Upload Section */}
      <div className="bg-white rounded-xl shadow-md p-6">
        <h2 className="text-xl font-bold text-gray-800 mb-4">Out-of-Time Validation</h2>
        <p className="text-sm text-gray-600 mb-4">
          Upload a test dataset (CSV) to score all records and compute validation metrics.
          The file should contain the same features as the training data, plus a target column.
        </p>

        <div className="flex items-center gap-4">
          <label className="flex-1">
            <input
              type="file"
              accept=".csv"
              onChange={handleFileChange}
              className="hidden"
              id="file-upload"
            />
            <div className="flex items-center gap-3 p-4 border-2 border-dashed border-gray-300 rounded-lg hover:border-blue-500 cursor-pointer transition-colors">
              {file ? (
                <>
                  <FileCheck className="w-5 h-5 text-green-600" />
                  <span className="text-gray-700 font-medium">{file.name}</span>
                </>
              ) : (
                <>
                  <Upload className="w-5 h-5 text-gray-400" />
                  <span className="text-gray-500">Click to select CSV file</span>
                </>
              )}
            </div>
          </label>
          <button
            onClick={handleUpload}
            disabled={!file || loading}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700 disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center gap-2"
          >
            {loading ? (
              <>
                <Loader2 className="w-4 h-4 animate-spin" />
                Processing...
              </>
            ) : (
              <>
                <Upload className="w-4 h-4" />
                Upload & Score
              </>
            )}
          </button>
        </div>

        {error && (
          <div className="mt-4 bg-red-50 border border-red-200 rounded-lg p-4">
            <p className="text-red-700 text-sm">{error}</p>
          </div>
        )}

        {loadingSaved && (
          <div className="mt-4 flex items-center gap-2 text-gray-600 text-sm">
            <Loader2 className="w-4 h-4 animate-spin" />
            Loading saved results...
          </div>
        )}

        {uploaded && validationData && (
          <div className="mt-4 bg-green-50 border border-green-200 rounded-lg p-4">
            <p className="text-green-700 text-sm font-medium">
              ✓ {validationData.uploaded_at 
                ? `Results from ${new Date(validationData.uploaded_at).toLocaleString()}`
                : 'Successfully processed'} {validationData.metrics?.n_samples || 0} records
            </p>
            {validationData.filename && (
              <p className="text-green-600 text-xs mt-1">
                File: {validationData.filename}
              </p>
            )}
          </div>
        )}
      </div>

      {/* Metrics Cards */}
      {validationData?.metrics && (
        <div className="grid grid-cols-2 gap-4">
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

      {/* Visualizations - Side by Side */}
      {validationData && (
        <div className="grid grid-cols-2 gap-6">
          {/* Distribution Histogram */}
          {validationData.histogram && (
            <div className="bg-white rounded-xl shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Distribution Histogram</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={validationData.histogram.bin_labels.map((label: string, i: number) => ({
                  label,
                  good: validationData.histogram.good_counts[i],
                  bad: validationData.histogram.bad_counts[i],
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" angle={-45} textAnchor="end" height={80} fontSize={10} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="good" stackId="a" fill="#10b981" name="Good" />
                  <Bar dataKey="bad" stackId="a" fill="#ef4444" name="Bad" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* ROC Curve */}
          {validationData.roc_curve && (
            <div className="bg-white rounded-xl shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">ROC Curve</h3>
              <ResponsiveContainer width="100%" height={300}>
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
                    name={`OOT Validation (AUC: ${(validationData.metrics.auc * 100).toFixed(2)}%)`}
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

          {/* Count per Scoreband */}
          {validationData.score_bands && (
            <div className="bg-white rounded-xl shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Count per Scoreband</h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={validationData.score_bands.map((band: any) => ({
                  range: band.range,
                  count: band.total,
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="range" angle={-45} textAnchor="end" height={80} fontSize={10} />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="count" fill="#93c5fd" name="Count" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Bad Rate by Scoreband */}
          {validationData.histogram && (
            <div className="bg-white rounded-xl shadow-md p-6">
              <h3 className="text-lg font-semibold text-gray-800 mb-4">Bad Rate by Scoreband</h3>
              <ResponsiveContainer width="100%" height={300}>
                <ComposedChart data={validationData.histogram.bin_labels.map((label: string, i: number) => ({
                  label,
                  count: validationData.histogram.total_counts[i],
                  bad_rate: validationData.histogram.bad_rate[i],
                }))}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="label" angle={-45} textAnchor="end" height={80} fontSize={10} />
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
        </div>
      )}

      {/* Score Bands Table */}
      {validationData?.score_bands && (
        <div className="bg-white rounded-xl shadow-md p-6">
          <h3 className="text-lg font-semibold text-gray-800 mb-4">Score Bands Detail</h3>
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead>
                <tr className="border-b">
                  <th className="text-left py-3 px-4 font-semibold text-gray-700">Score Band</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Count</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Good</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Bad</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">Bad Rate</th>
                  <th className="text-right py-3 px-4 font-semibold text-gray-700">% of Total</th>
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
                    <td className="py-3 px-4 text-right text-gray-600">{band.pct_total.toFixed(2)}%</td>
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

export default OutOfTimeValidation;

