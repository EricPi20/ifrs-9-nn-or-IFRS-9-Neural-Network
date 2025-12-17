// frontend/src/pages/Results.tsx

import React, { useState, useEffect } from 'react';
import { Link, useNavigate } from 'react-router-dom';
import { loadResults, deleteResult, TrainingResult } from '../stores/resultsStore';
import { Trash2, FileText, ExternalLink } from 'lucide-react';

const Results: React.FC = () => {
  const navigate = useNavigate();
  const [results, setResults] = useState<TrainingResult[]>([]);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState<string | null>(null);

  useEffect(() => {
    const storedResults = loadResults();
    setResults(storedResults);
  }, []);

  const handleDelete = (job_id: string) => {
    const updatedResults = deleteResult(job_id);
    setResults(updatedResults);
    setShowDeleteConfirm(null);
  };

  const formatDate = (dateString: string) => {
    return new Date(dateString).toLocaleString('en-US', {
      month: 'short',
      day: 'numeric',
      year: 'numeric',
      hour: '2-digit',
      minute: '2-digit',
    });
  };

  const shortenJobId = (jobId: string) => {
    return jobId.substring(0, 8) + '...';
  };

  return (
    <div className="container mx-auto px-4 py-8">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h1 className="text-2xl font-bold text-gray-800">Training Results</h1>
          <p className="text-sm text-gray-500 mt-1">
            {results.length} training run{results.length !== 1 ? 's' : ''} saved
          </p>
        </div>
        {results.length > 0 && (
          <button
            onClick={() => navigate('/training')}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
          >
            + New Training
          </button>
        )}
      </div>

      {results.length === 0 ? (
        <div className="bg-white rounded-xl shadow-sm p-12 text-center">
          <FileText className="w-16 h-16 text-gray-300 mx-auto mb-4" />
          <h3 className="text-lg font-medium text-gray-600 mb-2">No Training Results</h3>
          <p className="text-gray-500 mb-6">
            Complete a training run to see results here.
          </p>
          <button
            onClick={() => navigate('/training')}
            className="px-6 py-3 bg-blue-600 text-white rounded-lg hover:bg-blue-700"
          >
            Start Training
          </button>
        </div>
      ) : (
        <div className="bg-white rounded-xl shadow-sm overflow-hidden">
          <div className="overflow-x-auto">
            <table className="w-full">
              <thead className="bg-gray-50 border-b">
                <tr>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Training ID
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Date
                  </th>
                  <th className="px-4 py-3 text-left text-sm font-semibold text-gray-700">
                    Segment
                  </th>
                  <th className="px-4 py-3 text-right text-sm font-semibold text-gray-700">
                    Test AR
                  </th>
                  <th className="px-4 py-3 text-right text-sm font-semibold text-gray-700">
                    Test AUC
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                    Scorecard
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                    Config
                  </th>
                  <th className="px-4 py-3 text-center text-sm font-semibold text-gray-700">
                    Actions
                  </th>
                </tr>
              </thead>
              <tbody className="divide-y divide-gray-100">
                {results.map((result) => (
                  <tr key={result.job_id} className="hover:bg-gray-50">
                    {/* Training ID */}
                    <td className="px-4 py-3">
                      <code className="text-xs bg-gray-100 px-2 py-1 rounded font-mono text-gray-700">
                        {shortenJobId(result.job_id)}
                      </code>
                    </td>
                    
                    {/* Date */}
                    <td className="px-4 py-3 text-sm text-gray-600">
                      {formatDate(result.created_at)}
                    </td>
                    
                    {/* Segment */}
                    <td className="px-4 py-3">
                      <span className="px-2 py-1 bg-blue-100 text-blue-700 text-xs rounded-full font-medium">
                        {result.segment}
                      </span>
                    </td>
                    
                    {/* Test AR */}
                    <td className="px-4 py-3 text-right">
                      <span className={`font-mono font-semibold ${
                        result.metrics.test_ar > 0.5 ? 'text-green-600' : 
                        result.metrics.test_ar > 0.3 ? 'text-yellow-600' : 'text-gray-600'
                      }`}>
                        {result.metrics.test_ar?.toFixed(4) || '—'}
                      </span>
                    </td>
                    
                    {/* Test AUC */}
                    <td className="px-4 py-3 text-right">
                      <span className={`font-mono font-semibold ${
                        result.metrics.test_auc > 0.75 ? 'text-green-600' : 
                        result.metrics.test_auc > 0.65 ? 'text-yellow-600' : 'text-gray-600'
                      }`}>
                        {result.metrics.test_auc?.toFixed(4) || '—'}
                      </span>
                    </td>
                    
                    {/* Scorecard Link */}
                    <td className="px-4 py-3 text-center">
                      <Link
                        to={`/results/${result.job_id}/scorecard`}
                        className="inline-flex items-center gap-1 text-blue-600 hover:text-blue-800 text-sm font-medium"
                      >
                        View
                        <ExternalLink className="w-3 h-3" />
                      </Link>
                    </td>
                    
                    {/* Config Link */}
                    <td className="px-4 py-3 text-center">
                      <Link
                        to={`/results/${result.job_id}/config`}
                        className="inline-flex items-center gap-1 text-gray-600 hover:text-gray-800 text-sm font-medium"
                      >
                        View
                        <ExternalLink className="w-3 h-3" />
                      </Link>
                    </td>
                    
                    {/* Delete */}
                    <td className="px-4 py-3 text-center">
                      <button
                        onClick={() => setShowDeleteConfirm(result.job_id)}
                        className="p-2 text-red-500 hover:bg-red-50 rounded-lg transition-colors"
                        title="Delete"
                      >
                        <Trash2 className="w-4 h-4" />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showDeleteConfirm && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-xl shadow-xl p-6 max-w-md w-full mx-4">
            <h3 className="text-lg font-bold text-gray-800 mb-2">Delete Training Result?</h3>
            <p className="text-gray-600 mb-2">
              Training ID: <code className="text-sm bg-gray-100 px-2 py-1 rounded">{showDeleteConfirm}</code>
            </p>
            <p className="text-gray-500 text-sm mb-6">
              This will permanently remove this training result. This action cannot be undone.
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setShowDeleteConfirm(null)}
                className="px-4 py-2 bg-gray-200 text-gray-700 rounded-lg hover:bg-gray-300"
              >
                Cancel
              </button>
              <button
                onClick={() => handleDelete(showDeleteConfirm)}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700"
              >
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default Results;
