// frontend/src/pages/ConfigView.tsx

import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { ArrowLeft, Copy, Check, Download } from 'lucide-react';
import { getResult } from '../stores/resultsStore';
import { api } from '../services/api';

const ConfigView: React.FC = () => {
  const { jobId } = useParams<{ jobId: string }>();
  const [config, setConfig] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    const fetchConfig = async () => {
      if (!jobId) return;
      
      setLoading(true);
      
      // First try to get from local storage
      const storedResult = getResult(jobId);
      if (storedResult) {
        setConfig(storedResult);
        setLoading(false);
        return;
      }
      
      // If not in storage, try to get from API
      try {
        const status = await api.getTrainingStatus(jobId);
        setConfig({
          job_id: jobId,
          segment: status.config?.segment || 'Unknown',
          created_at: status.created_at,
          config: status.config,
          metrics: status.current_metrics,
        });
      } catch (err: any) {
        setError('Configuration not found. The training may have been deleted or the server restarted.');
      } finally {
        setLoading(false);
      }
    };
    
    fetchConfig();
  }, [jobId]);

  const handleCopyConfig = () => {
    if (config) {
      navigator.clipboard.writeText(JSON.stringify(config.config, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    }
  };

  const handleDownloadCSV = async () => {
    if (!jobId) return;
    
    setDownloading(true);
    try {
      const blob = await api.downloadConfigCSV(jobId);
      const url = window.URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `config_${jobId}.csv`;
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

  if (loading) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="flex items-center justify-center h-64">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-600"></div>
        </div>
      </div>
    );
  }

  if (error || !config) {
    return (
      <div className="container mx-auto px-4 py-8">
        <div className="bg-yellow-50 border border-yellow-200 rounded-lg p-6">
          <h3 className="text-yellow-800 font-semibold mb-2">Configuration Not Found</h3>
          <p className="text-yellow-700 mb-4">{error}</p>
          <Link
            to="/results"
            className="inline-flex items-center gap-2 text-yellow-600 hover:text-yellow-800"
          >
            <ArrowLeft className="w-4 h-4" />
            Back to Results
          </Link>
        </div>
      </div>
    );
  }

  const cfg = config.config || {};

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
          <h1 className="text-2xl font-bold text-gray-800">Training Configuration</h1>
          <p className="text-sm text-gray-500">
            Training ID: <code className="bg-gray-100 px-2 py-0.5 rounded">{jobId}</code>
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={handleDownloadCSV}
            disabled={downloading || !config}
            className="px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 text-sm disabled:opacity-50 disabled:cursor-not-allowed flex items-center gap-2"
          >
            <Download className="w-4 h-4" />
            {downloading ? 'Downloading...' : 'Download CSV'}
          </button>
          <Link
            to={`/results/${jobId}/scorecard`}
            className="px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 text-sm"
          >
            View Scorecard
          </Link>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Data Configuration */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            📊 Data Configuration
          </h2>
          <div className="space-y-3">
            <ConfigRow label="Segment" value={config.segment} />
            <ConfigRow label="Test Size" value={`${((cfg.test_size || 0.25) * 100).toFixed(0)}%`} />
            <ConfigRow label="Random Seed" value={cfg.random_seed || 42} />
            <ConfigRow label="Stratified Split" value={cfg.stratified_split ? 'Yes' : 'No'} />
            <ConfigRow 
              label="Features" 
              value={`${cfg.selected_features?.length || 0} selected`} 
            />
          </div>
          
          {cfg.selected_features && (
            <div className="mt-4 pt-4 border-t">
              <p className="text-sm text-gray-600 mb-2">Selected Features:</p>
              <div className="flex flex-wrap gap-2">
                {cfg.selected_features.map((feat: string, idx: number) => (
                  <span 
                    key={idx}
                    className="px-2 py-1 bg-blue-50 text-blue-700 text-xs rounded-full"
                  >
                    {feat}
                  </span>
                ))}
              </div>
            </div>
          )}
        </div>

        {/* Network Architecture */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            🧠 Neural Network Architecture
          </h2>
          <div className="space-y-3">
            <ConfigRow 
              label="Hidden Layers" 
              value={
                <code className="bg-gray-100 px-2 py-1 rounded text-sm">
                  [{cfg.network?.hidden_layers?.join(' → ') || '—'}]
                </code>
              } 
            />
            <ConfigRow label="Activation" value={cfg.network?.activation || 'relu'} />
            <ConfigRow label="Skip Connection" value={cfg.network?.skip_connection ? 'Yes' : 'No'} />
            <ConfigRow label="Dropout Rate" value={cfg.regularization?.dropout_rate || 0.3} />
            <ConfigRow label="L2 Lambda" value={cfg.regularization?.l2_lambda || 0.001} />
          </div>
          
          {/* Visual representation */}
          <div className="mt-4 pt-4 border-t">
            <p className="text-sm text-gray-600 mb-3">Network Visualization:</p>
            <div className="flex items-center justify-center gap-2 py-4 bg-gray-50 rounded-lg overflow-x-auto">
              <LayerBox label="Input" size={cfg.selected_features?.length || '?'} color="blue" />
              <Arrow />
              {cfg.network?.hidden_layers?.map((size: number, idx: number) => (
                <React.Fragment key={idx}>
                  <LayerBox label={`H${idx + 1}`} size={size} color="purple" />
                  <Arrow />
                </React.Fragment>
              ))}
              <LayerBox label="Output" size={1} color="green" />
            </div>
          </div>
        </div>

        {/* Training Parameters */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            ⚙️ Training Parameters
          </h2>
          <div className="space-y-3">
            <ConfigRow label="Learning Rate" value={cfg.learning_rate || 0.001} mono />
            <ConfigRow label="Batch Size" value={cfg.batch_size || 32} />
            <ConfigRow label="Epochs" value={cfg.epochs || 100} />
            <ConfigRow label="Loss Function" value={cfg.loss_type || cfg.loss?.loss_type || 'combined'} />
            {cfg.loss_type === 'combined' || cfg.loss?.loss_type === 'combined' ? (
              <>
                <ConfigRow label="Loss Alpha (BCE weight)" value={cfg.loss_alpha ?? cfg.loss?.loss_alpha ?? 0.3} mono />
                <ConfigRow label="AUC Gamma" value={cfg.auc_gamma ?? cfg.loss?.auc_gamma ?? 2.0} mono />
              </>
            ) : null}
            <ConfigRow label="Early Stopping" value={cfg.early_stopping?.enabled ? 'Enabled' : 'Disabled'} />
            {cfg.early_stopping?.enabled && (
              <ConfigRow label="Patience" value={cfg.early_stopping?.patience || 10} />
            )}
          </div>
        </div>

        {/* Performance Metrics */}
        <div className="bg-white rounded-xl shadow-sm p-6">
          <h2 className="text-lg font-semibold text-gray-800 mb-4 flex items-center gap-2">
            📈 Performance Metrics
          </h2>
          {config.metrics ? (
            <div className="grid grid-cols-2 gap-4">
              <MetricCard 
                label="Test AUC" 
                value={config.metrics.test_auc?.toFixed(4)} 
                color="blue"
              />
              <MetricCard 
                label="Test AR (Gini)" 
                value={config.metrics.test_ar?.toFixed(4)} 
                color="green"
              />
              <MetricCard 
                label="Test KS" 
                value={config.metrics.test_ks?.toFixed(4)} 
                color="purple"
              />
              <MetricCard 
                label="Train AUC" 
                value={config.metrics.train_auc?.toFixed(4)} 
                color="gray"
              />
            </div>
          ) : (
            <p className="text-gray-500 text-sm">Metrics not available</p>
          )}
        </div>
      </div>

      {/* Raw Config JSON */}
      <div className="mt-6 bg-white rounded-xl shadow-sm p-6">
        <div className="flex items-center justify-between mb-4">
          <h2 className="text-lg font-semibold text-gray-800">Raw Configuration (JSON)</h2>
          <button
            onClick={handleCopyConfig}
            className="flex items-center gap-2 px-3 py-1.5 text-sm bg-gray-100 hover:bg-gray-200 rounded-lg transition-colors"
          >
            {copied ? (
              <>
                <Check className="w-4 h-4 text-green-600" />
                <span className="text-green-600">Copied!</span>
              </>
            ) : (
              <>
                <Copy className="w-4 h-4" />
                <span>Copy</span>
              </>
            )}
          </button>
        </div>
        <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg overflow-x-auto text-sm">
          {JSON.stringify(cfg, null, 2)}
        </pre>
      </div>
    </div>
  );
};

// Helper Components
const ConfigRow: React.FC<{ label: string; value: any; mono?: boolean }> = ({ label, value, mono }) => (
  <div className="flex justify-between items-center py-2 border-b border-gray-100 last:border-0">
    <span className="text-gray-600 text-sm">{label}</span>
    <span className={`font-medium ${mono ? 'font-mono text-sm' : ''}`}>
      {typeof value === 'boolean' ? (value ? 'Yes' : 'No') : value}
    </span>
  </div>
);

const MetricCard: React.FC<{ label: string; value: string | undefined; color: string }> = ({ label, value, color }) => {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-50 text-blue-700',
    green: 'bg-green-50 text-green-700',
    purple: 'bg-purple-50 text-purple-700',
    gray: 'bg-gray-50 text-gray-700',
  };
  
  return (
    <div className={`p-4 rounded-lg ${colorClasses[color]}`}>
      <div className="text-xs opacity-75 mb-1">{label}</div>
      <div className="text-xl font-bold font-mono">{value || '—'}</div>
    </div>
  );
};

const LayerBox: React.FC<{ label: string; size: number | string; color: string }> = ({ label, size, color }) => {
  const colorClasses: Record<string, string> = {
    blue: 'bg-blue-100 border-blue-300 text-blue-700',
    purple: 'bg-purple-100 border-purple-300 text-purple-700',
    green: 'bg-green-100 border-green-300 text-green-700',
  };
  
  return (
    <div className={`flex flex-col items-center px-3 py-2 rounded-lg border-2 ${colorClasses[color]}`}>
      <span className="text-xs font-medium">{label}</span>
      <span className="text-lg font-bold">{size}</span>
    </div>
  );
};

const Arrow: React.FC = () => (
  <div className="text-gray-400 text-lg">→</div>
);

export default ConfigView;

