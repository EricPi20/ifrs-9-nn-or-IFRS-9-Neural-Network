/**
 * Model Metrics Component
 * 
 * Displays model evaluation metrics (AR, AUC, KS, etc.).
 */

import React from 'react';
import { Card } from '../common';
import { ModelResults } from '../../types';

interface ModelMetricsProps {
  results: ModelResults;
}

export const ModelMetrics: React.FC<ModelMetricsProps> = ({ results }) => {
  return (
    <Card title="Model Performance Metrics">
      <div className="grid grid-cols-2 gap-4">
        <div className="p-4 bg-blue-50 rounded-lg">
          <div className="text-sm text-gray-600">Train AR</div>
          <div className="text-2xl font-bold text-blue-600">
            {results.train_ar.toFixed(4)}
          </div>
        </div>
        <div className="p-4 bg-green-50 rounded-lg">
          <div className="text-sm text-gray-600">Test AR</div>
          <div className="text-2xl font-bold text-green-600">
            {results.test_ar.toFixed(4)}
          </div>
        </div>
        <div className="p-4 bg-purple-50 rounded-lg">
          <div className="text-sm text-gray-600">Train AUC</div>
          <div className="text-2xl font-bold text-purple-600">
            {results.train_auc.toFixed(4)}
          </div>
        </div>
        <div className="p-4 bg-orange-50 rounded-lg">
          <div className="text-sm text-gray-600">Test AUC</div>
          <div className="text-2xl font-bold text-orange-600">
            {results.test_auc.toFixed(4)}
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">Train KS</div>
          <div className="text-2xl font-bold text-gray-700">
            {results.train_ks.toFixed(4)}
          </div>
        </div>
        <div className="p-4 bg-gray-50 rounded-lg">
          <div className="text-sm text-gray-600">Test KS</div>
          <div className="text-2xl font-bold text-gray-700">
            {results.test_ks.toFixed(4)}
          </div>
        </div>
      </div>
    </Card>
  );
};

