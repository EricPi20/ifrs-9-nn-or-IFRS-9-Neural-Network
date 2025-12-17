/**
 * Dashboard Page
 * 
 * Main dashboard showing overview of training jobs and recent results.
 */

import React from 'react';
import { Card, Button } from '../components/common';
import { useNavigate } from 'react-router-dom';

export const Dashboard: React.FC = () => {
  const navigate = useNavigate();

  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Neural Network Scorecard Dashboard</h1>
      
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6 mb-8">
        <Card>
          <h3 className="text-lg font-semibold mb-2">Upload Data</h3>
          <p className="text-gray-600 mb-4">
            Upload your pre-processed credit risk data (WoE transformed)
          </p>
          <Button onClick={() => navigate('/upload')} className="w-full">
            Go to Upload
          </Button>
        </Card>
        
        <Card>
          <h3 className="text-lg font-semibold mb-2">Train Model</h3>
          <p className="text-gray-600 mb-4">
            Configure and train neural network scorecard models
          </p>
          <Button onClick={() => navigate('/training')} className="w-full">
            Go to Training
          </Button>
        </Card>
        
        <Card>
          <h3 className="text-lg font-semibold mb-2">View Results</h3>
          <p className="text-gray-600 mb-4">
            Review model performance and scorecard results
          </p>
          <Button onClick={() => navigate('/results')} className="w-full">
            Go to Results
          </Button>
        </Card>
      </div>
      
      <Card title="Recent Training Jobs">
        <p className="text-gray-600">No recent jobs. Start by uploading data and training a model.</p>
      </Card>
    </div>
  );
};

