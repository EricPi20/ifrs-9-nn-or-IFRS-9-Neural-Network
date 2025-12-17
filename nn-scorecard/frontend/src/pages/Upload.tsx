/**
 * Upload Page
 * 
 * Page for uploading CSV data files.
 */

import React from 'react';
import { FileUpload } from '../components/upload/FileUpload';
import { Card } from '../components/common';

export const Upload: React.FC = () => {
  return (
    <div className="container mx-auto px-4 py-8">
      <h1 className="text-3xl font-bold mb-8">Upload Data</h1>
      
      <div className="max-w-2xl mx-auto">
        <Card title="Upload Credit Risk Data">
          <div className="mb-4">
            <p className="text-gray-600 mb-2">
              Upload a CSV file containing pre-processed credit risk data.
            </p>
            <ul className="list-disc list-inside text-sm text-gray-600 space-y-1">
              <li>Features should be WoE transformed and pre-binned (2-6 bins per feature)</li>
              <li>Include a segment column for portfolio segmentation (optional)</li>
              <li>Include a target column (binary: 0/1)</li>
            </ul>
          </div>
          <FileUpload />
        </Card>
      </div>
    </div>
  );
};

