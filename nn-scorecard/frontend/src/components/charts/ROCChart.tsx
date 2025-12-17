/**
 * ROC Curve Chart Component
 * 
 * Displays Receiver Operating Characteristic (ROC) curve using Recharts.
 */

import React from 'react';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ROCChartProps {
  trainData: Array<{ fpr: number; tpr: number }>;
  testData: Array<{ fpr: number; tpr: number }>;
}

export const ROCChart: React.FC<ROCChartProps> = ({ trainData, testData }) => {
  // Combine data for chart
  const chartData = trainData.map((point, index) => ({
    fpr: point.fpr,
    train_tpr: point.tpr,
    test_tpr: testData[index]?.tpr || 0,
  }));

  return (
    <ResponsiveContainer width="100%" height={400}>
      <LineChart data={chartData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis
          dataKey="fpr"
          label={{ value: 'False Positive Rate', position: 'insideBottom', offset: -5 }}
        />
        <YAxis
          label={{ value: 'True Positive Rate', angle: -90, position: 'insideLeft' }}
        />
        <Tooltip />
        <Legend />
        <Line
          type="monotone"
          dataKey="train_tpr"
          stroke="#3b82f6"
          name="Train"
          dot={false}
        />
        <Line
          type="monotone"
          dataKey="test_tpr"
          stroke="#10b981"
          name="Test"
          dot={false}
        />
      </LineChart>
    </ResponsiveContainer>
  );
};

