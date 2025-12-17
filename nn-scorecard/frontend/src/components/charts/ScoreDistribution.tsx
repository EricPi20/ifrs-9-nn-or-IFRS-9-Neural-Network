/**
 * Score Distribution Chart Component
 * 
 * Displays distribution of credit scores using a histogram.
 */

import React from 'react';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';

interface ScoreDistributionProps {
  scores: number[];
  bins?: number;
}

export const ScoreDistribution: React.FC<ScoreDistributionProps> = ({
  scores,
  bins = 20,
}) => {
  // Create histogram data
  const minScore = Math.min(...scores);
  const maxScore = Math.max(...scores);
  const binWidth = (maxScore - minScore) / bins;

  const histogramData = Array.from({ length: bins }, (_, i) => {
    const binStart = minScore + i * binWidth;
    const binEnd = binStart + binWidth;
    const count = scores.filter(
      (score) => score >= binStart && score < binEnd
    ).length;
    return {
      range: `${binStart.toFixed(0)}-${binEnd.toFixed(0)}`,
      count,
    };
  });

  return (
    <ResponsiveContainer width="100%" height={400}>
      <BarChart data={histogramData}>
        <CartesianGrid strokeDasharray="3 3" />
        <XAxis dataKey="range" />
        <YAxis label={{ value: 'Frequency', angle: -90, position: 'insideLeft' }} />
        <Tooltip />
        <Legend />
        <Bar dataKey="count" fill="#3b82f6" name="Score Count" />
      </BarChart>
    </ResponsiveContainer>
  );
};

