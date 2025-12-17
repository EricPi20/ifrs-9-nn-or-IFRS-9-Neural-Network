/**
 * Training History Charts Component
 * 
 * Displays training history in 3 separate charts for better readability:
 * 1. Loss Chart (Train Loss vs Test Loss)
 * 2. AR/Gini Chart (Train AR vs Test AR)
 * 3. KS Statistic Chart (Train KS vs Test KS)
 */

import React from 'react';
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts';

interface EpochMetric {
  epoch: number;
  train_loss: number;
  test_loss: number;
  train_ar: number;
  test_ar: number;
  train_ks: number;
  test_ks: number;
}

interface TrainingHistoryChartsProps {
  history: EpochMetric[];
}

export const TrainingHistoryCharts: React.FC<TrainingHistoryChartsProps> = ({ history }) => {
  if (!history || history.length === 0) {
    return null;
  }

  const chartHeight = 250;
  
  // Find best epoch based on Test AR
  const bestEpoch = history.reduce((best, curr, idx) => 
    curr.test_ar > (history[best]?.test_ar || 0) ? idx : best, 0
  );
  
  return (
    <div className="space-y-6">
      <h3 className="text-lg font-semibold text-[#1E3A5F]">Training History</h3>
      
      {/* Loss Chart - Full Width */}
      <div className="bg-white rounded-lg p-6 border border-gray-200">
        <h4 className="text-sm font-medium text-gray-700 mb-4">
          Loss <span className="text-xs text-gray-500 ml-2">(Lower is better)</span>
        </h4>
        <div style={{ height: chartHeight }}>
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={history}>
              <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
              <XAxis 
                dataKey="epoch" 
                tick={{ fontSize: 12 }}
                label={{ value: 'Epoch', position: 'bottom', offset: -5, fontSize: 12 }}
              />
              <YAxis 
                tick={{ fontSize: 12 }}
                tickFormatter={(v) => v.toFixed(2)}
              />
              <Tooltip 
                formatter={(value: number) => value.toFixed(4)}
                labelFormatter={(label) => `Epoch ${label}`}
                contentStyle={{
                  backgroundColor: 'white',
                  border: '1px solid #E2E8F0',
                  borderRadius: '8px'
                }}
              />
              <Legend wrapperStyle={{ fontSize: 12 }} />
              <Line
                type="monotone"
                dataKey="train_loss"
                name="Train"
                stroke="#E53E3E"
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
                animationDuration={300}
                animationEasing="ease-in-out"
              />
              <Line
                type="monotone"
                dataKey="test_loss"
                name="Test"
                stroke="#F6AD55"
                strokeWidth={2}
                dot={false}
                isAnimationActive={true}
                animationDuration={300}
                animationEasing="ease-in-out"
              />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* AR and KS - 2 columns */}
      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        {/* AR (GINI) CHART */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-4">
            AR / Gini <span className="text-xs text-gray-500 ml-2">(Higher is better)</span>
          </h4>
          <div style={{ height: chartHeight }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="epoch" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Epoch', position: 'bottom', offset: -5, fontSize: 12 }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  domain={[0, 'auto']}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip 
                  formatter={(value: number) => value.toFixed(4)}
                  labelFormatter={(label) => `Epoch ${label}`}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #E2E8F0',
                    borderRadius: '8px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line
                  type="monotone"
                  dataKey="train_ar"
                  name="Train"
                  stroke="#1E3A5F"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={300}
                  animationEasing="ease-in-out"
                />
                <Line
                  type="monotone"
                  dataKey="test_ar"
                  name="Test"
                  stroke="#38B2AC"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={300}
                  animationEasing="ease-in-out"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* KS CHART */}
        <div className="bg-white rounded-lg p-6 border border-gray-200">
          <h4 className="text-sm font-medium text-gray-700 mb-4">
            KS Statistic <span className="text-xs text-gray-500 ml-2">(Higher is better)</span>
          </h4>
          <div style={{ height: chartHeight }}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={history}>
                <CartesianGrid strokeDasharray="3 3" stroke="#E2E8F0" />
                <XAxis 
                  dataKey="epoch" 
                  tick={{ fontSize: 12 }}
                  label={{ value: 'Epoch', position: 'bottom', offset: -5, fontSize: 12 }}
                />
                <YAxis 
                  tick={{ fontSize: 12 }}
                  domain={[0, 'auto']}
                  tickFormatter={(v) => v.toFixed(2)}
                />
                <Tooltip 
                  formatter={(value: number) => value.toFixed(4)}
                  labelFormatter={(label) => `Epoch ${label}`}
                  contentStyle={{
                    backgroundColor: 'white',
                    border: '1px solid #E2E8F0',
                    borderRadius: '8px'
                  }}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <Line
                  type="monotone"
                  dataKey="train_ks"
                  name="Train"
                  stroke="#805AD5"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={300}
                  animationEasing="ease-in-out"
                />
                <Line
                  type="monotone"
                  dataKey="test_ks"
                  name="Test"
                  stroke="#D53F8C"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={true}
                  animationDuration={300}
                  animationEasing="ease-in-out"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
      
      {/* Best Epoch Indicator */}
      <div className="text-sm text-gray-600 text-center">
        Best model saved at epoch {history[bestEpoch]?.epoch || 'N/A'} (based on Test AR: {history[bestEpoch]?.test_ar.toFixed(4) || 'N/A'})
      </div>
    </div>
  );
};

export default TrainingHistoryCharts;

