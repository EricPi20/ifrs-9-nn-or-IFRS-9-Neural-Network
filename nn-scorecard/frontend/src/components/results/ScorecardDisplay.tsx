/**
 * ScorecardDisplay Component
 * 
 * Displays the complete scorecard generated from the trained neural network,
 * showing feature weights, bin-level points, and score calculation details.
 */

import React, { useState } from 'react';
import { Scorecard, FeatureScore, BinScore } from '../../types';
import { Card } from '../common';

interface ScorecardDisplayProps {
  scorecard: Scorecard;
  onExport?: () => void;
}

export const ScorecardDisplay: React.FC<ScorecardDisplayProps> = ({
  scorecard,
  onExport,
}) => {
  const [expandedFeatures, setExpandedFeatures] = useState<Set<string>>(
    new Set(scorecard.features.slice(0, 3).map(f => f.feature_name)) // Expand top 3 by default
  );
  const [sortBy, setSortBy] = useState<'importance' | 'name' | 'weight'>('importance');
  const [showFormula, setShowFormula] = useState(false);

  // Toggle feature expansion
  const toggleFeature = (featureName: string) => {
    setExpandedFeatures(prev => {
      const next = new Set(prev);
      if (next.has(featureName)) {
        next.delete(featureName);
      } else {
        next.add(featureName);
      }
      return next;
    });
  };

  // Expand/collapse all
  const expandAll = () => {
    setExpandedFeatures(new Set(scorecard.features.map(f => f.feature_name)));
  };
  
  const collapseAll = () => {
    setExpandedFeatures(new Set());
  };

  // Sort features
  const sortedFeatures = [...scorecard.features].sort((a, b) => {
    switch (sortBy) {
      case 'importance':
        return a.importance_rank - b.importance_rank;
      case 'name':
        return a.feature_name.localeCompare(b.feature_name);
      case 'weight':
        return b.weight - a.weight;
      default:
        return 0;
    }
  });

  // Get color for points (all positive now, green gradient based on value)
  const getPointsColor = (points: number): string => {
    // All points are positive (0-100 range)
    if (points >= 75) return 'text-green-700 bg-green-100';
    if (points >= 50) return 'text-green-600 bg-green-50';
    if (points >= 25) return 'text-yellow-600 bg-yellow-50';
    return 'text-gray-600 bg-gray-50';
  };

  // Get importance badge color
  const getImportanceBadgeColor = (rank: number): string => {
    if (rank === 1) return 'bg-yellow-400 text-yellow-900';
    if (rank === 2) return 'bg-gray-300 text-gray-800';
    if (rank === 3) return 'bg-amber-600 text-white';
    return 'bg-gray-100 text-gray-600';
  };

  return (
    <div className="space-y-6">
      
      {/* === HEADER SECTION === */}
      <Card>
        <div className="flex justify-between items-start mb-6">
          <div>
            <h2 className="text-2xl font-bold text-[#1E3A5F]">
              Credit Scorecard
            </h2>
            <p className="text-gray-500 mt-1">
              Segment: <span className="font-medium">{scorecard.segment}</span>
              {' • '}
              Model: <span className="font-medium capitalize">{scorecard.model_type.replace('_', ' ')}</span>
              {' • '}
              Features: <span className="font-medium">{scorecard.features.length}</span>
            </p>
          </div>
          
          {onExport && (
            <button
              onClick={onExport}
              className="flex items-center gap-2 px-4 py-2 bg-[#1E3A5F] text-white 
                         rounded-lg hover:bg-[#2D4A6F] transition-colors"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} 
                      d="M12 10v6m0 0l-3-3m3 3l3-3m2 8H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
              </svg>
              Export Scorecard
            </button>
          )}
        </div>

        {/* Score Range Info */}
        <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
          <div className="bg-gradient-to-br from-[#1E3A5F] to-[#2D4A6F] rounded-lg p-4 text-white">
            <p className="text-sm opacity-80">Score Range</p>
            <p className="text-2xl font-bold">{scorecard.score_min} - {scorecard.score_max}</p>
          </div>
          <div className="bg-gradient-to-br from-[#38B2AC] to-[#2D9A94] rounded-lg p-4 text-white">
            <p className="text-sm opacity-80">Achievable Range</p>
            <p className="text-2xl font-bold">
              {scorecard.min_possible_score} - {scorecard.max_possible_score}
            </p>
          </div>
          <div className="bg-gray-100 rounded-lg p-4">
            <p className="text-sm text-gray-500">Test AUC</p>
            <p className="text-2xl font-bold text-[#1E3A5F]">
              {scorecard.metrics.test_auc.toFixed(4)}
            </p>
          </div>
          <div className="bg-gray-100 rounded-lg p-4">
            <p className="text-sm text-gray-500">Test AR (Gini)</p>
            <p className="text-2xl font-bold text-[#1E3A5F]">
              {scorecard.metrics.test_ar.toFixed(4)}
            </p>
          </div>
        </div>

        {/* Data Summary */}
        {scorecard.data_stats && (
          <div className="mt-4 p-4 bg-blue-50 rounded-lg">
            <h4 className="font-semibold text-blue-800 mb-2">Data Summary</h4>
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div>
                <span className="text-gray-600">Segment:</span>
                <span className="ml-2 font-medium">{scorecard.data_stats.segment || 'ALL'}</span>
              </div>
              <div>
                <span className="text-gray-600">Train:</span>
                <span className="ml-2 font-medium">{scorecard.data_stats.n_train?.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-gray-600">Test:</span>
                <span className="ml-2 font-medium">{scorecard.data_stats.n_test?.toLocaleString()}</span>
              </div>
              <div>
                <span className="text-gray-600">Bad Rate:</span>
                <span className="ml-2 font-medium">{scorecard.data_stats.bad_rate_overall}%</span>
              </div>
            </div>
          </div>
        )}
      </Card>

      {/* === SCORE FORMULA SECTION === */}
      <Card>
        <button
          onClick={() => setShowFormula(!showFormula)}
          className="w-full flex justify-between items-center"
        >
          <h3 className="text-lg font-semibold text-[#1E3A5F] flex items-center gap-2">
            <span>📐</span> Score Calculation Formula
          </h3>
          <svg 
            className={`w-5 h-5 text-gray-400 transition-transform ${showFormula ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </button>
        
        {showFormula && (
          <div className="mt-4 space-y-4">
            <div className="bg-gray-50 rounded-lg p-4 font-mono text-sm">
              <p className="text-gray-600 mb-2">// For each feature:</p>
              <p className="text-[#1E3A5F]">
                Points<sub>feature</sub> = Weight × Input_Value × Scale_Factor
              </p>
              <p className="text-gray-600 mt-3 mb-2">// Total score:</p>
              <p className="text-[#1E3A5F]">
                Score = Σ(Points<sub>feature</sub>) + Offset
              </p>
              <p className="text-gray-600 mt-3 mb-2">// Clamped to range:</p>
              <p className="text-[#1E3A5F]">
                Final_Score = max({scorecard.score_min}, min({scorecard.score_max}, Score))
              </p>
            </div>
            
            <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
              <div className="bg-blue-50 rounded-lg p-3">
                <p className="text-blue-600 font-medium">Scale Factor</p>
                <p className="text-blue-900 font-mono">{scorecard.scale_factor.toFixed(6)}</p>
              </div>
              <div className="bg-blue-50 rounded-lg p-3">
                <p className="text-blue-600 font-medium">Offset</p>
                <p className="text-blue-900 font-mono">{scorecard.offset.toFixed(2)}</p>
              </div>
              <div className="bg-blue-50 rounded-lg p-3">
                <p className="text-blue-600 font-medium">Raw Score Range</p>
                <p className="text-blue-900 font-mono">
                  [{scorecard.raw_min.toFixed(2)}, {scorecard.raw_max.toFixed(2)}]
                </p>
              </div>
              <div className="bg-blue-50 rounded-lg p-3">
                <p className="text-blue-600 font-medium">Input Scale</p>
                <p className="text-blue-900 font-mono">÷{scorecard.input_scale_factor}</p>
              </div>
            </div>

            <div className="text-sm text-gray-600 bg-yellow-50 border border-yellow-200 rounded-lg p-3">
              <p className="font-medium text-yellow-800 mb-1">📝 Note on Input Values:</p>
              <p>
                Input values in the scorecard are <strong>standardized log odds × (-50)</strong>.
                Positive values indicate lower risk, negative values indicate higher risk.
              </p>
            </div>
          </div>
        )}
      </Card>

      {/* === FEATURE WEIGHTS SUMMARY === */}
      <Card>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-[#1E3A5F] flex items-center gap-2">
            <span>⚖️</span> Feature Importance
          </h3>
          <div className="flex gap-2">
            <select
              value={sortBy}
              onChange={(e) => setSortBy(e.target.value as any)}
              className="text-sm border border-gray-300 rounded-lg px-3 py-1"
            >
              <option value="importance">Sort by Importance</option>
              <option value="weight">Sort by Weight</option>
              <option value="name">Sort by Name</option>
            </select>
          </div>
        </div>

        {/* Horizontal bar chart of feature importance */}
        <div className="space-y-3">
          {sortedFeatures.map((feature, idx) => {
            const maxWeight = Math.max(...scorecard.features.map(f => f.weight));
            const barWidth = (feature.weight / maxWeight) * 100;
            
            return (
              <div key={feature.feature_name} className="flex items-center gap-3">
                <div className="w-8 text-center">
                  <span className={`inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold
                                   ${getImportanceBadgeColor(feature.importance_rank)}`}>
                    {feature.importance_rank}
                  </span>
                </div>
                <div className="w-40 truncate text-sm font-medium text-gray-700">
                  {feature.feature_name}
                </div>
                <div className="flex-1">
                  <div className="h-6 bg-gray-100 rounded-full overflow-hidden">
                    <div
                      className="h-full rounded-full bg-[#38B2AC]"
                      style={{ width: `${barWidth}%` }}
                    />
                  </div>
                </div>
                {/* Weight as percentage */}
                <div className="w-20 text-right font-mono text-sm font-semibold text-[#1E3A5F]">
                  {feature.weight.toFixed(1)}%
                </div>
                {/* Point range - all positive now */}
                <div className="w-24 text-right text-sm text-gray-500">
                  [{feature.min_points} - {feature.max_points}]
                </div>
              </div>
            );
          })}
        </div>
      </Card>

      {/* === DETAILED FEATURE SCORECARDS === */}
      <Card>
        <div className="flex justify-between items-center mb-4">
          <h3 className="text-lg font-semibold text-[#1E3A5F] flex items-center gap-2">
            <span>📊</span> Feature Details & Bin Points
          </h3>
          <div className="flex gap-2">
            <button
              onClick={expandAll}
              className="text-sm text-[#38B2AC] hover:underline"
            >
              Expand All
            </button>
            <span className="text-gray-300">|</span>
            <button
              onClick={collapseAll}
              className="text-sm text-[#38B2AC] hover:underline"
            >
              Collapse All
            </button>
          </div>
        </div>

        <div className="space-y-3">
          {sortedFeatures.map((feature) => (
            <FeatureCard
              key={feature.feature_name}
              feature={feature}
              isExpanded={expandedFeatures.has(feature.feature_name)}
              onToggle={() => toggleFeature(feature.feature_name)}
              getPointsColor={getPointsColor}
              getImportanceBadgeColor={getImportanceBadgeColor}
            />
          ))}
        </div>
      </Card>

      {/* === LEGEND === */}
      <Card>
        <div className="py-2">
          <h4 className="text-sm font-semibold text-gray-700 mb-3">Legend</h4>
        <div className="flex flex-wrap gap-4 text-sm">
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-green-100 border border-green-300"></span>
            <span>High Points (75-100) - Lower Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-yellow-100 border border-yellow-300"></span>
            <span>Medium Points (25-75) - Moderate Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="w-4 h-4 rounded bg-gray-100 border border-gray-300"></span>
            <span>Low Points (0-25) - Higher Risk</span>
          </div>
          <div className="flex items-center gap-2">
            <span className="inline-flex items-center justify-center w-5 h-5 rounded-full bg-yellow-400 text-yellow-900 text-xs font-bold">1</span>
            <span>Most Important Feature</span>
          </div>
        </div>
        </div>
      </Card>
    </div>
  );
};

// === FEATURE CARD SUB-COMPONENT ===

interface FeatureCardProps {
  feature: FeatureScore;
  isExpanded: boolean;
  onToggle: () => void;
  getPointsColor: (points: number) => string;
  getImportanceBadgeColor: (rank: number) => string;
}

const FeatureCard: React.FC<FeatureCardProps> = ({
  feature,
  isExpanded,
  onToggle,
  getPointsColor,
  getImportanceBadgeColor,
}) => {
  return (
    <div className="border border-gray-200 rounded-lg overflow-hidden">
      {/* Header - Always visible */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between p-4 bg-gray-50 hover:bg-gray-100 transition-colors"
      >
        <div className="flex items-center gap-3">
          <span className={`inline-flex items-center justify-center w-7 h-7 rounded-full text-sm font-bold
                          ${getImportanceBadgeColor(feature.importance_rank)}`}>
            {feature.importance_rank}
          </span>
          <div className="text-left">
            <p className="font-semibold text-[#1E3A5F]">{feature.feature_name}</p>
            <p className="text-sm text-gray-500">
              {feature.bins.length} bins • Weight: <span className="font-semibold">{feature.weight.toFixed(1)}%</span>
            </p>
          </div>
        </div>
        
        <div className="flex items-center gap-4">
          <div className="text-right">
            <p className="text-sm text-gray-500">Point Range</p>
            <p className="font-mono font-medium">
              <span className="text-gray-600">{feature.min_points}</span>
              {' to '}
              <span className="text-green-600">{feature.max_points}</span>
            </p>
          </div>
          <svg 
            className={`w-5 h-5 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`}
            fill="none" stroke="currentColor" viewBox="0 0 24 24"
          >
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {/* Expanded Content - Bin Table */}
      {isExpanded && (
        <div className="p-4 border-t border-gray-200">
          <div className="overflow-x-auto">
            <table className="w-full text-sm">
              <thead>
                <tr className="border-b border-gray-200">
                  <th className="text-left py-2 px-3 font-medium text-gray-600">Bin Label</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">
                    Input Value
                    <span className="block text-xs font-normal text-gray-400">(std log odds × -50)</span>
                  </th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">Points</th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">
                    Train Count
                    <span className="block text-xs font-normal text-gray-400">(Bad Rate)</span>
                  </th>
                  <th className="text-right py-2 px-3 font-medium text-gray-600">
                    Test Count
                    <span className="block text-xs font-normal text-gray-400">(Bad Rate)</span>
                  </th>
                </tr>
              </thead>
              <tbody>
                {feature.bins.map((bin, idx) => (
                  <tr 
                    key={bin.bin_index}
                    className={`border-b border-gray-100 ${idx % 2 === 0 ? 'bg-white' : 'bg-gray-50'}`}
                  >
                    <td className="py-3 px-3">
                      <span className="font-medium text-gray-800">{bin.bin_label}</span>
                    </td>
                    <td className="py-3 px-3 text-right font-mono text-gray-600">
                      {bin.input_value >= 0 ? '+' : ''}{bin.input_value.toFixed(1)}
                    </td>
                    <td className="py-3 px-3 text-right">
                      <span className={`inline-block px-3 py-1 rounded-full font-bold font-mono
                                       ${getPointsColor(bin.scaled_points)}`}>
                        {bin.scaled_points}
                      </span>
                    </td>
                    <td className="py-3 px-3 text-right">
                      {bin.count_train !== undefined && bin.count_train !== null ? (
                        <>
                          <span className="font-medium">{bin.count_train.toLocaleString()}</span>
                          {bin.bad_rate_train !== undefined && bin.bad_rate_train !== null && (
                            <span className="text-gray-400 ml-1">
                              ({typeof bin.bad_rate_train === 'number' && bin.bad_rate_train < 1 
                                ? (bin.bad_rate_train * 100).toFixed(1) 
                                : bin.bad_rate_train.toFixed(1)}%)
                            </span>
                          )}
                        </>
                      ) : (
                        <span className="text-gray-400">—</span>
                      )}
                    </td>
                    <td className="py-3 px-3 text-right">
                      {bin.count_test !== undefined && bin.count_test !== null ? (
                        <>
                          <span className="font-medium">{bin.count_test.toLocaleString()}</span>
                          {bin.bad_rate_test !== undefined && bin.bad_rate_test !== null && (
                            <span className="text-gray-400 ml-1">
                              ({typeof bin.bad_rate_test === 'number' && bin.bad_rate_test < 1 
                                ? (bin.bad_rate_test * 100).toFixed(1) 
                                : bin.bad_rate_test.toFixed(1)}%)
                            </span>
                          )}
                        </>
                      ) : (
                        <span className="text-gray-400">—</span>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          {/* Bin visualization - mini bar chart */}
          <div className="mt-4 pt-4 border-t border-gray-100">
            <p className="text-xs text-gray-500 mb-2">Points Distribution</p>
            <div className="flex items-end gap-1 h-16">
              {feature.bins.map((bin) => {
                const maxPoints = Math.max(...feature.bins.map(b => b.scaled_points));
                const height = maxPoints > 0 
                  ? (bin.scaled_points / maxPoints) * 100 
                  : 0;
                
                return (
                  <div
                    key={bin.bin_index}
                    className="flex-1 flex flex-col items-center"
                  >
                    <div 
                      className="w-full rounded-t transition-all bg-green-400"
                      style={{ height: `${height}%`, minHeight: '4px' }}
                      title={`${bin.bin_label}: ${bin.scaled_points} points`}
                    />
                    <span className="text-xs text-gray-400 mt-1 truncate w-full text-center">
                      {bin.input_value}
                    </span>
                  </div>
                );
              })}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ScorecardDisplay;

