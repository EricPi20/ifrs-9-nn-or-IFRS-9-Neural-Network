import React, { useState, useMemo } from 'react';
import { ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';

interface FeatureInfo {
  name: string;
  num_bins: number;
  unique_values: number[];
  min_value: number;
  max_value: number;
  correlation: number;
}

interface FeatureListProps {
  features: FeatureInfo[];
  selectedFeatures: string[];
  onSelectionChange: (selected: string[]) => void;
}

type SortKey = 'name' | 'num_bins' | 'correlation';

export const FeatureList: React.FC<FeatureListProps> = ({
  features,
  selectedFeatures,
  onSelectionChange
}) => {
  const [sortKey, setSortKey] = useState<SortKey>('correlation');
  const [sortAsc, setSortAsc] = useState(false);
  const [expandedFeature, setExpandedFeature] = useState<string | null>(null);

  const sortedFeatures = useMemo(() => {
    return [...features].sort((a, b) => {
      let cmp = 0;
      switch (sortKey) {
        case 'name':
          cmp = a.name.localeCompare(b.name);
          break;
        case 'num_bins':
          cmp = a.num_bins - b.num_bins;
          break;
        case 'correlation':
          cmp = Math.abs(a.correlation) - Math.abs(b.correlation);
          break;
      }
      return sortAsc ? cmp : -cmp;
    });
  }, [features, sortKey, sortAsc]);

  const handleSort = (key: SortKey) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc);
    } else {
      setSortKey(key);
      setSortAsc(false);
    }
  };

  const toggleFeature = (name: string) => {
    if (selectedFeatures.includes(name)) {
      onSelectionChange(selectedFeatures.filter(f => f !== name));
    } else {
      onSelectionChange([...selectedFeatures, name]);
    }
  };

  const selectAll = () => onSelectionChange(features.map(f => f.name));
  const deselectAll = () => onSelectionChange([]);
  const selectTopN = (n: number) => {
    const top = [...features]
      .sort((a, b) => Math.abs(b.correlation) - Math.abs(a.correlation))
      .slice(0, n)
      .map(f => f.name);
    onSelectionChange(top);
  };

  const SortIcon = ({ active, asc }: { active: boolean; asc: boolean }) => (
    <span className={`ml-1 ${active ? 'text-[#38B2AC]' : 'text-gray-400'}`}>
      {asc ? '↑' : '↓'}
    </span>
  );

  return (
    <div className="space-y-4">
      {/* Actions */}
      <div className="flex gap-2 flex-wrap">
        <button
          onClick={selectAll}
          className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
        >
          Select All
        </button>
        <button
          onClick={deselectAll}
          className="px-3 py-1 text-sm bg-gray-100 hover:bg-gray-200 rounded"
        >
          Deselect All
        </button>
        <button
          onClick={() => selectTopN(10)}
          className="px-3 py-1 text-sm bg-[#38B2AC] text-white hover:bg-[#2C9A94] rounded"
        >
          Top 10 by Correlation
        </button>
        <span className="ml-auto text-sm text-gray-500">
          {selectedFeatures.length} / {features.length} selected
        </span>
      </div>

      {/* Table */}
      <div className="border rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-gray-50">
            <tr>
              <th className="w-10 px-4 py-3">
                <input
                  type="checkbox"
                  checked={selectedFeatures.length === features.length}
                  onChange={() => 
                    selectedFeatures.length === features.length 
                      ? deselectAll() 
                      : selectAll()
                  }
                  className="rounded"
                />
              </th>
              <th 
                className="px-4 py-3 text-left cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('name')}
              >
                Feature Name
                <SortIcon active={sortKey === 'name'} asc={sortAsc} />
              </th>
              <th 
                className="px-4 py-3 text-center cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('num_bins')}
              >
                Bins
                <SortIcon active={sortKey === 'num_bins'} asc={sortAsc} />
              </th>
              <th className="px-4 py-3 text-center">Value Range</th>
              <th 
                className="px-4 py-3 text-center cursor-pointer hover:bg-gray-100"
                onClick={() => handleSort('correlation')}
              >
                Correlation
                <SortIcon active={sortKey === 'correlation'} asc={sortAsc} />
              </th>
            </tr>
          </thead>
          <tbody>
            {sortedFeatures.map((feat) => (
              <React.Fragment key={feat.name}>
                <tr 
                  className={`border-t hover:bg-gray-50 ${
                    selectedFeatures.includes(feat.name) ? 'bg-[#38B2AC]/5' : ''
                  }`}
                >
                  <td className="px-4 py-3">
                    <input
                      type="checkbox"
                      checked={selectedFeatures.includes(feat.name)}
                      onChange={() => toggleFeature(feat.name)}
                      className="rounded"
                    />
                  </td>
                  <td className="px-4 py-3">
                    <button
                      className="flex items-center gap-2 text-left font-medium text-gray-800 hover:text-[#1E3A5F]"
                      onClick={() => setExpandedFeature(
                        expandedFeature === feat.name ? null : feat.name
                      )}
                    >
                      {expandedFeature === feat.name 
                        ? <ChevronUpIcon className="w-4 h-4" />
                        : <ChevronDownIcon className="w-4 h-4" />
                      }
                      {feat.name}
                    </button>
                  </td>
                  <td className="px-4 py-3 text-center">{feat.num_bins}</td>
                  <td className="px-4 py-3 text-center font-mono text-xs">
                    [{feat.min_value.toFixed(0)}, {feat.max_value.toFixed(0)}]
                  </td>
                  <td className="px-4 py-3">
                    <div className="flex items-center justify-center gap-2">
                      <div className="w-16 h-2 bg-gray-200 rounded-full overflow-hidden">
                        <div
                          className={`h-full ${
                            feat.correlation >= 0 ? 'bg-[#38B2AC]' : 'bg-[#F56565]'
                          }`}
                          style={{ width: `${Math.abs(feat.correlation) * 100}%` }}
                        />
                      </div>
                      <span className={`font-mono text-xs ${
                        Math.abs(feat.correlation) > 0.2 
                          ? 'text-[#1E3A5F] font-semibold' 
                          : 'text-gray-500'
                      }`}>
                        {feat.correlation.toFixed(3)}
                      </span>
                    </div>
                  </td>
                </tr>
                
                {/* Expanded: Show bin values */}
                {expandedFeature === feat.name && (
                  <tr className="bg-gray-50">
                    <td colSpan={5} className="px-8 py-4">
                      <div className="text-xs">
                        <span className="font-medium text-gray-600">
                          Bin Values (Standardized Log Odds × -50):
                        </span>
                        <div className="flex flex-wrap gap-2 mt-2">
                          {feat.unique_values.map((val, idx) => (
                            <span 
                              key={idx}
                              className={`px-2 py-1 rounded font-mono ${
                                val >= 0 
                                  ? 'bg-red-100 text-red-700' 
                                  : 'bg-green-100 text-green-700'
                              }`}
                            >
                              {val.toFixed(1)}
                            </span>
                          ))}
                        </div>
                        <p className="mt-2 text-gray-500">
                          Note: Negative values = lower risk, Positive values = higher risk
                        </p>
                      </div>
                    </td>
                  </tr>
                )}
              </React.Fragment>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};

export default FeatureList;

