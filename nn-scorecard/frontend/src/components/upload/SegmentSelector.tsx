import React from 'react';

interface SegmentStat {
  segment: string;
  count: number;
  bad_count: number;
  bad_rate: number;
}

interface SegmentSelectorProps {
  segments: SegmentStat[];
  selected: string;
  onChange: (segment: string) => void;
}

export const SegmentSelector: React.FC<SegmentSelectorProps> = ({
  segments,
  selected,
  onChange
}) => {
  return (
    <div className="space-y-3">
      <label className="block text-sm font-medium text-gray-700">
        Select Segment for Training
      </label>
      
      <div className="space-y-2">
        {segments.map((seg) => (
          <label
            key={seg.segment}
            className={`
              flex items-center gap-4 p-4 rounded-lg border-2 cursor-pointer
              transition-all duration-200
              ${selected === seg.segment
                ? 'border-[#1E3A5F] bg-[#1E3A5F]/5'
                : 'border-gray-200 hover:border-gray-300'
              }
            `}
          >
            <input
              type="radio"
              name="segment"
              value={seg.segment}
              checked={selected === seg.segment}
              onChange={() => onChange(seg.segment)}
              className="w-4 h-4 text-[#1E3A5F] focus:ring-[#38B2AC]"
            />
            
            <div className="flex-1">
              <span className="font-medium text-gray-800">{seg.segment}</span>
            </div>
            
            <div className="flex gap-6 text-sm">
              <div className="text-center">
                <div className="text-gray-500">Records</div>
                <div className="font-semibold">{seg.count.toLocaleString()}</div>
              </div>
              <div className="text-center">
                <div className="text-gray-500">Bads</div>
                <div className="font-semibold">{seg.bad_count.toLocaleString()}</div>
              </div>
              <div className="text-center">
                <div className="text-gray-500">Bad Rate</div>
                <div className="font-semibold text-[#F56565]">
                  {(seg.bad_rate * 100).toFixed(1)}%
                </div>
              </div>
            </div>
            
            {/* Visual Bad Rate Bar */}
            <div className="w-24">
              <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
                <div
                  className="h-full bg-[#F56565]"
                  style={{ width: `${Math.min(seg.bad_rate * 100 * 5, 100)}%` }}
                />
              </div>
            </div>
          </label>
        ))}
      </div>
    </div>
  );
};

export default SegmentSelector;

