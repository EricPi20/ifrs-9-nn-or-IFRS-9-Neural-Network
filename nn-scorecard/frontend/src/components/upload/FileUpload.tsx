/**
 * File Upload Component
 * 
 * Handles CSV file uploads for credit risk data with drag-and-drop support.
 */

import React, { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, DocumentIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { useApi } from '../../hooks/useApi';
import { api } from '../../services/api';
import { Button, Alert } from '../common';
import { SegmentSelector } from './SegmentSelector';
import { FeatureList } from './FeatureList';
import { UploadResponse } from '../../types';

export const FileUpload: React.FC = () => {
  const navigate = useNavigate();
  const { uploadFile, loading, error } = useApi();
  const uploadCompleteRef = useRef(false);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);
  
  // Load persisted upload response from sessionStorage
  const [uploadResponse, setUploadResponse] = useState<UploadResponse | null>(() => {
    try {
      const stored = sessionStorage.getItem('uploadResponse');
      if (stored) {
        return JSON.parse(stored);
      }
    } catch (e) {
      console.error('Failed to load upload response from sessionStorage:', e);
    }
    return null;
  });
  
  const [selectedSegment, setSelectedSegment] = useState<string>(() => {
    try {
      const stored = sessionStorage.getItem('selectedSegment');
      return stored || 'ALL';
    } catch (e) {
      return 'ALL';
    }
  });

  const [selectedFeatures, setSelectedFeatures] = useState<string[]>(() => {
    try {
      const stored = sessionStorage.getItem('selectedFeatures');
      return stored ? JSON.parse(stored) : [];
    } catch (e) {
      return [];
    }
  });

  const [currentFeatures, setCurrentFeatures] = useState<UploadResponse['features']>(null);
  const [loadingFeatures, setLoadingFeatures] = useState(false);
  
  // Persist upload response to sessionStorage
  const updateUploadResponse = (response: UploadResponse | null) => {
    setUploadResponse(response);
    if (response) {
      try {
        sessionStorage.setItem('uploadResponse', JSON.stringify(response));
      } catch (e) {
        console.error('Failed to save upload response to sessionStorage:', e);
      }
    } else {
      try {
        sessionStorage.removeItem('uploadResponse');
      } catch (e) {
        console.error('Failed to remove upload response from sessionStorage:', e);
      }
    }
  };
  
  // Persist selected segment
  React.useEffect(() => {
    try {
      sessionStorage.setItem('selectedSegment', selectedSegment);
    } catch (e) {
      console.error('Failed to save selected segment to sessionStorage:', e);
    }
  }, [selectedSegment]);

  // Persist selected features
  React.useEffect(() => {
    try {
      sessionStorage.setItem('selectedFeatures', JSON.stringify(selectedFeatures));
    } catch (e) {
      console.error('Failed to save selected features to sessionStorage:', e);
    }
  }, [selectedFeatures]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    // Only allow dropping new files if there's no upload response
    // If there is an upload response, user should use "Upload New File" button
    if (acceptedFiles.length > 0 && !uploadResponse) {
      setSelectedFile(acceptedFiles[0]);
      setUploadProgress(0);
      // Clear previous state when new file is selected
      setSelectedSegment('ALL');
      setSelectedFeatures([]);
      setCurrentFeatures(null);
    }
  }, [uploadResponse]);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    maxSize: 500 * 1024 * 1024, // 500MB
    disabled: !!uploadResponse // Disable dropzone after successful upload
  });

  const handleUpload = async (e?: React.MouseEvent) => {
    e?.preventDefault();
    e?.stopPropagation();
    
    if (!selectedFile) return;
    
    try {
      // Clear previous state when uploading new file
      setSelectedFeatures([]);
      setCurrentFeatures(null);
      setSelectedSegment('ALL');
      
      // Clear any cached training config
      try {
        sessionStorage.removeItem('rift_training_config');
        sessionStorage.removeItem('selectedFeatures');
        sessionStorage.removeItem('selectedSegment');
      } catch (e) {
        console.error('Failed to clear sessionStorage:', e);
      }
      
      // Simulate progress for better UX
      setUploadProgress(0);
      const progressInterval = setInterval(() => {
        setUploadProgress((prev) => {
          if (prev >= 90) {
            clearInterval(progressInterval);
            return 90;
          }
          return prev + 10;
        });
      }, 200);

      const response = await uploadFile(selectedFile);
      
      clearInterval(progressInterval);
      setUploadProgress(100);
      
      console.log('=== UPLOAD RESPONSE DEBUG ===');
      console.log('Full response object:', JSON.stringify(response, null, 2));
      console.log('Response type:', typeof response);
      console.log('Response keys:', Object.keys(response || {}));
      console.log('=== END DEBUG ===');
      
      // Validate response structure
      if (!response || !response.file_id) {
        throw new Error('Invalid upload response received');
      }
      
      // Normalize response to expected format (handle both old and new backend formats)
      let normalizedResponse: UploadResponse = {
        file_id: response.file_id,
        filename: response.filename || response.file_name || 'unknown',
        // Handle both 'num_records' and 'rows' field names
        num_records: response.num_records ?? response.rows ?? 0,
        // Handle both 'num_features' and 'columns' field names (subtract target, segment, account_id)
        num_features: response.num_features ?? (response.columns ? Math.max(0, response.columns - 3) : 0),
        segments: response.segments || [],
        // Use segment_stats if provided, otherwise empty array (will be fetched if needed)
        segment_stats: response.segment_stats || [],
        features: response.features,
        target_stats: response.target_stats
      };
      
      // ALWAYS fetch segment stats if we have segments but no valid stats
      const needsFetch = normalizedResponse.segments && 
                        normalizedResponse.segments.length > 0 && 
                        (!normalizedResponse.segment_stats || 
                         normalizedResponse.segment_stats.length === 0 ||
                         normalizedResponse.segment_stats.every(s => !s || s.count === 0));
      
      console.log('🔍 SEGMENT STATS FETCH CHECK:');
      console.log('  - segments:', normalizedResponse.segments);
      console.log('  - segments length:', normalizedResponse.segments?.length);
      console.log('  - segment_stats:', normalizedResponse.segment_stats);
      console.log('  - segment_stats length:', normalizedResponse.segment_stats?.length);
      console.log('  - needsFetch:', needsFetch);
      
      if (needsFetch) {
        console.log('🚀 STARTING FETCH - segments exist but stats are missing/empty!');
        try {
          console.log('🔍 Fetching segment stats for file_id:', normalizedResponse.file_id);
          const API_BASE_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
          const segmentsUrl = `${API_BASE_URL}/api/upload/${normalizedResponse.file_id}/segments`;
          console.log('🔍 Fetching from URL:', segmentsUrl);
          
          const segmentsResponse = await fetch(segmentsUrl);
          console.log('📡 Response status:', segmentsResponse.status);
          console.log('📡 Response ok?', segmentsResponse.ok);
          
          if (segmentsResponse.ok) {
            const segmentsData = await segmentsResponse.json();
            console.log('📊 Raw segments data from API:', segmentsData);
            console.log('📊 Type of segmentsData:', typeof segmentsData);
            console.log('📊 Has segments property?', !!segmentsData?.segments);
            console.log('📊 segments is array?', Array.isArray(segmentsData?.segments));
            
            if (segmentsData && segmentsData.segments && Array.isArray(segmentsData.segments) && segmentsData.segments.length > 0) {
              // Map the response to match SegmentStat interface
              normalizedResponse.segment_stats = segmentsData.segments.map((stat: any) => ({
                segment: stat.segment || '',
                count: Number(stat.count) || 0,
                bad_count: Number(stat.bad_count) || 0,
                bad_rate: Number(stat.bad_rate) || 0
              }));
              console.log('✅ Fetched and mapped segment stats:', normalizedResponse.segment_stats);
              console.log('✅ First segment stat:', normalizedResponse.segment_stats[0]);
            } else {
              console.warn('⚠️ Segment stats response format unexpected or empty:', segmentsData);
              console.warn('⚠️ segmentsData type:', typeof segmentsData);
              console.warn('⚠️ segmentsData keys:', Object.keys(segmentsData || {}));
            }
          } else {
            const errorText = await segmentsResponse.text();
            console.error('❌ Segment stats fetch failed with status:', segmentsResponse.status);
            console.error('❌ Error response:', errorText);
            
            // If 404, the backend might not have the file in memory
            // Don't create placeholder - let user know they need to restart backend
            if (segmentsResponse.status === 404) {
              console.error('❌ Backend endpoint returned 404. Please restart the backend server for the fix to take effect.');
              console.error('❌ The backend needs to be restarted to use the updated segments endpoint.');
              // Don't create placeholder stats - show error instead
              throw new Error(`Backend endpoint not found. Please restart the backend server. Status: ${segmentsResponse.status}`);
            } else {
              throw new Error(`Failed to fetch segment stats: ${segmentsResponse.status} ${errorText}`);
            }
          }
        } catch (e) {
          console.error('❌ Failed to fetch segment stats:', e);
          console.error('❌ Error details:', e instanceof Error ? e.message : String(e));
          
          // Don't create placeholder stats - the backend should return real data
          console.error('❌ Could not fetch segment stats. Please ensure the backend server is running and has been restarted with the latest code.');
          // Keep empty array - UI will show segments but without stats
        }
      } else {
        // Define variables for logging why fetch was skipped
        const hasValidStats = normalizedResponse.segment_stats && 
                              Array.isArray(normalizedResponse.segment_stats) && 
                              normalizedResponse.segment_stats.length > 0 &&
                              !normalizedResponse.segment_stats.every(s => !s || s.count === 0);
        const hasSegments = normalizedResponse.segments && 
                           Array.isArray(normalizedResponse.segments) && 
                           normalizedResponse.segments.length > 0;
        
        console.log('ℹ️ Skipping fetch - reason:', {
          hasValidStats,
          hasSegments,
          why: hasValidStats ? 'stats already valid' : !hasSegments ? 'no segments' : 'unknown'
        });
      }
      
      // Log final segment_stats for debugging
      console.log('📋 Final segment_stats before storing:', normalizedResponse.segment_stats);
      console.log('📋 Final segment_stats length:', normalizedResponse.segment_stats?.length);
      console.log('📋 Will store segment_stats?', !!normalizedResponse.segment_stats);
      
      console.log('Normalized response:', normalizedResponse);
      console.log('Segment stats check:', {
        hasSegmentStats: !!normalizedResponse.segment_stats,
        segmentStatsLength: normalizedResponse.segment_stats?.length,
        segmentStats: normalizedResponse.segment_stats
      });
      
      // Store normalized upload response and set default segment
      updateUploadResponse(normalizedResponse);
      
      // Verify what was stored
      setTimeout(() => {
        const stored = sessionStorage.getItem('uploadResponse');
        if (stored) {
          const parsed = JSON.parse(stored);
          console.log('Stored uploadResponse segment_stats:', parsed.segment_stats);
        }
      }, 100);
      uploadCompleteRef.current = true;
      console.log('Upload response stored in state and sessionStorage');
      
      // Keep the selectedFile so it doesn't disappear
      // Don't clear selectedFile - we want to keep showing it
      
      if (normalizedResponse.segment_stats && normalizedResponse.segment_stats.length > 0) {
        // Prefer 'ALL' segment if it exists, otherwise use first segment
        const allSegment = normalizedResponse.segment_stats.find(s => s.segment === 'ALL');
        const defaultSegment = allSegment ? 'ALL' : normalizedResponse.segment_stats[0].segment;
        setSelectedSegment(defaultSegment);
        console.log('Default segment set to:', defaultSegment);
      } else if (normalizedResponse.segments && normalizedResponse.segments.length > 0) {
        // If no segment_stats but we have segments list, use first segment
        setSelectedSegment(normalizedResponse.segments[0]);
        console.log('Default segment set to (from segments list):', normalizedResponse.segments[0]);
      }

      // Don't initialize features here - wait for segment-specific fetch
      // The useEffect below will fetch features for the selected segment
      
      // Reset progress after a short delay
      setTimeout(() => {
        setUploadProgress(0);
        // Verify state is still there after timeout
        console.log('After timeout - uploadResponse should still exist:', !!uploadResponse);
      }, 500);
      
      // Force a re-render check
      console.log('Upload complete. State should persist now.');
      console.log('uploadCompleteRef.current:', uploadCompleteRef.current);
    } catch (err) {
      console.error('Upload failed:', err);
      setUploadProgress(0);
      updateUploadResponse(null);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
    updateUploadResponse(null);
    uploadCompleteRef.current = false;
    setSelectedSegment('ALL');
    setSelectedFeatures([]);
    setCurrentFeatures(null);
    try {
      sessionStorage.removeItem('uploadResponse');
      sessionStorage.removeItem('selectedSegment');
      sessionStorage.removeItem('selectedFeatures');
      sessionStorage.removeItem('rift_training_config');
      localStorage.removeItem('trainingConfig');
      localStorage.removeItem('rift_training_config');
    } catch (e) {
      console.error('Failed to clear sessionStorage:', e);
    }
  };

  const handleProceedToTraining = () => {
    if (!uploadResponse || !selectedSegment || selectedFeatures.length === 0) {
      return;
    }

    // Store configuration for training page
    const trainingConfig = {
      file_id: uploadResponse.file_id,
      filename: uploadResponse.filename,
      segment: selectedSegment,
      selected_features: selectedFeatures,
      num_records: uploadResponse.num_records,
      num_features: selectedFeatures.length,
      segment_stats: uploadResponse.segment_stats,
      features: currentFeatures?.filter(f => selectedFeatures.includes(f.name)) || []
    };

    try {
      sessionStorage.setItem('rift_training_config', JSON.stringify(trainingConfig));
      console.log('Training config stored:', trainingConfig);
    } catch (e) {
      console.error('Failed to save training config to sessionStorage:', e);
      return;
    }
    
    // Navigate to training configuration page
    navigate('/training');
  };
  
  // Initialize uploadCompleteRef from sessionStorage
  React.useEffect(() => {
    if (uploadResponse) {
      uploadCompleteRef.current = true;
    }
  }, []);

  // Initialize currentFeatures when uploadResponse is first loaded
  React.useEffect(() => {
    if (uploadResponse && !currentFeatures && uploadResponse.features && uploadResponse.features.length > 0) {
      // This will be replaced by the segment-specific fetch, but provides initial data
      setCurrentFeatures(uploadResponse.features);
    }
  }, [uploadResponse]);

  // Debug: Log state changes
  React.useEffect(() => {
    console.log('FileUpload state:', {
      selectedFile: selectedFile?.name,
      uploadResponse: uploadResponse ? 'exists' : 'null',
      uploadResponseFileId: uploadResponse?.file_id,
      selectedSegment,
      loading
    });
  }, [selectedFile, uploadResponse, selectedSegment, loading]);
  
  // Monitor uploadResponse changes specifically
  React.useEffect(() => {
    if (uploadResponse) {
      console.log('✅ Upload response is set!', {
        fileId: uploadResponse.file_id,
        filename: uploadResponse.filename,
        hasSegmentStats: !!uploadResponse.segment_stats,
        segmentStatsCount: uploadResponse.segment_stats?.length || 0
      });
    } else {
      console.log('❌ Upload response is null/cleared');
    }
  }, [uploadResponse]);

  // Fetch features when segment changes OR when file is uploaded
  React.useEffect(() => {
    if (!uploadResponse?.file_id || !selectedSegment) {
      return;
    }

    const fetchFeaturesForSegment = async () => {
      setLoadingFeatures(true);
      try {
        console.log(`🔄 [FileUpload] Fetching features for segment: ${selectedSegment}, file_id: ${uploadResponse.file_id}`);
        const featuresData = await api.getFeatures(uploadResponse.file_id, selectedSegment);
        console.log(`✅ [FileUpload] Features fetched for segment ${selectedSegment}:`, featuresData);
        
        if (featuresData.features && featuresData.features.length > 0) {
          // Ensure min_value and max_value are set (calculate from unique_values if missing)
          const featuresWithMinMax = featuresData.features.map(f => ({
            ...f,
            min_value: f.min_value ?? (f.unique_values.length > 0 ? Math.min(...f.unique_values) : 0),
            max_value: f.max_value ?? (f.unique_values.length > 0 ? Math.max(...f.unique_values) : 0),
          }));
          
          // Always set current features from the NEW file/segment
          setCurrentFeatures(featuresWithMinMax);
          
          // Auto-select all features by default when fetching for a new segment or file
          const allFeatureNames = featuresWithMinMax.map(f => f.name);
          setSelectedFeatures(allFeatureNames);
          
          console.log(`[FileUpload] Auto-selected ${allFeatureNames.length} features for segment ${selectedSegment}`);
        }
      } catch (error) {
        console.error('❌ [FileUpload] Failed to fetch features for segment:', error);
        // Don't fallback to old features - clear them instead
        setCurrentFeatures(null);
        setSelectedFeatures([]);
      } finally {
        setLoadingFeatures(false);
      }
    };

    fetchFeaturesForSegment();
  }, [uploadResponse?.file_id, selectedSegment]);
  
  // Prevent accidental navigation or page reload
  React.useEffect(() => {
    const handleBeforeUnload = (e: BeforeUnloadEvent) => {
      if (uploadResponse) {
        // Don't prevent, but log
        console.log('Page unload detected, but uploadResponse exists in sessionStorage');
      }
    };
    
    window.addEventListener('beforeunload', handleBeforeUnload);
    return () => window.removeEventListener('beforeunload', handleBeforeUnload);
  }, [uploadResponse]);

  return (
    <div className="space-y-4">
      {/* Dropzone - Hide or minimize after successful upload */}
      {!uploadResponse && (
      <div
        {...getRootProps()}
        className={`
          border-2 border-dashed rounded-xl p-8 text-center cursor-pointer
          transition-colors duration-200
          ${isDragActive && !isDragReject 
            ? 'border-[#38B2AC] bg-[#38B2AC]/5' 
            : isDragReject
            ? 'border-red-400 bg-red-50'
            : 'border-gray-300 hover:border-[#38B2AC] hover:bg-gray-50'
          }
        `}
      >
        <input {...getInputProps()} />
        
        <CloudArrowUpIcon className={`
          w-16 h-16 mx-auto mb-4
          ${isDragActive ? 'text-[#38B2AC]' : 'text-gray-400'}
        `} />
        
        {isDragActive && !isDragReject ? (
          <p className="text-lg text-[#38B2AC] font-medium">Drop file here...</p>
        ) : isDragReject ? (
          <p className="text-lg text-red-500 font-medium">Only CSV files are accepted</p>
        ) : (
          <>
            <p className="text-lg text-gray-700 font-medium">
              Drag & drop your CSV file here
            </p>
            <p className="text-sm text-gray-500 mt-2">
              or click to browse
            </p>
          </>
        )}
        
        <p className="text-xs text-gray-400 mt-4">
          Supported: .csv files up to 500MB
        </p>
      </div>
      )}

      {/* Selected File - Show file info or uploaded file info */}
      {(selectedFile || uploadResponse) && (
        <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
          <DocumentIcon className="w-10 h-10 text-[#1E3A5F]" />
          <div className="flex-1">
            <p className="font-medium text-gray-800">
              {uploadResponse?.filename || selectedFile?.name}
            </p>
            <p className="text-sm text-gray-500">
              {uploadResponse 
                ? `${(uploadResponse.num_records || 0).toLocaleString()} records, ${uploadResponse.num_features || 0} features`
                : `${((selectedFile?.size || 0) / (1024 * 1024)).toFixed(2)} MB`
              }
            </p>
          </div>
          {!uploadResponse && (
          <button
            onClick={clearFile}
            className="p-2 hover:bg-gray-200 rounded-lg"
            disabled={loading}
          >
            <XMarkIcon className="w-5 h-5 text-gray-500" />
          </button>
          )}
        </div>
      )}

      {/* Progress */}
      {loading && (
        <div className="space-y-2">
          <div className="flex justify-between text-sm">
            <span>Uploading...</span>
            <span>{uploadProgress}%</span>
          </div>
          <div className="h-2 bg-gray-200 rounded-full overflow-hidden">
            <div 
              className="h-full bg-[#38B2AC] transition-all duration-300"
              style={{ width: `${uploadProgress}%` }}
            />
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <Alert variant="error" title="Upload Failed">
          {error}
        </Alert>
      )}

      {/* Success Message - Show if we have uploadResponse OR if uploadCompleteRef is true */}
      {(uploadResponse || uploadCompleteRef.current) && !loading && (
        uploadResponse ? (
          (uploadResponse.num_records === 0 || uploadResponse.num_features === 0) ? (
            <Alert variant="warning" title="File Processed with Issues">
              <div className="space-y-2">
                <p>File uploaded but contains no data or missing required columns.</p>
                <div className="text-sm space-y-1">
                  <p><strong>Records:</strong> {(uploadResponse.num_records || 0).toLocaleString()}</p>
                  <p><strong>Features:</strong> {uploadResponse.num_features || 0}</p>
                  <p className="mt-2 text-xs text-gray-600">
                    <strong>Required:</strong> CSV must have a <code className="bg-gray-100 px-1 rounded">target</code> column (0/1 values) and feature columns.
                  </p>
                </div>
              </div>
            </Alert>
          ) : (
            <Alert variant="success" title="Upload Successful">
              <div className="space-y-2">
                <p>File analyzed successfully!</p>
                <div className="text-sm space-y-1">
                  <p><strong>Records:</strong> {(uploadResponse.num_records || 0).toLocaleString()}</p>
                  <p><strong>Features:</strong> {uploadResponse.num_features || 0}</p>
                  {uploadResponse.target_stats && (
                    <p><strong>Overall Bad Rate:</strong> {(uploadResponse.target_stats.bad_rate * 100).toFixed(2)}%</p>
                  )}
                </div>
              </div>
            </Alert>
          )
        ) : (
          <Alert variant="info" title="Upload Complete">
            <p>Loading analysis results...</p>
          </Alert>
        )
      )}

      {/* Segment Selector */}
      {uploadResponse && uploadResponse.segment_stats && uploadResponse.segment_stats.length > 0 && (
        <div className="mt-6">
          <SegmentSelector
            segments={uploadResponse.segment_stats}
            selected={selectedSegment}
            onChange={setSelectedSegment}
          />
        </div>
      )}

      {/* Feature List */}
      {uploadResponse && currentFeatures && currentFeatures.length > 0 && (
        <div className="mt-6">
          <div className="flex items-center justify-between mb-4">
            <h3 className="text-lg font-semibold text-gray-800">Select Features for Training</h3>
            {loadingFeatures && (
              <span className="text-sm text-gray-500">Loading features for {selectedSegment}...</span>
            )}
          </div>
          <FeatureList
            features={currentFeatures}
            selectedFeatures={selectedFeatures}
            onSelectionChange={setSelectedFeatures}
          />
        </div>
      )}

      {/* Training Configuration Summary and Proceed Button */}
      {uploadResponse && (
        <div className="mt-8 space-y-4">
          {/* Summary */}
          <div className="bg-gray-50 rounded-lg p-4 flex justify-between items-center">
            <div>
              <p className="text-sm text-gray-600">Ready to configure training:</p>
              <p className="font-semibold text-[#1E3A5F]">
                {selectedSegment} segment • {selectedFeatures.length} features selected
              </p>
            </div>
            <div className="text-right text-sm text-gray-500">
              {uploadResponse.num_records?.toLocaleString()} records
            </div>
          </div>

          {/* Proceed Button */}
          <button
            onClick={handleProceedToTraining}
            disabled={!selectedSegment || selectedFeatures.length === 0}
            className={`
              w-full py-4 px-6 rounded-lg font-semibold text-lg
              flex items-center justify-center gap-2
              transition-all duration-200
              ${selectedFeatures.length > 0 && selectedSegment
                ? 'bg-[#1E3A5F] hover:bg-[#2D4A6F] text-white cursor-pointer'
                : 'bg-gray-300 text-gray-500 cursor-not-allowed'
              }
            `}
          >
            <span>Proceed to Training Configuration</span>
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 7l5 5m0 0l-5 5m5-5H6" />
            </svg>
          </button>

          {/* Validation message if needed */}
          {selectedFeatures.length === 0 && (
            <p className="text-center text-sm text-amber-600">
              Please select at least one feature to proceed
            </p>
          )}
        </div>
      )}

      {/* Upload Button */}
      {selectedFile && !loading && !uploadResponse && (
        <Button 
          type="button"
          variant="primary" 
          onClick={(e) => {
            e.preventDefault();
            e.stopPropagation();
            handleUpload(e);
          }}
          className="w-full"
        >
          Upload and Analyze
        </Button>
      )}

      {/* Reset Button */}
      {uploadResponse && (
        <Button 
          variant="secondary" 
          onClick={clearFile}
          className="w-full"
        >
          Upload New File
        </Button>
      )}
    </div>
  );
};

