import React, { useState, useCallback } from 'react';
import { useDropzone } from 'react-dropzone';
import { CloudArrowUpIcon, DocumentIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { Button, Alert, LoadingSpinner } from '../common';

interface FileUploadProps {
  onUpload: (file: File) => Promise<void>;
  isLoading?: boolean;
  error?: string;
}

export const FileUpload: React.FC<FileUploadProps> = ({ 
  onUpload, 
  isLoading = false,
  error 
}) => {
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [uploadProgress, setUploadProgress] = useState(0);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    if (acceptedFiles.length > 0) {
      setSelectedFile(acceptedFiles[0]);
    }
  }, []);

  const { getRootProps, getInputProps, isDragActive, isDragReject } = useDropzone({
    onDrop,
    accept: {
      'text/csv': ['.csv']
    },
    maxFiles: 1,
    maxSize: 100 * 1024 * 1024 // 100MB
  });

  const handleUpload = async () => {
    if (!selectedFile) return;
    
    try {
      await onUpload(selectedFile);
    } catch (err) {
      console.error('Upload failed:', err);
    }
  };

  const clearFile = () => {
    setSelectedFile(null);
    setUploadProgress(0);
  };

  return (
    <div className="space-y-4">
      {/* Dropzone */}
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
          Supported: .csv files up to 100MB
        </p>
      </div>

      {/* Selected File */}
      {selectedFile && (
        <div className="flex items-center gap-4 p-4 bg-gray-50 rounded-lg">
          <DocumentIcon className="w-10 h-10 text-[#1E3A5F]" />
          <div className="flex-1">
            <p className="font-medium text-gray-800">{selectedFile.name}</p>
            <p className="text-sm text-gray-500">
              {(selectedFile.size / (1024 * 1024)).toFixed(2)} MB
            </p>
          </div>
          <button
            onClick={clearFile}
            className="p-2 hover:bg-gray-200 rounded-lg"
          >
            <XMarkIcon className="w-5 h-5 text-gray-500" />
          </button>
        </div>
      )}

      {/* Progress */}
      {isLoading && (
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

      {/* Upload Button */}
      {selectedFile && !isLoading && (
        <Button 
          variant="primary" 
          onClick={handleUpload}
          className="w-full"
        >
          Upload and Analyze
        </Button>
      )}
    </div>
  );
};

export default FileUpload;

