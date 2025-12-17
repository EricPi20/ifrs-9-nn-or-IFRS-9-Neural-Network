/**
 * API Hook
 * 
 * Custom React hook for interacting with the backend API.
 */

import { useState } from 'react';
import { api } from '../services/api';
import { TrainingConfig, ModelResults, UploadResponse, TrainingResponse } from '../types';

export const useApi = () => {
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const uploadFile = async (file: File): Promise<UploadResponse> => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.uploadFile(file);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Upload failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const startTraining = async (
    fileId: string,
    config: TrainingConfig
  ): Promise<TrainingResponse> => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.startTraining(fileId, config);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Training start failed';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getTrainingStatus = async (jobId: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getTrainingStatus(jobId);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get training status';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const cancelTraining = async (jobId: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.cancelTraining(jobId);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to cancel training';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getTrainingHistory = async (jobId: string) => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getTrainingHistory(jobId);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get training history';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  const getModelResults = async (jobId: string): Promise<ModelResults> => {
    setLoading(true);
    setError(null);
    try {
      const response = await api.getModelResults(jobId);
      return response;
    } catch (err: unknown) {
      const errorMessage = err instanceof Error ? err.message : 'Failed to get model results';
      setError(errorMessage);
      throw err;
    } finally {
      setLoading(false);
    }
  };

  return {
    uploadFile,
    startTraining,
    getTrainingStatus,
    cancelTraining,
    getTrainingHistory,
    getModelResults,
    loading,
    error,
  };
};

