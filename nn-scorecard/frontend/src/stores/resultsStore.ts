// frontend/src/stores/resultsStore.ts

export interface TrainingResult {
  job_id: string;
  created_at: string;
  segment: string;
  file_name?: string;
  config: {
    learning_rate: number;
    hidden_layers: number[];
    activation: string;
    epochs: number;
    batch_size: number;
    dropout_rate: number;
    l2_lambda: number;
    skip_connection: boolean;
    random_seed: number;
    test_size: number;
    selected_features: string[];
    stratified_split: boolean;
    early_stopping_enabled: boolean;
    early_stopping_patience: number;
    loss_type: string;
    loss_alpha?: number;
    auc_gamma?: number;
  };
  metrics: {
    test_auc: number;
    test_ar: number;
    test_ks: number;
    train_auc: number;
    train_ar: number;
    final_loss: number;
  };
  data_stats?: {
    n_train: number;
    n_test: number;
    bad_rate: number;
  };
}

const STORAGE_KEY = 'rift_training_results';

export const loadResults = (): TrainingResult[] => {
  try {
    const stored = localStorage.getItem(STORAGE_KEY);
    return stored ? JSON.parse(stored) : [];
  } catch (error) {
    console.error('Failed to load results:', error);
    return [];
  }
};

export const saveResults = (results: TrainingResult[]): void => {
  try {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(results));
  } catch (error) {
    console.error('Failed to save results:', error);
  }
};

export const addResult = (result: TrainingResult): TrainingResult[] => {
  const results = loadResults();
  // Check if already exists
  const existingIndex = results.findIndex(r => r.job_id === result.job_id);
  if (existingIndex >= 0) {
    results[existingIndex] = result;
  } else {
    results.unshift(result); // Add to beginning (newest first)
  }
  saveResults(results);
  return results;
};

export const deleteResult = (job_id: string): TrainingResult[] => {
  let results = loadResults();
  results = results.filter(r => r.job_id !== job_id);
  saveResults(results);
  return results;
};

export const getResult = (job_id: string): TrainingResult | undefined => {
  const results = loadResults();
  return results.find(r => r.job_id === job_id);
};

export const clearAllResults = (): void => {
  localStorage.removeItem(STORAGE_KEY);
};

