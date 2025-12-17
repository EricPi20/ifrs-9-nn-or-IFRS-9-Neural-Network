/**
 * TypeScript Type Definitions
 * 
 * Centralized type definitions for the application.
 */

export interface SegmentStat {
  segment: string;
  count: number;
  bad_count: number;
  bad_rate: number;
}

export interface FeatureInfo {
  name: string;
  num_bins: number;
  unique_values: number[];
  min_value: number;
  max_value: number;
  mean_value?: number;
  correlation: number;
}

export interface UploadResponse {
  filename: string;
  file_id: string;
  num_records: number;
  num_features: number;
  segments: string[];
  segment_stats: SegmentStat[];
  features?: FeatureInfo[];
  target_stats?: {
    good_count: number;
    bad_count: number;
    bad_rate: number;
  };
}

export interface NeuralNetworkConfig {
  model_type: 'linear' | 'neural_network';
  hidden_layers: number[];
  activation: 'relu' | 'leaky_relu' | 'elu' | 'selu' | 'tanh';
  dropout_rate: number;
  use_batch_norm: boolean;
  skip_connection?: boolean;  // Skip connection from input to output
}

export interface RegularizationConfig {
  l1_lambda: number;
  l2_lambda: number;
  gradient_clip_norm: number;
}

export interface LossConfig {
  loss_type: 'bce' | 'pairwise_auc' | 'soft_auc' | 'wmw' | 'combined';
  loss_alpha: number;
  auc_gamma: number;
}

export interface EarlyStoppingConfig {
  enabled: boolean;
  patience: number;
  min_delta: number;
  monitor?: string;  // Optional: which metric to monitor (default: 'test_ar')
}

export interface TrainingConfig {
  segment: string;
  test_size: number;
  random_seed?: number;
  selected_features?: string[] | null;
  network: NeuralNetworkConfig;
  regularization: RegularizationConfig;
  loss: LossConfig;
  learning_rate: number;
  batch_size: number;
  epochs: number;
  early_stopping: EarlyStoppingConfig;
  use_class_weights: boolean;
}

export interface TrainingRequest {
  file_id: string;
  config: TrainingConfig;
}

export interface TrainingResponse {
  job_id: string;
  status: string;
  message: string;
  created_at: string;
}

export interface ModelResults {
  job_id: string;
  segment?: string;
  train_ar: number;
  test_ar: number;
  train_auc: number;
  test_auc: number;
  train_ks: number;
  test_ks: number;
  feature_importance: Record<string, number>;
  training_time: number;
  created_at: string;
}

export interface ScorecardResponse {
  job_id: string;
  segment?: string;
  scorecard_data: Record<string, unknown>;
  score_range: { min: number; max: number };
  created_at: string;
}

export interface ScoreRequest {
  job_id: string;
  features: Record<string, number>;
}

export interface ScoreResponse {
  score: number;
  probability: number;
  segment?: string;
}

// Bin-level score information
export interface BinScore {
  bin_index: number;
  bin_label: string;           // e.g., "Excellent (No late payments)"
  input_value: number;         // Original CSV value (std log odds × -50)
  raw_points: number;          // Weight × input_value
  scaled_points: number;       // Points scaled to contribute to 0-100 range
  count_train: number;         // Number of records in training set
  count_test: number;          // Number of records in test set
  bad_rate_train: number;      // Bad rate in training set
  bad_rate_test: number;       // Bad rate in test set
}

// Feature-level score information
export interface FeatureScore {
  feature_name: string;
  weight: number;              // Adjusted weight for original scale
  weight_normalized: number;   // Original weight from model
  importance_rank: number;     // 1 = most important
  bins: BinScore[];
  min_points: number;          // Minimum possible points for this feature
  max_points: number;          // Maximum possible points for this feature
}

// Complete scorecard
export interface Scorecard {
  job_id: string;
  segment: string;
  model_type: string;          // 'linear' or 'neural_network'
  created_at: string;
  
  // Score range
  score_min: number;           // 0
  score_max: number;           // 100
  min_possible_score: number;  // Actual minimum achievable
  max_possible_score: number;  // Actual maximum achievable
  
  // Scaling parameters
  raw_min: number;
  raw_max: number;
  scale_factor: number;
  offset: number;
  input_scale_factor: number;  // 50.0 (for documentation)
  
  // Features
  features: FeatureScore[];
  
  // Model performance
  metrics: {
    train_auc: number;
    test_auc: number;
    train_ar: number;
    test_ar: number;
    train_ks: number;
    test_ks: number;
  };
  
  // Data statistics (optional)
  data_stats?: {
    n_samples?: number;
    n_train?: number;
    n_test?: number;
    bad_rate_overall?: number;
    bad_rate_train?: number;
    bad_rate_test?: number;
    segment?: string;
    segment_column?: string;
  };
}

