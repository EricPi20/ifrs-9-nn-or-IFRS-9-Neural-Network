import React, { useEffect, useMemo } from 'react';
import { useForm, Controller } from 'react-hook-form';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import { Button, Card } from '../common';
import { TrainingConfig } from '../../types';

interface Props {
  defaultValues?: Partial<TrainingConfig>;
  uploadedFilePath: string;
  selectedFeatures: string[];
  segment: string;  // Segment from FileUpload (read-only)
  onSubmit: (config: TrainingConfig) => void;
  isSubmitting?: boolean;
}

const Tooltip: React.FC<{ text: string }> = ({ text }) => (
  <div className="group relative inline-block ml-1">
    <InformationCircleIcon className="w-4 h-4 text-gray-400 cursor-help" />
    <div className="invisible group-hover:visible absolute z-10 w-64 p-2 bg-gray-800 text-white text-xs rounded-lg -top-2 left-6">
      {text}
    </div>
  </div>
);

// Quick config presets
const PRESETS = {
  'quick_test': {
    name: 'Quick Test',
    description: 'Fast training for testing (10 epochs)',
    config: {
      epochs: 10,
      learning_rate: 0.01,
      network: { 
        hidden_layers: [8],
        model_type: 'neural_network' as const
      },
      early_stopping: {
        enabled: false,
        patience: 10,
        min_delta: 0.001,
        monitor: 'test_ar'
      },
    }
  },
  'balanced': {
    name: 'Balanced',
    description: 'Good balance of speed and accuracy',
    config: {
      epochs: 50,
      learning_rate: 0.001,
      network: { 
        hidden_layers: [16, 8],
        model_type: 'neural_network' as const
      },
      early_stopping: {
        enabled: true,
        patience: 10,
        min_delta: 0.001,
        monitor: 'test_ar'
      },
    }
  },
  'thorough': {
    name: 'Thorough',
    description: 'More epochs, aggressive regularization',
    config: {
      epochs: 200,
      learning_rate: 0.0005,
      network: { 
        hidden_layers: [32, 16, 8],
        model_type: 'neural_network' as const,
        dropout_rate: 0.4
      },
      regularization: { 
        l2_lambda: 0.01,
        l1_lambda: 0.0,
        gradient_clip_norm: 1.0
      },
      early_stopping: {
        enabled: true,
        patience: 20,
        min_delta: 0.001,
        monitor: 'test_ar'
      },
    }
  }
};

export const TrainingConfigForm: React.FC<Props> = ({
  defaultValues,
  uploadedFilePath,
  selectedFeatures,
  segment,
  onSubmit,
  isSubmitting
}) => {
  // Merge defaultValues with form defaults
  const formDefaults = useMemo<TrainingConfig>(() => ({
    segment: segment,  // Use segment from props (FileUpload)
    test_size: 0.30,
    random_seed: 42,
    selected_features: selectedFeatures,  // Use features from props (FileUpload)
    network: {
      model_type: 'neural_network',
      hidden_layers: [64, 32],
      activation: 'relu',
      dropout_rate: 0.2,
      use_batch_norm: true,
      skip_connection: false
    },
    regularization: {
      l1_lambda: 0.0,
      l2_lambda: 0.01,
      gradient_clip_norm: 1.0
    },
    loss: {
      loss_type: 'combined',
      loss_alpha: 0.3,
      auc_gamma: 2.0
    },
    learning_rate: 0.001,
    batch_size: 256,
    epochs: 100,
    early_stopping: {
      enabled: false,
      patience: 10,
      min_delta: 0.001,
      monitor: 'test_ar'
    },
    use_class_weights: true,
    ...defaultValues
  }), [selectedFeatures, segment, defaultValues]);

  const { control, handleSubmit, watch, setValue, reset, formState: { errors } } = useForm<TrainingConfig>({
    defaultValues: formDefaults
  });

  // Reset form when defaultValues changes (e.g., when returning from training)
  useEffect(() => {
    if (defaultValues) {
      const mergedDefaults: TrainingConfig = {
        segment: segment,  // Always use segment from props (FileUpload)
        test_size: defaultValues.test_size ?? 0.30,
        random_seed: defaultValues.random_seed ?? 42,
        selected_features: selectedFeatures,  // Always use features from props (FileUpload)
        network: {
          model_type: 'neural_network',
          hidden_layers: [64, 32],
          activation: 'relu',
          dropout_rate: 0.2,
          use_batch_norm: true,
          skip_connection: false,
          ...defaultValues.network,
        },
        regularization: {
          l1_lambda: 0.0,
          l2_lambda: 0.01,
          gradient_clip_norm: 1.0,
          ...defaultValues.regularization,
        },
        loss: {
          loss_type: 'combined',
          loss_alpha: 0.3,
          auc_gamma: 2.0,
          ...defaultValues.loss,
        },
        learning_rate: defaultValues.learning_rate ?? 0.001,
        batch_size: defaultValues.batch_size ?? 256,
        epochs: defaultValues.epochs ?? 100,
        early_stopping: {
          enabled: false,
          patience: 10,
          min_delta: 0.001,
          monitor: 'test_ar',
          ...defaultValues.early_stopping,
        },
        use_class_weights: defaultValues.use_class_weights ?? true,
      };
      reset(mergedDefaults);
    }
  }, [defaultValues, selectedFeatures, segment, reset]);

  // Show indicator that previous config is loaded
  const isPreviousConfig = defaultValues !== null && defaultValues !== undefined;

  const modelType = watch('network.model_type');
  const lossType = watch('loss.loss_type');
  const hiddenLayers = watch('network.hidden_layers');
  const earlyStoppingEnabled = watch('early_stopping.enabled');
  const skipConnectionEnabled = watch('network.skip_connection');
  const watchedSelectedFeatures = watch('selected_features');

  const addLayer = () => {
    if (hiddenLayers.length < 5) {
      const newSize = Math.max(16, Math.floor((hiddenLayers[hiddenLayers.length - 1] || 64) / 2));
      setValue('network.hidden_layers', [...hiddenLayers, newSize]);
    }
  };

  const removeLayer = () => {
    if (hiddenLayers.length > 1) {
      setValue('network.hidden_layers', hiddenLayers.slice(0, -1));
    }
  };

  const handleReset = () => {
    reset(formDefaults);
  };

  const applyPreset = (presetKey: keyof typeof PRESETS) => {
    const preset = PRESETS[presetKey];
    Object.entries(preset.config).forEach(([field, value]) => {
      if (field === 'network' && typeof value === 'object' && value !== null) {
        Object.entries(value).forEach(([nestedField, nestedValue]) => {
          if (nestedField === 'hidden_layers') {
            setValue('network.hidden_layers', nestedValue as number[]);
          } else if (nestedField === 'model_type') {
            setValue('network.model_type', nestedValue as 'linear' | 'neural_network');
          } else {
            setValue(`network.${nestedField}` as any, nestedValue);
          }
        });
      } else if (field === 'regularization' && typeof value === 'object' && value !== null) {
        Object.entries(value).forEach(([nestedField, nestedValue]) => {
          setValue(`regularization.${nestedField}` as any, nestedValue);
        });
      } else if (field === 'early_stopping' && typeof value === 'object' && value !== null) {
        Object.entries(value).forEach(([nestedField, nestedValue]) => {
          setValue(`early_stopping.${nestedField}` as any, nestedValue);
        });
      } else {
        setValue(field as keyof TrainingConfig, value as any);
      }
    });
  };

  const onSubmitWithErrorHandling = async (data: TrainingConfig) => {
    // Remove segment from form data - it comes from props (FileUpload)
    // Also ensure selected_features comes from props, not form
    const formConfig: TrainingConfig = {
      ...data,
      segment: segment,  // FROM PROPS (FileUpload)
      selected_features: selectedFeatures,  // FROM PROPS (FileUpload)
    };
    
    console.log('='.repeat(50));
    console.log('[FORM] Submitting training');
    console.log('[FORM] Segment from props (FileUpload):', segment);
    console.log('[FORM] Features from props (FileUpload):', selectedFeatures.length);
    console.log('[FORM] Form config (without segment/features override):', {
      ...data,
      segment: '[FROM PROPS]',
      selected_features: '[FROM PROPS]'
    });
    console.log('[FORM] Final config being sent:', formConfig);
    console.log('='.repeat(50));
    
    try {
      await onSubmit(formConfig);
    } catch (error) {
      console.error('[FORM] Error in onSubmit callback:', error);
    }
  };

  const onError = (errors: any) => {
    console.error('Form validation errors:', errors);
  };

  return (
    <form onSubmit={handleSubmit(onSubmitWithErrorHandling, onError)} noValidate className="space-y-6">
      
      {/* Previous Config Indicator */}
      {isPreviousConfig && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4 flex items-center justify-between">
          <div className="flex items-center gap-2">
            <span className="text-blue-600">ℹ️</span>
            <span className="text-blue-800 text-sm">
              Previous configuration loaded. Modify as needed.
            </span>
          </div>
          <button
            type="button"
            onClick={handleReset}
            className="text-blue-600 text-sm hover:underline"
          >
            Reset to Defaults
          </button>
        </div>
      )}

      {/* Quick Presets */}
      <div className="bg-gray-50 border border-gray-200 rounded-lg p-4">
        <div className="flex items-center gap-3 mb-2">
          <span className="text-sm font-medium text-gray-700">Quick Presets:</span>
          {Object.entries(PRESETS).map(([key, preset]) => (
            <button
              key={key}
              type="button"
              onClick={() => applyPreset(key as keyof typeof PRESETS)}
              className="px-3 py-1 text-xs bg-white hover:bg-gray-100 border border-gray-300 rounded-full transition-colors"
              title={preset.description}
            >
              {preset.name}
            </button>
          ))}
        </div>
        <p className="text-xs text-gray-500">
          Click a preset to quickly apply common configurations
        </p>
      </div>
      
      {/* Section 1: Data Split & Reproducibility */}
      <Card title="🎲 Data Split & Reproducibility">
        {/* Display segment and features as read-only info from FileUpload */}
        <div className="mb-4 p-3 bg-blue-50 rounded-lg border border-blue-200">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm text-gray-600">Segment:</span>
            <span className="font-semibold text-blue-700">{segment}</span>
          </div>
          <div className="flex justify-between items-center">
            <span className="text-sm text-gray-600">Features:</span>
            <span className="font-semibold text-blue-700">{selectedFeatures.length} selected</span>
          </div>
          <div className="mt-2 pt-2 border-t border-blue-200">
            <span className="text-xs text-blue-600">
              ℹ️ Segment and features are configured in the File Upload page
            </span>
          </div>
        </div>
        
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <Controller
            name="test_size"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Test Size
                  <Tooltip text="Proportion of data held out for testing. 30% is recommended." />
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={0.1}
                    max={0.5}
                    step={0.05}
                    {...field}
                    onChange={e => field.onChange(parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <span className="font-mono w-16 text-right">
                    {(field.value * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
          />

          <Controller
            name="random_seed"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Random Seed
                  <Tooltip text="Seed for reproducibility. Same seed = same train/test split and weight initialization." />
                </label>
                <div className="flex gap-2">
                  <input
                    type="number"
                    min={0}
                    max={999999}
                    step={1}
                    {...field}
                    value={field.value ?? 42}
                    onChange={(e) => field.onChange(parseInt(e.target.value) || 42)}
                    className="flex-1 border border-gray-300 rounded-lg px-3 py-2"
                  />
                  <button
                    type="button"
                    onClick={() => field.onChange(Math.floor(Math.random() * 100000))}
                    className="px-3 py-2 bg-gray-100 hover:bg-gray-200 rounded-lg text-sm transition-colors"
                    title="Generate random seed"
                  >
                    🎲
                  </button>
                </div>
              </div>
            )}
          />
        </div>
      </Card>

      {/* Section 2: Model Architecture */}
      <Card title="🧠 Model Architecture">
        <div className="space-y-4">
          <Controller
            name="network.model_type"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Model Type
                </label>
                <div className="flex gap-4">
                  <label className={`flex-1 p-4 border-2 rounded-lg cursor-pointer ${
                    field.value === 'linear' ? 'border-[#1E3A5F] bg-[#1E3A5F]/5' : 'border-gray-200'
                  }`}>
                    <input
                      type="radio"
                      {...field}
                      value="linear"
                      checked={field.value === 'linear'}
                      className="mr-2"
                    />
                    <span className="font-medium">Linear</span>
                    <p className="text-xs text-gray-500 mt-1">
                      Equivalent to logistic regression. More interpretable.
                    </p>
                  </label>
                  
                  <label className={`flex-1 p-4 border-2 rounded-lg cursor-pointer ${
                    field.value === 'neural_network' ? 'border-[#1E3A5F] bg-[#1E3A5F]/5' : 'border-gray-200'
                  }`}>
                    <input
                      type="radio"
                      {...field}
                      value="neural_network"
                      checked={field.value === 'neural_network'}
                      className="mr-2"
                    />
                    <span className="font-medium">Neural Network</span>
                    <p className="text-xs text-gray-500 mt-1">
                      Captures non-linear patterns. Better performance.
                    </p>
                  </label>
                </div>
              </div>
            )}
          />

          {modelType === 'neural_network' && (
            <>
              {/* Hidden Layers */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Hidden Layers
                  <Tooltip text="Add neurons per layer. Recommend decreasing sizes (e.g., 64→32→16)." />
                </label>
                <div className="flex flex-wrap gap-2 items-center">
                  {hiddenLayers.map((size, idx) => (
                    <Controller
                      key={idx}
                      name={`network.hidden_layers.${idx}`}
                      control={control}
                      render={({ field }) => (
                        <div className="flex items-center">
                          <span className="text-xs text-gray-500 mr-1">L{idx + 1}:</span>
                          <input
                            type="number"
                            min={8}
                            max={512}
                            step="any"
                            {...field}
                            onChange={e => field.onChange(parseInt(e.target.value))}
                            className="w-20 border rounded px-2 py-1 text-center"
                          />
                        </div>
                      )}
                    />
                  ))}
                  <button
                    type="button"
                    onClick={addLayer}
                    disabled={hiddenLayers.length >= 5}
                    className="px-2 py-1 bg-[#38B2AC] text-white rounded disabled:opacity-50"
                  >
                    + Add
                  </button>
                  <button
                    type="button"
                    onClick={removeLayer}
                    disabled={hiddenLayers.length <= 1}
                    className="px-2 py-1 bg-gray-200 rounded disabled:opacity-50"
                  >
                    − Remove
                  </button>
                </div>
              </div>

              {/* Activation & Batch Norm */}
              <div className="grid grid-cols-2 gap-4">
                <Controller
                  name="network.activation"
                  control={control}
                  render={({ field }) => (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 mb-1">
                        Activation Function
                      </label>
                      <select {...field} className="w-full border rounded-lg px-3 py-2">
                        <option value="relu">ReLU</option>
                        <option value="leaky_relu">Leaky ReLU</option>
                        <option value="elu">ELU</option>
                        <option value="selu">SELU</option>
                        <option value="tanh">Tanh</option>
                      </select>
                    </div>
                  )}
                />
                
                <Controller
                  name="network.use_batch_norm"
                  control={control}
                  render={({ field }) => (
                    <div className="flex items-center pt-6">
                      <input
                        type="checkbox"
                        checked={field.value}
                        onChange={e => field.onChange(e.target.checked)}
                        className="mr-2"
                      />
                      <label className="text-sm font-medium text-gray-700">
                        Use Batch Normalization
                        <Tooltip text="Stabilizes training. Recommended for deeper networks." />
                      </label>
                    </div>
                  )}
                />
              </div>

              {/* Skip Connection Toggle */}
              <div className="mt-6 pt-4 border-t border-gray-200">
                <div className="flex items-center justify-between p-4 bg-gradient-to-r from-blue-50 to-indigo-50 rounded-lg">
                  <div className="flex items-center gap-3">
                    <div className="text-2xl">⚡</div>
                    <div>
                      <label className="font-medium text-gray-800">Skip Connection to Output</label>
                      <p className="text-sm text-gray-600 mt-1">
                        Connect input directly to output layer (residual learning)
                      </p>
                      <p className="text-xs text-gray-500 mt-1">
                        Helps capture linear patterns while hidden layers learn non-linear adjustments
                      </p>
                    </div>
                  </div>
                  <Controller
                    name="network.skip_connection"
                    control={control}
                    render={({ field }) => (
                      <label className="relative inline-flex items-center cursor-pointer">
                        <input
                          type="checkbox"
                          checked={field.value || false}
                          onChange={field.onChange}
                          className="sr-only peer"
                        />
                        <div className="w-14 h-7 bg-gray-300 peer-focus:outline-none peer-focus:ring-2 
                                        peer-focus:ring-[#38B2AC] rounded-full peer 
                                        peer-checked:after:translate-x-full peer-checked:bg-[#38B2AC]
                                        after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                                        after:bg-white after:rounded-full after:h-6 after:w-6 
                                        after:transition-all after:shadow-md">
                        </div>
                      </label>
                    )}
                  />
                </div>
                
                {/* Visual diagram when enabled */}
                {skipConnectionEnabled && (
                  <div className="mt-4 p-4 bg-white border border-blue-200 rounded-lg">
                    <p className="text-sm text-gray-700 font-medium mb-2">Architecture Diagram:</p>
                    <div className="font-mono text-xs text-gray-600 bg-gray-50 p-3 rounded">
                      <pre>{`Input (${selectedFeatures.length || 6} features)
  │
  ├──────────────────────────────┐
  │                              │ (Skip Connection)
  ▼                              │
Hidden Layers                    │
  │                              │
  ▼                              │
  ⊕ ◄────────────────────────────┘
  │
  ▼
Output (P(default))`}</pre>
                    </div>
                  </div>
                )}
              </div>
            </>
          )}
        </div>
      </Card>

      {/* Section 3: Regularization */}
      <Card title="🛡️ Regularization">
        <div className="grid grid-cols-2 gap-4">
          <Controller
            name="network.dropout_rate"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Dropout Rate
                  <Tooltip text="Probability of dropping neurons during training. 0.2 is typical." />
                </label>
                <div className="flex items-center gap-4">
                  <input
                    type="range"
                    min={0}
                    max={0.5}
                    step={0.05}
                    {...field}
                    onChange={e => field.onChange(parseFloat(e.target.value))}
                    className="flex-1"
                  />
                  <span className="font-mono w-12">{field.value.toFixed(2)}</span>
                </div>
              </div>
            )}
          />


          <Controller
            name="regularization.l1_lambda"
            control={control}
            render={({ field }) => (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
                  L1 Lambda (Lasso)
                  <Tooltip text="Promotes sparsity. Set to 0 unless you want feature selection." />
            </label>
            <input
              type="number"
                  min={0}
                  max={0.1}
                  step="any"
                  {...field}
                  onChange={e => field.onChange(parseFloat(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
            />
          </div>
            )}
          />

          <Controller
            name="regularization.l2_lambda"
            control={control}
            render={({ field }) => (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
                  L2 Lambda (Ridge)
                  <Tooltip text="Prevents large weights. 0.01 is typical." />
            </label>
            <input
              type="number"
                  min={0}
                  max={0.1}
                  step="any"
                  {...field}
                  onChange={e => field.onChange(parseFloat(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
            />
          </div>
            )}
          />

          <Controller
            name="regularization.gradient_clip_norm"
            control={control}
            render={({ field }) => (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
                  Gradient Clip Norm
                  <Tooltip text="Prevents exploding gradients. 1.0 is typical." />
            </label>
            <input
              type="number"
                  min={0.1}
                  max={10}
                  step="any"
                  {...field}
                  onChange={e => field.onChange(parseFloat(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
            />
          </div>
            )}
          />
        </div>
      </Card>

      {/* Section 4: Early Stopping */}
      <Card title="⏱️ Early Stopping">
        <div className="space-y-4">
          {/* Enable Toggle */}
          <div className="flex items-center justify-between p-3 bg-gray-50 rounded-lg">
            <div>
              <label className="font-medium text-gray-700">Enable Early Stopping</label>
              <p className="text-sm text-gray-500">
                Optional: Stop training when validation metric stops improving
              </p>
            </div>
            <Controller
              name="early_stopping.enabled"
              control={control}
              render={({ field }) => (
                <label className="relative inline-flex items-center cursor-pointer">
                  <input
                    type="checkbox"
                    checked={field.value}
                    onChange={field.onChange}
                    className="sr-only peer"
                  />
                  <div className="w-11 h-6 bg-gray-300 peer-focus:outline-none peer-focus:ring-2 
                                  peer-focus:ring-[#38B2AC] rounded-full peer 
                                  peer-checked:after:translate-x-full peer-checked:bg-[#38B2AC]
                                  after:content-[''] after:absolute after:top-[2px] after:left-[2px] 
                                  after:bg-white after:rounded-full after:h-5 after:w-5 
                                  after:transition-all">
                  </div>
                </label>
              )}
            />
          </div>

          {/* Conditional settings - only show when enabled */}
          {earlyStoppingEnabled && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              {/* Patience */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Patience
                  <Tooltip text="Number of epochs to wait for improvement before stopping" />
                </label>
                <Controller
                  name="early_stopping.patience"
                  control={control}
                  render={({ field }) => (
                    <input
                      type="number"
                      min={1}
                      max={50}
                      step={1}
                      {...field}
                      onChange={(e) => field.onChange(parseInt(e.target.value))}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2"
                    />
                  )}
                />
              </div>

              {/* Min Delta */}
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Min Delta
                  <Tooltip text="Minimum change to qualify as an improvement" />
                </label>
                <Controller
                  name="early_stopping.min_delta"
                  control={control}
                  render={({ field }) => (
                    <input
                      type="number"
                      min={0}
                      max={0.1}
                      step="any"
                      {...field}
                      onChange={(e) => field.onChange(parseFloat(e.target.value))}
                      className="w-full border border-gray-300 rounded-lg px-3 py-2"
                    />
                  )}
                />
              </div>
            </div>
          )}
        </div>
      </Card>

      {/* Section 5: Loss Function */}
      <Card title="📉 Loss Function">
        <div className="space-y-4">
          <Controller
            name="loss.loss_type"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-2">
                  Loss Type
                </label>
                <select {...field} className="w-full border rounded-lg px-3 py-2">
                  <option value="bce">Binary Cross-Entropy (BCE)</option>
                  <option value="pairwise_auc">Pairwise AUC Loss</option>
                  <option value="soft_auc">Soft AUC Loss</option>
                  <option value="wmw">Wilcoxon-Mann-Whitney (WMW)</option>
                  <option value="combined">Combined (BCE + AUC)</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {lossType === 'bce' && 'Standard classification loss. Fast and stable.'}
                  {lossType === 'pairwise_auc' && 'Directly optimizes ranking. Good for AR.'}
                  {lossType === 'soft_auc' && 'Differentiable AUC approximation.'}
                  {lossType === 'wmw' && 'Wilcoxon-Mann-Whitney rank-based loss.'}
                  {lossType === 'combined' && 'Best of both: calibration + ranking.'}
                </p>
              </div>
            )}
          />

          {lossType === 'combined' && (
            <Controller
              name="loss.loss_alpha"
              control={control}
              render={({ field }) => (
                <div>
                  <label className="block text-sm font-medium text-gray-700 mb-1">
                    Alpha (α): BCE Weight
                    <Tooltip text="α × BCE + (1-α) × AUC Loss. Lower α = more AR focus." />
                  </label>
                  <div className="flex items-center gap-4">
                    <span className="text-xs">AUC Focus</span>
                    <input
                      type="range"
                      min={0}
                      max={1}
                      step={0.1}
                      {...field}
                      onChange={e => field.onChange(parseFloat(e.target.value))}
                      className="flex-1"
                    />
                    <span className="text-xs">BCE Focus</span>
                    <span className="font-mono w-12">{field.value.toFixed(1)}</span>
                  </div>
                  <p className="text-xs text-gray-500 mt-1">
                    {field.value * 100}% BCE + {(1 - field.value) * 100}% AUC Loss
                  </p>
                </div>
              )}
            />
          )}

          {(lossType === 'soft_auc' || lossType === 'combined') && (
            <Controller
              name="loss.auc_gamma"
              control={control}
              render={({ field }) => (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
                    AUC Gamma (γ)
                    <Tooltip text="Sharpness parameter for soft AUC. Higher = sharper approximation. 2.0 is typical." />
          </label>
          <input
                    type="number"
                    min={0.1}
                    max={10}
                    step="any"
                    {...field}
                    onChange={e => field.onChange(parseFloat(e.target.value))}
                    className="w-full border rounded-lg px-3 py-2"
                  />
                </div>
              )}
            />
          )}

          <Controller
            name="use_class_weights"
            control={control}
            render={({ field }) => (
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={field.value}
                  onChange={e => field.onChange(e.target.checked)}
                  className="mr-2"
                />
                <label className="text-sm font-medium text-gray-700">
                  Use Class Weights
                  <Tooltip text="Balances imbalanced data by weighting minority class higher." />
                </label>
              </div>
            )}
          />
        </div>
      </Card>

      {/* Section 6: Optimizer */}
      <Card title="⚡ Optimizer">
        <div className="grid grid-cols-3 gap-4">
          <Controller
            name="learning_rate"
            control={control}
            render={({ field }) => (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
                  Learning Rate
                  <Tooltip text="Step size for optimization. 0.001 is typical for Adam." />
          </label>
          <input
            type="number"
                  min={0.00001}
                  max={0.1}
                  step="any"
                  {...field}
                  onChange={e => field.onChange(parseFloat(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
                />
              </div>
            )}
          />

          <Controller
            name="batch_size"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Batch Size
                  <Tooltip text="Samples per gradient update. Larger = faster, less noisy." />
                </label>
                <select {...field} onChange={e => field.onChange(parseInt(e.target.value))} className="w-full border rounded-lg px-3 py-2">
                  <option value={32}>32</option>
                  <option value={64}>64</option>
                  <option value={128}>128</option>
                  <option value={256}>256</option>
                  <option value={512}>512</option>
                </select>
        </div>
            )}
          />

          <Controller
            name="epochs"
            control={control}
            render={({ field }) => (
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-1">
                  Max Epochs
                  <Tooltip text="Maximum training iterations. Training will complete all epochs unless early stopping is enabled." />
          </label>
                <input
                  type="number"
                  min={10}
                  max={500}
                  {...field}
                  onChange={e => field.onChange(parseInt(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
                />
              </div>
            )}
          />
        </div>
      </Card>

      {/* Form Errors Display */}
      {Object.keys(errors).length > 0 && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <h3 className="text-red-800 font-semibold mb-2">Form Validation Errors:</h3>
          <ul className="list-disc list-inside text-red-700 text-sm">
            {Object.entries(errors).map(([key, error]) => (
              <li key={key}>
                {key}: {error?.message || JSON.stringify(error)}
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Submit */}
      <div className="flex justify-end gap-4">
        <Button type="button" variant="ghost" onClick={handleReset}>
          Reset to Defaults
        </Button>
        <Button 
          type="submit" 
          variant="primary" 
          loading={isSubmitting}
          onClick={(e) => {
            console.log('Start Training button clicked');
            // Don't prevent default - let form handle submission
          }}
        >
          🚀 Start Training
        </Button>
      </div>
      </form>
  );
};

export default TrainingConfigForm;
