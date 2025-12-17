import React from 'react';
import { useForm, Controller } from 'react-hook-form';
import { InformationCircleIcon } from '@heroicons/react/24/outline';
import { Button, Card } from '../common';

interface TrainingConfig {
  segment: string;
  test_size: number;
  network: {
    model_type: 'linear' | 'neural_network';
    hidden_layers: number[];
    activation: string;
    dropout_rate: number;
    use_batch_norm: boolean;
  };
  regularization: {
    l1_lambda: number;
    l2_lambda: number;
    gradient_clip_norm: number;
  };
  loss: {
    loss_type: string;
    loss_alpha: number;
    use_class_weights: boolean;
  };
  learning_rate: number;
  batch_size: number;
  epochs: number;
  early_stopping_patience: number;
  selected_features: string[];
}

interface Props {
  defaultValues: Partial<TrainingConfig>;
  segments: string[];
  features: string[];
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

export const TrainingConfigForm: React.FC<Props> = ({
  defaultValues,
  segments,
  features,
  onSubmit,
  isSubmitting
}) => {
  const { control, handleSubmit, watch, setValue } = useForm<TrainingConfig>({
    defaultValues: {
      segment: 'ALL',
      test_size: 0.30,
      network: {
        model_type: 'neural_network',
        hidden_layers: [64, 32],
        activation: 'relu',
        dropout_rate: 0.2,
        use_batch_norm: true
      },
      regularization: {
        l1_lambda: 0.0,
        l2_lambda: 0.01,
        gradient_clip_norm: 1.0
      },
      loss: {
        loss_type: 'combined',
        loss_alpha: 0.3,
        use_class_weights: true
      },
      learning_rate: 0.001,
      batch_size: 256,
      epochs: 100,
      early_stopping_patience: 15,
      selected_features: features,
      ...defaultValues
    }
  });

  const modelType = watch('network.model_type');
  const lossType = watch('loss.loss_type');
  const hiddenLayers = watch('network.hidden_layers');

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

  return (
    <form onSubmit={handleSubmit(onSubmit)} noValidate className="space-y-6">
      
      {/* Section 1: Data Split */}
      <Card title="📊 Data Split">
        <div className="grid grid-cols-2 gap-6">
          <Controller
            name="segment"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Segment
                </label>
                <select {...field} className="w-full border rounded-lg px-3 py-2">
                  {segments.map(seg => (
                    <option key={seg} value={seg}>{seg}</option>
                  ))}
                </select>
              </div>
            )}
          />
          
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
            name="early_stopping_patience"
            control={control}
            render={({ field }) => (
              <div>
                <label className="block text-sm font-medium text-gray-700 mb-1">
                  Early Stopping Patience
                  <Tooltip text="Stop training if Test AR doesn't improve for this many epochs." />
                </label>
                <input
                  type="number"
                  min={5}
                  max={50}
                  {...field}
                  onChange={e => field.onChange(parseInt(e.target.value))}
                  className="w-full border rounded-lg px-3 py-2"
                />
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

      {/* Section 4: Loss Function */}
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
                  <option value="combined">Combined (BCE + AUC)</option>
                </select>
                <p className="text-xs text-gray-500 mt-1">
                  {lossType === 'bce' && 'Standard classification loss. Fast and stable.'}
                  {lossType === 'pairwise_auc' && 'Directly optimizes ranking. Good for AR.'}
                  {lossType === 'soft_auc' && 'Differentiable AUC approximation.'}
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

          <Controller
            name="loss.use_class_weights"
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

      {/* Section 5: Optimizer */}
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
                  <Tooltip text="Maximum training iterations. Early stopping will kick in." />
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

      {/* Submit */}
      <div className="flex justify-end gap-4">
        <Button type="button" variant="ghost">
          Reset to Defaults
        </Button>
        <Button type="submit" variant="primary" loading={isSubmitting}>
          🚀 Start Training
        </Button>
      </div>
    </form>
  );
};

export default TrainingConfigForm;

