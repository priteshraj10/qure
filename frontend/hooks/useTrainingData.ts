import { useMemo } from 'react';
import { TrainingMetrics } from '../types/training';

export function useTrainingData(metrics: TrainingMetrics | null) {
  return useMemo(() => {
    if (!metrics) return null;

    return {
      chartData: {
        labels: Array.from({ length: metrics.step }, (_, i) => i.toString()),
        datasets: [
          {
            label: 'Training Loss',
            data: metrics.lossHistory || [],
            borderColor: 'rgb(75, 192, 192)',
            backgroundColor: 'rgba(75, 192, 192, 0.5)',
          },
          {
            label: 'Validation Loss',
            data: metrics.validationLossHistory || [],
            borderColor: 'rgb(255, 99, 132)',
            backgroundColor: 'rgba(255, 99, 132, 0.5)',
          },
        ],
      },
      metrics: [
        {
          label: 'Training Loss',
          value: metrics.loss.toFixed(4),
          change: metrics.lossChange,
          icon: 'activity'
        },
        {
          label: 'Accuracy',
          value: `${(metrics.accuracy * 100).toFixed(2)}%`,
          change: metrics.accuracyChange,
          icon: 'target'
        },
        {
          label: 'Learning Rate',
          value: metrics.learningRate.toExponential(2),
          change: 0,
          icon: 'trending-up'
        },
        {
          label: 'Epoch',
          value: `${metrics.epoch + 1}/${metrics.maxEpochs}`,
          change: 0,
          icon: 'repeat'
        },
      ],
    };
  }, [metrics]);
} 