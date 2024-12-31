import { useState, useEffect } from 'react';
import { TrainingMetrics, ModelConfig, RAGConfig } from '../types/training';
import { LineChart, Card, Button, Select, Input } from './ui';
import { WebSocketClient } from '../lib/websocket';

interface TrainingDashboardProps {
  onStartTraining: (config: ModelConfig & RAGConfig) => void;
  onStopTraining: () => void;
}

export const TrainingDashboard: React.FC<TrainingDashboardProps> = ({
  onStartTraining,
  onStopTraining
}) => {
  const [metrics, setMetrics] = useState<TrainingMetrics[]>([]);
  const [config, setConfig] = useState<ModelConfig & RAGConfig>({
    modelName: "unsloth/Llama-3.2-3B",
    batchSize: 2,
    learningRate: 2e-4,
    maxEpochs: 5,
    deviceType: 'cuda',
    embeddingModel: 'medical/instructor-xl',
    chunkSize: 512,
    chunkOverlap: 50,
    topK: 4
  });

  useEffect(() => {
    const ws = new WebSocketClient('ws://localhost:8000/ws/training');
    ws.onMessage((data: TrainingMetrics) => {
      setMetrics(prev => [...prev, data]);
    });
    return () => ws.close();
  }, []);

  return (
    <div className="p-6 space-y-6">
      <Card>
        <h2>Training Configuration</h2>
        <div className="grid grid-cols-2 gap-4">
          <Select
            label="Model"
            value={config.modelName}
            onChange={(value) => setConfig({...config, modelName: value})}
            options={[
              { value: "unsloth/Llama-3.2-3B", label: "Llama 3.2 3B" },
              { value: "unsloth/Llama-3.2-1B", label: "Llama 3.2 1B" }
            ]}
          />
          {/* Other configuration inputs */}
        </div>
        <Button 
          onClick={() => onStartTraining(config)}
          className="mt-4"
        >
          Start Training
        </Button>
      </Card>

      <Card>
        <h2>Training Progress</h2>
        <LineChart
          data={metrics}
          series={[
            { key: 'loss', label: 'Training Loss' },
            { key: 'validationLoss', label: 'Validation Loss' }
          ]}
        />
      </Card>
    </div>
  );
}; 