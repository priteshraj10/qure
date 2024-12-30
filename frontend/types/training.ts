export interface TrainingMetrics {
  loss: number;
  accuracy: number;
  epoch: number;
  step: number;
  learningRate: number;
  validationLoss?: number;
  validationAccuracy?: number;
  timestamp: number;
}

export interface ModelConfig {
  modelName: string;
  batchSize: number;
  learningRate: number;
  maxEpochs: number;
  deviceType: 'cuda' | 'cpu';
}

export interface RAGConfig {
  embeddingModel: string;
  chunkSize: number;
  chunkOverlap: number;
  topK: number;
} 