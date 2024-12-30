import { createContext, useContext, useState, useEffect } from 'react';
import { TrainingMetrics, ModelConfig } from '../types/training';
import { QureAPI } from '../lib/api';

interface TrainingContextType {
  metrics: TrainingMetrics | null;
  startTraining: (config: ModelConfig) => Promise<void>;
  isTraining: boolean;
  error: string | null;
  progress: number;
}

const TrainingContext = createContext<TrainingContextType | undefined>(undefined);

export function TrainingProvider({ children }: { children: React.ReactNode }) {
  const [metrics, setMetrics] = useState<TrainingMetrics | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [progress, setProgress] = useState(0);
  const api = new QureAPI(process.env.NEXT_PUBLIC_API_URL);

  useEffect(() => {
    const wsUrl = process.env.NEXT_PUBLIC_WS_URL || 'ws://localhost:3000';
    api.connectWebSocket((data) => {
      if (data.type === 'metrics') {
        setMetrics(data.payload);
        setProgress((data.payload.step / data.payload.totalSteps) * 100);
      } else if (data.type === 'status') {
        if (data.payload.status === 'completed') {
          setIsTraining(false);
        }
      }
    });

    return () => api.disconnectWebSocket();
  }, []);

  const startTraining = async (config: ModelConfig) => {
    try {
      setError(null);
      setIsTraining(true);
      await api.startTraining(config);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Training failed');
      setIsTraining(false);
    }
  };

  return (
    <TrainingContext.Provider value={{ 
      metrics, 
      startTraining, 
      isTraining, 
      error,
      progress 
    }}>
      {children}
    </TrainingContext.Provider>
  );
}

export const useTraining = () => {
  const context = useContext(TrainingContext);
  if (context === undefined) {
    throw new Error('useTraining must be used within a TrainingProvider');
  }
  return context;
}; 