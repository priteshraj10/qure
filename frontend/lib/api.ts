import { TrainingMetrics, ModelConfig, RAGConfig } from '../types/training';

export class QureAPI {
  private baseUrl: string;
  private ws: WebSocket | null = null;
  private reconnectAttempts = 0;
  private maxReconnectAttempts = 5;

  constructor(baseUrl: string = 'http://localhost:3000') {
    this.baseUrl = baseUrl;
  }

  private setupWebSocket(wsUrl: string, onMessage: (data: any) => void) {
    this.ws = new WebSocket(wsUrl);

    this.ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        onMessage(data);
      } catch (error) {
        console.error('WebSocket message parsing error:', error);
      }
    };

    this.ws.onclose = () => {
      if (this.reconnectAttempts < this.maxReconnectAttempts) {
        setTimeout(() => {
          this.reconnectAttempts++;
          this.setupWebSocket(wsUrl, onMessage);
        }, 1000 * Math.pow(2, this.reconnectAttempts));
      }
    };

    this.ws.onerror = (error) => {
      console.error('WebSocket error:', error);
    };
  }

  connectWebSocket(onMessage: (data: any) => void): void {
    const wsUrl = `${this.baseUrl.replace('http', 'ws')}/ws/metrics`;
    this.setupWebSocket(wsUrl, onMessage);
  }

  disconnectWebSocket(): void {
    if (this.ws) {
      this.ws.close();
      this.ws = null;
    }
  }

  async startTraining(config: ModelConfig): Promise<Response> {
    const response = await fetch(`${this.baseUrl}/api/training/start`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.message || 'Training failed to start');
    }

    return response;
  }

  async configureRAG(config: RAGConfig): Promise<Response> {
    return fetch(`${this.baseUrl}/api/rag/configure`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(config),
    });
  }
} 