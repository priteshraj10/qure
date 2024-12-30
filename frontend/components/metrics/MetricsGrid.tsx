import { Card } from '../ui/card';
import { 
  TrendingUp, 
  TrendingDown, 
  Activity,
  Brain,
  Clock,
  Database
} from 'lucide-react';

interface Metric {
  label: string;
  value: string | number;
  change: number;
  icon: JSX.Element;
}

export function MetricsGrid() {
  const metrics: Metric[] = [
    {
      label: 'Training Loss',
      value: '0.342',
      change: -5.2,
      icon: <Activity className="w-6 h-6" />
    },
    {
      label: 'Validation Accuracy',
      value: '94.3%',
      change: 2.1,
      icon: <Brain className="w-6 h-6" />
    },
    {
      label: 'Training Time',
      value: '2h 34m',
      change: 0,
      icon: <Clock className="w-6 h-6" />
    },
    {
      label: 'Dataset Size',
      value: '1.2M',
      change: 0,
      icon: <Database className="w-6 h-6" />
    }
  ];

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
      {metrics.map((metric) => (
        <Card key={metric.label} className="p-6">
          <div className="flex items-center justify-between">
            <div className="flex items-center space-x-4">
              <div className="p-2 bg-blue-100 rounded-lg">
                {metric.icon}
              </div>
              <div>
                <p className="text-sm text-gray-500">{metric.label}</p>
                <p className="text-2xl font-bold">{metric.value}</p>
              </div>
            </div>
            {metric.change !== 0 && (
              <div className={`flex items-center ${
                metric.change > 0 ? 'text-green-500' : 'text-red-500'
              }`}>
                {metric.change > 0 ? (
                  <TrendingUp className="w-4 h-4" />
                ) : (
                  <TrendingDown className="w-4 h-4" />
                )}
                <span className="ml-1">{Math.abs(metric.change)}%</span>
              </div>
            )}
          </div>
        </Card>
      ))}
    </div>
  );
} 