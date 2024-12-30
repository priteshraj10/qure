import { ArrowUpIcon, ArrowDownIcon } from '@heroicons/react/24/solid';

interface MetricsCardProps {
  title: string;
  value: string | number;
  trend: 'increasing' | 'decreasing' | 'neutral';
}

export default function MetricsCard({ title, value, trend }: MetricsCardProps) {
  return (
    <div className="bg-white rounded-lg shadow p-6">
      <h3 className="text-gray-500 text-sm font-medium">{title}</h3>
      <div className="mt-2 flex items-center">
        <span className="text-3xl font-bold">{value}</span>
        {trend !== 'neutral' && (
          <span className={`ml-2 ${trend === 'increasing' ? 'text-green-500' : 'text-red-500'}`}>
            {trend === 'increasing' ? (
              <ArrowUpIcon className="w-5 h-5" />
            ) : (
              <ArrowDownIcon className="w-5 h-5" />
            )}
          </span>
        )}
      </div>
    </div>
  );
} 