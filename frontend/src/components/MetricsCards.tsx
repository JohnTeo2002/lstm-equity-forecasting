import React from 'react';
import { ForecastResponse } from '../types';

interface MetricsCardsProps {
  forecast: ForecastResponse;
}

export const MetricsCards: React.FC<MetricsCardsProps> = ({ forecast }) => {
  const changePct = forecast.metrics?.projected_change_pct ?? 0;
  const isPositive = changePct >= 0;
  const lastPrice = forecast.last_historical_close;
  const finalPrice = forecast.predictions[forecast.predictions.length - 1]?.predicted_price ?? lastPrice;

  return (
    <div style={{
      display: 'grid',
      gridTemplateColumns: 'repeat(auto-fit, minmax(200px, 1fr))',
      gap: '1rem',
      marginTop: '1.5rem',
      marginBottom: '1.5rem',
    }}>
      <div style={{
        backgroundColor: '#1f2937',
        border: '1px solid #374151',
        borderRadius: '0.5rem',
        padding: '1rem',
      }}>
        <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>LAST CLOSE</div>
        <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ffffff', marginTop: '0.25rem' }}>
          ${lastPrice.toFixed(2)}
        </div>
      </div>

      <div style={{
        backgroundColor: '#1f2937',
        border: '1px solid #374151',
        borderRadius: '0.5rem',
        padding: '1rem',
      }}>
        <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>PROJECTED TARGET</div>
        <div style={{ fontSize: '1.5rem', fontWeight: 'bold', color: '#ffffff', marginTop: '0.25rem' }}>
          ${finalPrice.toFixed(2)}
        </div>
      </div>

      <div style={{
        backgroundColor: '#1f2937',
        border: '1px solid #374151',
        borderRadius: '0.5rem',
        padding: '1rem',
      }}>
        <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>EXPECTED RETURN</div>
        <div style={{
          fontSize: '1.5rem',
          fontWeight: 'bold',
          color: isPositive ? '#10b981' : '#ef4444',
          marginTop: '0.25rem',
        }}>
          {isPositive ? '+' : ''}{changePct.toFixed(2)}%
        </div>
      </div>

      <div style={{
        backgroundColor: '#1f2937',
        border: '1px solid #374151',
        borderRadius: '0.5rem',
        padding: '1rem',
      }}>
        <div style={{ fontSize: '0.75rem', color: '#9ca3af' }}>OUTLOOK</div>
        <div style={{
          fontSize: '1.5rem',
          fontWeight: 'bold',
          textTransform: 'capitalize',
          color: isPositive ? '#10b981' : '#ef4444',
          marginTop: '0.25rem',
        }}>
          {forecast.metrics?.expected_direction || 'Neutral'}
        </div>
      </div>
    </div>
  );
};

