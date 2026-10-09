import React, { useState } from 'react';

interface ForecastFormProps {
  onForecast: (ticker: string, lookback: number, horizon: number) => void;
  loading: boolean;
}

export const ForecastForm: React.FC<ForecastFormProps> = ({ onForecast, loading }) => {
  const [ticker, setTicker] = useState('AAPL');
  const [lookback, setLookback] = useState(60);
  const [horizon, setHorizon] = useState(5);

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (ticker.trim()) {
      onForecast(ticker.trim().toUpperCase(), lookback, horizon);
    }
  };

  return (
    <form
      onSubmit={handleSubmit}
      style={{
        display: 'flex',
        flexWrap: 'wrap',
        gap: '1rem',
        alignItems: 'flex-end',
        backgroundColor: '#1f2937',
        padding: '1.25rem',
        borderRadius: '0.5rem',
        border: '1px solid #374151',
      }}
    >
      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label htmlFor="ticker-input" style={{ fontSize: '0.75rem', fontWeight: '600', color: '#9ca3af' }}>
          TICKER SYMBOL
        </label>
        <input
          id="ticker-input"
          type="text"
          value={ticker}
          onChange={(e) => setTicker(e.target.value)}
          placeholder="e.g. AAPL, NVDA"
          style={{
            backgroundColor: '#111827',
            border: '1px solid #4b5563',
            color: '#f9fafb',
            borderRadius: '0.375rem',
            padding: '0.5rem 0.75rem',
            fontSize: '0.875rem',
            textTransform: 'uppercase',
            width: '120px',
          }}
          required
        />
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label htmlFor="lookback-select" style={{ fontSize: '0.75rem', fontWeight: '600', color: '#9ca3af' }}>
          LOOKBACK WINDOW (DAYS)
        </label>
        <select
          id="lookback-select"
          value={lookback}
          onChange={(e) => setLookback(Number(e.target.value))}
          style={{
            backgroundColor: '#111827',
            border: '1px solid #4b5563',
            color: '#f9fafb',
            borderRadius: '0.375rem',
            padding: '0.5rem 0.75rem',
            fontSize: '0.875rem',
          }}
        >
          <option value={30}>30 Days</option>
          <option value={60}>60 Days (Default)</option>
          <option value={90}>90 Days</option>
          <option value={120}>120 Days</option>
        </select>
      </div>

      <div style={{ display: 'flex', flexDirection: 'column', gap: '0.25rem' }}>
        <label htmlFor="horizon-select" style={{ fontSize: '0.75rem', fontWeight: '600', color: '#9ca3af' }}>
          FORECAST HORIZON
        </label>
        <select
          id="horizon-select"
          value={horizon}
          onChange={(e) => setHorizon(Number(e.target.value))}
          style={{
            backgroundColor: '#111827',
            border: '1px solid #4b5563',
            color: '#f9fafb',
            borderRadius: '0.375rem',
            padding: '0.5rem 0.75rem',
            fontSize: '0.875rem',
          }}
        >
          <option value={3}>3 Days</option>
          <option value={5}>5 Days (1 Week)</option>
          <option value={10}>10 Days (2 Weeks)</option>
          <option value={15}>15 Days</option>
        </select>
      </div>

      <button
        type="submit"
        disabled={loading}
        style={{
          backgroundColor: loading ? '#4b5563' : '#2563eb',
          color: '#ffffff',
          fontWeight: '600',
          fontSize: '0.875rem',
          padding: '0.55rem 1.25rem',
          borderRadius: '0.375rem',
          border: 'none',
          cursor: loading ? 'not-allowed' : 'pointer',
          transition: 'background-color 0.2s',
        }}
      >
        {loading ? 'Running Inference...' : 'Generate Forecast'}
      </button>
    </form>
  );
};

