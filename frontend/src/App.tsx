import React, { useState, useEffect } from 'react';
import { Header } from './components/Header';
import { ForecastForm } from './components/ForecastForm';
import { ForecastChart } from './components/ForecastChart';
import { MetricsCards } from './components/MetricsCards';
import { fetchHistoricalData, generatePrediction } from './services/api';
import { ForecastResponse, HistoricalBar } from './types';

export const App: React.FC = () => {
  const [ticker, setTicker] = useState('AAPL');
  const [historical, setHistorical] = useState<HistoricalBar[]>([]);
  const [forecast, setForecast] = useState<ForecastResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const loadData = async (symbol: string, lookback: number, horizon: number) => {
    setLoading(true);
    setError(null);
    try {
      setTicker(symbol);
      const [histData, predData] = await Promise.all([
        fetchHistoricalData(symbol, lookback),
        generatePrediction(symbol, lookback, horizon),
      ]);
      setHistorical(histData.data);
      setForecast(predData);
    } catch (err: any) {
      setError(err.message || 'An unexpected error occurred.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    loadData('AAPL', 60, 5);
  }, []);

  return (
    <div style={{ minHeight: '100vh', display: 'flex', flexDirection: 'column' }}>
      <Header />

      <main style={{ maxWidth: '1100px', width: '100%', margin: '0 auto', padding: '2rem 1.5rem', boxSizing: 'border-box' }}>
        <ForecastForm onForecast={loadData} loading={loading} />

        {error && (
          <div style={{
            marginTop: '1rem',
            padding: '1rem',
            backgroundColor: '#7f1d1d',
            border: '1px solid #b91c1c',
            borderRadius: '0.375rem',
            color: '#fecaca',
            fontSize: '0.875rem',
          }}>
            ⚠️ {error}
          </div>
        )}

        {forecast && <MetricsCards forecast={forecast} />}

        {historical.length > 0 && forecast && (
          <ForecastChart
            historical={historical}
            forecast={forecast.predictions}
            ticker={ticker}
          />
        )}
      </main>
    </div>
  );
};

export default App;

