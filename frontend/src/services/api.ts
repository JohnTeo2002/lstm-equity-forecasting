import { ForecastResponse, HistoricalDataResponse } from '../types';

const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || '/api/v1';

export async function fetchHistoricalData(ticker: string, days = 60): Promise<HistoricalDataResponse> {
  const res = await fetch(`${API_BASE_URL}/stocks/${encodeURIComponent(ticker)}/historical?days=${days}`);
  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to fetch data for ${ticker}`);
  }
  return res.json();
}

export async function generatePrediction(
  ticker: string,
  lookbackWindow = 60,
  horizonDays = 5
): Promise<ForecastResponse> {
  const res = await fetch(`${API_BASE_URL}/forecast/predict`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({
      ticker: ticker.toUpperCase(),
      lookback_window: lookbackWindow,
      horizon_days: horizonDays,
      include_confidence_intervals: true,
    }),
  });

  if (!res.ok) {
    const errorData = await res.json().catch(() => ({}));
    throw new Error(errorData.detail || `Failed to forecast for ${ticker}`);
  }
  return res.json();
}

