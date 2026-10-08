export interface HistoricalBar {
  date: string;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

export interface ForecastPoint {
  step: number;
  target_date: string;
  predicted_price: number;
  lower_bound?: number;
  upper_bound?: number;
}

export interface ForecastResponse {
  ticker: string;
  generated_at: string;
  model_version: string;
  last_historical_close: number;
  predictions: ForecastPoint[];
  metrics: {
    expected_direction?: 'bullish' | 'bearish' | 'neutral';
    projected_change_pct?: number;
    estimated_volatility?: number;
    [key: string]: any;
  };
}

export interface HistoricalDataResponse {
  ticker: string;
  count: number;
  data: HistoricalBar[];
}

