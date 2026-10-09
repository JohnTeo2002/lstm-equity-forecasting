import React from 'react';
import { HistoricalBar, ForecastPoint } from '../types';

interface ForecastChartProps {
  historical: HistoricalBar[];
  forecast: ForecastPoint[];
  ticker: string;
}

export const ForecastChart: React.FC<ForecastChartProps> = ({ historical, forecast, ticker }) => {
  if (!historical.length) return null;

  const allCloses = historical.map((h) => h.close);
  const forecastCloses = forecast.map((f) => f.predicted_price);
  const uppers = forecast.filter((f) => f.upper_bound !== undefined).map((f) => f.upper_bound as number);
  const lowers = forecast.filter((f) => f.lower_bound !== undefined).map((f) => f.lower_bound as number);

  const minVal = Math.min(...allCloses, ...forecastCloses, ...(lowers.length ? lowers : [])) * 0.98;
  const maxVal = Math.max(...allCloses, ...forecastCloses, ...(uppers.length ? uppers : [])) * 1.02;
  const range = maxVal - minVal || 1;

  const width = 800;
  const height = 350;
  const padding = 50;
  const chartW = width - padding * 2;
  const chartH = height - padding * 2;

  const totalPoints = historical.length + forecast.length;
  const getX = (index: number) => padding + (index / (totalPoints - 1)) * chartW;
  const getY = (price: number) => padding + chartH - ((price - minVal) / range) * chartH;

  // Historical path
  const histPoints = historical.map((bar, i) => `${getX(i)},${getY(bar.close)}`).join(' L ');
  const histPath = `M ${histPoints}`;

  // Forecast path connects from last historical point
  const startIndex = historical.length - 1;
  const lastHistoricalClose = historical[startIndex].close;
  const forecastPoints = [
    `${getX(startIndex)},${getY(lastHistoricalClose)}`,
    ...forecast.map((pt, i) => `${getX(startIndex + 1 + i)},${getY(pt.predicted_price)}`),
  ].join(' L ');
  const forecastPath = `M ${forecastPoints}`;

  // Confidence interval band path
  const hasConfidenceIntervals = forecast.some(
    (pt) => pt.lower_bound !== undefined && pt.upper_bound !== undefined
  );

  let confidenceBandPath = '';
  let upperBoundPath = '';
  let lowerBoundPath = '';

  if (hasConfidenceIntervals && forecast.length > 0) {
    const upperPoints = [
      `${getX(startIndex)},${getY(lastHistoricalClose)}`,
      ...forecast.map((pt, i) => `${getX(startIndex + 1 + i)},${getY(pt.upper_bound ?? pt.predicted_price)}`),
    ];
    upperBoundPath = `M ${upperPoints.join(' L ')}`;

    const lowerPointsReversed = [
      ...forecast
        .map((pt, i) => `${getX(startIndex + 1 + i)},${getY(pt.lower_bound ?? pt.predicted_price)}`)
        .reverse(),
      `${getX(startIndex)},${getY(lastHistoricalClose)}`,
    ];
    lowerBoundPath = `M ${lowerPointsReversed.join(' L ')}`;

    confidenceBandPath = `M ${upperPoints.join(' L ')} L ${lowerPointsReversed.join(' L ')} Z`;
  }

  return (
    <div style={{
      backgroundColor: '#1f2937',
      borderRadius: '0.5rem',
      border: '1px solid #374151',
      padding: '1.5rem',
      marginTop: '1.5rem',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        marginBottom: '1rem',
      }}>
        <h2 style={{ fontSize: '1rem', fontWeight: '600', color: '#f3f4f6', margin: 0 }}>
          {ticker} Price Trajectory & LSTM Projections
        </h2>
        <div style={{ display: 'flex', gap: '1.25rem', fontSize: '0.75rem', color: '#9ca3af' }}>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: '12px', height: '3px', backgroundColor: '#3b82f6', display: 'inline-block' }}></span>
            Historical Close
          </span>
          <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
            <span style={{ width: '12px', height: '3px', backgroundColor: '#10b981', display: 'inline-block', borderTop: '2px dashed #10b981' }}></span>
            LSTM Forecast
          </span>
          {hasConfidenceIntervals && (
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <span style={{ width: '12px', height: '8px', backgroundColor: 'rgba(16, 185, 129, 0.25)', display: 'inline-block', borderRadius: '2px' }}></span>
              Confidence Band
            </span>
          )}
        </div>
      </div>

      <div style={{ width: '100%', overflowX: 'auto' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: 'auto', display: 'block' }}>
          {/* Grid lines */}
          {[0, 0.25, 0.5, 0.75, 1].map((pct) => {
            const y = padding + chartH * pct;
            const price = maxVal - pct * range;
            return (
              <g key={pct}>
                <line x1={padding} y1={y} x2={width - padding} y2={y} stroke="#374151" strokeDasharray="3 3" />
                <text x={padding - 8} y={y + 4} fill="#6b7280" fontSize="10" textAnchor="end">
                  ${price.toFixed(1)}
                </text>
              </g>
            );
          })}

          {/* Confidence interval shaded band */}
          {confidenceBandPath && (
            <path
              d={confidenceBandPath}
              fill="rgba(16, 185, 129, 0.15)"
              stroke="none"
            />
          )}

          {/* Confidence interval boundary lines */}
          {upperBoundPath && (
            <path
              d={upperBoundPath}
              fill="none"
              stroke="#059669"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.6"
            />
          )}
          {lowerBoundPath && (
            <path
              d={lowerBoundPath}
              fill="none"
              stroke="#059669"
              strokeWidth="1"
              strokeDasharray="2 2"
              opacity="0.6"
            />
          )}

          {/* Historical line */}
          <path d={histPath} fill="none" stroke="#3b82f6" strokeWidth="2.5" />

          {/* Forecast trajectory */}
          <path d={forecastPath} fill="none" stroke="#10b981" strokeWidth="2.5" strokeDasharray="5 4" />

          {/* Points */}
          {forecast.map((pt, i) => {
            const cx = getX(startIndex + 1 + i);
            const cy = getY(pt.predicted_price);
            return (
              <g key={i}>
                <circle cx={cx} cy={cy} r="4" fill="#10b981" />
                <text x={cx} y={cy - 10} fill="#6ee7b7" fontSize="10" textAnchor="middle">
                  ${pt.predicted_price.toFixed(1)}
                </text>
              </g>
            );
          })}
        </svg>
      </div>
    </div>
  );
};
