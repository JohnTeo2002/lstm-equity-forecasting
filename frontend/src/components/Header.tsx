import React from 'react';

export const Header: React.FC = () => {
  return (
    <header style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'space-between',
      padding: '1.25rem 2rem',
      backgroundColor: '#111827',
      borderBottom: '1px solid #1f2937'
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
        <div style={{
          width: '2rem',
          height: '2rem',
          borderRadius: '0.375rem',
          backgroundColor: '#3b82f6',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          fontWeight: 'bold',
          color: '#ffffff'
        }}>
          📈
        </div>
        <div>
          <h1 style={{ fontSize: '1.25rem', fontWeight: '700', margin: 0, color: '#ffffff' }}>
            LSTM Equity Forecasting
          </h1>
          <p style={{ fontSize: '0.75rem', color: '#9ca3af', margin: 0 }}>
            Deep Learning Time-Series Predictive Engine
          </p>
        </div>
      </div>
      <div style={{
        display: 'inline-flex',
        alignItems: 'center',
        padding: '0.25rem 0.75rem',
        borderRadius: '9999px',
        fontSize: '0.75rem',
        backgroundColor: '#064e3b',
        color: '#6ee7b7',
        border: '1px solid #047857'
      }}>
        ● Model Ready (v2.1)
      </div>
    </header>
  );
};

