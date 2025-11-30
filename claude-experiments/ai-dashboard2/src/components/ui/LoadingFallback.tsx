import { FC } from 'react';

export const LoadingFallback: FC = () => {
  return (
    <div className="app" style={{
      display: 'flex',
      alignItems: 'center',
      justifyContent: 'center',
      height: '100vh',
      backgroundColor: '#0d1117',
      color: '#58a6ff',
      fontFamily: 'system-ui',
      fontSize: '1.2rem'
    }}>
      <div style={{ textAlign: 'center' }}>
        <div style={{ marginBottom: 12 }}>Loading dashboard...</div>
        <div style={{
          width: 40,
          height: 40,
          border: '3px solid rgba(88, 166, 255, 0.2)',
          borderTopColor: '#58a6ff',
          borderRadius: '50%',
          margin: '0 auto',
          animation: 'spin 1s linear infinite'
        }} />
      </div>
    </div>
  );
};
