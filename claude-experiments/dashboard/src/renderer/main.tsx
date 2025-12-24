import React from 'react';
import ReactDOM from 'react-dom/client';
import Desktop from './components/Desktop';

const rootElement = document.getElementById('root');
if (!rootElement) {
  throw new Error('Root element not found');
}

ReactDOM.createRoot(rootElement).render(
  <React.StrictMode>
    <Desktop />
  </React.StrictMode>
);
