import { useState, useEffect } from 'react';
import type { WidgetConfig } from '../types';

interface UseWidgetDataResult<T> {
  data: T | null;
  loading: boolean;
  error: string | null;
}

export function useWidgetData<T = unknown>(
  config: WidgetConfig,
  reloadTrigger?: number
): UseWidgetDataResult<T> {
  const [data, setData] = useState<T | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Check if widget has inline data (type narrowing)
    if ('data' in config && config.data !== undefined) {
      setData(config.data as T);
      setLoading(false);
      setError(null);
      return;
    }

    // If dataSource is provided, load from file
    if (config.dataSource) {
      setLoading(true);
      setError(null);

      // Use the dashboard API to load the file
      if (window.dashboardAPI && window.dashboardAPI.loadDataFile) {
        window.dashboardAPI.loadDataFile(config.dataSource)
          .then(loadedData => {
            setData(loadedData as T);
            setLoading(false);
          })
          .catch(err => {
            console.error('[useWidgetData] Failed to load data from', config.dataSource, err);
            setError(err.message || 'Failed to load data');
            setLoading(false);
          });
      } else {
        // Fallback: try to fetch as a relative URL
        fetch(config.dataSource)
          .then(response => {
            if (!response.ok) throw new Error(`HTTP ${response.status}`);
            return response.json();
          })
          .then(loadedData => {
            setData(loadedData as T);
            setLoading(false);
          })
          .catch(err => {
            console.error('[useWidgetData] Failed to fetch data from', config.dataSource, err);
            setError(err.message || 'Failed to load data');
            setLoading(false);
          });
      }
    }
  }, [config, reloadTrigger]);

  return { data, loading, error };
}
