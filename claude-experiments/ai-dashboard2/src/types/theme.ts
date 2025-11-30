export interface Theme {
  name?: string;
  icon?: string;
  bgApp?: string;
  bgSecondary?: string;
  widgetBg?: string;
  widgetRadius?: string;
  textBody?: string;
  textHeader?: string;
  accent?: string;
  positive?: string;
  negative?: string;
  shadow?: string;
}

export const DEFAULT_THEME: Theme = {
  bgApp: '#0d1117',
  bgSecondary: '#161b22',
  widgetBg: 'rgba(22, 27, 34, 0.85)',
  widgetRadius: '8px',
  textBody: 'Inter, system-ui, sans-serif',
  textHeader: 'Inter, system-ui, sans-serif',
  accent: '#00d9ff',
  positive: '#39ff14',
  negative: '#ff4757',
  shadow: 'rgba(0, 0, 0, 0.4)',
};
