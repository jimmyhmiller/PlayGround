import { CSSProperties, ReactElement } from 'react';

const colorMap: Record<number, string> = {
  30: '#000', 31: '#e74c3c', 32: '#2ecc71', 33: '#f39c12',
  34: '#3498db', 35: '#9b59b6', 36: '#1abc9c', 37: '#ecf0f1',
  90: '#7f8c8d', 91: '#ff6b6b', 92: '#51cf66', 93: '#ffd43b',
  94: '#4dabf7', 95: '#da77f2', 96: '#3bc9db', 97: '#f8f9fa'
};

function applyCode(code: string, style: CSSProperties): CSSProperties {
  const num = parseInt(code);
  if (num === 0) return {}; // Reset
  if (num === 1) return { ...style, fontWeight: 'bold' };
  if (num === 3) return { ...style, fontStyle: 'italic' };
  if (num === 4) return { ...style, textDecoration: 'underline' };
  if (colorMap[num]) return { ...style, color: colorMap[num] };
  return style;
}

export function parseAnsiToReact(text: string): ReactElement[] | string {
  const ansiRegex = /\x1b\[([0-9;]+)m/g;
  const elements: ReactElement[] = [];
  let lastIndex = 0;
  let currentStyle: CSSProperties = {};

  let match;
  while ((match = ansiRegex.exec(text)) !== null) {
    // Add text before this code
    if (match.index > lastIndex) {
      const textContent = text.substring(lastIndex, match.index);
      elements.push(
        <span key={elements.length} style={currentStyle}>
          {textContent}
        </span>
      );
    }

    // Apply the ANSI code(s)
    const codes = match[1].split(';');
    for (const code of codes) {
      currentStyle = applyCode(code, currentStyle);
    }

    lastIndex = match.index + match[0].length;
  }

  // Add remaining text
  if (lastIndex < text.length) {
    elements.push(
      <span key={elements.length} style={currentStyle}>
        {text.substring(lastIndex)}
      </span>
    );
  }

  return elements.length > 0 ? elements : text;
}
