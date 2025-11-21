import { useState } from 'react';
import { useThemeStore } from '../store/useThemeStore';
import './style-control.css';

interface StyleControlProps {
  onGenerate: (prompt: string) => void;
}

export function StyleControl({ onGenerate }: StyleControlProps) {
  const [prompt, setPrompt] = useState('');
  const { isGenerating, history } = useThemeStore();

  const handleSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (prompt.trim()) {
      onGenerate(prompt.trim());
    }
  };

  const quickPrompts = [
    'Art Deco with gold accents',
    'Cyberpunk with neon colors',
    'Brutalist monochrome',
    'Vaporwave aesthetic',
    '1960s NASA control room',
    'Organic art nouveau',
  ];

  return (
    <div className="style-control">
      <form onSubmit={handleSubmit} className="style-form">
        <input
          type="text"
          value={prompt}
          onChange={(e) => setPrompt(e.target.value)}
          placeholder="Describe the aesthetic... (e.g., 'art deco with gold')"
          disabled={isGenerating}
          className="style-input"
        />
        <button type="submit" disabled={isGenerating || !prompt.trim()} className="style-button">
          {isGenerating ? 'Generating...' : 'Generate Style'}
        </button>
      </form>

      <div className="quick-prompts">
        <span className="quick-prompts-label">Quick styles:</span>
        {quickPrompts.map((p) => (
          <button
            key={p}
            onClick={() => onGenerate(p)}
            disabled={isGenerating}
            className="quick-prompt-button"
          >
            {p}
          </button>
        ))}
      </div>

      {history.length > 0 && (
        <div className="style-history">
          <span className="history-label">History ({history.length}):</span>
          {history.slice(-5).reverse().map((style) => (
            <div key={style.id} className="history-item">
              <span className="history-prompt">{style.prompt}</span>
              <span className="history-time">
                {new Date(style.timestamp).toLocaleTimeString()}
              </span>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
