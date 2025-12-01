import { FC, useState, useEffect, useCallback } from 'react';
import ReactMarkdown from 'react-markdown';
import { Prism as SyntaxHighlighter } from 'react-syntax-highlighter';
import { oneDark } from 'react-syntax-highlighter/dist/esm/styles/prism';
import type { Theme } from '../types';

// Global state for widgets (preserves state across re-renders)
export const globalChatInputs = new Map<string, string>(); // conversationId -> input text
export const globalCodeEditorStates = new Map<string, { code: string; language: string }>(); // widgetId -> { code, language }
export const globalCommandOutputs = new Map<string, string>(); // widgetId -> output text

// Language configuration for CodeEditor
export const LANGUAGE_CONFIG: Record<string, { command: string; extension: string; monacoLang: string }> = {
  javascript: { command: 'node {file}', extension: 'js', monacoLang: 'javascript' },
  typescript: { command: 'ts-node {file}', extension: 'ts', monacoLang: 'typescript' },
  python: { command: 'python3 {file}', extension: 'py', monacoLang: 'python' },
  python2: { command: 'python {file}', extension: 'py', monacoLang: 'python' },
  ruby: { command: 'ruby {file}', extension: 'rb', monacoLang: 'ruby' },
  php: { command: 'php {file}', extension: 'php', monacoLang: 'php' },
  perl: { command: 'perl {file}', extension: 'pl', monacoLang: 'perl' },
  lua: { command: 'lua {file}', extension: 'lua', monacoLang: 'lua' },
  bash: { command: 'bash {file}', extension: 'sh', monacoLang: 'shell' },
  sh: { command: 'sh {file}', extension: 'sh', monacoLang: 'shell' },
  zsh: { command: 'zsh {file}', extension: 'sh', monacoLang: 'shell' },
  powershell: { command: 'pwsh {file}', extension: 'ps1', monacoLang: 'powershell' },
  go: { command: 'go run {file}', extension: 'go', monacoLang: 'go' },
  rust: { command: 'rustc {file} -o {tempdir}/a.out && {tempdir}/a.out', extension: 'rs', monacoLang: 'rust' },
  c: { command: 'gcc {file} -o {tempdir}/a.out && {tempdir}/a.out', extension: 'c', monacoLang: 'c' },
  cpp: { command: 'g++ {file} -o {tempdir}/a.out && {tempdir}/a.out', extension: 'cpp', monacoLang: 'cpp' },
  java: { command: 'java {file}', extension: 'java', monacoLang: 'java' },
  kotlin: { command: 'kotlinc {file} -include-runtime -d {tempdir}/out.jar && java -jar {tempdir}/out.jar', extension: 'kt', monacoLang: 'kotlin' },
  swift: { command: 'swift {file}', extension: 'swift', monacoLang: 'swift' },
  r: { command: 'Rscript {file}', extension: 'R', monacoLang: 'r' },
  clojure: { command: 'clj -M {file}', extension: 'clj', monacoLang: 'clojure' },
  deno: { command: 'deno run {file}', extension: 'ts', monacoLang: 'typescript' },
  bun: { command: 'bun run {file}', extension: 'ts', monacoLang: 'typescript' },
  scala: { command: 'scala {file}', extension: 'scala', monacoLang: 'scala' },
  haskell: { command: 'runhaskell {file}', extension: 'hs', monacoLang: 'haskell' },
  elixir: { command: 'elixir {file}', extension: 'ex', monacoLang: 'elixir' },
  erlang: { command: 'escript {file}', extension: 'erl', monacoLang: 'erlang' },
  fsharp: { command: 'dotnet fsi {file}', extension: 'fsx', monacoLang: 'fsharp' },
  csharp: { command: 'dotnet script {file}', extension: 'csx', monacoLang: 'csharp' },
};

// ANSI to React converter for terminal output
export function parseAnsiToReact(text: string): JSX.Element[] | string {
  const ansiRegex = /\x1b\[([0-9;]+)m/g;
  const elements: JSX.Element[] = [];
  let lastIndex = 0;
  let currentStyle: React.CSSProperties = {};

  const colorMap: Record<number, string> = {
    30: '#000', 31: '#e74c3c', 32: '#2ecc71', 33: '#f39c12',
    34: '#3498db', 35: '#9b59b6', 36: '#1abc9c', 37: '#ecf0f1',
    90: '#7f8c8d', 91: '#ff6b6b', 92: '#51cf66', 93: '#ffd43b',
    94: '#4dabf7', 95: '#da77f2', 96: '#3bc9db', 97: '#f8f9fa'
  };

  const applyCode = (code: string, style: React.CSSProperties): React.CSSProperties => {
    const num = parseInt(code);
    if (num === 0) return {}; // Reset
    if (num === 1) return { ...style, fontWeight: 'bold' };
    if (num === 3) return { ...style, fontStyle: 'italic' };
    if (num === 4) return { ...style, textDecoration: 'underline' };
    if (colorMap[num]) return { ...style, color: colorMap[num] };
    return style;
  };

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

// Format tool call descriptions for Chat widget
export function formatToolDescription(toolName: string, input: any): string {
  if (!input) return toolName;

  switch (toolName) {
    case 'Read':
      return `Read ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Write':
      return `Write ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Edit':
      return `Edit ${input.file_path?.split('/').pop() || input.file_path || ''}`;
    case 'Bash':
      const cmd = input.command || '';
      const shortCmd = cmd.length > 40 ? cmd.substring(0, 40) + '...' : cmd;
      return `Run: ${shortCmd}`;
    case 'Grep':
      return `Search for "${input.pattern || ''}"`;
    case 'Glob':
      return `Find files: ${input.pattern || ''}`;
    case 'Task':
      return `${input.description || 'Start task'}`;
    case 'WebFetch':
      const url = input.url || '';
      const domain = url.replace(/^https?:\/\//, '').split('/')[0];
      return `Fetch ${domain}`;
    case 'WebSearch':
      return `Search: ${input.query || ''}`;
    default:
      return toolName;
  }
}

// Chat message component
interface ChatMessageProps {
  msg: {
    from: string;
    text: string;
    toolCalls?: Array<{ name: string; description?: string; id?: string; input?: any }>;
  };
  theme: Theme;
}

export const ChatMessage: FC<ChatMessageProps> = ({ msg, theme }) => {
  const components = {
    code({ node, inline, className, children, ...props }: any) {
      const match = /language-(\w+)/.exec(className || '');
      return !inline && match ? (
        <SyntaxHighlighter
          style={oneDark}
          language={match[1]}
          PreTag="div"
          customStyle={{ margin: '8px 0', borderRadius: '6px', fontSize: '0.7rem' }}
          {...props}
        >
          {String(children).replace(/\n$/, '')}
        </SyntaxHighlighter>
      ) : (
        <code className="inline-code" {...props}>{children}</code>
      );
    }
  };

  return (
    <>
      {msg.toolCalls && msg.toolCalls.length > 0 && (
        <div style={{
          display: 'flex',
          flexWrap: 'wrap',
          gap: '6px',
          marginBottom: '8px'
        }}>
          {msg.toolCalls.map((tool, idx) => (
            <div
              key={idx}
              style={{
                fontFamily: theme.textBody,
                fontSize: '0.7rem',
                padding: '4px 8px',
                backgroundColor: `${theme.accent}22`,
                border: `1px solid ${theme.accent}44`,
                borderRadius: '4px',
                color: theme.accent,
                display: 'flex',
                alignItems: 'center',
                gap: '4px'
              }}
            >
              <span style={{ opacity: 0.6 }}>🔧</span>
              {tool.description || tool.name}
            </div>
          ))}
        </div>
      )}
      <div className={`chat-bubble ${msg.from}`} style={{
        fontFamily: theme.textBody,
        backgroundColor: msg.from === 'user' ? 'rgba(255,255,255,0.1)' : `${theme.accent}22`,
        borderColor: msg.from === 'assistant' ? theme.accent : 'transparent',
      }}>
        <ReactMarkdown components={components}>{msg.text}</ReactMarkdown>
      </div>
    </>
  );
};

// Question prompt component for Chat widget (AskUserQuestion tool)
interface QuestionPromptProps {
  question: {
    question: string;
    options?: string[];
    allowMultiple?: boolean;
  };
  theme: Theme;
  onAnswer: (answer: any) => void;
  showSubmit?: boolean;
}

export const QuestionPrompt: FC<QuestionPromptProps> = ({ question, theme, onAnswer, showSubmit = true }) => {
  const [selectedOptions, setSelectedOptions] = useState<string[]>([]);
  const [customAnswer, setCustomAnswer] = useState('');
  const [showCustomInput, setShowCustomInput] = useState(!question.options || question.options.length === 0);

  // When showSubmit is false and custom input is being used, report answer as user types
  useEffect(() => {
    if (!showSubmit && showCustomInput && customAnswer.trim()) {
      onAnswer(customAnswer);
    }
  }, [customAnswer, showSubmit, showCustomInput, onAnswer]);

  const handleOptionClick = (option: string) => {
    let newOptions: string[];
    if (question.allowMultiple) {
      newOptions = selectedOptions.includes(option)
        ? selectedOptions.filter(o => o !== option)
        : [...selectedOptions, option];
      setSelectedOptions(newOptions);
    } else {
      newOptions = [option];
      setSelectedOptions(newOptions);
    }

    // If showSubmit is false, immediately report the answer
    if (!showSubmit) {
      if (question.allowMultiple) {
        onAnswer(newOptions);
      } else {
        onAnswer(option);
      }
    }
  };

  const handleSubmit = () => {
    if (showCustomInput) {
      onAnswer(customAnswer);
    } else if (question.allowMultiple) {
      onAnswer(selectedOptions);
    } else {
      onAnswer(selectedOptions[0] || '');
    }
  };

  const canSubmit = showCustomInput ? customAnswer.trim() : selectedOptions.length > 0;

  return (
    <div style={{
      padding: '16px',
      margin: '12px 0',
      backgroundColor: `${theme.accent}11`,
      border: `2px solid ${theme.accent}`,
      borderRadius: '12px',
      fontFamily: theme.textBody
    }}>
      <div style={{
        fontSize: '0.9rem',
        fontWeight: 'bold',
        marginBottom: '12px',
        color: theme.accent
      }}>
        ❓ Question
      </div>

      <div style={{
        fontSize: '0.85rem',
        marginBottom: '16px',
        lineHeight: '1.5',
        color: theme.textColor
      }}>
        {question.question}
      </div>

      {!showCustomInput && question.options && question.options.length > 0 && (
        <div style={{
          display: 'flex',
          flexDirection: 'column',
          gap: '8px',
          marginBottom: '12px'
        }}>
          {question.options.map((option, idx) => (
            <button
              key={idx}
              onClick={() => handleOptionClick(option)}
              style={{
                padding: '10px 14px',
                backgroundColor: selectedOptions.includes(option)
                  ? `${theme.accent}33`
                  : 'rgba(255,255,255,0.05)',
                border: `1px solid ${selectedOptions.includes(option) ? theme.accent : 'rgba(255,255,255,0.1)'}`,
                borderRadius: '8px',
                color: theme.textColor,
                cursor: 'pointer',
                fontFamily: theme.textBody,
                fontSize: '0.8rem',
                textAlign: 'left',
                transition: 'all 0.2s ease'
              }}
              onMouseEnter={(e) => {
                if (!selectedOptions.includes(option)) {
                  e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.08)';
                }
              }}
              onMouseLeave={(e) => {
                if (!selectedOptions.includes(option)) {
                  e.currentTarget.style.backgroundColor = 'rgba(255,255,255,0.05)';
                }
              }}
            >
              {option}
            </button>
          ))}
        </div>
      )}

      {(!question.options || question.options.length === 0 || showCustomInput) && (
        <div style={{ marginBottom: '12px' }}>
          <textarea
            value={customAnswer}
            onChange={(e) => setCustomAnswer(e.target.value)}
            placeholder="Type your answer here..."
            style={{
              width: '100%',
              minHeight: '80px',
              padding: '10px',
              backgroundColor: 'rgba(0,0,0,0.3)',
              border: `1px solid ${theme.accent}44`,
              borderRadius: '8px',
              color: theme.textColor,
              fontFamily: theme.textBody,
              fontSize: '0.8rem',
              resize: 'vertical'
            }}
          />
        </div>
      )}

      {question.options && question.options.length > 0 && (
        <button
          onClick={() => setShowCustomInput(!showCustomInput)}
          style={{
            fontSize: '0.75rem',
            color: theme.accent,
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            marginBottom: '12px',
            textDecoration: 'underline',
            fontFamily: theme.textBody
          }}
        >
          {showCustomInput ? 'Choose from options' : 'Or type a custom answer'}
        </button>
      )}

      {showSubmit && (
        <div style={{
          display: 'flex',
          justifyContent: 'flex-end'
        }}>
          <button
            onClick={handleSubmit}
            disabled={!canSubmit}
            style={{
              padding: '10px 24px',
              backgroundColor: canSubmit ? theme.accent : 'rgba(255,255,255,0.1)',
              border: 'none',
              borderRadius: '8px',
              color: canSubmit ? theme.bgApp : 'rgba(255,255,255,0.3)',
              cursor: canSubmit ? 'pointer' : 'not-allowed',
              fontFamily: theme.textBody,
              fontSize: '0.85rem',
              fontWeight: 'bold',
              opacity: canSubmit ? 1 : 0.5
            }}
          >
            Submit
          </button>
        </div>
      )}
    </div>
  );
};

// Multi-question prompt component for Chat widget
interface MultiQuestionPromptProps {
  questions: Array<{ id: string; question: string; options?: string[]; allowMultiple?: boolean }>;
  theme: Theme;
  onAnswer: (answers: Record<string, any>) => void;
}

export const MultiQuestionPrompt: FC<MultiQuestionPromptProps> = ({ questions, theme, onAnswer }) => {
  const [answers, setAnswers] = useState<Record<string, any>>({});

  const handleQuestionAnswer = useCallback((questionId: string, answer: any) => {
    console.log('Question answered:', questionId, answer);
    setAnswers(prev => {
      const newAnswers = { ...prev, [questionId]: answer };
      console.log('Updated answers:', newAnswers);
      return newAnswers;
    });
  }, []);

  const answeredCount = Object.keys(answers).length;
  const allQuestionsAnswered = questions.every(q => answers[q.id] !== undefined);

  const handleSubmitAll = () => {
    onAnswer(answers);
  };

  return (
    <div style={{
      padding: '16px',
      margin: '12px 0',
      backgroundColor: `${theme.accent}11`,
      border: `2px solid ${theme.accent}`,
      borderRadius: '12px',
      fontFamily: theme.textBody
    }}>
      <div style={{
        fontSize: '0.9rem',
        fontWeight: 'bold',
        marginBottom: '16px',
        color: theme.accent,
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center'
      }}>
        <span>❓ Questions</span>
        <span style={{ fontSize: '0.75rem', opacity: 0.8 }}>
          {answeredCount}/{questions.length} answered
        </span>
      </div>

      <div style={{
        display: 'flex',
        flexDirection: 'column',
        gap: '16px'
      }}>
        {questions.map((q, idx) => (
          <div key={q.id} style={{
            padding: '12px',
            backgroundColor: answers[q.id] !== undefined ? `${theme.accent}08` : 'transparent',
            border: `1px solid ${answers[q.id] !== undefined ? theme.accent + '33' : 'rgba(255,255,255,0.1)'}`,
            borderRadius: '8px'
          }}>
            <div style={{
              fontSize: '0.75rem',
              opacity: 0.6,
              marginBottom: '8px',
              textTransform: 'uppercase',
              letterSpacing: '0.5px',
              color: theme.textColor
            }}>
              Question {idx + 1}
            </div>
            <QuestionPrompt
              question={q}
              theme={theme}
              onAnswer={(answer) => handleQuestionAnswer(q.id, answer)}
              showSubmit={false}
            />
          </div>
        ))}
      </div>

      <div style={{
        display: 'flex',
        justifyContent: 'flex-end',
        marginTop: '16px',
        paddingTop: '16px',
        borderTop: `1px solid rgba(255,255,255,0.1)`
      }}>
        <button
          onClick={handleSubmitAll}
          disabled={!allQuestionsAnswered}
          style={{
            padding: '10px 24px',
            backgroundColor: allQuestionsAnswered ? theme.accent : 'rgba(255,255,255,0.1)',
            border: 'none',
            borderRadius: '8px',
            color: allQuestionsAnswered ? theme.bgApp : 'rgba(255,255,255,0.3)',
            cursor: allQuestionsAnswered ? 'pointer' : 'not-allowed',
            fontFamily: theme.textBody,
            fontSize: '0.85rem',
            fontWeight: 'bold',
            opacity: allQuestionsAnswered ? 1 : 0.5
          }}
        >
          Submit All Answers
        </button>
      </div>
    </div>
  );
};
