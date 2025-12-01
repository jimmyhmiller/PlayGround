import { FC, useState, useEffect } from 'react';

const icons: Record<string, JSX.Element> = {
  botanist: (
    <svg viewBox="0 0 60 60">
      <path d="M10 10 Q30 5 50 10 Q55 30 50 50 Q30 55 10 50 Q5 30 10 10" />
      <path d="M30 50 Q 45 30 30 10 Q 15 30 30 50" />
    </svg>
  ),
  grid: (
    <svg viewBox="0 0 60 60">
      <path d="M10 5 L50 5 L55 15 L55 55 L15 55 L5 45 Z" />
      <rect x="20" y="20" width="20" height="20" />
    </svg>
  ),
  dream: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="5" width="50" height="50" rx="20" />
      <circle cx="30" cy="30" r="10" />
    </svg>
  ),
  console: (
    <svg viewBox="0 0 60 60">
      <rect x="5" y="10" width="50" height="40" rx="4" />
      <path d="M15 25 L25 32 L15 39" />
      <line x1="30" y1="39" x2="45" y2="39" />
    </svg>
  ),
  square: <svg viewBox="0 0 60 60"><rect x="10" y="10" width="40" height="40" /></svg>,
  circle: <svg viewBox="0 0 60 60"><circle cx="30" cy="30" r="20" /></svg>,
};

interface ProjectNodeProps {
  icon?: string;
  active?: boolean;
  accent: string;
  hoverAccent: string;
  onClick: () => void;
  onDelete?: () => void;
}

export const ProjectNode: FC<ProjectNodeProps> = ({ icon, active, accent, hoverAccent, onClick, onDelete }) => {
  const [showContextMenu, setShowContextMenu] = useState(false);
  const [menuPosition, setMenuPosition] = useState({ x: 0, y: 0 });

  // Support inline SVG string or predefined icon name
  const iconContent = typeof icon === 'string' && icons[icon]
    ? icons[icon]
    : typeof icon === 'string' && icon.trim().startsWith('<svg')
      ? <span dangerouslySetInnerHTML={{ __html: icon }} />
      : icons.square;

  const handleContextMenu = (e: React.MouseEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setMenuPosition({ x: e.clientX, y: e.clientY });
    setShowContextMenu(true);
  };

  const handleDelete = () => {
    setShowContextMenu(false);
    if (onDelete) {
      onDelete();
    }
  };

  useEffect(() => {
    if (showContextMenu) {
      const handleClick = () => setShowContextMenu(false);
      document.addEventListener('click', handleClick);
      return () => document.removeEventListener('click', handleClick);
    }
  }, [showContextMenu]);

  return (
    <>
      <div
        className={`project-node ${active ? 'active' : ''}`}
        style={{ '--accent': accent, '--hover-accent': hoverAccent } as any}
        onClick={onClick}
        onContextMenu={handleContextMenu}
      >
        {iconContent}
      </div>
      {showContextMenu && (
        <div
          style={{
            position: 'fixed',
            left: menuPosition.x,
            top: menuPosition.y,
            backgroundColor: '#1a1a1a',
            border: '1px solid #333',
            borderRadius: '6px',
            padding: '4px 0',
            zIndex: 10000,
            minWidth: 150,
            boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
          }}
          onClick={(e) => e.stopPropagation()}
        >
          <div
            onClick={handleDelete}
            style={{
              padding: '8px 12px',
              cursor: 'pointer',
              color: '#f85149',
              fontSize: '0.85rem',
              fontFamily: 'system-ui',
              transition: 'background-color 0.2s'
            }}
            onMouseEnter={(e) => (e.target as HTMLDivElement).style.backgroundColor = 'rgba(248, 81, 73, 0.1)'}
            onMouseLeave={(e) => (e.target as HTMLDivElement).style.backgroundColor = 'transparent'}
          >
            Remove Project
          </div>
        </div>
      )}
    </>
  );
};
