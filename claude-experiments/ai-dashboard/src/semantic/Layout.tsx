// Semantic layout components

import { ReactNode } from 'react';

export function DashboardRoot({ children }: { children: ReactNode }) {
  return <div className="dashboard-root">{children}</div>;
}

export function Sidebar({ children }: { children: ReactNode }) {
  return <aside className="sidebar">{children}</aside>;
}

export function MainContent({ children }: { children: ReactNode }) {
  return <main className="main-content">{children}</main>;
}

export function ProjectSelector({
  projects,
  activeId,
  onSelect,
}: {
  projects: { id: string; name: string }[];
  activeId: string;
  onSelect: (id: string) => void;
}) {
  return (
    <nav className="project-selector">
      {projects.map((project) => (
        <button
          key={project.id}
          className={`project-item ${project.id === activeId ? 'active' : ''}`}
          onClick={() => onSelect(project.id)}
          data-project-id={project.id}
        >
          <span className="project-name">{project.name}</span>
        </button>
      ))}
    </nav>
  );
}
