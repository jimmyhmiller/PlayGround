// Semantic widget container

import { ReactNode } from 'react';

export interface WidgetContainerProps {
  title: string;
  children: ReactNode;
  id?: string;
}

export function WidgetContainer({ title, children, id }: WidgetContainerProps) {
  return (
    <div className="widget-container" data-widget-id={id}>
      <div className="widget-header">
        <h3 className="widget-title">{title}</h3>
      </div>
      <div className="widget-content">{children}</div>
    </div>
  );
}
