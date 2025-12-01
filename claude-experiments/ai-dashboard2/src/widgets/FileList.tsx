import { FC } from 'react';
import type { BaseWidgetComponentProps } from '../components/ui/Widget';

interface FileItem {
  name: string;
  status: string;
}

interface FileListConfig {
  id: string;
  type: 'fileList' | 'file-list';
  label: string;
  files: FileItem[];
  x?: number;
  y?: number;
  width?: number;
  height?: number;
}

export const FileList: FC<BaseWidgetComponentProps> = ({ theme, config }) => {
  const fileConfig = config as FileListConfig;

  return (
    <>
      <div className="widget-label" style={{ fontFamily: theme.textBody }}>
        {fileConfig.label}
      </div>
      <div className="file-list">
        {fileConfig.files.map((file, i) => (
          <div
            key={i}
            className="file-item"
            style={{
              fontFamily: theme.textBody,
              color: theme.textColor
            }}
          >
            <span className="file-name">{file.name}</span>
            <span
              className="file-status"
              style={{
                color: file.status === 'created' ? theme.positive : theme.accent
              }}
            >
              {file.status}
            </span>
          </div>
        ))}
      </div>
    </>
  );
};
