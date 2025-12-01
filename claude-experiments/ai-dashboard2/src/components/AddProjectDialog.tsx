import { FC, useState, useEffect } from 'react';
import type { Theme } from '../types';

interface AddProjectDialogProps {
  theme: Theme;
  onClose: () => void;
  onAdd: (project: any) => void;
}

export const AddProjectDialog: FC<AddProjectDialogProps> = ({ theme, onClose, onAdd }) => {
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [status, setStatus] = useState('');
  const [pickerOpen, setPickerOpen] = useState(true);

  const handleSelectFolder = async () => {
    try {
      console.log('[AddProject] Starting folder selection...');

      if (!(window as any).projectAPI) {
        console.error('[AddProject] window.projectAPI is not available!');
        setError('Project API not available');
        setPickerOpen(false);
        return;
      }

      const result = await (window as any).projectAPI.selectFolder();
      console.log('[AddProject] Folder selection result:', result);

      setPickerOpen(false);

      if (result.canceled) {
        console.log('[AddProject] User canceled folder selection');
        onClose();
        return;
      }

      if (result.success && result.path) {
        console.log('[AddProject] Starting project creation for:', result.path);
        setIsSubmitting(true);
        setError(null);
        setStatus('Generating AI-powered icon and theme...');

        const projectName = result.path.split('/').pop() || result.path.split('\\').pop();
        console.log('[AddProject] Project name:', projectName);

        const designResult = await (window as any).projectAPI.generateDesign(projectName);
        console.log('[AddProject] Design result:', designResult);
        const design = designResult.design;

        setStatus('Creating project...');

        const addResult = await (window as any).projectAPI.addProject(
          result.path,
          'embedded',
          undefined,
          design
        );

        console.log('[AddProject] Add result:', addResult);

        if (addResult.success) {
          setStatus('Project added successfully!');
          setTimeout(() => {
            onAdd(addResult.project);
            onClose();
          }, 500);
        } else {
          setError(addResult.error || 'Failed to add project');
          setIsSubmitting(false);
        }
      } else {
        setError(result.error || 'Failed to select folder');
      }
    } catch (err: any) {
      console.error('[AddProject] Error:', err);
      setError(err.message || 'An error occurred');
      setIsSubmitting(false);
      setPickerOpen(false);
    }
  };

  useEffect(() => {
    if (pickerOpen) {
      handleSelectFolder();
    }
  }, []);

  if (pickerOpen) {
    return null;
  }

  if (isSubmitting) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
      }}>
        <div style={{
          backgroundColor: theme.widgetBg || '#1a1a1a',
          border: theme.widgetBorder || '1px solid #333',
          borderRadius: theme.widgetRadius || '6px',
          padding: 24,
          width: 400,
          maxWidth: '90%',
          fontFamily: theme.textBody || 'system-ui',
          textAlign: 'center'
        }}>
          <div style={{ fontSize: '2rem', marginBottom: 16 }}>⟳</div>
          <div style={{ color: theme.textColor || '#fff', fontSize: '1rem', marginBottom: 8 }}>
            {status}
          </div>
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div style={{
        position: 'fixed',
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        backgroundColor: 'rgba(0, 0, 0, 0.7)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000
      }} onClick={onClose}>
        <div style={{
          backgroundColor: theme.widgetBg || '#1a1a1a',
          border: theme.widgetBorder || '1px solid #333',
          borderRadius: theme.widgetRadius || '6px',
          padding: 24,
          width: 400,
          maxWidth: '90%',
          fontFamily: theme.textBody || 'system-ui'
        }} onClick={(e) => e.stopPropagation()}>
          <h2 style={{
            margin: '0 0 16px 0',
            color: theme.negative || '#f85149',
            fontSize: '1.2rem'
          }}>
            Error Adding Project
          </h2>
          <p style={{
            margin: '0 0 20px 0',
            color: theme.textColor || '#fff',
            opacity: 0.8
          }}>
            {error}
          </p>
          <button
            onClick={onClose}
            style={{
              width: '100%',
              padding: '8px 16px',
              backgroundColor: theme.accent,
              border: 'none',
              borderRadius: 4,
              color: '#000',
              cursor: 'pointer',
              fontFamily: theme.textBody || 'system-ui',
              fontWeight: 600
            }}
          >
            Close
          </button>
        </div>
      </div>
    );
  }

  return null;
};
