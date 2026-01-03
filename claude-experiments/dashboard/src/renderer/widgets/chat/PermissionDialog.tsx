/**
 * PermissionDialog - Modal for tool permission requests
 */

import React, { memo } from 'react';

interface PermissionRequest {
  requestId: string;
  toolCallId: string;
  title: string;
  description?: string;
}

interface PermissionDialogProps {
  request: PermissionRequest;
  onAllow: () => void;
  onDeny: () => void;
}

export const PermissionDialog = memo(function PermissionDialog({
  request,
  onAllow,
  onDeny,
}: PermissionDialogProps) {
  const overlayStyle: React.CSSProperties = {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    backgroundColor: 'rgba(0, 0, 0, 0.6)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    zIndex: 1000,
  };

  const dialogStyle: React.CSSProperties = {
    backgroundColor: 'var(--theme-bg-primary)',
    border: '1px solid var(--theme-border-primary)',
    borderRadius: 'var(--theme-radius-lg)',
    padding: '20px',
    maxWidth: '400px',
    width: '90%',
    boxShadow: 'var(--theme-window-shadow)',
  };

  const headerStyle: React.CSSProperties = {
    display: 'flex',
    alignItems: 'center',
    gap: '12px',
    marginBottom: '16px',
  };

  const iconStyle: React.CSSProperties = {
    fontSize: '24px',
  };

  const titleStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-lg)',
    fontWeight: 600,
    color: 'var(--theme-text-primary)',
  };

  const descStyle: React.CSSProperties = {
    fontSize: 'var(--theme-font-size-md)',
    color: 'var(--theme-text-secondary)',
    marginBottom: '20px',
    lineHeight: '1.5',
  };

  const toolInfoStyle: React.CSSProperties = {
    backgroundColor: 'var(--theme-bg-secondary)',
    borderRadius: 'var(--theme-radius-md)',
    padding: '12px',
    marginBottom: '20px',
    fontFamily: 'var(--theme-font-mono)',
    fontSize: 'var(--theme-font-size-sm)',
    color: 'var(--theme-text-primary)',
  };

  const buttonsStyle: React.CSSProperties = {
    display: 'flex',
    gap: '12px',
    justifyContent: 'flex-end',
  };

  const buttonBaseStyle: React.CSSProperties = {
    padding: '10px 20px',
    borderRadius: 'var(--theme-radius-md)',
    fontSize: 'var(--theme-font-size-md)',
    fontWeight: 500,
    cursor: 'pointer',
    border: 'none',
    transition: 'background-color 0.2s',
  };

  const denyButtonStyle: React.CSSProperties = {
    ...buttonBaseStyle,
    backgroundColor: 'var(--theme-bg-secondary)',
    color: 'var(--theme-text-primary)',
    border: '1px solid var(--theme-border-primary)',
  };

  const allowButtonStyle: React.CSSProperties = {
    ...buttonBaseStyle,
    backgroundColor: 'var(--theme-accent-primary)',
    color: '#fff',
  };

  return (
    <div style={overlayStyle} onClick={onDeny}>
      <div style={dialogStyle} onClick={(e) => e.stopPropagation()}>
        <div style={headerStyle}>
          <span style={iconStyle}>🔐</span>
          <span style={titleStyle}>Permission Required</span>
        </div>

        <div style={descStyle}>
          The assistant wants to perform an action that requires your approval.
        </div>

        <div style={toolInfoStyle}>
          <strong>{request.title}</strong>
          {request.description && (
            <div style={{ marginTop: '8px', color: 'var(--theme-text-muted)' }}>
              {request.description}
            </div>
          )}
        </div>

        <div style={buttonsStyle}>
          <button style={denyButtonStyle} onClick={onDeny}>
            Deny
          </button>
          <button style={allowButtonStyle} onClick={onAllow}>
            Allow
          </button>
        </div>
      </div>
    </div>
  );
});
