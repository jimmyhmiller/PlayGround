import { createContext, useContext, useCallback, memo } from 'react';
import { useComponentsState } from '../hooks/useBackendState';

/**
 * Component Registry System
 *
 * Allows dynamic instantiation of registered component types.
 * Each instance can have its own event subscription pattern.
 * Instance state is managed by the backend for persistence.
 */

// Registry of available component types (in-memory, for component lookups)
const componentTypes = new Map();

/**
 * Register a component type
 * @param {string} type - Component type name (e.g., 'codemirror', 'git-diff')
 * @param {React.Component} Component - The React component
 * @param {Object} defaultProps - Default props for this component type
 */
export function registerComponent(type, Component, defaultProps = {}) {
  componentTypes.set(type, { Component, defaultProps });
}

/**
 * Get a registered component type
 */
export function getComponentType(type) {
  return componentTypes.get(type);
}

/**
 * Get all registered component types
 */
export function getRegisteredTypes() {
  return Array.from(componentTypes.keys());
}

// Context for component instances
const ComponentInstancesContext = createContext(null);

/**
 * Provider for managing dynamic component instances
 * Uses backend state for persistence
 */
export function ComponentInstancesProvider({ children }) {
  const {
    instances,
    addInstance: backendAddInstance,
    removeInstance: backendRemoveInstance,
    updateInstanceProps: backendUpdateInstanceProps,
    loading,
  } = useComponentsState();

  const addInstance = useCallback(async (type, props = {}) => {
    const typeInfo = componentTypes.get(type);

    if (!typeInfo) {
      console.error(`Unknown component type: ${type}`);
      return null;
    }

    const mergedProps = { ...typeInfo.defaultProps, ...props };
    const result = await backendAddInstance(type, mergedProps);
    return result?.id;
  }, [backendAddInstance]);

  const removeInstance = useCallback((id) => {
    backendRemoveInstance(id);
  }, [backendRemoveInstance]);

  const updateInstanceProps = useCallback((id, newProps) => {
    backendUpdateInstanceProps(id, newProps);
  }, [backendUpdateInstanceProps]);

  const value = {
    instances,
    addInstance,
    removeInstance,
    updateInstanceProps,
    getRegisteredTypes,
    loading,
  };

  return (
    <ComponentInstancesContext.Provider value={value}>
      {children}
    </ComponentInstancesContext.Provider>
  );
}

/**
 * Hook to access component instances
 */
export function useComponentInstances() {
  const context = useContext(ComponentInstancesContext);
  if (!context) {
    throw new Error('useComponentInstances must be used within ComponentInstancesProvider');
  }
  return context;
}

/**
 * Renders a single dynamic component instance
 */
const DynamicComponentInstance = memo(function DynamicComponentInstance({ instance, onRemove }) {
  const typeInfo = componentTypes.get(instance.type);

  if (!typeInfo) {
    return (
      <div style={{ padding: '10px', background: '#ffebee', color: '#c62828' }}>
        Unknown component type: {instance.type}
      </div>
    );
  }

  const { Component } = typeInfo;

  return (
    <div style={{
      border: '1px solid #ddd',
      borderRadius: '8px',
      overflow: 'hidden',
      background: '#fff',
    }}>
      <div style={{
        display: 'flex',
        justifyContent: 'space-between',
        alignItems: 'center',
        padding: '8px 12px',
        background: '#f5f5f5',
        borderBottom: '1px solid #ddd',
        fontSize: '12px',
      }}>
        <span>
          <strong>{instance.type}</strong>
          <span style={{ color: '#888', marginLeft: '8px' }}>
            {instance.props.subscribePattern && `[${instance.props.subscribePattern}]`}
          </span>
        </span>
        <button
          onClick={() => onRemove(instance.id)}
          style={{
            background: 'none',
            border: 'none',
            cursor: 'pointer',
            fontSize: '16px',
            color: '#888',
          }}
        >
          ×
        </button>
      </div>
      <Component {...instance.props} instanceId={instance.id} />
    </div>
  );
});

/**
 * Renders all dynamic component instances
 */
export function DynamicComponentPanel() {
  const { instances, removeInstance, loading } = useComponentInstances();

  if (loading) {
    return (
      <div style={{
        padding: '20px',
        textAlign: 'center',
        color: '#888',
        background: '#fafafa',
        borderRadius: '8px',
      }}>
        Loading components...
      </div>
    );
  }

  if (instances.length === 0) {
    return (
      <div style={{
        padding: '20px',
        textAlign: 'center',
        color: '#888',
        background: '#fafafa',
        borderRadius: '8px',
      }}>
        No components instantiated. Use the controls to add components.
      </div>
    );
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: '15px' }}>
      {instances.map(instance => (
        <DynamicComponentInstance
          key={instance.id}
          instance={instance}
          onRemove={removeInstance}
        />
      ))}
    </div>
  );
}
