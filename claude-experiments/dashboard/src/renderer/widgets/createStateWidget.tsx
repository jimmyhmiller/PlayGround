/**
 * State Widget Factory
 *
 * Creates state-connected React components from declarative definitions.
 * Widgets automatically subscribe to backend state and get actions injected.
 */

import { memo, useMemo, type ComponentType, type ReactElement } from 'react';
import { useBackendStateSelector, useDispatch } from '../hooks/useBackendState';
import type {
  WidgetDefinition,
  WidgetProps,
  CommandDefinition,
} from '../../types/globalUI';
import type { CommandResult } from '../../types/state';

/**
 * Creates a state-connected widget component from a definition
 *
 * @example
 * const DashboardList = createStateWidget(
 *   {
 *     id: 'dashboard-list',
 *     displayName: 'Dashboard List',
 *     state: {
 *       path: 'projects',
 *       select: (state) => ({
 *         dashboards: getDashboardsForCurrentProject(state),
 *         activeDashboardId: state?.activeProject?.activeDashboardId,
 *       }),
 *     },
 *     commands: {
 *       switchDashboard: { type: 'dashboards.switch' },
 *     },
 *   },
 *   ({ data, actions }) => (
 *     <ul>
 *       {data.dashboards.map(d => (
 *         <li key={d.id} onClick={() => actions.switchDashboard({ id: d.id })}>
 *           {d.name}
 *         </li>
 *       ))}
 *     </ul>
 *   )
 * );
 */
export function createStateWidget<
  TState,
  TSelected,
  TCommands extends Record<string, CommandDefinition> = Record<string, never>
>(
  definition: WidgetDefinition<TState, TSelected, TCommands>,
  Component: ComponentType<WidgetProps<TSelected, TCommands>>
): ComponentType {
  const WrappedWidget = memo(function WrappedWidget(): ReactElement | null {
    // Subscribe to state with selector
    const [selectedState, loading] = useBackendStateSelector<TState, TSelected>(
      definition.state.path,
      definition.state.select,
      definition.state.equals
    );

    // Get dispatch function
    const dispatch = useDispatch();

    // Create action functions from command definitions
    const actions = useMemo(() => {
      if (!definition.commands) {
        return {} as WidgetProps<TSelected, TCommands>['actions'];
      }

      const actionMap: Record<string, (payload?: unknown) => Promise<CommandResult>> = {};

      for (const [name, commandDef] of Object.entries(definition.commands)) {
        actionMap[name] = async (payload?: unknown) => {
          const transformedPayload = commandDef.transform
            ? commandDef.transform(payload)
            : payload;
          return dispatch(commandDef.type, transformedPayload);
        };
      }

      return actionMap as WidgetProps<TSelected, TCommands>['actions'];
    }, [dispatch]);

    // Don't render while loading
    if (loading || selectedState === null) {
      return null;
    }

    return (
      <Component
        data={selectedState}
        loading={loading}
        actions={actions}
      />
    );
  });

  WrappedWidget.displayName = `StateWidget(${definition.id})`;

  return WrappedWidget;
}
