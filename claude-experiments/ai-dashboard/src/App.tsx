import { useState, useEffect } from 'react';
import { StyleProvider, SVGDefs } from './engine/StyleProvider';
import { useThemeStore } from './store/useThemeStore';
import {
  DashboardRoot,
  Sidebar,
  MainContent,
  ProjectSelector,
  WidgetContainer,
  MetricDisplay,
  DataSeries,
  StatusItem,
} from './semantic';
import { StyleControl } from './ui/StyleControl';

function App() {
  const {
    currentStyle,
    currentProjectId,
    setProjectStyle,
    switchProject,
    setGenerating,
    setStreamingStyle
  } = useThemeStore();
  const [activeProject, setActiveProject] = useState('project-1');

  // Sample data
  const projects = [
    { id: 'project-1', name: 'AI Research' },
    { id: 'project-2', name: 'Data Viz' },
    { id: 'project-3', name: 'Backend' },
  ];

  const sampleData = [45, 67, 23, 89, 56, 78, 34, 90, 45, 67, 82, 55];

  // Initialize first project on mount
  useEffect(() => {
    switchProject(activeProject);
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  // Handle project switching
  const handleProjectSelect = (projectId: string) => {
    setActiveProject(projectId);
    switchProject(projectId);
  };

  // Handle style generation (streaming)
  const handleGenerateStyle = (prompt: string) => {
    setGenerating(true);

    // Send the request
    window.electronAPI.generateStyle({
      prompt,
      context: {
        currentTheme: currentStyle,
      },
    });
  };

  // Set up streaming listeners - only once on mount
  useEffect(() => {
    console.log('[App] Setting up IPC listeners');

    // Handle CSS chunks as they arrive
    window.electronAPI.onStyleChunk((chunk: string) => {
      console.log('[App] Received chunk:', chunk);
      setStreamingStyle(chunk);
    });

    // Handle completion - use the state from the store directly
    window.electronAPI.onStyleComplete((style: any) => {
      console.log('[App] Style generation complete:', style.id);
      const state = useThemeStore.getState();
      const projectId = state.currentProjectId;
      console.log('[App] Current project ID:', projectId);

      // Save to current project
      if (projectId) {
        console.log('[App] Saving style to project:', projectId);
        state.setProjectStyle(projectId, style);
      } else {
        console.error('[App] No current project ID! Cannot save style.');
      }
      state.setGenerating(false);
    });

    // Handle errors
    window.electronAPI.onStyleError((error: string) => {
      console.error('[App] Style generation failed:', error);
      alert(`Failed to generate style: ${error}`);
      useThemeStore.getState().setGenerating(false);
    });

    // Listen for external agent registrations and data updates
    window.electronAPI.onAgentRegistration((message) => {
      console.log('[App] Agent registration:', message);
      // TODO: Handle component registration
    });

    window.electronAPI.onDataUpdate((message) => {
      console.log('[App] Data update:', message);
      // TODO: Update component data
    });
  }, []); // Empty deps - only run once on mount

  return (
    <StyleProvider style={currentStyle}>
      <SVGDefs content={currentStyle?.svgDefs} />

      <DashboardRoot>
        <Sidebar>
          <ProjectSelector
            projects={projects}
            activeId={activeProject}
            onSelect={handleProjectSelect}
          />
        </Sidebar>

        <MainContent>
          <StyleControl onGenerate={handleGenerateStyle} />

          <div className="grid">
            <WidgetContainer title="Metrics" id="metrics">
              <MetricDisplay value="99.4%" label="Success Rate" />
              <MetricDisplay value="12ms" label="Latency" unit="ms" />
              <MetricDisplay value="1.2k" label="Requests" unit="/min" />
            </WidgetContainer>

            <WidgetContainer title="Activity" id="activity">
              <DataSeries points={sampleData} />
            </WidgetContainer>

            <WidgetContainer title="System Status" id="status">
              <StatusItem label="API" value="Healthy" state="ok" />
              <StatusItem label="Database" value="Connected" state="ok" />
              <StatusItem label="Cache" value="87%" state="warn" />
              <StatusItem label="Queue" value="245 jobs" state="ok" />
            </WidgetContainer>
          </div>
        </MainContent>
      </DashboardRoot>
    </StyleProvider>
  );
}

export default App;
