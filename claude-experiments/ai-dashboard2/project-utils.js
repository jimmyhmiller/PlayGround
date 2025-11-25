const fs = require('fs');
const path = require('path');
const crypto = require('crypto');

/**
 * Generate a unique project ID
 * @returns {string} Unique project ID
 */
function generateProjectId() {
  return `project-${Date.now()}-${crypto.randomBytes(4).toString('hex')}`;
}

/**
 * Detect if a path contains an embedded project (.ai-dashboard folder)
 * @param {string} dirPath - Directory path to check
 * @returns {boolean} True if .ai-dashboard exists
 */
function hasEmbeddedProject(dirPath) {
  const aiDashboardPath = path.join(dirPath, '.ai-dashboard');
  return fs.existsSync(aiDashboardPath) && fs.statSync(aiDashboardPath).isDirectory();
}

/**
 * Detect project type from path
 * @param {string} projectPath - Path to project directory
 * @returns {'embedded'|'standalone'} Project type
 */
function detectProjectType(projectPath) {
  // If path ends with .ai-dashboard or is in a dedicated dashboards folder, it's standalone
  if (projectPath.endsWith('.ai-dashboard') || projectPath.includes('ai-dashboards')) {
    return 'standalone';
  }
  // Otherwise it's embedded (project contains .ai-dashboard subfolder)
  return 'embedded';
}

/**
 * Get the actual project config directory
 * For embedded: /project/root/.ai-dashboard
 * For standalone: /path/to/project/folder
 * @param {string} projectPath - Project root path
 * @param {'embedded'|'standalone'} type - Project type
 * @returns {string} Config directory path
 */
function getProjectConfigDir(projectPath, type) {
  if (type === 'embedded') {
    return path.join(projectPath, '.ai-dashboard');
  }
  return projectPath;
}

/**
 * Find all dashboard JSON files in a project
 * @param {string} configDir - Project config directory
 * @returns {string[]} Array of dashboard file paths
 */
function findDashboardFiles(configDir) {
  const dashboards = [];

  try {
    // Check for dashboard.json in root
    const mainDashboard = path.join(configDir, 'dashboard.json');
    if (fs.existsSync(mainDashboard)) {
      dashboards.push(mainDashboard);
    }

    // Check for dashboards/ subdirectory
    const dashboardsDir = path.join(configDir, 'dashboards');
    if (fs.existsSync(dashboardsDir) && fs.statSync(dashboardsDir).isDirectory()) {
      const files = fs.readdirSync(dashboardsDir);
      files.forEach(file => {
        if (file.endsWith('.json')) {
          dashboards.push(path.join(dashboardsDir, file));
        }
      });
    }
  } catch (e) {
    console.error('[ProjectUtils] Error finding dashboards:', e);
  }

  return dashboards;
}

/**
 * Initialize project structure
 * @param {string} projectPath - Project root path
 * @param {'embedded'|'standalone'} type - Project type
 * @param {Object} options - Configuration options
 * @param {string} options.name - Project name
 * @param {string} [options.description] - Project description
 * @returns {Object} Result object with success/error
 */
function initializeProject(projectPath, type, options = {}) {
  try {
    const configDir = getProjectConfigDir(projectPath, type);

    // Create config directory if it doesn't exist
    if (!fs.existsSync(configDir)) {
      fs.mkdirSync(configDir, { recursive: true });
    }

    // Create project.json
    const projectConfig = {
      id: generateProjectId(),
      name: options.name || path.basename(projectPath),
      type,
      rootPath: projectPath,
      description: options.description || '',
      createdAt: new Date().toISOString(),
      dashboards: [],
      settings: {
        allowFileAccess: true,
        cwd: type === 'embedded' ? projectPath : configDir
      }
    };

    const projectJsonPath = path.join(configDir, 'project.json');
    fs.writeFileSync(projectJsonPath, JSON.stringify(projectConfig, null, 2));

    // Create dashboards subdirectory
    const dashboardsDir = path.join(configDir, 'dashboards');
    if (!fs.existsSync(dashboardsDir)) {
      fs.mkdirSync(dashboardsDir);
    }

    // Create data subdirectory
    const dataDir = path.join(configDir, 'data');
    if (!fs.existsSync(dataDir)) {
      fs.mkdirSync(dataDir);
    }

    // Create a default dashboard.json if none exists
    const defaultDashboard = path.join(configDir, 'dashboard.json');
    if (!fs.existsSync(defaultDashboard)) {
      const dashboardConfig = {
        id: `${projectConfig.id}-main`,
        title: projectConfig.name,
        subtitle: 'Project Dashboard',
        icon: '📊',
        theme: {
          background: '#1a1a1a',
          textColor: '#e0e0e0',
          textBody: 'system-ui, sans-serif',
          textCode: 'Menlo, Monaco, monospace',
          accent: '#4a9eff',
          positive: '#4caf50',
          negative: '#f44336',
          neutral: '#9e9e9e'
        },
        layout: {
          columns: '1fr 1fr',
          rows: 'auto 1fr',
          areas: '". ." ". ."',
          gridSize: 16,
          widgetGap: 12,
          buffer: 12
        },
        widgets: [
          {
            id: 'chat',
            type: 'chat',
            label: 'Project Assistant',
            backend: 'claude',
            claudeOptions: {
              model: 'claude-sonnet-4-5-20250929'
            },
            x: 0,
            y: 0,
            width: 400,
            height: 500
          }
        ]
      };
      fs.writeFileSync(defaultDashboard, JSON.stringify(dashboardConfig, null, 2));
    }

    return {
      success: true,
      projectConfig,
      configDir
    };
  } catch (error) {
    return {
      success: false,
      error: error.message
    };
  }
}

/**
 * Load project configuration from directory
 * @param {string} configDir - Project config directory
 * @returns {Object|null} Project config or null if not found
 */
function loadProjectConfig(configDir) {
  try {
    const projectJsonPath = path.join(configDir, 'project.json');
    if (!fs.existsSync(projectJsonPath)) {
      return null;
    }
    const content = fs.readFileSync(projectJsonPath, 'utf-8');
    return JSON.parse(content);
  } catch (e) {
    console.error('[ProjectUtils] Error loading project config:', e);
    return null;
  }
}

/**
 * Validate if a path can be a project
 * @param {string} dirPath - Directory path to validate
 * @returns {Object} Validation result
 */
function validateProjectPath(dirPath) {
  try {
    if (!fs.existsSync(dirPath)) {
      return { valid: false, error: 'Path does not exist' };
    }

    const stats = fs.statSync(dirPath);
    if (!stats.isDirectory()) {
      return { valid: false, error: 'Path is not a directory' };
    }

    return { valid: true };
  } catch (error) {
    return { valid: false, error: error.message };
  }
}

module.exports = {
  generateProjectId,
  hasEmbeddedProject,
  detectProjectType,
  getProjectConfigDir,
  findDashboardFiles,
  initializeProject,
  loadProjectConfig,
  validateProjectPath
};
