'use strict'

const { spawn, spawnSync } = require('node:child_process')
const { createHash } = require('node:crypto')
const fs = require('node:fs')
const path = require('node:path')
const readline = require('node:readline')

const PROTOCOL_VERSION = 1
const projects = new WeakSet()
const endpoints = new WeakSet()
const subscriptions = new WeakSet()

function required(name) {
  const value = process.env[name]
  if (!value) throw new Error(`@diffpack/next-bindings requires ${name}`)
  return value
}

function nativeBindingPath() {
  if (process.env.DIFFPACK_NEXT_BASE_BINDING) {
    return process.env.DIFFPACK_NEXT_BASE_BINDING
  }
  const repository = required('DIFFPACK_NEXT_REPO')
  const abi = `${process.platform}-${process.arch === 'x64' ? 'x64' : process.arch}`
  return path.join(
    repository,
    'packages/next-swc/native',
    `next-swc.${abi}.node`
  )
}

const native = require(nativeBindingPath())

function assertProduction(project) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  if (project.options.dev || project.options.watch?.enable) {
    throw new Error(
      '@diffpack/next-bindings currently implements production builds only; dev subscriptions and HMR are the next milestone'
    )
  }
}

function endpoint(kind = 'none', route, outputDir, project) {
  const value = { __napiType: 'Endpoint', kind, route, outputDir, project }
  endpoints.add(value)
  return value
}

function entrypointsFor(routes = [], outputDir, project) {
  return {
    routes: routes.map((route) => {
      if (route.kind === 'app-page') {
        return {
          pathname: route.pathname,
          type: 'app-page',
          pages: [{
            originalName: route.original_name,
            htmlEndpoint: endpoint('app-page-html', route, outputDir, project),
            rscHmrEndpoint: endpoint('app-page-rsc', route, outputDir, project),
          }],
        }
      }
      if (route.kind === 'pages-api') {
        return {
          pathname: route.pathname,
          type: 'page-api',
          endpoint: endpoint('pages-api', route, outputDir, project),
        }
      }
      if (route.kind === 'pages-page') {
        return {
          pathname: route.pathname,
          type: 'page',
          htmlEndpoint: endpoint('pages-page-html', route, outputDir, project),
          dataEndpoint: endpoint('pages-page-data', route, outputDir, project),
        }
      }
      return {
        pathname: route.pathname,
        originalName: route.original_name,
        type: 'app-route',
        endpoint: endpoint('app-route', route, outputDir, project),
      }
    }),
    pagesDocumentEndpoint: endpoint(),
    pagesAppEndpoint: endpoint(),
    pagesErrorEndpoint: endpoint(),
    issues: [],
  }
}

function projectPaths(project) {
  const projectRoot = path.resolve(project.options.rootPath, project.options.projectPath)
  return {
    projectRoot,
    outputDir: path.resolve(projectRoot, project.options.distDir),
  }
}

function buildProject(project) {
  const { projectRoot, outputDir } = projectPaths(project)
  const bridge = required('DIFFPACK_NEXT_BRIDGE')
  const nextConfig = JSON.parse(project.options.nextConfig)
  const request = {
    operation: 'build-production',
    protocol_version: PROTOCOL_VERSION,
    project_root: projectRoot,
    output_dir: outputDir,
    next_config_output: nextConfig.output ?? null,
  }
  const result = spawnSync(bridge, [], {
    input: `${JSON.stringify(request)}\n`,
    encoding: 'utf8',
    env: process.env,
    stdio: ['pipe', 'pipe', 'inherit'],
  })
  if (result.error) {
    throw new Error(`cannot start Diffpack Next bridge ${bridge}: ${result.error.message}`)
  }
  let response
  try {
    response = JSON.parse(result.stdout)
  } catch {
    throw new Error(`Diffpack Next bridge returned invalid JSON: ${result.stdout}`)
  }
  if (result.status !== 0 || !response.ok) {
    throw new Error(response.error || `Diffpack Next bridge exited ${result.status}`)
  }
  project.entrypoints = entrypointsFor(response.routes, outputDir, project)
  return project.entrypoints
}

function subscription(dispose = () => {}) {
  const value = { __napiType: 'RootTask', dispose }
  subscriptions.add(value)
  return value
}

function notify(callbacks, error, value) {
  for (const callback of callbacks) callback(error, value)
}

function startDevelopment(project) {
  if (project.development) return project.development
  const { projectRoot, outputDir } = projectPaths(project)
  const nextConfig = JSON.parse(project.options.nextConfig)
  const child = spawn(required('DIFFPACK_NEXT_BRIDGE'), [], {
    env: process.env,
    stdio: ['pipe', 'pipe', 'inherit'],
  })
  const development = {
    child,
    built: false,
    entrypoints: new Set(),
    serverHmr: new Set(),
    endpoints: new Set(),
  }
  project.development = development
  const lines = readline.createInterface({ input: child.stdout })
  lines.on('line', (line) => {
    let response
    try {
      response = JSON.parse(line)
    } catch (error) {
      notify(development.entrypoints, error)
      return
    }
    if (!response.ok) {
      const error = new Error(response.error || 'Diffpack development build failed')
      notify(development.entrypoints, error)
      notify(development.endpoints, error)
      return
    }
    const wasBuilt = development.built
    development.built = true
    project.entrypoints = entrypointsFor(response.routes, outputDir, project)
    notify(development.entrypoints, undefined, project.entrypoints)
    if (wasBuilt) {
      notify(development.endpoints, undefined, { issues: [] })
      notify(development.serverHmr, undefined, { type: 'restart', issues: [] })
    }
  })
  child.once('error', (error) => notify(development.entrypoints, error))
  child.once('exit', (code, signal) => {
    if (project.development !== development) return
    const error = new Error(
      `Diffpack development bridge stopped (${signal || `exit ${code}`})`
    )
    notify(development.entrypoints, error)
    notify(development.endpoints, error)
    notify(development.serverHmr, error)
  })
  child.stdin.end(`${JSON.stringify({
    operation: 'watch-development',
    protocol_version: PROTOCOL_VERSION,
    project_root: projectRoot,
    output_dir: outputDir,
    next_config_output: nextConfig.output ?? null,
    poll_interval_ms: project.options.watch?.pollIntervalMs ?? null,
  })}\n`)
  return development
}

async function projectNew(options, turboEngineOptions, callbacks) {
  const project = {
    __napiType: 'Project',
    options,
    turboEngineOptions,
    callbacks,
    entrypoints: null,
  }
  projects.add(project)
  return project
}

async function projectWriteAllEntrypointsToDisk(project) {
  assertProduction(project)
  return buildProject(project)
}

async function endpointWriteToDisk(value) {
  if (!endpoints.has(value)) throw new TypeError('invalid Diffpack endpoint handle')
  if (value.route && value.outputDir) {
    const prefix = value.route.kind.startsWith('pages-') ? 'server/pages' : 'server/app'
    const entryPath = `${prefix}/${value.route.original_name.replace(/^\//, '')}.js`
    const sharedPaths = value.route.kind.startsWith('pages-')
      ? ['server/diffpack-pages-entries.js']
      : ['server/diffpack-app-entries.js', 'server/diffpack-ssr.js']
    const serverPaths = [entryPath, ...sharedPaths]
      .filter((candidate) => fs.existsSync(path.join(value.outputDir, candidate)))
      .map((candidate) => ({
        path: candidate,
        contentHash: createHash('sha256')
          .update(fs.readFileSync(path.join(value.outputDir, candidate)))
          .digest('hex'),
      }))
    return {
      type: 'nodejs',
      entryPath,
      clientPaths: [],
      serverPaths,
      config: {},
      issues: [],
    }
  }
  return {
    type: 'none',
    clientPaths: [],
    serverPaths: [],
    config: {},
    issues: [],
  }
}

function projectEntrypointsSubscribe(project, callback) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  const development = startDevelopment(project)
  development.entrypoints.add(callback)
  if (development.built) queueMicrotask(() => callback(undefined, project.entrypoints))
  return subscription(() => development.entrypoints.delete(callback))
}

function rootTaskDispose(task) {
  if (!subscriptions.has(task)) return native.rootTaskDispose(task)
  subscriptions.delete(task)
  task.dispose()
}

async function projectUpdate(project, options) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  project.options = { ...project.options, ...options }
}

function initialSubscription(value, callback) {
  const task = subscription()
  queueMicrotask(() => {
    if (subscriptions.has(task)) callback(undefined, value)
  })
  return task
}

function projectAllHmrEvents(project, _target, callback) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  const development = startDevelopment(project)
  development.serverHmr.add(callback)
  queueMicrotask(() => callback(undefined, { type: 'restart', issues: [] }))
  return subscription(() => development.serverHmr.delete(callback))
}

function projectHmrEvents(project, _chunkName, _target, callback) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  return initialSubscription({ type: 'issues', issues: [] }, callback)
}

function projectHmrChunkNamesSubscribe(project, _target, callback) {
  if (!projects.has(project)) throw new TypeError('invalid Diffpack project handle')
  return initialSubscription({ chunkNames: [], issues: [] }, callback)
}

function endpointChangedSubscribe(value, callback) {
  if (!endpoints.has(value)) throw new TypeError('invalid Diffpack endpoint handle')
  const development = value.project?.development
  if (!development) return initialSubscription({ issues: [] }, callback)
  development.endpoints.add(callback)
  queueMicrotask(() => callback(undefined, { issues: [] }))
  return subscription(() => development.endpoints.delete(callback))
}

function unsupported(name) {
  return () => {
    throw new Error(`@diffpack/next-bindings does not implement ${name} yet`)
  }
}

async function stopDevelopment(project) {
  const development = project.development
  if (!development) return
  project.development = null
  development.child.kill()
}

module.exports = {
  ...native,
  projectNew,
  projectWriteAllEntrypointsToDisk,
  projectFeatureUsage: async () => [],
  projectGetAllCompilationIssues: async () => ({ issues: [] }),
  projectShutdown: stopDevelopment,
  projectOnExit: stopDevelopment,
  projectInvalidateFileSystemCache: async () => {},
  endpointWriteToDisk,
  rootTaskDispose,
  projectUpdate,
  projectEntrypointsSubscribe,
  projectAllHmrEvents,
  projectHmrEvents,
  projectHmrChunkNamesSubscribe,
  endpointServerChangedSubscribe: (value, _includeIssues, callback) =>
    endpointChangedSubscribe(value, callback),
  endpointClientChangedSubscribe: endpointChangedSubscribe,
  projectUpdateInfoSubscribe: () => {},
  projectCompilationEventsSubscribe: () => {},
}

module.exports.__diffpack = {
  PROTOCOL_VERSION,
  entrypointsFor,
  endpointWriteToDisk,
  projectNew,
}
