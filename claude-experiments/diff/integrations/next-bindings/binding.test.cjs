'use strict'

const test = require('node:test')
const assert = require('node:assert/strict')

const binding = require('./binding.cjs')

test('custom binding preserves the native SWC surface', () => {
  assert.equal(typeof binding.transform, 'function')
  assert.equal(typeof binding.minify, 'function')
})

test('production project and endpoint handles satisfy Next raw shapes', async () => {
  const project = await binding.__diffpack.projectNew(
    { rootPath: '/workspace', projectPath: 'app', distDir: '.next', dev: false },
    {},
    {}
  )
  assert.equal(project.__napiType, 'Project')
  const entrypoints = binding.__diffpack.entrypointsFor([
    { pathname: '/', original_name: '/page', kind: 'app-page' },
    { pathname: '/api/ping', original_name: '/api/ping/route', kind: 'app-route' },
    { pathname: '/api/legacy', original_name: '/api/legacy', kind: 'pages-api' },
    { pathname: '/embed', original_name: '/router/embed', kind: 'pages-page' },
  ])
  assert.equal(entrypoints.routes[0].type, 'app-page')
  assert.equal(entrypoints.routes[0].pages[0].originalName, '/page')
  assert.equal(entrypoints.routes[1].type, 'app-route')
  assert.equal(entrypoints.routes[2].type, 'page-api')
  assert.equal(entrypoints.routes[3].type, 'page')
  assert.equal(entrypoints.routes[3].htmlEndpoint.kind, 'pages-page-html')
  assert.equal(entrypoints.routes[3].dataEndpoint.kind, 'pages-page-data')
  for (const value of [
    entrypoints.pagesDocumentEndpoint,
    entrypoints.pagesAppEndpoint,
    entrypoints.pagesErrorEndpoint,
  ]) {
    assert.equal(value.__napiType, 'Endpoint')
    assert.deepEqual(await binding.__diffpack.endpointWriteToDisk(value), {
      type: 'none',
      clientPaths: [],
      serverPaths: [],
      config: {},
      issues: [],
    })
  }
})

test('development project exposes a disposable entrypoint subscription', async () => {
  const project = await binding.__diffpack.projectNew(
    { rootPath: '/workspace', projectPath: 'app', distDir: '.next', dev: true },
    {},
    {}
  )
  assert.equal(typeof binding.projectEntrypointsSubscribe, 'function')
  assert.equal(typeof binding.rootTaskDispose, 'function')
})

test('development republishes stable endpoint handles', async () => {
  const project = await binding.__diffpack.projectNew(
    { rootPath: '/workspace', projectPath: 'app', distDir: '.next/dev', dev: true },
    {},
    {}
  )
  const route = { pathname: '/', original_name: '/page', kind: 'app-page' }
  const first = binding.__diffpack.entrypointsFor([route], '/workspace/app/.next/dev', project)
  const second = binding.__diffpack.entrypointsFor([{ ...route }], '/workspace/app/.next/dev', project)
  assert.equal(first.routes[0].pages[0].htmlEndpoint, second.routes[0].pages[0].htmlEndpoint)
  assert.equal(first.routes[0].pages[0].rscHmrEndpoint, second.routes[0].pages[0].rscHmrEndpoint)
})
