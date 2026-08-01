import { createAppPageEntrypoint } from 'next/dist/build/templates/app-page-runtime'
import { interopDefault } from 'next/dist/server/app-render/interop-default'

/*__DIFFPACK_ROUTE_IMPORTS__*/
/*__DIFFPACK_REFERENCE_IMPORTS__*/

const tree = /*__DIFFPACK_LOADER_TREE__*/
const rscModules = {
  /*__DIFFPACK_REFERENCE_TABLE__*/
}

const loadSsrTable = process
  .getBuiltinModule('module')
  .createRequire(`${process.cwd()}/package.json`)(/*__DIFFPACK_SSR_PATH__*/)

const ssrTablePromise = Promise.resolve(loadSsrTable).then(
  (module) => module.default ?? module
)
let ssrTable = null
const pendingSsrModules = new Map()

function missingModule(id) {
  throw new Error(
    `Diffpack native Next module id ${id} is absent from both the RSC and SSR module tables`
  )
}

function requireModule(id) {
  const rscModule = rscModules[id]
  if (rscModule != null) return rscModule
  if (ssrTable != null) return ssrTable[id] ?? missingModule(id)

  let pendingModule = pendingSsrModules.get(id)
  if (pendingModule == null) {
    pendingModule = ssrTablePromise.then((table) => {
      ssrTable = table
      return table[id] ?? missingModule(id)
    })
    pendingSsrModules.set(id, pendingModule)
  }
  return pendingModule
}

async function loadChunk() {
  ssrTable ??= await ssrTablePromise
}

const entrypoint = createAppPageEntrypoint({
  tree,
  page: /*__DIFFPACK_PAGE__*/,
  pathname: /*__DIFFPACK_PATHNAME__*/,
  require: requireModule,
  loadChunk,
  interopDefault,
})

export const __next_app__ = entrypoint.__next_app__
export const routeModule = entrypoint.routeModule
export const handler = entrypoint.handler
export * from 'next/dist/server/app-render/entry-base'
