import path from 'path'
import { fileURLToPath } from 'url'

const __dirname = path.dirname(fileURLToPath(import.meta.url))

/** @type {import('next').NextConfig} */
const nextConfig = {
  webpack: (config) => {
    // Force dagre to use its CJS bundle where require('@dagrejs/graphlib')
    // is a static call that webpack can resolve. The ESM bundle wraps require
    // in a dynamic shim that webpack replaces with webpackEmptyContext.
    config.resolve.alias['@dagrejs/dagre'] = path.resolve(
      __dirname, 'node_modules/@dagrejs/dagre/dist/dagre.cjs.js'
    )
    return config
  },
  async rewrites() {
    return [
      {
        source: '/control-api/:path*',
        destination: 'http://localhost:3311/:path*',
      },
    ]
  },
}

export default nextConfig
