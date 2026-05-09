'use client'

import { useState, useMemo, ReactNode } from 'react'
import { Highlight, themes } from 'prism-react-renderer'
import { parseDebugOutput, DebugNode } from '@/lib/debug-parser'

// ─── Custom Renderer Registry ────────────────────────────────
//
// Register custom renderers by type name. A custom renderer
// receives the parsed DebugNode and can return JSX, or null
// to fall back to the default renderer.

type CustomRenderer = (node: DebugNode & { type: 'struct' | 'tuple_struct' }) => ReactNode | null

const customRenderers = new Map<string, CustomRenderer>()

export function registerCellRenderer(typeName: string, renderer: CustomRenderer) {
  customRenderers.set(typeName, renderer)
}

// ─── Built-in custom renderers ──────────────────────────────

// Rope: Show the length and render the actual string content
registerCellRenderer('Rope', (node) => {
  if (node.type !== 'struct') return null
  const lengthField = node.fields.find(f => f.name === 'length')
  const dataField = node.fields.find(f => f.name === 'data')
  if (!lengthField || !dataField) return null

  const content = extractStringContent(dataField.value)
  const length = lengthField.value.type === 'number' ? lengthField.value.value : '?'

  return (
    <div className="space-y-1">
      <div className="text-[9px] text-zinc-500">{length} bytes</div>
      {content ? (
        <ResizableCodeBlock content={content} />
      ) : (
        <DebugNodeView node={dataField.value} depth={0} />
      )}
    </div>
  )
})

// Register all code-containing types
registerCodeContentRenderer('EcmascriptModuleContent')
registerCodeContentRenderer('EcmascriptChunkItemContent')

// Generic code content renderer for types with inner_code / source_map fields
function registerCodeContentRenderer(typeName: string) {
  registerCellRenderer(typeName, (node) => {
    if (node.type !== 'struct') return null
    const fields = new Map(node.fields.map(f => [f.name, f.value]))

    // Look for any field containing code-like content
    const codeFieldNames = ['inner_code', 'code', 'source']
    const mapFieldNames = ['source_map', 'sourcemap']
    const boolFieldNames = ['is_esm', 'strict']

    let codeContent: string | null = null
    let codeFieldName: string | null = null
    for (const name of codeFieldNames) {
      const field = fields.get(name)
      if (field) {
        codeContent = extractStringContent(field)
        codeFieldName = name
        break
      }
    }

    let sourceMapContent: string | null = null
    let sourceMapFieldName: string | null = null
    for (const name of mapFieldNames) {
      const field = fields.get(name)
      if (field) {
        sourceMapContent = extractStringContent(field)
        sourceMapFieldName = name
        break
      }
    }

    const handledFields = new Set([...codeFieldNames, ...mapFieldNames, ...boolFieldNames].filter(n => fields.has(n)))

    return (
      <div className="space-y-2">
        {/* Bool flags */}
        <div className="flex gap-2 text-[9px]">
          {fields.get('is_esm')?.type === 'bool' && (
            <span className={`rounded px-1.5 py-0.5 ${(fields.get('is_esm') as any).value ? 'bg-blue-900/40 text-blue-300' : 'bg-zinc-800 text-zinc-500'}`}>
              {(fields.get('is_esm') as any).value ? 'ESM' : 'CJS'}
            </span>
          )}
          {fields.get('strict')?.type === 'bool' && (fields.get('strict') as any).value && (
            <span className="rounded bg-amber-900/40 px-1.5 py-0.5 text-amber-300">strict</span>
          )}
        </div>

        {/* Code */}
        {codeContent ? (
          <FieldSection label={codeFieldName || 'Code'}>
            <ResizableCodeBlock content={codeContent} language="javascript" />
          </FieldSection>
        ) : codeFieldName && fields.get(codeFieldName) ? (
          <FieldSection label={codeFieldName}>
            <DebugNodeView node={fields.get(codeFieldName)!} depth={0} />
          </FieldSection>
        ) : null}

        {/* Source Map */}
        {sourceMapContent ? (
          <CollapsibleField label={sourceMapFieldName || 'Source Map'} defaultOpen={false}>
            <ResizableCodeBlock content={sourceMapContent} language="json" />
          </CollapsibleField>
        ) : sourceMapFieldName && fields.get(sourceMapFieldName) ? (
          <CollapsibleField label={sourceMapFieldName} defaultOpen={false}>
            <DebugNodeView node={fields.get(sourceMapFieldName)!} depth={0} />
          </CollapsibleField>
        ) : null}

        {/* Remaining fields */}
        {node.fields
          .filter(f => !handledFields.has(f.name))
          .map(f => (
            <FieldSection key={f.name} label={f.name}>
              <DebugNodeView node={f.value} depth={0} />
            </FieldSection>
          ))}
      </div>
    )
  })
}

registerCodeContentRenderer('module_factory_with_code_generation_issue')

// ─── Helpers ─────────────────────────────────────────────────

function unescapeString(s: string): string {
  return s
    .replace(/\\n/g, '\n')
    .replace(/\\t/g, '\t')
    .replace(/\\r/g, '\r')
    .replace(/\\"/g, '"')
    .replace(/\\\\/g, '\\')
}

// Walk the tree to find a string or byte_string buried in Rope → InnerRope → Local(b"...")
function extractStringContent(node: DebugNode): string | null {
  if (node.type === 'string') return unescapeString(node.value)
  if (node.type === 'byte_string') return unescapeString(node.value)

  if (node.type === 'struct') {
    const dataField = node.fields.find(f => f.name === 'data')
    if (dataField) return extractStringContent(dataField.value)
    if (node.fields.length > 0) return extractStringContent(node.fields[0].value)
  }

  if (node.type === 'tuple_struct') {
    for (const v of node.values) {
      const result = extractStringContent(v)
      if (result) return result
    }
  }

  if (node.type === 'array') {
    const parts: string[] = []
    for (const item of node.items) {
      const s = extractStringContent(item)
      if (s != null) parts.push(s)
    }
    if (parts.length > 0) return parts.join('')
  }

  return null
}

// ─── Components ──────────────────────────────────────────────

function ResizableCodeBlock({ content, language }: { content: string; language?: string }) {
  return (
    <div
      className="overflow-auto rounded ring-1 ring-zinc-800"
      style={{ resize: 'both', minHeight: 60, minWidth: 200, maxWidth: '100%' }}
    >
      {language ? (
        <SyntaxHighlightedCode code={content} language={language} />
      ) : (
        <pre className="whitespace-pre-wrap break-all p-2 font-mono text-[10px] text-zinc-300 bg-zinc-950">
          {content}
        </pre>
      )}
    </div>
  )
}

function SyntaxHighlightedCode({ code, language }: { code: string; language: string }) {
  return (
    <Highlight theme={themes.nightOwl} code={code} language={language}>
      {({ style, tokens, getLineProps, getTokenProps }) => (
        <pre
          className="p-2 font-mono text-[10px]"
          style={{ ...style, background: 'transparent', margin: 0 }}
        >
          {tokens.map((line, i) => (
            <div key={i} {...getLineProps({ line })} className="flex">
              <span className="mr-3 inline-block w-6 select-none text-right text-zinc-600">
                {i + 1}
              </span>
              <span>
                {line.map((token, key) => (
                  <span key={key} {...getTokenProps({ token })} />
                ))}
              </span>
            </div>
          ))}
        </pre>
      )}
    </Highlight>
  )
}

function FieldSection({ label, children }: { label: string; children: ReactNode }) {
  return (
    <div>
      <div className="text-[9px] font-medium uppercase tracking-wider text-zinc-500">{label}</div>
      <div className="mt-0.5">{children}</div>
    </div>
  )
}

function CollapsibleField({ label, defaultOpen = true, children }: { label: string; defaultOpen?: boolean; children: ReactNode }) {
  const [open, setOpen] = useState(defaultOpen)
  return (
    <div>
      <button
        className="flex items-center gap-1 text-[9px] font-medium uppercase tracking-wider text-zinc-500 hover:text-zinc-400"
        onClick={() => setOpen(v => !v)}
      >
        <span className="text-[8px]">{open ? '\u25BC' : '\u25B6'}</span>
        {label}
      </button>
      {open && <div className="mt-0.5">{children}</div>}
    </div>
  )
}

// ─── Main DebugNode renderer ─────────────────────────────────

interface DebugNodeViewProps {
  node: DebugNode
  depth: number
  inline?: boolean
}

function DebugNodeView({ node, depth, inline }: DebugNodeViewProps) {
  switch (node.type) {
    case 'string':
      return <span className="text-green-400">&quot;{node.value}&quot;</span>

    case 'byte_string':
      return <span className="text-green-400">b&quot;{node.value}&quot;</span>

    case 'number':
      return <span className="text-amber-300">{node.value}</span>

    case 'bool':
      return <span className="text-cyan-300">{String(node.value)}</span>

    case 'ident':
      return <span className="text-purple-300">{node.name}</span>

    case 'raw':
      return <span className="text-zinc-400">{node.text}</span>

    case 'array':
      if (node.items.length === 0) return <span className="text-zinc-500">[]</span>
      if (node.items.length <= 3 && node.items.every(isSimple)) {
        return (
          <span>
            <span className="text-zinc-500">[</span>
            {node.items.map((item, i) => (
              <span key={i}>
                {i > 0 && <span className="text-zinc-600">, </span>}
                <DebugNodeView node={item} depth={depth + 1} inline />
              </span>
            ))}
            <span className="text-zinc-500">]</span>
          </span>
        )
      }
      return <ArrayView items={node.items} depth={depth} />

    case 'tuple_struct':
      if (node.values.length === 0) return <span className="text-zinc-300">{node.name || '()'}</span>
      if (node.values.length === 1 && (node.name === 'Some' || node.name === 'Ok' || node.name === 'Err')) {
        return (
          <span>
            <span className="text-purple-300">{node.name}</span>
            <span className="text-zinc-500">(</span>
            <DebugNodeView node={node.values[0]} depth={depth} inline />
            <span className="text-zinc-500">)</span>
          </span>
        )
      }
      if (node.values.length === 1 && isSimple(node.values[0])) {
        return (
          <span>
            {node.name && <span className="text-zinc-300">{node.name}</span>}
            <span className="text-zinc-500">(</span>
            <DebugNodeView node={node.values[0]} depth={depth} inline />
            <span className="text-zinc-500">)</span>
          </span>
        )
      }
      return <TupleStructView name={node.name} values={node.values} depth={depth} />

    case 'struct': {
      if (node.name) {
        const custom = customRenderers.get(node.name)
        if (custom) {
          const result = custom(node)
          if (result) return <>{result}</>
        }
      }
      if (node.fields.length === 0) {
        return <span className="text-zinc-300">{node.name || '{}'}</span>
      }
      if (node.fields.length <= 2 && node.fields.every(f => isSimple(f.value)) && inline) {
        return (
          <span>
            {node.name && <span className="text-zinc-300">{node.name} </span>}
            <span className="text-zinc-500">{'{ '}</span>
            {node.fields.map((f, i) => (
              <span key={f.name}>
                {i > 0 && <span className="text-zinc-600">, </span>}
                <span className="text-zinc-500">{f.name}: </span>
                <DebugNodeView node={f.value} depth={depth + 1} inline />
              </span>
            ))}
            <span className="text-zinc-500">{' }'}</span>
          </span>
        )
      }
      return <StructView name={node.name} fields={node.fields} depth={depth} />
    }
  }
}

function isSimple(node: DebugNode): boolean {
  return node.type === 'string' || node.type === 'byte_string' ||
    node.type === 'number' || node.type === 'bool' || node.type === 'ident' ||
    (node.type === 'raw' && node.text.length < 40)
}

function StructView({ name, fields, depth }: { name: string; fields: { name: string; value: DebugNode }[]; depth: number }) {
  const [collapsed, setCollapsed] = useState(depth > 3)

  return (
    <div className="font-mono text-[10px]">
      <button
        className="flex items-center gap-1 text-zinc-300 hover:text-zinc-100"
        onClick={() => setCollapsed(v => !v)}
      >
        <span className="text-[8px] text-zinc-600">{collapsed ? '\u25B6' : '\u25BC'}</span>
        {name && <span className="font-semibold">{name}</span>}
        {collapsed && <span className="text-zinc-600">{`{ ${fields.length} field${fields.length !== 1 ? 's' : ''} }`}</span>}
      </button>
      {!collapsed && (
        <div className="ml-3 border-l border-zinc-800 pl-2">
          {fields.map((f) => (
            <div key={f.name} className="py-px">
              <span className="text-zinc-500">{f.name}: </span>
              <DebugNodeView node={f.value} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function TupleStructView({ name, values, depth }: { name: string; values: DebugNode[]; depth: number }) {
  const [collapsed, setCollapsed] = useState(depth > 3)

  return (
    <div className="font-mono text-[10px]">
      <button
        className="flex items-center gap-1 text-zinc-300 hover:text-zinc-100"
        onClick={() => setCollapsed(v => !v)}
      >
        <span className="text-[8px] text-zinc-600">{collapsed ? '\u25B6' : '\u25BC'}</span>
        {name && <span className="font-semibold">{name}</span>}
        {collapsed && <span className="text-zinc-600">({values.length} item{values.length !== 1 ? 's' : ''})</span>}
      </button>
      {!collapsed && (
        <div className="ml-3 border-l border-zinc-800 pl-2">
          {values.map((v, i) => (
            <div key={i} className="py-px">
              <DebugNodeView node={v} depth={depth + 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function ArrayView({ items, depth }: { items: DebugNode[]; depth: number }) {
  const [collapsed, setCollapsed] = useState(depth > 3)
  const [showAll, setShowAll] = useState(false)
  const LIMIT = 20
  const visibleItems = showAll ? items : items.slice(0, LIMIT)

  return (
    <div className="font-mono text-[10px]">
      <button
        className="flex items-center gap-1 text-zinc-300 hover:text-zinc-100"
        onClick={() => setCollapsed(v => !v)}
      >
        <span className="text-[8px] text-zinc-600">{collapsed ? '\u25B6' : '\u25BC'}</span>
        <span className="text-zinc-600">[{items.length} item{items.length !== 1 ? 's' : ''}]</span>
      </button>
      {!collapsed && (
        <div className="ml-3 border-l border-zinc-800 pl-2">
          {visibleItems.map((item, i) => (
            <div key={i} className="py-px">
              <DebugNodeView node={item} depth={depth + 1} />
            </div>
          ))}
          {items.length > LIMIT && !showAll && (
            <button className="text-[9px] text-blue-400 hover:text-blue-300" onClick={() => setShowAll(true)}>
              +{items.length - LIMIT} more
            </button>
          )}
        </div>
      )}
    </div>
  )
}

// ─── Public API ──────────────────────────────────────────────

export function CellContentRenderer({ preview, typeName }: { preview: string; typeName?: string }) {
  const parsed = useMemo(() => parseDebugOutput(preview), [preview])

  // If parsing produced just raw text, fall back to resizable pre
  if (parsed.type === 'raw') {
    return (
      <div
        className="overflow-auto rounded ring-1 ring-zinc-800"
        style={{ resize: 'both', minHeight: 40, minWidth: 200, maxWidth: '100%' }}
      >
        <pre className="whitespace-pre-wrap break-all bg-zinc-950 p-2 font-mono text-[10px] text-zinc-300">
          {preview}
        </pre>
      </div>
    )
  }

  // Check custom renderer for the top-level type or the typeName from the cell
  if (parsed.type === 'struct' || parsed.type === 'tuple_struct') {
    const name = parsed.name || typeName || ''
    if (name) {
      const custom = customRenderers.get(name)
      if (custom) {
        const result = custom(parsed)
        if (result) return <div className="font-mono text-[10px]">{result}</div>
      }
    }
  }

  return (
    <div
      className="overflow-auto"
      style={{ resize: 'both', minHeight: 40, minWidth: 200, maxWidth: '100%' }}
    >
      <DebugNodeView node={parsed} depth={0} />
    </div>
  )
}
