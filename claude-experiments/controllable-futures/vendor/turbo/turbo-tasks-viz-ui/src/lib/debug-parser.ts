// ─── Rust Debug Format Parser ────────────────────────────────
//
// Parses Rust's Debug trait output into a structured tree.
// Supports: named structs, tuple structs, enums, arrays,
// strings, byte strings, numbers, bools, identifiers.

export type DebugNode =
  | { type: 'struct'; name: string; fields: { name: string; value: DebugNode }[] }
  | { type: 'tuple_struct'; name: string; values: DebugNode[] }
  | { type: 'array'; items: DebugNode[] }
  | { type: 'string'; value: string }
  | { type: 'byte_string'; value: string }
  | { type: 'number'; value: string }
  | { type: 'bool'; value: boolean }
  | { type: 'ident'; name: string }
  | { type: 'raw'; text: string }

class Parser {
  private pos = 0
  private input: string

  constructor(input: string) {
    this.input = input
  }

  private peek(): string {
    return this.input[this.pos] ?? ''
  }

  private advance(): string {
    return this.input[this.pos++] ?? ''
  }

  private skipWhitespace() {
    while (this.pos < this.input.length && /\s/.test(this.input[this.pos])) {
      this.pos++
    }
  }

  private isAtEnd(): boolean {
    return this.pos >= this.input.length
  }

  private expect(ch: string) {
    this.skipWhitespace()
    if (this.peek() !== ch) {
      throw new Error(`Expected '${ch}' at pos ${this.pos}, got '${this.peek()}'`)
    }
    this.advance()
  }

  private readString(): string {
    // Opening quote already consumed or about to be consumed
    this.advance() // consume "
    let result = ''
    while (!this.isAtEnd() && this.peek() !== '"') {
      if (this.peek() === '\\') {
        result += this.advance() // backslash
        if (!this.isAtEnd()) result += this.advance() // escaped char
      } else {
        result += this.advance()
      }
    }
    if (this.peek() === '"') this.advance() // consume closing "
    return result
  }

  private readIdentifier(): string {
    let result = ''
    while (!this.isAtEnd() && /[a-zA-Z0-9_]/.test(this.peek())) {
      result += this.advance()
    }
    return result
  }

  // Read a full path like Foo::Bar::Baz
  private readPath(): string {
    let result = this.readIdentifier()
    while (this.pos + 1 < this.input.length && this.input[this.pos] === ':' && this.input[this.pos + 1] === ':') {
      result += '::'
      this.pos += 2
      result += this.readIdentifier()
    }
    return result
  }

  private readNumber(): string {
    let result = ''
    if (this.peek() === '-') result += this.advance()
    while (!this.isAtEnd() && /[0-9.]/.test(this.peek())) {
      result += this.advance()
    }
    // Handle suffixes like u32, i64, f64, etc.
    if (!this.isAtEnd() && /[a-zA-Z_]/.test(this.peek())) {
      const suffix = this.readIdentifier()
      result += suffix
    }
    return result
  }

  parseValue(): DebugNode {
    this.skipWhitespace()
    if (this.isAtEnd()) return { type: 'raw', text: '' }

    const ch = this.peek()

    // String literal
    if (ch === '"') {
      return { type: 'string', value: this.readString() }
    }

    // Byte string literal: b"..."
    if (ch === 'b' && this.pos + 1 < this.input.length && this.input[this.pos + 1] === '"') {
      this.advance() // consume 'b'
      return { type: 'byte_string', value: this.readString() }
    }

    // Array
    if (ch === '[') {
      return this.parseArray()
    }

    // Anonymous struct/map { ... }
    if (ch === '{') {
      return this.parseAnonymousStruct()
    }

    // Number (or negative number)
    if (ch === '-' || (ch >= '0' && ch <= '9')) {
      return { type: 'number', value: this.readNumber() }
    }

    // Identifier, struct, tuple struct, enum, bool
    if (/[a-zA-Z_]/.test(ch)) {
      return this.parseIdentifierOrStruct()
    }

    // Parenthesized group (treat as tuple)
    if (ch === '(') {
      return this.parseTupleValues()
    }

    // Fallback: consume until a delimiter
    return this.parseRawUntilDelimiter()
  }

  private parseArray(): DebugNode {
    this.expect('[')
    const items: DebugNode[] = []
    this.skipWhitespace()
    while (!this.isAtEnd() && this.peek() !== ']') {
      items.push(this.parseValue())
      this.skipWhitespace()
      if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
    }
    if (this.peek() === ']') this.advance()
    return { type: 'array', items }
  }

  private parseAnonymousStruct(): DebugNode {
    this.expect('{')
    this.skipWhitespace()

    // Could be a map or a struct — check if first thing is `key: value`
    const fields: { name: string; value: DebugNode }[] = []
    while (!this.isAtEnd() && this.peek() !== '}') {
      const savedPos = this.pos
      // Try to read field_name: value
      if (/[a-zA-Z_"]/.test(this.peek())) {
        let fieldName: string
        if (this.peek() === '"') {
          fieldName = this.readString()
        } else {
          fieldName = this.readIdentifier()
        }
        this.skipWhitespace()
        if (this.peek() === ':') {
          this.advance() // consume :
          this.skipWhitespace()
          const value = this.parseValue()
          fields.push({ name: fieldName, value })
          this.skipWhitespace()
          if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
          continue
        }
        // Not a field — restore and treat as array-like
        this.pos = savedPos
      }
      // Fallback: parse as array item
      fields.push({ name: String(fields.length), value: this.parseValue() })
      this.skipWhitespace()
      if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
    }
    if (this.peek() === '}') this.advance()
    return { type: 'struct', name: '', fields }
  }

  private parseTupleValues(): DebugNode {
    this.expect('(')
    const values: DebugNode[] = []
    this.skipWhitespace()
    while (!this.isAtEnd() && this.peek() !== ')') {
      values.push(this.parseValue())
      this.skipWhitespace()
      if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
    }
    if (this.peek() === ')') this.advance()
    if (values.length === 1) return values[0]
    return { type: 'tuple_struct', name: '', values }
  }

  private parseIdentifierOrStruct(): DebugNode {
    const name = this.readPath()

    // Check for bool
    if (name === 'true') return { type: 'bool', value: true }
    if (name === 'false') return { type: 'bool', value: false }

    this.skipWhitespace()

    // Named struct: Foo { field: value, ... }
    if (this.peek() === '{') {
      this.advance() // consume {
      this.skipWhitespace()

      const fields: { name: string; value: DebugNode }[] = []
      while (!this.isAtEnd() && this.peek() !== '}') {
        const savedPos = this.pos
        const fieldName = this.readIdentifier()
        this.skipWhitespace()
        if (fieldName && this.peek() === ':') {
          this.advance() // consume :
          this.skipWhitespace()
          const value = this.parseValue()
          fields.push({ name: fieldName, value })
        } else {
          // Not field:value — might be a value inside braces
          this.pos = savedPos
          fields.push({ name: String(fields.length), value: this.parseValue() })
        }
        this.skipWhitespace()
        if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
      }
      if (this.peek() === '}') this.advance()
      return { type: 'struct', name, fields }
    }

    // Tuple struct or enum variant: Foo(...) or Foo::Bar(...)
    if (this.peek() === '(') {
      this.advance() // consume (
      const values: DebugNode[] = []
      this.skipWhitespace()
      while (!this.isAtEnd() && this.peek() !== ')') {
        values.push(this.parseValue())
        this.skipWhitespace()
        if (this.peek() === ',') { this.advance(); this.skipWhitespace() }
      }
      if (this.peek() === ')') this.advance()
      return { type: 'tuple_struct', name, values }
    }

    // Plain identifier (like None, or a unit variant)
    return { type: 'ident', name }
  }

  private parseRawUntilDelimiter(): DebugNode {
    let text = ''
    while (!this.isAtEnd() && !/[,\]\}\)]/.test(this.peek())) {
      text += this.advance()
    }
    return { type: 'raw', text: text.trim() }
  }

  parse(): DebugNode {
    const result = this.parseValue()
    return result
  }
}

export function parseDebugOutput(input: string): DebugNode {
  try {
    const parser = new Parser(input.trim())
    return parser.parse()
  } catch {
    // If parsing fails, return the raw text
    return { type: 'raw', text: input }
  }
}
