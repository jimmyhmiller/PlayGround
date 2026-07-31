// A NON-EMPTY `useMDXComponents` override map. The pinned third-party `next-mdx` example
// ships an empty one, so the override path was never observed by the suite; every override
// here changes the rendered DOM in a way the e2e probe records (a `data-testid`, an extra
// wrapper element, an extra class).
export function useMDXComponents(inherited = {}) {
  return {
    ...inherited,
    h1: ({ children, ...rest }) => (
      <h1 {...rest} className="mdx-h1" data-testid="override-h1">
        {children}
      </h1>
    ),
    // An override that changes the ELEMENT SHAPE, not just its attributes: a wrapper the
    // structure channel sees.
    table: ({ children, ...rest }) => (
      <div className="mdx-table-wrapper" data-testid="override-table">
        <table {...rest}>{children}</table>
      </div>
    ),
    del: ({ children, ...rest }) => (
      <del {...rest} className="mdx-del" data-testid="override-del">
        {children}
      </del>
    ),
    a: ({ children, ...rest }) => (
      <a {...rest} className="mdx-link" data-testid="override-a">
        {children}
      </a>
    ),
  };
}
