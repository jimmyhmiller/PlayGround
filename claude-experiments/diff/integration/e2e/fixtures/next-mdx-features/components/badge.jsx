// A component imported BY an MDX file and used as JSX inside it.
export default function Badge({ children }) {
  return (
    <span className="badge" data-testid="badge">
      {children}
    </span>
  );
}
