// Imported by an MDX page and used as JSX inside it.
export default function Note({ children }) {
  return (
    <aside className="note" data-testid="note">
      {children}
    </aside>
  );
}
