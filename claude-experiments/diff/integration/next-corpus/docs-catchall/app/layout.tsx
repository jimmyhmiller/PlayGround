// Root layout — owns <html>, exposes #app-shell.
export const metadata = {
  title: "docs-catchall",
  description: "optional catch-all + boundaries + route handlers (hermetic corpus app)",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div id="app-shell" data-app="docs-catchall">
          {children}
        </div>
      </body>
    </html>
  );
}
