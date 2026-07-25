// Root layout (Server Component) — wraps every route. Owns <html>, exposes the
// stable #app-shell wrapper the Tier-2 SSR smoke asserts, and exports Metadata.
export const metadata = {
  title: "blog-static",
  description: "SSG + route groups + nested layouts (hermetic corpus app)",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div id="app-shell" data-app="blog-static">
          {children}
        </div>
      </body>
    </html>
  );
}
