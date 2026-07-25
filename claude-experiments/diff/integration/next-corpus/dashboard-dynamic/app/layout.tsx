// Root layout — owns <html>, exposes #app-shell.
export const metadata = {
  title: "dashboard-dynamic",
  description: "force-dynamic + request-state reads + redirect (hermetic corpus app)",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div id="app-shell" data-app="dashboard-dynamic">
          {children}
        </div>
      </body>
    </html>
  );
}
