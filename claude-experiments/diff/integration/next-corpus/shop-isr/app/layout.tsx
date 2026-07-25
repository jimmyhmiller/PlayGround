// Root layout — imports globals.css (exercises the CSS-in-head path) and owns <html>.
import "./globals.css";

export const metadata = {
  title: "shop-isr",
  description: "ISR listing + SSG product pages (hermetic corpus app)",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <div id="app-shell" data-app="shop-isr">
          {children}
        </div>
      </body>
    </html>
  );
}
