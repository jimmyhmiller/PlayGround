// Real Next.js app-router root layout. In the app-router file convention this
// wraps every page: the RSC render composes `<RootLayout><Page/></RootLayout>`.
// It is a Server Component (no directive), so its code never ships to the browser
// — only the flight it produces. diffpack's next app-router adapter generates the
// three RSC entries (react-server render / SSR / client) that compose this layout
// around the matched page, exactly as Next's app-router runtime does.
//
// Uses `next/font/google` exactly as the stock create-next-app template does: the
// call is a build-time macro diffpack rewrites (next_font.rs) to a static object,
// and the adapter injects the companion CSS (Google @import + the CSS-variable
// class) as a React-hoisted <style> — so `${geist.variable}` on <html> resolves.
import "./globals.css";
import { Geist } from "next/font/google";

const geist = Geist({ variable: "--font-geist", subsets: ["latin"] });

export const metadata = {
  title: "diffpack next app-router",
  description: "RSC app-router app built and served by diffpack",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en" className={geist.variable}>
      <body>
        <div id="app-shell">{children}</div>
      </body>
    </html>
  );
}
