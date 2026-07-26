import type { Metadata } from "next";
export const metadata: Metadata = {
  title: "Meta Demo",
  description: "A page exercising the full Metadata API",
  keywords: ["diffpack", "next", "metadata"],
  metadataBase: new URL("https://example.com"),
  openGraph: {
    title: "Meta Demo OG",
    description: "OG description",
    url: "/meta-demo",
    siteName: "Diffpack",
    images: ["/og.png"],
    type: "article",
  },
  twitter: { card: "summary_large_image", title: "Meta Demo TW", images: ["/tw.png"] },
  robots: { index: false, follow: true },
  alternates: { canonical: "/meta-demo" },
  icons: { icon: "/favicon.ico" },
};
export const viewport = { themeColor: "#0af", width: "device-width", initialScale: 1 };
export default function MetaDemo() { return <h1 id="meta-demo">Meta demo</h1>; }
