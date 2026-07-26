import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  typescript: { ignoreBuildErrors: true },
  eslint: { ignoreDuringBuilds: true },
  images: {
    remotePatterns: [{ protocol: "https", hostname: "**.imgix.net" }],
  },
  async redirects() {
    return [
      { source: "/old-about", destination: "/about", permanent: false },
    ];
  },
  async rewrites() {
    return [
      { source: "/rw-about", destination: "/about" },
    ];
  },
  async headers() {
    return [
      { source: "/:path*", headers: [{ key: "x-diffpack-config", value: "on" }] },
    ];
  },
};

export default nextConfig;
