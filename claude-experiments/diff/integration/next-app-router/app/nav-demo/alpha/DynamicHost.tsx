// A `"use client"` host that uses next/dynamic with ssr:false + a loading fallback.
// dynamic() must be called in a client component for ssr:false (it uses client hooks).
"use client";

import dynamic from "next/dynamic";

const Heavy = dynamic(() => import("./Heavy"), {
  ssr: false,
  loading: () => <div id="heavy-loading">loading-heavy...</div>,
});

export default function DynamicHost() {
  return (
    <div id="dynamic-host">
      <Heavy />
    </div>
  );
}
