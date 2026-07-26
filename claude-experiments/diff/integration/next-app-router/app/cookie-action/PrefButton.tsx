"use client";

// A client island that invokes the `setPrefCookie` Server Action (passed as a prop, so it
// serializes into the flight as a server reference the browser calls over `/_action/`).
// The action writes cookies server-side; the Set-Cookie rides back on the action response.
import { useState } from "react";

export function PrefButton({ setPref }: { setPref: () => Promise<string> }) {
  const [result, setResult] = useState<string>("none");
  return (
    <div id="pref-island">
      <button
        id="set-pref"
        onClick={async () => {
          setResult(await setPref());
        }}
      >
        set-pref
      </button>
      <span id="pref-result">pref: {result}</span>
    </div>
  );
}
