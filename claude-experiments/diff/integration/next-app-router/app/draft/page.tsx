// A Server Component that branches on draftMode().isEnabled — the preview-mode read side.
// When the request carries a valid signed `__prerender_bypass` cookie (set by the
// /api/draft enable handler), diffpack's next/headers shim verifies it against the baked
// DRAFT_SECRET and this renders the "Draft ON" branch; otherwise "Draft OFF". `next build`
// accepts draftMode() in a Server Component (read-only there, exactly as in real Next).
import { draftMode } from "next/headers";

export default async function DraftPage() {
  const { isEnabled } = await draftMode();
  return (
    <main id="draft">
      <h1 id="draft-status">{isEnabled ? "Draft ON" : "Draft OFF"}</h1>
    </main>
  );
}
