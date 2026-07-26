// A Route Handler that toggles Draft Mode (the preview-mode write side). `POST /api/draft`
// enables it (draftMode().enable() sets a signed `__prerender_bypass` Set-Cookie);
// `POST /api/draft?action=disable` clears it (an immediately-expired cookie). This is the
// canonical place Next expects draftMode().enable()/disable() to be called (a Route
// Handler or Server Action). `next build`/`next start` accept draftMode() in a handler.
import { draftMode } from "next/headers";

export async function POST(request: Request): Promise<Response> {
  const url = new URL(request.url);
  const dm = await draftMode();
  if (url.searchParams.get("action") === "disable") {
    dm.disable();
  } else {
    dm.enable();
  }
  return new Response(JSON.stringify({ draft: url.searchParams.get("action") !== "disable" }), {
    status: 200,
    headers: { "content-type": "application/json" },
  });
}
