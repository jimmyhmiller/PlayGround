// A Server Component that issues a server-side `redirect()` (next/navigation). On the
// server, `redirect('/about')` throws Next's `NEXT_REDIRECT` digest; diffpack's
// react-server render captures it via `onError` and reports it to the orchestrator on
// the fd-3 control channel, which then issues a REAL HTTP 307 to `/about` (it never
// SSRs the errored tree). `next build` accepts this unchanged.
//
// `force-dynamic` opts this route out of Next's static prerender: a page that
// unconditionally redirects is resolved at REQUEST time (diffpack's per-request model).
import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default function Go() {
  redirect("/about");
}
