// A Server Component that issues a server-side redirect(). `force-dynamic` opts it out
// of prerender (a route that unconditionally redirects resolves at request time) →
// classified `dynamic`. On the server redirect() throws the NEXT_REDIRECT digest the
// orchestrator turns into a real HTTP 307.
import { redirect } from "next/navigation";

export const dynamic = "force-dynamic";

export default function Go() {
  redirect("/");
}
