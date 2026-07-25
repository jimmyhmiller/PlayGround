// Reads request-scoped state via next/headers (cookies + headers) at the top of an
// async Server Component → classified `dynamic` (a request-state read forces
// per-request rendering).
import { cookies, headers } from "next/headers";

export default async function WhoAmI() {
  const cookieStore = await cookies();
  const headerList = await headers();
  const theme = cookieStore.get("theme")?.value ?? "none";
  const agent = headerList.get("user-agent") ?? "unknown";
  return (
    <main id="whoami">
      <h1>whoami</h1>
      <p id="theme">theme: {theme}</p>
      <p id="agent">agent-known: {String(agent.length > 0)}</p>
    </main>
  );
}
