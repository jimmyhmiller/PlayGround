// A Server Component that passes the `setPrefCookie` Server Action into a client island,
// exactly mirroring how the index page wires `increment` into Counter. Clicking the button
// invokes the action over `/_action/`, which writes cookies server-side and returns them as
// Set-Cookie on the action response. `next build` builds this tree unchanged.
import { PrefButton } from "./PrefButton";
import { setPrefCookie } from "../actions";

export default function CookieActionPage() {
  return (
    <main id="cookie-action">
      <h1>cookie action</h1>
      <PrefButton setPref={setPrefCookie} />
    </main>
  );
}
