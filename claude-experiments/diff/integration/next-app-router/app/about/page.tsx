// A second app-router route, so `next/link href="/about"` resolves to a real page.
// Demonstrates the app-router multi-route file convention AND client-side soft
// navigation: the `next/link` back to `/` is intercepted by the client Router,
// which fetches `/` flight (`?__rsc=1`) and diff-renders it without a full reload.
import Link from "next/link";

export default function About() {
  return (
    <main id="about">
      About page (app-router route).
      <p>
        <Link id="home-link" href="/">
          Home
        </Link>
      </p>
    </main>
  );
}
