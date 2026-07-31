import Link from "next/link";
import Intro, { revision } from "./intro.mdx";

// `.mdx` used as a COMPONENT (not a route), including a named export read by the
// importing module.
export default function Home() {
  return (
    <main>
      <h1 data-testid="home-title">MDX features</h1>
      <p data-testid="revision">intro revision: {revision}</p>
      <Intro />
      <p>
        <Link href="/docs" data-testid="to-docs">
          Read the MDX route
        </Link>
      </p>
    </main>
  );
}
