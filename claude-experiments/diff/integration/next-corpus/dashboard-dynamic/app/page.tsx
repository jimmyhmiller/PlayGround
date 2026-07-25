// The dashboard index ("/"). No request read, no dynamic segment, no config →
// classified `static`.
import Link from "next/link";

export default function Home() {
  return (
    <main id="home">
      <h1>dashboard</h1>
      <nav>
        <Link href="/live">live</Link>
        <Link href="/whoami">whoami</Link>
        <Link href="/search">search</Link>
      </nav>
    </main>
  );
}
