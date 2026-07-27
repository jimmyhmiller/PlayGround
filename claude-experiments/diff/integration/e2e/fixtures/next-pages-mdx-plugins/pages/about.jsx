import Link from "next/link";

// A plain pages-router page, so the MDX page has somewhere to navigate to.
export default function About() {
  return (
    <main>
      <h1 data-testid="about-title">About</h1>
      <p>
        <Link href="/" data-testid="to-home">
          Back to the MDX page
        </Link>
      </p>
    </main>
  );
}
