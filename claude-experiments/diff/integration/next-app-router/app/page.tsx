// The app-router index page (`app/page.tsx`), a real async Server Component. It
// `await`s data on the server, renders a `next/link` (the core app-router
// navigation primitive), and embeds a `"use client"` island — passing a
// `"use server"` action into it as a prop. In the flight the island becomes a
// client reference and the action a server reference; neither's code ships from
// here. `next build` accepts this unchanged; diffpack builds the same tree through
// its RSC spine + a minimal `next/*` shim layer.
import Link from "next/link";
import Image from "next/image";
import styles from "./page.module.css";
import { Counter } from "./Counter";
import { increment } from "./actions";

export default async function Home() {
  const data = await Promise.resolve("from-server");
  return (
    <main id="page" className={styles.page}>
      <h1 id="heading">Server:{data}</h1>
      {/* Raster next/image: responsive srcset from build-emitted variants + a
          priority preload (Slice J / gap 4.2). */}
      <Image
        id="hero"
        src="/hero.png"
        alt="hero"
        width={1200}
        height={300}
        sizes="(max-width: 600px) 100vw, 600px"
        placeholder="blur"
        priority
      />
      {/* SVG next/image: raw src, no srcset (byte-faithful to Next). */}
      <Image id="logo" src="/next.svg" alt="logo" width={100} height={20} />
      <p>
        <Link id="about-link" href="/about">
          About
        </Link>
      </p>
      <Counter initial={5} increment={increment} />
    </main>
  );
}
