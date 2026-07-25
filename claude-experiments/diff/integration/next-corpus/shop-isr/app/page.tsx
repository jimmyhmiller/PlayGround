// The product listing at "/". `export const revalidate = 5` on an otherwise-static
// route → classified `isr`: prerendered at build AND regenerated on demand once the
// cached copy is older than 5 seconds. Renders a committed SVG through next/image
// (unoptimized: raw src, no srcset — byte-faithful to Next).
import Image from "next/image";
import Link from "next/link";
import styles from "./page.module.css";
import { PRODUCTS } from "./products";

export const revalidate = 5;

export default function Listing() {
  return (
    <main id="listing" className={styles.listing}>
      <h1>shop (ISR listing)</h1>
      <Image id="tile" src="/tile.svg" alt="tile" width={48} height={48} />
      <ul>
        {PRODUCTS.map((p) => (
          <li key={p.id}>
            <Link href={`/products/${p.id}`}>
              {p.name} — ${p.price}
            </Link>
          </li>
        ))}
      </ul>
    </main>
  );
}
