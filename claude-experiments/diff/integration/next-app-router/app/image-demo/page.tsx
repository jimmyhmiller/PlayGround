// A Server Component page exercising diffpack's build-time image features:
//  - a STATIC image import (`import shot from './shot.png'`), which arrives as
//    Next's `{ src, width, height, blurDataURL, variants }` object with a real
//    responsive srcset from build-emitted variants (no image server), and
//  - `placeholder="blur"`, whose tiny auto-generated blurDataURL is painted as
//    the img's CSS background under the foreground image (zero client JS).
// `next build` accepts this unchanged (both are stable Next APIs); diffpack builds
// the same tree through its RSC spine + the `next/image` shim.
import Image from "next/image";
import shot from "./shot.png";

export default function ImageDemo() {
  return (
    <main id="image-demo">
      <h1 id="image-demo-heading">Image demo</h1>
      {/* Static import + auto blur placeholder + responsive variants. */}
      <Image id="shot" src={shot} alt="shot" placeholder="blur" sizes="100vw" />
    </main>
  );
}
