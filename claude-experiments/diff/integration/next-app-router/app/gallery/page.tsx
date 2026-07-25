import Link from "next/link";
export default function GalleryPage() {
  return (
    <main>
      <h1 id="gallery-heading">Gallery</h1>
      <Link id="photo-link" href="/gallery/photo/1">Open photo 1</Link>
    </main>
  );
}
